// =============================================================================
// color-segment/region.rs — Connected Component Labeling (Two-Pass + Union-Find)
// =============================================================================
//
// 将量化后的聚类索引图转换为连通区域。
//
// 算法: 经典两遍连通组件标记 (CCL), 基于 Union-Find 等价类合并。
// 连通性: 4-连通 (仅检查左邻和上邻)。
// 针对动漫图像中典型的大块平涂颜色区域优化 —— 区域少、面积大、
// 边界规整, Union-Find 合并次数极少。
//
// 流程:
//   1. Pass 1: 逐像素分配临时标签, 左/上邻同簇则合并等价类
//   2. Pass 2: 路径压缩解析最终标签, 去重编号
//   3. 按 min_region_area 过滤小区域
//   4. 统计每区域: 面积、质心、包围盒 (不在此阶段计算 LAB 均值 ——
//      extract_regions 仅接收聚类索引, 无原始像素 LAB 数据)

use crate::params::SegmentParams;

// =============================================================================
// Union-Find — 带路径压缩和按秩合并的不相交集
// =============================================================================

/// 不相交集数据结构，用于 CCL 等价类管理。
///
/// `parent[i]` 指向父节点，根节点的 `parent[i] == i`。
/// `rank[i]` 为秩上界，用于按秩合并控制树高。
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    /// 创建容量为 `size` 的空并查集 (每个元素自成一类)。
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    /// 查找 `x` 的根，沿途做路径压缩。
    fn find(&mut self, x: usize) -> usize {
        // 手动递归以避免 std 内部迭代器的潜在开销
        let mut cur = x;
        while self.parent[cur] != cur {
            self.parent[cur] = self.parent[self.parent[cur]]; // 路径压缩: 跳祖父
            cur = self.parent[cur];
        }
        cur
    }

    /// 合并 `x` 和 `y` 所在集合 (按秩合并)。
    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] = self.rank[rx].saturating_add(1);
            }
        }
    }

    /// 完成所有查找 → 路径压缩，并将根重编号为连续 0..K-1。
    ///
    /// 返回 `(new_labels, num_classes)`:
    /// - `new_labels[i]` = 原标签 `i` 对应的最终连续编号
    /// - `num_classes` = 独立等价类的总数
    fn resolve(&mut self, num_labels: usize) -> (Vec<usize>, usize) {
        // 强制全部路径压缩到根
        let roots: Vec<usize> = (0..num_labels).map(|i| self.find(i)).collect();

        // 将根重编号为 0, 1, 2, ...
        let mut root_to_new: Vec<Option<usize>> = vec![None; num_labels];
        let mut next_id: usize = 0;
        for &root in &roots {
            if root_to_new[root].is_none() {
                root_to_new[root] = Some(next_id);
                next_id += 1;
            }
        }

        let new_labels: Vec<usize> = roots
            .iter()
            .map(|&r| root_to_new[r].expect("root must be mapped"))
            .collect();

        (new_labels, next_id)
    }
}

// =============================================================================
// 输出类型
// =============================================================================

/// 单个连通区域的统计信息。
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// 最终连续区域 ID (0, 1, ...)
    pub id: usize,
    /// 该区域所属的聚类索引 (0..N-1, 对应 quantize 输出)
    pub cluster_id: usize,
    /// 区域面积 (像素数)
    pub area: usize,
    /// 质心坐标 (x, y), 亚像素精度
    pub centroid: (f64, f64),
    /// 包围盒 (min_x, min_y, max_x, max_y), 闭区间
    pub bbox: (u32, u32, u32, u32),
    /// 像素数 (与 area 相同, 保留以兼容下游代码)
    pub pixel_count: usize,
}

/// 完整区域图: 每像素的最终区域标签 + 区域统计。
///
/// `labels[idx] = Some(region_id)` 表示该像素属于 `regions[region_id]`。
/// `labels[idx] = None` 表示该像素所在区域因面积 < `min_region_area` 被过滤。
#[derive(Debug, Clone)]
pub struct RegionMap {
    /// 每像素的区域 ID (或 None, 若被过滤)
    pub labels: Vec<Option<usize>>,
    /// 所有保留区域的统计信息
    pub regions: Vec<Region>,
    /// 图像宽度
    pub width: u32,
    /// 图像高度
    pub height: u32,
}

// =============================================================================
// 主入口
// =============================================================================

/// 从量化聚类索引中提取连通区域 (精简版, 仅返回区域列表)。
///
/// 等价于 `extract_region_map(...).regions`。
pub fn extract_regions(
    cluster_indices: &[usize],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> Vec<Region> {
    extract_region_map(cluster_indices, width, height, params).regions
}

/// 从量化聚类索引中提取连通区域 (完整版, 含像素级标签映射)。
///
/// # 参数
/// - `cluster_indices`: 每像素的聚类分配 (0..N-1), 长度 = width * height
/// - `width`, `height`: 图像尺寸
/// - `params`: 分割参数, 仅使用 `params.min_region_area` 进行面积过滤
///
/// # 算法
/// 两遍连通组件标记 (4-连通), Union-Find 合并等价类。
pub fn extract_region_map(
    cluster_indices: &[usize],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> RegionMap {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;

    assert_eq!(
        cluster_indices.len(),
        total,
        "cluster_indices length must equal width * height"
    );

    // ===== Pass 1: 临时标签 + Union-Find 合并 =====
    //
    // 最多 total 个临时标签 (每个像素都可能独立)
    let mut uf = UnionFind::new(total);
    let mut provisional: Vec<usize> = vec![0; total];
    let mut next_label: usize = 0;

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let idx = row_offset + x;
            let cur = cluster_indices[idx];

            // 检查左邻 (4-连通)
            let left_label = if x > 0 && cluster_indices[idx - 1] == cur {
                Some(provisional[idx - 1])
            } else {
                None
            };

            // 检查上邻 (4-连通)
            let top_label = if y > 0 && cluster_indices[idx - w] == cur {
                Some(provisional[idx - w])
            } else {
                None
            };

            let label = match (left_label, top_label) {
                (Some(l), Some(t)) => {
                    // 左邻和上邻都是同簇 → 两者必须在同一等价类
                    if l != t {
                        uf.union(l, t);
                    }
                    l
                }
                (Some(l), None) => l,
                (None, Some(t)) => t,
                (None, None) => {
                    let l = next_label;
                    next_label += 1;
                    l
                }
            };

            provisional[idx] = label;
        }
    }

    // ===== Pass 2: 解析等价类 → 连续最终标签 =====
    let (label_map, num_final) = uf.resolve(next_label);

    let mut final_labels: Vec<usize> = vec![0; total];
    for i in 0..total {
        final_labels[i] = label_map[provisional[i]];
    }

    // ===== 统计: 面积 / 质心 / 包围盒 =====
    let mut areas: Vec<usize> = vec![0; num_final];
    let mut sum_x: Vec<f64> = vec![0.0; num_final];
    let mut sum_y: Vec<f64> = vec![0.0; num_final];
    let mut min_x: Vec<u32> = vec![width; num_final];
    let mut min_y: Vec<u32> = vec![height; num_final];
    let mut max_x: Vec<u32> = vec![0; num_final];
    let mut max_y: Vec<u32> = vec![0; num_final];
    let mut cluster_ids: Vec<usize> = vec![0; num_final];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let idx = row_offset + x;
            let label = final_labels[idx];

            areas[label] += 1;
            sum_x[label] += x as f64;
            sum_y[label] += y as f64;

            let ux = x as u32;
            let uy = y as u32;
            if ux < min_x[label] {
                min_x[label] = ux;
            }
            if ux > max_x[label] {
                max_x[label] = ux;
            }
            if uy < min_y[label] {
                min_y[label] = uy;
            }
            if uy > max_y[label] {
                max_y[label] = uy;
            }

            // 区域内所有像素簇号相同, 记录任意一个即可
            if areas[label] == 1 {
                cluster_ids[label] = cluster_indices[idx];
            }
        }
    }

    // ===== 过滤 + 重编号 =====
    let min_area = params.min_region_area;

    // 找出保留的区域 → 新 ID 映射
    let mut old_to_new: Vec<Option<usize>> = vec![None; num_final];
    let mut regions: Vec<Region> = Vec::new();

    for old_id in 0..num_final {
        if areas[old_id] >= min_area {
            let new_id = regions.len();
            old_to_new[old_id] = Some(new_id);
            regions.push(Region {
                id: new_id,
                cluster_id: cluster_ids[old_id],
                area: areas[old_id],
                pixel_count: areas[old_id],
                centroid: (
                    sum_x[old_id] / areas[old_id] as f64,
                    sum_y[old_id] / areas[old_id] as f64,
                ),
                bbox: (
                    min_x[old_id],
                    min_y[old_id],
                    max_x[old_id],
                    max_y[old_id],
                ),
            });
        }
    }

    // ===== 构建像素级标签映射 (过滤后) =====
    let labels: Vec<Option<usize>> = final_labels
        .iter()
        .map(|&old| old_to_new[old])
        .collect();

    RegionMap {
        labels,
        regions,
        width,
        height,
    }
}

// =============================================================================
// 单元测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SegmentParams;

    /// 辅助: 创建宽松参数 (min_region_area = 0, 不过滤)
    fn relaxed_params() -> SegmentParams {
        let mut p = SegmentParams::default();
        p.min_region_area = 0;
        p
    }

    // ===== 测试: 2×2 均匀图像 → 1 个区域 =====

    #[test]
    fn test_uniform_2x2_single_region() {
        // 全部像素属于 cluster 0
        let indices = vec![0, 0, 0, 0];
        let result = extract_region_map(&indices, 2, 2, &relaxed_params());

        assert_eq!(result.regions.len(), 1, "uniform image should produce 1 region");
        let r = &result.regions[0];
        assert_eq!(r.id, 0);
        assert_eq!(r.cluster_id, 0);
        assert_eq!(r.area, 4);
        assert_eq!(r.pixel_count, 4);
        assert_eq!(r.bbox, (0, 0, 1, 1));
        // 质心: (0+1+0+1)/4 = 0.5, (0+0+1+1)/4 = 0.5
        assert!((r.centroid.0 - 0.5).abs() < 1e-9);
        assert!((r.centroid.1 - 0.5).abs() < 1e-9);

        // 所有像素标签应一致
        assert!(result.labels.iter().all(|l| l == &Some(0)));
    }

    // ===== 测试: 棋盘格 → 9 个孤立区域 (4-连通) =====

    #[test]
    fn test_checkerboard_isolated_regions() {
        // 3×3 棋盘格: 4-连通下每个像素都是孤立区域
        // 0 1 0
        // 1 0 1
        // 0 1 0
        let indices = vec![
            0, 1, 0,
            1, 0, 1,
            0, 1, 0,
        ];
        let result = extract_region_map(&indices, 3, 3, &relaxed_params());

        // 每个像素孤立 → 9 个区域
        assert_eq!(result.regions.len(), 9, "3x3 checkerboard (4-connected) produces 9 isolated regions");

        // 所有区域面积 = 1
        for r in &result.regions {
            assert_eq!(r.area, 1);
            assert_eq!(r.pixel_count, 1);
            // 包围盒退化为单点
            assert_eq!(r.bbox.0, r.bbox.2);
            assert_eq!(r.bbox.1, r.bbox.3);
        }

        // cluster_id 分布: 5 个 0, 4 个 1
        let zeros = result.regions.iter().filter(|r| r.cluster_id == 0).count();
        let ones = result.regions.iter().filter(|r| r.cluster_id == 1).count();
        assert_eq!(zeros, 5);
        assert_eq!(ones, 4);

        // 验证标签: 每个像素都有唯一标签
        let unique_labels: std::collections::HashSet<usize> = result
            .labels
            .iter()
            .map(|l| l.unwrap())
            .collect();
        assert_eq!(unique_labels.len(), 9);
    }

    // ===== 测试: 连通区域正确分离 =====

    #[test]
    fn test_connected_regions_correct_labels() {
        // 4×4: 两个不连通的 cluster-0 区域包围一个 cluster-1 区域
        // 0 0 0 0
        // 0 1 1 0
        // 0 1 1 0
        // 0 0 0 0
        let indices = vec![
            0, 0, 0, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 0, 0, 0,
        ];
        let result = extract_region_map(&indices, 4, 4, &relaxed_params());

        // 4-连通: 外围 cluster-0 是一个连通区域 (面积 12), 中心 cluster-1 是另一个 (面积 4)
        assert_eq!(result.regions.len(), 2, "expected 2 connected regions");

        let outer: Vec<_> = result.regions.iter().filter(|r| r.cluster_id == 0).collect();
        let inner: Vec<_> = result.regions.iter().filter(|r| r.cluster_id == 1).collect();

        assert_eq!(outer.len(), 1);
        assert_eq!(inner.len(), 1);

        assert_eq!(outer[0].area, 12);
        assert_eq!(inner[0].area, 4);

        // 包围盒
        assert_eq!(outer[0].bbox, (0, 0, 3, 3));
        assert_eq!(inner[0].bbox, (1, 1, 2, 2));

        // 验证: 属于同一区域的像素有相同标签, 不同区域标签不同
        let outer_label = result.labels[0]; // (0,0) = cluster 0
        let inner_label = result.labels[5]; // (1,1) = cluster 1
        assert_ne!(outer_label, inner_label);

        // 外围所有 cluster-0 像素应有相同标签
        for y in 0..4 {
            for x in 0..4 {
                let idx = y * 4 + x;
                if indices[idx] == 0 {
                    assert_eq!(result.labels[idx], outer_label);
                } else {
                    assert_eq!(result.labels[idx], inner_label);
                }
            }
        }
    }

    // ===== 测试: min_region_area 过滤 =====

    #[test]
    fn test_min_region_area_filtering() {
        // 4×4: 外围 cluster-0 (面积 12), 中心 cluster-1 (面积 4)
        let indices = vec![
            0, 0, 0, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 0, 0, 0,
        ];

        // min_region_area = 5 → 过滤掉 cluster-1 (面积 4)
        let mut params = SegmentParams::default();
        params.min_region_area = 5;
        let result = extract_region_map(&indices, 4, 4, &params);

        assert_eq!(result.regions.len(), 1);
        assert_eq!(result.regions[0].cluster_id, 0);
        assert_eq!(result.regions[0].area, 12);

        // 被过滤的像素标签应为 None
        for y in 0..4 {
            for x in 0..4 {
                let idx = y * 4 + x;
                if indices[idx] == 0 {
                    assert_eq!(result.labels[idx], Some(0));
                } else {
                    assert_eq!(result.labels[idx], None);
                }
            }
        }

        // min_region_area = 13 → 全部过滤
        params.min_region_area = 13;
        let result = extract_region_map(&indices, 4, 4, &params);
        assert!(result.regions.is_empty());
        assert!(result.labels.iter().all(|l| l.is_none()));
    }

    // ===== 测试: 同簇不连通 → 不同标签 =====

    #[test]
    fn test_same_cluster_disconnected_different_labels() {
        // 2×4: cluster 0 在左侧和右侧, 中间隔开 (4-连通下不连通)
        // 0 0 1 0
        // 0 0 1 0
        let indices = vec![
            0, 0, 1, 0,
            0, 0, 1, 0,
        ];
        let result = extract_region_map(&indices, 4, 2, &relaxed_params());

        // 应产生 3 个区域: 左 cluster-0 (2×2=4), 中 cluster-1 (2×1=2), 右 cluster-0 (2×1=2)
        assert_eq!(result.regions.len(), 3);

        let left_label = result.labels[0];  // (0,0)
        let right_label = result.labels[3]; // (3,0)

        // 左右同簇但不同标签
        assert_eq!(result.regions[left_label.unwrap()].cluster_id, 0);
        assert_eq!(result.regions[right_label.unwrap()].cluster_id, 0);
        assert_ne!(left_label, right_label);

        // 左区域面积 = 4
        assert_eq!(result.regions[left_label.unwrap()].area, 4);
        // 右区域面积 = 2
        assert_eq!(result.regions[right_label.unwrap()].area, 2);
    }

    // ===== 测试: 空输入 (min_region_area 过滤掉所有) =====

    #[test]
    fn test_empty_result_on_total_filter() {
        let indices = vec![0, 1, 1, 0];
        let mut params = SegmentParams::default();
        params.min_region_area = 100;
        let result = extract_region_map(&indices, 2, 2, &params);
        assert!(result.regions.is_empty());
        assert_eq!(result.labels.len(), 4);
    }

    // ===== 测试: RegionMap 元数据 =====

    #[test]
    fn test_region_map_metadata() {
        let indices = vec![0; 9]; // 3x3 uniform
        let result = extract_region_map(&indices, 3, 3, &relaxed_params());
        assert_eq!(result.width, 3);
        assert_eq!(result.height, 3);
        assert_eq!(result.labels.len(), 9);
        assert_eq!(result.regions.len(), 1);
    }
}
