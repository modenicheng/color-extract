// =============================================================================
// Median Cut 颜色量化 — CIELAB 空间迭代中位切分
// =============================================================================
//
// 本模块实现经典的 Median Cut 颜色量化算法，在 CIELAB 色彩空间中
// 对 RGB 像素进行迭代切分，生成不超过 N 个颜色聚类。
//
// 算法流程：
//   1. 将 RGB 像素转换为 CIELAB L*a*b* 坐标
//   2. 初始化一个包含所有像素的聚类
//   3. 每次切分方差最大的聚类：沿包围盒最长轴，在中位数处一分为二
//   4. 重复直到达到 max_clusters 或所有聚类方差低于 variance_threshold

use palette::{FromColor, IntoColor, Lab, Srgb};

// =============================================================================
// 类型定义
// =============================================================================

/// 单个像素的完整信息 —— 原始索引、LAB 坐标、RGB 原始值
#[derive(Debug, Clone)]
pub struct Pixel {
    /// 在原像素数组中的位置
    pub index: usize,
    /// CIELAB L*a*b* 坐标
    pub lab: [f64; 3],
    /// 原始归一化 sRGB [0, 1]
    pub rgb: [f64; 3],
}

/// 一个颜色聚类 —— 在 LAB 空间中被 Median Cut 切分出的像素集合
#[derive(Debug, Clone)]
pub struct Cluster {
    /// 聚类质心（LAB 空间均值）
    pub centroid: [f64; 3],
    /// 成员像素在原数组中的索引
    pub members: Vec<usize>,
    /// LAB 空间包围盒 (min_l, max_l, min_a, max_a, min_b, max_b)
    pub bbox: (f64, f64, f64, f64, f64, f64),
    /// 聚类内方差 —— 成员到质心的平均平方欧氏距离
    pub variance: f64,
}

/// 量化结果的 sRGB 调色盘
#[derive(Debug, Clone)]
pub struct Palette {
    /// sRGB 显示色 [0, 255]
    pub colors: Vec<[u8; 3]>,
    /// 每个颜色对应的像素计数
    pub counts: Vec<usize>,
}

// =============================================================================
// 前向引用 —— 从 params 模块导入 SegmentParams
// =============================================================================

pub use crate::params::SegmentParams;

// =============================================================================
// RGB ↔ LAB 转换辅助函数
// =============================================================================

/// 归一化 RGB [0, 1] → CIELAB L*a*b*
fn rgb_to_lab(rgb: &[f64; 3]) -> [f64; 3] {
    let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
    let lab: Lab = srgb.into_color();
    [lab.l as f64, lab.a as f64, lab.b as f64]
}

/// CIELAB L*a*b* → 归一化 RGB [0, 1]
#[allow(dead_code)]
fn lab_to_rgb(lab: &[f64; 3]) -> [f64; 3] {
    let lab_color = Lab::new(lab[0] as f32, lab[1] as f32, lab[2] as f32);
    let srgb: Srgb = Srgb::from_color(lab_color);
    [srgb.red as f64, srgb.green as f64, srgb.blue as f64]
}

// =============================================================================
// 聚类操作辅助函数
// =============================================================================

/// 计算聚类的质心（LAB 均值）和包围盒
///
/// 遍历所有成员像素，累计 LAB 各通道均值和 min/max，
/// 将结果写入 `cluster.centroid` 和 `cluster.bbox`。
pub fn compute_centroid(cluster: &mut Cluster, pixels: &[Pixel]) {
    if cluster.members.is_empty() {
        cluster.centroid = [0.0; 3];
        cluster.bbox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    let n = cluster.members.len() as f64;
    let mut sum_l = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;

    let mut min_l = f64::MAX;
    let mut max_l = f64::MIN;
    let mut min_a = f64::MAX;
    let mut max_a = f64::MIN;
    let mut min_b = f64::MAX;
    let mut max_b = f64::MIN;

    for &idx in &cluster.members {
        let lab = pixels[idx].lab;
        sum_l += lab[0];
        sum_a += lab[1];
        sum_b += lab[2];
        min_l = min_l.min(lab[0]);
        max_l = max_l.max(lab[0]);
        min_a = min_a.min(lab[1]);
        max_a = max_a.max(lab[1]);
        min_b = min_b.min(lab[2]);
        max_b = max_b.max(lab[2]);
    }

    cluster.centroid = [sum_l / n, sum_a / n, sum_b / n];
    cluster.bbox = (min_l, max_l, min_a, max_a, min_b, max_b);
}

/// 计算聚类内方差 —— 成员到质心的平均平方欧氏距离
///
/// 假定 `cluster.centroid` 已经由 `compute_centroid` 计算完成。
pub fn compute_variance(cluster: &Cluster, pixels: &[Pixel]) -> f64 {
    if cluster.members.is_empty() {
        return 0.0;
    }

    let [cl, ca, cb] = cluster.centroid;
    let n = cluster.members.len() as f64;

    let sum_sq: f64 = cluster
        .members
        .iter()
        .map(|&idx| {
            let [l, a, b] = pixels[idx].lab;
            let dl = l - cl;
            let da = a - ca;
            let db = b - cb;
            dl * dl + da * da + db * db
        })
        .sum();

    sum_sq / n
}

/// 将聚类沿包围盒最长轴在中位数处一分为二
///
/// 返回两个新的聚类（左/右），原聚类的 `members` 被清空。
///
/// 分割轴选择：比较包围盒各维度的跨度 (max - min)，取最大者。
/// 分割点：沿选定轴排序后取中位索引 split_off。
pub fn split_cluster(cluster: &mut Cluster, pixels: &[Pixel]) -> (Cluster, Cluster) {
    let (min_l, max_l, min_a, max_a, min_b, max_b) = cluster.bbox;
    let range_l = max_l - min_l;
    let range_a = max_a - min_a;
    let range_b = max_b - min_b;

    // 选择跨度最大的维度作为切分轴
    let axis: usize = if range_l >= range_a && range_l >= range_b {
        0 // L*
    } else if range_a >= range_b {
        1 // a*
    } else {
        2 // b*
    };

    // 沿选定轴排序成员
    cluster.members.sort_by(|&a, &b| {
        pixels[a].lab[axis]
            .partial_cmp(&pixels[b].lab[axis])
            .unwrap()
    });

    // 在中位数处切分
    let mid = cluster.members.len() / 2;
    let right_members = cluster.members.split_off(mid);
    let left_members = std::mem::take(&mut cluster.members);

    let mut c1 = Cluster {
        centroid: [0.0; 3],
        members: left_members,
        bbox: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        variance: 0.0,
    };
    let mut c2 = Cluster {
        centroid: [0.0; 3],
        members: right_members,
        bbox: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        variance: 0.0,
    };

    compute_centroid(&mut c1, pixels);
    compute_centroid(&mut c2, pixels);
    c1.variance = compute_variance(&c1, pixels);
    c2.variance = compute_variance(&c2, pixels);

    // 清空原聚类
    cluster.members.clear();
    cluster.variance = 0.0;
    cluster.centroid = [0.0; 3];
    cluster.bbox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    (c1, c2)
}

/// 将聚类列表转换为 sRGB 调色盘
///
/// 每个聚类的颜色由其成员的原始 RGB 均值计算（避免 LAB←→RGB 往返精度损失）。
pub fn to_rgb_palette(clusters: &[Cluster], pixels: &[Pixel]) -> Palette {
    let mut colors: Vec<[u8; 3]> = Vec::with_capacity(clusters.len());
    let mut counts: Vec<usize> = Vec::with_capacity(clusters.len());

    for cluster in clusters {
        if cluster.members.is_empty() {
            colors.push([0, 0, 0]);
            counts.push(0);
            continue;
        }

        let n = cluster.members.len() as f64;
        let (sum_r, sum_g, sum_b) =
            cluster
                .members
                .iter()
                .fold((0.0f64, 0.0f64, 0.0f64), |(sr, sg, sb), &idx| {
                    let rgb = pixels[idx].rgb;
                    (sr + rgb[0], sg + rgb[1], sb + rgb[2])
                });

        colors.push([
            ((sum_r / n).clamp(0.0, 1.0) * 255.0).round() as u8,
            ((sum_g / n).clamp(0.0, 1.0) * 255.0).round() as u8,
            ((sum_b / n).clamp(0.0, 1.0) * 255.0).round() as u8,
        ]);
        counts.push(cluster.members.len());
    }

    Palette { colors, counts }
}

// =============================================================================
// 主入口：Median Cut 量化
// =============================================================================

/// 对归一化 RGB 像素执行 Median Cut 颜色量化
///
/// # 参数
///
/// * `pixels` - 归一化 RGB 像素切片，每个分量为 [0, 1]
/// * `params` - 分割参数（max_clusters、variance_threshold 等）
///
/// # 返回
///
/// 聚类列表，每个 Cluster 包含 LAB 空间质心、成员索引和包围盒。
/// 聚类数量 ≤ `params.max_clusters`，且所有聚类的 variance ≥ `params.variance_threshold`。
pub fn quantize(pixels: &[[f64; 3]], params: &SegmentParams) -> Vec<Cluster> {
    if pixels.is_empty() {
        return vec![];
    }

    let n = pixels.len();

    // ── 步骤 1: RGB → LAB 转换 ──
    let lab_pixels: Vec<Pixel> = pixels
        .iter()
        .enumerate()
        .map(|(i, rgb)| Pixel {
            index: i,
            lab: rgb_to_lab(rgb),
            rgb: *rgb,
        })
        .collect();

    // ── 步骤 2: 初始化单个聚类，包含全部像素 ──
    let mut root = Cluster {
        centroid: [0.0; 3],
        members: (0..n).collect(),
        bbox: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        variance: 0.0,
    };
    compute_centroid(&mut root, &lab_pixels);
    root.variance = compute_variance(&root, &lab_pixels);

    let mut clusters = vec![root];

    // ── 步骤 3-4: 迭代切分最高方差聚类 ──
    while clusters.len() < params.max_clusters {
        // 找到方差最大的聚类
        let (split_idx, _) = clusters
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.variance.partial_cmp(&b.variance).unwrap())
            .unwrap();

        // 终止条件：方差低于阈值
        if clusters[split_idx].variance < params.variance_threshold {
            break;
        }

        // 终止条件：成员不足 2 个，无法切分
        if clusters[split_idx].members.len() < 2 {
            break;
        }

        let (c1, c2) = split_cluster(&mut clusters[split_idx], &lab_pixels);
        clusters[split_idx] = c1;
        clusters.push(c2);
    }

    clusters
}

// =============================================================================
// 单元测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助：创建默认参数（宽松阈值，允许较多切分）
    fn default_params() -> SegmentParams {
        SegmentParams {
            max_clusters: 8,
            variance_threshold: 0.01,
            ..SegmentParams::default()
        }
    }

    // ── 基础正确性 ──

    #[test]
    fn test_empty_input() {
        let pixels: Vec<[f64; 3]> = vec![];
        let params = default_params();
        let clusters = quantize(&pixels, &params);
        assert!(clusters.is_empty(), "empty input → no clusters");
    }

    #[test]
    fn test_all_identical_pixels() {
        // 全部相同像素 → 方差≈0 → 低于阈值停止切分 → 1 个聚类
        let pixels: Vec<[f64; 3]> = vec![[0.3, 0.6, 0.9]; 100];
        let params = default_params();
        let clusters = quantize(&pixels, &params);
        assert_eq!(clusters.len(), 1, "identical pixels → single cluster");
        assert_eq!(clusters[0].members.len(), 100);
    }

    #[test]
    fn test_clusters_non_empty() {
        let pixels: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let t = i as f64 / 200.0;
                [t, 1.0 - t, 0.5]
            })
            .collect();
        let params = default_params();
        let clusters = quantize(&pixels, &params);

        for cluster in &clusters {
            assert!(
                !cluster.members.is_empty(),
                "every cluster must have members"
            );
        }

        // 总成员数 = 总像素数
        let total: usize = clusters.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, pixels.len(), "no pixels lost during quantization");
    }

    // ── 质心范围验证 ──

    #[test]
    fn test_centroids_in_valid_lab_range() {
        let pixels: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let t = i as f64 / 200.0;
                [t * 0.8, (1.0 - t) * 0.6, 0.3 + t * 0.5]
            })
            .collect();
        let params = default_params();
        let clusters = quantize(&pixels, &params);

        for cluster in &clusters {
            let [l, a, b] = cluster.centroid;
            assert!((0.0..=100.0).contains(&l), "L* {} not in [0, 100]", l);
            assert!((-128.0..=128.0).contains(&a), "a* {} not in [-128, 128]", a);
            assert!((-128.0..=128.0).contains(&b), "b* {} not in [-128, 128]", b);
        }
    }

    // ── 聚类数量约束 ──

    #[test]
    fn test_respects_max_clusters() {
        // 高度分散的像素 → 每个像素都是独立颜色
        let pixels: Vec<[f64; 3]> = (0..500)
            .map(|i| {
                let t = i as f64 / 500.0;
                // 在 RGB 空间中均匀分布的色块
                [t, (t * 1.3) % 1.0, (t * 0.7) % 1.0]
            })
            .collect();

        let params = SegmentParams {
            max_clusters: 4,
            variance_threshold: 0.0, // 不因方差提前终止
            ..SegmentParams::default()
        };

        let clusters = quantize(&pixels, &params);
        assert!(
            clusters.len() <= 4,
            "must not exceed max_clusters (got {})",
            clusters.len()
        );
    }

    #[test]
    fn test_respects_variance_threshold() {
        // 几乎相同的像素 → 方差极小
        let mut pixels: Vec<[f64; 3]> = Vec::with_capacity(100);
        for i in 0..100 {
            let tiny = i as f64 * 0.0001;
            pixels.push([0.5 + tiny, 0.5, 0.5]);
        }

        let params = SegmentParams {
            max_clusters: 16,
            variance_threshold: 10.0, // 极高阈值 → 任何方差都低于此
            ..SegmentParams::default()
        };

        let clusters = quantize(&pixels, &params);
        assert_eq!(clusters.len(), 1, "high variance_threshold → no splitting");
    }

    // ── Median Cut 正确分离不同颜色 ──

    #[test]
    fn test_separates_two_distinct_colors() {
        let mut pixels = Vec::with_capacity(200);
        // 红色区域 (在 LAB 中远离蓝色)
        for _ in 0..100 {
            pixels.push([1.0, 0.0, 0.0]);
        }
        // 蓝色区域
        for _ in 0..100 {
            pixels.push([0.0, 0.0, 1.0]);
        }

        let params = SegmentParams {
            max_clusters: 8,
            variance_threshold: 0.1,
            ..SegmentParams::default()
        };

        let clusters = quantize(&pixels, &params);

        // 红和蓝在 LAB 中距离很远 → 至少切出 2 个聚类
        assert!(
            clusters.len() >= 2,
            "should separate red and blue (got {} clusters)",
            clusters.len()
        );

        // 验证两个聚类的质心在 LAB 空间中明显不同
        // 红蓝在 b* 轴上差距很大：红 b*≈+67，蓝 b*≈-108
        if clusters.len() >= 2 {
            let c0 = clusters[0].centroid;
            let c1 = clusters[1].centroid;
            let d_b = (c0[2] - c1[2]).abs();
            assert!(
                d_b > 20.0,
                "red and blue should separate along b* axis (Δb = {}), centroids: ({:.1}, {:.1}, {:.1}) vs ({:.1}, {:.1}, {:.1})",
                d_b,
                c0[0],
                c0[1],
                c0[2],
                c1[0],
                c1[1],
                c1[2],
            );
        }
    }

    #[test]
    fn test_separates_three_distinct_colors() {
        let mut pixels = Vec::with_capacity(300);
        for _ in 0..100 {
            pixels.push([1.0, 0.0, 0.0]);
        } // red
        for _ in 0..100 {
            pixels.push([0.0, 1.0, 0.0]);
        } // green
        for _ in 0..100 {
            pixels.push([0.0, 0.0, 1.0]);
        } // blue

        let params = SegmentParams {
            max_clusters: 8,
            variance_threshold: 0.1,
            ..SegmentParams::default()
        };

        let clusters = quantize(&pixels, &params);
        assert!(
            clusters.len() >= 3,
            "should separate all 3 primary colors (got {} clusters)",
            clusters.len()
        );
    }

    // ── Palette 生成 ──

    #[test]
    fn test_to_rgb_palette() {
        let pixels: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let params = SegmentParams {
            max_clusters: 2,
            variance_threshold: 0.1,
            ..SegmentParams::default()
        };

        let clusters = quantize(&pixels, &params);
        let lab_pixels: Vec<Pixel> = pixels
            .iter()
            .enumerate()
            .map(|(i, rgb)| Pixel {
                index: i,
                lab: rgb_to_lab(rgb),
                rgb: *rgb,
            })
            .collect();
        let palette = to_rgb_palette(&clusters, &lab_pixels);

        assert_eq!(palette.colors.len(), clusters.len());
        assert_eq!(palette.counts.len(), clusters.len());

        // 每个聚类的 counts 应等于其 members 长度
        for (i, cluster) in clusters.iter().enumerate() {
            assert_eq!(palette.counts[i], cluster.members.len());
        }

        // 每个颜色分量在 [0, 255] 内
        for color in &palette.colors {
            assert!((0..=255).contains(&color[0]));
            assert!((0..=255).contains(&color[1]));
            assert!((0..=255).contains(&color[2]));
        }
    }

    #[test]
    fn test_single_pixel() {
        let pixels = vec![[0.2, 0.4, 0.6]];
        let params = default_params();
        let clusters = quantize(&pixels, &params);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 1);
        assert_eq!(clusters[0].members, vec![0]);
    }
}
