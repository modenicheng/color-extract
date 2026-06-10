// =============================================================================
// color-segment/refine.rs — Edge-aware region boundary refinement
// =============================================================================
//
// 智能层：利用边缘检测数据改进区域边界。包含两个阶段：
//
//   Phase A (Merge): 合并由弱边缘分隔的相邻区域。
//     量化产生的伪边界在边缘检测中响应弱 → 合并为同一区域。
//
//   Phase B (Split):  分裂包含强内部边缘的区域。
//     统一区域内部的强边缘表明量化遗漏了颜色边界 → 沿边缘分裂。
//
// 针对动漫平涂风格优化：量化噪声形成的弱边界被吸收，漏检的颜色
// 边界被恢复。
//
// 流程:
//   1. Phase A — 构建区域邻接图，按边缘强度判断合并
//   2. Phase B — 逐区域检测内部强边缘，形态学膨胀 → CCL 分裂
//   3. 吸收极小分裂子区域（面积 < min_region_area）
//   4. 重建区域统计（面积、质心、包围盒）

use crate::params::SegmentParams;
use crate::region::Region;
use std::collections::HashMap;

// =============================================================================
// Simple Union-Find (inline — not using region.rs private UnionFind)
// =============================================================================

/// 轻量 Union-Find，带路径压缩。
///
/// 用于合并阶段的区域等价类管理和分裂阶段的 CCL。
struct Uf {
    parent: Vec<usize>,
}

impl Uf {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut cur = x;
        while self.parent[cur] != cur {
            self.parent[cur] = self.parent[self.parent[cur]];
            cur = self.parent[cur];
        }
        cur
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[ry] = rx;
        }
    }

    /// 将所有根重编号为 0..K-1，返回 (new_id_map, num_classes)。
    fn resolve(&mut self, num_items: usize) -> (Vec<usize>, usize) {
        // 强制路径压缩
        for i in 0..num_items {
            self.find(i);
        }
        // 收集根并重编号
        let mut root_to_new: Vec<Option<usize>> = vec![None; num_items];
        let mut next_id = 0;
        for i in 0..num_items {
            let r = self.parent[i];
            if root_to_new[r].is_none() {
                root_to_new[r] = Some(next_id);
                next_id += 1;
            }
        }
        let new_ids: Vec<usize> = (0..num_items)
            .map(|i| root_to_new[self.parent[i]].expect("root must be mapped"))
            .collect();
        (new_ids, next_id)
    }
}

#[derive(Debug, Clone, Copy)]
struct AdjacentPair {
    a: usize,
    b: usize,
    mean_edge: f64,
    boundary_len: usize,
}

// =============================================================================
// Public API
// =============================================================================

/// 边缘感知区域边界精炼：合并 + 分裂两阶段。
///
/// # 参数
/// - `regions`: CCL 阶段提取的连通区域列表
/// - `labels`: 每像素区域标签（`Some(region_id)` 或 `None`）
/// - `edge_strength`: 边缘检测输出，每像素 [0, 1]，值越高边缘越强
/// - `width`, `height`: 图像尺寸
/// - `params`: 分割参数，使用 `edge_merge_strength`、`edge_split_strength`、`min_region_area`
///
/// # 返回
/// `(regions, labels)` — 精炼后的区域列表和像素级标签映射。
///
/// # 算法
///
/// **Phase A — MERGE**: 扫描所有内部像素的 4-邻域，收集不同区域间的
/// 邻接对及边界边缘强度。若某对区域间的平均边缘强度 ≤ `edge_merge_strength`，
/// 则该边界为量化伪影 → 合并两区域。
///
/// **Phase B — SPLIT**: 对每个（合并后）区域，找出内部边缘强度 ≥
/// `edge_split_strength` 的像素，经形态学膨胀（1px 半径）创建分水岭屏障，
/// 然后在区域内剩余像素上运行 CCL 以生成子区域。面积 < `min_region_area`
/// 的子区域被吸收至最近邻。
pub fn refine(
    regions: &[Region],
    labels: &[Option<usize>],
    edge_strength: &[f64],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>) {
    refine_impl(regions, labels, edge_strength, None, width, height, params)
}

/// 边缘 + 颜色感知区域精炼。
///
/// `cluster_centroids` 是 quantize 阶段输出的 CIELAB 质心，用于判断相邻区域
/// 是否只是被量化过程切得过细。颜色近、共享边界不强的区域会被合并。
pub fn refine_with_colors(
    regions: &[Region],
    labels: &[Option<usize>],
    edge_strength: &[f64],
    cluster_centroids: &[[f64; 3]],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>) {
    refine_impl(
        regions,
        labels,
        edge_strength,
        Some(cluster_centroids),
        width,
        height,
        params,
    )
}

fn refine_impl(
    regions: &[Region],
    labels: &[Option<usize>],
    edge_strength: &[f64],
    cluster_centroids: Option<&[[f64; 3]]>,
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>) {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;

    assert_eq!(
        labels.len(),
        total,
        "labels length must equal width * height"
    );
    assert_eq!(
        edge_strength.len(),
        total,
        "edge_strength length must equal width * height"
    );

    if regions.is_empty() {
        return (Vec::new(), vec![None; total]);
    }

    // ===== Phase A — Merge =====
    let (merged_regions, merged_labels, _old_to_new) = perform_merge(
        regions,
        labels,
        edge_strength,
        cluster_centroids,
        width,
        height,
        params,
    );

    // ===== Phase B — Split =====
    let (final_regions, final_labels) = perform_split(
        &merged_regions,
        &merged_labels,
        edge_strength,
        width,
        height,
        params,
    );

    (final_regions, final_labels)
}

// =============================================================================
// Phase A — Merge: 弱边缘邻接区域合并
// =============================================================================

/// 执行合并阶段：构建邻接图、按边缘强度合并、重建区域。
fn perform_merge(
    regions: &[Region],
    labels: &[Option<usize>],
    edge_strength: &[f64],
    cluster_centroids: Option<&[[f64; 3]]>,
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>, Vec<usize>) {
    let w = width as usize;
    let _h = height as usize;
    let total = w * _h;

    let num_regions = regions.len();

    // 构建邻接: 每对区域共享边界上的平均边缘强度
    let pairs = compute_adjacency_pairs(labels, edge_strength, width, height);

    // Union-Find 合并
    let mut uf = Uf::new(num_regions);
    for pair in &pairs {
        if should_merge_pair(pair, regions, cluster_centroids, params) {
            uf.union(pair.a, pair.b);
        }
    }
    absorb_small_regions(&mut uf, regions, &pairs, cluster_centroids, params);

    let (old_to_new, num_merged) = uf.resolve(num_regions);

    // 重建合并后区域统计
    let mut merged_regions: Vec<Region> = (0..num_merged)
        .map(|new_id| Region {
            id: new_id,
            cluster_id: 0, // 合并后 cluster_id 取第一个区域的
            area: 0,
            centroid: (0.0, 0.0),
            bbox: (width, height, 0, 0),
            pixel_count: 0,
        })
        .collect();

    // 累加统计
    for old_id in 0..num_regions {
        let new_id = old_to_new[old_id];
        let r = &regions[old_id];
        let mr = &mut merged_regions[new_id];

        // cluster_id: 取最大区域的 cluster_id
        if r.area > mr.area {
            mr.cluster_id = r.cluster_id;
        }

        mr.area += r.area;
        mr.pixel_count += r.pixel_count;

        // 质心: 加权合并
        let w_old = r.area as f64;
        let w_new = mr.area as f64 - w_old;
        if mr.area > r.area {
            // 已累加过，回退计算合并质心
            mr.centroid.0 = (mr.centroid.0 * w_new + r.centroid.0 * w_old) / mr.area as f64;
            mr.centroid.1 = (mr.centroid.1 * w_new + r.centroid.1 * w_old) / mr.area as f64;
        } else {
            mr.centroid = r.centroid;
        }

        // 包围盒: 扩展
        mr.bbox.0 = mr.bbox.0.min(r.bbox.0);
        mr.bbox.1 = mr.bbox.1.min(r.bbox.1);
        mr.bbox.2 = mr.bbox.2.max(r.bbox.2);
        mr.bbox.3 = mr.bbox.3.max(r.bbox.3);
    }

    // 重新编号 id
    for (i, mr) in merged_regions.iter_mut().enumerate() {
        mr.id = i;
    }

    // 更新像素标签
    let merged_labels: Vec<Option<usize>> = (0..total)
        .map(|i| labels[i].map(|old_id| old_to_new[old_id]))
        .collect();

    (merged_regions, merged_labels, old_to_new)
}

/// 判断一对相邻区域是否应合并。
///
/// 规则分两层：
/// - 边界非常弱：视为量化伪边界，直接合并；
/// - 颜色足够接近且边界没有达到“明确分割线”的强度：合并。
fn should_merge_pair(
    pair: &AdjacentPair,
    regions: &[Region],
    cluster_centroids: Option<&[[f64; 3]]>,
    params: &SegmentParams,
) -> bool {
    if pair.mean_edge <= params.edge_merge_strength {
        return true;
    }

    let Some(color_dist) =
        region_color_distance(&regions[pair.a], &regions[pair.b], cluster_centroids)
    else {
        return false;
    };

    let color_edge_limit = (params.edge_split_strength * 0.75)
        .max(params.edge_merge_strength)
        .min(1.0);

    color_dist <= params.color_merge_distance && pair.mean_edge <= color_edge_limit
}

/// 将小区域吸收到最合适的邻接区域，减少孤立碎片。
fn absorb_small_regions(
    uf: &mut Uf,
    regions: &[Region],
    pairs: &[AdjacentPair],
    cluster_centroids: Option<&[[f64; 3]]>,
    params: &SegmentParams,
) {
    if !params.merge_small_regions || params.min_region_area == 0 {
        return;
    }

    let mut small_ids: Vec<usize> = regions
        .iter()
        .filter(|r| r.area < params.min_region_area)
        .map(|r| r.id)
        .collect();
    small_ids.sort_by_key(|&rid| regions[rid].area);

    for rid in small_ids {
        let root = uf.find(rid);
        let mut best: Option<(usize, f64)> = None;

        for pair in pairs {
            let other = if pair.a == rid {
                pair.b
            } else if pair.b == rid {
                pair.a
            } else {
                continue;
            };

            if uf.find(other) == root {
                continue;
            }

            let color_dist =
                region_color_distance(&regions[rid], &regions[other], cluster_centroids);
            let color_ok = color_dist
                .map(|d| d <= params.small_region_color_distance)
                .unwrap_or(false);
            let weak_edge_ok = pair.mean_edge <= params.edge_merge_strength;
            let tiny_soft_edge_ok = regions[rid].area <= (params.min_region_area / 4).max(1)
                && pair.mean_edge <= params.edge_split_strength;

            if !(color_ok || weak_edge_ok || tiny_soft_edge_ok) {
                continue;
            }

            let color_score = color_dist.unwrap_or(params.small_region_color_distance);
            let boundary_bonus = (pair.boundary_len as f64).sqrt();
            let score = color_score + pair.mean_edge * 16.0 - boundary_bonus;

            match best {
                Some((_, best_score)) if best_score <= score => {}
                _ => best = Some((other, score)),
            }
        }

        if let Some((target, _)) = best {
            uf.union(rid, target);
        }
    }
}

fn region_color_distance(
    a: &Region,
    b: &Region,
    cluster_centroids: Option<&[[f64; 3]]>,
) -> Option<f64> {
    if a.cluster_id == b.cluster_id {
        return Some(0.0);
    }

    let centroids = cluster_centroids?;
    let ca = centroids.get(a.cluster_id)?;
    let cb = centroids.get(b.cluster_id)?;
    Some(lab_distance(*ca, *cb))
}

fn lab_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// 构建区域邻接三元组列表: (r1, r2, mean_edge_strength), r1 < r2。
///
/// 扫描所有内部像素，对每个像素检查右邻和下邻。若两像素属于不同区域，
/// 记录该区域对的边界边缘强度（取两像素边缘强度的均值）。
fn compute_adjacency_pairs(
    labels: &[Option<usize>],
    edge_strength: &[f64],
    width: u32,
    height: u32,
) -> Vec<AdjacentPair> {
    let w = width as usize;
    let h = height as usize;
    let mut accum: HashMap<(usize, usize), (f64, usize)> = HashMap::new();

    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let idx = row + x;
            let id_a = match labels[idx] {
                Some(id) => id,
                None => continue,
            };

            if x + 1 < w {
                push_adjacency_sample(&mut accum, labels, edge_strength, idx, idx + 1, id_a);
            }
            if y + 1 < h {
                push_adjacency_sample(&mut accum, labels, edge_strength, idx, idx + w, id_a);
            }
        }
    }

    let mut pairs: Vec<AdjacentPair> = accum
        .into_iter()
        .map(|((a, b), (sum, count))| AdjacentPair {
            a,
            b,
            mean_edge: sum / count as f64,
            boundary_len: count,
        })
        .collect();
    pairs.sort_by_key(|p| (p.a, p.b));
    pairs
}

fn push_adjacency_sample(
    accum: &mut HashMap<(usize, usize), (f64, usize)>,
    labels: &[Option<usize>],
    edge_strength: &[f64],
    idx_a: usize,
    idx_b: usize,
    id_a: usize,
) {
    let Some(id_b) = labels[idx_b] else {
        return;
    };
    if id_a == id_b {
        return;
    }

    let (a, b) = if id_a < id_b {
        (id_a, id_b)
    } else {
        (id_b, id_a)
    };
    let mean_e = (edge_strength[idx_a] + edge_strength[idx_b]) * 0.5;
    let entry = accum.entry((a, b)).or_insert((0.0, 0));
    entry.0 += mean_e;
    entry.1 += 1;
}

fn compute_adjacency_triples(
    labels: &[Option<usize>],
    edge_strength: &[f64],
    width: u32,
    height: u32,
    _num_regions: usize,
) -> Vec<(usize, usize, f64)> {
    compute_adjacency_pairs(labels, edge_strength, width, height)
        .into_iter()
        .map(|p| (p.a, p.b, p.mean_edge))
        .collect()
}

/// 邻接列表格式（用于测试/诊断），每个区域返回 [(neighbor_id, mean_edge_strength)]。
///
/// 注意：每条边出现两次（每个方向一次）。
#[allow(dead_code)]
fn compute_region_adjacency(
    labels: &[Option<usize>],
    width: u32,
    height: u32,
    num_regions: usize,
    edge_strength: &[f64],
) -> Vec<Vec<(usize, f64)>> {
    let triples = compute_adjacency_triples(labels, edge_strength, width, height, num_regions);
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_regions];
    for &(a, b, mean) in &triples {
        adj[a].push((b, mean));
        adj[b].push((a, mean));
    }
    adj
}

// =============================================================================
// Phase B — Split: 强内部边缘区域分裂
// =============================================================================

/// 执行分裂阶段：对每个合并后区域，检测内部强边缘并分裂。
///
/// 对每个区域调用 `split_region`，获取子区域列表和像素→子区域映射，
/// 再统一重编号到全局区域空间。
fn perform_split(
    regions: &[Region],
    labels: &[Option<usize>],
    edge_strength: &[f64],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>) {
    let total = (width as usize) * (height as usize);

    let mut final_regions: Vec<Region> = Vec::new();
    let mut final_labels: Vec<Option<usize>> = vec![None; total];

    for region in regions {
        let (sub_regions, sub_label_map) =
            split_region(region, labels, edge_strength, width, height, params);

        if sub_regions.len() <= 1 {
            // 无需分裂，原样保留
            let new_id = final_regions.len();
            let mut r = if sub_regions.is_empty() {
                region.clone()
            } else {
                sub_regions[0].clone()
            };
            r.id = new_id;
            final_regions.push(r);
            for i in 0..total {
                if labels[i] == Some(region.id) {
                    final_labels[i] = Some(new_id);
                }
            }
        } else {
            // 分裂为多个子区域，分配全局 ID
            let base_id = final_regions.len();
            for sub in &sub_regions {
                let mut r = sub.clone();
                r.id = final_regions.len();
                final_regions.push(r);
            }
            for i in 0..total {
                if let Some(sub_idx) = sub_label_map[i] {
                    final_labels[i] = Some(base_id + sub_idx);
                }
            }
        }
    }

    (final_regions, final_labels)
}

/// 对单个区域执行内部边缘分裂。
///
/// # 算法
/// 1. 找出区域内部边缘强度 ≥ `edge_split_strength` 的像素
/// 2. 形态学膨胀（1px 半径）创建分水岭屏障
/// 3. 在区域非屏障像素上运行 4-连通 CCL
/// 4. 吸收面积 < `min_region_area` 的子区域至最近邻
///
/// # 返回
/// `(sub_regions, label_map)`:
/// - `sub_regions`: 最终子区域列表（已吸收极小区域并重编号）
/// - `label_map[i]`: 像素 i 在 sub_regions 中的索引（仅对属于原区域的像素为 `Some`）
fn split_region(
    region: &Region,
    labels: &[Option<usize>],
    edge_strength: &[f64],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>) {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    let region_id = region.id;

    let empty_result = (vec![region.clone()], vec![None; total]);

    // ===== Step 1: 收集区域像素 =====
    let mut region_pixels: Vec<usize> = Vec::new();
    for i in 0..total {
        if labels[i] == Some(region_id) {
            region_pixels.push(i);
        }
    }

    if region_pixels.is_empty() {
        return empty_result;
    }

    // ===== Step 2: 标记强边缘像素 =====
    let mut barrier: Vec<bool> = vec![false; total];
    let mut has_strong_edge = false;
    for &i in &region_pixels {
        if edge_strength[i] >= params.edge_split_strength {
            barrier[i] = true;
            has_strong_edge = true;
        }
    }

    if !has_strong_edge {
        return empty_result;
    }

    // ===== Step 3: 形态学膨胀 (1px) =====
    dilate_boundary(&mut barrier, width, height, 1);

    // ===== Step 4: 区域内 CCL (跳过屏障像素) =====
    let raw_sub_labels = ccl_within_region(&region_pixels, &barrier, width, height);

    // 统计子区域
    let num_subs = raw_sub_labels.iter().max().map(|&m| m + 1).unwrap_or(0);

    if num_subs <= 1 {
        return empty_result;
    }

    // ===== Step 5: 收集每个原始子区域的像素与统计 =====
    struct SubCandidate {
        pixels: Vec<usize>,
        centroid: (f64, f64),
        bbox: (u32, u32, u32, u32),
        area: usize,
    }

    let mut candidates: Vec<SubCandidate> = Vec::new();
    for sub_id in 0..num_subs {
        let sub_pixels: Vec<usize> = region_pixels
            .iter()
            .copied()
            .filter(|&i| raw_sub_labels[i] == sub_id)
            .collect();

        if sub_pixels.is_empty() {
            candidates.push(SubCandidate {
                pixels: Vec::new(),
                centroid: (0.0, 0.0),
                bbox: (0, 0, 0, 0),
                area: 0,
            });
            continue;
        }

        let (centroid, bbox) = compute_pixel_stats(&sub_pixels, width);
        let area = sub_pixels.len();

        candidates.push(SubCandidate {
            pixels: sub_pixels,
            centroid,
            bbox,
            area,
        });
    }

    // ===== Step 6: 吸收极小区域到最近邻 =====
    // 极小区域 → 将其像素重分配给最近的非极小区域
    let min_area = params.min_region_area;
    let mut absorbed: Vec<bool> = vec![false; num_subs];
    let mut final_subs: Vec<usize> = Vec::new(); // surviving sub_ids

    for sub_id in 0..num_subs {
        if candidates[sub_id].area >= min_area {
            final_subs.push(sub_id);
        }
    }

    // 若无幸存者，保持原区域不变
    if final_subs.is_empty() {
        return empty_result;
    }

    // 单幸存者 → 不分裂
    if final_subs.len() <= 1 {
        return empty_result;
    }

    // 对每个极小区域，找最近幸存者
    for sub_id in 0..num_subs {
        if candidates[sub_id].area >= min_area {
            continue;
        }
        if candidates[sub_id].pixels.is_empty() {
            absorbed[sub_id] = true;
            continue;
        }

        let tc = candidates[sub_id].centroid;
        let mut best_target = None;
        let mut best_dist = f64::MAX;
        for &fs in &final_subs {
            let dx = tc.0 - candidates[fs].centroid.0;
            let dy = tc.1 - candidates[fs].centroid.1;
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_target = Some(fs);
            }
        }

        if let Some(target) = best_target {
            // 吸收: 将这些像素归入 target
            let stolen: Vec<usize> = std::mem::take(&mut candidates[sub_id].pixels);
            candidates[target].pixels.extend(stolen);
            candidates[target].area = candidates[target].pixels.len();
            let (new_centroid, new_bbox) = compute_pixel_stats(&candidates[target].pixels, width);
            candidates[target].centroid = new_centroid;
            candidates[target].bbox = new_bbox;
            absorbed[sub_id] = true;
        }
        // 找不到目标 → 像素丢失（极罕见：所有区域都极小）
    }

    // ===== Step 7: 构建最终子区域列表 =====
    let mut sub_regions: Vec<Region> = Vec::new();
    // 旧 sub_id → 新 sub_idx 映射
    let mut old_to_new_idx: Vec<Option<usize>> = vec![None; num_subs];

    for &fs in &final_subs {
        let idx = sub_regions.len();
        old_to_new_idx[fs] = Some(idx);
        let c = &candidates[fs];
        sub_regions.push(Region {
            id: idx,
            cluster_id: region.cluster_id,
            area: c.area,
            pixel_count: c.area,
            centroid: c.centroid,
            bbox: c.bbox,
        });
    }

    // ===== Step 8: 为被吸收的子区域也分配映射 =====
    for sub_id in 0..num_subs {
        if absorbed[sub_id] {
            // 找最近幸存者作为其映射目标
            let tc = candidates[sub_id].centroid;
            if tc == (0.0, 0.0) && candidates[sub_id].pixels.is_empty() {
                continue;
            }
            let mut best = None;
            let mut best_dist = f64::MAX;
            for &fs in &final_subs {
                let dx = tc.0 - candidates[fs].centroid.0;
                let dy = tc.1 - candidates[fs].centroid.1;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best = Some(fs);
                }
            }
            if let Some(target) = best {
                old_to_new_idx[sub_id] = old_to_new_idx[target];
            }
        }
    }

    // ===== Step 9: 构建像素→子区域标签映射 =====
    let mut label_map: Vec<Option<usize>> = vec![None; total];
    for i in 0..total {
        if labels[i] == Some(region_id) {
            let raw = raw_sub_labels[i];
            if raw < num_subs {
                if let Some(new_idx) = old_to_new_idx[raw] {
                    label_map[i] = Some(new_idx);
                }
            }
        }
    }

    (sub_regions, label_map)
}

// =============================================================================
// Split helpers
// =============================================================================

/// 形态学膨胀: 将 mask 中 true 像素向 4-邻域扩展 `radius` 像素。
///
/// 就地修改 `mask`。
fn dilate_boundary(mask: &mut [bool], width: u32, height: u32, radius: u32) {
    if radius == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;

    // 逐轮膨胀
    for _ in 0..radius {
        let mut additions: Vec<usize> = Vec::new();
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                if mask[idx] {
                    continue;
                }
                // 检查 4-邻域是否有 true
                let has_neighbor = (x > 0 && mask[idx - 1])
                    || (x + 1 < w && mask[idx + 1])
                    || (y > 0 && mask[idx - w])
                    || (y + 1 < h && mask[idx + w]);

                if has_neighbor {
                    additions.push(idx);
                }
            }
        }
        for &i in &additions {
            mask[i] = true;
        }
        if additions.is_empty() {
            break;
        }
    }
}

/// 在指定像素集合内运行 4-连通 CCL，跳过屏障像素。
///
/// `region_pixels`: 属于该区域的所有像素索引（已去重）。
/// `barrier`: 屏障掩码（true = 屏障，不可穿越）。
///
/// 返回每像素的子区域标签（0..K-1），非区域或屏障像素保持 0。
fn ccl_within_region(
    region_pixels: &[usize],
    barrier: &[bool],
    width: u32,
    height: u32,
) -> Vec<usize> {
    let w = width as usize;
    let _h = height as usize;
    let total = w * _h;

    // 标记哪些像素参与 CCL
    let mut participant: Vec<bool> = vec![false; total];
    for &i in region_pixels {
        if !barrier[i] {
            participant[i] = true;
        }
    }

    // ===== Pass 1: 临时标签 =====
    let mut uf = Uf::new(region_pixels.len());
    let mut label_of_pixel: Vec<usize> = vec![usize::MAX; total];
    let mut next_label: usize = 0;

    // 为了高效查找已分配标签的像素，按行扫描
    for y in 0.._h {
        let row = y * w;
        for x in 0..w {
            let idx = row + x;
            if !participant[idx] {
                continue;
            }

            // 检查左邻
            let left_label = if x > 0 && participant[idx - 1] {
                Some(label_of_pixel[idx - 1])
            } else {
                None
            };

            // 检查上邻
            let top_label = if y > 0 && participant[idx - w] {
                Some(label_of_pixel[idx - w])
            } else {
                None
            };

            let label = match (left_label, top_label) {
                (Some(l), Some(t)) => {
                    if l != t {
                        uf.union(l, t);
                    }
                    l
                }
                (Some(l), None) => l,
                (None, Some(t)) => t,
                (None, None) => {
                    let lb = next_label;
                    next_label += 1;
                    lb
                }
            };

            label_of_pixel[idx] = label;
        }
    }

    if next_label == 0 {
        // 无参与像素
        return vec![0; total];
    }

    // ===== Pass 2: 路径压缩 + 重编号 =====
    let (label_map, num_final) = uf.resolve(next_label);

    let mut result: Vec<usize> = vec![0; total];
    for idx in 0..total {
        if participant[idx] {
            let raw = label_of_pixel[idx];
            result[idx] = label_map[raw];
        }
    }

    // 将 usize::MAX 替换为 0（统一无标签像素）
    for r in &mut result {
        if *r == usize::MAX {
            *r = 0;
        }
    }

    // 保留 num_final 信息在 result 中？不需要。调用方可以直接用 result。
    // 但我们加一个额外的约定：result 中所有属于区域的像素都有 0..num_final-1 的标签。
    // 调用方通过 result.iter().max() 可获取标签数。

    let _ = num_final;
    result
}

// =============================================================================
// Stats helpers
// =============================================================================

/// 从像素索引列表计算质心和包围盒。
fn compute_pixel_stats(pixels: &[usize], width: u32) -> ((f64, f64), (u32, u32, u32, u32)) {
    let w = width as u32;
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;
    let mut min_x: u32 = width;
    let mut min_y: u32 = u32::MAX;
    let mut max_x: u32 = 0;
    let mut max_y: u32 = 0;

    for &idx in pixels {
        let x = (idx as u32) % w;
        let y = (idx as u32) / w;
        sum_x += x as f64;
        sum_y += y as f64;
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    let n = pixels.len() as f64;
    ((sum_x / n, sum_y / n), (min_x, min_y, max_x, max_y))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SegmentParams;
    use crate::region::Region;

    // ===== Helpers =====

    /// 创建宽松参数（不过滤小区域）
    fn relaxed_params() -> SegmentParams {
        SegmentParams {
            min_region_area: 0,
            edge_merge_strength: 0.05,
            edge_split_strength: 0.4,
            ..SegmentParams::default()
        }
    }

    /// 从 labels 和区域尺寸构建一个简单的 regions 列表（使用近似统计）。
    fn regions_from_labels(
        labels: &[Option<usize>],
        width: u32,
        num_regions: usize,
    ) -> Vec<Region> {
        let w = width as usize;
        let total = labels.len();

        let mut regions = Vec::new();
        for rid in 0..num_regions {
            let mut sum_x: f64 = 0.0;
            let mut sum_y: f64 = 0.0;
            let mut min_x: u32 = width;
            let mut min_y: u32 = total as u32;
            let mut max_x: u32 = 0;
            let mut max_y: u32 = 0;
            let mut count = 0;

            for i in 0..total {
                if labels[i] == Some(rid) {
                    let x = (i % w) as u32;
                    let y = (i / w) as u32;
                    sum_x += x as f64;
                    sum_y += y as f64;
                    if x < min_x {
                        min_x = x;
                    }
                    if y < min_y {
                        min_y = y;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                    count += 1;
                }
            }

            // 取第一个像素的 cluster 作为 cluster_id（任意值，测试中不重要）
            let cid = labels
                .iter()
                .position(|l| l == &Some(rid))
                .map(|_| 0)
                .unwrap_or(0);

            regions.push(Region {
                id: rid,
                cluster_id: cid,
                area: count,
                pixel_count: count,
                centroid: (sum_x / count.max(1) as f64, sum_y / count.max(1) as f64),
                bbox: (min_x, min_y, max_x, max_y),
            });
        }
        regions
    }

    // ===== Test: merge — two regions with weak edge → merged =====

    #[test]
    fn test_merge_weak_edge() {
        // 4×2 图像: 左 2 列 region 0, 右 2 列 region 1
        // 边界处边缘强度很低 → 应合并
        let w = 4u32;
        let h = 2u32;
        let total = (w * h) as usize;

        let labels: Vec<Option<usize>> = vec![
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
        ];

        let regions = regions_from_labels(&labels, w, 2);

        // 边缘强度: 边界处极弱 (0.01), 其余任意
        let mut edge = vec![0.0; total];
        // 边界在列 1-2 之间: 像素 (1,*), (2,*) 邻接
        // 给边界像素低边缘强度
        edge[1] = 0.01; // (1,0)
        edge[3] = 0.01; // (3,0) → wait, (3,0) is column 3, not boundary
        edge[5] = 0.01; // (1,1)
        edge[7] = 0.01; // (3,1) → same

        // Actually: pixels at column 1 and 2 are adjacent.
        // Their edge strengths affect the boundary.
        edge[1] = 0.01; // col 1 row 0
        edge[2] = 0.01; // col 2 row 0
        edge[5] = 0.01; // col 1 row 1
        edge[6] = 0.01; // col 2 row 1

        let params = relaxed_params();

        let (result_regions, result_labels) = refine(&regions, &labels, &edge, w, h, &params);

        assert_eq!(result_regions.len(), 1, "weak edge → single merged region");
        assert_eq!(result_regions[0].area, 8);

        // 所有像素应属于同一区域
        assert!(result_labels.iter().all(|l| l == &Some(0)));
    }

    // ===== Test: no-merge — two regions with strong edge → stay separate =====

    #[test]
    fn test_no_merge_strong_edge() {
        let w = 4u32;
        let h = 2u32;
        let total = (w * h) as usize;

        let labels: Vec<Option<usize>> = vec![
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
        ];

        let regions = regions_from_labels(&labels, w, 2);

        // 边界处边缘强度高 (0.5) → 不应合并
        let mut edge = vec![0.0; total];
        edge[1] = 0.5;
        edge[2] = 0.5;
        edge[5] = 0.5;
        edge[6] = 0.5;

        let params = relaxed_params();

        let (result_regions, _result_labels) = refine(&regions, &labels, &edge, w, h, &params);

        assert_eq!(
            result_regions.len(),
            2,
            "strong edge → regions stay separate"
        );
        assert_eq!(result_regions[0].area, 4);
        assert_eq!(result_regions[1].area, 4);
    }

    // ===== Test: split — a region with strong internal edge → split =====

    #[test]
    fn test_split_internal_edge() {
        // 6×4 单区域，中间一列强边缘把左右分开
        // 6 列宽保证 1px 膨胀后两侧仍有非屏障像素
        let w = 6u32;
        let h = 4u32;
        let total = (w * h) as usize;

        let labels: Vec<Option<usize>> = vec![Some(0); total];

        let regions = regions_from_labels(&labels, w, 1);

        // 在中间列 (col 3) 放强边缘
        let mut edge = vec![0.0; total];
        for y in 0..h as usize {
            edge[y * 6 + 3] = 0.5; // column 3, all rows
        }

        let mut params = relaxed_params();
        params.edge_split_strength = 0.4;

        let (result_regions, _result_labels) = refine(&regions, &labels, &edge, w, h, &params);

        // 强边缘 + 膨胀应把区域切成左 (col 0-1) 和右 (col 4-5)
        assert!(
            result_regions.len() >= 2,
            "internal strong edge should split region (got {} regions)",
            result_regions.len()
        );
    }

    // ===== Test: no-split — uniform region → unchanged =====

    #[test]
    fn test_no_split_uniform_region() {
        let w = 4u32;
        let h = 4u32;
        let total = (w * h) as usize;

        let labels: Vec<Option<usize>> = vec![Some(0); total];
        let regions = regions_from_labels(&labels, w, 1);

        // 全图低边缘强度
        let edge = vec![0.0; total];

        let params = relaxed_params();

        let (result_regions, _result_labels) = refine(&regions, &labels, &edge, w, h, &params);

        assert_eq!(result_regions.len(), 1, "uniform region → no split");
        assert_eq!(result_regions[0].area, 16);
    }

    // ===== Test: combined — weak edge merges + strong edge splits =====

    #[test]
    fn test_combined_merge_and_split() {
        // 6×2: 三列一组
        // col 0-1: region 0, col 2-3: region 1, col 4-5: region 2
        // region 0-1 间弱边缘 → merge
        // region 1-2 间强边缘 → stay
        // region 1 内部强边缘 → split
        let w = 6u32;
        let h = 2u32;
        let total = (w * h) as usize;

        let labels: Vec<Option<usize>> = vec![
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
        ];

        let regions = regions_from_labels(&labels, w, 3);

        let mut edge = vec![0.0; total];
        // region 0-1 边界 (col 1-2): 弱边缘
        edge[1] = 0.01;
        edge[2] = 0.01;
        edge[7] = 0.01;
        edge[8] = 0.01;

        // region 1-2 边界 (col 3-4): 强边缘
        edge[3] = 0.5;
        edge[4] = 0.5;
        edge[9] = 0.5;
        edge[10] = 0.5;

        // region 1 内部也有强边缘 (模拟漏检边界)
        // region 1 是 col 2-3, 内部 col 2 和 col 3 间的像素
        // 但它们已经同一区域，加内部强边缘让它们分裂
        // 实际上 col 2 和 3 是不同列，它们间的边界就是 (2,*) ~ (3,*)
        // That's the same as the region 1's internal split point.
        // Let me think: region 1 occupies col 2-3. An edge between col 2 and 3
        // would split region 1 into two sub-regions: col 2 and col 3.
        // But col 3 also borders region 2. So col 3 would get split off from col 2.
        // Actually the region 1-2 boundary IS at col 3-4. So col 3 is region 1,
        // col 4 is region 2. The strong edge at col 3-4 won't split region 1.

        // Let me put strong edge IN region 0 (the one that will merge with 1):
        // After merge, region 0+1 = col 0,1,2,3. Put strong edge at col 2:
        edge[2] = 0.5;
        edge[3] = 0.5;
        edge[8] = 0.5;
        edge[9] = 0.5;

        let params = relaxed_params();

        let (result_regions, _result_labels) = refine(&regions, &labels, &edge, w, h, &params);

        // 预期: region 0+1 合并 → 但内部强边缘导致分裂 → ≥ 2 个区域
        // region 2 保持独立
        assert!(
            result_regions.len() >= 2,
            "combined: expect at least 2 regions after merge+split, got {}",
            result_regions.len()
        );
    }

    // ===== Test: edge case — empty regions → empty output =====

    #[test]
    fn test_empty_regions() {
        let regions: Vec<Region> = Vec::new();
        let labels: Vec<Option<usize>> = vec![None; 4];
        let edge = vec![0.0; 4];
        let params = relaxed_params();

        let (result_regions, result_labels) = refine(&regions, &labels, &edge, 2, 2, &params);

        assert!(result_regions.is_empty());
        assert_eq!(result_labels.len(), 4);
        assert!(result_labels.iter().all(|l| l.is_none()));
    }

    // ===== Test: dilate_boundary =====

    #[test]
    fn test_dilate_boundary_1px() {
        let w = 5u32;
        let h = 5u32;
        let total = (w * h) as usize;

        // 中心单点为 true
        let mut mask = vec![false; total];
        mask[2 * 5 + 2] = true; // (2,2)

        dilate_boundary(&mut mask, w, h, 1);

        // 中心 + 4-邻域应为 true
        let center = 2 * 5 + 2;
        assert!(mask[center]);
        assert!(mask[center - 1]); // left
        assert!(mask[center + 1]); // right
        assert!(mask[center - 5]); // up
        assert!(mask[center + 5]); // down

        // 角落不应受影响
        assert!(!mask[0]);
        assert!(!mask[4]);
        assert!(!mask[20]);
        assert!(!mask[24]);
    }

    // ===== Test: adjacency computation =====

    #[test]
    fn test_adjacency_two_regions() {
        let w = 4u32;
        let h = 2u32;

        let labels: Vec<Option<usize>> = vec![
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
        ];

        let edge = vec![0.1; 8];

        let triples = compute_adjacency_triples(&labels, &edge, w, h, 2);

        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].0, 0);
        assert_eq!(triples[0].1, 1);
        // mean edge at boundary (col 1-2): pixels 1,2,5,6
        // each pair contributes edge[1]+edge[2] or edge[5]+edge[6]
        // mean of all boundary edge strengths
        assert!(
            (triples[0].2 - 0.1).abs() < 1e-9,
            "mean edge should be 0.1, got {}",
            triples[0].2
        );
    }
}
