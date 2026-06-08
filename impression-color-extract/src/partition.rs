// =============================================================================
// 色域切分 (Median Cut recursion) + 背景分离 (边界采样 + BFS + 形态学)
// =============================================================================

use std::collections::VecDeque;

use crate::params::ColorPartitionParams;

// ── Median Cut 递归二分 ──

#[derive(Debug, Clone)]
pub struct Cluster {
    pub pixels: Vec<(f64, f64, f64)>, // LAB
    pub indices: Vec<usize>,          // 对应原图索引
    pub mean_l: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub var_l: f64,
    pub var_a: f64,
    pub var_b: f64,
    pub bg_score: f64, // [0,1] 该簇属于背景的置信度
}

impl Cluster {
    #[allow(dead_code)]
    fn variance(&self) -> f64 {
        self.var_l + self.var_a + self.var_b
    }
}

/// 递归 Median Cut 切分 LAB 空间
fn median_cut_partition(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    indices: Vec<usize>,
    depth: usize,
    params: &ColorPartitionParams,
    clusters: &mut Vec<Cluster>,
) {
    if indices.len() <= 1 || depth >= params.max_depth {
        // 叶子簇
        clusters.push(build_cluster(lab_l, lab_a, lab_b, indices));
        return;
    }

    // 找到方差最大的通道
    let (mean_l, var_l) = mean_var_sel(&lab_l, &indices);
    let (mean_a, var_a) = mean_var_sel(&lab_a, &indices);
    let (mean_b, var_b) = mean_var_sel(&lab_b, &indices);

    let total_var = var_l + var_a + var_b;
    if total_var < params.variance_threshold {
        clusters.push(Cluster {
            pixels: vec![],
            indices,
            mean_l, mean_a, mean_b,
            var_l, var_a, var_b,
            bg_score: 0.0,
        });
        return;
    }

    let (channel, _) = if var_l >= var_a && var_l >= var_b {
        (0, &lab_l)  // L*
    } else if var_a >= var_b {
        (1, &lab_a)  // a*
    } else {
        (2, &lab_b)  // b*
    };

    let channel_data: &[f64] = if channel == 0 { lab_l } else if channel == 1 { lab_a } else { lab_b };

    // 中位数分割
    let median = median_value(channel_data, &indices);

    let mut left = Vec::with_capacity(indices.len() / 2);
    let mut right = Vec::with_capacity(indices.len() / 2);
    for &idx in &indices {
        if channel_data[idx] <= median {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    if left.is_empty() || right.is_empty() {
        clusters.push(build_cluster(lab_l, lab_a, lab_b, indices));
        return;
    }

    if left.len() as f64 / (left.len() + right.len()) as f64 > 0.95 ||
       right.len() as f64 / (left.len() + right.len()) as f64 > 0.95 {
        // 避免分裂过于不均衡
        clusters.push(build_cluster(lab_l, lab_a, lab_b, indices));
        return;
    }

    median_cut_partition(lab_l, lab_a, lab_b, left, depth + 1, params, clusters);
    median_cut_partition(lab_l, lab_a, lab_b, right, depth + 1, params, clusters);
}

fn build_cluster(lab_l: &[f64], lab_a: &[f64], lab_b: &[f64], indices: Vec<usize>) -> Cluster {
    let (mean_l, var_l) = mean_var_sel(lab_l, &indices);
    let (mean_a, var_a) = mean_var_sel(lab_a, &indices);
    let (mean_b, var_b) = mean_var_sel(lab_b, &indices);
    Cluster { pixels: vec![], indices, mean_l, mean_a, mean_b, var_l, var_a, var_b, bg_score: 0.0 }
}

fn mean_var_sel(data: &[f64], indices: &[usize]) -> (f64, f64) {
    if indices.is_empty() { return (0.0, 0.0); }
    let n = indices.len() as f64;
    let sum: f64 = indices.iter().map(|&i| data[i]).sum();
    let mean = sum / n;
    let var: f64 = indices.iter().map(|&i| { let d = data[i] - mean; d * d }).sum::<f64>() / n;
    (mean, var)
}

fn median_value(data: &[f64], indices: &[usize]) -> f64 {
    let mut vals: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

// ── 背景评分：基于边界采样 + 边界连通 BFS ──

/// 对每个簇计算背景分数: 簇内像素落在边界 band 的比例 + 簇中心到边界样本的距离
fn score_clusters(
    clusters: &mut [Cluster],
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    w: u32, h: u32, params: &ColorPartitionParams,
) {
    let n_pixels = (w * h) as usize;
    if n_pixels == 0 { return; }

    let band = params.border_band.max(1);

    // 标记边界像素
    let mut border_mask = vec![false; n_pixels];
    // Top
    for y in 0..band.min(h) {
        for x in 0..w { border_mask[(y * w + x) as usize] = true; }
    }
    // Bottom
    for y in (h.saturating_sub(band))..h {
        for x in 0..w { border_mask[(y * w + x) as usize] = true; }
    }
    // Left
    for y in band..h.saturating_sub(band) {
        for x in 0..band.min(w) { border_mask[(y * w + x) as usize] = true; }
    }
    // Right
    for y in band..h.saturating_sub(band) {
        for x in (w.saturating_sub(band))..w { border_mask[(y * w + x) as usize] = true; }
    }

    // 采样边界 LAB 值做背景模型
    let border_l: Vec<f64> = (0..n_pixels).filter(|&i| border_mask[i]).map(|i| lab_l[i]).collect();
    let border_a: Vec<f64> = (0..n_pixels).filter(|&i| border_mask[i]).map(|i| lab_a[i]).collect();
    let border_b: Vec<f64> = (0..n_pixels).filter(|&i| border_mask[i]).map(|i| lab_b[i]).collect();

    if border_l.is_empty() { return; }

    // 边界 LAB 稳健中心
    let bg_l = robust_center(&border_l);
    let bg_a = robust_center(&border_a);
    let bg_b = robust_center(&border_b);

    for cluster in clusters.iter_mut() {
        if cluster.indices.is_empty() {
            cluster.bg_score = 1.0;
            continue;
        }

        // 簇内边界像素比例
        let border_count = cluster.indices.iter().filter(|&&i| border_mask[i]).count();
        let border_ratio = border_count as f64 / cluster.indices.len() as f64;

        // 簇中心到背景模型的 LAB 距离
        let d_l = (cluster.mean_l - bg_l) / 100.0;
        let d_a = (cluster.mean_a - bg_a) / 128.0;
        let d_b = (cluster.mean_b - bg_b) / 128.0;
        let center_dist = (d_l * d_l + d_a * d_a + d_b * d_b).sqrt();

        // 综合评分: 边界比例高 + 距背景模型近 => bg_score 高
        let dist_factor = (1.0 - center_dist).clamp(0.0, 1.0);
        cluster.bg_score = (border_ratio * 0.6 + dist_factor * 0.4).clamp(0.0, 1.0);
    }
}

fn robust_center(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    // 简单 median
    sorted[n / 2]
}

// ── 生成背景 mask ──

/// 从簇的 bg_score 生成初始背景 mask
fn clusters_to_bg_mask(clusters: &[Cluster], n_pixels: usize, threshold: f64) -> Vec<f64> {
    let mut mask = vec![0.0; n_pixels];
    for cluster in clusters {
        let v = if cluster.bg_score >= threshold { 1.0 } else { 0.0 };
        for &idx in &cluster.indices {
            mask[idx] = v;
        }
    }
    mask
}

// ── 边界连通 BFS 精化 ──

/// BFS flood-fill 从边界扩散，连通区域标记为背景
fn bfs_connected_bg(
    raw_mask: &[f64],
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    w: u32, h: u32, params: &ColorPartitionParams,
) -> Vec<f64> {
    let n = (w * h) as usize;
    if n == 0 { return vec![]; }

    let band = params.border_band.max(1);
    let score_threshold = params.bg_score_threshold;
    let connect_threshold = params.bg_connect_threshold;
    let mut visited = vec![false; n];
    let mut mask = vec![0.0; n];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();

    // 将边界 band 中 raw_mask >= threshold 的像素入队
    // Top
    for y in 0..band.min(h) {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if raw_mask[i] >= score_threshold {
                visited[i] = true;
                mask[i] = 1.0;
                queue.push_back((x, y));
            }
        }
    }
    // Bottom
    for y in (h.saturating_sub(band))..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if !visited[i] && raw_mask[i] >= score_threshold {
                visited[i] = true; mask[i] = 1.0; queue.push_back((x, y));
            }
        }
    }
    // Left / Right (non-corner)
    for y in band..h.saturating_sub(band) {
        for x in 0..band.min(w) {
            let i = (y * w + x) as usize;
            if !visited[i] && raw_mask[i] >= score_threshold {
                visited[i] = true; mask[i] = 1.0; queue.push_back((x, y));
            }
        }
        for x in (w.saturating_sub(band))..w {
            let i = (y * w + x) as usize;
            if !visited[i] && raw_mask[i] >= score_threshold {
                visited[i] = true; mask[i] = 1.0; queue.push_back((x, y));
            }
        }
    }

    // 4-neighbor BFS
    while let Some((cx, cy)) = queue.pop_front() {
        let ci = (cy * w + cx) as usize;
        let cl = lab_l[ci];
        let ca = lab_a[ci];
        let cb = lab_b[ci];
        for (dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
            let ni = (ny as u32 * w + nx as u32) as usize;
            if visited[ni] { continue; }
            // 判断 LAB 距离是否在阈值内
            let d_l = (lab_l[ni] - cl) / 100.0;
            let d_a = (lab_a[ni] - ca) / 128.0;
            let d_b = (lab_b[ni] - cb) / 128.0;
            let dist = (d_l * d_l + d_a * d_a + d_b * d_b).sqrt();
            if dist <= connect_threshold {
                visited[ni] = true;
                mask[ni] = 1.0;
                queue.push_back((nx as u32, ny as u32));
            }
        }
    }

    mask
}

// ── 形态学操作 ──

fn erode(mask: &[f64], w: u32, h: u32, radius: u32) -> Vec<f64> {
    let r = radius as i32;
    let mut out = vec![0.0; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let mut all_bg = true;
            'outer: for dy in -r..=r {
                for dx in -r..=r {
                    let px = x as i32 + dx;
                    let py = y as i32 + dy;
                    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 { continue; }
                    if mask[(py as u32 * w + px as u32) as usize] < 0.5 {
                        all_bg = false;
                        break 'outer;
                    }
                }
            }
            out[i] = if all_bg { 1.0 } else { 0.0 };
        }
    }
    out
}

fn dilate(mask: &[f64], w: u32, h: u32, radius: u32) -> Vec<f64> {
    let r = radius as i32;
    let mut out = vec![0.0; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let mut any_bg = false;
            'outer: for dy in -r..=r {
                for dx in -r..=r {
                    let px = x as i32 + dx;
                    let py = y as i32 + dy;
                    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 { continue; }
                    if mask[(py as u32 * w + px as u32) as usize] >= 0.5 {
                        any_bg = true;
                        break 'outer;
                    }
                }
            }
            out[i] = if any_bg { 1.0 } else { 0.0 };
        }
    }
    out
}

fn opening(mask: &[f64], w: u32, h: u32, radius: u32) -> Vec<f64> {
    let eroded = erode(mask, w, h, radius);
    dilate(&eroded, w, h, radius)
}

fn closing(mask: &[f64], w: u32, h: u32, radius: u32) -> Vec<f64> {
    let dilated = dilate(mask, w, h, radius);
    erode(&dilated, w, h, radius)
}

fn mask_mean(mask: &[f64]) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }
    mask.iter().sum::<f64>() / mask.len() as f64
}

// ── 公共 API ──

/// 主入口：色域切分 → 背景评分 → BFS → 形态学 → 前景置信度
pub struct PartitionResult {
    pub clusters: Vec<Cluster>,
    pub bg_mask_raw: Vec<f64>,
    pub bg_mask_morph: Vec<f64>,
    pub fg_confidence: Vec<f64>,
    pub color_clusters_rgb: Vec<[f64; 3]>, // 每个像素用簇平均色填充
}

pub fn partition_and_separate(
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    rgb: &[[f64; 3]],
    w: u32, h: u32, params: &ColorPartitionParams,
) -> PartitionResult {
    let n = (w * h) as usize;
    if n == 0 {
        return PartitionResult {
            clusters: vec![],
            bg_mask_raw: vec![],
            bg_mask_morph: vec![],
            fg_confidence: vec![],
            color_clusters_rgb: vec![],
        };
    }

    if !params.enabled {
        // 禁用色域切分：全部视为前景
        return PartitionResult {
            clusters: vec![],
            bg_mask_raw: vec![0.0; n],
            bg_mask_morph: vec![0.0; n],
            fg_confidence: vec![1.0; n],
            color_clusters_rgb: rgb.to_vec(),
        };
    }

    // 1. 色域切分
    let all_indices: Vec<usize> = (0..n).collect();
    let mut clusters: Vec<Cluster> = Vec::new();
    median_cut_partition(lab_l, lab_a, lab_b, all_indices, 0, params, &mut clusters);

    // 合并小簇到最近的邻居
    merge_small_clusters(&mut clusters, lab_l, lab_a, lab_b, params);
    // 限制最大簇数：合并最小簇，而不是按递归顺序截断丢弃颜色。
    reduce_cluster_count(&mut clusters, lab_l, lab_a, lab_b, params.max_clusters);

    // 2. 背景评分
    score_clusters(&mut clusters, lab_l, lab_a, lab_b, w, h, params);

    // 3. 生成初始 mask
    let raw_mask = clusters_to_bg_mask(&clusters, n, params.bg_score_threshold);

    // 4. BFS 连通
    let bfs_mask = bfs_connected_bg(&raw_mask, lab_l, lab_a, lab_b, w, h, params);
    let connected_mask = if mask_mean(&bfs_mask) > params.max_bg_ratio {
        raw_mask.clone()
    } else {
        bfs_mask
    };

    // 5. 形态学净化
    let morph_mask = if params.close_radius > 0 {
        let closed = closing(&connected_mask, w, h, params.close_radius);
        if params.open_radius > 0 {
            opening(&closed, w, h, params.open_radius)
        } else { closed }
    } else if params.open_radius > 0 {
        opening(&connected_mask, w, h, params.open_radius)
    } else { connected_mask.clone() };

    let erode_mask = if params.erode_radius > 0 {
        erode(&morph_mask, w, h, params.erode_radius)
    } else { morph_mask.clone() };
    let erode_mask = if mask_mean(&erode_mask) > params.max_bg_ratio {
        raw_mask.clone()
    } else {
        erode_mask
    };

    // 6. 前景置信度
    let fg_confidence: Vec<f64> = erode_mask.iter().map(|&m| (1.0 - m).clamp(0.0, 1.0)).collect();

    // 7. 色块渲染图
    let color_clusters_rgb = render_clusters(&clusters, n, rgb);

    PartitionResult {
        clusters,
        bg_mask_raw: connected_mask,
        bg_mask_morph: erode_mask,
        fg_confidence,
        color_clusters_rgb,
    }
}

fn render_clusters(clusters: &[Cluster], n: usize, rgb: &[[f64; 3]]) -> Vec<[f64; 3]> {
    if clusters.is_empty() {
        return rgb.to_vec();
    }
    let mut out = vec![[0.0; 3]; n];
    for cluster in clusters {
        let color = mean_rgb_for_cluster(cluster, rgb);
        for &idx in &cluster.indices {
            out[idx] = color;
        }
    }
    out
}

fn mean_rgb_for_cluster(cluster: &Cluster, rgb: &[[f64; 3]]) -> [f64; 3] {
    if cluster.indices.is_empty() {
        return [0.0; 3];
    }
    let mut sum = [0.0; 3];
    for &idx in &cluster.indices {
        sum[0] += rgb[idx][0];
        sum[1] += rgb[idx][1];
        sum[2] += rgb[idx][2];
    }
    let n = cluster.indices.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ── 辅助: 合并小簇 ──

fn merge_small_clusters(
    clusters: &mut Vec<Cluster>,
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    params: &ColorPartitionParams,
) {
    let total = lab_l.len();
    let min_area = (total as f64 * params.min_cluster_area_ratio) as usize;
    if min_area < 2 { return; }

    let mut i = 0;
    while i < clusters.len() {
        if clusters[i].indices.len() < min_area {
            // 找到最近的簇合并
            let mut best_j = None;
            let mut best_dist = f64::MAX;
            for j in 0..clusters.len() {
                if j == i || clusters[j].indices.len() < min_area { continue; }
                let d_l = (clusters[i].mean_l - clusters[j].mean_l) / 100.0;
                let d_a = (clusters[i].mean_a - clusters[j].mean_a) / 128.0;
                let d_b = (clusters[i].mean_b - clusters[j].mean_b) / 128.0;
                let dist = d_l * d_l + d_a * d_a + d_b * d_b;
                if dist < best_dist { best_dist = dist; best_j = Some(j); }
            }
            if let Some(j) = best_j {
                // 合并 i 到 j
                let mut cluster_i = clusters.remove(i);
                let target = &mut clusters[if j < i { j } else { j.saturating_sub(1) }];
                target.indices.append(&mut cluster_i.indices);
                recompute_cluster_stats(target, lab_l, lab_a, lab_b);
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

fn reduce_cluster_count(
    clusters: &mut Vec<Cluster>,
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    max_clusters: usize,
) {
    let max_clusters = max_clusters.max(1);
    while clusters.len() > max_clusters {
        let Some(small_i) = clusters
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.indices.len())
            .map(|(i, _)| i)
        else {
            return;
        };

        let mut best_j = None;
        let mut best_dist = f64::MAX;
        for (j, c) in clusters.iter().enumerate() {
            if j == small_i {
                continue;
            }
            let d_l = (clusters[small_i].mean_l - c.mean_l) / 100.0;
            let d_a = (clusters[small_i].mean_a - c.mean_a) / 128.0;
            let d_b = (clusters[small_i].mean_b - c.mean_b) / 128.0;
            let dist = d_l * d_l + d_a * d_a + d_b * d_b;
            if dist < best_dist {
                best_dist = dist;
                best_j = Some(j);
            }
        }

        let Some(best_j) = best_j else {
            return;
        };
        let mut small = clusters.remove(small_i);
        let target_i = if best_j > small_i { best_j - 1 } else { best_j };
        clusters[target_i].indices.append(&mut small.indices);
        recompute_cluster_stats(&mut clusters[target_i], lab_l, lab_a, lab_b);
    }
}

fn recompute_cluster_stats(cluster: &mut Cluster, lab_l: &[f64], lab_a: &[f64], lab_b: &[f64]) {
    let (mean_l, var_l) = mean_var_sel(lab_l, &cluster.indices);
    let (mean_a, var_a) = mean_var_sel(lab_a, &cluster.indices);
    let (mean_b, var_b) = mean_var_sel(lab_b, &cluster.indices);
    cluster.mean_l = mean_l;
    cluster.mean_a = mean_a;
    cluster.mean_b = mean_b;
    cluster.var_l = var_l;
    cluster.var_a = var_a;
    cluster.var_b = var_b;
}
