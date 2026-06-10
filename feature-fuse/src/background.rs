// =============================================================================
// Background Estimation: 三阶段管线（色域切分 + BFS 连通 + 软 mask）
// =============================================================================
// Phase 1 — Color partition via Median Cut → cluster scoring → bg mask
// Phase 2 — BFS boundary connectivity → morphological cleanup
// Phase 3 — Soft mask: border prototype likelihood → blend → fg confidence
// =============================================================================

use std::collections::VecDeque;

use crate::params::{
    BackgroundFloodBarrierParams, BackgroundParams, SoftMaskParams, SubjectPriorParams,
};

pub struct BackgroundFeatureInputs<'a> {
    pub dct: &'a [f64],
    pub lab_grad: &'a [f64],
    pub spectral: &'a [f64],
    pub local_light: &'a [f64],
    /// 使用 local LAB a/b residual 的合成图作为局部色度/饱和度保护代理。
    pub local_sat: &'a [f64],
}

pub struct BackgroundDiagnostics {
    pub bg_candidate: Vec<f64>,
    pub bg_barrier: Vec<f64>,
    pub bg_mask_before_protect: Vec<f64>,
    pub foreground_protect: Vec<f64>,
    pub bg_mask_after_protect: Vec<f64>,
    pub fg_confidence: Vec<f64>,
}

pub struct BackgroundFeatureResult {
    pub bg_mask_morph: Vec<f64>,
    pub fg_confidence: Vec<f64>,
    pub diagnostics: BackgroundDiagnostics,
}

// =============================================================================
// Phase 1: Color Partition via Median Cut
// =============================================================================

#[derive(Debug, Clone)]
struct Cluster {
    #[allow(dead_code)]
    pixels: Vec<(f64, f64, f64)>,
    indices: Vec<usize>,
    mean_l: f64,
    mean_a: f64,
    mean_b: f64,
    var_l: f64,
    var_a: f64,
    var_b: f64,
    bg_score: f64,
}

/// 递归 Median Cut 切分 LAB 空间
fn median_cut_partition(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    indices: Vec<usize>,
    depth: usize,
    max_depth: usize,
    variance_threshold: f64,
    clusters: &mut Vec<Cluster>,
) {
    if indices.len() <= 1 || depth >= max_depth {
        clusters.push(build_cluster(lab_l, lab_a, lab_b, indices));
        return;
    }

    let (mean_l, var_l) = mean_var_sel(lab_l, &indices);
    let (mean_a, var_a) = mean_var_sel(lab_a, &indices);
    let (mean_b, var_b) = mean_var_sel(lab_b, &indices);

    let total_var = var_l + var_a + var_b;
    if total_var < variance_threshold {
        clusters.push(Cluster {
            pixels: vec![],
            indices,
            mean_l,
            mean_a,
            mean_b,
            var_l,
            var_a,
            var_b,
            bg_score: 0.0,
        });
        return;
    }

    let channel = if var_l >= var_a && var_l >= var_b {
        0 // L*
    } else if var_a >= var_b {
        1 // a*
    } else {
        2 // b*
    };

    let channel_data: &[f64] = match channel {
        0 => lab_l,
        1 => lab_a,
        _ => lab_b,
    };

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

    if left.is_empty()
        || right.is_empty()
        || left.len() as f64 / (left.len() + right.len()) as f64 > 0.95
        || right.len() as f64 / (left.len() + right.len()) as f64 > 0.95
    {
        clusters.push(build_cluster(lab_l, lab_a, lab_b, indices));
        return;
    }

    median_cut_partition(
        lab_l,
        lab_a,
        lab_b,
        left,
        depth + 1,
        max_depth,
        variance_threshold,
        clusters,
    );
    median_cut_partition(
        lab_l,
        lab_a,
        lab_b,
        right,
        depth + 1,
        max_depth,
        variance_threshold,
        clusters,
    );
}

fn build_cluster(lab_l: &[f64], lab_a: &[f64], lab_b: &[f64], indices: Vec<usize>) -> Cluster {
    let (mean_l, var_l) = mean_var_sel(lab_l, &indices);
    let (mean_a, var_a) = mean_var_sel(lab_a, &indices);
    let (mean_b, var_b) = mean_var_sel(lab_b, &indices);
    Cluster {
        pixels: vec![],
        indices,
        mean_l,
        mean_a,
        mean_b,
        var_l,
        var_a,
        var_b,
        bg_score: 0.0,
    }
}

fn mean_var_sel(data: &[f64], indices: &[usize]) -> (f64, f64) {
    if indices.is_empty() {
        return (0.0, 0.0);
    }
    let n = indices.len() as f64;
    let sum: f64 = indices.iter().map(|&i| data[i]).sum();
    let mean = sum / n;
    let var: f64 = indices
        .iter()
        .map(|&i| {
            let d = data[i] - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var)
}

fn median_value(data: &[f64], indices: &[usize]) -> f64 {
    let mut vals: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

/// Simple median for robust center estimation
fn robust_center(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

/// 对每个簇计算背景分数: 边界比例 + 距边界 LAB 中心距离
fn score_clusters(
    clusters: &mut [Cluster],
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: u32,
    h: u32,
    border_band: u32,
) {
    let n_pixels = (w * h) as usize;
    if n_pixels == 0 {
        return;
    }

    let band = border_band.max(1);

    // 标记边界像素
    let mut border_mask = vec![false; n_pixels];
    for y in 0..band.min(h) {
        for x in 0..w {
            border_mask[(y * w + x) as usize] = true;
        }
    }
    for y in (h.saturating_sub(band))..h {
        for x in 0..w {
            border_mask[(y * w + x) as usize] = true;
        }
    }
    for y in band..h.saturating_sub(band) {
        for x in 0..band.min(w) {
            border_mask[(y * w + x) as usize] = true;
        }
    }
    for y in band..h.saturating_sub(band) {
        for x in (w.saturating_sub(band))..w {
            border_mask[(y * w + x) as usize] = true;
        }
    }

    // 边界 LAB 值列表
    let border_l: Vec<f64> = (0..n_pixels)
        .filter(|&i| border_mask[i])
        .map(|i| lab_l[i])
        .collect();
    let border_a: Vec<f64> = (0..n_pixels)
        .filter(|&i| border_mask[i])
        .map(|i| lab_a[i])
        .collect();
    let border_b: Vec<f64> = (0..n_pixels)
        .filter(|&i| border_mask[i])
        .map(|i| lab_b[i])
        .collect();

    if border_l.is_empty() {
        return;
    }

    let bg_l = robust_center(&border_l);
    let bg_a = robust_center(&border_a);
    let bg_b = robust_center(&border_b);

    for cluster in clusters.iter_mut() {
        if cluster.indices.is_empty() {
            cluster.bg_score = 1.0;
            continue;
        }

        let border_count = cluster.indices.iter().filter(|&&i| border_mask[i]).count();
        let border_ratio = border_count as f64 / cluster.indices.len() as f64;

        let d_l = (cluster.mean_l - bg_l) / 100.0;
        let d_a = (cluster.mean_a - bg_a) / 128.0;
        let d_b = (cluster.mean_b - bg_b) / 128.0;
        let center_dist = (d_l * d_l + d_a * d_a + d_b * d_b).sqrt();

        let dist_factor = (1.0 - center_dist).clamp(0.0, 1.0);
        cluster.bg_score = (border_ratio * 0.6 + dist_factor * 0.4).clamp(0.0, 1.0);
    }
}

fn clusters_to_bg_mask(clusters: &[Cluster], n_pixels: usize, threshold: f64) -> Vec<f64> {
    let mut mask = vec![0.0; n_pixels];
    for cluster in clusters {
        let v = if cluster.bg_score >= threshold {
            1.0
        } else {
            0.0
        };
        for &idx in &cluster.indices {
            mask[idx] = v;
        }
    }
    mask
}

// ── Cluster merge / reduce helpers ──

fn merge_small_clusters(
    clusters: &mut Vec<Cluster>,
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    min_cluster_area_ratio: f64,
) {
    let total = lab_l.len();
    let min_area = (total as f64 * min_cluster_area_ratio) as usize;
    if min_area < 2 {
        return;
    }

    let mut i = 0;
    while i < clusters.len() {
        if clusters[i].indices.len() < min_area {
            let mut best_j = None;
            let mut best_dist = f64::MAX;
            for j in 0..clusters.len() {
                if j == i || clusters[j].indices.len() < min_area {
                    continue;
                }
                let d_l = (clusters[i].mean_l - clusters[j].mean_l) / 100.0;
                let d_a = (clusters[i].mean_a - clusters[j].mean_a) / 128.0;
                let d_b = (clusters[i].mean_b - clusters[j].mean_b) / 128.0;
                let dist = d_l * d_l + d_a * d_a + d_b * d_b;
                if dist < best_dist {
                    best_dist = dist;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                let mut cluster_i = clusters.remove(i);
                let target = &mut clusters[if j <= i { j } else { j.saturating_sub(1) }];
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
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
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

// =============================================================================
// Phase 2: BFS Connectivity + Morphology
// =============================================================================

fn build_flood_barrier(
    features: &BackgroundFeatureInputs,
    border_bg: &[f64],
    params: &BackgroundFloodBarrierParams,
) -> Vec<f64> {
    let n = features.lab_grad.len();
    if !params.enabled {
        return vec![0.0; n];
    }

    (0..n)
        .map(|i| {
            if border_bg.get(i).copied().unwrap_or(0.0) >= params.barrier_color_relax_threshold {
                return 0.0;
            }
            let stop = features.lab_grad.get(i).copied().unwrap_or(0.0) > params.grad_stop
                || features.dct.get(i).copied().unwrap_or(0.0) > params.dct_stop
                || features.local_light.get(i).copied().unwrap_or(0.0) > params.local_light_stop
                || features.spectral.get(i).copied().unwrap_or(0.0) > params.spectral_stop;
            if stop {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn build_bg_candidate(border_bg: &[f64], color_threshold: f64) -> Vec<f64> {
    border_bg
        .iter()
        .map(|&v| if v > color_threshold { 1.0 } else { 0.0 })
        .collect()
}

fn build_foreground_protect(
    features: &BackgroundFeatureInputs,
    params: &BackgroundFloodBarrierParams,
) -> Vec<f64> {
    let n = features.lab_grad.len();
    if n == 0 {
        return Vec::new();
    }

    let raw: Vec<f64> = (0..n)
        .map(|i| {
            features.lab_grad.get(i).copied().unwrap_or(0.0) * params.protect_grad_weight
                + features.dct.get(i).copied().unwrap_or(0.0) * params.protect_dct_weight
                + features.spectral.get(i).copied().unwrap_or(0.0) * params.protect_spectral_weight
                + features.local_light.get(i).copied().unwrap_or(0.0)
                    * params.protect_local_light_weight
                + features.local_sat.get(i).copied().unwrap_or(0.0)
                    * params.protect_local_sat_weight
        })
        .collect();

    percentile_normalize_local(&raw, params.protect_p_low, params.protect_p_high)
}

fn apply_foreground_protect(bg_mask: &[f64], protect: &[f64], protect_strength: f64) -> Vec<f64> {
    bg_mask
        .iter()
        .enumerate()
        .map(|(i, &bg)| {
            let p = protect.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            (bg * (1.0 - protect_strength.clamp(0.0, 1.0) * p)).clamp(0.0, 1.0)
        })
        .collect()
}

/// BFS flood-fill 从边界扩散，连通区域标记为背景
fn bfs_connected_bg(
    _raw_mask: &[f64],
    bg_candidate: &[f64],
    bg_barrier: &[f64],
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: u32,
    h: u32,
    border_band: u32,
    _bg_score_threshold: f64,
    bg_connect_threshold: f64,
) -> Vec<f64> {
    let n = (w * h) as usize;
    if n == 0 {
        return vec![];
    }

    let band = border_band.max(1);
    let connect_threshold = bg_connect_threshold;
    let mut visited = vec![false; n];
    let mut mask = vec![0.0; n];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();

    // 将边界 band 中 raw_mask >= threshold 的像素入队
    // Top
    for y in 0..band.min(h) {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if bg_candidate[i] >= 0.5 {
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
            if !visited[i] && bg_candidate[i] >= 0.5 {
                visited[i] = true;
                mask[i] = 1.0;
                queue.push_back((x, y));
            }
        }
    }
    // Left / Right (non-corner)
    for y in band..h.saturating_sub(band) {
        for x in 0..band.min(w) {
            let i = (y * w + x) as usize;
            if !visited[i] && bg_candidate[i] >= 0.5 {
                visited[i] = true;
                mask[i] = 1.0;
                queue.push_back((x, y));
            }
        }
        for x in (w.saturating_sub(band))..w {
            let i = (y * w + x) as usize;
            if !visited[i] && bg_candidate[i] >= 0.5 {
                visited[i] = true;
                mask[i] = 1.0;
                queue.push_back((x, y));
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
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let ni = (ny as u32 * w + nx as u32) as usize;
            if visited[ni] {
                continue;
            }
            if bg_candidate[ni] < 0.5 || bg_barrier[ni] >= 0.5 {
                continue;
            }
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
                    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                        continue;
                    }
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
                    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                        continue;
                    }
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

// =============================================================================
// Phase 3: Soft Mask Refinement
// =============================================================================

/// 基于边界采样原型的背景相似度（颜色距离 → Gaussian 衰减）
fn border_background_likelihood(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: u32,
    h: u32,
    border_band: u32,
    soft_mask: &SoftMaskParams,
) -> Vec<f64> {
    let wu = w as usize;
    let hu = h as usize;
    let n = wu.saturating_mul(hu);
    if n == 0 {
        return Vec::new();
    }

    let band = (border_band as usize).max(1).min(wu.max(1)).min(hu.max(1));
    let mut border_indices = Vec::new();
    for y in 0..hu {
        for x in 0..wu {
            if x < band || y < band || x + band >= wu || y + band >= hu {
                border_indices.push(y * wu + x);
            }
        }
    }
    if border_indices.is_empty() {
        return vec![0.0; n];
    }

    let max_prototypes = 96usize;
    let step = (border_indices.len() / max_prototypes).max(1);
    let prototypes: Vec<usize> = border_indices
        .iter()
        .step_by(step)
        .take(max_prototypes)
        .copied()
        .collect();

    let mut bg = vec![0.0; n];
    for y in 0..hu {
        for x in 0..wu {
            let i = y * wu + x;
            let mut best = f64::MAX;
            for &p in &prototypes {
                let d_l = (lab_l[i] - lab_l[p]) / 100.0;
                let d_a = (lab_a[i] - lab_a[p]) / 128.0;
                let d_b = (lab_b[i] - lab_b[p]) / 128.0;
                let d = (d_l * d_l + d_a * d_a + d_b * d_b).sqrt();
                if d < best {
                    best = d;
                }
            }
            let color_bg = (-(best / 0.16).powi(2)).exp();
            // NOTE: use 1.0 as edge_bg placeholder (no center_prior call here)
            let edge_bg = 1.0;
            bg[i] = (color_bg * (0.58 + 0.42 * edge_bg)).clamp(0.0, 1.0);
        }
    }

    let radius = blur_radius(wu.min(hu), soft_mask.border_bg_blur_radius, 48, 4, 16);
    box_blur_mask(&bg, wu, hu, radius)
}

/// Gaussian center bias (subject prior)
fn subject_prior(x: usize, y: usize, w: usize, h: usize, params: &SubjectPriorParams) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let nx = (x as f64 + 0.5) / w as f64;
    let ny = (y as f64 + 0.5) / h as f64;
    let dx = (nx - params.center_x) / params.radius_x.max(1e-6);
    let dy = (ny - params.center_y) / params.radius_y.max(1e-6);
    (-(dx * dx + dy * dy)).exp().clamp(0.0, 1.0)
}

/// Compute box blur radius: if `fixed > 0`, use it directly.
/// Otherwise use adaptive formula: clamp(min(w,h)/divisor, min_clamp, max_clamp)
fn blur_radius(
    image_short_side: usize,
    fixed: u32,
    divisor: usize,
    min_clamp: usize,
    max_clamp: usize,
) -> usize {
    if fixed > 0 {
        fixed as usize
    } else {
        (image_short_side / divisor).clamp(min_clamp, max_clamp)
    }
}

/// Separable box blur via prefix sums, O(1) per pixel
fn box_blur_mask(mask: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    if mask.is_empty() || w == 0 || h == 0 || radius == 0 {
        return mask.to_vec();
    }

    // Horizontal pass
    let mut horizontal = vec![0.0; mask.len()];
    for y in 0..h {
        let mut prefix = vec![0.0; w + 1];
        for x in 0..w {
            prefix[x + 1] = prefix[x] + mask[y * w + x];
        }
        for x in 0..w {
            let left = x.saturating_sub(radius);
            let right = (x + radius).min(w - 1);
            let sum = prefix[right + 1] - prefix[left];
            horizontal[y * w + x] = sum / (right - left + 1) as f64;
        }
    }

    // Vertical pass
    let mut out = vec![0.0; mask.len()];
    for x in 0..w {
        let mut prefix = vec![0.0; h + 1];
        for y in 0..h {
            prefix[y + 1] = prefix[y] + horizontal[y * w + x];
        }
        for y in 0..h {
            let top = y.saturating_sub(radius);
            let bottom = (y + radius).min(h - 1);
            let sum = prefix[bottom + 1] - prefix[top];
            out[y * w + x] = sum / (bottom - top + 1) as f64;
        }
    }
    out
}

/// Weighted average of 3 components
fn weighted3(a: f64, aw: f64, b: f64, bw: f64, c: f64, cw: f64) -> f64 {
    let aw = aw.max(0.0);
    let bw = bw.max(0.0);
    let cw = cw.max(0.0);
    let sum = aw + bw + cw;
    if sum < 1e-12 {
        return 0.0;
    }
    ((a * aw + b * bw + c * cw) / sum).clamp(0.0, 1.0)
}

struct MaskStats {
    mean: f64,
    unique_values: usize,
}

fn mask_stats(mask: &[f64]) -> MaskStats {
    if mask.is_empty() {
        return MaskStats {
            mean: 0.0,
            unique_values: 0,
        };
    }
    let mean = mask.iter().sum::<f64>() / mask.len() as f64;
    let mut seen = Vec::new();
    for &v in mask {
        let q = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        if !seen.contains(&q) {
            seen.push(q);
            if seen.len() > 2 {
                break;
            }
        }
    }
    MaskStats {
        mean,
        unique_values: seen.len(),
    }
}

/// Blend hard partition mask with soft border-background likelihood
fn soft_background_from_partition(mask: &[f64], border_bg: &[f64]) -> Vec<f64> {
    let stats = mask_stats(mask);
    let partition_weight = if stats.unique_values <= 2 && (stats.mean <= 0.02 || stats.mean >= 0.98)
    {
        0.0
    } else if stats.unique_values <= 2 {
        0.08
    } else {
        0.15
    };
    mask.iter()
        .zip(border_bg.iter())
        .map(|(&m, &b)| (m * partition_weight + b * (1.0 - partition_weight)).clamp(0.0, 1.0))
        .collect()
}

fn unsharp_mask(mask: &[f64], w: usize, h: usize, radius: u32, amount: f64) -> Vec<f64> {
    if amount <= 0.0 || radius == 0 {
        return mask.to_vec();
    }

    let blurred = box_blur_mask(mask, w, h, radius as usize);
    mask.iter()
        .zip(blurred.iter())
        .map(|(&v, &b)| (v + (v - b) * amount).clamp(0.0, 1.0))
        .collect()
}

/// Compute foreground confidence from background: color_fg blended with saliency + subject_prior
fn soft_foreground_from_background(
    bg: &[f64],
    w: u32,
    h: u32,
    subject_prior_map: &[f64],
    soft_mask: &SoftMaskParams,
) -> Vec<f64> {
    const FG_COLOR_W: f64 = 0.45;
    const FG_SALIENCY_W: f64 = 0.15;
    const FG_SUBJECT_W: f64 = 0.40;

    let n = bg.len();
    let saliency_fg = 0.5; // placeholder constant

    let mut fg = vec![0.0; n];
    let wu = w as usize;
    let hu = h as usize;
    for y in 0..hu {
        for x in 0..wu {
            let i = y * wu + x;
            let color_fg = 1.0 - bg[i];
            fg[i] = weighted3(
                color_fg,
                FG_COLOR_W,
                saliency_fg,
                FG_SALIENCY_W,
                subject_prior_map[i],
                FG_SUBJECT_W,
            );
        }
    }

    let radius = blur_radius(wu.min(hu), soft_mask.fg_confidence_blur_radius, 40, 6, 24);
    let blurred = box_blur_mask(&fg, wu, hu, radius);
    let sharpen_radius = soft_mask.fg_confidence_sharpen_radius.max(1);
    let sharpened = unsharp_mask(
        &blurred,
        wu,
        hu,
        sharpen_radius,
        soft_mask.fg_confidence_sharpen_amount,
    );

    // Percentile normalize to [0, 1]
    percentile_normalize_local(&sharpened, 2.0, 98.0)
}

/// Local percentile normalization (same logic as fusion::percentile_normalize)
fn percentile_normalize_local(data: &[f64], p_low: f64, p_high: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let lo_idx = ((n as f64) * p_low / 100.0).floor() as usize;
    let hi_idx = ((n as f64) * p_high / 100.0).ceil() as usize;
    let lo_val = sorted[lo_idx.min(n - 1)];
    let hi_val = sorted[hi_idx.min(n - 1)];
    let range = (hi_val - lo_val).max(1e-12);
    data.iter()
        .map(|&v| ((v - lo_val) / range).clamp(0.0, 1.0))
        .collect()
}

// =============================================================================
// Public API
// =============================================================================

/// Compute background mask (morph) and foreground confidence.
/// Returns (bg_mask_morph, fg_confidence) — both express "foregroundness" (high = foreground).
pub fn compute_background_features(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: u32,
    h: u32,
    feature_inputs: &BackgroundFeatureInputs,
    params: &BackgroundParams,
) -> BackgroundFeatureResult {
    let n = (w * h) as usize;
    if n == 0 {
        return BackgroundFeatureResult {
            bg_mask_morph: vec![],
            fg_confidence: vec![],
            diagnostics: BackgroundDiagnostics {
                bg_candidate: vec![],
                bg_barrier: vec![],
                bg_mask_before_protect: vec![],
                foreground_protect: vec![],
                bg_mask_after_protect: vec![],
                fg_confidence: vec![],
            },
        };
    }

    let partition = &params.partition;
    let morph = &params.morphology;

    // 禁用 → 全部视为前景
    if !partition.enabled {
        return BackgroundFeatureResult {
            bg_mask_morph: vec![1.0; n],
            fg_confidence: vec![1.0; n],
            diagnostics: BackgroundDiagnostics {
                bg_candidate: vec![0.0; n],
                bg_barrier: vec![0.0; n],
                bg_mask_before_protect: vec![0.0; n],
                foreground_protect: vec![0.0; n],
                bg_mask_after_protect: vec![0.0; n],
                fg_confidence: vec![1.0; n],
            },
        };
    }

    // Phase 1: Median Cut partition
    let all_indices: Vec<usize> = (0..n).collect();
    let mut clusters: Vec<Cluster> = Vec::new();
    median_cut_partition(
        lab_l,
        lab_a,
        lab_b,
        all_indices,
        0,
        partition.max_depth,
        partition.variance_threshold,
        &mut clusters,
    );

    merge_small_clusters(
        &mut clusters,
        lab_l,
        lab_a,
        lab_b,
        partition.min_cluster_area_ratio,
    );
    reduce_cluster_count(&mut clusters, lab_l, lab_a, lab_b, partition.max_clusters);

    // Score clusters
    score_clusters(
        &mut clusters,
        lab_l,
        lab_a,
        lab_b,
        w,
        h,
        partition.border_band,
    );

    // Phase 2: Generate mask → barrier-aware BFS → morphology
    let raw_mask = clusters_to_bg_mask(&clusters, n, partition.bg_score_threshold);
    let border_bg = border_background_likelihood(
        lab_l,
        lab_a,
        lab_b,
        w,
        h,
        partition.border_band,
        &params.soft_mask,
    );
    let bg_candidate = build_bg_candidate(&border_bg, params.flood_barrier.color_threshold);
    let bg_barrier = build_flood_barrier(feature_inputs, &border_bg, &params.flood_barrier);
    let foreground_protect = build_foreground_protect(feature_inputs, &params.flood_barrier);

    let bfs_mask = bfs_connected_bg(
        &raw_mask,
        &bg_candidate,
        &bg_barrier,
        lab_l,
        lab_a,
        lab_b,
        w,
        h,
        partition.border_band,
        partition.bg_score_threshold,
        partition.bg_connect_threshold,
    );

    // max_bg_ratio guard: if too much is bg, fall back to raw_mask
    let connected_mask = if mask_mean(&bfs_mask) > partition.max_bg_ratio {
        raw_mask.clone()
    } else {
        bfs_mask
    };

    // Morphology: closing → opening → erode
    let morph_mask = if morph.close_radius > 0 {
        let closed = closing(&connected_mask, w, h, morph.close_radius);
        if morph.open_radius > 0 {
            opening(&closed, w, h, morph.open_radius)
        } else {
            closed
        }
    } else if morph.open_radius > 0 {
        opening(&connected_mask, w, h, morph.open_radius)
    } else {
        connected_mask.clone()
    };

    let morph_mask = if morph.erode_radius > 0 {
        erode(&morph_mask, w, h, morph.erode_radius)
    } else {
        morph_mask
    };

    // max_bg_ratio guard again
    let morph_mask = if mask_mean(&morph_mask) > partition.max_bg_ratio {
        raw_mask.clone()
    } else {
        morph_mask
    };

    let bg_mask_before_protect = morph_mask.clone();
    let morph_mask = if params.flood_barrier.enabled {
        apply_foreground_protect(
            &morph_mask,
            &foreground_protect,
            params.flood_barrier.protect_strength,
        )
    } else {
        morph_mask
    };
    let bg_mask_after_protect = morph_mask.clone();

    // Blend hard morph mask with soft border_bg
    let soft_bg = soft_background_from_partition(&morph_mask, &border_bg);

    // Compute foreground confidence (uses subject_prior with default params)
    let default_subject = SubjectPriorParams::default();
    let subject_map: Vec<f64> = (0..n)
        .map(|i| {
            let x = i % w as usize;
            let y = i / w as usize;
            subject_prior(x, y, w as usize, h as usize, &default_subject)
        })
        .collect();

    let fg_confidence =
        soft_foreground_from_background(&soft_bg, w, h, &subject_map, &params.soft_mask);

    // Use the edge-friendly soft foreground map as the foreground-oriented mask feature.
    // 硬 mask 仅参与软背景估计的弱先验；直接输出它会把 Median Cut 的块状边界带到汇总图。
    let morph_stats = mask_stats(&morph_mask);
    let bg_mask_morph: Vec<f64> = if morph_stats.mean <= 0.02 || morph_stats.mean >= 0.98 {
        fg_confidence.clone()
    } else {
        morph_mask
            .iter()
            .zip(fg_confidence.iter())
            .map(|(&m, &f)| {
                let hard_fg = 1.0 - m;
                (hard_fg * 0.20 + f * 0.80).clamp(0.0, 1.0)
            })
            .collect()
    };

    let diagnostics = BackgroundDiagnostics {
        bg_candidate,
        bg_barrier,
        bg_mask_before_protect,
        foreground_protect,
        bg_mask_after_protect,
        fg_confidence: fg_confidence.clone(),
    };

    BackgroundFeatureResult {
        bg_mask_morph,
        fg_confidence,
        diagnostics,
    }
}

/// Compute subject prior (Gaussian center bias) for every pixel.
pub fn compute_subject_prior(w: u32, h: u32, params: &SubjectPriorParams) -> Vec<f64> {
    let n = (w * h) as usize;
    if n == 0 {
        return vec![];
    }
    let wu = w as usize;
    let hu = h as usize;
    let mut out = Vec::with_capacity(n);
    for y in 0..hu {
        for x in 0..wu {
            out.push(subject_prior(x, y, wu, hu, params));
        }
    }
    out
}
