// =============================================================================
// Percentile Normalize + Hybrid Fusion + Filter
// =============================================================================

use crate::params::{ClusterScoringParams, FilterParams, ImpressionParams, Weights};
use serde::Serialize;

/// Percentile 归一化到 [0,1]：低于 p_low% 置 0，高于 p_high% 置 1，中间线性拉伸
pub fn percentile_normalize(data: &[f64], p_low: f64, p_high: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
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

/// 对已经是 [0,1] 语义的概率/掩膜特征做归一化。
///
/// 当百分位上下界重合时，普通 percentile normalize 会把全 1 前景 mask 映射成全 0。
/// 这类特征的常量值本身有语义，因此退化时直接保留原值。
pub fn percentile_normalize_unit_feature(data: &[f64], p_low: f64, p_high: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
    let n = sorted.len();
    let lo_idx = ((n as f64) * p_low / 100.0).floor() as usize;
    let hi_idx = ((n as f64) * p_high / 100.0).ceil() as usize;
    let lo_val = sorted[lo_idx.min(n - 1)];
    let hi_val = sorted[hi_idx.min(n - 1)];
    let range = hi_val - lo_val;

    let min_val = sorted[0];
    let max_val = sorted[n - 1];
    let already_unit = min_val >= -1e-9 && max_val <= 1.0 + 1e-9;
    if range.abs() < 1e-12 && already_unit {
        return data.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    }

    let range = range.max(1e-12);
    data.iter()
        .map(|&v| ((v - lo_val) / range).clamp(0.0, 1.0))
        .collect()
}

/// 从 Params 中提取权重数组（长度 = feature 数量）
pub fn weights_to_array(w: &Weights) -> Vec<f64> {
    vec![
        w.dct,
        w.lab_grad,
        w.spectral,
        w.global_light,
        w.global_lab_a,
        w.global_lab_b,
        w.global_sat,
        w.local_light,
        w.local_lab_a,
        w.local_lab_b,
        w.local_sat,
        w.background_mask_morph,
        w.background_fg_confidence,
        w.subject_prior,
        w.abs_light,
        w.abs_lab_a,
        w.abs_lab_b,
        w.abs_sat,
        w.segment_foreground,
    ]
}

/// Hybrid Fusion: 加权加法分支 + 软乘法分支 混合，各自独立权重
pub fn hybrid_fusion(
    features: &[&[f64]],
    add_w: &[f64],
    mul_w: &[f64],
    alpha: f64,
    gamma: f64,
    epsilon: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(add_w.len(), features.len());
    assert_eq!(mul_w.len(), features.len());

    let n = features[0].len();

    // 归一化加法/乘法权重（total=0 时该分支退化为全零）
    let add_total: f64 = add_w.iter().sum();
    let add_norm: Vec<f64> = if add_total > 0.0 {
        add_w.iter().map(|&w| w / add_total).collect()
    } else {
        vec![0.0; add_w.len()]
    };
    let mul_total: f64 = mul_w.iter().sum();
    let mul_norm: Vec<f64> = if mul_total > 0.0 {
        mul_w.iter().map(|&w| w / mul_total).collect()
    } else {
        vec![0.0; mul_w.len()]
    };

    let mut add_score = vec![0.0; n];
    let mut mul_score = vec![0.0; n];
    let mut mul_has_data = vec![false; n];

    for (fi, feat) in features.iter().enumerate() {
        let wa = add_norm[fi];
        let wm = mul_norm[fi];
        for j in 0..n {
            add_score[j] += feat[j] * wa;
            let base = epsilon + (1.0 - epsilon) * feat[j];
            if !mul_has_data[j] {
                mul_score[j] = base.ln() * wm;
                mul_has_data[j] = true;
            } else {
                mul_score[j] += base.ln() * wm;
            }
        }
    }

    // Exponentiate softmul
    for j in 0..n {
        mul_score[j] = mul_score[j].exp();
    }

    // Blend
    let mut hybrid = vec![0.0; n];
    for j in 0..n {
        hybrid[j] = alpha * add_score[j] + (1.0 - alpha) * mul_score[j];
        hybrid[j] = hybrid[j].powf(gamma);
        hybrid[j] = hybrid[j].clamp(0.0, 1.0);
    }

    // Gamma adjust for individual branches too
    for j in 0..n {
        add_score[j] = add_score[j].powf(gamma).clamp(0.0, 1.0);
        mul_score[j] = mul_score[j].powf(gamma).clamp(0.0, 1.0);
    }

    (add_score, mul_score, hybrid)
}

// =============================================================================
// Original × FiltHybrid 复合图 + 印象色
// =============================================================================

/// 将原图 RGB 与过滤后的 FuseHybrid 逐像素相乘，生成复合图。
///
/// 复合图 = original × filtered_hybrid（非显著区域 → 0）。
pub fn composite_with_hybrid(rgb: &[[f64; 3]], filtered: &[f64]) -> Vec<[f64; 3]> {
    rgb.iter()
        .zip(filtered.iter())
        .map(|(&px, &w)| [px[0] * w, px[1] * w, px[2] * w])
        .collect()
}

/// 将原图 RGB 与未过滤的 FuseHybrid 逐像素相乘（无阈值过滤）。
///
/// 通过 `normalize_before` 控制在乘法前是否对 hybrid 做 [0,1] 归一化。
/// 与 `composite_with_hybrid` 不同，这里接受的是未经过滤（全像素都保留）的 hybrid。
pub fn composite_with_hybrid_direct(
    rgb: &[[f64; 3]],
    hybrid: &[f64],
    normalize_before: bool,
) -> Vec<[f64; 3]> {
    let weights: Vec<f64> = if normalize_before {
        let min = hybrid.iter().cloned().fold(f64::MAX, f64::min);
        let max = hybrid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-12);
        hybrid
            .iter()
            .map(|&v| ((v - min) / range).clamp(0.0, 1.0))
            .collect()
    } else {
        hybrid.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    };
    rgb.iter()
        .zip(weights.iter())
        .map(|(&px, &w)| [px[0] * w, px[1] * w, px[2] * w])
        .collect()
}

// =============================================================================
// K-Means++ 印象色提取
// =============================================================================

/// 简单 LCG 随机数生成器 (MMIX-style)
struct Lcg64(u64);

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1442695040888963407))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

#[inline]
fn sq_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// 使用加权 k-means++ 聚类从最终聚类权重图中提取印象色。
///
/// 权重图既作为二值门控（>0 才参与聚类），其值本身也作为该像素的权重。
/// 像素权重越高，对质心位置的牵引力越大。
///
/// 采样方式由 `impression.sample_method` 控制:
///   - "stride": 间隔规律网格采样，按 sample_stride 在 2D 网格上每隔 stride 个像素选一个
///   - "all":    使用全部权重图 > 0 的像素
///
/// k-means++ 初始化使用固定种子 `impression.seed`（默认 42），保证可复现。
/// 最终胜出簇由 `impression.cluster_scoring` 控制。
pub fn kmeans_impression_color(
    rgb: &[[f64; 3]],
    filtered: &[f64],
    w: usize,
    h: usize,
    params: &ImpressionParams,
    background_hint: Option<&[f64]>,
) -> [f64; 3] {
    let k = params.k;
    let max_iter = params.max_iter;

    // ── 收集显著像素点（含原始索引，用于查权重）──
    let (indices, pts): (Vec<usize>, Vec<[f64; 3]>) = if params.sample_method == "stride" {
        let stride = params.sample_stride.max(1);
        let mut idxs = Vec::new();
        let mut sampled = Vec::new();
        for y in (0..h).step_by(stride) {
            for x in (0..w).step_by(stride) {
                let idx = y * w + x;
                if filtered[idx] > 0.0 {
                    idxs.push(idx);
                    sampled.push(rgb[idx]);
                }
            }
        }
        (idxs, sampled)
    } else {
        let idxs: Vec<usize> = (0..rgb.len()).filter(|&i| filtered[i] > 0.0).collect();
        let pts_all: Vec<[f64; 3]> = idxs.iter().map(|&i| rgb[i]).collect();
        (idxs, pts_all)
    };

    let n = pts.len();
    if n == 0 {
        return [0.0; 3];
    }
    if n <= k {
        // 点数不足 k，返回加权均值
        let mut s = [0.0; 3];
        let mut w_sum = 0.0;
        for &i in &indices {
            let wi = filtered[i];
            s[0] += rgb[i][0] * wi;
            s[1] += rgb[i][1] * wi;
            s[2] += rgb[i][2] * wi;
            w_sum += wi;
        }
        if w_sum > 0.0 {
            return [s[0] / w_sum, s[1] / w_sum, s[2] / w_sum];
        }
        return [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64];
    }

    // ── K-means++ 初始化 ── 使用固定种子保证可复现
    let mut rng = Lcg64::new(params.seed);

    let mut centroids = Vec::with_capacity(k);
    centroids.push(pts[rng.next_f64() as usize % n]);

    let mut min_d2 = vec![f64::MAX; n];

    for _ in 1..k {
        // 更新到最新质心的距离，并计算 D² 总和
        let last = &centroids[centroids.len() - 1];
        let total: f64 = pts
            .iter()
            .zip(min_d2.iter_mut())
            .map(|(p, md)| {
                let d2 = sq_dist(p, last);
                *md = (*md).min(d2);
                *md
            })
            .sum();

        if total <= 0.0 {
            // 所有剩余点与已有质心重合 — 补足质心数后退出
            while centroids.len() < k {
                centroids.push(centroids[0]);
            }
            break;
        }

        // 按 D² 权重随机选下一个质心（idx 默认 n-1，防止浮点舍入导致选到 idx=0）
        let mut r = rng.next_f64() * total;
        let mut idx = n - 1;
        for i in 0..n {
            r -= min_d2[i];
            if r <= 0.0 {
                idx = i;
                break;
            }
        }
        centroids.push(pts[idx]);
    }

    // ── Lloyd 迭代（加权质心更新）──
    let mut assign = vec![0usize; n];

    for _ in 0..max_iter {
        // Assignment: 每个点归入最近质心（基于 RGB 距离，无加权）
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = sq_dist(&pts[i], &centroids[0]);
            for j in 1..k {
                let d = sq_dist(&pts[i], &centroids[j]);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update: 加权质心（每像素权重 = filtered[原始索引]）
        let mut sums = vec![[0.0; 3]; k];
        let mut weight_sums = vec![0.0f64; k];
        for i in 0..n {
            let c = assign[i];
            let pt_idx = indices[i];
            let w_i = filtered[pt_idx];
            sums[c][0] += pts[i][0] * w_i;
            sums[c][1] += pts[i][1] * w_i;
            sums[c][2] += pts[i][2] * w_i;
            weight_sums[c] += w_i;
        }
        for j in 0..k {
            if weight_sums[j] > 0.0 {
                centroids[j] = [
                    sums[j][0] / weight_sums[j],
                    sums[j][1] / weight_sums[j],
                    sums[j][2] / weight_sums[j],
                ];
            }
        }
    }

    let (best, _, _) = select_best_cluster(
        &assign,
        &indices,
        &centroids,
        filtered,
        background_hint,
        params,
    );
    centroids[best]
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterDiagnostics {
    pub cluster_id: usize,
    pub rgb: [f64; 3],
    pub hex: String,
    pub count: u32,
    pub area_ratio: f64,
    pub weight_sum: f64,
    pub mean_weight: f64,
    pub p90_weight: f64,
    pub max_weight: f64,
    pub saturation: f64,
    pub background_mean: f64,
    pub area_component: f64,
    pub mean_weight_component: f64,
    pub peak_weight_component: f64,
    pub base_score: f64,
    pub saturation_multiplier: f64,
    pub neutral_penalty: f64,
    pub background_penalty: f64,
    pub skin_likeness: f64,
    pub skin_penalty: f64,
    pub min_area_gate: f64,
    pub final_score: f64,
    pub legacy_weight_sum_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterScoringDiagnostics {
    pub enabled: bool,
    pub winner: usize,
    pub winner_hex: String,
    pub winner_score: f64,
    pub clusters: Vec<ClusterDiagnostics>,
}

/// 加权聚类：对原图像素做 k-means，使用传入的融合图作为每像素权重。
///
/// 与 `kmeans_impression_color` 不同：
///   - 传入 raw hybrid 时所有权重大于 0 的像素参与；传入 FiltHybrid 时只保留过滤后的像素
///   - 每个像素的权重 = fusion map 值（可配是否先归一化）
///   - 聚类更新时使用加权质心
///   - 输出 `impression.cluster_scoring` 评分最高的簇的质心
///
/// 采样方式由 `impression.sample_method` 控制（同印象色聚类）。
pub fn kmeans_weighted_color(
    rgb: &[[f64; 3]],
    hybrid: &[f64],
    w: usize,
    h: usize,
    params: &ImpressionParams,
    normalize_before: bool,
    background_hint: Option<&[f64]>,
) -> ([f64; 3], f64, ClusterScoringDiagnostics) {
    let k = params.k;
    let max_iter = params.max_iter;

    // ── 归一化 hybrid 权重 ──
    let weights: Vec<f64> = if normalize_before {
        let min = hybrid.iter().cloned().fold(f64::MAX, f64::min);
        let max = hybrid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-12);
        hybrid
            .iter()
            .map(|&v| ((v - min) / range).clamp(0.0, 1.0))
            .collect()
    } else {
        hybrid.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    };

    // ── 采样（stride 或 all）──
    let (indices, pts): (Vec<usize>, Vec<[f64; 3]>) = if params.sample_method == "stride" {
        let stride = params.sample_stride.max(1);
        let mut idxs = Vec::new();
        let mut pts_sampled = Vec::new();
        for y in (0..h).step_by(stride) {
            for x in (0..w).step_by(stride) {
                let idx = y * w + x;
                if weights[idx] > 0.0 {
                    idxs.push(idx);
                    pts_sampled.push(rgb[idx]);
                }
            }
        }
        (idxs, pts_sampled)
    } else {
        let idxs: Vec<usize> = (0..rgb.len()).filter(|&i| weights[i] > 0.0).collect();
        let pts_all: Vec<[f64; 3]> = idxs.iter().map(|&i| rgb[i]).collect();
        (idxs, pts_all)
    };

    let n = pts.len();
    if n == 0 {
        return (
            [0.0; 3],
            0.0,
            empty_cluster_diagnostics(params.cluster_scoring.enabled),
        );
    }
    if n <= k {
        let mut s = [0.0; 3];
        let mut w_sum = 0.0;
        for &i in &indices {
            s[0] += rgb[i][0] * weights[i];
            s[1] += rgb[i][1] * weights[i];
            s[2] += rgb[i][2] * weights[i];
            w_sum += weights[i];
        }
        if w_sum > 0.0 {
            let color = [s[0] / w_sum, s[1] / w_sum, s[2] / w_sum];
            return (
                color,
                w_sum,
                fallback_cluster_diagnostics(color, n as u32, w_sum, params),
            );
        }
        let color = [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64];
        return (
            color,
            0.0,
            fallback_cluster_diagnostics(color, n as u32, 0.0, params),
        );
    }

    // ── K-means++ 初始化 ──
    let mut rng = Lcg64::new(params.seed);
    let mut centroids = Vec::with_capacity(k);
    centroids.push(pts[rng.next_f64() as usize % n]);

    let mut min_d2 = vec![f64::MAX; n];
    for _ in 1..k {
        let last = &centroids[centroids.len() - 1];
        let total: f64 = pts
            .iter()
            .zip(min_d2.iter_mut())
            .map(|(p, md)| {
                let d2 = sq_dist(p, last);
                *md = (*md).min(d2);
                *md
            })
            .sum();

        if total <= 0.0 {
            while centroids.len() < k {
                centroids.push(centroids[0]);
            }
            break;
        }

        let mut r = rng.next_f64() * total;
        let mut idx = n - 1;
        for i in 0..n {
            r -= min_d2[i];
            if r <= 0.0 {
                idx = i;
                break;
            }
        }
        centroids.push(pts[idx]);
    }

    // ── Lloyd 迭代（加权质心更新）──
    let mut assign = vec![0usize; n];
    for _iter in 0..max_iter {
        // Assignment: 无加权，归入最近质心
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = sq_dist(&pts[i], &centroids[0]);
            for j in 1..k {
                let d = sq_dist(&pts[i], &centroids[j]);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update: 加权质心（每像素权重 = weights[原始索引]）
        let mut sums = vec![[0.0; 3]; k];
        let mut weight_sums = vec![0.0f64; k];
        for i in 0..n {
            let c = assign[i];
            let pt_idx = indices[i];
            let w_i = weights[pt_idx];
            sums[c][0] += pts[i][0] * w_i;
            sums[c][1] += pts[i][1] * w_i;
            sums[c][2] += pts[i][2] * w_i;
            weight_sums[c] += w_i;
        }
        for j in 0..k {
            if weight_sums[j] > 0.0 {
                centroids[j] = [
                    sums[j][0] / weight_sums[j],
                    sums[j][1] / weight_sums[j],
                    sums[j][2] / weight_sums[j],
                ];
            }
        }
    }

    let (best, score, diagnostics) = select_best_cluster(
        &assign,
        &indices,
        &centroids,
        &weights,
        background_hint,
        params,
    );
    (centroids[best], score, diagnostics)
}

fn select_best_cluster(
    assign: &[usize],
    indices: &[usize],
    centroids: &[[f64; 3]],
    weights: &[f64],
    background_hint: Option<&[f64]>,
    params: &ImpressionParams,
) -> (usize, f64, ClusterScoringDiagnostics) {
    let k = centroids.len();
    if k == 0 || assign.is_empty() {
        let diag = empty_cluster_diagnostics(params.cluster_scoring.enabled);
        return (0, 0.0, diag);
    }

    let mut counts = vec![0u32; k];
    let mut weight_sums = vec![0.0f64; k];
    let mut max_weights = vec![0.0f64; k];
    let mut bg_sums = vec![0.0f64; k];
    let mut cluster_weights = vec![Vec::<f64>::new(); k];

    for (sample_i, &cluster_id) in assign.iter().enumerate() {
        if cluster_id >= k {
            continue;
        }
        let idx = indices[sample_i];
        let weight = weights.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        counts[cluster_id] += 1;
        weight_sums[cluster_id] += weight;
        max_weights[cluster_id] = max_weights[cluster_id].max(weight);
        cluster_weights[cluster_id].push(weight);

        if let Some(bg) = background_hint {
            bg_sums[cluster_id] += bg.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        }
    }

    let clusters: Vec<ClusterDiagnostics> = (0..k)
        .map(|cluster_id| {
            let count = counts[cluster_id];
            let area_ratio = count as f64 / assign.len().max(1) as f64;
            let weight_sum = weight_sums[cluster_id];
            let mean_weight = if count > 0 {
                weight_sum / count as f64
            } else {
                0.0
            };
            let p90_weight = percentile_from_values(&mut cluster_weights[cluster_id], 0.90);
            let background_mean = if count > 0 {
                bg_sums[cluster_id] / count as f64
            } else {
                0.0
            };

            score_cluster(
                cluster_id,
                centroids[cluster_id],
                count,
                area_ratio,
                weight_sum,
                mean_weight,
                p90_weight,
                max_weights[cluster_id],
                background_mean,
                &params.cluster_scoring,
            )
        })
        .collect();

    let winner = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.final_score
                .partial_cmp(&b.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let winner_score = clusters[winner].final_score;
    let winner_hex = clusters[winner].hex.clone();
    let diagnostics = ClusterScoringDiagnostics {
        enabled: params.cluster_scoring.enabled,
        winner,
        winner_hex,
        winner_score,
        clusters,
    };

    (winner, winner_score, diagnostics)
}

fn score_cluster(
    cluster_id: usize,
    rgb: [f64; 3],
    count: u32,
    area_ratio: f64,
    weight_sum: f64,
    mean_weight: f64,
    p90_weight: f64,
    max_weight: f64,
    background_mean: f64,
    cfg: &ClusterScoringParams,
) -> ClusterDiagnostics {
    let saturation = rgb_saturation(rgb);
    let area_power = cfg.area_power.max(1e-6);
    let mean_power = cfg.mean_weight_power.max(1e-6);
    let peak_power = cfg.peak_weight_power.max(1e-6);

    let area_component = area_ratio.clamp(0.0, 1.0).powf(area_power);
    let mean_weight_component = mean_weight.clamp(0.0, 1.0).powf(mean_power);
    let peak_weight_component = p90_weight.clamp(0.0, 1.0).powf(peak_power);

    let base_score = if cfg.enabled {
        weighted_average3(
            area_component,
            cfg.area_weight,
            mean_weight_component,
            cfg.mean_weight_weight,
            peak_weight_component,
            cfg.peak_weight_weight,
        )
    } else {
        weight_sum
    };

    let saturation_multiplier = if cfg.enabled {
        1.0 + cfg.saturation_bonus.max(0.0) * saturation
    } else {
        1.0
    };

    let low_sat =
        1.0 - smoothstep(cfg.neutral_sat_threshold, cfg.neutral_sat_threshold + 0.10, saturation);
    let neutral_area_gate = smoothstep(
        cfg.neutral_min_area_ratio,
        cfg.neutral_full_area_ratio,
        area_ratio,
    );
    let neutral_penalty = if cfg.enabled {
        (1.0 - cfg.neutral_penalty_strength.clamp(0.0, 1.0) * low_sat * neutral_area_gate)
            .clamp(0.02, 1.0)
    } else {
        1.0
    };

    let background_penalty = if cfg.enabled {
        let vivid_relax = 1.0 - 0.65 * saturation.clamp(0.0, 1.0);
        (1.0
            - cfg.background_penalty_strength.clamp(0.0, 1.0)
                * background_mean.clamp(0.0, 1.0)
                * vivid_relax)
            .clamp(0.05, 1.0)
    } else {
        1.0
    };

    let skin_likeness_val = skin_likeness(rgb);
    let skin_penalty = if cfg.enabled && cfg.skin_penalty_strength > 0.0 {
        (1.0 - cfg.skin_penalty_strength.clamp(0.0, 1.0) * skin_likeness_val).clamp(0.02, 1.0)
    } else {
        1.0
    };

    let min_area_gate = if cfg.enabled && cfg.min_cluster_area_ratio > 0.0 {
        smoothstep(
            cfg.min_cluster_area_ratio * 0.5,
            cfg.min_cluster_area_ratio,
            area_ratio,
        )
    } else {
        1.0
    };

    let final_score = if cfg.enabled {
        base_score * saturation_multiplier * neutral_penalty * background_penalty * skin_penalty * min_area_gate
    } else {
        weight_sum
    };

    ClusterDiagnostics {
        cluster_id,
        rgb,
        hex: rgb_to_hex(rgb),
        count,
        area_ratio,
        weight_sum,
        mean_weight,
        p90_weight,
        max_weight,
        saturation,
        background_mean,
        area_component,
        mean_weight_component,
        peak_weight_component,
        base_score,
        saturation_multiplier,
        neutral_penalty,
        background_penalty,
        skin_likeness: skin_likeness_val,
        skin_penalty,
        min_area_gate,
        final_score,
        legacy_weight_sum_score: weight_sum,
    }
}

fn empty_cluster_diagnostics(enabled: bool) -> ClusterScoringDiagnostics {
    ClusterScoringDiagnostics {
        enabled,
        winner: 0,
        winner_hex: "#000000".to_string(),
        winner_score: 0.0,
        clusters: Vec::new(),
    }
}

fn fallback_cluster_diagnostics(
    color: [f64; 3],
    count: u32,
    score: f64,
    params: &ImpressionParams,
) -> ClusterScoringDiagnostics {
    let cluster = score_cluster(
        0,
        color,
        count,
        1.0,
        score,
        if count > 0 { score / count as f64 } else { 0.0 },
        if count > 0 { score / count as f64 } else { 0.0 },
        if count > 0 { score / count as f64 } else { 0.0 },
        0.0,
        &params.cluster_scoring,
    );
    ClusterScoringDiagnostics {
        enabled: params.cluster_scoring.enabled,
        winner: 0,
        winner_hex: cluster.hex.clone(),
        winner_score: cluster.final_score,
        clusters: vec![cluster],
    }
}

fn percentile_from_values(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
    let idx = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn weighted_average3(a: f64, aw: f64, b: f64, bw: f64, c: f64, cw: f64) -> f64 {
    let aw = aw.max(0.0);
    let bw = bw.max(0.0);
    let cw = cw.max(0.0);
    let total = aw + bw + cw;
    if total <= 1e-12 {
        return (a + b + c) / 3.0;
    }
    (a * aw + b * bw + c * cw) / total
}

fn rgb_saturation(rgb: [f64; 3]) -> f64 {
    let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_c = rgb[0].min(rgb[1]).min(rgb[2]);
    if max_c <= 1e-12 {
        0.0
    } else {
        ((max_c - min_c) / max_c).clamp(0.0, 1.0)
    }
}

/// 肤色似然度 [0, 1]：基于 RGB 色序 R>G>B（偏红暖色调）+ 饱和度/亮度门控。
/// 值越高说明该颜色越像人类肤色。
fn skin_likeness(rgb: [f64; 3]) -> f64 {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    if max_c < 0.06 {
        return 0.0; // 太暗，不可能是肤色
    }

    // 核心条件：R > G > B（偏红的暖色调）
    let rg_norm = ((r - g) / max_c).max(0.0); // R 必须领先 G
    let gb_norm = ((g - b) / max_c).max(0.0); // G 必须领先 B

    // R 需显著领先 G，G 略领先 B 即可
    let rg_score = smoothstep(0.02, 0.10, rg_norm);
    let gb_score = smoothstep(0.0, 0.05, gb_norm);

    // 饱和度门控：太灰的颜色不是肤色
    let spread = (max_c - min_c) / max_c.max(1e-6);
    let sat_gate = smoothstep(0.04, 0.10, spread);

    // 亮度门控：太暗（阴影/曝光不足）或太亮（过曝）排除
    let lightness = (r + g + b) / 3.0;
    let light_gate =
        smoothstep(0.06, 0.12, lightness) * (1.0 - smoothstep(0.90, 0.98, lightness));

    (rg_score * gb_score * sat_gate * light_gate).clamp(0.0, 1.0)
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < 1e-12 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rgb_to_hex(rgb: [f64; 3]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
        (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
        (rgb[2].clamp(0.0, 1.0) * 255.0) as u8
    )
}

// =============================================================================
// 最终 Fuse 图过滤（阈值法 / 分位数法）
// =============================================================================

/// 对融合图做亮度过滤：低于阈值的像素置 0（保留原图高亮区域）
pub fn apply_filter(data: &[f64], filter: &FilterParams) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let brightness: Vec<f64>;
    let threshold: f64;

    match filter.method.as_str() {
        "threshold" => {
            let raw: Vec<f64> = data.to_vec();
            brightness = if filter.normalize_before.unwrap_or(false) {
                let min = raw.iter().cloned().fold(f64::MAX, f64::min);
                let max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = (max - min).max(1e-12);
                raw.iter().map(|&v| (v - min) / range).collect()
            } else {
                raw
            };
            threshold = filter.threshold.unwrap(); // validated
        }
        "quantile" => {
            brightness = data.to_vec();
            let q = filter.quantile.unwrap() / 100.0; // validated
            let mut sorted = data.to_vec();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((1.0 - q) * n as f64).floor() as usize;
            threshold = sorted[idx.min(n - 1)];
        }
        _ => unreachable!(),
    }

    let filtered: Vec<f64> = data
        .iter()
        .zip(brightness.iter())
        .map(|(&orig, &b)| if b >= threshold { orig } else { 0.0 })
        .collect();

    if filter.post_normalize {
        post_normalize_filtered(
            &filtered,
            filter.post_normalize_min,
            filter.post_normalize_gamma,
        )
    } else {
        filtered
    }
}

/// 阈值过滤后只重拉伸保留区域，过滤掉的 0 继续保持为 0。
fn post_normalize_filtered(data: &[f64], min_out: f64, gamma: f64) -> Vec<f64> {
    let min_out = min_out.clamp(0.0, 1.0);
    let gamma = if gamma.is_finite() && gamma > 0.0 {
        gamma
    } else {
        1.0
    };
    let mut lo = f64::MAX;
    let mut hi = f64::NEG_INFINITY;

    for &v in data {
        if v > 0.0 {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }

    if !lo.is_finite() || !hi.is_finite() {
        return data.to_vec();
    }

    let range = hi - lo;
    if range <= 1e-12 {
        return data
            .iter()
            .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
            .collect();
    }

    data.iter()
        .map(|&v| {
            if v > 0.0 {
                let t = ((v - lo) / range).clamp(0.0, 1.0).powf(gamma);
                min_out + t * (1.0 - min_out)
            } else {
                0.0
            }
        })
        .collect()
}
