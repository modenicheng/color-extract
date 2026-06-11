// =============================================================================
// Percentile Normalize + Hybrid Fusion + Filter
// =============================================================================

use crate::params::{FilterParams, ImpressionParams, Weights};

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

/// 使用加权 k-means++ 聚类从过滤后的显著像素中提取印象色。
///
/// `filtered` 既作为二值门控（>0 才参与聚类），其值本身也作为该像素的权重。
/// 像素的 FiltHyb 值越高，对质心位置的牵引力越大。
///
/// 采样方式由 `impression.sample_method` 控制:
///   - "stride": 间隔规律网格采样，按 sample_stride 在 2D 网格上每隔 stride 个像素选一个
///   - "all":    使用全部 filtered > 0 的像素（旧行为）
///
/// k-means++ 初始化使用固定种子 `impression.seed`（默认 42），保证可复现。
/// 簇得分 = weight_sum × count，得分最高的簇的质心作为印象色返回。
pub fn kmeans_impression_color(
    rgb: &[[f64; 3]],
    filtered: &[f64],
    w: usize,
    h: usize,
    params: &ImpressionParams,
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

    // 加权得分：weight_sum × count，选得分最高的簇
    let mut counts = vec![0u32; k];
    let mut weight_sums = vec![0.0f64; k];
    for i in 0..n {
        let c = assign[i];
        counts[c] += 1;
        weight_sums[c] += filtered[indices[i]];
    }
    let best = (0..k)
        .max_by(|&a, &b| {
            let sa = weight_sums[a] * counts[a] as f64;
            let sb = weight_sums[b] * counts[b] as f64;
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    centroids[best]
}

/// 加权聚类：对所有原图像素做 k-means，使用 FuseHybrid 作为每像素权重。
///
/// 与 `kmeans_impression_color` 不同：
///   - 不使用阈值过滤（所有像素都参与聚类）
///   - 每个像素的权重 = hybrid 值（可配是否先归一化）
///   - 聚类更新时使用加权质心
///   - 输出「权 × 簇大小」最大的簇的质心（而非纯粹像素数最多的簇）
///
/// 采样方式由 `impression.sample_method` 控制（同印象色聚类）。
pub fn kmeans_weighted_color(
    rgb: &[[f64; 3]],
    hybrid: &[f64],
    w: usize,
    h: usize,
    params: &ImpressionParams,
    normalize_before: bool,
) -> ([f64; 3], f64) {
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
        return ([0.0; 3], 0.0);
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
            return ([s[0] / w_sum, s[1] / w_sum, s[2] / w_sum], w_sum * n as f64);
        }
        return ([s[0] / n as f64, s[1] / n as f64, s[2] / n as f64], 0.0);
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

    // ── 计算每簇的「权 × 大小」得分 ──
    let mut counts = vec![0u32; k];
    let mut weight_sums = vec![0.0f64; k];
    for i in 0..n {
        let c = assign[i];
        counts[c] += 1;
        weight_sums[c] += weights[indices[i]];
    }
    // 得分 = 总权重 × 像素数（权越大、簇越大 → 得分越高）
    let best = (0..k)
        .max_by(|&a, &b| {
            let sa = weight_sums[a] * counts[a] as f64;
            let sb = weight_sums[b] * counts[b] as f64;
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let score = weight_sums[best] * counts[best] as f64;
    (centroids[best], score)
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

    data.iter()
        .zip(brightness.iter())
        .map(|(&orig, &b)| if b >= threshold { orig } else { 0.0 })
        .collect()
}
