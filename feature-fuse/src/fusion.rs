// =============================================================================
// Percentile Normalize + Hybrid Fusion + Filter
// =============================================================================

use crate::params::{FilterParams, Weights};

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

/// 从 Params 中提取权重数组（长度 = feature 数量）
pub fn weights_to_array(w: &Weights) -> Vec<f64> {
    vec![
        w.dct, w.lab_grad, w.spectral,
        w.global_light, w.global_sat,
        w.local_light, w.local_sat,
        w.background_lab, w.background_fg_confidence,
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

    // 归一化加法/乘法权重
    let add_total: f64 = add_w.iter().sum();
    let add_norm: Vec<f64> = add_w.iter().map(|&w| w / add_total).collect();
    let mul_total: f64 = mul_w.iter().sum();
    let mul_norm: Vec<f64> = mul_w.iter().map(|&w| w / mul_total).collect();

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
pub fn composite_with_hybrid(
    rgb: &[[f64; 3]],
    filtered: &[f64],
) -> Vec<[f64; 3]> {
    rgb.iter()
        .zip(filtered.iter())
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
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

/// 使用 k-means++ 聚类从显著像素中提取印象色。
///
/// 仅对 filtered > 0 的像素做聚类，k-means++ 初始化，Lloyd 迭代。
/// 返回像素数最多的簇的质心作为印象色。
pub fn kmeans_impression_color(
    rgb: &[[f64; 3]],
    filtered: &[f64],
    k: usize,
    max_iter: usize,
) -> [f64; 3] {
    // 收集非零像素点
    let pts: Vec<[f64; 3]> = rgb.iter()
        .zip(filtered.iter())
        .filter(|&(_, &w)| w > 0.0)
        .map(|(&c, _)| c)
        .collect();

    let n = pts.len();
    if n == 0 {
        return [0.0; 3];
    }
    if n <= k {
        // 点数不足 k，直接返回均值
        let mut s = [0.0; 3];
        for &p in &pts { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }
        return [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64];
    }

    // ── K-means++ 初始化 ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default() as u64;
    let mut rng = Lcg64::new(seed);

    let mut centroids = Vec::with_capacity(k);
    centroids.push(pts[rng.next_f64() as usize % n]);

    let mut min_d2 = vec![f64::MAX; n];

    for _ in 1..k {
        // 更新到最新质心的距离，并计算 D² 总和
        let last = &centroids[centroids.len() - 1];
        let total: f64 = pts.iter()
            .zip(min_d2.iter_mut())
            .map(|(p, md)| {
                let d2 = sq_dist(p, last);
                *md = (*md).min(d2);
                *md
            })
            .sum();

        if total <= 0.0 {
            break; // 所有剩余点重合，提前退出
        }

        // 按 D² 权重随机选下一个质心
        let mut r = rng.next_f64() * total;
        let mut idx = 0;
        for i in 0..n {
            r -= min_d2[i];
            if r <= 0.0 { idx = i; break; }
        }
        centroids.push(pts[idx]);
    }

    // ── Lloyd 迭代 ──
    let mut assign = vec![0usize; n];

    for _ in 0..max_iter {
        // Assignment: 每个点归入最近质心
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = sq_dist(&pts[i], &centroids[0]);
            for j in 1..k {
                let d = sq_dist(&pts[i], &centroids[j]);
                if d < best_d { best_d = d; best = j; }
            }
            if assign[i] != best { assign[i] = best; changed = true; }
        }
        if !changed { break; }

        // Update: 重新计算质心
        let mut sums = vec![[0.0; 3]; k];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assign[i];
            sums[c][0] += pts[i][0];
            sums[c][1] += pts[i][1];
            sums[c][2] += pts[i][2];
            counts[c] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                centroids[j] = [
                    sums[j][0] / counts[j] as f64,
                    sums[j][1] / counts[j] as f64,
                    sums[j][2] / counts[j] as f64,
                ];
            }
        }
    }

    // 找最大簇
    let mut counts = vec![0u32; k];
    for &a in &assign { counts[a] += 1; }
    let best = (0..k).max_by_key(|&j| counts[j]).unwrap();
    centroids[best]
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
