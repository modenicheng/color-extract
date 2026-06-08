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
