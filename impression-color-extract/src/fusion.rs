// =============================================================================
// Percentile 归一化 + 特征融合
// =============================================================================

/// Percentile 归一化到 [0,1]
pub fn percentile_normalize(data: &[f64], p_low: f64, p_high: f64) -> Vec<f64> {
    if data.is_empty() { return Vec::new(); }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
    let n = sorted.len();
    let lo_idx = ((n as f64) * p_low / 100.0).floor() as usize;
    let hi_idx = ((n as f64) * p_high / 100.0).ceil() as usize;
    let lo_val = sorted[lo_idx.min(n - 1)];
    let hi_val = sorted[hi_idx.min(n - 1)];
    let range = (hi_val - lo_val).max(1e-12);
    data.iter().map(|&v| ((v - lo_val) / range).clamp(0.0, 1.0)).collect()
}

/// 带权融合: Σ(w_i × feat_i) / Σ(w_i)
pub fn fuse_features(features: &[&[f64]], weights: &[f64]) -> Vec<f64> {
    let n = features[0].len();
    let w_sum: f64 = weights.iter().sum();
    if w_sum < 1e-12 { return vec![0.0; n]; }
    let mut result = vec![0.0; n];
    for (fi, feat) in features.iter().enumerate() {
        let w = weights[fi] / w_sum;
        for j in 0..n {
            result[j] += feat[j] * w;
        }
    }
    result
}
