// =============================================================================
// Dynamic Feature Weights — 根据特征图统计量动态调整融合权重
// =============================================================================
//
// 对每张特征图计算方差、对比度范围、峰度三项统计量，然后映射为 multiplier。
// multiplier 乘到 base weight 上得到 dynamic weight。
//
// 公式:
//   dynamic_weight_i = base_weight_i × multiplier_i
//   如果 base_weight_i == 0 → dynamic_weight_i = 0（不复活用户禁用的特征）
//   如果 per_feature.enabled == false → multiplier_i = 1.0
//
// =============================================================================

use crate::params::{DynamicWeightsConfig, DynamicWeightsPerFeature};

// =============================================================================
// 诊断数据结构
// =============================================================================

/// 每个特征图的统计量和动态权重诊断信息（用于输出 JSON / 控制台表格）
#[derive(Debug, Clone, serde::Serialize)]
pub struct FeatureDiagnostics {
    /// 特征名称
    pub feature: String,
    /// 加法分支 base weight
    pub base_add_weight: f64,
    /// 乘法分支 base weight
    pub base_mul_weight: f64,
    /// percentile-clip 后的方差
    pub variance: f64,
    /// 第 5 百分位
    pub p5: f64,
    /// 第 95 百分位
    pub p95: f64,
    /// p95 - p5
    pub range: f64,
    /// 均值
    pub mean: f64,
    /// 峰度 (p95 / (mean+eps) - 1.0)
    pub peakiness: f64,
    /// 方差得分 (clamped)
    pub variance_score: f64,
    /// 范围得分 (clamped)
    pub range_score: f64,
    /// 峰度得分 (clamped)
    pub peakiness_score: f64,
    /// 综合显著度分数
    pub stat_score: f64,
    /// multiplier (应用于 base weight)
    pub multiplier: f64,
    /// 动态加法权重 = base_add × multiplier
    pub dynamic_add_weight: f64,
    /// 动态乘法权重 = base_mul × multiplier
    pub dynamic_mul_weight: f64,
}

/// 动态权重计算结果
pub struct DynamicWeightsResult {
    /// 动态加法权重数组
    pub add_weights: Vec<f64>,
    /// 动态乘法权重数组
    pub mul_weights: Vec<f64>,
    /// 各特征诊断信息
    pub diagnostics: Vec<FeatureDiagnostics>,
}

// =============================================================================
// Percentile 计算辅助
// =============================================================================

/// 从已排序的数据中取第 p 百分位的值（p ∈ [0, 100]）
#[inline]
fn percentile_from_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * p / 100.0).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// 计算均值
#[inline]
fn compute_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// 计算 percentile-clip 后的方差
///
/// 先对特征图做 [p_low, p_high] percentile clip，再计算方差。
/// 这样可以排除离群值对方差的过度影响。
fn compute_clipped_variance(data: &[f64], sorted: &[f64], p_low: f64, p_high: f64) -> f64 {
    if data.is_empty() || sorted.is_empty() {
        return 0.0;
    }

    let lo = percentile_from_sorted(sorted, p_low);
    let hi = percentile_from_sorted(sorted, p_high);

    // 如果上下界重合（纯色图或极端情况），方差为 0
    if (hi - lo).abs() < 1e-12 {
        return 0.0;
    }

    // 第一遍：clip 后计算均值
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in data {
        let clipped = v.clamp(lo, hi);
        sum += clipped;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f64;

    // 第二遍：计算方差
    let mut var_sum = 0.0;
    for &v in data {
        let clipped = v.clamp(lo, hi);
        let d = clipped - mean;
        var_sum += d * d;
    }

    var_sum / count as f64
}

// =============================================================================
// 单特征统计量计算
// =============================================================================

/// 计算单个特征图的所有统计量
fn compute_feature_stats(
    feature: &[f64],
    sorted: &[f64],
    p_low: f64,
    p_high: f64,
    eps: f64,
) -> (f64, f64, f64, f64, f64) {
    if feature.is_empty() || sorted.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // percentile-clip 后的方差
    let variance = compute_clipped_variance(feature, sorted, p_low, p_high);

    // p5, p95（基于原始数据）
    let p5 = percentile_from_sorted(sorted, 5.0);
    let p95 = percentile_from_sorted(sorted, 95.0);
    let range = p95 - p5;

    // 均值（原始数据）
    let mean = compute_mean(feature);

    // 峰度: p95 / (mean + eps) - 1.0
    let peakiness = if mean + eps > 0.0 {
        (p95 / (mean + eps) - 1.0).max(0.0)
    } else {
        0.0
    };

    (variance, p5, p95, range, peakiness)
}

// =============================================================================
// Multiplier 计算
// =============================================================================

/// 将统计量映射为 multiplier
///
/// 步骤:
///   1. 分别计算 variance_score, range_score, peakiness_score
///   2. 加权求和得到 stat_score
///   3. stat_score 线性映射到 [min_multiplier, max_multiplier]
fn compute_multiplier(
    variance: f64,
    range: f64,
    peakiness: f64,
    cfg: &DynamicWeightsConfig,
) -> (f64, f64, f64, f64, f64) {
    // ── 计算各项得分 ──
    let variance_score = (variance / cfg.variance_ref).clamp(0.0, 1.0);
    let range_score = (range / cfg.range_ref).clamp(0.0, 1.0);
    let peakiness_score = (peakiness / cfg.peakiness_ref).clamp(0.0, 1.0);

    // ── 加权合成显著度分数 ──
    let stat_score = cfg.stat_mix.variance * variance_score
        + cfg.stat_mix.range * range_score
        + cfg.stat_mix.peakiness * peakiness_score;
    let stat_score = stat_score.clamp(0.0, 1.0);

    // ── 映射到 multiplier ──
    let multiplier = cfg.min_multiplier + (cfg.max_multiplier - cfg.min_multiplier) * stat_score;

    (
        variance_score,
        range_score,
        peakiness_score,
        stat_score,
        multiplier,
    )
}

// =============================================================================
// 主入口：计算动态权重
// =============================================================================

/// 对应 Weights 结构体字段顺序的特征名称
const FEATURE_NAMES: [&str; 14] = [
    "dct",
    "lab_grad",
    "spectral",
    "global_light",
    "global_lab_a",
    "global_lab_b",
    "global_sat",
    "local_light",
    "local_lab_a",
    "local_lab_b",
    "local_sat",
    "background_mask_morph",
    "background_fg_confidence",
    "subject_prior",
];

/// 从 per_feature 配置中提取各特征是否启用动态权重
fn per_feature_enabled(per_feat: &DynamicWeightsPerFeature) -> [bool; 14] {
    [
        per_feat.dct.enabled,
        per_feat.lab_grad.enabled,
        per_feat.spectral.enabled,
        per_feat.global_light.enabled,
        per_feat.global_lab_a.enabled,
        per_feat.global_lab_b.enabled,
        per_feat.global_sat.enabled,
        per_feat.local_light.enabled,
        per_feat.local_lab_a.enabled,
        per_feat.local_lab_b.enabled,
        per_feat.local_sat.enabled,
        per_feat.background_mask_morph.enabled,
        per_feat.background_fg_confidence.enabled,
        per_feat.subject_prior.enabled,
    ]
}

/// 计算所有特征的动态权重
///
/// # Arguments
/// * `features` - 14 个特征图（顺序与 Weights 结构体字段一致）
/// * `base_add` - 加法分支 base weight 数组（长度 14）
/// * `base_mul` - 乘法分支 base weight 数组（长度 14）
/// * `cfg` - 动态权重配置
///
/// # Returns
/// * `DynamicWeightsResult` 包含动态权重和诊断信息
pub fn compute_dynamic_weights(
    features: &[&[f64]],
    base_add: &[f64],
    base_mul: &[f64],
    cfg: &DynamicWeightsConfig,
) -> DynamicWeightsResult {
    assert_eq!(features.len(), 14);
    assert_eq!(base_add.len(), 14);
    assert_eq!(base_mul.len(), 14);

    let enabled_flags = per_feature_enabled(&cfg.per_feature);
    let n_features = 14usize;

    let mut add_weights = vec![0.0_f64; n_features];
    let mut mul_weights = vec![0.0_f64; n_features];
    let mut diagnostics = Vec::with_capacity(n_features);

    for i in 0..n_features {
        let feat = features[i];
        let ba = base_add[i];
        let bm = base_mul[i];
        let name = FEATURE_NAMES[i];

        // 如果 base weight 为零，动态后仍为零
        if ba == 0.0 && bm == 0.0 {
            diagnostics.push(FeatureDiagnostics {
                feature: name.to_string(),
                base_add_weight: ba,
                base_mul_weight: bm,
                variance: 0.0,
                p5: 0.0,
                p95: 0.0,
                range: 0.0,
                mean: 0.0,
                peakiness: 0.0,
                variance_score: 0.0,
                range_score: 0.0,
                peakiness_score: 0.0,
                stat_score: 0.0,
                multiplier: 0.0,
                dynamic_add_weight: 0.0,
                dynamic_mul_weight: 0.0,
            });
            add_weights[i] = 0.0;
            mul_weights[i] = 0.0;
            continue;
        }

        // 排序一次，供所有统计量复用
        let mut sorted = feat.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));

        // 计算统计量
        let (variance, p5, p95, range, peakiness) = compute_feature_stats(
            feat,
            &sorted,
            cfg.percentile.low,
            cfg.percentile.high,
            cfg.eps,
        );

        let mean = compute_mean(feat);

        // 计算 multiplier（如果该特征禁用了动态权重，则 multiplier = 1.0）
        let (variance_score, range_score, peakiness_score, stat_score, multiplier) =
            if !enabled_flags[i] {
                (0.0, 0.0, 0.0, 0.0, 1.0)
            } else {
                compute_multiplier(variance, range, peakiness, cfg)
            };

        // 计算动态权重：base × multiplier（base==0 的情况已经在前面处理）
        let dynamic_add = ba * multiplier;
        let dynamic_mul = bm * multiplier;

        diagnostics.push(FeatureDiagnostics {
            feature: name.to_string(),
            base_add_weight: ba,
            base_mul_weight: bm,
            variance,
            p5,
            p95,
            range,
            mean,
            peakiness,
            variance_score,
            range_score,
            peakiness_score,
            stat_score,
            multiplier,
            dynamic_add_weight: dynamic_add,
            dynamic_mul_weight: dynamic_mul,
        });

        add_weights[i] = dynamic_add;
        mul_weights[i] = dynamic_mul;
    }

    DynamicWeightsResult {
        add_weights,
        mul_weights,
        diagnostics,
    }
}

// =============================================================================
// 诊断输出
// =============================================================================

/// 将诊断信息输出为 JSON 字符串
pub fn diagnostics_to_json(diagnostics: &[FeatureDiagnostics]) -> String {
    serde_json::to_string_pretty(diagnostics).unwrap_or_else(|e| format!("JSON error: {e}"))
}

/// 将诊断信息输出为格式化的控制台表格
pub fn diagnostics_to_table(diagnostics: &[FeatureDiagnostics]) -> String {
    let mut lines = Vec::new();

    // 表头
    lines.push(format!(
        "{:<28} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Feature",
        "BaseAdd",
        "BaseMul",
        "Var",
        "Range",
        "Peak",
        "Score",
        "Mult",
        "DynAdd",
        "DynMul",
    ));

    // 分隔线
    lines.push("-".repeat(28 + 7 * 9 + 9));

    // 数据行
    for d in diagnostics {
        lines.push(format!(
            "{:<28} {:7.4} {:7.4} {:7.4} {:7.4} {:7.2} {:7.4} {:7.4} {:7.4} {:7.4}",
            d.feature,
            d.base_add_weight,
            d.base_mul_weight,
            d.variance,
            d.range,
            d.peakiness,
            d.stat_score,
            d.multiplier,
            d.dynamic_add_weight,
            d.dynamic_mul_weight,
        ));
    }

    lines.join("\n")
}
