// =============================================================================
// YAML 参数定义 + 加载
// =============================================================================

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Params {
    pub max_dim: u32,
    pub gauss_sigma: f32,
    pub percentile: PercentileParams,
    pub global_residual: GlobalResidualParams,
    pub weights_add: Weights,
    pub weights_mul: Weights,
    pub fusion: FusionParams,
    pub filter: Option<FilterParams>,
    pub contact_sheet: ContactSheetParams,
    #[serde(default)]
    pub spectral_residual: SpectralResidualParams,
    #[serde(default)]
    pub dct: DctParams,
    pub background: BackgroundParams,
    #[serde(default)]
    pub subject_prior: SubjectPriorParams,
    #[serde(default)]
    pub impression: ImpressionParams,
    #[serde(default)]
    pub dynamic_weights: DynamicWeightsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PercentileParams {
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Deserialize)]
pub struct Weights {
    pub dct: f64,
    pub lab_grad: f64,
    pub spectral: f64,
    pub global_light: f64,
    pub global_lab_a: f64,
    pub global_lab_b: f64,
    pub local_light: f64,
    pub local_lab_a: f64,
    pub local_lab_b: f64,
    pub background_mask_morph: f64,
    pub background_fg_confidence: f64,
    pub subject_prior: f64,
}

// =============================================================================
// Dynamic Feature Weights 配置
// =============================================================================

/// 每个特征的动态权重开关
#[derive(Debug, Deserialize, Clone)]
pub struct FeatureDynamicConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for FeatureDynamicConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// 各特征动态权重开关（12 个特征）
#[derive(Debug, Deserialize, Clone)]
pub struct DynamicWeightsPerFeature {
    #[serde(default)]
    pub dct: FeatureDynamicConfig,
    #[serde(default)]
    pub lab_grad: FeatureDynamicConfig,
    #[serde(default)]
    pub spectral: FeatureDynamicConfig,
    #[serde(default)]
    pub global_light: FeatureDynamicConfig,
    #[serde(default)]
    pub global_lab_a: FeatureDynamicConfig,
    #[serde(default)]
    pub global_lab_b: FeatureDynamicConfig,
    #[serde(default)]
    pub local_light: FeatureDynamicConfig,
    #[serde(default)]
    pub local_lab_a: FeatureDynamicConfig,
    #[serde(default)]
    pub local_lab_b: FeatureDynamicConfig,
    #[serde(default = "default_dw_per_feat_disabled")]
    pub background_mask_morph: FeatureDynamicConfig,
    #[serde(default = "default_dw_per_feat_disabled")]
    pub background_fg_confidence: FeatureDynamicConfig,
    #[serde(default = "default_dw_per_feat_disabled")]
    pub subject_prior: FeatureDynamicConfig,
}

impl Default for DynamicWeightsPerFeature {
    fn default() -> Self {
        Self {
            dct: FeatureDynamicConfig::default(),
            lab_grad: FeatureDynamicConfig::default(),
            spectral: FeatureDynamicConfig::default(),
            global_light: FeatureDynamicConfig::default(),
            global_lab_a: FeatureDynamicConfig::default(),
            global_lab_b: FeatureDynamicConfig::default(),
            local_light: FeatureDynamicConfig::default(),
            local_lab_a: FeatureDynamicConfig::default(),
            local_lab_b: FeatureDynamicConfig::default(),
            background_mask_morph: FeatureDynamicConfig { enabled: false },
            background_fg_confidence: FeatureDynamicConfig { enabled: false },
            subject_prior: FeatureDynamicConfig { enabled: false },
        }
    }
}

fn default_dw_per_feat_disabled() -> FeatureDynamicConfig {
    FeatureDynamicConfig { enabled: false }
}

/// stat_mix: 方差/范围/峰度三项在显著度分数中的权重
#[derive(Debug, Deserialize, Clone)]
pub struct StatMixParams {
    #[serde(default = "default_dw_variance_mix")]
    pub variance: f64,
    #[serde(default = "default_dw_range_mix")]
    pub range: f64,
    #[serde(default = "default_dw_peakiness_mix")]
    pub peakiness: f64,
}

impl Default for StatMixParams {
    fn default() -> Self {
        Self {
            variance: 0.45,
            range: 0.35,
            peakiness: 0.20,
        }
    }
}

fn default_dw_variance_mix() -> f64 {
    0.45
}
fn default_dw_range_mix() -> f64 {
    0.35
}
fn default_dw_peakiness_mix() -> f64 {
    0.20
}

/// 动态特征权重完整配置
#[derive(Debug, Deserialize, Clone)]
pub struct DynamicWeightsConfig {
    /// 是否启用动态权重；false 时行为与旧版完全一致
    #[serde(default = "default_dw_enabled")]
    pub enabled: bool,
    /// multiplier 下限
    #[serde(default = "default_dw_min_multiplier")]
    pub min_multiplier: f64,
    /// multiplier 上限
    #[serde(default = "default_dw_max_multiplier")]
    pub max_multiplier: f64,
    /// 方差参考值
    #[serde(default = "default_dw_variance_ref")]
    pub variance_ref: f64,
    /// 范围参考值
    #[serde(default = "default_dw_range_ref")]
    pub range_ref: f64,
    /// 峰度参考值
    #[serde(default = "default_dw_peakiness_ref")]
    pub peakiness_ref: f64,
    /// 防止除零的 epsilon
    #[serde(default = "default_dw_eps")]
    pub eps: f64,
    /// 三项统计量在显著度分数中的混合权重
    #[serde(default)]
    pub stat_mix: StatMixParams,
    /// percentile clip 参数（用于方差计算前的截断）
    #[serde(default = "default_dw_percentile")]
    pub percentile: PercentileParams,
    /// 各特征动态权重开关
    #[serde(default)]
    pub per_feature: DynamicWeightsPerFeature,
}

impl Default for DynamicWeightsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_multiplier: 0.35,
            max_multiplier: 1.80,
            variance_ref: 0.05,
            range_ref: 0.65,
            peakiness_ref: 4.0,
            eps: 1e-6,
            stat_mix: StatMixParams::default(),
            percentile: PercentileParams {
                low: 2.0,
                high: 98.0,
            },
            per_feature: DynamicWeightsPerFeature::default(),
        }
    }
}

fn default_dw_enabled() -> bool {
    false
}
fn default_dw_min_multiplier() -> f64 {
    0.35
}
fn default_dw_max_multiplier() -> f64 {
    1.80
}
fn default_dw_variance_ref() -> f64 {
    0.05
}
fn default_dw_range_ref() -> f64 {
    0.65
}
fn default_dw_peakiness_ref() -> f64 {
    4.0
}
fn default_dw_eps() -> f64 {
    1e-6
}
fn default_dw_percentile() -> PercentileParams {
    PercentileParams {
        low: 2.0,
        high: 98.0,
    }
}

#[derive(Debug, Deserialize)]
pub struct FusionParams {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
}

#[derive(Debug, Deserialize)]
pub struct GlobalResidualParams {
    pub light: RobustCenterParams,
    pub lab_a: RobustCenterParams,
    pub lab_b: RobustCenterParams,
}

/// 稳健亮度/饱和度中心估计参数
#[derive(Debug, Deserialize)]
pub struct RobustCenterParams {
    /// 感知压缩方式: "gamma" 或 "log"
    pub compression: String,
    /// 低端 trim 百分位 (如 2.0 表示 p2)
    pub trim_low: f64,
    /// 高端 trim 百分位 (如 98.0 表示 p98)
    pub trim_high: f64,
    /// trimmed_mean 混合系数 (默认 0.65)
    #[serde(default = "default_trimmed_mean_weight")]
    pub trimmed_mean_weight: f64,
    /// median 混合系数 (默认 0.35)
    #[serde(default = "default_median_weight")]
    pub median_weight: f64,
    /// gamma 压缩指数 (默认 0.5)
    #[serde(default = "default_gamma_power")]
    pub gamma_power: f64,
    /// log 底数 (默认 e)
    #[serde(default = "default_log_base")]
    pub log_base: f64,
}

fn default_trimmed_mean_weight() -> f64 {
    0.65
}
fn default_median_weight() -> f64 {
    0.35
}
fn default_gamma_power() -> f64 {
    0.5
}
fn default_log_base() -> f64 {
    std::f64::consts::E
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SpectralResidualParams {
    /// 均值滤波核大小（奇数），1 或 0 = 不滤波
    pub mean_filter_kernel: u32,
    /// 频谱残差 IFFT 后的 Gaussian blur sigma
    pub gaussian_sigma: f64,
    /// 输入 gamma 校正指数（FFT 前对像素值做 powf），1.0 = 无变化
    pub gamma: f64,
    /// 后处理 gamma 压缩指数（L₂ 融合 + 归一化后），<1 放大残差、>1 压制，1.0 = 无变化
    #[serde(default = "default_one")]
    pub post_gamma: f64,
}

fn default_one() -> f64 {
    1.0
}

impl Default for SpectralResidualParams {
    fn default() -> Self {
        Self {
            mean_filter_kernel: 3,
            gaussian_sigma: 3.0,
            gamma: 1.0,
            post_gamma: 1.0,
        }
    }
}

// =============================================================================
// DCT 纹理复杂度参数
// =============================================================================

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct DctParams {
    /// 高频判定阈值：u+v >= 此值视为高频分量。调大→更严格（仅极高频率算纹理），调小→更宽松
    pub high_freq_threshold: usize,
}

impl Default for DctParams {
    fn default() -> Self {
        Self {
            high_freq_threshold: 4,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ContactSheetParams {
    pub cols: u32,
    pub rows: u32,
    pub pad: u32,
    pub thumb_w: u32,
    pub label_h: u32,
}

#[derive(Debug, Deserialize)]
pub struct FilterParams {
    pub method: String,
    pub threshold: Option<f64>,
    pub normalize_before: Option<bool>,
    pub quantile: Option<f64>,
}

// =============================================================================
// 印象色 K-Means 聚类参数
// =============================================================================

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ImpressionParams {
    /// 聚类数 (k)
    pub k: usize,
    /// 最大迭代次数
    pub max_iter: usize,
    /// 采样方式: "stride" — 间隔规律网格采样, "all" — 使用全部显著像素
    pub sample_method: String,
    /// 网格采样步长 (仅 sample_method="stride" 时生效)
    pub sample_stride: usize,
    /// K-Means++ 初始化随机数种子 (固定可保证结果可复现)
    pub seed: u64,
}

impl Default for ImpressionParams {
    fn default() -> Self {
        Self {
            k: 4,
            max_iter: 10,
            sample_method: "stride".to_string(),
            sample_stride: 4,
            seed: 42,
        }
    }
}

// =============================================================================
// Background 参数（三阶段管线: 色域切分 + BFS 连通 + 软 mask）
// =============================================================================

/// Phase 3 软 mask 精炼参数（box blur + 前景锐化控制）
#[derive(Debug, Deserialize)]
pub struct SoftMaskParams {
    /// Phase 3 边界背景似然图的 box blur 半径。
    /// 0 = 自适应（基于图像边长: clamp(min(w,h)/48, 4, 16)）。>0 = 固定像素半径。
    #[serde(default = "default_soft_blur")]
    pub border_bg_blur_radius: u32,
    /// Phase 3 前景置信度图的 box blur 半径。
    /// 0 = 自适应（基于图像边长: clamp(min(w,h)/40, 6, 24)）。>0 = 固定像素半径。
    #[serde(default = "default_soft_blur")]
    pub fg_confidence_blur_radius: u32,
    /// Phase 3 前景置信度图的反遮罩锐化强度。0 = 禁用；建议 0.2~0.8。
    #[serde(default)]
    pub fg_confidence_sharpen_amount: f64,
    /// 前景置信度反遮罩锐化半径。仅在 sharpen_amount > 0 时生效。
    #[serde(default = "default_fg_confidence_sharpen_radius")]
    pub fg_confidence_sharpen_radius: u32,
}

impl Default for SoftMaskParams {
    fn default() -> Self {
        Self {
            border_bg_blur_radius: 0,
            fg_confidence_blur_radius: 0,
            fg_confidence_sharpen_amount: 0.0,
            fg_confidence_sharpen_radius: 1,
        }
    }
}

fn default_soft_blur() -> u32 {
    0
}
fn default_fg_confidence_sharpen_radius() -> u32 {
    1
}
fn default_grad_stop() -> f64 {
    0.35
}
fn default_dct_stop() -> f64 {
    0.45
}
fn default_local_light_stop() -> f64 {
    0.40
}
fn default_spectral_stop() -> f64 {
    0.50
}
fn default_color_threshold() -> f64 {
    0.45
}
fn default_barrier_color_relax_threshold() -> f64 {
    0.92
}
fn default_protect_strength() -> f64 {
    0.75
}
fn default_protect_p_low() -> f64 {
    70.0
}
fn default_protect_p_high() -> f64 {
    99.0
}
fn default_protect_grad_weight() -> f64 {
    0.30
}
fn default_protect_dct_weight() -> f64 {
    0.25
}
fn default_protect_spectral_weight() -> f64 {
    0.15
}
fn default_protect_local_light_weight() -> f64 {
    0.15
}
fn default_protect_local_sat_weight() -> f64 {
    0.15
}

#[derive(Debug, Deserialize)]
pub struct BackgroundParams {
    #[serde(default)]
    pub partition: ColorPartitionParams,
    #[serde(default)]
    pub morphology: MorphologyParams,
    #[serde(default)]
    pub soft_mask: SoftMaskParams,
    #[serde(default)]
    pub flood_barrier: BackgroundFloodBarrierParams,
}

/// 背景 flood fill 的前景结构阻断与保护参数。
#[derive(Debug, Deserialize)]
pub struct BackgroundFloodBarrierParams {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_grad_stop")]
    pub grad_stop: f64,
    #[serde(default = "default_dct_stop")]
    pub dct_stop: f64,
    #[serde(default = "default_local_light_stop")]
    pub local_light_stop: f64,
    #[serde(default = "default_spectral_stop")]
    pub spectral_stop: f64,
    #[serde(default = "default_color_threshold")]
    pub color_threshold: f64,
    #[serde(default = "default_barrier_color_relax_threshold")]
    pub barrier_color_relax_threshold: f64,
    #[serde(default = "default_protect_strength")]
    pub protect_strength: f64,
    #[serde(default = "default_protect_p_low")]
    pub protect_p_low: f64,
    #[serde(default = "default_protect_p_high")]
    pub protect_p_high: f64,
    #[serde(default = "default_protect_grad_weight")]
    pub protect_grad_weight: f64,
    #[serde(default = "default_protect_dct_weight")]
    pub protect_dct_weight: f64,
    #[serde(default = "default_protect_spectral_weight")]
    pub protect_spectral_weight: f64,
    #[serde(default = "default_protect_local_light_weight")]
    pub protect_local_light_weight: f64,
    #[serde(default = "default_protect_local_sat_weight")]
    pub protect_local_sat_weight: f64,
}

impl Default for BackgroundFloodBarrierParams {
    fn default() -> Self {
        Self {
            enabled: true,
            grad_stop: 0.35,
            dct_stop: 0.45,
            local_light_stop: 0.40,
            spectral_stop: 0.50,
            color_threshold: 0.45,
            barrier_color_relax_threshold: 0.92,
            protect_strength: 0.75,
            protect_p_low: 70.0,
            protect_p_high: 99.0,
            protect_grad_weight: 0.30,
            protect_dct_weight: 0.25,
            protect_spectral_weight: 0.15,
            protect_local_light_weight: 0.15,
            protect_local_sat_weight: 0.15,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ColorPartitionParams {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    #[serde(default = "default_max_clusters")]
    pub max_clusters: usize,
    #[serde(default = "default_variance_threshold")]
    pub variance_threshold: f64,
    #[serde(default = "default_min_cluster_area_ratio")]
    pub min_cluster_area_ratio: f64,
    #[serde(default = "default_border_band")]
    pub border_band: u32,
    #[serde(default = "default_bg_score_threshold")]
    pub bg_score_threshold: f64,
    #[serde(default = "default_bg_connect_threshold")]
    pub bg_connect_threshold: f64,
    #[serde(default = "default_max_bg_ratio")]
    pub max_bg_ratio: f64,
}

impl Default for ColorPartitionParams {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 5,
            max_clusters: 16,
            variance_threshold: 0.1,
            min_cluster_area_ratio: 0.01,
            border_band: 3,
            bg_score_threshold: 0.55,
            bg_connect_threshold: 0.08,
            max_bg_ratio: 0.85,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_max_depth() -> usize {
    5
}
fn default_max_clusters() -> usize {
    16
}
fn default_variance_threshold() -> f64 {
    0.1
}
fn default_min_cluster_area_ratio() -> f64 {
    0.01
}
fn default_border_band() -> u32 {
    3
}
fn default_bg_score_threshold() -> f64 {
    0.55
}
fn default_bg_connect_threshold() -> f64 {
    0.08
}
fn default_max_bg_ratio() -> f64 {
    0.85
}
fn default_open_radius() -> u32 {
    2
}
fn default_close_radius() -> u32 {
    8
}
fn default_erode_radius() -> u32 {
    3
}

// Note: open_radius/close_radius/erode_radius live in MorphologyParams, not ColorPartitionParams.
// These helpers serve MorphologyParams::default().

#[derive(Debug, Deserialize)]
pub struct MorphologyParams {
    #[serde(default = "default_open_radius")]
    pub open_radius: u32,
    #[serde(default = "default_close_radius")]
    pub close_radius: u32,
    #[serde(default = "default_erode_radius")]
    pub erode_radius: u32,
}

impl Default for MorphologyParams {
    fn default() -> Self {
        Self {
            open_radius: 2,
            close_radius: 8,
            erode_radius: 3,
        }
    }
}

// =============================================================================
// Subject Prior 参数
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct SubjectPriorParams {
    #[serde(default = "default_center_x")]
    pub center_x: f64,
    #[serde(default = "default_center_y")]
    pub center_y: f64,
    #[serde(default = "default_radius_x")]
    pub radius_x: f64,
    #[serde(default = "default_radius_y")]
    pub radius_y: f64,
}

impl Default for SubjectPriorParams {
    fn default() -> Self {
        Self {
            center_x: 0.5,
            center_y: 0.55,
            radius_x: 0.35,
            radius_y: 0.45,
        }
    }
}

fn default_center_x() -> f64 {
    0.5
}
fn default_center_y() -> f64 {
    0.55
}
fn default_radius_x() -> f64 {
    0.35
}
fn default_radius_y() -> f64 {
    0.45
}

/// 校验 filter 配置：互斥检查 + 值域检查
pub fn validate_filter(filter: &FilterParams) -> Result<(), anyhow::Error> {
    match filter.method.as_str() {
        "threshold" => {
            if filter.quantile.is_some() {
                anyhow::bail!(
                    "Filter config conflict: method='{}' but quantile is also set. \
                     These two methods are mutually exclusive.",
                    filter.method
                );
            }
            let t = filter.threshold.ok_or_else(|| {
                anyhow::anyhow!("filter.threshold is required when method='threshold'")
            })?;
            if !(0.0..=1.0).contains(&t) {
                anyhow::bail!("filter.threshold must be in [0, 1], got {t}");
            }
        }
        "quantile" => {
            if filter.threshold.is_some() {
                anyhow::bail!(
                    "Filter config conflict: method='{}' but threshold is also set. \
                     These two methods are mutually exclusive.",
                    filter.method
                );
            }
            let q = filter.quantile.ok_or_else(|| {
                anyhow::anyhow!("filter.quantile is required when method='quantile'")
            })?;
            if !(q > 0.0 && q <= 100.0) {
                anyhow::bail!("filter.quantile must be in (0, 100], got {q}");
            }
        }
        other => anyhow::bail!("filter.method must be 'threshold' or 'quantile', got '{other}'"),
    }
    Ok(())
}

pub fn load_params(path: &Path) -> Result<Params> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let params: Params = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(params)
}
