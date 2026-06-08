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
}

#[derive(Debug, Deserialize)]
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
    pub global_sat: f64,
    pub local_light: f64,
    pub local_sat: f64,
    pub background_lab: f64,
    pub background_fg_confidence: f64,
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
    pub sat: RobustCenterParams,
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

fn default_trimmed_mean_weight() -> f64 { 0.65 }
fn default_median_weight() -> f64 { 0.35 }
fn default_gamma_power() -> f64 { 0.5 }
fn default_log_base() -> f64 { std::f64::consts::E }

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SpectralResidualParams {
    /// 均值滤波核大小（奇数），1 或 0 = 不滤波
    pub mean_filter_kernel: u32,
    /// 频谱残差 IFFT 后的 Gaussian blur sigma
    pub gaussian_sigma: f64,
    /// 输入 gamma 校正指数（FFT 前对像素值做 powf），1.0 = 无变化
    pub gamma: f64,
}

impl Default for SpectralResidualParams {
    fn default() -> Self {
        Self {
            mean_filter_kernel: 3,
            gaussian_sigma: 3.0,
            gamma: 1.0,
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
// Background 参数
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BackgroundParams {
    /// 边界采样 band 宽度（像素），默认 3
    #[serde(default = "default_border_band")]
    pub border_band: u32,
    /// 低端 trim 百分位，默认 10.0
    #[serde(default = "default_bg_trim_low")]
    pub trim_low: f64,
    /// 高端 trim 百分位，默认 90.0
    #[serde(default = "default_bg_trim_high")]
    pub trim_high: f64,
    /// trimmed_mean 混合系数，默认 0.7
    #[serde(default = "default_bg_trimmed_mean_weight")]
    pub trimmed_mean_weight: f64,
    /// median 混合系数，默认 0.3
    #[serde(default = "default_bg_median_weight")]
    pub median_weight: f64,
    /// 连通性参数
    #[serde(default)]
    pub connectedness: BackgroundConnectednessParams,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct BackgroundConnectednessParams {
    /// BFS 距离阈值 = bg_max_dist × dist_threshold_factor，默认 1.5
    pub dist_threshold_factor: f64,
    /// mask blur sigma（0=不 blur），默认 2.0
    pub blur_sigma: f32,
    /// 前景置信度 strength，默认 0.85
    pub strength: f64,
}

impl Default for BackgroundConnectednessParams {
    fn default() -> Self {
        Self {
            dist_threshold_factor: 1.5,
            blur_sigma: 2.0,
            strength: 0.85,
        }
    }
}

fn default_border_band() -> u32 { 3 }
fn default_bg_trim_low() -> f64 { 10.0 }
fn default_bg_trim_high() -> f64 { 90.0 }
fn default_bg_trimmed_mean_weight() -> f64 { 0.7 }
fn default_bg_median_weight() -> f64 { 0.3 }

/// 校验 filter 配置：互斥检查 + 值域检查
pub fn validate_filter(filter: &FilterParams) {
    match filter.method.as_str() {
        "threshold" => {
            if filter.quantile.is_some() {
                panic!(
                    "Filter config conflict: method='{}' but quantile is also set. \
                     These two methods are mutually exclusive.",
                    filter.method
                );
            }
            let t = filter.threshold.unwrap_or_else(|| {
                panic!("filter.threshold is required when method='threshold'");
            });
            assert!(
                (0.0..=1.0).contains(&t),
                "filter.threshold must be in [0, 1], got {t}"
            );
        }
        "quantile" => {
            if filter.threshold.is_some() {
                panic!(
                    "Filter config conflict: method='{}' but threshold is also set. \
                     These two methods are mutually exclusive.",
                    filter.method
                );
            }
            let q = filter.quantile.unwrap_or_else(|| {
                panic!("filter.quantile is required when method='quantile'");
            });
            assert!(
                q > 0.0 && q <= 100.0,
                "filter.quantile must be in (0, 100], got {q}"
            );
        }
        other => panic!("filter.method must be 'threshold' or 'quantile', got '{other}'"),
    }
}

pub fn load_params(path: &Path) -> Result<Params> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let params: Params = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(params)
}
