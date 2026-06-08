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
}

#[derive(Debug, Deserialize)]
pub struct FusionParams {
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
}

#[derive(Debug, Deserialize)]
pub struct GlobalResidualParams {
    pub baseline: String,
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
