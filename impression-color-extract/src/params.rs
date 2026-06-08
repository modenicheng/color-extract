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
    pub color_partition: ColorPartitionParams,
    pub feature_weights: FeatureWeights,
    pub palette: PaletteParams,
    pub output: OutputParams,
}

#[derive(Debug, Deserialize)]
pub struct PercentileParams {
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Deserialize)]
pub struct ColorPartitionParams {
    pub enabled: bool,
    pub max_clusters: usize,
    pub max_depth: usize,
    pub target_samples: usize,
    pub variance_threshold: f64,
    pub min_cluster_area_ratio: f64,
    pub border_band: u32,
    pub bg_score_threshold: f64,
    pub bg_connect_threshold: f64,
    pub max_bg_ratio: f64,
    pub open_radius: u32,
    pub close_radius: u32,
    pub erode_radius: u32,
}

#[derive(Debug, Deserialize)]
pub struct FeatureWeights {
    pub dct: f64,
    pub lab_grad: f64,
    pub spectral: f64,
    pub local_light: f64,
    pub local_sat: f64,
    pub bg_mask: f64,
    pub fg_confidence: f64,
}

#[derive(Debug, Deserialize)]
pub struct PaletteParams {
    pub algorithm: String,
    pub n_colors: usize,
    pub min_fg_ratio: f64,
}

#[derive(Debug, Deserialize)]
pub struct OutputParams {
    pub dir: String,
    pub contact_sheet_cols: u32,
    pub contact_sheet_thumb_w: u32,
}

pub fn load_params(path: &Path) -> Result<Params> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let params: Params = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(params)
}
