pub mod kmeans;
pub mod median_cut;
pub mod minibatch_kmeans;
pub mod octree;

use crate::colorspace::ColorSpace;
use anyhow::Result;

/// The four color extraction algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    KMeansPlusPlus,
    MiniBatchKMeans,
    MedianCut,
    Octree,
}

impl Algorithm {
    pub fn all() -> [Self; 4] {
        [
            Self::KMeansPlusPlus,
            Self::MiniBatchKMeans,
            Self::MedianCut,
            Self::Octree,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::KMeansPlusPlus => "KMeans++",
            Self::MiniBatchKMeans => "Mini-Batch KMeans",
            Self::MedianCut => "Median Cut",
            Self::Octree => "Octree Quantization",
        }
    }
}

/// A single color in the extracted palette, with metadata for HTML display.
#[derive(Debug, Clone)]
pub struct PaletteEntry {
    #[allow(dead_code)]
    pub rgb: [u8; 3],
    pub hex: String,
    pub lab_l: f64,
    pub proportion: f64,
}

/// The result of running one (algorithm, colorspace) combination.
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    pub palette: Vec<PaletteEntry>,
    pub dominant: PaletteEntry,
    pub duration: std::time::Duration,
}

/// Helper: build a PaletteEntry from normalized RGB and optional proportion.
fn make_entry(rgb_norm: [f64; 3], proportion: f64) -> PaletteEntry {
    let r = (rgb_norm[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgb_norm[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgb_norm[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    PaletteEntry {
        rgb: [r, g, b],
        hex: format!("#{r:02x}{g:02x}{b:02x}"),
        lab_l: crate::colorspace::perceptual_lightness(rgb_norm),
        proportion,
    }
}

/// Sort palette entries from dark to light (by CIELAB L* ascending).
fn sort_by_lightness(palette: &mut [PaletteEntry]) {
    palette.sort_by(|a, b| {
        a.lab_l
            .partial_cmp(&b.lab_l)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Run one (algorithm, colorspace) combination on the given normalized RGB pixels.
/// `k` is the palette size (target = 10), `rng_seed` for reproducible randomness.
pub fn run_combination(
    pixels: &[[f64; 3]],
    algo: Algorithm,
    cs: ColorSpace,
    k: usize,
    rng_seed: u64,
) -> Result<AlgorithmResult> {
    match algo {
        Algorithm::KMeansPlusPlus => kmeans::run(pixels, cs, k, rng_seed),
        Algorithm::MiniBatchKMeans => minibatch_kmeans::run(pixels, cs, k, rng_seed),
        Algorithm::MedianCut => median_cut::run(pixels, cs, k),
        Algorithm::Octree => octree::run(pixels, cs, k),
    }
}
