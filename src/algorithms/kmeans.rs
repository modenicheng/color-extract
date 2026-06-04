use super::{make_entry, sort_by_lightness, AlgorithmResult, PaletteEntry};
use crate::colorspace::ColorSpace;
use crate::timing::timed;
use anyhow::Result;
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Run KMeans++ color extraction on normalized RGB pixels.
///
/// Converts pixels to the target color space, clusters with KMeans++
/// (default init), then converts centroids back to RGB for the palette.
pub fn run(pixels: &[[f64; 3]], cs: ColorSpace, k: usize, rng_seed: u64) -> Result<AlgorithmResult> {
    let n_pixels = pixels.len();
    let k = k.min(n_pixels);

    // No pixels → empty palette
    if k == 0 {
        return Ok(AlgorithmResult {
            palette: vec![],
            dominant: make_entry([0.0, 0.0, 0.0], 0.0),
            duration: std::time::Duration::ZERO,
        });
    }

    let (build_result, elapsed) = timed(|| -> Result<AlgorithmResult> {
        // a: Convert all pixels from normalized RGB to target color space
        let converted = cs.convert_batch_to(pixels);
        let n = converted.len();

        // b: Build ndarray::Array2<f64> of shape (n, 3)
        let flat: Vec<f64> = converted.iter().flat_map(|c| c.iter().copied()).collect();
        let array = Array2::from_shape_vec((n, 3), flat)?;

        // c: Create DatasetBase from array (unit targets)
        let dataset = DatasetBase::from(array);

        // d: Create reproducible RNG
        let rng = Xoshiro256Plus::seed_from_u64(rng_seed);

        // e: Configure KMeans — KMeansPlusPlus init is the default
        let model = KMeans::params_with_rng(k, rng)
            .n_runs(1)
            .tolerance(1e-3)
            .max_n_iterations(50)
            .fit(&dataset)?;

        // h: Predict cluster assignments and count per cluster
        let assignments: Array1<usize> = model.predict(&dataset);
        let total = assignments.len() as f64;
        let mut counts = vec![0usize; k];
        for &c in assignments.iter() {
            counts[c] += 1;
        }

        // i: Dominant = centroid of cluster with the largest membership
        let dominant_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, cnt)| *cnt)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // g+j: Extract centroids and convert back to normalized RGB
        let centroid_slice = model.centroids().as_slice().unwrap();
        let centroid_rows: Vec<[f64; 3]> = (0..k)
            .map(|i| {
                let b = i * 3;
                [
                    centroid_slice[b],
                    centroid_slice[b + 1],
                    centroid_slice[b + 2],
                ]
            })
            .collect();
        let rgb_centroids = cs.convert_batch_from(&centroid_rows);

        // k: Build PaletteEntry for each centroid with its pixel proportion
        let mut palette: Vec<PaletteEntry> = rgb_centroids
            .iter()
            .enumerate()
            .map(|(i, rgb)| {
                let proportion = if counts[i] > 0 {
                    counts[i] as f64 / total
                } else {
                    0.0
                };
                make_entry(*rgb, proportion)
            })
            .collect();

        // Snapshot dominant entry before sorting (dominant_idx refers to centroid order)
        let dominant = palette[dominant_idx].clone();

        // l: Sort palette by CIELAB L* ascending (dark → light)
        sort_by_lightness(&mut palette);

        Ok(AlgorithmResult {
            palette,
            dominant,
            duration: std::time::Duration::ZERO, // filled in below
        })
    });

    let mut result = build_result?;
    result.duration = elapsed;
    Ok(result)
}
