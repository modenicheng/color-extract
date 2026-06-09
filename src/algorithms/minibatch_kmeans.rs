use super::{AlgorithmResult, make_entry, sort_by_lightness};
use crate::colorspace::ColorSpace;
use crate::timing::timed;
use anyhow::{Result, anyhow};
use linfa::DatasetBase;
use linfa::traits::{FitWith, Predict};
use linfa_clustering::{IncrKMeansError, KMeans};
use ndarray::Array2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const BATCH_SIZE: usize = 2048;

/// Run the Mini-Batch KMeans color extraction algorithm.
///
/// # Pipeline
/// 1. Convert all pixels to the target color space.
/// 2. Build an ndarray matrix of shape (n_pixels, 3).
/// 3. Shuffle the dataset with a seeded RNG.
/// 4. Perform incremental Mini-Batch KMeans using `linfa::traits::FitWith`.
/// 5. After convergence (or max iterations), predict cluster assignments for all pixels.
/// 6. Build a palette from the resulting centroids, sorted by perceptual lightness.
pub fn run(
    pixels: &[[f64; 3]],
    cs: ColorSpace,
    k: usize,
    rng_seed: u64,
) -> Result<AlgorithmResult> {
    let (result, duration) = timed(|| -> Result<AlgorithmResult> {
        // Convert all pixels to the target color space, filtering NaN / inf.
        let converted: Vec<[f64; 3]> = cs
            .convert_batch_to(pixels)
            .into_iter()
            .filter(|c| c.iter().all(|v| v.is_finite()))
            .collect();

        let n = converted.len();
        if n == 0 {
            return Err(anyhow!("no valid pixels remain after conversion"));
        }
        // Clip k to the available pixel count.
        let k = k.min(n);

        // Build an ndarray matrix of shape (n, 3).
        let data: Vec<f64> = converted.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
        let array = Array2::from_shape_vec((n, 3), data)
            .map_err(|e| anyhow!("failed to build ndarray from {} pixels: {}", n, e))?;

        // Keep a copy of the array for the final full-dataset prediction.
        let full_array = array.clone();
        // Use Array1<()> targets so `shuffle` and other dataset operations work.
        let targets = ndarray::Array1::from_elem(n, ());
        let dataset = DatasetBase::new(array, targets);

        // Shuffle with a seeded RNG for reproducible stochastic minibatch sampling.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
        let dataset = dataset.shuffle(&mut rng);

        // Create the KMeans parameter builder with a fresh RNG clone.
        let rng_params = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
        let params = KMeans::params_with_rng(k, rng_params)
            .tolerance(1e-2)
            .max_n_iterations(30);

        // ── 5. Mini-batch training loop ─────────────────────────────────
        // Cycle through chunks up to 3× the number of batches, capped at 100.
        let num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
        let max_iters = (num_batches / 4).min(20).max(5);

        // Pull batches from the shuffled dataset.
        let mut batches = dataset.sample_chunks(BATCH_SIZE).cycle().take(max_iters);

        // First batch initialises the model — establishes concrete type for inference.
        let first_batch = batches
            .next()
            .ok_or_else(|| anyhow!("no sample chunks available"))?;
        let mut model = match params.fit_with(None, &first_batch) {
            Ok(m) => m,
            Err(IncrKMeansError::NotConverged(m)) => m,
            Err(e) => return Err(anyhow!("Mini-Batch KMeans init failed: {}", e)),
        };

        // Subsequent batches update the model incrementally.
        for batch in batches {
            match params.fit_with(Some(model), &batch) {
                Ok(m) => {
                    model = m;
                    break; // converged
                }
                Err(IncrKMeansError::NotConverged(m)) => {
                    model = m;
                }
                Err(e) => return Err(anyhow!("Mini-Batch KMeans update failed: {}", e)),
            }
        }

        // ── 6. Predict cluster assignments for every pixel ──────────────
        let full_dataset = DatasetBase::new(full_array, ());
        let assignments = model.predict(&full_dataset); // returns Array1<usize>

        // Count how many pixels belong to each cluster.
        let mut counts = vec![0usize; k];
        for &cluster in assignments.iter() {
            counts[cluster] += 1;
        }

        // The dominant colour is the centroid with the most assigned pixels.
        let dominant_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // ── Build the palette from centroids ──────────────────────────────
        let centroids = model.centroids();
        let mut palette = Vec::with_capacity(k);

        for i in 0..k {
            let coords = [centroids[[i, 0]], centroids[[i, 1]], centroids[[i, 2]]];
            let rgb = cs.convert_from(coords);
            let proportion = counts[i] as f64 / n as f64;
            palette.push(make_entry(rgb, proportion));
        }

        sort_by_lightness(&mut palette);

        // Build the dominant entry.
        let dominant_coords = [
            centroids[[dominant_idx, 0]],
            centroids[[dominant_idx, 1]],
            centroids[[dominant_idx, 2]],
        ];
        let dominant_rgb = cs.convert_from(dominant_coords);
        let dominant = make_entry(dominant_rgb, counts[dominant_idx] as f64 / n as f64);

        Ok(AlgorithmResult {
            palette,
            dominant,
            duration: std::time::Duration::ZERO, // filled in by caller
        })
    });

    let mut r = result?;
    r.duration = duration;
    Ok(r)
}
