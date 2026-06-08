use anyhow::Result;
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array2;
use palette::{FromColor, Lab, Srgb};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// One colour cluster with its statistics.
#[derive(Debug, Clone)]
pub struct Cluster4D {
    #[allow(dead_code)]
    pub rgb: [u8; 3],
    pub hex: String,
    pub proportion: f64,       // fraction of pixels belonging to this cluster
    pub avg_complexity: f64,  // mean c (4th dimension) of the cluster
    pub dominant_score: f64,  // softmax(p) × softmax(c) — both normalized across clusters
    pub lab_l: f64,
    pub avg_x: f64,           // mean normalized x (0..1) — 0.0 when not available
    pub avg_y: f64,           // mean normalized y (0..1) — 0.0 when not available
}

/// Result of a 4‑D clustering run.
#[derive(Debug, Clone)]
pub struct ClusterResult4D {
    pub clusters: Vec<Cluster4D>,
    pub dominant: Cluster4D,
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert CIELAB → normalized sRGB [0,1], clamped to gamut.
fn lab_to_rgb_norm(lab: [f64; 3]) -> [f64; 3] {
    let lab_c = Lab::new(lab[0] as f32, lab[1] as f32, lab[2] as f32);
    let srgb: Srgb = Srgb::from_color(lab_c);
    [
        (srgb.red as f64).clamp(0.0, 1.0),
        (srgb.green as f64).clamp(0.0, 1.0),
        (srgb.blue as f64).clamp(0.0, 1.0),
    ]
}

fn make_cluster(
    rgb_norm: [f64; 3],
    proportion: f64,
    avg_c: f64,
    lightness: f64,
    avg_x: f64,
    avg_y: f64,
) -> Cluster4D {
    let r = (rgb_norm[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgb_norm[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgb_norm[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    Cluster4D {
        rgb: [r, g, b],
        hex: format!("#{r:02x}{g:02x}{b:02x}"),
        proportion,
        avg_complexity: avg_c,
        dominant_score: proportion * avg_c,
        lab_l: lightness,
        avg_x,
        avg_y,
    }
}

/// Sort clusters by L* ascending (dark → light).
fn sort_by_lightness(clusters: &mut [Cluster4D]) {
    clusters.sort_by(|a, b| a.lab_l.partial_cmp(&b.lab_l).unwrap_or(std::cmp::Ordering::Equal));
}

/// Apply softmax to both proportion and avg_complexity across clusters,
/// then set `dominant_score` = softmax(p) × softmax(c).
fn softmax_dominant_scores(clusters: &mut [Cluster4D]) {
    if clusters.is_empty() {
        return;
    }
    let max_p = clusters
        .iter()
        .map(|c| c.proportion)
        .fold(f64::NEG_INFINITY, f64::max);
    let denom_p: f64 = clusters
        .iter()
        .map(|c| (c.proportion - max_p).exp())
        .sum();
    let max_c = clusters
        .iter()
        .map(|c| c.avg_complexity)
        .fold(f64::NEG_INFINITY, f64::max);
    let denom_c: f64 = clusters
        .iter()
        .map(|c| (c.avg_complexity - max_c).exp())
        .sum();
    for c in clusters.iter_mut() {
        let sm_p = (c.proportion - max_p).exp() / denom_p;
        let sm_c = (c.avg_complexity - max_c).exp() / denom_c;
        c.dominant_score = sm_p * sm_c;
    }
}

// ---------------------------------------------------------------------------
// Internal: run KMeans++ on 3D data, return centroid-based clusters
// ---------------------------------------------------------------------------

fn run_kmeans_3d(
    data: &[[f64; 3]],
    k: usize,
    rng_seed: u64,
) -> Result<(Vec<Cluster4D>, Cluster4D, Duration)> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok((vec![], make_cluster([0.0; 3], 0.0, 0.0, 0.0, 0.0, 0.0), Duration::ZERO));
    }

    let start = std::time::Instant::now();

    let flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let array = Array2::from_shape_vec((n, 3), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1)
        .tolerance(1e-3)
        .max_n_iterations(50)
        .fit(&dataset)?;

    let assignments = model.predict(&dataset);
    let total = n as f64;
    let mut counts = vec![0usize; k];
    for &cl in assignments.iter() { counts[cl] += 1; }

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 3;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = counts[i] as f64 / total;
            make_cluster(rgb, proportion, 0.0, lab[0].clamp(0.0, 100.0), 0.0, 0.0)
        })
        .collect();

    let dominant_idx = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.proportion
                .partial_cmp(&b.proportion)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok((clusters, dominant, start.elapsed()))
}

/// KMeans++ on plain CIELAB 3D (no complexity) — baseline.
pub fn kmeans_baseline(data: &[[f64; 3]], k: usize, rng_seed: u64) -> Result<ClusterResult4D> {
    let (clusters, dominant, duration) = run_kmeans_3d(data, k, rng_seed)?;
    Ok(ClusterResult4D { clusters, dominant, duration })
}

// ---------------------------------------------------------------------------
// K‑Means++ (4D)
// ---------------------------------------------------------------------------

/// Run K‑Means++ on 4‑D data [L, a, b, c] (CIELAB + complexity).
pub fn kmeans_plus_plus(
    data: &[[f64; 4]],
    k: usize,
    rng_seed: u64,
) -> Result<ClusterResult4D> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok(ClusterResult4D {
            clusters: vec![],
            dominant: make_cluster([0.0; 3], 0.0, 0.0, 0.0, 0.0, 0.0),
            duration: Duration::ZERO,
        });
    }

    let start = std::time::Instant::now();

    // Build (n, 4) ndarray
    let flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let array = Array2::from_shape_vec((n, 4), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1)
        .tolerance(1e-3)
        .max_n_iterations(50)
        .fit(&dataset)?;

    let assignments = model.predict(&dataset);
    let total = assignments.len() as f64;
    let mut counts = vec![0usize; k];
    let mut sum_c = vec![0.0; k]; // sum of c per cluster
    for (i, &cl) in assignments.iter().enumerate() {
        counts[cl] += 1;
        sum_c[cl] += data[i][3];
    }

    let centroid_slice = model.centroids().as_slice().unwrap();

    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 4;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = if counts[i] > 0 { counts[i] as f64 / total } else { 0.0 };
            let avg_c = if counts[i] > 0 { sum_c[i] / counts[i] as f64 } else { 0.0 };
            make_cluster(rgb, proportion, avg_c, lab[0].clamp(0.0, 100.0), 0.0, 0.0)
        })
        .collect();

    // Softmax-normalize p and c, then pick dominant by score
    softmax_dominant_scores(&mut clusters);
    let dominant_idx = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.dominant_score
                .partial_cmp(&b.dominant_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok(ClusterResult4D {
        clusters,
        dominant,
        duration: start.elapsed(),
    })
}

// ---------------------------------------------------------------------------
// Internal: run KMeans on a minibatch subset, predict on all
// ---------------------------------------------------------------------------

fn run_minibatch_3d(
    data: &[[f64; 3]],
    k: usize,
    rng_seed: u64,
) -> Result<(Vec<Cluster4D>, Cluster4D, Duration)> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok((vec![], make_cluster([0.0; 3], 0.0, 0.0, 0.0, 0.0, 0.0), Duration::ZERO));
    }

    let start = std::time::Instant::now();

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
    indices.as_mut_slice().shuffle(&mut rng);

    let batch_n = BATCH_SIZE.min(n);
    let batch_data: Vec<[f64; 3]> = indices[..batch_n].iter().map(|&i| data[i]).collect();

    let flat: Vec<f64> = batch_data.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let array = Array2::from_shape_vec((batch_n, 3), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1)
        .tolerance(1e-3)
        .max_n_iterations(50)
        .fit(&dataset)?;

    let full_flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let full_array = Array2::from_shape_vec((n, 3), full_flat)?;
    let full_dataset = DatasetBase::new(full_array, ());
    let assignments = model.predict(&full_dataset);

    let total = n as f64;
    let mut counts = vec![0usize; k];
    for &cl in assignments.iter() { counts[cl] += 1; }

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 3;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = counts[i] as f64 / total;
            make_cluster(rgb, proportion, 0.0, lab[0].clamp(0.0, 100.0), 0.0, 0.0)
        })
        .collect();

    let dominant_idx = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.proportion
                .partial_cmp(&b.proportion)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok((clusters, dominant, start.elapsed()))
}

/// Mini-Batch KMeans on plain CIELAB 3D (no complexity) — baseline.
pub fn minibatch_baseline(data: &[[f64; 3]], k: usize, rng_seed: u64) -> Result<ClusterResult4D> {
    let (clusters, dominant, duration) = run_minibatch_3d(data, k, rng_seed)?;
    Ok(ClusterResult4D { clusters, dominant, duration })
}

// ---------------------------------------------------------------------------
// Mini‑Batch K‑Means (4D) — direct minibatch: standard KMeans on a subset
// ---------------------------------------------------------------------------

const BATCH_SIZE: usize = 2048;

pub fn mini_batch_kmeans(
    data: &[[f64; 4]],
    k: usize,
    rng_seed: u64,
) -> Result<ClusterResult4D> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok(ClusterResult4D {
            clusters: vec![],
            dominant: make_cluster([0.0; 3], 0.0, 0.0, 0.0, 0.0, 0.0),
            duration: Duration::ZERO,
        });
    }

    let start = std::time::Instant::now();

    // Shuffle & take a minibatch, then run standard KMeans on it
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
    indices.as_mut_slice().shuffle(&mut rng);

    let batch_n = BATCH_SIZE.min(n);
    let batch_data: Vec<[f64; 4]> = indices[..batch_n].iter().map(|&i| data[i]).collect();

    let flat: Vec<f64> = batch_data.iter().flat_map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let array = Array2::from_shape_vec((batch_n, 4), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1)
        .tolerance(1e-3)
        .max_n_iterations(50)
        .fit(&dataset)?;

    // Predict on ALL pixels
    let full_flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let full_array = Array2::from_shape_vec((n, 4), full_flat)?;
    let full_dataset = DatasetBase::new(full_array, ());
    let assignments = model.predict(&full_dataset);

    let total = n as f64;
    let mut counts = vec![0usize; k];
    let mut sum_c = vec![0.0; k];
    for (i, &cl) in assignments.iter().enumerate() {
        counts[cl] += 1;
        sum_c[cl] += data[i][3];
    }

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 4;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = counts[i] as f64 / total;
            let avg_c = if counts[i] > 0 { sum_c[i] / counts[i] as f64 } else { 0.0 };
            make_cluster(rgb, proportion, avg_c, lab[0].clamp(0.0, 100.0), 0.0, 0.0)
        })
        .collect();

    // Softmax-normalize p and c, then pick dominant by score
    softmax_dominant_scores(&mut clusters);
    let dominant_idx = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.dominant_score
                .partial_cmp(&b.dominant_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok(ClusterResult4D {
        clusters,
        dominant,
        duration: start.elapsed(),
    })
}

// ---------------------------------------------------------------------------
// 6‑D clustering [L, a, b, c, nx, ny] — coordinate‑enhanced
// ---------------------------------------------------------------------------

/// Run K‑Means++ on 6‑D data [L, a, b, c, nx, ny] (CIELAB + complexity + pixel coordinates).
pub fn kmeans_plus_plus_6d(
    data: &[[f64; 6]],
    k: usize,
    rng_seed: u64,
) -> Result<ClusterResult4D> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok(ClusterResult4D {
            clusters: vec![],
            dominant: make_cluster([0.0; 3], 0.0, 0.0, 0.0, 0.0, 0.0),
            duration: Duration::ZERO,
        });
    }

    let start = std::time::Instant::now();

    let flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
    let array = Array2::from_shape_vec((n, 6), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1)
        .tolerance(1e-3)
        .max_n_iterations(50)
        .fit(&dataset)?;

    let assignments = model.predict(&dataset);
    let total = assignments.len() as f64;
    let mut counts = vec![0usize; k];
    let mut sum_c = vec![0.0; k];
    let mut sum_x = vec![0.0; k];
    let mut sum_y = vec![0.0; k];
    for (i, &cl) in assignments.iter().enumerate() {
        counts[cl] += 1;
        sum_c[cl] += data[i][3];
        sum_x[cl] += data[i][4];
        sum_y[cl] += data[i][5];
    }

    let centroid_slice = model.centroids().as_slice().unwrap();

    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 6;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = if counts[i] > 0 { counts[i] as f64 / total } else { 0.0 };
            let avg_c = if counts[i] > 0 { sum_c[i] / counts[i] as f64 } else { 0.0 };
            let avg_x = if counts[i] > 0 { sum_x[i] / counts[i] as f64 } else { 0.0 };
            let avg_y = if counts[i] > 0 { sum_y[i] / counts[i] as f64 } else { 0.0 };
            make_cluster(rgb, proportion, avg_c, lab[0].clamp(0.0, 100.0), avg_x, avg_y)
        })
        .collect();

    // Softmax-normalize p and c, then pick dominant by score
    softmax_dominant_scores(&mut clusters);
    let dominant_idx = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.dominant_score
                .partial_cmp(&b.dominant_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok(ClusterResult4D {
        clusters,
        dominant,
        duration: start.elapsed(),
    })
}

/// Mini‑Batch K‑Means on 6‑D data [L, a, b, c, nx, ny] (CIELAB + complexity + coordinates).
pub fn mini_batch_kmeans_6d(
    data: &[[f64; 6]],
    k: usize,
    rng_seed: u64,
) -> Result<ClusterResult4D> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok(ClusterResult4D {
            clusters: vec![],
            dominant: make_cluster([0.0; 3], 0.0, 0.0, 0.0, 0.0, 0.0),
            duration: Duration::ZERO,
        });
    }

    let start = std::time::Instant::now();

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
    indices.as_mut_slice().shuffle(&mut rng);

    let batch_n = BATCH_SIZE.min(n);
    let batch_data: Vec<[f64; 6]> = indices[..batch_n].iter().map(|&i| data[i]).collect();

    let flat: Vec<f64> = batch_data.iter().flat_map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
    let array = Array2::from_shape_vec((batch_n, 6), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1)
        .tolerance(1e-3)
        .max_n_iterations(50)
        .fit(&dataset)?;

    // Predict on ALL pixels
    let full_flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
    let full_array = Array2::from_shape_vec((n, 6), full_flat)?;
    let full_dataset = DatasetBase::new(full_array, ());
    let assignments = model.predict(&full_dataset);

    let total = n as f64;
    let mut counts = vec![0usize; k];
    let mut sum_c = vec![0.0; k];
    let mut sum_x = vec![0.0; k];
    let mut sum_y = vec![0.0; k];
    for (i, &cl) in assignments.iter().enumerate() {
        counts[cl] += 1;
        sum_c[cl] += data[i][3];
        sum_x[cl] += data[i][4];
        sum_y[cl] += data[i][5];
    }

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 6;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = counts[i] as f64 / total;
            let avg_c = if counts[i] > 0 { sum_c[i] / counts[i] as f64 } else { 0.0 };
            let avg_x = if counts[i] > 0 { sum_x[i] / counts[i] as f64 } else { 0.0 };
            let avg_y = if counts[i] > 0 { sum_y[i] / counts[i] as f64 } else { 0.0 };
            make_cluster(rgb, proportion, avg_c, lab[0].clamp(0.0, 100.0), avg_x, avg_y)
        })
        .collect();

    // Softmax-normalize p and c, then pick dominant by score
    softmax_dominant_scores(&mut clusters);
    let dominant_idx = clusters
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.dominant_score
                .partial_cmp(&b.dominant_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok(ClusterResult4D {
        clusters,
        dominant,
        duration: start.elapsed(),
    })
}
