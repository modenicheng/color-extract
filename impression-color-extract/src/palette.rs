// =============================================================================
// 调色板提取 — KMeans++ / Mini-Batch KMeans / Median Cut / Octree
// =============================================================================

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use crate::params::PaletteParams;

const MAX_KMEANS_TRAIN_PIXELS: usize = 20_000;

#[derive(Debug, Clone)]
pub struct PaletteEntry {
    pub rgb: [f64; 3],
    pub hex: String,
    pub proportion: f64,
}

/// 提取前景主色（仅在前景置信度高的区域采样）
pub fn extract_palette(
    rgb: &[[f64; 3]],
    fg_confidence: &[f64],
    params: &PaletteParams,
) -> Vec<PaletteEntry> {
    let n = rgb.len();
    if n == 0 {
        return vec![];
    }

    // 前景置信度只作为聚类权重，不改写 RGB 本身。
    // 之前把 RGB 乘以权重会把低置信度像素压向黑色，HTML 调色板因此大量变黑。
    let foreground: Vec<([f64; 3], f64)> = rgb
        .iter()
        .zip(fg_confidence.iter())
        .filter_map(|(&c, &w)| {
            let w = w.clamp(0.0, 1.0);
            if w > 1e-6 { Some((c, w)) } else { None }
        })
        .collect();

    let total_weight: f64 = foreground.iter().map(|(_, w)| *w).sum();
    let min_fg_weight = n as f64 * params.min_fg_ratio.clamp(0.0, 1.0);
    let (pixels, weights): (Vec<[f64; 3]>, Vec<f64>) =
        if foreground.is_empty() || total_weight < min_fg_weight {
            (rgb.to_vec(), vec![1.0; n])
        } else {
            (
                foreground.iter().map(|(c, _)| *c).collect(),
                foreground.iter().map(|(_, w)| *w).collect(),
            )
        };

    let k = params.n_colors.min(pixels.len()).max(1);

    let mut entries = match params.algorithm.as_str() {
        "kmeanspp" => kmeans_pp(&pixels, k, &weights),
        "minibatch" => mini_batch_kmeans(&pixels, k, &weights),
        "mediancut" => median_cut_palette(&pixels, k, &weights),
        "octree" => octree_palette(&pixels, k),
        _ => kmeans_pp(&pixels, k, &weights),
    };

    entries.sort_by(|a, b| {
        b.proportion
            .partial_cmp(&a.proportion)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(params.n_colors.max(1));
    entries
}

// ── KMeans++ ──

fn kmeans_pp(pixels: &[[f64; 3]], k: usize, weights: &[f64]) -> Vec<PaletteEntry> {
    let n = pixels.len();
    if n == 0 || k == 0 {
        return vec![];
    }
    let k = k.min(n);

    let train_indices = sample_indices(n, MAX_KMEANS_TRAIN_PIXELS);
    let train_n = train_indices.len();

    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // 初始化：KMeans++ seeding
    let mut centroids: Vec<[f64; 3]> = Vec::with_capacity(k);
    // 第一个中心：按权重随机选择
    let first_train_idx = weighted_choice_sampled(weights, &train_indices, &mut rng);
    centroids.push(pixels[first_train_idx]);

    let mut min_dists = vec![f64::MAX; train_n];
    for _ in 1..k {
        let mut total_dist = 0.0;
        for (ti, &i) in train_indices.iter().enumerate() {
            let p = &pixels[i];
            let d = sq_dist(p, &centroids[centroids.len() - 1]);
            if d < min_dists[ti] {
                min_dists[ti] = d;
            }
            total_dist += min_dists[ti] * weights[i];
        }
        if total_dist < 1e-30 {
            break;
        }
        let threshold = rng.r#gen::<f64>() * total_dist;
        let mut cum = 0.0;
        let mut chosen = train_indices[0];
        for (ti, &d) in min_dists.iter().enumerate() {
            let i = train_indices[ti];
            cum += d * weights[i];
            if cum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(pixels[chosen]);
    }

    let actual_k = centroids.len();

    // 迭代 Lloyd
    let max_iter = 20;
    let mut assignments = vec![usize::MAX; train_n];
    for _iter in 0..max_iter {
        // assign
        let mut changed = false;
        for (ti, &i) in train_indices.iter().enumerate() {
            let p = &pixels[i];
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (j, c) in centroids.iter().enumerate() {
                let d = sq_dist(p, c);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            if assignments[ti] != best {
                assignments[ti] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // update
        let mut sums = vec![[0.0; 3]; actual_k];
        let mut counts = vec![0.0; actual_k];
        for (ti, &i) in train_indices.iter().enumerate() {
            let p = &pixels[i];
            let a = assignments[ti];
            let w = weights[i];
            sums[a][0] += p[0] * w;
            sums[a][1] += p[1] * w;
            sums[a][2] += p[2] * w;
            counts[a] += w;
        }
        for j in 0..actual_k {
            if counts[j] > 0.0 {
                centroids[j] = [
                    sums[j][0] / counts[j],
                    sums[j][1] / counts[j],
                    sums[j][2] / counts[j],
                ];
            }
        }
    }

    // 用全量前景像素重新统计比例，并用全量加权均值修正最终色心。
    let mut total_weight: f64 = weights.iter().sum();
    if total_weight < 1e-10 {
        total_weight = 1.0;
    }
    let mut cluster_weight = vec![0.0; actual_k];
    let mut final_sums = vec![[0.0; 3]; actual_k];
    for (i, p) in pixels.iter().enumerate() {
        let mut best = 0;
        let mut best_d = f64::MAX;
        for (j, c) in centroids.iter().enumerate() {
            let d = sq_dist(p, c);
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        let w = weights[i];
        cluster_weight[best] += w;
        final_sums[best][0] += p[0] * w;
        final_sums[best][1] += p[1] * w;
        final_sums[best][2] += p[2] * w;
    }
    for j in 0..actual_k {
        if cluster_weight[j] > 0.0 {
            centroids[j] = [
                final_sums[j][0] / cluster_weight[j],
                final_sums[j][1] / cluster_weight[j],
                final_sums[j][2] / cluster_weight[j],
            ];
        }
    }

    centroids
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let hex = rgb_to_hex(c);
            let proportion = cluster_weight[i] / total_weight;
            PaletteEntry {
                rgb: c,
                hex,
                proportion,
            }
        })
        .collect()
}

// ── Mini-Batch KMeans ──

fn mini_batch_kmeans(pixels: &[[f64; 3]], k: usize, weights: &[f64]) -> Vec<PaletteEntry> {
    // Simplified: use standard KMeans++ (mini-batch not critical for this scale)
    kmeans_pp(pixels, k, weights)
}

// ── Median Cut (调色板提取版本) ──

fn median_cut_palette(pixels: &[[f64; 3]], k: usize, weights: &[f64]) -> Vec<PaletteEntry> {
    let n = pixels.len();
    if n == 0 || k == 0 {
        return vec![];
    }
    let k = k.min(n);

    let indices: Vec<usize> = (0..n).collect();
    let leaves = median_cut_recursive(pixels, indices, k, 0);

    let total_weight: f64 = weights.iter().sum();
    let total_weight = if total_weight < 1e-10 {
        1.0
    } else {
        total_weight
    };

    leaves
        .into_iter()
        .map(|idx_set| {
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            let mut w_sum = 0.0;
            for &i in &idx_set {
                let w = weights[i];
                sum_r += pixels[i][0] * w;
                sum_g += pixels[i][1] * w;
                sum_b += pixels[i][2] * w;
                w_sum += w;
            }
            let w_sum = if w_sum < 1e-10 { 1.0 } else { w_sum };
            let mean = [sum_r / w_sum, sum_g / w_sum, sum_b / w_sum];
            let proportion = w_sum / total_weight;
            PaletteEntry {
                rgb: mean,
                hex: rgb_to_hex(mean),
                proportion,
            }
        })
        .collect()
}

fn median_cut_recursive(
    pixels: &[[f64; 3]],
    indices: Vec<usize>,
    max_colors: usize,
    depth: usize,
) -> Vec<Vec<usize>> {
    if indices.len() <= 1 || depth > 10 || max_colors <= 1 {
        return vec![indices];
    }

    // 找最大范围通道
    let (mut min_r, mut max_r) = (f64::MAX, f64::MIN);
    let (mut min_g, mut max_g) = (f64::MAX, f64::MIN);
    let (mut min_b, mut max_b) = (f64::MAX, f64::MIN);
    for &i in &indices {
        let p = pixels[i];
        if p[0] < min_r {
            min_r = p[0];
        }
        if p[0] > max_r {
            max_r = p[0];
        }
        if p[1] < min_g {
            min_g = p[1];
        }
        if p[1] > max_g {
            max_g = p[1];
        }
        if p[2] < min_b {
            min_b = p[2];
        }
        if p[2] > max_b {
            max_b = p[2];
        }
    }

    let range_r = max_r - min_r;
    let range_g = max_g - min_g;
    let range_b = max_b - min_b;

    let channel = if range_r >= range_g && range_r >= range_b {
        0
    } else if range_g >= range_b {
        1
    } else {
        2
    };

    let mut sorted: Vec<usize> = indices;
    sorted.sort_by(|&a, &b| pixels[a][channel].partial_cmp(&pixels[b][channel]).unwrap());

    let mid = sorted.len() / 2;
    let left = sorted[..mid].to_vec();
    let right = sorted[mid..].to_vec();

    if left.is_empty() || right.is_empty() {
        return vec![sorted];
    }

    let colors_per_child = (max_colors + 1) / 2;
    let mut result = Vec::new();
    result.extend(median_cut_recursive(
        pixels,
        left,
        colors_per_child,
        depth + 1,
    ));
    result.extend(median_cut_recursive(
        pixels,
        right,
        max_colors - colors_per_child,
        depth + 1,
    ));
    result
}

// ── Octree Quantization ──

fn octree_palette(pixels: &[[f64; 3]], max_colors: usize) -> Vec<PaletteEntry> {
    // 简化的八叉树量化
    // 为保持代码简洁，回退到 Median Cut
    let uniform_weights: Vec<f64> = vec![1.0; pixels.len()];
    median_cut_palette(pixels, max_colors, &uniform_weights)
}

// ── 辅助函数 ──

fn sample_indices(n: usize, max_samples: usize) -> Vec<usize> {
    if n <= max_samples {
        return (0..n).collect();
    }
    (0..max_samples)
        .map(|i| ((i as f64 + 0.5) * n as f64 / max_samples as f64) as usize)
        .map(|i| i.min(n - 1))
        .collect()
}

fn weighted_choice_sampled(weights: &[f64], indices: &[usize], rng: &mut Xoshiro256Plus) -> usize {
    let total: f64 = indices.iter().map(|&i| weights[i]).sum();
    if total < 1e-10 {
        return indices[0];
    }
    let threshold = rng.r#gen::<f64>() * total;
    let mut cum = 0.0;
    for &i in indices {
        cum += weights[i];
        if cum >= threshold {
            return i;
        }
    }
    *indices.last().unwrap()
}

fn sq_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

fn rgb_to_hex(rgb: [f64; 3]) -> String {
    let r = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
    let g = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
    let b = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}
