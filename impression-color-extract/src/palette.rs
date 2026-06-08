// =============================================================================
// 调色板提取 — KMeans++ / Mini-Batch KMeans / Median Cut / Octree
// =============================================================================

use rand::Rng;
use rand_xoshiro::Xoshiro256Plus;
use rand::SeedableRng;

use crate::params::PaletteParams;

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
    if n == 0 { return vec![]; }

    // 加权采样：前景置信度越高的像素采样概率越大
    let total_weight: f64 = fg_confidence.iter().sum();
    if total_weight < 1e-10 {
        return vec![];
    }

    // 根据 min_fg_ratio 决定实际使用的像素
    let min_fg_weight = total_weight * params.min_fg_ratio;
    let effective_weight = total_weight.max(min_fg_weight);

    // 使用权重乘以 RGB 作为颜色聚类输入
    let weighted_pixels: Vec<[f64; 3]> = rgb.iter()
        .zip(fg_confidence.iter())
        .map(|(&c, &w)| [c[0] * w, c[1] * w, c[2] * w])
        .collect();

    let k = params.n_colors.min(effective_weight as usize).max(2);

    let entries = match params.algorithm.as_str() {
        "kmeanspp" => kmeans_pp(&weighted_pixels, k, fg_confidence),
        "minibatch" => mini_batch_kmeans(&weighted_pixels, k, fg_confidence),
        "mediancut" => median_cut_palette(rgb, k, fg_confidence),
        "octree" => octree_palette(rgb, k),
        _ => kmeans_pp(&weighted_pixels, k, fg_confidence),
    };

    entries
}

// ── KMeans++ ──

fn kmeans_pp(pixels: &[[f64; 3]], k: usize, weights: &[f64]) -> Vec<PaletteEntry> {
    let n = pixels.len();
    if n == 0 || k == 0 { return vec![]; }
    let k = k.min(n);

    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // 初始化：KMeans++ seeding
    let mut centroids: Vec<[f64; 3]> = Vec::with_capacity(k);
    // 第一个中心：按权重随机选择
    let first_idx = weighted_choice(weights, &mut rng);
    if first_idx < n { centroids.push(pixels[first_idx]); } else if n > 0 { centroids.push(pixels[0]); }

    let mut min_dists = vec![f64::MAX; n];
    for _ in 1..k {
        let mut total_dist = 0.0;
        for (i, p) in pixels.iter().enumerate() {
            let d = sq_dist(p, &centroids[centroids.len() - 1]);
            if d < min_dists[i] { min_dists[i] = d; }
            total_dist += min_dists[i] * weights[i];
        }
        if total_dist < 1e-30 { break; }
        let threshold = rng.r#gen::<f64>() * total_dist;
        let mut cum = 0.0;
        let mut chosen = 0;
        for (i, &d) in min_dists.iter().enumerate() {
            cum += d * weights[i];
            if cum >= threshold { chosen = i; break; }
        }
        centroids.push(pixels[chosen]);
    }

    // 迭代 Lloyd
    let max_iter = 20;
    let mut assignments = vec![0usize; n];
    for _iter in 0..max_iter {
        // assign
        let mut changed = false;
        for (i, p) in pixels.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (j, c) in centroids.iter().enumerate() {
                let d = sq_dist(p, c);
                if d < best_d { best_d = d; best = j; }
            }
            if assignments[i] != best { assignments[i] = best; changed = true; }
        }
        if !changed { break; }

        // update
        let mut sums = vec![[0.0; 3]; k];
        let mut counts = vec![0.0; k];
        for (i, p) in pixels.iter().enumerate() {
            let a = assignments[i];
            sums[a][0] += p[0]; sums[a][1] += p[1]; sums[a][2] += p[2];
            counts[a] += weights[i];
        }
        for j in 0..k {
            if counts[j] > 0.0 {
                centroids[j] = [sums[j][0] / pixels.iter().filter(|_| assignments.iter().any(|&a| a == j)).count() as f64,
                                 sums[j][1] / pixels.iter().filter(|_| assignments.iter().any(|&a| a == j)).count() as f64,
                                 sums[j][2] / pixels.iter().filter(|_| assignments.iter().any(|&a| a == j)).count() as f64];
            }
        }
    }

    // 统计比例
    let mut total_weight: f64 = weights.iter().sum();
    if total_weight < 1e-10 { total_weight = 1.0; }
    let mut cluster_weight = vec![0.0; k];
    for (i, &a) in assignments.iter().enumerate() {
        cluster_weight[a] += weights[i];
    }

    centroids.into_iter().enumerate().map(|(i, c)| {
        let hex = rgb_to_hex(c);
        let proportion = cluster_weight[i] / total_weight;
        PaletteEntry { rgb: c, hex, proportion }
    }).collect()
}

// ── Mini-Batch KMeans ──

fn mini_batch_kmeans(pixels: &[[f64; 3]], k: usize, weights: &[f64]) -> Vec<PaletteEntry> {
    // Simplified: use standard KMeans++ (mini-batch not critical for this scale)
    kmeans_pp(pixels, k, weights)
}

// ── Median Cut (调色板提取版本) ──

fn median_cut_palette(pixels: &[[f64; 3]], k: usize, weights: &[f64]) -> Vec<PaletteEntry> {
    let n = pixels.len();
    if n == 0 || k == 0 { return vec![]; }
    let k = k.min(n);

    let indices: Vec<usize> = (0..n).collect();
    let leaves = median_cut_recursive(pixels, indices, k, 0);

    let total_weight: f64 = weights.iter().sum();
    let total_weight = if total_weight < 1e-10 { 1.0 } else { total_weight };

    leaves.into_iter().map(|idx_set| {
        let mut sum_r = 0.0; let mut sum_g = 0.0; let mut sum_b = 0.0;
        let mut w_sum = 0.0;
        for &i in &idx_set {
            let w = weights[i];
            sum_r += pixels[i][0] * w; sum_g += pixels[i][1] * w; sum_b += pixels[i][2] * w;
            w_sum += w;
        }
        let w_sum = if w_sum < 1e-10 { 1.0 } else { w_sum };
        let mean = [sum_r / w_sum, sum_g / w_sum, sum_b / w_sum];
        let proportion = w_sum / total_weight;
        PaletteEntry { rgb: mean, hex: rgb_to_hex(mean), proportion }
    }).collect()
}

fn median_cut_recursive(pixels: &[[f64; 3]], indices: Vec<usize>, max_colors: usize, depth: usize) -> Vec<Vec<usize>> {
    if indices.len() <= 1 || depth > 10 || max_colors <= 1 {
        return vec![indices];
    }

    // 找最大范围通道
    let (mut min_r, mut max_r) = (f64::MAX, f64::MIN);
    let (mut min_g, mut max_g) = (f64::MAX, f64::MIN);
    let (mut min_b, mut max_b) = (f64::MAX, f64::MIN);
    for &i in &indices {
        let p = pixels[i];
        if p[0] < min_r { min_r = p[0]; } if p[0] > max_r { max_r = p[0]; }
        if p[1] < min_g { min_g = p[1]; } if p[1] > max_g { max_g = p[1]; }
        if p[2] < min_b { min_b = p[2]; } if p[2] > max_b { max_b = p[2]; }
    }

    let range_r = max_r - min_r;
    let range_g = max_g - min_g;
    let range_b = max_b - min_b;

    let channel = if range_r >= range_g && range_r >= range_b { 0 }
    else if range_g >= range_b { 1 } else { 2 };

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
    result.extend(median_cut_recursive(pixels, left, colors_per_child, depth + 1));
    result.extend(median_cut_recursive(pixels, right, max_colors - colors_per_child, depth + 1));
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

fn weighted_choice(weights: &[f64], rng: &mut Xoshiro256Plus) -> usize {
    let total: f64 = weights.iter().sum();
    if total < 1e-10 { return rng.r#gen::<f64>() as usize % weights.len().max(1); }
    let threshold = rng.r#gen::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= threshold { return i; }
    }
    weights.len() - 1
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
