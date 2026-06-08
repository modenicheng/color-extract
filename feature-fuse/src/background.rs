// =============================================================================
// Background Estimation: LAB residual + connectedness mask
// =============================================================================

use std::collections::VecDeque;

use crate::params::BackgroundParams;

// ── helpers ──

/// 百分位值（线性插值）
fn percentile_value(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let p = p.clamp(0.0, 100.0);
    let idx = p / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (idx.ceil() as usize).min(n - 1);
    let frac = idx - lo as f64;
    if lo == hi { sorted[lo] } else { sorted[lo] * (1.0 - frac) + sorted[hi] * frac }
}

/// Percentile trimmed mean + median 混合的稳健中心估计
fn robust_center_percentile_trimmed(
    data: &[f64], p_low: f64, p_high: f64,
    tm_weight: f64, med_weight: f64,
) -> f64 {
    let n = data.len();
    if n == 0 { return 0.0; }

    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let lo_val = percentile_value(&sorted, p_low);
    let hi_val = percentile_value(&sorted, p_high);

    // Trimmed mean (clip to [lo_val, hi_val])
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in data {
        if v >= lo_val && v <= hi_val {
            sum += v;
            count += 1;
        }
    }
    let trimmed_mean = if count > 0 { sum / count as f64 } else { sorted[n / 2] };

    let median = sorted[n / 2];

    tm_weight * trimmed_mean + med_weight * median
}

/// 从图像四条边界采样像素 band
fn sample_border_pixels(
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    w: u32, h: u32, band: u32,
) -> Vec<(f64, f64, f64)> {
    let band = band.max(1);
    let mut pixels = Vec::new();

    // Top
    for y in 0..band.min(h) {
        for x in 0..w {
            let i = (y * w + x) as usize;
            pixels.push((lab_l[i], lab_a[i], lab_b[i]));
        }
    }
    // Bottom
    for y in (h.saturating_sub(band))..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            pixels.push((lab_l[i], lab_a[i], lab_b[i]));
        }
    }
    // Left (exclude corners already sampled)
    for y in band..h.saturating_sub(band) {
        for x in 0..band.min(w) {
            let i = (y * w + x) as usize;
            pixels.push((lab_l[i], lab_a[i], lab_b[i]));
        }
    }
    // Right (exclude corners already sampled)
    for y in band..h.saturating_sub(band) {
        for x in (w.saturating_sub(band))..w {
            let i = (y * w + x) as usize;
            pixels.push((lab_l[i], lab_a[i], lab_b[i]));
        }
    }
    pixels
}

// ── public API ──

/// 计算每个像素到边界背景 LAB 中心（稳健估计）的加权欧氏距离
///
/// 距离公式:
///   d = sqrt( ((L-bg_L)/100)^2 + ((a-bg_a)/128)^2 + ((b-bg_b)/128)^2 )
///
/// 返回值是未归一化的原始距离（后续由 percentile_normalize 处理到 [0,1]）。
pub fn compute_background_lab_residual(
    lab_l: &[f64], lab_a: &[f64], lab_b: &[f64],
    w: u32, h: u32,
    params: &BackgroundParams,
) -> Vec<f64> {
    let n = (w * h) as usize;
    if n == 0 { return vec![]; }

    // sample border band
    let border = sample_border_pixels(lab_l, lab_a, lab_b, w, h, params.border_band);
    if border.is_empty() {
        return vec![0.0; n];
    }

    // 提取各通道
    let b_l: Vec<f64> = border.iter().map(|&(l, _, _)| l).collect();
    let b_a: Vec<f64> = border.iter().map(|&(_, a, _)| a).collect();
    let b_b: Vec<f64> = border.iter().map(|&(_, _, b)| b).collect();

    // 稳健中心
    let bg_l = robust_center_percentile_trimmed(
        &b_l, params.trim_low, params.trim_high,
        params.trimmed_mean_weight, params.median_weight,
    );
    let bg_a = robust_center_percentile_trimmed(
        &b_a, params.trim_low, params.trim_high,
        params.trimmed_mean_weight, params.median_weight,
    );
    let bg_b = robust_center_percentile_trimmed(
        &b_b, params.trim_low, params.trim_high,
        params.trimmed_mean_weight, params.median_weight,
    );

    // 逐像素加权距离
    let mut dist = Vec::with_capacity(n);
    for i in 0..n {
        let d_l = (lab_l[i] - bg_l) / 100.0;
        let d_a = (lab_a[i] - bg_a) / 128.0;
        let d_b = (lab_b[i] - bg_b) / 128.0;
        dist.push((d_l * d_l + d_a * d_a + d_b * d_b).sqrt());
    }
    dist
}

/// BFS 从边界连通区域检测背景 mask
///
/// 利用 raw_dist (compute_background_lab_residual 的原始输出)，
/// 以边界 band 像素为种子做 4-neighbor flood fill。
/// 连通条件: raw_dist ≤ bg_max_dist × dist_threshold_factor。
/// 返回 [0,1] mask，1.0 = 背景（边界连通区域）。
pub fn compute_background_connected_mask(
    raw_dist: &[f64],
    w: u32, h: u32,
    params: &BackgroundParams,
) -> Vec<f64> {
    let n = (w * h) as usize;
    if n == 0 { return vec![]; }

    let band = params.border_band.max(1);
    let conn = &params.connectedness;

    // 1. 计算边界像素在 raw_dist 中的最大值作为阈值基
    let mut border_dists = Vec::new();
    let mut seed_queue: VecDeque<(u32, u32)> = VecDeque::new();
    let mut visited = vec![false; n];
    let mut mask = vec![0.0; n];

    // 遍历所有边界 band 像素：标记为 visited + 设置 mask=1.0 + 若 dist≤threshold 则入队
    // --- Top ---
    for y in 0..band.min(h) {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = raw_dist[i];
            border_dists.push(d);
            visited[i] = true;
            mask[i] = 1.0;
            // 暂不入队，threshold 还没算出来
        }
    }
    // --- Bottom ---
    for y in (h.saturating_sub(band))..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let d = raw_dist[i];
            if !visited[i] {
                border_dists.push(d);
                visited[i] = true;
                mask[i] = 1.0;
            }
        }
    }
    // --- Left (non-corner) ---
    for y in band..h.saturating_sub(band) {
        for x in 0..band.min(w) {
            let i = (y * w + x) as usize;
            let d = raw_dist[i];
            if !visited[i] {
                border_dists.push(d);
                visited[i] = true;
                mask[i] = 1.0;
            }
        }
    }
    // --- Right (non-corner) ---
    for y in band..h.saturating_sub(band) {
        for x in (w.saturating_sub(band))..w {
            let i = (y * w + x) as usize;
            let d = raw_dist[i];
            if !visited[i] {
                border_dists.push(d);
                visited[i] = true;
                mask[i] = 1.0;
            }
        }
    }

    // 2. 计算阈值
    let bg_max = border_dists.iter().cloned().fold(0.0f64, f64::max);
    let threshold = if bg_max.is_finite() && bg_max >= 0.0 {
        bg_max * conn.dist_threshold_factor
    } else {
        0.01
    };

    // 3. 重新扫描一遍边界，将 dist≤threshold 的像素入队作为 BFS 起点
    let enqueue_border = |queue: &mut VecDeque<(u32, u32)>, dist: &[f64], ww: u32, hh: u32, b: u32| {
        // Top
        for y in 0..b.min(hh) {
            for x in 0..ww {
                if dist[(y * ww + x) as usize] <= threshold {
                    queue.push_back((x, y));
                }
            }
        }
        // Bottom
        for y in (hh.saturating_sub(b))..hh {
            for x in 0..ww {
                let i = (y * ww + x) as usize;
                if !(y < b) && dist[i] <= threshold {
                    queue.push_back((x, y));
                }
            }
        }
        // Left
        for y in b..hh.saturating_sub(b) {
            for x in 0..b.min(ww) {
                if dist[(y * ww + x) as usize] <= threshold {
                    queue.push_back((x, y));
                }
            }
            for x in (ww.saturating_sub(b))..ww {
                if dist[(y * ww + x) as usize] <= threshold {
                    queue.push_back((x, y));
                }
            }
        }
    };
    enqueue_border(&mut seed_queue, raw_dist, w, h, band);

    // 4. 4-neighbor BFS
    while let Some((cx, cy)) = seed_queue.pop_front() {
        macro_rules! try_neighbor {
            ($nx:expr, $ny:expr) => {
                let ni = ($ny * w + $nx) as usize;
                if !visited[ni] && raw_dist[ni] <= threshold {
                    visited[ni] = true;
                    mask[ni] = 1.0;
                    seed_queue.push_back(($nx, $ny));
                }
            };
        }
        if cy > 0 { try_neighbor!(cx, cy - 1); }
        if cy + 1 < h { try_neighbor!(cx, cy + 1); }
        if cx > 0 { try_neighbor!(cx - 1, cy); }
        if cx + 1 < w { try_neighbor!(cx + 1, cy); }
    }

    // 5. 可选 blur
    if conn.blur_sigma > 0.0 {
        mask = blur_mask(&mask, w, h, conn.blur_sigma);
    }

    mask
}

/// 对 mask 做 Gaussian blur 以软化硬边缘
fn blur_mask(mask: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    use image::{GrayImage, ImageBuffer, Luma};

    let img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
        let v = (mask[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });

    let blurred = image::imageops::blur(&img, sigma);

    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            out.push(blurred.get_pixel(x, y)[0] as f64 / 255.0);
        }
    }
    out
}

/// 从背景 mask 生成前景置信度: 1.0 - mask × strength
///
/// 结果在 [0,1] 区间，越亮表示越不像边界连通背景。
/// 此输出可直接作为特征图参与融合。
pub fn mask_to_foreground_confidence(mask: &[f64], strength: f64) -> Vec<f64> {
    mask.iter().map(|&m| (1.0 - m * strength).clamp(0.0, 1.0)).collect()
}
