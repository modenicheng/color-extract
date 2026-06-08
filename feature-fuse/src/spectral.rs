// =============================================================================
// 频谱残差显著性检测 (复用 spectral-residual 算法)
// =============================================================================

use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

fn fft2d_real(data: &mut [Complex<f64>], w: usize, h: usize, forward: bool) {
    let mut planner = FftPlanner::new();
    let fft_row: Arc<dyn rustfft::Fft<f64>> = if forward {
        planner.plan_fft_forward(w)
    } else {
        planner.plan_fft_inverse(w)
    };
    for y in 0..h {
        fft_row.process(&mut data[y * w..(y + 1) * w]);
    }
    let fft_col: Arc<dyn rustfft::Fft<f64>> = if forward {
        planner.plan_fft_forward(h)
    } else {
        planner.plan_fft_inverse(h)
    };
    let mut col = vec![Complex::new(0.0, 0.0); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = data[y * w + x];
        }
        fft_col.process(&mut col);
        for y in 0..h {
            data[y * w + x] = col[y];
        }
    }
}

fn mean_filter_3x3(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut cnt = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let px = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let py = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    sum += src[py * w + px];
                    cnt += 1;
                }
            }
            out[y * w + x] = sum / cnt as f64;
        }
    }
    out
}

fn gaussian_blur_1d(src: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let r = (sigma * 2.0).round() as usize;
    let r = r.max(1).min(20);
    let mut kernel = Vec::with_capacity(2 * r + 1);
    let mut ksum = 0.0;
    for i in 0..=2 * r {
        let x = i as f64 - r as f64;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(v);
        ksum += v;
    }
    for k in &mut kernel {
        *k /= ksum;
    }

    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for kx in 0..=2 * r {
                let sx = (x as i32 + kx as i32 - r as i32).clamp(0, w as i32 - 1) as usize;
                s += src[y * w + sx] * kernel[kx];
            }
            tmp[y * w + x] = s;
        }
    }

    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for ky in 0..=2 * r {
                let sy = (y as i32 + ky as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
                s += tmp[sy * w + x] * kernel[ky];
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// 对单通道计算频谱残差显著性，返回 [0,1] 归一化图
fn spectral_residual_single(ch: &[f64], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;

    // 1. FFT
    let mut data: Vec<Complex<f64>> = ch.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2d_real(&mut data, w, h, true);

    let mut log_amp = vec![0.0; n];
    let mut phase = vec![0.0; n];
    for (i, &c) in data.iter().enumerate() {
        let mag = (c.norm_sqr() + 1e-20).sqrt();
        log_amp[i] = mag.ln();
        phase[i] = c.im.atan2(c.re);
    }

    // 2. 平均 log 幅度谱 (3×3 mean filter)
    let avg_log_amp = mean_filter_3x3(&log_amp, w, h);

    // 3. 频谱残差: R = log_amp - avg_log_amp
    let mut residual = vec![0.0; n];
    for i in 0..n {
        residual[i] = log_amp[i] - avg_log_amp[i];
    }

    // 4. 重构: F' = exp(R + i·P)
    let mut recon: Vec<Complex<f64>> = residual
        .iter()
        .zip(phase.iter())
        .map(|(&r, &p)| {
            let mag = r.exp();
            Complex::new(mag * p.cos(), mag * p.sin())
        })
        .collect();

    // 5. IFFT
    fft2d_real(&mut recon, w, h, false);
    let norm = n as f64;
    let mut saliency: Vec<f64> = recon.iter().map(|c| c.re / norm).collect();

    // 6. Gaussian blur 去噪
    saliency = gaussian_blur_1d(&saliency, w, h, 3.0);

    // 7. 归一化到 [0, 1]
    let smin = saliency.iter().cloned().fold(f64::MAX, f64::min);
    let smax = saliency.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let srange = (smax - smin).max(1e-12);
    for v in &mut saliency {
        *v = (*v - smin) / srange;
    }
    saliency
}

/// 计算频谱残差显著性：LAB 三通道分别计算后 L₂ 融合
pub fn compute_spectral_residual(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: usize,
    h: usize,
) -> Vec<f64> {
    let sal_l = spectral_residual_single(lab_l, w, h);
    let sal_a = spectral_residual_single(lab_a, w, h);
    let sal_b = spectral_residual_single(lab_b, w, h);

    let n = w * h;
    let mut fused = Vec::with_capacity(n);
    for i in 0..n {
        fused.push((sal_l[i] * sal_l[i] + sal_a[i] * sal_a[i] + sal_b[i] * sal_b[i]).sqrt());
    }
    // L₂ 融合后再归一化
    let smin = fused.iter().cloned().fold(f64::MAX, f64::min);
    let smax = fused.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let srange = (smax - smin).max(1e-12);
    for v in &mut fused {
        *v = (*v - smin) / srange;
    }
    fused
}
