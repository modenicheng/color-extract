use anyhow::{Context, Result};
use image::{GenericImageView, ImageBuffer, ImageReader, Rgb};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// 2D FFT helpers (row–column, in‑place)
// ---------------------------------------------------------------------------

fn fft2d_real(data: &mut [Complex<f64>], w: usize, h: usize, forward: bool) {
    let planner = FftPlanner::new();
    // rows
    let fft_row: Arc<dyn rustfft::Fft<f64>> = if forward {
        planner.plan_fft_forward(w)
    } else {
        planner.plan_fft_inverse(w)
    };
    for y in 0..h {
        let row = &mut data[y * w..(y + 1) * w];
        fft_row.process(row);
    }

    // columns
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

/// 2D mean filter (3×3, symmetric padding via border‑clamp)
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

/// 2D Gaussian blur (separable, σ = 3, kernel radius = 6)
fn gaussian_blur(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let sigma = 3.0;
    let r = 6usize;
    let mut kernel = Vec::with_capacity(2 * r + 1);
    let mut ksum = 0.0;
    for i in 0..=2 * r {
        let x = i as f64 - r as f64;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(v);
        ksum += v;
    }
    for k in &mut kernel { *k /= ksum; }

    // horizontal pass
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

    // vertical pass
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

// ---------------------------------------------------------------------------
// Spectral residual saliency
// ---------------------------------------------------------------------------

fn spectral_residual(gray: &[f64], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;

    // 1. FFT → log amplitude + phase
    let mut data: Vec<Complex<f64>> = gray.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2d_real(&mut data, w, h, true);

    let mut log_amp = vec![0.0; n];
    let mut phase = vec![0.0; n];
    for (i, &c) in data.iter().enumerate() {
        let mag = (c.norm_sqr() + 1e-20).sqrt();
        log_amp[i] = mag.ln();
        phase[i] = c.im.atan2(c.re);
    }

    // 2. Averaged log amplitude (3×3 mean filter)
    let avg_log_amp = mean_filter_3x3(&log_amp, w, h);

    // 3. Spectral residual: R = log_amp - avg_log_amp
    let mut residual = vec![0.0; n];
    for i in 0..n {
        residual[i] = log_amp[i] - avg_log_amp[i];
    }

    // 4. Reconstruct: F' = exp(R + i·P)
    let mut recon: Vec<Complex<f64>> = residual.iter().zip(phase.iter())
        .map(|(&r, &p)| {
            let mag = r.exp();
            Complex::new(mag * p.cos(), mag * p.sin())
        })
        .collect();

    // 5. IFFT → spatial saliency map
    fft2d_real(&mut recon, w, h, false);
    let norm = (w * h) as f64;
    let mut saliency: Vec<f64> = recon.iter().map(|c| c.re / norm).collect();

    // 6. Gaussian blur (optional but reduces noise)
    saliency = gaussian_blur(&saliency, w, h);

    // 7. Normalize to [0, 1]
    let min = saliency.iter().cloned().fold(f64::MAX, f64::min);
    let max = saliency.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-12);
    for v in &mut saliency { *v = (*v - min) / range; }

    saliency
}

// ---------------------------------------------------------------------------
// Colormap (same heatmap as dct-viz for consistency)
// ---------------------------------------------------------------------------

fn heatmap(v: f64) -> Rgb<u8> {
    let v = v.clamp(0.0, 1.0);
    let (r, g, b) = if v < 0.25 {
        let t = v / 0.25;
        (0.0, t * 4.0, 1.0)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - t * 4.0)
    } else if v < 0.75 {
        let t = (v - 0.50) / 0.25;
        (t * 4.0, 1.0, 0.0)
    } else {
        let t = (v - 0.75) / 0.25;
        (1.0, 1.0 - t * 4.0, 0.0)
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fit_dim(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w <= max_dim && h <= max_dim { return (w, h); }
    let s = max_dim as f64 / w.max(h) as f64;
    ((w as f64 * s) as u32, (h as f64 * s) as u32).max((1, 1))
}

fn to_gray(pixels: &[[f64; 3]]) -> Vec<f64> {
    pixels.iter().map(|&[r, g, b]| 0.299 * r + 0.587 * g + 0.114 * b).collect()
}

fn load_resize_gray(path: &Path) -> Result<(String, u32, u32, Vec<f64>)> {
    let img = ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .decode()
        .with_context(|| format!("decode {}", path.display()))?;

    let (w, h) = img.dimensions();
    let (nw, nh) = fit_dim(w, h, 1024);
    let resized = if nw != w || nh != h {
        img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let rgb = resized.to_rgb8();
    let (fw, fh) = rgb.dimensions();
    let pixels: Vec<[f64; 3]> = rgb.pixels().map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0]).collect();
    let gray = to_gray(&pixels);

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("img").to_string();
    Ok((stem, fw, fh, gray))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let out_dir = Path::new("output/spectral_residual");
    std::fs::create_dir_all(out_dir)?;

    let img_dir = Path::new("imgs");
    let mut entries: Vec<_> = std::fs::read_dir(img_dir)
        .context("reading imgs/")?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in &entries {
        let path = entry.path();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if ext != "jpg" && ext != "jpeg" && ext != "png" { continue; }

        let (stem, w, h, gray) = load_resize_gray(&path)?;
        println!("{} {}×{}", stem, w, h);

        let saliency = spectral_residual(&gray, w as usize, h as usize);

        // --- 热力图 (同 dct-viz 风格) ---
        let img_heat: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
            let i = (y * w + x) as usize;
            heatmap(saliency[i])
        });
        img_heat.save(out_dir.join(format!("{}_sr_heat.png", stem)))?;

        // --- 原图灰度叠加: 显著区域红色高亮 ---
        let img_overlay: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
            let i = (y * w + x) as usize;
            let c = saliency[i];
            let base = gray[i] * 255.0;
            let r = (base * (1.0 - c) + 255.0 * c) as u8;
            let g = (base * (1.0 - c) + 40.0 * c) as u8;
            let b = (base * (1.0 - c) + 40.0 * c) as u8;
            Rgb([r, g, b])
        });
        img_overlay.save(out_dir.join(format!("{}_sr_overlay.png", stem)))?;
    }

    println!("Done. Output in {}/", out_dir.display());
    Ok(())
}
