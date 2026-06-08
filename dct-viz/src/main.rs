use anyhow::{Context, Result};
use image::{GenericImageView, GrayImage, ImageBuffer, ImageReader, Rgb};
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::path::Path;

// ---------------------------------------------------------------------------
// DCT constants
// ---------------------------------------------------------------------------
const N: usize = 8;
const THRESHOLD: usize = 4;
const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// DCT matrix helpers (Type-II)
// ---------------------------------------------------------------------------

fn dct_matrix() -> [[f64; N]; N] {
    let mut t = [[0.0; N]; N];
    let inv_sqrt_n = 1.0 / (N as f64).sqrt();
    let sqrt_2_over_n = (2.0 / N as f64).sqrt();
    for i in 0..N {
        let alpha = if i == 0 { inv_sqrt_n } else { sqrt_2_over_n };
        for j in 0..N {
            t[i][j] = alpha * ((2.0 * j as f64 + 1.0) * i as f64 * PI / (2.0 * N as f64)).cos();
        }
    }
    t
}

fn transpose(m: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N { for j in 0..N { out[j][i] = m[i][j]; } }
    out
}

fn dct_2d(block: &[[f64; N]; N], t: &[[f64; N]; N]) -> [[f64; N]; N] {
    let tt = transpose(t);
    let mut rows_dct = [[0.0; N]; N];
    for r in 0..N { for c in 0..N { for k in 0..N { rows_dct[r][c] += block[r][k] * tt[k][c]; } } }
    let mut out = [[0.0; N]; N];
    for r in 0..N { for c in 0..N { for k in 0..N { out[r][c] += t[r][k] * rows_dct[k][c]; } } }
    out
}

fn high_freq_ratio(coeffs: &[[f64; N]; N]) -> f64 {
    let mut total_ac = 0.0;
    let mut high_freq = 0.0;
    for u in 0..N { for v in 0..N {
        if u == 0 && v == 0 { continue; }
        let e = coeffs[u][v] * coeffs[u][v];
        total_ac += e;
        if u + v >= THRESHOLD { high_freq += e; }
    }}
    high_freq / (total_ac + 1e-10)
}

// ---------------------------------------------------------------------------
// Complexity map
// ---------------------------------------------------------------------------

fn complexity_map(gray: &[f64], w: usize, h: usize) -> Vec<f64> {
    let offset = (N / 2) as i32;
    let t = dct_matrix();
    let mut map = vec![0.0; w * h];

    map.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut block = [[0.0; N]; N];
            for dy in 0..N { for dx in 0..N {
                let px = (x as i32 + dx as i32 - offset).clamp(0, w as i32 - 1) as usize;
                let py = (y as i32 + dy as i32 - offset).clamp(0, h as i32 - 1) as usize;
                block[dy][dx] = gray[py * w + px];
            }}
            let coeffs = dct_2d(&block, &t);
            row[x] = high_freq_ratio(&coeffs);
        }
    });
    map
}

// ---------------------------------------------------------------------------
// Colormap: value in [0,1] → RGB heatmap (dark blue → cyan → yellow → red)
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

fn main() -> Result<()> {
    let out_dir = Path::new("output/dct_viz");
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

        let map = complexity_map(&gray, w as usize, h as usize);

        // --- 彩色热力图 ---
        let img_heat: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
            let i = (y * w + x) as usize;
            heatmap(map[i])
        });
        img_heat.save(out_dir.join(format!("{}_dct_heat.png", stem)))?;

        // --- 纯灰度图: 复杂度值直接映射到亮度, 无分段/overlay ---
        let img_gray: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
            let i = (y * w + x) as usize;
            let v = (map[i] * 255.0).clamp(0.0, 255.0) as u8;
            image::Luma([v])
        });
        img_gray.save(out_dir.join(format!("{}_dct_gray.png", stem)))?;

        // --- 灰度加权图: 把复杂度叠加到原图亮度上 ---
        let img_light: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
            let i = (y * w + x) as usize;
            let c = map[i];
            // 平滑区(c≈0)显示原图灰度, 纹理区(c≈1)显示为亮绿/青色高亮
            let base = gray[i] * 255.0;
            let r = (base * (1.0 - c) + 60.0 * c) as u8;
            let g = (base * (1.0 - c) + 220.0 * c) as u8;
            let b = (base * (1.0 - c) + 180.0 * c) as u8;
            Rgb([r, g, b])
        });
        img_light.save(out_dir.join(format!("{}_dct_overlay.png", stem)))?;
    }

    println!("Done. Output in {}/", out_dir.display());
    Ok(())
}
