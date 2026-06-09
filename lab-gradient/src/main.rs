use anyhow::{Context, Result};
use image::{GenericImageView, ImageBuffer, ImageReader, Rgb};
use palette::{IntoColor, Lab, Srgb};
use std::path::Path;

/// Fit dimensions keeping aspect ratio, longest side ≤ max_dim.
fn fit_dim(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (w, h);
    }
    let s = max_dim as f64 / w.max(h) as f64;
    ((w as f64 * s) as u32, (h as f64 * s) as u32).max((1, 1))
}

/// Load an image, resize, return (filename_stem, width, height, LAB channels).
fn load_resize_lab(path: &Path) -> Result<(String, u32, u32, Vec<f64>, Vec<f64>, Vec<f64>)> {
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
    let n = (fw * fh) as usize;
    let mut l = Vec::with_capacity(n);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);

    for p in rgb.pixels() {
        let srgb = Srgb::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        );
        let lab: Lab = srgb.into_color();
        l.push(lab.l as f64);
        a.push(lab.a as f64);
        b.push(lab.b as f64);
    }

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("img")
        .to_string();
    Ok((stem, fw, fh, l, a, b))
}

/// Sobel gradient magnitude for a single channel (2D grid w×h).
fn sobel_magnitude(ch: &[f64], w: u32, h: u32) -> Vec<f64> {
    let n = (w * h) as usize;
    let mut mag = vec![0.0; n];
    let wu = w as usize;

    // Sobel kernels – skip border pixels (use clamp-to-0 or just skip)
    // Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    // Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
    // divisor: sum(|kernel|) = 8 for normalization to approx derivative per pixel
    for y in 1..(h as usize - 1) {
        for x in 1..(wu - 1) {
            let i = y * wu + x;
            let gx = -1.0 * ch[i - wu - 1] + 1.0 * ch[i - wu + 1] - 2.0 * ch[i - 1]
                + 2.0 * ch[i + 1]
                - 1.0 * ch[i + wu - 1]
                + 1.0 * ch[i + wu + 1];
            let gy = -1.0 * ch[i - wu - 1] - 2.0 * ch[i - wu] - 1.0 * ch[i - wu + 1]
                + 1.0 * ch[i + wu - 1]
                + 2.0 * ch[i + wu]
                + 1.0 * ch[i + wu + 1];
            // Normalize by 8 to get approx per-pixel delta, then magnitude
            mag[i] = ((gx * gx + gy * gy).sqrt()) / 8.0;
        }
    }
    mag
}

/// Map three channels (mag_r, mag_g, mag_b) to R/G/B, normalize each independently.
fn to_rgb_image(
    ch_r: &[f64],
    ch_g: &[f64],
    ch_b: &[f64],
    w: u32,
    h: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let norm = |ch: &[f64]| -> Vec<u8> {
        let min = ch.iter().cloned().fold(f64::MAX, f64::min);
        let max = ch.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-12);
        ch.iter()
            .map(|&v| ((v - min) / range * 255.0) as u8)
            .collect()
    };
    let r = norm(ch_r);
    let g = norm(ch_g);
    let b = norm(ch_b);
    ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        Rgb([r[i], g[i], b[i]])
    })
}

fn main() -> Result<()> {
    let out_dir = Path::new("output/lab_gradient");
    std::fs::create_dir_all(out_dir)?;

    let img_dir = Path::new("imgs");
    let mut entries: Vec<_> = std::fs::read_dir(img_dir)
        .context("reading imgs/")?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in &entries {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext != "jpg" && ext != "jpeg" && ext != "png" {
            continue;
        }

        let (stem, w, h, l, a, b) = load_resize_lab(&path)?;
        println!("{} {}×{}", stem, w, h);

        let mag_l = sobel_magnitude(&l, w, h);
        let mag_a = sobel_magnitude(&a, w, h);
        let mag_b = sobel_magnitude(&b, w, h);
        let img = to_rgb_image(&mag_l, &mag_a, &mag_b, w, h);
        let out_path = out_dir.join(format!("{}_grad.png", stem));
        img.save(&out_path)?;
    }

    println!("Done. Output in {}/", out_dir.display());
    Ok(())
}
