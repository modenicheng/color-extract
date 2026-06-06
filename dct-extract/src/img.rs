use anyhow::{Context, Result};
use base64::Engine;
use image::{DynamicImage, GenericImageView, ImageReader};
use std::path::Path;

pub struct LoadedImage {
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f64; 3]>, // normalized RGB 0..1, row-major
    pub base64_preview: String,
}

/// Load all JPEG/PNG images from a directory, resize to fit max_dim.
pub fn load_all(img_dir: &str, max_dim: u32) -> Result<Vec<LoadedImage>> {
    let dir = Path::new(img_dir);
    if !dir.is_dir() {
        anyhow::bail!("'{}' is not a directory", img_dir);
    }

    let mut images = Vec::new();
    for entry in std::fs::read_dir(dir).context("reading img directory")? {
        let entry = entry?;
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext != "jpg" && ext != "jpeg" && ext != "png" {
            continue;
        }

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let img = ImageReader::open(&path)
            .with_context(|| format!("opening {}", filename))?
            .decode()
            .with_context(|| format!("decoding {}", filename))?;

        let (w, h) = img.dimensions();
        let (new_w, new_h) = fit_dimensions(w, h, max_dim);
        let resized = if new_w != w || new_h != h {
            img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };

        let rgb = resized.to_rgb8();
        let (final_w, final_h) = rgb.dimensions();

        let pixels: Vec<[f64; 3]> = rgb
            .pixels()
            .map(|p| {
                [
                    p[0] as f64 / 255.0,
                    p[1] as f64 / 255.0,
                    p[2] as f64 / 255.0,
                ]
            })
            .collect();

        let thumb = resize_for_preview(&resized, 200);
        let base64_preview = image_to_base64_jpeg(&thumb)?;

        images.push(LoadedImage {
            filename,
            width: final_w,
            height: final_h,
            pixels,
            base64_preview,
        });
    }

    if images.is_empty() {
        anyhow::bail!("no JPEG/PNG images found in '{}'", img_dir);
    }

    Ok(images)
}

fn fit_dimensions(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (w, h);
    }
    let scale = max_dim as f64 / w.max(h) as f64;
    ((w as f64 * scale) as u32, (h as f64 * scale) as u32).max((1, 1))
}

fn resize_for_preview(img: &DynamicImage, max_dim: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let (new_w, new_h) = fit_dimensions(w, h, max_dim);
    if new_w != w || new_h != h {
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    }
}

fn image_to_base64_jpeg(img: &DynamicImage) -> Result<String> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg)
        .context("encoding thumbnail to JPEG")?;
    let bytes = buf.into_inner();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/jpeg;base64,{}", b64))
}
