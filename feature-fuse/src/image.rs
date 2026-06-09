// =============================================================================
// 图片加载 + Colorspace 转换
// =============================================================================

use anyhow::{Context, Result};
use image::{GenericImageView, ImageReader};
use palette::{Hsl, IntoColor, Lab, Srgb};
use std::path::Path;

/// 加载图片，统一 resize，返回所有需要的通道数据。
pub struct ImageData {
    pub stem: String,
    pub w: u32,
    pub h: u32,
    /// RGB 像素，每个分量 [0, 1]
    pub rgb: Vec<[f64; 3]>,
    /// CIELAB L* [0, 100]
    pub lab_l: Vec<f64>,
    /// CIELAB a*
    pub lab_a: Vec<f64>,
    /// CIELAB b*
    pub lab_b: Vec<f64>,
    /// HSL saturation [0, 1]
    #[allow(dead_code)]
    pub hsl_s: Vec<f64>,
    /// HSL lightness [0, 1]
    pub hsl_l: Vec<f64>,
}

pub fn load_image(path: &Path, max_dim: u32) -> Result<ImageData> {
    let img = ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .decode()
        .with_context(|| format!("decode {}", path.display()))?;

    let (w, h) = img.dimensions();
    let (nw, nh) = fit_dimensions(w, h, max_dim);
    let resized = if nw != w || nh != h {
        img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let (fw, fh) = resized.dimensions();
    let rgb = resized.to_rgb8();

    let n = (fw * fh) as usize;
    let mut out_rgb = Vec::with_capacity(n);
    let mut lab_l = Vec::with_capacity(n);
    let mut lab_a = Vec::with_capacity(n);
    let mut lab_b = Vec::with_capacity(n);
    let mut hsl_s = Vec::with_capacity(n);
    let mut hsl_l = Vec::with_capacity(n);

    for p in rgb.pixels() {
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        let srgb = Srgb::new(r, g, b);
        let lab: Lab = srgb.into_color();
        let hsl: Hsl = srgb.into_color();
        out_rgb.push([r as f64, g as f64, b as f64]);
        lab_l.push(lab.l as f64);
        lab_a.push(lab.a as f64);
        lab_b.push(lab.b as f64);
        hsl_s.push(hsl.saturation as f64);
        hsl_l.push(hsl.lightness as f64);
    }

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("img")
        .to_string();

    Ok(ImageData {
        stem,
        w: fw,
        h: fh,
        rgb: out_rgb,
        lab_l,
        lab_a,
        lab_b,
        hsl_s,
        hsl_l,
    })
}

fn fit_dimensions(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (w, h);
    }
    let scale = max_dim as f64 / w.max(h) as f64;
    let nw = (w as f64 * scale) as u32;
    let nh = (h as f64 * scale) as u32;
    (nw.max(1), nh.max(1))
}
