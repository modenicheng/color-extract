// =============================================================================
// Global / Local Residual (全局均值残差 & Gaussian 局部残差)
// =============================================================================

use image::{GrayImage, ImageBuffer, Luma};

// ── Global Residual ──

/// 计算 HSL lightness 全局残差: |pixel - baseline|, baseline = mean 或 median
pub fn compute_global_light_residual(hsl_l: &[f64], baseline: &str) -> Vec<f64> {
    let center = center_value(hsl_l, baseline);
    hsl_l.iter().map(|&v| (v - center).abs()).collect()
}

/// 计算 HSL saturation 全局残差: |pixel - baseline|, baseline = mean 或 median
pub fn compute_global_sat_residual(hsl_s: &[f64], baseline: &str) -> Vec<f64> {
    let center = center_value(hsl_s, baseline);
    hsl_s.iter().map(|&v| (v - center).abs()).collect()
}

/// 根据 baseline 策略计算全局中心值
fn center_value(data: &[f64], baseline: &str) -> f64 {
    let n = data.len();
    match baseline {
        "mean" => data.iter().sum::<f64>() / n as f64,
        "median" => {
            let mut sorted = data.to_vec();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[n / 2]
        }
        _ => panic!("global_residual.baseline must be 'mean' or 'median', got '{baseline}'"),
    }
}

// ── Local (Gaussian) Residual ──

/// 使用 image crate 的 Gaussian blur 计算残差: |original - blurred|
fn compute_gaussian_residual(ch: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    // 将 f64 [0,1] 转为 GrayImage
    let src_img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let v = (ch[i].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });

    let blurred = image::imageops::blur(&src_img, sigma);

    let n = (w * h) as usize;
    let mut residual = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let orig = src_img.get_pixel(x, y)[0] as f64;
            let blr = blurred.get_pixel(x, y)[0] as f64;
            residual.push((orig - blr).abs() / 255.0);
        }
    }
    residual
}

pub fn compute_local_light_residual(hsl_l: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    compute_gaussian_residual(hsl_l, w, h, sigma)
}

pub fn compute_local_sat_residual(hsl_s: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    compute_gaussian_residual(hsl_s, w, h, sigma)
}
