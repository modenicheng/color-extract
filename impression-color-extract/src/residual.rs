// =============================================================================
// Local (Gaussian) 残差
// =============================================================================

use image::{GrayImage, ImageBuffer, Luma};

fn compute_gaussian_residual(ch: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    let src_img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
        let v = (ch[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0) as u8;
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
