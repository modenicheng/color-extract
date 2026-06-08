// =============================================================================
// Global / Local Residual (稳健亮度中心残差 & Gaussian 局部残差)
// =============================================================================

use image::{GrayImage, ImageBuffer, Luma};

use crate::params::RobustCenterParams;

// ── Global Residual ──

/// 计算 HSL lightness 全局残差: |pixel - robust_center| (稳健亮度中心)
pub fn compute_global_light_residual(
    hsl_l: &[f64],
    rcp: &RobustCenterParams,
) -> Vec<f64> {
    let center = robust_center(hsl_l, rcp);
    hsl_l.iter().map(|&v| (v - center).abs()).collect()
}

/// 计算 HSL saturation 全局残差: |pixel - robust_center| (稳健亮度中心)
pub fn compute_global_sat_residual(
    hsl_s: &[f64],
    rcp: &RobustCenterParams,
) -> Vec<f64> {
    let center = robust_center(hsl_s, rcp);
    hsl_s.iter().map(|&v| (v - center).abs()).collect()
}

/// 稳健视觉亮度/饱和度中心估计
///
/// 流程:
///   1. 感知压缩 (gamma 或 log)
///   2. 在压缩域计算 p{trim_low} 和 p{trim_high}
///   3. clip 到 [p_low, p_high] 后计算 trimmed_mean
///   4. 混合: trimmed_mean_weight × trimmed_mean + median_weight × median
///   5. 从压缩域还原回原始亮度域 [0, 1]
fn robust_center(data: &[f64], rcp: &RobustCenterParams) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let eps = 1e-10_f64;

    // 1. 感知压缩到 [0, 1] 域
    let compressed: Vec<f64> = match rcp.compression.as_str() {
        "gamma" => {
            let p = rcp.gamma_power;
            data.iter().map(|&v| v.powf(p)).collect()
        }
        "log" => {
            let base = rcp.log_base;
            let ln_base = base.ln();
            let denom = (eps + 1.0_f64).ln() / ln_base; // = log_base(1+eps)
            data.iter().map(|&v| (eps + v).ln() / ln_base / denom).collect()
        }
        _ => panic!(
            "global_residual.compression must be 'gamma' or 'log', got '{}'",
            rcp.compression
        ),
    };

    // 2. 排序 (用于百分位 + median)
    let mut sorted = compressed.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // 3. 百分位截断
    let p_low = percentile_value(&sorted, rcp.trim_low);
    let p_high = percentile_value(&sorted, rcp.trim_high);

    // 4. Trimmed mean (剔除 >= p_high 和 <= p_low 的边界值)
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for &v in &compressed {
        if v >= p_low && v <= p_high {
            sum += v;
            count += 1;
        }
    }
    let trimmed_mean = if count > 0 {
        sum / count as f64
    } else {
        sorted[n / 2]
    };

    // 5. Median (压缩域)
    let median = sorted[n / 2];

    // 6. 混合
    let center_compressed = rcp.trimmed_mean_weight * trimmed_mean + rcp.median_weight * median;

    // 7. 从压缩域还原到原始亮度域 [0, 1]
    match rcp.compression.as_str() {
        "gamma" => center_compressed.powf(1.0 / rcp.gamma_power),
        "log" => {
            let base = rcp.log_base;
            let ln_base = base.ln();
            let denom = (eps + 1.0_f64).ln() / ln_base;
            (base.powf(center_compressed * denom) - eps).clamp(0.0, 1.0)
        }
        _ => unreachable!(),
    }
}

/// 计算排序数组的指定百分位值 (线性插值)
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
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
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
