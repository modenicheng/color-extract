// =============================================================================
// feature-fuse — 全链路特征图计算 + Hybrid Fusion
//
// 从 imgs/ 读取原始图片，统一缩放到 ≤720×720，在同一程序中计算所有特征图：
//   DCT complexity | LAB gradient | Spectral residual
//   Global light/sat residual | Local (Gaussian) light/sat residual
// 然后做 Hybrid Fusion (加权加法 + 软乘法混合)，输出各特征图 + 融合图 + 拼贴图。
//
// 调参区域：下面所有 `const` 均可按需修改，改完即跑。
// =============================================================================

use anyhow::{Context, Result};
use image::{GenericImageView, GrayImage, ImageBuffer, ImageReader, Luma, Rgb};
use palette::{Hsl, IntoColor, Lab, Srgb};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// =============================================================================
// [调参区]
// 编译时参数（DCT 块大小等算法结构常量）在此。
// 运行时可调参数请编辑 `params.yaml`，无需重新编译。
// =============================================================================

/// DCT 块大小（算法结构常量，改需重新编译）
const DCT_N: usize = 8;
/// DCT 高频阈值: u+v >= 此值视为高频
const DCT_THRESHOLD: usize = 4;

// =============================================================================
// YAML 参数加载
// =============================================================================

#[derive(Debug, Deserialize)]
struct Params {
    max_dim: u32,
    gauss_sigma: f32,
    percentile: PercentileParams,
    weights_add: Weights,
    weights_mul: Weights,
    fusion: FusionParams,
    contact_sheet: ContactSheetParams,
}

#[derive(Debug, Deserialize)]
struct PercentileParams {
    low: f64,
    high: f64,
}

#[derive(Debug, Deserialize)]
struct Weights {
    dct: f64,
    lab_grad: f64,
    spectral: f64,
    global_light: f64,
    global_sat: f64,
    local_light: f64,
    local_sat: f64,
}

#[derive(Debug, Deserialize)]
struct FusionParams {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
}

#[derive(Debug, Deserialize)]
struct ContactSheetParams {
    cols: u32,
    rows: u32,
    pad: u32,
    thumb_w: u32,
    label_h: u32,
}

fn load_params(path: &Path) -> Result<Params> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let params: Params = serde_yaml::from_str(&content)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(params)
}

// =============================================================================
// 1. 图片加载
// =============================================================================

/// 加载图片，统一 resize，返回所有需要的通道数据。
struct ImageData {
    stem: String,
    w: u32,
    h: u32,
    /// RGB 像素，每个分量 [0, 1]
    rgb: Vec<[f64; 3]>,
    /// CIELAB L* [0, 100]
    lab_l: Vec<f64>,
    /// CIELAB a*
    lab_a: Vec<f64>,
    /// CIELAB b*
    lab_b: Vec<f64>,
    /// HSL saturation [0, 1]
    hsl_s: Vec<f64>,
    /// HSL lightness [0, 1]
    hsl_l: Vec<f64>,
}

fn load_image(path: &Path, max_dim: u32) -> Result<ImageData> {
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

// =============================================================================
// 2. DCT 纹理复杂度 (复用 dct-viz 算法)
// =============================================================================

fn dct_matrix() -> [[f64; DCT_N]; DCT_N] {
    let mut t = [[0.0; DCT_N]; DCT_N];
    let inv_sqrt_n = 1.0 / (DCT_N as f64).sqrt();
    let sqrt_2_over_n = (2.0 / DCT_N as f64).sqrt();
    for i in 0..DCT_N {
        let alpha = if i == 0 { inv_sqrt_n } else { sqrt_2_over_n };
        for j in 0..DCT_N {
            t[i][j] = alpha
                * ((2.0 * j as f64 + 1.0) * i as f64 * std::f64::consts::PI
                    / (2.0 * DCT_N as f64))
                    .cos();
        }
    }
    t
}

fn transpose(m: &[[f64; DCT_N]; DCT_N]) -> [[f64; DCT_N]; DCT_N] {
    let mut out = [[0.0; DCT_N]; DCT_N];
    for i in 0..DCT_N {
        for j in 0..DCT_N {
            out[j][i] = m[i][j];
        }
    }
    out
}

fn dct_2d(block: &[[f64; DCT_N]; DCT_N], t: &[[f64; DCT_N]; DCT_N]) -> [[f64; DCT_N]; DCT_N] {
    let tt = transpose(t);
    let mut rows_dct = [[0.0; DCT_N]; DCT_N];
    for r in 0..DCT_N {
        for c in 0..DCT_N {
            for k in 0..DCT_N {
                rows_dct[r][c] += block[r][k] * tt[k][c];
            }
        }
    }
    let mut out = [[0.0; DCT_N]; DCT_N];
    for r in 0..DCT_N {
        for c in 0..DCT_N {
            for k in 0..DCT_N {
                out[r][c] += t[r][k] * rows_dct[k][c];
            }
        }
    }
    out
}

fn high_freq_ratio(coeffs: &[[f64; DCT_N]; DCT_N]) -> f64 {
    let mut total_ac = 0.0;
    let mut high_freq = 0.0;
    for u in 0..DCT_N {
        for v in 0..DCT_N {
            if u == 0 && v == 0 {
                continue;
            }
            let e = coeffs[u][v] * coeffs[u][v];
            total_ac += e;
            if u + v >= DCT_THRESHOLD {
                high_freq += e;
            }
        }
    }
    high_freq / (total_ac + 1e-10)
}

/// 计算 DCT 纹理复杂度图 (高频能量占比)
fn compute_dct_complexity(gray: &[f64], w: usize, h: usize) -> Vec<f64> {
    let offset = (DCT_N / 2) as i32;
    let t = dct_matrix();
    let mut map = vec![0.0; w * h];

    map.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut block = [[0.0; DCT_N]; DCT_N];
            for dy in 0..DCT_N {
                for dx in 0..DCT_N {
                    let px = (x as i32 + dx as i32 - offset).clamp(0, w as i32 - 1) as usize;
                    let py = (y as i32 + dy as i32 - offset).clamp(0, h as i32 - 1) as usize;
                    block[dy][dx] = gray[py * w + px];
                }
            }
            let coeffs = dct_2d(&block, &t);
            row[x] = high_freq_ratio(&coeffs);
        }
    });
    map
}

// =============================================================================
// 3. LAB Sobel 梯度 (复用 lab-gradient 算法)
// =============================================================================

/// Sobel 梯度幅值：对单通道二维网格计算 |∇f|
fn sobel_magnitude(ch: &[f64], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut mag = vec![0.0; n];

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let i = y * w + x;
            let gx = -1.0 * ch[i - w - 1]
                + 1.0 * ch[i - w + 1]
                - 2.0 * ch[i - 1]
                + 2.0 * ch[i + 1]
                - 1.0 * ch[i + w - 1]
                + 1.0 * ch[i + w + 1];
            let gy = -1.0 * ch[i - w - 1]
                - 2.0 * ch[i - w]
                - 1.0 * ch[i - w + 1]
                + 1.0 * ch[i + w - 1]
                + 2.0 * ch[i + w]
                + 1.0 * ch[i + w + 1];
            // 除以 8 归一化到近似每像素 delta
            mag[i] = ((gx * gx + gy * gy).sqrt()) / 8.0;
        }
    }
    mag
}

/// 计算 LAB 梯度融合图：sqrt(gL² + ga² + gb²)
fn compute_lab_gradient(lab_l: &[f64], lab_a: &[f64], lab_b: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mag_l = sobel_magnitude(lab_l, w, h);
    let mag_a = sobel_magnitude(lab_a, w, h);
    let mag_b = sobel_magnitude(lab_b, w, h);

    let n = w * h;
    let mut fused = Vec::with_capacity(n);
    for i in 0..n {
        fused.push((mag_l[i] * mag_l[i] + mag_a[i] * mag_a[i] + mag_b[i] * mag_b[i]).sqrt());
    }
    fused
}

// =============================================================================
// 4. 频谱残差显著性检测 (复用 spectral-residual 算法)
// =============================================================================

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
fn compute_spectral_residual(
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

// =============================================================================
// 5. Global Residual (全局均值残差)
// =============================================================================

/// 计算 HSL lightness 全局均值残差: |pixel - global_mean|
fn compute_global_light_residual(hsl_l: &[f64]) -> Vec<f64> {
    let n = hsl_l.len();
    let mean = hsl_l.iter().sum::<f64>() / n as f64;
    hsl_l.iter().map(|&v| (v - mean).abs()).collect()
}

/// 计算 HSL saturation 全局均值残差: |pixel - global_mean|
fn compute_global_sat_residual(hsl_s: &[f64]) -> Vec<f64> {
    let n = hsl_s.len();
    let mean = hsl_s.iter().sum::<f64>() / n as f64;
    hsl_s.iter().map(|&v| (v - mean).abs()).collect()
}

// =============================================================================
// 6. Local (Gaussian) Residual
// =============================================================================

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

fn compute_local_light_residual(hsl_l: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    compute_gaussian_residual(hsl_l, w, h, sigma)
}

fn compute_local_sat_residual(hsl_s: &[f64], w: u32, h: u32, sigma: f32) -> Vec<f64> {
    compute_gaussian_residual(hsl_s, w, h, sigma)
}

// =============================================================================
// 7. Percentile Normalize
// =============================================================================

/// Percentile 归一化到 [0,1]：低于 p_low% 置 0，高于 p_high% 置 1，中间线性拉伸
fn percentile_normalize(data: &[f64], p_low: f64, p_high: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let lo_idx = ((n as f64) * p_low / 100.0).floor() as usize;
    let hi_idx = ((n as f64) * p_high / 100.0).ceil() as usize;
    let lo_val = sorted[lo_idx.min(n - 1)];
    let hi_val = sorted[hi_idx.min(n - 1)];
    let range = (hi_val - lo_val).max(1e-12);

    data.iter()
        .map(|&v| ((v - lo_val) / range).clamp(0.0, 1.0))
        .collect()
}

// =============================================================================
// 8. Hybrid Fusion
// =============================================================================

/// 从 Params 中提取 7 元素权重数组
fn weights_to_array(w: &Weights) -> [f64; 7] {
    [w.dct, w.lab_grad, w.spectral, w.global_light, w.global_sat, w.local_light, w.local_sat]
}

/// Hybrid Fusion: 加权加法分支 + 软乘法分支 混合，各自独立权重
fn hybrid_fusion(
    features: &[&[f64]],
    add_w: &[f64; 7],
    mul_w: &[f64; 7],
    alpha: f64,
    gamma: f64,
    epsilon: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(add_w.len(), features.len());
    assert_eq!(mul_w.len(), features.len());

    let n = features[0].len();

    // 归一化加法/乘法权重
    let add_total: f64 = add_w.iter().sum();
    let add_norm: Vec<f64> = add_w.iter().map(|&w| w / add_total).collect();
    let mul_total: f64 = mul_w.iter().sum();
    let mul_norm: Vec<f64> = mul_w.iter().map(|&w| w / mul_total).collect();

    let mut add_score = vec![0.0; n];
    let mut mul_score = vec![0.0; n];
    let mut mul_has_data = vec![false; n];

    for (fi, feat) in features.iter().enumerate() {
        let wa = add_norm[fi];
        let wm = mul_norm[fi];
        for j in 0..n {
            add_score[j] += feat[j] * wa;
            let base = epsilon + (1.0 - epsilon) * feat[j];
            if !mul_has_data[j] {
                mul_score[j] = base.ln() * wm;
                mul_has_data[j] = true;
            } else {
                mul_score[j] += base.ln() * wm;
            }
        }
    }

    // Exponentiate softmul
    for j in 0..n {
        mul_score[j] = mul_score[j].exp();
    }

    // Blend
    let mut hybrid = vec![0.0; n];
    for j in 0..n {
        hybrid[j] = alpha * add_score[j] + (1.0 - alpha) * mul_score[j];
        hybrid[j] = hybrid[j].powf(gamma);
        hybrid[j] = hybrid[j].clamp(0.0, 1.0);
    }

    // Gamma adjust for individual branches too
    for j in 0..n {
        add_score[j] = add_score[j].powf(gamma).clamp(0.0, 1.0);
        mul_score[j] = mul_score[j].powf(gamma).clamp(0.0, 1.0);
    }

    (add_score, mul_score, hybrid)
}

// =============================================================================
// 9. 保存单通道特征图为 PNG
// =============================================================================

fn save_gray_png(data: &[f64], w: u32, h: u32, path: &Path) -> Result<()> {
    let img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let v = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });
    img.save(path).context(format!("save {}", path.display()))
}

fn save_rgb_png(rgb: &[[f64; 3]], w: u32, h: u32, path: &Path) -> Result<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let r = (rgb[i][0].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (rgb[i][1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (rgb[i][2].clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([r, g, b])
    });
    img.save(path).context(format!("save {}", path.display()))
}

// =============================================================================
// 10. 8×8 位图字体标签绘制
// =============================================================================

/// 在 RGB 图像上绘制文本（使用 font8x8 位图字体，8×8 等宽）
fn draw_text_rgb(canvas: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, text: &str, x: i32, y: i32, color: Rgb<u8>) {
    let (cw, ch) = canvas.dimensions();
    for (ci, &ch_byte) in text.as_bytes().iter().enumerate() {
        let idx = ch_byte as usize;
        if idx >= font8x8::legacy::BASIC_LEGACY.len() {
            continue;
        }
        let glyph = &font8x8::legacy::BASIC_LEGACY[idx];
        for row in 0..8 {
            let row_data = glyph[row];
            for col in 0..8 {
                // MSB = leftmost pixel in font8x8
                if (row_data >> col) & 1 != 0 {
                    let px = x + (ci as i32) * 8 + col as i32;
                    let py = y + row as i32;
                    if px >= 0 && py >= 0 && px < cw as i32 && py < ch as i32 {
                        canvas.put_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }
}

// =============================================================================
// 11. Contact Sheet 拼贴图（带标签）
// =============================================================================

/// 创建包含原图、7 张特征图、3 张融合图的 contact sheet 拼贴图
/// 每张子图下方带说明标签，右下角空位印时间戳
fn make_contact_sheet_full(
    original: &[[f64; 3]],
    features: &[(&str, &[f64])],
    fused: &[(&str, &[f64])],
    w: u32,
    h: u32,
    layout: &ContactSheetParams,
    timestamp: &str,
    path: &Path,
) -> Result<()> {
    let cell_w = layout.thumb_w.min(w);
    let cell_h = (cell_w as f64 * h as f64 / w as f64).round() as u32;

    // sheet 高度每行多出 label_h 用于标签
    let step_h = cell_h + layout.label_h;
    let sheet_w = layout.cols * (cell_w + layout.pad) + layout.pad;
    let sheet_h = layout.rows * (step_h + layout.pad) + layout.pad;

    // 收集所有缩略图 (RGB)
    let mut thumbs: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::with_capacity((layout.cols * layout.rows) as usize);

    // 原图缩略
    thumbs.push(make_thumb_rgb(original, w, h, cell_w, cell_h));

    // 7 张特征图
    for &(_name, feat) in features {
        thumbs.push(make_thumb_gray_f32(feat, w, h, cell_w, cell_h));
    }

    // 3 张融合图
    for &(_name, f) in fused {
        thumbs.push(make_thumb_gray_f32(f, w, h, cell_w, cell_h));
    }

    // 每张图对应的标签文字
    const CELL_LABELS: [&str; 11] = [
        "Original",
        "DCT",
        "LAB Grad",
        "Spect Res",
        "G-Light",
        "G-Sat",
        "L-Light",
        "L-Sat",
        "Fuse Add",
        "Fuse Mult",
        "Fuse Hybrid",
    ];

    let mut sheet: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(sheet_w, sheet_h);
    // 白色背景
    for pixel in sheet.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    for (idx, thumb) in thumbs.iter().enumerate() {
        if idx >= (layout.cols * layout.rows) as usize {
            break;
        }
        let col = idx as u32 % layout.cols;
        let row = idx as u32 / layout.cols;
        let ox = layout.pad + col * (cell_w + layout.pad);
        let oy = layout.pad + row * (step_h + layout.pad);

        // overlay thumbnail
        image::imageops::overlay(&mut sheet, thumb, ox as i64, oy as i64);

        // draw label centered below the thumbnail
        if let Some(&label) = CELL_LABELS.get(idx) {
            let text_w = label.len() as i32 * 8;
            let label_x = ox as i32 + (cell_w as i32 - text_w) / 2;
            let label_y = oy as i32 + cell_h as i32 + 4;
            draw_text_rgb(&mut sheet, label, label_x.max(0), label_y, Rgb([40, 40, 40]));
        }
    }

    // ── 右下角（第 12 格）印时间戳 ──
    let ts_col = layout.cols - 1;
    let ts_row = layout.rows - 1;
    let ts_ox = layout.pad + ts_col * (cell_w + layout.pad);
    let ts_oy = layout.pad + ts_row * (step_h + layout.pad);
    let ts_text_w = timestamp.len() as i32 * 8;
    let ts_label_x = ts_ox as i32 + (cell_w as i32 - ts_text_w) / 2;
    let ts_label_y = ts_oy as i32 + (cell_h as i32 / 2) - 4; // 垂直居中
    draw_text_rgb(&mut sheet, timestamp, ts_label_x.max(0), ts_label_y, Rgb([120, 120, 120]));

    sheet.save(path).context(format!("save contact sheet {}", path.display()))?;
    Ok(())
}

/// 将 f32 [0,1] 特征图转为 RGB 缩略图（灰度映射到 RGB）
fn make_thumb_gray_f32(data: &[f64], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let gray: GrayImage = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let i = (y * src_w + x) as usize;
        let v = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });
    let thumb = image::imageops::resize(&gray, dst_w, dst_h, image::imageops::FilterType::Lanczos3);
    // 转为 RGB（三通道相同）
    ImageBuffer::from_fn(dst_w, dst_h, |x, y| {
        let Luma([g]) = thumb.get_pixel(x, y);
        Rgb([*g, *g, *g])
    })
}

/// 将 RGB [0,1] 原图转为 RGB 缩略图
fn make_thumb_rgb(data: &[[f64; 3]], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let orig: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let i = (y * src_w + x) as usize;
        let r = (data[i][0].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (data[i][1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (data[i][2].clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([r, g, b])
    });
    image::imageops::resize(&orig, dst_w, dst_h, image::imageops::FilterType::Lanczos3)
}

// =============================================================================
// 11. Main
// =============================================================================

fn main() -> Result<()> {
    // ── 加载 YAML 参数 ──
    let params_path = Path::new("feature-fuse/params.yaml");
    let params = load_params(params_path)?;

    // ── CLI 参数可覆盖 max_dim ──
    let max_dim: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(params.max_dim);

    let out_base: PathBuf = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("output/feature-fuse"));

    println!("═══ feature-fuse: 全链路特征图计算 + Hybrid Fusion ═══");
    println!("max dim: {max_dim}, output: {}", out_base.display());

    let start_total = std::time::Instant::now();

    std::fs::create_dir_all(&out_base).context("create output base dir")?;

    // ── 扫描 imgs/ ──
    let img_dir = Path::new("imgs");
    let mut entries: Vec<_> = std::fs::read_dir(img_dir)
        .context("reading imgs/")?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    let image_paths: Vec<PathBuf> = entries
        .iter()
        .filter(|e| {
            let ext = e
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            ext == "jpg" || ext == "jpeg" || ext == "png"
        })
        .map(|e| e.path())
        .collect();

    if image_paths.is_empty() {
        anyhow::bail!("No images found in imgs/");
    }

    println!("Found {} image(s)", image_paths.len());

    // ── 多图片并行处理 ──
    let results: Vec<Result<String>> = image_paths
        .par_iter()
        .map(|path| {
            process_one_image(path, &params, max_dim, &out_base)
        })
        .collect();

    // ── 汇总 ──
    let mut success = 0;
    let mut errors = Vec::new();
    for r in results {
        match r {
            Ok(stem) => {
                println!("  ✓ {stem}");
                success += 1;
            }
            Err(e) => {
                errors.push(e);
            }
        }
    }

    let elapsed = start_total.elapsed();
    println!(
        "\nDone! {success}/{} image(s) processed in {:.2}s",
        image_paths.len(),
        elapsed.as_secs_f64()
    );
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("  Error: {e:#}");
        }
    }
    Ok(())
}

/// 处理单张图片：计算所有特征 → 归一化 → 融合 → 输出
fn process_one_image(path: &Path, params: &Params, max_dim: u32, out_base: &Path) -> Result<String> {
    // ── 加载图片 ──
    let data = load_image(path, max_dim)?;
    let stem = data.stem.clone();
    let w = data.w;
    let h = data.h;

    // 创建输出目录
    let out_dir = out_base.join(&stem);
    std::fs::create_dir_all(&out_dir)?;

    println!("  {} ({}×{}) — 7 features …", stem, w, h);

    // ── 保存 resize 后的原图 ──
    save_rgb_png(&data.rgb, w, h, &out_dir.join("resized.png"))?;

    // ── 计算灰度图 (用于 DCT) ──
    let gray: Vec<f64> = data
        .rgb
        .iter()
        .map(|&[r, g, b]| 0.299 * r + 0.587 * g + 0.114 * b)
        .collect();

    let (p_low, p_high) = (params.percentile.low, params.percentile.high);

    // ── (1) DCT 纹理复杂度 ──
    let t0 = std::time::Instant::now();
    let dct_raw = compute_dct_complexity(&gray, w as usize, h as usize);
    let dct_norm = percentile_normalize(&dct_raw, p_low, p_high);
    save_gray_png(&dct_norm, w, h, &out_dir.join("dct_complexity.png"))?;
    let t_dct = t0.elapsed();

    // ── (2) LAB 梯度 ──
    let t0 = std::time::Instant::now();
    let lab_grad_raw = compute_lab_gradient(&data.lab_l, &data.lab_a, &data.lab_b, w as usize, h as usize);
    let lab_grad_norm = percentile_normalize(&lab_grad_raw, p_low, p_high);
    save_gray_png(&lab_grad_norm, w, h, &out_dir.join("lab_gradient.png"))?;
    let t_lab = t0.elapsed();

    // ── (3) 频谱残差显著性 ──
    let t0 = std::time::Instant::now();
    let sr_raw = compute_spectral_residual(&data.lab_l, &data.lab_a, &data.lab_b, w as usize, h as usize);
    let sr_norm = percentile_normalize(&sr_raw, p_low, p_high);
    save_gray_png(&sr_norm, w, h, &out_dir.join("spectral_residual.png"))?;
    let t_sr = t0.elapsed();

    // ── (4) Global light residual ──
    let t0 = std::time::Instant::now();
    let gl_l_raw = compute_global_light_residual(&data.hsl_l);
    let gl_l_norm = percentile_normalize(&gl_l_raw, p_low, p_high);
    save_gray_png(&gl_l_norm, w, h, &out_dir.join("global_light_residual.png"))?;
    let t_gl = t0.elapsed();

    // ── (5) Global sat residual ──
    let t0 = std::time::Instant::now();
    let gl_s_raw = compute_global_sat_residual(&data.hsl_s);
    let gl_s_norm = percentile_normalize(&gl_s_raw, p_low, p_high);
    save_gray_png(&gl_s_norm, w, h, &out_dir.join("global_sat_residual.png"))?;
    let t_gs = t0.elapsed();

    // ── (6) Local (Gaussian) light residual ──
    let t0 = std::time::Instant::now();
    let loc_l_raw = compute_local_light_residual(&data.hsl_l, w, h, params.gauss_sigma);
    let loc_l_norm = percentile_normalize(&loc_l_raw, p_low, p_high);
    save_gray_png(&loc_l_norm, w, h, &out_dir.join("local_light_residual.png"))?;
    let t_ll = t0.elapsed();

    // ── (7) Local (Gaussian) sat residual ──
    let t0 = std::time::Instant::now();
    let loc_s_raw = compute_local_sat_residual(&data.hsl_s, w, h, params.gauss_sigma);
    let loc_s_norm = percentile_normalize(&loc_s_raw, p_low, p_high);
    save_gray_png(&loc_s_norm, w, h, &out_dir.join("local_sat_residual.png"))?;
    let t_ls = t0.elapsed();

    println!(
        "    DCT={:.1}s LAB={:.1}s SR={:.1}s GL={:.1}s GS={:.1}s LL={:.1}s LS={:.1}s — fusion …",
        t_dct.as_secs_f64(),
        t_lab.as_secs_f64(),
        t_sr.as_secs_f64(),
        t_gl.as_secs_f64(),
        t_gs.as_secs_f64(),
        t_ll.as_secs_f64(),
        t_ls.as_secs_f64(),
    );

    // ── 归一化后的所有特征 ──
    let features: [&[f64]; 7] = [
        &dct_norm,
        &lab_grad_norm,
        &sr_norm,
        &gl_l_norm,
        &gl_s_norm,
        &loc_l_norm,
        &loc_s_norm,
    ];
    let feature_names: [&str; 7] = [
        "dct_complexity",
        "lab_gradient",
        "spectral_residual",
        "global_light_residual",
        "global_sat_residual",
        "local_light_residual",
        "local_sat_residual",
    ];

    // ── 从 YAML 提取权重数组 ──
    let add_w = weights_to_array(&params.weights_add);
    let mul_w = weights_to_array(&params.weights_mul);

    // ── Hybrid Fusion ──
    let (fused_add, fused_mul, fused_hybrid) =
        hybrid_fusion(&features, &add_w, &mul_w, params.fusion.alpha, params.fusion.gamma, params.fusion.epsilon);

    save_gray_png(&fused_add, w, h, &out_dir.join("fused_add.png"))?;
    save_gray_png(&fused_mul, w, h, &out_dir.join("fused_softmul.png"))?;
    save_gray_png(&fused_hybrid, w, h, &out_dir.join("fused_hybrid.png"))?;

    // ── Contact Sheet ──
    let feat_slices: Vec<(&str, &[f64])> = feature_names.iter().zip(features.iter()).map(|(&n, &f)| (n, f)).collect();
    let fused_slices: [(&str, &[f64]); 3] = [
        ("fused_add", &fused_add),
        ("fused_softmul", &fused_mul),
        ("fused_hybrid", &fused_hybrid),
    ];

    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();

    make_contact_sheet_full(
        &data.rgb,
        &feat_slices,
        &fused_slices,
        w,
        h,
        &params.contact_sheet,
        &ts,
        &out_dir.join("contact_sheet.png"),
    )?;

    println!("    ✓ {stem} — all outputs in {}/", out_dir.display());

    Ok(stem)
}
