// =============================================================================
// impression-color-extract — 背景/主体分离对印象色提取的影响探索
// =============================================================================

mod params;
mod image;
mod dct;
mod gradient;
mod spectral;
mod residual;
mod partition;
mod palette;
mod fusion;
mod render;

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use crate::params::{load_params, Params, SoftMaskParams};
use crate::image::load_image;
use crate::dct::compute_dct_complexity;
use crate::gradient::compute_lab_gradient;
use crate::spectral::compute_spectral_residual;
use crate::residual::{compute_local_light_residual, compute_local_sat_residual};
use crate::partition::partition_and_separate;
use crate::palette::extract_palette;
use crate::fusion::percentile_normalize;
use crate::render::{
    save_gray_png, save_gray_png_with_centroid, save_rgb_png, make_contact_sheet, generate_html_report,
};

fn main() -> Result<()> {
    let start_total = std::time::Instant::now();

    // ── 加载 YAML ──
    let params_path = Path::new("impression-color-extract/params.yaml");
    let params = load_params(params_path)?;

    // CLI 覆盖
    let max_dim: u32 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(params.max_dim);

    let out_base: PathBuf = std::env::args().nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&params.output.dir));

    println!("═══ impression-color-extract ═══");
    println!("max_dim={max_dim}, output={}", out_base.display());
    std::fs::create_dir_all(&out_base)?;

    // ── 扫描 imgs/ ──
    let img_dir = Path::new("imgs");
    let mut entries: Vec<_> = std::fs::read_dir(img_dir)
        .context("reading imgs/")?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    let image_paths: Vec<PathBuf> = entries.iter()
        .filter(|e| {
            let ext = e.path().extension()
                .and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            ext == "jpg" || ext == "jpeg" || ext == "png"
        })
        .map(|e| e.path())
        .collect();

    if image_paths.is_empty() {
        anyhow::bail!("No images found in imgs/");
    }
    println!("Found {} image(s)", image_paths.len());

    // ── 并行处理 ──
    let results: Vec<Result<String>> = image_paths.par_iter()
        .map(|path| process_one(path, &params, max_dim, &out_base))
        .collect();

    // ── 汇总 ──
    let mut success = 0;
    let mut errors = Vec::new();
    for r in results {
        match r {
            Ok(stem) => { println!("  ✓ {stem}"); success += 1; }
            Err(e) => errors.push(e),
        }
    }

    let elapsed = start_total.elapsed();
    println!("\nDone! {success}/{} image(s) in {:.2}s", image_paths.len(), elapsed.as_secs_f64());
    for e in &errors { eprintln!("  Error: {e:#}"); }

    Ok(())
}

fn process_one(path: &Path, params: &Params, max_dim: u32, out_base: &Path) -> Result<String> {
    let data = load_image(path, max_dim)?;
    let stem = data.stem.clone();
    let w = data.w;
    let h = data.h;

    let out_dir = out_base.join(&stem);
    std::fs::create_dir_all(&out_dir)?;

    println!("  {} ({}×{}) …", stem, w, h);

    let (p_low, p_high) = (params.percentile.low, params.percentile.high);
    let gs = params.gauss_sigma;

    // ── (1) DCT 纹理复杂度 ──
    let dct_raw = compute_dct_complexity(&data.gray, w as usize, h as usize);
    let dct_norm = percentile_normalize(&dct_raw, p_low, p_high);
    save_gray_png_with_centroid(&dct_norm, w, h, &out_dir.join("dct_complexity.png"))?;

    // ── (2) LAB 梯度 ──
    let lab_raw = compute_lab_gradient(&data.lab_l, &data.lab_a, &data.lab_b, w as usize, h as usize);
    let lab_norm = percentile_normalize(&lab_raw, p_low, p_high);
    save_gray_png(&lab_norm, w, h, &out_dir.join("lab_gradient.png"))?;

    // ── (3) 频谱残差 ──
    let sr_raw = compute_spectral_residual(
        &data.lab_l,
        &data.lab_a,
        &data.lab_b,
        w as usize,
        h as usize,
        &params.spectral_residual,
    );
    let sr_norm = percentile_normalize(&sr_raw, p_low, p_high);
    save_gray_png_with_centroid(&sr_norm, w, h, &out_dir.join("spectral_residual.png"))?;

    // ── (4) 局部残差 ──
    let loc_l_raw = compute_local_light_residual(&data.hsl_l, w, h, gs);
    let loc_l_norm = percentile_normalize(&loc_l_raw, p_low, p_high);
    save_gray_png(&loc_l_norm, w, h, &out_dir.join("local_light_residual.png"))?;

    let loc_s_raw = compute_local_sat_residual(&data.hsl_s, w, h, gs);
    let loc_s_norm = percentile_normalize(&loc_s_raw, p_low, p_high);
    save_gray_png(&loc_s_norm, w, h, &out_dir.join("local_sat_residual.png"))?;

    // ── (5) 色域切分 + 背景分离 ──
    let pr = partition_and_separate(
        &data.lab_l, &data.lab_a, &data.lab_b,
        &data.rgb, w, h, &params.color_partition,
    );

    let n_pixels = (w * h) as usize;
    let has_partition = params.color_partition.enabled && !pr.bg_mask_raw.is_empty();
    let saliency_fg = soft_foreground_saliency(
        &[&dct_norm, &lab_norm, &sr_norm, &loc_l_norm, &loc_s_norm],
        &[
            params.soft_mask.saliency_dct,
            params.soft_mask.saliency_lab_grad,
            params.soft_mask.saliency_spectral,
            params.soft_mask.saliency_local_light,
            params.soft_mask.saliency_local_sat,
        ],
        w,
        h,
        &params.soft_mask,
    );
    let border_bg = border_background_likelihood(
        &data.lab_l,
        &data.lab_a,
        &data.lab_b,
        w as usize,
        h as usize,
        params.color_partition.border_band,
    );
    let (bg_mask_raw, bg_mask_morph, fg_confidence) = if has_partition {
        refine_soft_masks(&pr.bg_mask_raw, &pr.bg_mask_morph, &saliency_fg, &border_bg, w, h, &params.soft_mask)
    } else {
        let fg = soft_foreground_from_background(&border_bg, &saliency_fg, w, h, &params.soft_mask);
        let bg = invert_mask(&fg);
        (bg.clone(), bg, fg)
    };

    // 保存背景相关图
    if has_partition {
        save_rgb_png(&pr.color_clusters_rgb, w, h, &out_dir.join("color_clusters.png"))?;

        // 由于 clusters 顺序可能变化，重新构建与像素顺序一致的 bg_candidate
        let mut bg_candidate_full = vec![0.0; n_pixels];
        for c in &pr.clusters {
            let v = if c.bg_score >= params.color_partition.bg_score_threshold {
                1.0
            } else {
                0.0
            };
            for &idx in &c.indices { bg_candidate_full[idx] = v; }
        }
        save_gray_png(&bg_candidate_full, w, h, &out_dir.join("bg_candidate.png"))?;
        save_gray_png(&border_bg, w, h, &out_dir.join("border_bg.png"))?;
        save_gray_png(&bg_mask_raw, w, h, &out_dir.join("bg_mask_raw.png"))?;
        save_gray_png(&bg_mask_morph, w, h, &out_dir.join("bg_mask_morph.png"))?;
        save_gray_png(&fg_confidence, w, h, &out_dir.join("fg_confidence.png"))?;
    }

    // ── (6) 特征融合（使用权重的软加法）──
    let weights = &params.feature_weights;
    let features: [&[f64]; 7] = [
        &dct_norm, &lab_norm, &sr_norm,
        &loc_l_norm, &loc_s_norm,
        &bg_mask_morph,
        &fg_confidence,
    ];
    let weight_vals = [
        weights.dct, weights.lab_grad, weights.spectral,
        weights.local_light, weights.local_sat,
        weights.bg_mask, weights.fg_confidence,
    ];

    let w_sum: f64 = weight_vals.iter().sum();
    let w_sum = if w_sum < 1e-12 { 1.0 } else { w_sum };

    let mut fused = vec![0.0; n_pixels];
    for (fi, feat) in features.iter().enumerate() {
        let w = weight_vals[fi] / w_sum;
        if w < 1e-12 { continue; }
        for j in 0..n_pixels { fused[j] += feat[j] * w; }
    }
    // Gamma correction
    for j in 0..n_pixels { fused[j] = fused[j].clamp(0.0, 1.0); }
    save_gray_png(&fused, w, h, &out_dir.join("fused.png"))?;

    // ── (7) 调色板提取（仅前景区域）──
    let palette = if has_partition {
        extract_palette(&data.rgb, &fg_confidence, &params.palette)
    } else {
        extract_palette(&data.rgb, &fg_confidence, &params.palette)
    };

    // ── (8) 输出 ──
    // 保存原图
    save_rgb_png(&data.rgb, w, h, &out_dir.join("resized.png"))?;

    // Contact Sheet
    let feature_slices: Vec<(&str, &[f64])> = vec![
        ("dct", &dct_norm),
        ("lab_grad", &lab_norm),
        ("spectral", &sr_norm),
        ("local_light", &loc_l_norm),
        ("local_sat", &loc_s_norm),
        ("bg_mask", &bg_mask_morph),
        ("fg_confidence", &fg_confidence),
        ("fused", &fused),
    ];

    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    make_contact_sheet(
        &data.rgb, &feature_slices, &[],
        w, h, params.output.contact_sheet_cols, params.output.contact_sheet_thumb_w,
        &palette, &ts,
        &out_base.join(format!("{stem}_contact_sheet.png")),
    )?;

    // 单张 contact sheet（所有特征 + 调色板）
    let mut all_slices: Vec<(&str, &[f64])> = feature_slices.clone();
    let empty_slice: &[f64] = &[];
    all_slices.push(("palette", empty_slice));

    // HTML 报告
    let yaml_content = std::fs::read_to_string("impression-color-extract/params.yaml")?;
    generate_html_report(&stem, &palette, &yaml_content, &out_dir.join("report.html"))?;

    Ok(stem)
}

fn soft_foreground_saliency(
    features: &[&[f64]],
    weights: &[f64],
    w: u32,
    h: u32,
    params: &SoftMaskParams,
) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }
    let n = features[0].len();
    let w_sum: f64 = weights.iter().sum();
    if n == 0 || w_sum < 1e-12 {
        return vec![0.0; n];
    }
    let mut saliency = vec![0.0; n];
    for (feature, &weight) in features.iter().zip(weights.iter()) {
        let w = weight / w_sum;
        for i in 0..n {
            saliency[i] += feature[i] * w;
        }
    }
    let base = percentile_normalize(&saliency, 5.0, 95.0);
    let radius = ((w.min(h) / 24).clamp(8, 32)) as usize;
    let smooth = box_blur_mask(&base, w as usize, h as usize, radius);
    let mut soft = vec![0.0; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            let subject = subject_prior(x, y, w as usize, h as usize, params);
            soft[i] = weighted3(
                base[i],
                params.saliency_base_weight,
                smooth[i],
                params.saliency_smooth_weight,
                subject,
                params.saliency_subject_weight,
            );
        }
    }
    let mut soft = percentile_normalize(&soft, 2.0, 98.0);
    for v in &mut soft {
        *v = v.powf(0.85).clamp(0.0, 1.0);
    }
    soft
}

fn border_background_likelihood(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: usize,
    h: usize,
    border_band: u32,
) -> Vec<f64> {
    let n = w.saturating_mul(h);
    if n == 0 {
        return Vec::new();
    }

    let band = (border_band as usize).max(1).min(w.max(1)).min(h.max(1));
    let mut border_indices = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if x < band || y < band || x + band >= w || y + band >= h {
                border_indices.push(y * w + x);
            }
        }
    }
    if border_indices.is_empty() {
        return vec![0.0; n];
    }

    let max_prototypes = 96usize;
    let step = (border_indices.len() / max_prototypes).max(1);
    let prototypes: Vec<usize> = border_indices
        .iter()
        .step_by(step)
        .take(max_prototypes)
        .copied()
        .collect();

    let mut bg = vec![0.0; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut best = f64::MAX;
            for &p in &prototypes {
                let d_l = (lab_l[i] - lab_l[p]) / 100.0;
                let d_a = (lab_a[i] - lab_a[p]) / 128.0;
                let d_b = (lab_b[i] - lab_b[p]) / 128.0;
                let d = (d_l * d_l + d_a * d_a + d_b * d_b).sqrt();
                if d < best {
                    best = d;
                }
            }
            let color_bg = (-(best / 0.16).powi(2)).exp();
            let edge_bg = 1.0 - center_prior(x, y, w, h);
            bg[i] = (color_bg * (0.58 + 0.42 * edge_bg)).clamp(0.0, 1.0);
        }
    }

    let radius = ((w.min(h) / 48).clamp(4, 16)) as usize;
    box_blur_mask(&bg, w, h, radius)
}

fn center_prior(x: usize, y: usize, w: usize, h: usize) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let nx = (x as f64 + 0.5) / w as f64 - 0.5;
    let ny = (y as f64 + 0.5) / h as f64 - 0.5;
    let dist = (nx * nx + ny * ny).sqrt() / 0.70710678118;
    (1.0 - dist).clamp(0.0, 1.0)
}

fn subject_prior(x: usize, y: usize, w: usize, h: usize, params: &SoftMaskParams) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let nx = (x as f64 + 0.5) / w as f64;
    let ny = (y as f64 + 0.5) / h as f64;
    let dx = (nx - params.subject_center_x) / params.subject_radius_x.max(1e-6);
    let dy = (ny - params.subject_center_y) / params.subject_radius_y.max(1e-6);
    (-(dx * dx + dy * dy)).exp().clamp(0.0, 1.0)
}

fn box_blur_mask(mask: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    if mask.is_empty() || w == 0 || h == 0 || radius == 0 {
        return mask.to_vec();
    }

    let mut horizontal = vec![0.0; mask.len()];
    for y in 0..h {
        let mut prefix = vec![0.0; w + 1];
        for x in 0..w {
            prefix[x + 1] = prefix[x] + mask[y * w + x];
        }
        for x in 0..w {
            let left = x.saturating_sub(radius);
            let right = (x + radius).min(w - 1);
            let sum = prefix[right + 1] - prefix[left];
            horizontal[y * w + x] = sum / (right - left + 1) as f64;
        }
    }

    let mut out = vec![0.0; mask.len()];
    for x in 0..w {
        let mut prefix = vec![0.0; h + 1];
        for y in 0..h {
            prefix[y + 1] = prefix[y] + horizontal[y * w + x];
        }
        for y in 0..h {
            let top = y.saturating_sub(radius);
            let bottom = (y + radius).min(h - 1);
            let sum = prefix[bottom + 1] - prefix[top];
            out[y * w + x] = sum / (bottom - top + 1) as f64;
        }
    }
    out
}

fn refine_soft_masks(
    bg_raw: &[f64],
    bg_morph: &[f64],
    saliency_fg: &[f64],
    border_bg: &[f64],
    w: u32,
    h: u32,
    params: &SoftMaskParams,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw_bg = soft_background_from_partition(bg_raw, border_bg);
    let morph_bg = soft_background_from_partition(bg_morph, border_bg);
    let fg = soft_foreground_from_background(&morph_bg, saliency_fg, w, h, params);
    let morph_bg = invert_mask(&fg);
    (raw_bg, morph_bg, fg)
}

fn soft_background_from_partition(mask: &[f64], border_bg: &[f64]) -> Vec<f64> {
    let stats = mask_stats(mask);
    let partition_weight = if stats.unique_values <= 2 && (stats.mean <= 0.02 || stats.mean >= 0.98) {
        0.0
    } else if stats.unique_values <= 2 {
        0.40
    } else {
        0.55
    };
    mask.iter()
        .zip(border_bg.iter())
        .map(|(&m, &b)| (m * partition_weight + b * (1.0 - partition_weight)).clamp(0.0, 1.0))
        .collect()
}

fn soft_foreground_from_background(
    bg: &[f64],
    saliency_fg: &[f64],
    w: u32,
    h: u32,
    params: &SoftMaskParams,
) -> Vec<f64> {
    let mut fg = vec![0.0; bg.len()];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            let color_fg = 1.0 - bg[i];
            let subject = subject_prior(x, y, w as usize, h as usize, params);
            fg[i] = weighted3(
                color_fg,
                params.foreground_color_weight,
                saliency_fg[i],
                params.foreground_saliency_weight,
                subject,
                params.foreground_subject_weight,
            );
        }
    }
    let radius = ((w.min(h) / 40).clamp(6, 24)) as usize;
    percentile_normalize(&box_blur_mask(&fg, w as usize, h as usize, radius), 2.0, 98.0)
}

fn weighted3(a: f64, aw: f64, b: f64, bw: f64, c: f64, cw: f64) -> f64 {
    let aw = aw.max(0.0);
    let bw = bw.max(0.0);
    let cw = cw.max(0.0);
    let sum = aw + bw + cw;
    if sum < 1e-12 {
        return 0.0;
    }
    ((a * aw + b * bw + c * cw) / sum).clamp(0.0, 1.0)
}

fn invert_mask(mask: &[f64]) -> Vec<f64> {
    mask.iter().map(|&v| (1.0 - v).clamp(0.0, 1.0)).collect()
}

struct MaskStats {
    mean: f64,
    unique_values: usize,
}

fn mask_stats(mask: &[f64]) -> MaskStats {
    if mask.is_empty() {
        return MaskStats { mean: 0.0, unique_values: 0 };
    }
    let mean = mask.iter().sum::<f64>() / mask.len() as f64;
    let mut seen = Vec::new();
    for &v in mask {
        let q = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        if !seen.contains(&q) {
            seen.push(q);
            if seen.len() > 2 {
                break;
            }
        }
    }
    MaskStats { mean, unique_values: seen.len() }
}
