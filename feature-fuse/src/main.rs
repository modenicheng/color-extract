// =============================================================================
// feature-fuse — 全链路特征图计算 + Hybrid Fusion
// =============================================================================

mod params;
mod image;
mod dct;
mod gradient;
mod spectral;
mod residual;
mod background;
mod fusion;
mod render;

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use crate::params::{Params, load_params, validate_filter};
use crate::fusion::{percentile_normalize, hybrid_fusion, apply_filter, weights_to_array, composite_with_hybrid, kmeans_impression_color};
use crate::image::load_image;
use crate::dct::compute_dct_complexity;
use crate::gradient::compute_lab_gradient;
use crate::spectral::compute_spectral_residual;
use crate::residual::{
    compute_global_light_residual, compute_global_sat_residual,
    compute_local_light_residual, compute_local_sat_residual,
};
use crate::background::{
    compute_background_lab_residual,
    compute_background_connected_mask,
    mask_to_foreground_confidence,
};

use crate::render::{save_gray_png, save_rgb_png, make_contact_sheet_full};

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    // ── 加载 YAML 参数 ──
    let params_path = Path::new("feature-fuse/params.yaml");
    let params = load_params(params_path)?;

    // ── 校验 filter 配置（互斥检查）──
    if let Some(ref filter) = params.filter {
        validate_filter(filter)?;
        println!("  filter: method={} ✓", filter.method);
    }

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

    println!("  {} ({}×{}) — 9 features …", stem, w, h);

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
    let dct_raw = compute_dct_complexity(&gray, w as usize, h as usize, params.dct.high_freq_threshold);
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
    let sr_raw = compute_spectral_residual(
        &data.lab_l, &data.lab_a, &data.lab_b, w as usize, h as usize,
        params.spectral_residual.mean_filter_kernel,
        params.spectral_residual.gaussian_sigma,
        params.spectral_residual.gamma,
        params.spectral_residual.post_gamma,
    );
    let sr_norm = percentile_normalize(&sr_raw, p_low, p_high);
    save_gray_png(&sr_norm, w, h, &out_dir.join("spectral_residual.png"))?;
    let t_sr = t0.elapsed();

    // ── (4) Global light residual (稳健亮度中心) ──
    let t0 = std::time::Instant::now();
    let gl_l_raw = compute_global_light_residual(&data.hsl_l, &params.global_residual.light);
    let gl_l_norm = percentile_normalize(&gl_l_raw, p_low, p_high);
    save_gray_png(&gl_l_norm, w, h, &out_dir.join("global_light_residual.png"))?;
    let t_gl = t0.elapsed();

    // ── (5) Global sat residual (稳健亮度中心) ──
    let t0 = std::time::Instant::now();
    let gl_s_raw = compute_global_sat_residual(&data.hsl_s, &params.global_residual.sat);
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

    // ── (8) Background LAB residual ──
    let t0 = std::time::Instant::now();
    let bg_raw = compute_background_lab_residual(
        &data.lab_l, &data.lab_a, &data.lab_b, w, h, &params.background,
    );
    let bg_norm = percentile_normalize(&bg_raw, p_low, p_high);
    save_gray_png(&bg_norm, w, h, &out_dir.join("background_lab_residual.png"))?;
    let t_bgl = t0.elapsed();

    // ── (9) Background connected mask + foreground confidence ──
    let t0 = std::time::Instant::now();
    let bg_mask = compute_background_connected_mask(&bg_raw, w, h, &params.background);
    save_gray_png(&bg_mask, w, h, &out_dir.join("background_connected_mask.png"))?;
    let bg_fg = mask_to_foreground_confidence(&bg_mask, params.background.connectedness.strength);
    save_gray_png(&bg_fg, w, h, &out_dir.join("background_foreground_confidence.png"))?;
    let t_bgc = t0.elapsed();

    println!(
        "    DCT={:.1}s LAB={:.1}s SR={:.1}s GL={:.1}s GS={:.1}s LL={:.1}s LS={:.1}s BGL={:.1}s BGC={:.1}s — fusion …",
        t_dct.as_secs_f64(),
        t_lab.as_secs_f64(),
        t_sr.as_secs_f64(),
        t_gl.as_secs_f64(),
        t_gs.as_secs_f64(),
        t_ll.as_secs_f64(),
        t_ls.as_secs_f64(),
        t_bgl.as_secs_f64(),
        t_bgc.as_secs_f64(),
    );

    // ── 归一化后的所有特征 ──
    let features: [&[f64]; 9] = [
        &dct_norm,
        &lab_grad_norm,
        &sr_norm,
        &gl_l_norm,
        &gl_s_norm,
        &loc_l_norm,
        &loc_s_norm,
        &bg_norm,
        &bg_fg,
    ];
    let feature_names: [&str; 9] = [
        "dct_complexity",
        "lab_gradient",
        "spectral_residual",
        "global_light_residual",
        "global_sat_residual",
        "local_light_residual",
        "local_sat_residual",
        "background_lab_residual",
        "background_foreground_confidence",
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

    // ── 对融合图应用过滤（阈值法/分位数法）──
    let (fused_add_filt, fused_mul_filt, fused_hyb_filt) = if let Some(ref flt) = params.filter {
        (
            apply_filter(&fused_add, flt),
            apply_filter(&fused_mul, flt),
            apply_filter(&fused_hybrid, flt),
        )
    } else {
        // 无 filter 时直接复用未过滤版本，避免合成图全黑
        (fused_add.clone(), fused_mul.clone(), fused_hybrid.clone())
    };

    if params.filter.is_some() {
        save_gray_png(&fused_add_filt, w, h, &out_dir.join("fused_add_filtered.png"))?;
        save_gray_png(&fused_mul_filt, w, h, &out_dir.join("fused_softmul_filtered.png"))?;
        save_gray_png(&fused_hyb_filt, w, h, &out_dir.join("fused_hybrid_filtered.png"))?;
    }

    // ── Original × FiltHybrid 复合图 ──
    let composite_rgb = composite_with_hybrid(&data.rgb, &fused_hyb_filt);
    save_rgb_png(&composite_rgb, w, h, &out_dir.join("fused_original_hybrid.png"))?;

    // ── 印象色：k-means++ 聚类，取最大簇 ──
    let k = params.impression.k;
    let max_iter = params.impression.max_iter;
    let impression_color = kmeans_impression_color(&data.rgb, &fused_hyb_filt, k, max_iter);
    let ic_hex = format!(
        "#{:02x}{:02x}{:02x}",
        (impression_color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (impression_color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (impression_color[2].clamp(0.0, 1.0) * 255.0) as u8,
    );
    println!("    impression color: {ic_hex} (k-means cluster, k={k}, iter={max_iter})");

    // ── Contact Sheet ──
    let feat_slices: Vec<(&str, &[f64])> = feature_names.iter().zip(features.iter()).map(|(&n, &f)| (n, f)).collect();
    let fused_slices: [(&str, &[f64]); 6] = [
        ("fused_add", &fused_add),
        ("fused_softmul", &fused_mul),
        ("fused_hybrid", &fused_hybrid),
        ("fused_add_filtered", &fused_add_filt),
        ("fused_softmul_filtered", &fused_mul_filt),
        ("fused_hybrid_filtered", &fused_hyb_filt),
    ];

    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();

    make_contact_sheet_full(
        &data.rgb,
        &feat_slices,
        &fused_slices,
        &[("fused_original_hybrid", composite_rgb.as_slice())],
        Some(impression_color),
        w,
        h,
        &params.contact_sheet,
        &ts,
        &out_base.join(format!("contact_sheet_{stem}.png")),
    )?;

    println!("    ✓ {stem} — all outputs in {}/", out_dir.display());

    Ok(stem)
}
