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

use crate::params::{load_params, Params};
use crate::image::load_image;
use crate::dct::compute_dct_complexity;
use crate::gradient::compute_lab_gradient;
use crate::spectral::compute_spectral_residual;
use crate::residual::{compute_local_light_residual, compute_local_sat_residual};
use crate::partition::partition_and_separate;
use crate::palette::extract_palette;
use crate::fusion::percentile_normalize;
use crate::render::{
    save_gray_png, save_rgb_png, make_contact_sheet, generate_html_report,
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
    save_gray_png(&dct_norm, w, h, &out_dir.join("dct_complexity.png"))?;

    // ── (2) LAB 梯度 ──
    let lab_raw = compute_lab_gradient(&data.lab_l, &data.lab_a, &data.lab_b, w as usize, h as usize);
    let lab_norm = percentile_normalize(&lab_raw, p_low, p_high);
    save_gray_png(&lab_norm, w, h, &out_dir.join("lab_gradient.png"))?;

    // ── (3) 频谱残差 ──
    let sr_raw = compute_spectral_residual(&data.lab_l, &data.lab_a, &data.lab_b, w as usize, h as usize);
    let sr_norm = percentile_normalize(&sr_raw, p_low, p_high);
    save_gray_png(&sr_norm, w, h, &out_dir.join("spectral_residual.png"))?;

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

    // 保存背景相关图
    if has_partition {
        save_rgb_png(&pr.color_clusters_rgb, w, h, &out_dir.join("color_clusters.png"))?;

        // 背景候选 = bg_mask_raw（直接用 bg_mask 的特征做可视化）
        let _bg_candidate: Vec<f64> = pr.clusters.iter()
            .flat_map(|c| {
                let v = if c.bg_score > 0.5 { 1.0 } else { 0.0 };
                std::iter::repeat(v).take(c.indices.len())
            })
            .collect();
        // 由于 clusters 顺序可能变化，重新构建与像素顺序一致的 bg_candidate
        let mut bg_candidate_full = vec![0.0; n_pixels];
        for c in &pr.clusters {
            let v = if c.bg_score > 0.5 { 1.0 } else { 0.0 };
            for &idx in &c.indices { bg_candidate_full[idx] = v; }
        }
        save_gray_png(&bg_candidate_full, w, h, &out_dir.join("bg_candidate.png"))?;
        save_gray_png(&pr.bg_mask_raw, w, h, &out_dir.join("bg_mask_raw.png"))?;
        save_gray_png(&pr.bg_mask_morph, w, h, &out_dir.join("bg_mask_morph.png"))?;
        save_gray_png(&pr.fg_confidence, w, h, &out_dir.join("fg_confidence.png"))?;
    }

    // ── (6) 特征融合（使用权重的软加法）──
    let weights = &params.feature_weights;
    let features: [&[f64]; 7] = [
        &dct_norm, &lab_norm, &sr_norm,
        &loc_l_norm, &loc_s_norm,
        &pr.bg_mask_morph,
        &pr.fg_confidence,
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
        extract_palette(&data.rgb, &pr.fg_confidence, &params.palette)
    } else {
        let ones: Vec<f64> = vec![1.0; n_pixels];
        extract_palette(&data.rgb, &ones, &params.palette)
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
        ("bg_mask", &pr.bg_mask_morph),
        ("fg_confidence", &pr.fg_confidence),
        ("fused", &fused),
    ];

    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
    make_contact_sheet(
        &data.rgb, &feature_slices, &[],
        w, h, params.output.contact_sheet_cols, params.output.contact_sheet_thumb_w,
        &palette, &ts,
        &out_dir.join("contact_sheet.png"),
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
