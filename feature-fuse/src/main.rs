// =============================================================================
// feature-fuse — 全链路特征图计算 + Hybrid Fusion
// =============================================================================

mod background;
mod dct;
mod dynamic_weights;
mod fusion;
mod gradient;
mod html;
mod image;
mod params;
mod render;
mod residual;
mod segment_features;
mod spectral;

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use crate::background::{
    BackgroundFeatureInputs, compute_background_features, compute_subject_prior,
};
use crate::dct::compute_dct_complexity;
use crate::dynamic_weights::{compute_dynamic_weights, diagnostics_to_json, diagnostics_to_table};
use crate::fusion::{
    apply_filter, composite_with_hybrid, composite_with_hybrid_direct, hybrid_fusion,
    kmeans_impression_color, kmeans_weighted_color, percentile_normalize,
    percentile_normalize_unit_feature, weights_to_array,
};
use crate::gradient::compute_lab_gradient;
use crate::image::load_image;
use crate::params::{Params, load_params, validate_filter};
use crate::residual::{
    compute_global_lab_a_residual, compute_global_lab_b_residual, compute_global_light_residual,
    compute_global_sat_residual, compute_local_lab_a_residual, compute_local_lab_b_residual,
    compute_local_light_residual, compute_local_sat_residual,
};
use crate::segment_features::{
    SegmentFeatureResult, apply_segment_foreground_prior, compute_segment_features,
};
use crate::spectral::compute_spectral_residual;

use crate::render::{make_contact_sheet_full, save_gray_png, save_rgb_png};

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
    let results: Vec<Result<(String, String, String)>> = image_paths
        .par_iter()
        .map(|path| process_one_image(path, &params, max_dim, &out_base))
        .collect();

    // ── 汇总 ──
    let mut success = 0;
    let mut errors = Vec::new();
    let mut entries: Vec<html::ImageEntry> = Vec::new();
    for r in results {
        match r {
            Ok((stem, ic_hex, rc_hex)) => {
                println!("  ✓ {stem}");
                entries.push(html::ImageEntry {
                    stem,
                    ic_hex,
                    rc_hex,
                });
                success += 1;
            }
            Err(e) => {
                errors.push(e);
            }
        }
    }

    // ── 生成 HTML 总览页 ──
    if !entries.is_empty() {
        let html_path = out_base.join("all.html");
        html::generate_overview(&entries, &html_path)?;
        println!("  HTML overview: {}", html_path.display());
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
fn process_one_image(
    path: &Path,
    params: &Params,
    max_dim: u32,
    out_base: &Path,
) -> Result<(String, String, String)> {
    // ── 加载图片 ──
    let data = load_image(path, max_dim)?;
    let stem = data.stem.clone();
    let w = data.w;
    let h = data.h;

    // 创建输出目录
    let out_dir = out_base.join(&stem);
    std::fs::create_dir_all(&out_dir)?;

    println!("  {} ({}×{}) — 19 features …", stem, w, h);

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
    let dct_raw = compute_dct_complexity(
        &gray,
        w as usize,
        h as usize,
        params.dct.high_freq_threshold,
    );
    let dct_norm = percentile_normalize(&dct_raw, p_low, p_high);
    save_gray_png(&dct_norm, w, h, &out_dir.join("dct_complexity.png"))?;
    let t_dct = t0.elapsed();

    // ── (2) LAB 梯度 ──
    let t0 = std::time::Instant::now();
    let lab_grad_raw = compute_lab_gradient(
        &data.lab_l,
        &data.lab_a,
        &data.lab_b,
        w as usize,
        h as usize,
    );
    let lab_grad_norm = percentile_normalize(&lab_grad_raw, p_low, p_high);
    save_gray_png(&lab_grad_norm, w, h, &out_dir.join("lab_gradient.png"))?;
    let t_lab = t0.elapsed();

    // ── (3) 频谱残差显著性 ──
    let t0 = std::time::Instant::now();
    let sr_raw = compute_spectral_residual(
        &data.lab_l,
        &data.lab_a,
        &data.lab_b,
        w as usize,
        h as usize,
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

    // ── (5a) Global LAB a* residual (红-绿轴稳健中心) ──
    let t0 = std::time::Instant::now();
    let gl_a_raw = compute_global_lab_a_residual(&data.lab_a, &params.global_residual.lab_a);
    let gl_a_norm = percentile_normalize(&gl_a_raw, p_low, p_high);
    save_gray_png(&gl_a_norm, w, h, &out_dir.join("global_lab_a_residual.png"))?;
    let t_ga = t0.elapsed();

    // ── (5b) Global LAB b* residual (黄-蓝轴稳健中心) ──
    let t0 = std::time::Instant::now();
    let gl_b_raw = compute_global_lab_b_residual(&data.lab_b, &params.global_residual.lab_b);
    let gl_b_norm = percentile_normalize(&gl_b_raw, p_low, p_high);
    save_gray_png(&gl_b_norm, w, h, &out_dir.join("global_lab_b_residual.png"))?;
    let t_gb = t0.elapsed();

    // ── (5c) Global HSL 饱和度 residual (稳健中心) ──
    let t0 = std::time::Instant::now();
    let gl_sat_raw = compute_global_sat_residual(&data.hsl_s, &params.global_residual.sat);
    let gl_sat_norm = percentile_normalize(&gl_sat_raw, p_low, p_high);
    save_gray_png(&gl_sat_norm, w, h, &out_dir.join("global_sat_residual.png"))?;
    let t_gs = t0.elapsed();

    // ── (6) Local (Gaussian) light residual ──
    let t0 = std::time::Instant::now();
    let loc_l_raw = compute_local_light_residual(&data.hsl_l, w, h, params.gauss_sigma);
    let loc_l_norm = percentile_normalize(&loc_l_raw, p_low, p_high);
    save_gray_png(&loc_l_norm, w, h, &out_dir.join("local_light_residual.png"))?;
    let t_ll = t0.elapsed();

    // ── (7a) Local (Gaussian) LAB a* residual ──
    let t0 = std::time::Instant::now();
    let loc_a_raw = compute_local_lab_a_residual(&data.lab_a, w, h, params.gauss_sigma);
    let loc_a_norm = percentile_normalize(&loc_a_raw, p_low, p_high);
    save_gray_png(&loc_a_norm, w, h, &out_dir.join("local_lab_a_residual.png"))?;
    let t_la = t0.elapsed();

    // ── (7b) Local (Gaussian) LAB b* residual ──
    let t0 = std::time::Instant::now();
    let loc_b_raw = compute_local_lab_b_residual(&data.lab_b, w, h, params.gauss_sigma);
    let loc_b_norm = percentile_normalize(&loc_b_raw, p_low, p_high);
    save_gray_png(&loc_b_norm, w, h, &out_dir.join("local_lab_b_residual.png"))?;
    let t_lb = t0.elapsed();

    // ── (7c) Local (Gaussian) HSL 饱和度 residual (含 post-gamma 后处理) ──
    let t0 = std::time::Instant::now();
    let loc_sat_gamma = compute_local_sat_residual(
        &data.hsl_s,
        w,
        h,
        params.gauss_sigma,
        params.saturation.local_post_gamma,
    );
    let loc_sat_norm = percentile_normalize(&loc_sat_gamma, p_low, p_high);
    save_gray_png(&loc_sat_norm, w, h, &out_dir.join("local_sat_residual.png"))?;
    let t_ls = t0.elapsed();

    // ── 构建背景与区域先验共用的局部色度保护代理 ──
    let local_lab_ab: Vec<f64> = loc_a_norm
        .iter()
        .zip(loc_b_norm.iter())
        .map(|(&a, &b)| a.max(b))
        .collect();
    let bg_feature_inputs = BackgroundFeatureInputs {
        dct: &dct_norm,
        lab_grad: &lab_grad_norm,
        spectral: &sr_norm,
        local_light: &loc_l_norm,
        local_sat: &local_lab_ab,
    };

    // ── (8) Subject Prior（提前计算，供 segment-aware 区域统计复用） ──
    let t0 = std::time::Instant::now();
    let sp_raw = compute_subject_prior(w, h, &params.subject_prior);
    let sp_norm = percentile_normalize(&sp_raw, p_low, p_high);
    save_gray_png(&sp_norm, w, h, &out_dir.join("subject_prior.png"))?;
    let t_sp = t0.elapsed();

    // ── (9) Segment-aware region priors ──
    let t0 = std::time::Instant::now();
    let segment_result: Option<SegmentFeatureResult> = if params.segment_fusion.enabled {
        let seg = compute_segment_features(
            &data.rgb,
            &data.lab_l,
            &data.lab_a,
            &data.lab_b,
            w,
            h,
            &bg_feature_inputs,
            &sp_norm,
            &params.segment_fusion,
        )?;
        if params.segment_fusion.diagnostics {
            save_gray_png(
                &seg.region_id_map,
                w,
                h,
                &out_dir.join("segment_region_id.png"),
            )?;
            save_gray_png(
                &seg.boundary_map,
                w,
                h,
                &out_dir.join("segment_boundary.png"),
            )?;
            save_gray_png(
                &seg.bg_probability,
                w,
                h,
                &out_dir.join("segment_bg_probability.png"),
            )?;
            save_gray_png(&seg.saliency, w, h, &out_dir.join("segment_saliency.png"))?;
            save_gray_png(
                &seg.subject_confidence,
                w,
                h,
                &out_dir.join("segment_subject_confidence.png"),
            )?;
            save_gray_png(
                &seg.foreground,
                w,
                h,
                &out_dir.join("segment_foreground.png"),
            )?;
            let json = serde_json::to_string_pretty(&seg.diagnostics(w, h))
                .context("serialize segment diagnostics")?;
            std::fs::write(out_dir.join("segment_diagnostics.json"), json)
                .context("write segment_diagnostics.json")?;
        }
        Some(seg)
    } else {
        None
    };
    let t_seg = t0.elapsed();

    let t0 = std::time::Instant::now();
    let bg_result = compute_background_features(
        &data.lab_l,
        &data.lab_a,
        &data.lab_b,
        w,
        h,
        &bg_feature_inputs,
        &params.background,
    );
    let (bg_mask_raw, bg_fg_raw) = if let Some(ref seg) = segment_result {
        (
            apply_segment_foreground_prior(
                &bg_result.bg_mask_morph,
                seg,
                &params.segment_fusion.region_scoring,
            ),
            apply_segment_foreground_prior(
                &bg_result.fg_confidence,
                seg,
                &params.segment_fusion.region_scoring,
            ),
        )
    } else {
        (
            bg_result.bg_mask_morph.clone(),
            bg_result.fg_confidence.clone(),
        )
    };
    let bg_mask_norm = percentile_normalize_unit_feature(&bg_mask_raw, p_low, p_high);
    let bg_fg_norm = percentile_normalize_unit_feature(&bg_fg_raw, p_low, p_high);
    save_gray_png(
        &bg_mask_norm,
        w,
        h,
        &out_dir.join("background_mask_morph.png"),
    )?;
    save_gray_png(
        &bg_fg_norm,
        w,
        h,
        &out_dir.join("background_fg_confidence.png"),
    )?;
    save_gray_png(
        &bg_result.diagnostics.bg_candidate,
        w,
        h,
        &out_dir.join("bg_candidate.png"),
    )?;
    save_gray_png(
        &bg_result.diagnostics.bg_barrier,
        w,
        h,
        &out_dir.join("bg_barrier.png"),
    )?;
    save_gray_png(
        &bg_result.diagnostics.bg_mask_before_protect,
        w,
        h,
        &out_dir.join("bg_mask_before_protect.png"),
    )?;
    save_gray_png(
        &bg_result.diagnostics.foreground_protect,
        w,
        h,
        &out_dir.join("foreground_protect.png"),
    )?;
    save_gray_png(
        &bg_result.diagnostics.bg_mask_after_protect,
        w,
        h,
        &out_dir.join("bg_mask_after_protect.png"),
    )?;
    save_gray_png(
        &bg_result.diagnostics.fg_confidence,
        w,
        h,
        &out_dir.join("fg_confidence.png"),
    )?;
    let t_bg = t0.elapsed();

    // ── (11a) Absolute Light (L* raw channel) ──
    let t0 = std::time::Instant::now();
    let abs_l_norm = percentile_normalize(&data.lab_l, p_low, p_high);
    save_gray_png(&abs_l_norm, w, h, &out_dir.join("abs_light.png"))?;
    let t_al = t0.elapsed();

    // ── (11b) Absolute LAB a* (red-green raw channel) ──
    let t0 = std::time::Instant::now();
    let abs_a_raw: Vec<f64> = data.lab_a.iter().map(|&v| (v + 128.0) / 256.0).collect();
    let abs_a_norm = percentile_normalize(&abs_a_raw, p_low, p_high);
    save_gray_png(&abs_a_norm, w, h, &out_dir.join("abs_lab_a.png"))?;
    let t_aa = t0.elapsed();

    // ── (11c) Absolute LAB b* (yellow-blue raw channel) ──
    let t0 = std::time::Instant::now();
    let abs_b_raw: Vec<f64> = data.lab_b.iter().map(|&v| (v + 128.0) / 256.0).collect();
    let abs_b_norm = percentile_normalize(&abs_b_raw, p_low, p_high);
    save_gray_png(&abs_b_norm, w, h, &out_dir.join("abs_lab_b.png"))?;
    let t_ab = t0.elapsed();

    // ── (11d) Absolute HSL Saturation (raw channel) ──
    let t0 = std::time::Instant::now();
    let abs_sat_norm = percentile_normalize(&data.hsl_s, p_low, p_high);
    save_gray_png(&abs_sat_norm, w, h, &out_dir.join("abs_sat.png"))?;
    let t_as = t0.elapsed();

    // ── (11e) Segment Foreground (color-segment 派生的第 19D 前景概率) ──
    let segment_fg_norm: Vec<f64> = segment_result
        .as_ref()
        .map(|seg| seg.foreground.iter().map(|&v| v.clamp(0.0, 1.0)).collect())
        .unwrap_or_else(|| vec![0.0; (w * h) as usize]);

    println!(
        "    DCT={:.1}s LAB={:.1}s SR={:.1}s GL={:.1}s GA={:.1}s GB={:.1}s GS={:.1}s LL={:.1}s LA={:.1}s LB={:.1}s LS={:.1}s SEG={:.1}s BG={:.1}s SP={:.1}s AL={:.1}s AA={:.1}s AB={:.1}s AS={:.1}s — fusion …",
        t_dct.as_secs_f64(),
        t_lab.as_secs_f64(),
        t_sr.as_secs_f64(),
        t_gl.as_secs_f64(),
        t_ga.as_secs_f64(),
        t_gb.as_secs_f64(),
        t_gs.as_secs_f64(),
        t_ll.as_secs_f64(),
        t_la.as_secs_f64(),
        t_lb.as_secs_f64(),
        t_ls.as_secs_f64(),
        t_seg.as_secs_f64(),
        t_bg.as_secs_f64(),
        t_sp.as_secs_f64(),
        t_al.as_secs_f64(),
        t_aa.as_secs_f64(),
        t_ab.as_secs_f64(),
        t_as.as_secs_f64(),
    );

    // ── 归一化后的所有特征 ──
    let features: [&[f64]; 19] = [
        &dct_norm,
        &lab_grad_norm,
        &sr_norm,
        &gl_l_norm,
        &gl_a_norm,
        &gl_b_norm,
        &gl_sat_norm,
        &loc_l_norm,
        &loc_a_norm,
        &loc_b_norm,
        &loc_sat_norm,
        &bg_mask_norm,
        &bg_fg_norm,
        &sp_norm,
        &abs_l_norm,
        &abs_a_norm,
        &abs_b_norm,
        &abs_sat_norm,
        &segment_fg_norm,
    ];
    let feature_names: [&str; 19] = [
        "dct_complexity",
        "lab_gradient",
        "spectral_residual",
        "global_light_residual",
        "global_lab_a_residual",
        "global_lab_b_residual",
        "global_sat_residual",
        "local_light_residual",
        "local_lab_a_residual",
        "local_lab_b_residual",
        "local_sat_residual",
        "background_mask_morph",
        "background_fg_confidence",
        "subject_prior",
        "abs_light",
        "abs_lab_a",
        "abs_lab_b",
        "abs_sat",
        "segment_foreground",
    ];

    // ── 从 YAML 提取 base weight 数组 ──
    let base_add = weights_to_array(&params.weights_add);
    let base_mul = weights_to_array(&params.weights_mul);

    // ── 计算动态权重（若 dynamic_weights.enabled == true）──
    let (add_w, mul_w) = if params.dynamic_weights.enabled {
        let segment_weight_ctx = segment_result.as_ref().map(|seg| seg.weight_context());
        let result = compute_dynamic_weights(
            &features,
            &base_add,
            &base_mul,
            &params.dynamic_weights,
            segment_weight_ctx.as_ref(),
            Some(&params.segment_fusion.dynamic_weights),
        );
        // 输出诊断表格到控制台
        let table = diagnostics_to_table(&result.diagnostics);
        println!("\n  ── Dynamic Weights ──\n{table}\n");
        // 保存 JSON 诊断文件
        let json_str = diagnostics_to_json(&result.diagnostics);
        std::fs::write(out_dir.join("dynamic_weights.json"), &json_str)
            .context("write dynamic_weights.json")?;
        (result.add_weights, result.mul_weights)
    } else {
        (base_add.clone(), base_mul.clone())
    };

    // ── Hybrid Fusion ──
    let (fused_add, fused_mul, fused_hybrid) = hybrid_fusion(
        &features,
        &add_w,
        &mul_w,
        params.fusion.alpha,
        params.fusion.gamma,
        params.fusion.epsilon,
    );

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
        save_gray_png(
            &fused_add_filt,
            w,
            h,
            &out_dir.join("fused_add_filtered.png"),
        )?;
        save_gray_png(
            &fused_mul_filt,
            w,
            h,
            &out_dir.join("fused_softmul_filtered.png"),
        )?;
        save_gray_png(
            &fused_hyb_filt,
            w,
            h,
            &out_dir.join("fused_hybrid_filtered.png"),
        )?;
    }

    // ── Original × FiltHybrid 复合图 ──
    let composite_rgb = composite_with_hybrid(&data.rgb, &fused_hyb_filt);
    save_rgb_png(
        &composite_rgb,
        w,
        h,
        &out_dir.join("fused_original_hybrid.png"),
    )?;

    // ── Original × Hybrid（无阈值，直接乘）──
    let composite_nothresh = composite_with_hybrid_direct(
        &data.rgb,
        &fused_hybrid,
        params.direct_blend.normalize_before,
    );
    save_rgb_png(
        &composite_nothresh,
        w,
        h,
        &out_dir.join("fused_original_hybrid_nothreshold.png"),
    )?;

    // ── 加权聚类色：对原图做加权 k-means，FiltHybrid 为权重 ──
    let (weighted_color, wc_score) = kmeans_weighted_color(
        &data.rgb,
        &fused_hyb_filt,
        w as usize,
        h as usize,
        &params.impression,
        false,
    );
    let wc_hex = format!(
        "#{:02x}{:02x}{:02x}",
        (weighted_color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (weighted_color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (weighted_color[2].clamp(0.0, 1.0) * 255.0) as u8,
    );
    println!("    weighted cluster: {wc_hex} (score={wc_score:.0})");

    // ── 印象色：k-means++ 聚类，取过滤权重总和最高的簇 ──
    let impression_color = kmeans_impression_color(
        &data.rgb,
        &fused_hyb_filt,
        w as usize,
        h as usize,
        &params.impression,
    );
    let k = params.impression.k;
    let max_iter = params.impression.max_iter;
    let ic_hex = format!(
        "#{:02x}{:02x}{:02x}",
        (impression_color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (impression_color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (impression_color[2].clamp(0.0, 1.0) * 255.0) as u8,
    );
    println!("    impression color: {ic_hex} (k-means cluster, k={k}, iter={max_iter})");

    // ── 区域感知主色：按 segment region 得分聚合 ──
    let (region_color, region_score) = if let Some(ref seg) = segment_result {
        (seg.region_color.rgb, seg.region_color.score)
    } else {
        (weighted_color, 0.0)
    };
    let rc_hex = format!(
        "#{:02x}{:02x}{:02x}",
        (region_color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (region_color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (region_color[2].clamp(0.0, 1.0) * 255.0) as u8,
    );
    if segment_result.is_some() {
        println!("    region color: {rc_hex} (segment-aware score={region_score:.4})");
    } else {
        println!("    region color: {rc_hex} (segment fusion disabled; fallback=weighted)");
    }

    // ── Contact Sheet ──
    let mut feat_slices: Vec<(&str, &[f64])> = feature_names
        .iter()
        .zip(features.iter())
        .map(|(&n, &f)| (n, f))
        .collect();
    feat_slices.extend_from_slice(&[
        ("bg_candidate", &bg_result.diagnostics.bg_candidate),
        ("bg_barrier", &bg_result.diagnostics.bg_barrier),
        (
            "bg_mask_before_protect",
            &bg_result.diagnostics.bg_mask_before_protect,
        ),
        (
            "foreground_protect",
            &bg_result.diagnostics.foreground_protect,
        ),
        (
            "bg_mask_after_protect",
            &bg_result.diagnostics.bg_mask_after_protect,
        ),
        ("fg_confidence", &bg_result.diagnostics.fg_confidence),
    ]);
    if let Some(ref seg) = segment_result {
        feat_slices.extend_from_slice(&[
            ("segment_region_id", &seg.region_id_map),
            ("segment_boundary", &seg.boundary_map),
            ("segment_bg_probability", &seg.bg_probability),
            ("segment_saliency", &seg.saliency),
            ("segment_subject_confidence", &seg.subject_confidence),
        ]);
    }
    let fused_slices: [(&str, &[f64]); 6] = [
        ("fused_add", &fused_add),
        ("fused_softmul", &fused_mul),
        ("fused_hybrid", &fused_hybrid),
        ("fused_add_filtered", &fused_add_filt),
        ("fused_softmul_filtered", &fused_mul_filt),
        ("fused_hybrid_filtered", &fused_hyb_filt),
    ];

    let ts = chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S%.3f")
        .to_string();

    make_contact_sheet_full(
        &data.rgb,
        &feat_slices,
        &fused_slices,
        &[
            ("fused_original_hybrid", composite_rgb.as_slice()),
            (
                "fused_original_hybrid_nothreshold",
                composite_nothresh.as_slice(),
            ),
        ],
        Some(impression_color),
        Some(weighted_color),
        Some(region_color),
        w,
        h,
        &params.contact_sheet,
        &ts,
        &out_base.join(format!("contact_sheet_{stem}.png")),
    )?;

    println!("    ✓ {stem} — all outputs in {}/", out_dir.display());

    Ok((stem, ic_hex, rc_hex))
}
