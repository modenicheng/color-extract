// =============================================================================
// Segment-aware region priors — color-segment 集成层
// =============================================================================

use anyhow::{Context, Result};
use color_segment::{Region, SegmentResult, segment};
use image::{Rgb, RgbImage};
use serde::Serialize;

use crate::background::BackgroundFeatureInputs;
use crate::params::{
    ImpressionBackgroundParams, RegionDynamicWeightsParams, RegionScoringParams,
    SegmentFusionParams,
};

#[derive(Debug, Clone, Serialize)]
pub struct RegionFeature {
    pub id: usize,
    pub cluster_id: usize,
    pub area: usize,
    pub area_ratio: f64,
    pub border_ratio: f64,
    pub center_prior: f64,
    pub saliency_mean: f64,
    pub saliency_peak: f64,
    pub edge_mean: f64,
    pub border_color_similarity: f64,
    pub color_stability: f64,
    pub mean_saturation: f64,
    pub red_green_affinity: f64,
    pub background_penalty: f64,
    pub vivid_rg_bonus: f64,
    pub bg_probability: f64,
    pub subject_confidence: f64,
    pub color_score: f64,
    pub mean_rgb: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
pub struct RegionColorResult {
    pub rgb: [f64; 3],
    pub score: f64,
    pub contributing_regions: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentDiagnostics {
    pub width: u32,
    pub height: u32,
    pub region_count: usize,
    pub region_color: RegionColorResult,
    pub regions: Vec<RegionFeature>,
}

#[derive(Debug, Clone)]
pub struct SegmentFeatureResult {
    pub labels: Vec<Option<usize>>,
    pub regions: Vec<RegionFeature>,
    pub region_id_map: Vec<f64>,
    pub boundary_map: Vec<f64>,
    pub bg_probability: Vec<f64>,
    pub saliency: Vec<f64>,
    pub subject_confidence: Vec<f64>,
    pub foreground: Vec<f64>,
    pub region_color: RegionColorResult,
}

pub struct RegionWeightContext<'a> {
    pub labels: &'a [Option<usize>],
    pub region_count: usize,
    pub saliency: &'a [f64],
    pub bg_probability: &'a [f64],
}

impl SegmentFeatureResult {
    pub fn weight_context(&self) -> RegionWeightContext<'_> {
        RegionWeightContext {
            labels: &self.labels,
            region_count: self.regions.len(),
            saliency: &self.saliency,
            bg_probability: &self.bg_probability,
        }
    }

    pub fn diagnostics(&self, width: u32, height: u32) -> SegmentDiagnostics {
        SegmentDiagnostics {
            width,
            height,
            region_count: self.regions.len(),
            region_color: self.region_color.clone(),
            regions: self.regions.clone(),
        }
    }
}

pub fn compute_segment_features(
    rgb: &[[f64; 3]],
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: u32,
    h: u32,
    feature_inputs: &BackgroundFeatureInputs<'_>,
    subject_prior: &[f64],
    cfg: &SegmentFusionParams,
) -> Result<SegmentFeatureResult> {
    let img = rgb_to_image(rgb, w, h)?;
    let segmented = segment(&img, &cfg.segment).context("running color-segment")?;
    anyhow::ensure!(
        segmented.width == w && segmented.height == h,
        "segment result size {}x{} does not match feature-fuse size {}x{}",
        segmented.width,
        segmented.height,
        w,
        h
    );

    let saliency = build_segment_saliency(feature_inputs, &cfg.region_scoring);
    let boundary_map = build_boundary_map(&segmented);
    let mut regions = compute_region_stats(
        rgb,
        lab_l,
        lab_a,
        lab_b,
        &saliency,
        subject_prior,
        &segmented,
        &cfg.background_bias,
        &cfg.region_scoring,
    );
    let bg_probability = labels_to_region_map(&segmented.labels, &regions, |r| r.bg_probability);
    let subject_confidence =
        labels_to_region_map(&segmented.labels, &regions, |r| r.subject_confidence);
    let foreground = build_segment_foreground(&bg_probability, &subject_confidence);
    let region_id_map = build_region_id_map(&segmented.labels, regions.len());
    compute_region_color_scores(&mut regions, &cfg.region_scoring);
    let region_color = region_color_from_scores(&regions);

    Ok(SegmentFeatureResult {
        labels: segmented.labels,
        regions,
        region_id_map,
        boundary_map,
        bg_probability,
        saliency,
        subject_confidence,
        foreground,
        region_color,
    })
}

pub fn apply_segment_foreground_prior(
    foreground: &[f64],
    segment: &SegmentFeatureResult,
    cfg: &RegionScoringParams,
) -> Vec<f64> {
    let strength = cfg.background_correction_strength.clamp(0.0, 1.0);
    foreground
        .iter()
        .enumerate()
        .map(|(i, &fg)| {
            let seg_fg = segment.foreground.get(i).copied().unwrap_or(0.0);
            (fg * (1.0 - strength) + seg_fg * strength).clamp(0.0, 1.0)
        })
        .collect()
}

/// 为最终颜色聚类生成背景氛围权重。
///
/// 这个权重不改变背景/前景判定本身，只让“大面积、稳定、有明确色调且与整体画面协调”的背景
/// 在最终印象色聚类中保留一部分发言权。
pub fn compute_background_impression_weight(
    segment: Option<&SegmentFeatureResult>,
    cfg: &ImpressionBackgroundParams,
    n_pixels: usize,
) -> Vec<f64> {
    if !cfg.enabled || cfg.strength <= 0.0 || n_pixels == 0 {
        return vec![0.0; n_pixels];
    }
    let Some(segment) = segment else {
        return vec![0.0; n_pixels];
    };
    if segment.regions.is_empty() {
        return vec![0.0; n_pixels];
    }

    let global_tone = weighted_region_tone(&segment.regions, |region| region.area as f64)
        .unwrap_or([0.0, 0.0, 0.0]);
    let bg_tone = weighted_region_tone(&segment.regions, |region| {
        region.area as f64 * region.bg_probability.clamp(0.0, 1.0)
    })
    .unwrap_or(global_tone);

    let component_total = cfg.colorfulness_weight.max(0.0)
        + cfg.tone_harmony_weight.max(0.0)
        + cfg.stability_weight.max(0.0);
    let max_weight = cfg.max_weight.clamp(0.0, 1.0);
    let strength = cfg.strength.max(0.0);
    let effective_bg_areas = effective_background_areas(&segment.regions);

    let region_weights: Vec<f64> = segment
        .regions
        .iter()
        .enumerate()
        .map(|(idx, region)| {
            let effective_area = region
                .area_ratio
                .max(effective_bg_areas.get(idx).copied().unwrap_or(0.0));
            let area_gate = smoothstep(
                cfg.min_area_ratio.clamp(0.0, 1.0),
                cfg.full_area_ratio.clamp(0.0, 1.0),
                effective_area,
            );
            let colorfulness = if cfg.min_saturation <= 1e-9 {
                1.0
            } else {
                smoothstep(0.0, cfg.min_saturation.clamp(1e-6, 1.0), region.mean_saturation)
            };
            let tone_harmony = rgb_tone_affinity(region.mean_rgb, global_tone)
                .max(rgb_tone_affinity(region.mean_rgb, bg_tone));
            let stability = region.color_stability.clamp(0.0, 1.0);
            let quality = if component_total <= 1e-12 {
                1.0
            } else {
                (colorfulness * cfg.colorfulness_weight.max(0.0)
                    + tone_harmony * cfg.tone_harmony_weight.max(0.0)
                    + stability * cfg.stability_weight.max(0.0))
                    / component_total
            };

            let non_subject = (1.0 - region.subject_confidence).clamp(0.0, 1.0);
            (strength
                * region.bg_probability.clamp(0.0, 1.0)
                * area_gate
                * quality.clamp(0.0, 1.0)
                * non_subject)
                .clamp(0.0, max_weight)
        })
        .collect();

    let mut out: Vec<f64> = segment
        .labels
        .iter()
        .take(n_pixels)
        .map(|label| match label {
            Some(rid) if *rid < region_weights.len() => region_weights[*rid],
            _ => 0.0,
        })
        .collect();
    out.resize(n_pixels, 0.0);
    out
}

/// 计算大面积黑/白/灰背景的聚类权重惩罚。
///
/// 只在「像背景 + 低饱和 + 近黑/近白 + 相似背景累计面积大」同时成立时生效，
/// 用于避免白底/黑底图片的最终主色被画布本身吞掉。
pub fn compute_neutral_background_suppression(
    segment: Option<&SegmentFeatureResult>,
    lab_l: &[f64],
    cfg: &ImpressionBackgroundParams,
    n_pixels: usize,
) -> Vec<f64> {
    if !cfg.neutral_suppression_enabled
        || cfg.neutral_suppression_strength <= 0.0
        || n_pixels == 0
    {
        return vec![0.0; n_pixels];
    }
    let Some(segment) = segment else {
        return vec![0.0; n_pixels];
    };
    if segment.regions.is_empty() {
        return vec![0.0; n_pixels];
    }

    let effective_bg_areas = effective_background_areas(&segment.regions);
    let strength = cfg.neutral_suppression_strength.clamp(0.0, 1.0);
    let sat_threshold = cfg.neutral_sat_threshold.clamp(1e-6, 1.0);
    let bg_threshold = cfg.neutral_bg_probability_threshold.clamp(0.0, 1.0);
    let bg_full = (bg_threshold + 0.25).min(1.0);
    let sat_fade_end = (sat_threshold + 0.10).min(1.0);
    let region_penalties: Vec<f64> = segment
        .regions
        .iter()
        .enumerate()
        .map(|(idx, region)| {
            let effective_area = region
                .area_ratio
                .max(effective_bg_areas.get(idx).copied().unwrap_or(0.0));
            let area_gate = smoothstep(
                cfg.neutral_min_effective_area_ratio.clamp(0.0, 1.0),
                cfg.neutral_full_effective_area_ratio.clamp(0.0, 1.0),
                effective_area,
            );
            let bg_gate = smoothstep(bg_threshold, bg_full, region.bg_probability);
            let low_sat = 1.0 - smoothstep(sat_threshold, sat_fade_end, region.mean_saturation);
            let non_subject = smoothstep(0.20, 0.55, 1.0 - region.subject_confidence);
            strength * area_gate * bg_gate * low_sat * non_subject
        })
        .collect();

    // ── 诊断：region 级 penalty 分解 ──
    {
        let mut nonzero_penalties: Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> = Vec::new();
        for (idx, region) in segment.regions.iter().enumerate() {
            let rp = region_penalties[idx];
            if rp > 0.001 {
                let effective_area = region
                    .area_ratio
                    .max(effective_bg_areas.get(idx).copied().unwrap_or(0.0));
                let area_gate = smoothstep(
                    cfg.neutral_min_effective_area_ratio.clamp(0.0, 1.0),
                    cfg.neutral_full_effective_area_ratio.clamp(0.0, 1.0),
                    effective_area,
                );
                let bg_gate = smoothstep(bg_threshold, bg_full, region.bg_probability);
                let low_sat = 1.0 - smoothstep(sat_threshold, sat_fade_end, region.mean_saturation);
                let non_subject = smoothstep(0.20, 0.55, 1.0 - region.subject_confidence);
                nonzero_penalties.push((
                    idx, region.area_ratio, effective_area, area_gate,
                    region.bg_probability, bg_gate,
                    region.mean_saturation, low_sat,
                    region.subject_confidence, non_subject, rp,
                ));
            }
        }
        nonzero_penalties.sort_by(|a, b| b.10.partial_cmp(&a.10).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "  [diag] region_penalties>0.001: {}/{} regions",
            nonzero_penalties.len(),
            segment.regions.len(),
        );
        for (idx, area, eff_area, a_gate, bg_prob, b_gate, sat, l_sat, subj, n_subj, rp)
            in nonzero_penalties.iter().take(12)
        {
            eprintln!(
                "    r[{idx:>3}]: area={area:.4} eff={eff_area:.4} a_gate={a_gate:.4} | bg={bg_prob:.4} b_gate={b_gate:.4} | sat={sat:.4} l_sat={l_sat:.4} | subj={subj:.4} n_subj={n_subj:.4} → penalty={rp:.4}",
            );
        }
    }

    let softness = cfg.neutral_light_softness.max(1e-6);
    let black_edge0 = cfg.neutral_black_l_threshold;
    let black_edge1 = cfg.neutral_black_l_threshold + softness;
    let white_edge0 = cfg.neutral_white_l_threshold - softness;
    let white_edge1 = cfg.neutral_white_l_threshold;

    let mut out: Vec<f64> = segment
        .labels
        .iter()
        .take(n_pixels)
        .enumerate()
        .map(|(idx, label)| {
            let region_penalty = match label {
                Some(rid) if *rid < region_penalties.len() => region_penalties[*rid],
                _ => 0.0,
            };
            if region_penalty <= 0.0 {
                return 0.0;
            }
            let l = lab_l.get(idx).copied().unwrap_or(50.0);
            let black = 1.0 - smoothstep(black_edge0, black_edge1, l);
            let white = smoothstep(white_edge0, white_edge1, l);
            let extreme_light = black.max(white).clamp(0.0, 1.0);
            (region_penalty * extreme_light).clamp(0.0, 1.0)
        })
        .collect();
    out.resize(n_pixels, 0.0);
    out
}

fn build_segment_foreground(bg_probability: &[f64], subject_confidence: &[f64]) -> Vec<f64> {
    bg_probability
        .iter()
        .enumerate()
        .map(|(i, &bg)| {
            let subj = subject_confidence.get(i).copied().unwrap_or(0.0);
            ((1.0 - bg) * 0.65 + subj * 0.35).clamp(0.0, 1.0)
        })
        .collect()
}

pub fn region_dynamic_score_for_feature(
    feature: &[f64],
    ctx: &RegionWeightContext<'_>,
    cfg: &RegionDynamicWeightsParams,
) -> f64 {
    if feature.is_empty() || ctx.region_count == 0 {
        return 0.0;
    }

    let separation = region_separation_score(feature, ctx.labels, ctx.region_count);
    let corr = positive_correlation(feature, ctx.saliency);
    let bg_suppression = foreground_background_separation(feature, ctx.bg_probability);
    let total = cfg.separation_weight + cfg.saliency_corr_weight + cfg.bg_suppression_weight;
    if total <= 1e-12 {
        return 0.0;
    }
    ((separation * cfg.separation_weight
        + corr * cfg.saliency_corr_weight
        + bg_suppression * cfg.bg_suppression_weight)
        / total)
        .clamp(0.0, 1.0)
}

fn rgb_to_image(rgb: &[[f64; 3]], w: u32, h: u32) -> Result<RgbImage> {
    anyhow::ensure!(
        rgb.len() == (w * h) as usize,
        "rgb buffer length {} does not match {}x{}",
        rgb.len(),
        w,
        h
    );
    Ok(RgbImage::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        Rgb([
            (rgb[i][0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (rgb[i][1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (rgb[i][2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ])
    }))
}

fn build_segment_saliency(
    features: &BackgroundFeatureInputs<'_>,
    cfg: &RegionScoringParams,
) -> Vec<f64> {
    let n = features.lab_grad.len();
    let total = cfg.saliency_dct_weight
        + cfg.saliency_lab_grad_weight
        + cfg.saliency_spectral_weight
        + cfg.saliency_local_light_weight
        + cfg.saliency_local_sat_weight;
    if total <= 1e-12 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| {
            (features.dct.get(i).copied().unwrap_or(0.0) * cfg.saliency_dct_weight
                + features.lab_grad.get(i).copied().unwrap_or(0.0) * cfg.saliency_lab_grad_weight
                + features.spectral.get(i).copied().unwrap_or(0.0) * cfg.saliency_spectral_weight
                + features.local_light.get(i).copied().unwrap_or(0.0)
                    * cfg.saliency_local_light_weight
                + features.local_sat.get(i).copied().unwrap_or(0.0) * cfg.saliency_local_sat_weight)
                / total
        })
        .map(|v| v.clamp(0.0, 1.0))
        .collect()
}

fn build_boundary_map(segmented: &SegmentResult) -> Vec<f64> {
    let w = segmented.width as usize;
    let h = segmented.height as usize;
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let cur = segmented.labels[i];
            let mut boundary = 0.0_f64;
            if x > 0 && segmented.labels[i - 1] != cur {
                boundary = 1.0;
            }
            if x + 1 < w && segmented.labels[i + 1] != cur {
                boundary = 1.0;
            }
            if y > 0 && segmented.labels[i - w] != cur {
                boundary = 1.0;
            }
            if y + 1 < h && segmented.labels[i + w] != cur {
                boundary = 1.0;
            }
            out[i] = boundary.max(segmented.edge_map.get(i).copied().unwrap_or(0.0) * 0.65);
        }
    }
    out
}

fn compute_region_stats(
    rgb: &[[f64; 3]],
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    saliency: &[f64],
    subject_prior: &[f64],
    segmented: &SegmentResult,
    background_bias: &str,
    cfg: &RegionScoringParams,
) -> Vec<RegionFeature> {
    let region_count = segmented.regions.len();
    let n = segmented.labels.len();
    let w = segmented.width as usize;
    let h = segmented.height as usize;
    let mut sums = vec![Accum::default(); region_count];
    let border_band = (cfg.border_band as usize)
        .max(1)
        .min(w.max(1))
        .min(h.max(1));
    let border_center = border_lab_center(lab_l, lab_a, lab_b, w, h, border_band);

    for i in 0..n {
        let Some(rid) = segmented.labels[i] else {
            continue;
        };
        if rid >= region_count {
            continue;
        }
        let x = i % w;
        let y = i / w;
        let on_border =
            x < border_band || y < border_band || x + border_band >= w || y + border_band >= h;
        sums[rid].add(
            rgb[i],
            lab_l[i],
            lab_a[i],
            lab_b[i],
            saliency.get(i).copied().unwrap_or(0.0),
            segmented.edge_map.get(i).copied().unwrap_or(0.0),
            subject_prior.get(i).copied().unwrap_or(0.0),
            on_border,
        );
    }

    segmented
        .regions
        .iter()
        .map(|region| build_region_feature(region, &sums, n, border_center, background_bias, cfg))
        .collect()
}

fn build_region_feature(
    region: &Region,
    sums: &[Accum],
    total_pixels: usize,
    border_center: [f64; 3],
    background_bias: &str,
    cfg: &RegionScoringParams,
) -> RegionFeature {
    let acc = sums.get(region.id).copied().unwrap_or_default();
    let area = acc.count.max(region.area).max(1);
    let area_f = area as f64;
    let mean_l = acc.sum_l / area_f;
    let mean_a = acc.sum_a / area_f;
    let mean_b = acc.sum_b_lab / area_f;
    let var_l = (acc.sum_l2 / area_f - mean_l * mean_l).max(0.0);
    let var_a = (acc.sum_a2 / area_f - mean_a * mean_a).max(0.0);
    let var_b = (acc.sum_b2 / area_f - mean_b * mean_b).max(0.0);
    let lab_std = ((var_l / 10000.0) + (var_a / 16384.0) + (var_b / 16384.0)).sqrt();
    let color_stability = (1.0 - lab_std / 0.18).clamp(0.0, 1.0);

    let d_l = (mean_l - border_center[0]) / 100.0;
    let d_a = (mean_a - border_center[1]) / 128.0;
    let d_b = (mean_b - border_center[2]) / 128.0;
    let border_dist = (d_l * d_l + d_a * d_a + d_b * d_b).sqrt();
    let border_color_similarity = (-(border_dist / 0.18).powi(2)).exp().clamp(0.0, 1.0);

    let area_ratio = area_f / total_pixels.max(1) as f64;
    let border_ratio = acc.border_count as f64 / area_f;
    let saliency_mean = acc.sum_saliency / area_f;
    let saliency_peak = acc.max_saliency;
    let edge_mean = acc.sum_edge / area_f;
    let center_prior = acc.sum_center / area_f;
    let subject_raw = weighted3(
        center_prior,
        cfg.subject_center_weight,
        (saliency_mean * 0.65 + saliency_peak * 0.35).clamp(0.0, 1.0),
        cfg.subject_saliency_weight,
        edge_mean,
        cfg.subject_edge_weight,
    );
    let bg_raw = weighted5(
        border_ratio,
        cfg.bg_border_weight,
        border_color_similarity,
        cfg.bg_color_weight,
        1.0 - saliency_mean,
        cfg.bg_low_saliency_weight,
        1.0 - center_prior,
        cfg.bg_low_center_weight,
        1.0 - edge_mean,
        cfg.bg_low_edge_weight,
    );

    let bias_scale = match background_bias {
        "subject" => 1.20,
        "background" => 0.82,
        _ => 1.0,
    };
    let protected_bg = (bg_raw
        * (1.0 - cfg.subject_protect_strength.clamp(0.0, 1.0) * subject_raw * bias_scale))
        .clamp(0.0, 1.0);
    let bg_probability = if background_bias == "background" {
        (protected_bg * 0.85 + bg_raw * 0.15).clamp(0.0, 1.0)
    } else {
        protected_bg
    };
    let subject_confidence = ((1.0 - bg_probability) * 0.55 + subject_raw * 0.45).clamp(0.0, 1.0);

    let mean_rgb = [acc.sum_r / area_f, acc.sum_g / area_f, acc.sum_b / area_f];
    let (mean_hue, mean_saturation) = rgb_to_hsl_hue_sat(mean_rgb);
    let red_green_affinity = red_green_hue_affinity(mean_hue, cfg.vivid_rg_hue_width);

    RegionFeature {
        id: region.id,
        cluster_id: region.cluster_id,
        area,
        area_ratio,
        border_ratio,
        center_prior,
        saliency_mean,
        saliency_peak,
        edge_mean,
        border_color_similarity,
        color_stability,
        mean_saturation,
        red_green_affinity,
        background_penalty: 0.0,
        vivid_rg_bonus: 0.0,
        bg_probability,
        subject_confidence,
        color_score: 0.0,
        mean_rgb,
    }
}

fn compute_region_color_scores(regions: &mut [RegionFeature], cfg: &RegionScoringParams) {
    for region in regions {
        let area_factor = region.area_ratio.max(1e-9).powf(cfg.area_power.max(0.05));
        let saliency = (region.saliency_mean * 0.70 + region.saliency_peak * 0.30).clamp(0.0, 1.0);
        let large_area = smoothstep(
            cfg.bg_flat_area_min_ratio,
            cfg.bg_flat_area_full_ratio,
            region.area_ratio,
        );
        let low_sat =
            (1.0 - region.mean_saturation / cfg.bg_flat_sat_threshold.max(1e-6)).clamp(0.0, 1.0);
        let high_sat = ((region.mean_saturation - cfg.vivid_rg_sat_threshold)
            / (1.0 - cfg.vivid_rg_sat_threshold).max(1e-6))
        .clamp(0.0, 1.0);
        let background_penalty = (large_area * region.color_stability * low_sat).clamp(0.0, 1.0);
        let vivid_rg_bonus = (large_area * high_sat * region.red_green_affinity).clamp(0.0, 1.0);
        let base = area_factor
            * region.subject_confidence
            * saliency.max(0.05)
            * (1.0 - region.bg_probability).clamp(0.0, 1.0);
        let stability_factor =
            1.0 - cfg.color_stability_weight + cfg.color_stability_weight * region.color_stability;
        let penalty_factor =
            1.0 - cfg.bg_flat_penalty_strength.clamp(0.0, 1.0) * background_penalty;
        let bonus_factor = 1.0 + cfg.vivid_rg_bonus_strength.max(0.0) * vivid_rg_bonus;

        region.background_penalty = background_penalty;
        region.vivid_rg_bonus = vivid_rg_bonus;
        region.color_score =
            (base * stability_factor * penalty_factor * bonus_factor).max(0.0);
    }
}

fn rgb_to_hsl_hue_sat(rgb: [f64; 3]) -> (f64, f64) {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    if d <= 1e-12 {
        return (0.0, 0.0);
    }

    let l = (max + min) * 0.5;
    let s = d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-12);
    let h = if (max - r).abs() < 1e-12 {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if (max - g).abs() < 1e-12 {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h.rem_euclid(360.0), s.clamp(0.0, 1.0))
}

fn red_green_hue_affinity(hue: f64, width_degrees: f64) -> f64 {
    let width = width_degrees.max(1.0);
    let red = circular_hue_affinity(hue, 0.0, width);
    let green = circular_hue_affinity(hue, 120.0, width);
    red.max(green)
}

fn circular_hue_affinity(hue: f64, center: f64, width: f64) -> f64 {
    let d = (hue - center).rem_euclid(360.0);
    let dist = d.min(360.0 - d);
    (1.0 - dist / width).clamp(0.0, 1.0)
}

fn weighted_region_tone<F>(regions: &[RegionFeature], weight: F) -> Option<[f64; 3]>
where
    F: Fn(&RegionFeature) -> f64,
{
    let mut sum = [0.0; 3];
    let mut total = 0.0;
    for region in regions {
        let w = weight(region).max(0.0);
        if w <= 0.0 {
            continue;
        }
        sum[0] += region.mean_rgb[0] * w;
        sum[1] += region.mean_rgb[1] * w;
        sum[2] += region.mean_rgb[2] * w;
        total += w;
    }
    if total <= 1e-12 {
        None
    } else {
        Some([sum[0] / total, sum[1] / total, sum[2] / total])
    }
}

fn effective_background_areas(regions: &[RegionFeature]) -> Vec<f64> {
    regions
        .iter()
        .map(|region| {
            regions
                .iter()
                .map(|other| {
                    other.area_ratio
                        * other.bg_probability.clamp(0.0, 1.0)
                        * rgb_tone_affinity(region.mean_rgb, other.mean_rgb)
                })
                .sum::<f64>()
                .clamp(0.0, 1.0)
        })
        .collect()
}

fn rgb_tone_affinity(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    let dist = ((dr * dr + dg * dg + db * db) / 3.0).sqrt();
    (-(dist / 0.32).powi(2)).exp().clamp(0.0, 1.0)
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn region_color_from_scores(regions: &[RegionFeature]) -> RegionColorResult {
    let mut ranked: Vec<&RegionFeature> = regions.iter().filter(|r| r.color_score > 0.0).collect();
    ranked.sort_by(|a, b| {
        b.color_score
            .partial_cmp(&a.color_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(region) = ranked.first() {
        return RegionColorResult {
            rgb: region.mean_rgb,
            score: region.color_score,
            contributing_regions: vec![region.id],
        };
    }

    RegionColorResult {
        rgb: [0.0; 3],
        score: 0.0,
        contributing_regions: Vec::new(),
    }
}

fn labels_to_region_map<F>(
    labels: &[Option<usize>],
    regions: &[RegionFeature],
    value: F,
) -> Vec<f64>
where
    F: Fn(&RegionFeature) -> f64,
{
    labels
        .iter()
        .map(|label| match label {
            Some(rid) if *rid < regions.len() => value(&regions[*rid]).clamp(0.0, 1.0),
            _ => 0.0,
        })
        .collect()
}

fn build_region_id_map(labels: &[Option<usize>], region_count: usize) -> Vec<f64> {
    if region_count <= 1 {
        return vec![0.0; labels.len()];
    }
    let denom = (region_count - 1) as f64;
    labels
        .iter()
        .map(|label| label.map(|rid| rid as f64 / denom).unwrap_or(0.0))
        .collect()
}

fn border_lab_center(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: usize,
    h: usize,
    band: usize,
) -> [f64; 3] {
    let mut l = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if x < band || y < band || x + band >= w || y + band >= h {
                let i = y * w + x;
                l.push(lab_l[i]);
                a.push(lab_a[i]);
                b.push(lab_b[i]);
            }
        }
    }
    [median(&mut l), median(&mut a), median(&mut b)]
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn region_separation_score(feature: &[f64], labels: &[Option<usize>], region_count: usize) -> f64 {
    let mut sums = vec![0.0; region_count];
    let mut counts = vec![0usize; region_count];
    let mut global_sum = 0.0;
    let mut global_count = 0usize;
    for (i, &v) in feature.iter().enumerate() {
        global_sum += v;
        global_count += 1;
        if let Some(rid) = labels.get(i).copied().flatten() {
            if rid < region_count {
                sums[rid] += v;
                counts[rid] += 1;
            }
        }
    }
    if global_count == 0 {
        return 0.0;
    }
    let global_mean = global_sum / global_count as f64;
    let mut between = 0.0;
    let mut total = 0.0;
    for (&sum, &count) in sums.iter().zip(counts.iter()) {
        if count == 0 {
            continue;
        }
        let mean = sum / count as f64;
        let d = mean - global_mean;
        between += d * d * count as f64;
    }
    for &v in feature {
        let d = v - global_mean;
        total += d * d;
    }
    if total <= 1e-12 {
        0.0
    } else {
        (between / total).clamp(0.0, 1.0)
    }
}

fn positive_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va <= 1e-12 || vb <= 1e-12 {
        return 0.0;
    }
    (cov / (va.sqrt() * vb.sqrt())).max(0.0).clamp(0.0, 1.0)
}

fn foreground_background_separation(feature: &[f64], bg_probability: &[f64]) -> f64 {
    if feature.len() != bg_probability.len() || feature.is_empty() {
        return 0.0;
    }
    let mut fg_sum = 0.0;
    let mut fg_w = 0.0;
    let mut bg_sum = 0.0;
    let mut bg_w = 0.0;
    for (&v, &bg) in feature.iter().zip(bg_probability.iter()) {
        let b = bg.clamp(0.0, 1.0);
        let f = 1.0 - b;
        fg_sum += v * f;
        fg_w += f;
        bg_sum += v * b;
        bg_w += b;
    }
    if fg_w <= 1e-12 || bg_w <= 1e-12 {
        return 0.0;
    }
    let fg_mean = fg_sum / fg_w;
    let bg_mean = bg_sum / bg_w;
    ((fg_mean - bg_mean) / 0.45).clamp(0.0, 1.0)
}

#[derive(Default, Debug, Clone, Copy)]
struct Accum {
    count: usize,
    border_count: usize,
    sum_r: f64,
    sum_g: f64,
    sum_b: f64,
    sum_l: f64,
    sum_a: f64,
    sum_b_lab: f64,
    sum_l2: f64,
    sum_a2: f64,
    sum_b2: f64,
    sum_saliency: f64,
    max_saliency: f64,
    sum_edge: f64,
    sum_center: f64,
}

impl Accum {
    fn add(
        &mut self,
        rgb: [f64; 3],
        l: f64,
        a: f64,
        b: f64,
        saliency: f64,
        edge: f64,
        center: f64,
        on_border: bool,
    ) {
        self.count += 1;
        self.border_count += usize::from(on_border);
        self.sum_r += rgb[0];
        self.sum_g += rgb[1];
        self.sum_b += rgb[2];
        self.sum_l += l;
        self.sum_a += a;
        self.sum_b_lab += b;
        self.sum_l2 += l * l;
        self.sum_a2 += a * a;
        self.sum_b2 += b * b;
        self.sum_saliency += saliency;
        self.max_saliency = self.max_saliency.max(saliency);
        self.sum_edge += edge;
        self.sum_center += center;
    }
}

fn weighted3(a: f64, aw: f64, b: f64, bw: f64, c: f64, cw: f64) -> f64 {
    let total = aw.max(0.0) + bw.max(0.0) + cw.max(0.0);
    if total <= 1e-12 {
        return 0.0;
    }
    ((a * aw.max(0.0) + b * bw.max(0.0) + c * cw.max(0.0)) / total).clamp(0.0, 1.0)
}

fn weighted5(
    a: f64,
    aw: f64,
    b: f64,
    bw: f64,
    c: f64,
    cw: f64,
    d: f64,
    dw: f64,
    e: f64,
    ew: f64,
) -> f64 {
    let total = aw.max(0.0) + bw.max(0.0) + cw.max(0.0) + dw.max(0.0) + ew.max(0.0);
    if total <= 1e-12 {
        return 0.0;
    }
    ((a * aw.max(0.0) + b * bw.max(0.0) + c * cw.max(0.0) + d * dw.max(0.0) + e * ew.max(0.0))
        / total)
        .clamp(0.0, 1.0)
}
