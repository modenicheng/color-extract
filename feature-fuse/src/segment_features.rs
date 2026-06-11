// =============================================================================
// Segment-aware region priors — color-segment 集成层
// =============================================================================

use anyhow::{Context, Result};
use color_segment::{Region, SegmentResult, segment};
use image::{Rgb, RgbImage};
use serde::Serialize;

use crate::background::BackgroundFeatureInputs;
use crate::params::{RegionDynamicWeightsParams, RegionScoringParams, SegmentFusionParams};

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
    let region_id_map = build_region_id_map(&segmented.labels, regions.len());
    compute_region_color_scores(&mut regions, &cfg.region_scoring);
    let region_color = region_color_from_scores(&regions, &cfg.region_scoring);

    Ok(SegmentFeatureResult {
        labels: segmented.labels,
        regions,
        region_id_map,
        boundary_map,
        bg_probability,
        saliency,
        subject_confidence,
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
            let seg_fg = (1.0 - segment.bg_probability.get(i).copied().unwrap_or(1.0)) * 0.65
                + segment.subject_confidence.get(i).copied().unwrap_or(0.0) * 0.35;
            (fg * (1.0 - strength) + seg_fg * strength).clamp(0.0, 1.0)
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
        bg_probability,
        subject_confidence,
        color_score: 0.0,
        mean_rgb: [acc.sum_r / area_f, acc.sum_g / area_f, acc.sum_b / area_f],
    }
}

fn compute_region_color_scores(regions: &mut [RegionFeature], cfg: &RegionScoringParams) {
    for region in regions {
        let area_factor = region.area_ratio.max(1e-9).powf(cfg.area_power.max(0.05));
        let saliency = (region.saliency_mean * 0.70 + region.saliency_peak * 0.30).clamp(0.0, 1.0);
        let base = area_factor
            * region.subject_confidence
            * saliency.max(0.05)
            * (1.0 - region.bg_probability).clamp(0.0, 1.0);
        region.color_score = (base
            * (1.0 - cfg.color_stability_weight
                + cfg.color_stability_weight * region.color_stability))
            .max(0.0);
    }
}

fn region_color_from_scores(
    regions: &[RegionFeature],
    cfg: &RegionScoringParams,
) -> RegionColorResult {
    let mut ranked: Vec<&RegionFeature> = regions.iter().filter(|r| r.color_score > 0.0).collect();
    ranked.sort_by(|a, b| {
        b.color_score
            .partial_cmp(&a.color_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_k = cfg.region_color_top_k.max(1);
    let mut rgb = [0.0; 3];
    let mut score_sum = 0.0;
    let mut ids = Vec::new();
    for r in ranked.into_iter().take(top_k) {
        rgb[0] += r.mean_rgb[0] * r.color_score;
        rgb[1] += r.mean_rgb[1] * r.color_score;
        rgb[2] += r.mean_rgb[2] * r.color_score;
        score_sum += r.color_score;
        ids.push(r.id);
    }

    if score_sum > 0.0 {
        rgb[0] /= score_sum;
        rgb[1] /= score_sum;
        rgb[2] /= score_sum;
    }

    RegionColorResult {
        rgb,
        score: score_sum,
        contributing_regions: ids,
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
