// =============================================================================
// color-segment/anime.rs — Fast anime-style block segmentation
// =============================================================================
//
// 面向二次元/插画图的快速色块分割：
//   1. 一次扫描提取 RGB/YCoCg/LAB 特征
//   2. 用 LAB Sobel + gamma 压缩生成边缘权重
//   3. 以强边缘为墙，按颜色距离做扫描式区域生长
//   4. 在区域邻接图上合并相近色块，并吸收抗锯齿/压缩产生的小碎片
//
// 这条路径避免了全图 Median Cut 的多轮排序，也不会先把渐变/抗锯齿切碎。

use crate::SegmentResult;
use crate::params::SegmentParams;
use crate::quantize::Palette;
use crate::region::Region;
use image::RgbImage;
use palette::{IntoColor, Lab, Srgb};

#[derive(Debug, Clone, Copy)]
struct PixelFeature {
    r: u8,
    g: u8,
    b: u8,
    y: i16,
    co: i16,
    cg: i16,
    lab_l: f32,
    lab_a: f32,
    lab_b: f32,
}

impl PixelFeature {
    fn from_rgb(rgb: [u8; 3]) -> Self {
        let r = rgb[0] as i16;
        let g = rgb[1] as i16;
        let b = rgb[2] as i16;
        let srgb = Srgb::new(
            rgb[0] as f32 / 255.0,
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
        );
        let lab: Lab = srgb.into_color();
        Self {
            r: rgb[0],
            g: rgb[1],
            b: rgb[2],
            y: (r + 2 * g + b) / 4,
            co: r - b,
            cg: 2 * g - r - b,
            lab_l: lab.l,
            lab_a: lab.a,
            lab_b: lab.b,
        }
    }
}

#[derive(Debug, Clone)]
struct RegionStats {
    id: usize,
    area: usize,
    sum_x: u64,
    sum_y: u64,
    sum_r: u64,
    sum_g: u64,
    sum_b: u64,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

impl RegionStats {
    fn new(id: usize, width: u32, height: u32) -> Self {
        Self {
            id,
            area: 0,
            sum_x: 0,
            sum_y: 0,
            sum_r: 0,
            sum_g: 0,
            sum_b: 0,
            min_x: width,
            min_y: height,
            max_x: 0,
            max_y: 0,
        }
    }

    fn add_pixel(&mut self, idx: usize, width: usize, px: PixelFeature) {
        let x = (idx % width) as u32;
        let y = (idx / width) as u32;
        self.area += 1;
        self.sum_x += x as u64;
        self.sum_y += y as u64;
        self.sum_r += px.r as u64;
        self.sum_g += px.g as u64;
        self.sum_b += px.b as u64;
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    fn mean_rgb(&self) -> [u8; 3] {
        if self.area == 0 {
            return [0, 0, 0];
        }
        let n = self.area as u64;
        [
            ((self.sum_r + n / 2) / n) as u8,
            ((self.sum_g + n / 2) / n) as u8,
            ((self.sum_b + n / 2) / n) as u8,
        ]
    }

    fn mean_rgb_f64(&self) -> [f64; 3] {
        if self.area == 0 {
            return [0.0, 0.0, 0.0];
        }
        let n = self.area as f64;
        [
            self.sum_r as f64 / n,
            self.sum_g as f64 / n,
            self.sum_b as f64 / n,
        ]
    }
}

#[derive(Debug, Clone, Copy)]
struct AdjacentStats {
    a: usize,
    b: usize,
    sum_edge: f64,
    sum_color: f64,
    len: usize,
}

struct Uf {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Uf {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut cur = x;
        while self.parent[cur] != cur {
            self.parent[cur] = self.parent[self.parent[cur]];
            cur = self.parent[cur];
        }
        cur
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }

    fn resolve(&mut self) -> (Vec<usize>, usize) {
        let n = self.parent.len();
        for i in 0..n {
            self.find(i);
        }

        let mut root_to_new = vec![usize::MAX; n];
        let mut next = 0usize;
        let mut map = vec![0usize; n];
        for (i, slot) in map.iter_mut().enumerate() {
            let root = self.parent[i];
            if root_to_new[root] == usize::MAX {
                root_to_new[root] = next;
                next += 1;
            }
            *slot = root_to_new[root];
        }
        (map, next)
    }
}

// =============================================================================
// Public entry
// =============================================================================

pub fn segment_anime_blocks(
    img: &RgbImage,
    params: &SegmentParams,
) -> anyhow::Result<SegmentResult> {
    let profile = std::env::var_os("COLOR_SEGMENT_PROFILE").is_some();
    let started = std::time::Instant::now();
    let (width, height) = img.dimensions();
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    anyhow::ensure!(total > 0, "image must have at least 1 pixel");

    let features = extract_features(img);
    let feature_ms = started.elapsed().as_secs_f64() * 1000.0;
    let edge_map = detect_edges_fast(&features, width, height, params);
    let edge_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (raw_labels, raw_regions) = grow_regions(&features, &edge_map, width, height, params);
    let grow_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (regions, labels, palette) = merge_regions(
        &features,
        &edge_map,
        &raw_labels,
        &raw_regions,
        width,
        height,
        params,
    );
    if profile {
        eprintln!(
            "profile: feature={:.1}ms edge={:.1}ms grow={:.1}ms merge={:.1}ms raw_regions={} final_regions={}",
            feature_ms,
            edge_ms - feature_ms,
            grow_ms - edge_ms,
            started.elapsed().as_secs_f64() * 1000.0 - grow_ms,
            raw_regions.len(),
            regions.len()
        );
    }

    Ok(SegmentResult {
        regions,
        labels,
        palette,
        edge_map,
        width,
        height,
    })
}

// =============================================================================
// Feature + edge
// =============================================================================

fn extract_features(img: &RgbImage) -> Vec<PixelFeature> {
    img.pixels().map(|p| PixelFeature::from_rgb(p.0)).collect()
}

fn detect_edges_fast(
    features: &[PixelFeature],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    if w < 3 || h < 3 {
        return vec![1.0; total];
    }

    let mut raw = vec![0.0f64; total];
    let mut max_v = 1.0f64;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let p00 = features[idx - w - 1];
            let p01 = features[idx - w];
            let p02 = features[idx - w + 1];
            let p10 = features[idx - 1];
            let p12 = features[idx + 1];
            let p20 = features[idx + w - 1];
            let p21 = features[idx + w];
            let p22 = features[idx + w + 1];

            let gx_l =
                -p00.lab_l - 2.0 * p10.lab_l - p20.lab_l + p02.lab_l + 2.0 * p12.lab_l + p22.lab_l;
            let gy_l =
                -p00.lab_l - 2.0 * p01.lab_l - p02.lab_l + p20.lab_l + 2.0 * p21.lab_l + p22.lab_l;
            let gx_a =
                -p00.lab_a - 2.0 * p10.lab_a - p20.lab_a + p02.lab_a + 2.0 * p12.lab_a + p22.lab_a;
            let gy_a =
                -p00.lab_a - 2.0 * p01.lab_a - p02.lab_a + p20.lab_a + 2.0 * p21.lab_a + p22.lab_a;
            let gx_b =
                -p00.lab_b - 2.0 * p10.lab_b - p20.lab_b + p02.lab_b + 2.0 * p12.lab_b + p22.lab_b;
            let gy_b =
                -p00.lab_b - 2.0 * p01.lab_b - p02.lab_b + p20.lab_b + 2.0 * p21.lab_b + p22.lab_b;

            let l_energy = (gx_l.abs() + gy_l.abs()) as f64;
            let a_energy = (gx_a.abs() + gy_a.abs()) as f64;
            let b_energy = (gx_b.abs() + gy_b.abs()) as f64;
            let v = l_energy + a_energy * 0.95 + b_energy * 0.95;
            raw[idx] = v;
            max_v = max_v.max(v);
        }
    }

    for x in 0..w {
        raw[x] = max_v;
        raw[(h - 1) * w + x] = max_v;
    }
    for y in 0..h {
        raw[y * w] = max_v;
        raw[y * w + w - 1] = max_v;
    }

    let cutoff = (params.edge_threshold.clamp(0.0, 1.0) * max_v).max(1.0);
    let gamma = params.edge_gamma.clamp(0.2, 3.0);
    raw.into_iter()
        .map(|v| {
            let edge = if v <= cutoff {
                0.0
            } else {
                ((v - cutoff) / (max_v - cutoff).max(1.0)).clamp(0.0, 1.0)
            };
            if gamma == 1.0 { edge } else { edge.powf(gamma) }
        })
        .collect()
}

// =============================================================================
// Region growing
// =============================================================================

fn grow_regions(
    features: &[PixelFeature],
    edge_map: &[f64],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Option<usize>>, Vec<RegionStats>) {
    let w = width as usize;
    let h = height as usize;
    let total = w * h;
    let mut uf = Uf::new(total);
    let local_limit = grow_color_limit(params) * 1.3;
    let local_limit_sq = local_limit * local_limit;
    let edge_wall = (params.edge_split_strength * 0.68).clamp(0.45, 0.85);

    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let idx = row + x;
            if x > 0
                && can_connect_neighbors(
                    idx,
                    idx - 1,
                    features,
                    edge_map,
                    local_limit_sq,
                    edge_wall,
                )
            {
                uf.union(idx, idx - 1);
            }
            if y > 0
                && can_connect_neighbors(
                    idx,
                    idx - w,
                    features,
                    edge_map,
                    local_limit_sq,
                    edge_wall,
                )
            {
                uf.union(idx, idx - w);
            }
        }
    }

    let (pixel_to_region, num_regions) = uf.resolve();
    let mut regions: Vec<RegionStats> = (0..num_regions)
        .map(|id| RegionStats::new(id, width, height))
        .collect();
    let mut labels = vec![None; total];
    for idx in 0..total {
        let rid = pixel_to_region[idx];
        labels[idx] = Some(rid);
        regions[rid].add_pixel(idx, w, features[idx]);
    }
    (labels, regions)
}

fn can_connect_neighbors(
    idx: usize,
    nidx: usize,
    features: &[PixelFeature],
    edge_map: &[f64],
    local_limit_sq: f64,
    edge_wall: f64,
) -> bool {
    let current = features[idx];
    let next = features[nidx];
    let sobel_hint = (edge_map[idx] + edge_map[nidx]) * 0.175;
    if is_strong_neighbor_boundary(current, next, edge_wall) || sobel_hint >= edge_wall {
        return false;
    }

    rgb_dist_sq_u8(current, next) <= local_limit_sq
}

fn grow_color_limit(params: &SegmentParams) -> f64 {
    // `color_merge_distance` 原本按 LAB ΔE 理解；快速路径使用 RGB 空间，
    // 这里映射到一个对抗锯齿友好但不会跨越明确色块的半径。
    (params.color_merge_distance * 1.75 + 4.0).clamp(12.0, 30.0)
}

// =============================================================================
// Merge + small-region absorption
// =============================================================================

fn merge_regions(
    features: &[PixelFeature],
    edge_map: &[f64],
    raw_labels: &[Option<usize>],
    raw_regions: &[RegionStats],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> (Vec<Region>, Vec<Option<usize>>, Palette) {
    if raw_regions.is_empty() {
        return (
            Vec::new(),
            vec![None; (width as usize) * (height as usize)],
            Palette {
                colors: Vec::new(),
                counts: Vec::new(),
            },
        );
    }

    let pairs = compute_adjacency(raw_labels, features, edge_map, width, height);
    let mut uf = Uf::new(raw_regions.len());

    let merge_limit = (params.color_merge_distance * 1.9 + 5.0).clamp(14.0, 32.0);
    let split_guard = params.edge_split_strength.clamp(0.25, 0.98);

    for pair in &pairs {
        let mean_edge = pair.sum_edge / pair.len as f64;
        let mean_color = pair.sum_color / pair.len as f64;
        if mean_edge <= params.edge_merge_strength
            || (mean_color <= merge_limit && mean_edge < split_guard * 0.45)
        {
            uf.union(pair.a, pair.b);
        }
    }

    if params.merge_small_regions {
        absorb_small_regions(&mut uf, raw_regions, &pairs, width, height, params);
    }

    let (old_to_new, num_final) = uf.resolve();
    let final_labels: Vec<Option<usize>> = raw_labels
        .iter()
        .map(|label| label.map(|rid| old_to_new[rid]))
        .collect();
    let final_stats = rebuild_stats(features, &final_labels, num_final, width, height);
    let (regions, palette) = stats_to_output(final_stats);

    (regions, final_labels, palette)
}

fn absorb_small_regions(
    uf: &mut Uf,
    regions: &[RegionStats],
    pairs: &[AdjacentStats],
    width: u32,
    height: u32,
    params: &SegmentParams,
) {
    let total = (width as usize) * (height as usize);
    let ratio_area = (total as f64 * params.min_cluster_area_ratio).round() as usize;
    let min_area = params.min_region_area.max(ratio_area).max(1);
    let tiny_area = (min_area / 4).max(2);
    let mut ids: Vec<usize> = regions
        .iter()
        .filter(|r| r.area < min_area)
        .map(|r| r.id)
        .collect();
    ids.sort_by_key(|&id| regions[id].area);

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); regions.len()];
    for (idx, pair) in pairs.iter().enumerate() {
        adjacency[pair.a].push(idx);
        adjacency[pair.b].push(idx);
    }

    for rid in ids {
        let root = uf.find(rid);
        let mut best: Option<(usize, f64)> = None;

        for &pair_idx in &adjacency[rid] {
            let pair = &pairs[pair_idx];
            let other = if pair.a == rid { pair.b } else { pair.a };

            if uf.find(other) == root {
                continue;
            }

            let mean_edge = pair.sum_edge / pair.len as f64;
            let mean_color = rgb_mean_distance(&regions[rid], &regions[other]);
            let very_tiny = regions[rid].area <= tiny_area;

            if !very_tiny
                && mean_edge >= params.edge_split_strength
                && mean_color > params.small_region_color_distance * 1.6
            {
                continue;
            }

            let boundary_bonus = (pair.len as f64).sqrt() * 3.0;
            let area_bonus = ((regions[other].area as f64).ln_1p()).min(12.0);
            let score = mean_color + mean_edge * 35.0 - boundary_bonus - area_bonus;
            match best {
                Some((_, best_score)) if best_score <= score => {}
                _ => best = Some((other, score)),
            }
        }

        if let Some((target, _)) = best {
            uf.union(rid, target);
        }
    }
}

fn compute_adjacency(
    labels: &[Option<usize>],
    features: &[PixelFeature],
    edge_map: &[f64],
    width: u32,
    height: u32,
) -> Vec<AdjacentStats> {
    let w = width as usize;
    let h = height as usize;
    let mut samples: Vec<(u64, f64, f64)> = Vec::with_capacity(w * h / 4);

    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let idx = row + x;
            let Some(a) = labels[idx] else {
                continue;
            };
            if x + 1 < w {
                push_adj_sample(&mut samples, labels, features, edge_map, idx, idx + 1, a);
            }
            if y + 1 < h {
                push_adj_sample(&mut samples, labels, features, edge_map, idx, idx + w, a);
            }
        }
    }

    if samples.is_empty() {
        return Vec::new();
    }

    samples.sort_unstable_by_key(|sample| sample.0);

    let mut pairs = Vec::new();
    let mut current_key = samples[0].0;
    let mut sum_edge = 0.0;
    let mut sum_color = 0.0;
    let mut len = 0usize;

    for (key, edge, color) in samples {
        if key != current_key {
            let (a, b) = unpack_pair_key(current_key);
            pairs.push(AdjacentStats {
                a,
                b,
                sum_edge,
                sum_color,
                len,
            });
            current_key = key;
            sum_edge = 0.0;
            sum_color = 0.0;
            len = 0;
        }

        sum_edge += edge;
        sum_color += color;
        len += 1;
    }

    let (a, b) = unpack_pair_key(current_key);
    pairs.push(AdjacentStats {
        a,
        b,
        sum_edge,
        sum_color,
        len,
    });
    pairs
}

fn push_adj_sample(
    samples: &mut Vec<(u64, f64, f64)>,
    labels: &[Option<usize>],
    features: &[PixelFeature],
    edge_map: &[f64],
    idx: usize,
    nidx: usize,
    a: usize,
) {
    let Some(b) = labels[nidx] else {
        return;
    };
    if a == b {
        return;
    }

    let key = if a < b {
        pack_pair_key(a, b)
    } else {
        pack_pair_key(b, a)
    };
    let edge = direct_edge_strength(features[idx], features[nidx])
        .max((edge_map[idx] + edge_map[nidx]) * 0.5);
    let color = rgb_dist_sq_u8(features[idx], features[nidx]).sqrt();
    samples.push((key, edge, color));
}

fn pack_pair_key(a: usize, b: usize) -> u64 {
    ((a as u64) << 32) | b as u64
}

fn unpack_pair_key(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, (key & 0xffff_ffff) as usize)
}

fn rebuild_stats(
    features: &[PixelFeature],
    labels: &[Option<usize>],
    num_regions: usize,
    width: u32,
    height: u32,
) -> Vec<RegionStats> {
    let w = width as usize;
    let mut stats: Vec<RegionStats> = (0..num_regions)
        .map(|id| RegionStats::new(id, width, height))
        .collect();
    for (idx, label) in labels.iter().enumerate() {
        if let Some(rid) = label {
            stats[*rid].add_pixel(idx, w, features[idx]);
        }
    }
    stats.into_iter().filter(|r| r.area > 0).collect()
}

fn stats_to_output(stats: Vec<RegionStats>) -> (Vec<Region>, Palette) {
    let mut regions = Vec::with_capacity(stats.len());
    let mut colors = Vec::with_capacity(stats.len());
    let mut counts = Vec::with_capacity(stats.len());

    for (new_id, stat) in stats.into_iter().enumerate() {
        let color = stat.mean_rgb();
        colors.push(color);
        counts.push(stat.area);
        let area_f = stat.area as f64;
        regions.push(Region {
            id: new_id,
            cluster_id: new_id,
            area: stat.area,
            pixel_count: stat.area,
            centroid: (stat.sum_x as f64 / area_f, stat.sum_y as f64 / area_f),
            bbox: (stat.min_x, stat.min_y, stat.max_x, stat.max_y),
        });
    }

    (regions, Palette { colors, counts })
}

// =============================================================================
// Distance helpers
// =============================================================================

fn rgb_dist_sq_u8(a: PixelFeature, b: PixelFeature) -> f64 {
    let dr = a.r as f64 - b.r as f64;
    let dg = a.g as f64 - b.g as f64;
    let db = a.b as f64 - b.b as f64;
    dr * dr + dg * dg + db * db
}

fn rgb_mean_distance(a: &RegionStats, b: &RegionStats) -> f64 {
    let ar = a.mean_rgb_f64();
    let br = b.mean_rgb_f64();
    let dr = ar[0] - br[0];
    let dg = ar[1] - br[1];
    let db = ar[2] - br[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

fn direct_edge_strength(a: PixelFeature, b: PixelFeature) -> f64 {
    let rgb = rgb_dist_sq_u8(a, b).sqrt() / 72.0;
    let dy = (a.y - b.y).abs() as f64 / 42.0;
    let chroma = ((a.co - b.co).abs() as f64 + (a.cg - b.cg).abs() as f64) / 180.0;
    rgb.max(dy).max(chroma).clamp(0.0, 1.0)
}

fn is_strong_neighbor_boundary(a: PixelFeature, b: PixelFeature, edge_wall: f64) -> bool {
    let rgb_limit = edge_wall * 72.0;
    if rgb_dist_sq_u8(a, b) >= rgb_limit * rgb_limit {
        return true;
    }

    let y_limit = (edge_wall * 42.0) as i16;
    if (a.y - b.y).abs() >= y_limit {
        return true;
    }

    let chroma_limit = (edge_wall * 180.0) as i16;
    (a.co - b.co).abs() + (a.cg - b.cg).abs() >= chroma_limit
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn keeps_flat_blocks_connected() {
        let mut img = RgbImage::new(12, 6);
        for y in 0..6 {
            for x in 0..12 {
                let color = if x < 6 { [240, 80, 90] } else { [70, 110, 230] };
                img.put_pixel(x, y, Rgb(color));
            }
        }

        let params = SegmentParams {
            min_region_area: 1,
            ..SegmentParams::default()
        };
        let result = segment_anime_blocks(&img, &params).expect("segment");
        assert_eq!(result.regions.len(), 2);
        assert!(result.labels.iter().all(|l| l.is_some()));
    }

    #[test]
    fn absorbs_single_pixel_noise() {
        let mut img = RgbImage::new(10, 10);
        for p in img.pixels_mut() {
            *p = Rgb([220, 220, 230]);
        }
        img.put_pixel(5, 5, Rgb([225, 219, 231]));

        let params = SegmentParams {
            min_region_area: 8,
            ..SegmentParams::default()
        };
        let result = segment_anime_blocks(&img, &params).expect("segment");
        assert_eq!(result.regions.len(), 1);
    }
}
