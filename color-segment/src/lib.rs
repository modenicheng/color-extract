// color-segment — Color-based image region segmentation library
// Target: anime-style (二次元) images

pub mod edge;
pub mod params;
pub mod quantize;
pub mod refine;
pub mod region;

// ===== Re-exports =====

pub use image;
pub use params::{ColorSpace, SegmentParams};
pub use quantize::{Palette, Pixel as QPixel};
pub use region::{Region, RegionMap};

// ===== Pipeline result =====

/// Complete segmentation result from a single [`segment()`] call.
#[derive(Debug, Clone)]
pub struct SegmentResult {
    /// Final refined regions after edge-aware merge + split
    pub regions: Vec<Region>,
    /// Per-pixel region assignment: `Some(region_id)` or `None` if filtered
    pub labels: Vec<Option<usize>>,
    /// Quantized sRGB colour palette
    pub palette: Palette,
    /// Per-pixel edge strength in [0, 1]
    pub edge_map: Vec<f64>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
}

// ===== Main pipeline =====

/// Run the full colour segmentation pipeline on an sRGB image.
///
/// # Pipeline steps
///
/// 1. **Quantize** — Median Cut in CIELAB space → cluster indices + palette
/// 2. **Extract regions** — 4-connected component labelling on cluster indices
/// 3. **Detect edges** — LAB Sobel gradient magnitude with smoothstep threshold
/// 4. **Refine** — edge-aware merge (weak boundaries) + split (strong internal edges)
///
/// # Arguments
///
/// * `img` — 8-bit sRGB image
/// * `params` — segmentation parameters (max clusters, edge thresholds, etc.)
///
/// # Errors
///
/// Returns an error if the image has zero pixels.  All other steps are
/// infallible for valid inputs.
pub fn segment(img: &image::RgbImage, params: &SegmentParams) -> anyhow::Result<SegmentResult> {
    let (width, height) = img.dimensions();
    let n = (width * height) as usize;

    anyhow::ensure!(n > 0, "image must have at least 1 pixel");

    // ===== Extract RGB pixels (normalised [0, 1]) =====
    let rgb_pixels: Vec<[f64; 3]> = img
        .pixels()
        .map(|p| {
            [
                p[0] as f64 / 255.0,
                p[1] as f64 / 255.0,
                p[2] as f64 / 255.0,
            ]
        })
        .collect();

    // ===== Quantize: Median Cut in CIELAB =====
    let clusters = quantize::quantize(&rgb_pixels, params);

    // Reconstruct per-pixel cluster assignment from cluster member lists
    let mut cluster_indices = vec![0usize; n];
    for (cid, cluster) in clusters.iter().enumerate() {
        for &idx in &cluster.members {
            cluster_indices[idx] = cid;
        }
    }

    // Build sRGB palette from cluster member RGB means (avoids LAB↔RGB round-trip)
    let palette = build_palette(&clusters, &rgb_pixels);
    let cluster_centroids: Vec<[f64; 3]> = clusters.iter().map(|c| c.centroid).collect();

    // ===== Extract connected regions (4-CCL) =====
    let region_params = if params.merge_small_regions {
        let mut p = params.clone();
        p.min_region_area = 1;
        p
    } else {
        params.clone()
    };
    let region_map = region::extract_region_map(&cluster_indices, width, height, &region_params);

    // ===== Edge detection (LAB Sobel) =====
    let edge_map = edge::detect_edges(&rgb_pixels, width, height, params);

    // ===== Refine: merge weak edges + split strong internal edges =====
    let (refined_regions, refined_labels) = refine::refine_with_colors(
        &region_map.regions,
        &region_map.labels,
        &edge_map,
        &cluster_centroids,
        width,
        height,
        params,
    );

    Ok(SegmentResult {
        regions: refined_regions,
        labels: refined_labels,
        palette,
        edge_map,
        width,
        height,
    })
}

// ===== Helpers =====

/// Build an sRGB [`Palette`] from quantization clusters.
///
/// Each cluster's colour is the mean of its members' original RGB values,
/// preserving perceptual accuracy by avoiding a LAB→RGB round-trip.
fn build_palette(clusters: &[quantize::Cluster], rgb_pixels: &[[f64; 3]]) -> Palette {
    let mut colors: Vec<[u8; 3]> = Vec::with_capacity(clusters.len());
    let mut counts: Vec<usize> = Vec::with_capacity(clusters.len());

    for cluster in clusters {
        if cluster.members.is_empty() {
            colors.push([0, 0, 0]);
            counts.push(0);
            continue;
        }

        let n = cluster.members.len() as f64;
        let (sr, sg, sb) =
            cluster
                .members
                .iter()
                .fold((0.0f64, 0.0f64, 0.0f64), |(sr, sg, sb), &idx| {
                    let rgb = rgb_pixels[idx];
                    (sr + rgb[0], sg + rgb[1], sb + rgb[2])
                });

        colors.push([
            ((sr / n).clamp(0.0, 1.0) * 255.0).round() as u8,
            ((sg / n).clamp(0.0, 1.0) * 255.0).round() as u8,
            ((sb / n).clamp(0.0, 1.0) * 255.0).round() as u8,
        ]);
        counts.push(cluster.members.len());
    }

    Palette { colors, counts }
}

// ===== Integration tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    /// Synthetic 4×4 image with two distinct colour regions.
    /// Verifies `segment()` produces correct dimensions, valid edge values,
    /// and at least one connected region.
    #[test]
    fn test_segment_synthetic_4x4() {
        let mut img = RgbImage::new(4, 4);
        // Left half: red, right half: blue
        for y in 0..4 {
            for x in 0..4 {
                if x < 2 {
                    img.put_pixel(x, y, image::Rgb([255, 0, 0]));
                } else {
                    img.put_pixel(x, y, image::Rgb([0, 0, 255]));
                }
            }
        }

        let params = SegmentParams {
            min_region_area: 1, // 4×4 image only has 16 pixels total
            ..SegmentParams::default()
        };
        let result = segment(&img, &params).expect("segmentation should succeed");

        // Dimensions
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);

        // Edge map: one value per pixel, all in [0,1]
        assert_eq!(result.edge_map.len(), 16);
        for &e in &result.edge_map {
            assert!((0.0..=1.0).contains(&e), "edge strength {e} out of [0,1]");
        }

        // Labels: one per pixel
        assert_eq!(result.labels.len(), 16);

        // Region extraction succeeds
        assert!(
            !result.regions.is_empty(),
            "should find at least one connected region"
        );

        // Palette colours are valid 8-bit values
        for col in &result.palette.colors {
            assert!((0..=255).contains(&col[0]));
            assert!((0..=255).contains(&col[1]));
            assert!((0..=255).contains(&col[2]));
        }

        // Palette count matches colours length
        assert_eq!(result.palette.colors.len(), result.palette.counts.len());
    }

    /// Zero-size image should return an error.
    #[test]
    fn test_segment_empty_image() {
        let img = RgbImage::new(0, 0);
        let params = SegmentParams::default();
        let result = segment(&img, &params);
        assert!(result.is_err(), "empty image must fail");
    }
}
