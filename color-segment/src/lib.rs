// color-segment — Color-based image region segmentation library
// Target: anime-style (二次元) images

pub mod anime;
pub mod params;
pub mod quantize;
pub mod region;

// ===== Re-exports =====

pub use image;
pub use params::SegmentParams;
pub use quantize::{Palette, Pixel};
pub use region::Region;

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
    anime::segment_anime_blocks(img, params)
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
