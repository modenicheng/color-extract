// =============================================================================
// CIELAB Sobel edge detection for anime-style images
// =============================================================================
//
// Computes per-pixel edge strength from LAB gradient magnitudes using a 3×3
// Sobel operator on each LAB channel.  The three channel gradients are combined
// via L₂ norm and then passed through a smoothstep soft threshold that is
// tuned for the clean line art and flat colour boundaries typical of anime.
//
// No Canny extension, no non-maximum suppression — raw gradient magnitude only.
// NMS is deferred to `refine.rs` for a clean separation of concerns.

use crate::params::SegmentParams;
use palette::{IntoColor, Lab, Srgb};

// =============================================================================
// Public API
// =============================================================================

/// Compute per-pixel edge strength from LAB gradients.
///
/// # Arguments
///
/// * `pixels` — RGB pixels normalised to `[0, 1]`, in row-major order.
/// * `width`, `height` — image dimensions in pixels.
/// * `params` — segmentation parameters; only `edge_threshold` is used.
///
/// # Returns
///
/// `Vec<f64>` of length `width × height` with each value in `[0, 1]`.
/// Higher values indicate stronger edges.
///
/// * Border pixels (first/last row, first/last column) always receive `1.0`
///   because the Sobel kernel cannot be applied there.
/// * Interior values below `params.edge_threshold` are attenuated to near zero
///   via a smoothstep soft threshold.
pub fn detect_edges(
    pixels: &[[f64; 3]],
    width: u32,
    height: u32,
    params: &SegmentParams,
) -> Vec<f64> {
    let n = pixels.len();
    assert_eq!(n, (width * height) as usize, "pixel count mismatch");

    // ===== Step 1 — Convert RGB → CIELAB L*a*b* =====
    let mut lab = Vec::with_capacity(n);
    for px in pixels {
        let srgb = Srgb::new(px[0] as f32, px[1] as f32, px[2] as f32);
        let l: Lab = srgb.into_color();
        lab.push([l.l as f64, l.a as f64, l.b as f64]);
    }

    let w = width as usize;
    let h = height as usize;
    let mut mag = vec![0.0_f64; n];

    if w < 3 || h < 3 {
        return vec![1.0; n];
    }

    // ===== Step 2 — 3×3 Sobel on each LAB channel =====
    //
    // Gx = [[-1,  0,  1],      Gy = [[-1, -2, -1],
    //       [-2,  0,  2],            [ 0,  0,  0],
    //       [-1,  0,  1]]            [ 1,  2,  1]]
    //
    // Divisor = 8 (sum of |kernel|) for per-pixel delta approximation.
    // Skip the 1-px border — handled separately in Step 5.
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let i = y * w + x;

            // --- L-channel gradients ---
            let gx_l = -lab[i - w - 1][0] + lab[i - w + 1][0] - 2.0 * lab[i - 1][0]
                + 2.0 * lab[i + 1][0]
                - lab[i + w - 1][0]
                + lab[i + w + 1][0];
            let gy_l = -lab[i - w - 1][0] - 2.0 * lab[i - w][0] - lab[i - w + 1][0]
                + lab[i + w - 1][0]
                + 2.0 * lab[i + w][0]
                + lab[i + w + 1][0];

            // --- a-channel gradients ---
            let gx_a = -lab[i - w - 1][1] + lab[i - w + 1][1] - 2.0 * lab[i - 1][1]
                + 2.0 * lab[i + 1][1]
                - lab[i + w - 1][1]
                + lab[i + w + 1][1];
            let gy_a = -lab[i - w - 1][1] - 2.0 * lab[i - w][1] - lab[i - w + 1][1]
                + lab[i + w - 1][1]
                + 2.0 * lab[i + w][1]
                + lab[i + w + 1][1];

            // --- b-channel gradients ---
            let gx_b = -lab[i - w - 1][2] + lab[i - w + 1][2] - 2.0 * lab[i - 1][2]
                + 2.0 * lab[i + 1][2]
                - lab[i + w - 1][2]
                + lab[i + w + 1][2];
            let gy_b = -lab[i - w - 1][2] - 2.0 * lab[i - w][2] - lab[i - w + 1][2]
                + lab[i + w - 1][2]
                + 2.0 * lab[i + w][2]
                + lab[i + w + 1][2];

            // ===== Step 3 — Combine per-channel magnitudes =====
            // Normalise by 8 → per-pixel delta on each channel,
            // then combine via sqrt(G²_L + G²_a + G²_b).
            let grad_l = (gx_l * gx_l + gy_l * gy_l).sqrt() / 8.0;
            let grad_a = (gx_a * gx_a + gy_a * gy_a).sqrt() / 8.0;
            let grad_b = (gx_b * gx_b + gy_b * gy_b).sqrt() / 8.0;

            mag[i] = (grad_l * grad_l + grad_a * grad_a + grad_b * grad_b).sqrt();
        }
    }

    // ===== Step 4 — Normalise to [0, 1] via global max =====
    let max_mag = mag.iter().cloned().fold(0.0_f64, f64::max).max(1e-12);
    for m in &mut mag {
        *m /= max_mag;
    }

    // ===== Step 5 — Soft threshold (smoothstep attenuation) =====
    //
    // Values below `edge_threshold` are smoothly attenuated toward 0.
    // The transition region is [thr, thr*2.5] clamped to [0, 1].
    // This eliminates low-confidence edges (noise / subtle gradients)
    // while preserving crisp colour boundaries typical of anime line art.
    let thr = params.edge_threshold;
    if thr > 0.0 {
        let thr_hi = (thr * 2.5).min(1.0);
        for m in &mut mag {
            *m = smoothstep(thr, thr_hi, *m);
        }
    }

    // ===== Step 6 — Border pixels → 1.0 (image boundary) =====
    for y in 0..h {
        if y == 0 || y == h - 1 {
            // Full top / bottom rows
            for x in 0..w {
                mag[y * w + x] = 1.0;
            }
        } else {
            // Left / right columns (corners already covered above)
            mag[y * w] = 1.0;
            mag[y * w + w - 1] = 1.0;
        }
    }

    mag
}

// =============================================================================
// Helpers
// =============================================================================

/// Safe indexed access into the LAB pixel buffer, returning `None` for
/// out-of-bounds coordinates so callers can handle border conditions.
#[allow(dead_code)]
fn at(x: i32, y: i32, w: u32, lab: &[[f64; 3]]) -> Option<[f64; 3]> {
    let h = (lab.len() / w as usize) as i32;
    if x < 0 || y < 0 || x >= w as i32 || y >= h {
        return None;
    }
    Some(lab[(y as usize) * (w as usize) + (x as usize)])
}

/// Cubic Hermite smoothstep: 0 when x ≤ edge0, 1 when x ≥ edge1,
/// smoothly interpolated in between.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-12)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Output vector length must equal pixel count.
    #[test]
    fn test_output_length() {
        let params = SegmentParams::default();
        let pixels = vec![[0.5; 3]; 50 * 30];
        let edges = detect_edges(&pixels, 50, 30, &params);
        assert_eq!(edges.len(), 1500);
    }

    /// Every output value must be in [0, 1].
    #[test]
    fn test_values_in_range() {
        let params = SegmentParams::default();
        // Gradient image with a sharp diagonal boundary
        let w = 16;
        let h = 16;
        let mut pixels = vec![[0.0; 3]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                pixels[i] = if x + y < w as u32 / 2 + h / 2 {
                    [1.0, 0.2, 0.2]
                } else {
                    [0.2, 0.2, 1.0]
                };
            }
        }
        let edges = detect_edges(&pixels, w, h, &params);
        for &e in &edges {
            assert!(e >= 0.0, "negative edge: {}", e);
            assert!(e <= 1.0, "edge > 1.0: {}", e);
        }
    }

    /// A perfectly uniform image should produce near-zero edge strength
    /// for every interior pixel (border pixels are always 1.0).
    #[test]
    fn test_uniform_image_near_zero() {
        let params = SegmentParams::default();
        let w = 10;
        let h = 10;
        let pixels = vec![[0.6, 0.3, 0.1]; (w * h) as usize];
        let edges = detect_edges(&pixels, w, h, &params);

        // Interior pixels (excluding 1-px border) should be zero
        for y in 1..(h as usize - 1) {
            for x in 1..(w as usize - 1) {
                let e = edges[y * (w as usize) + x];
                assert!(
                    e < 1e-6,
                    "uniform image interior should be 0, got {} at ({},{})",
                    e,
                    x,
                    y
                );
            }
        }

        // Border pixels are 1.0
        for x in 0..w as usize {
            assert!((edges[x] - 1.0).abs() < 1e-6);
            assert!((edges[(h as usize - 1) * (w as usize) + x] - 1.0).abs() < 1e-6);
        }
        for y in 1..(h as usize - 1) {
            assert!((edges[y * (w as usize)] - 1.0).abs() < 1e-6);
            assert!((edges[y * (w as usize) + (w as usize) - 1] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tiny_image_no_sobel_window() {
        let params = SegmentParams::default();
        let pixels = vec![[0.2, 0.3, 0.4]; 4];
        let edges = detect_edges(&pixels, 2, 2, &params);
        assert_eq!(edges, vec![1.0; 4]);
    }

    /// A sharp colour boundary (vertical split: red vs blue) must produce
    /// significantly higher edge response at the boundary than in flat regions.
    #[test]
    fn test_sharp_boundary_high_response() {
        let params = SegmentParams {
            edge_threshold: 0.0, // disable threshold so raw magnitudes are tested
            ..SegmentParams::default()
        };
        let w = 10;
        let h = 10;
        let mid = (w / 2) as usize;
        let mut pixels = vec![[0.0; 3]; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * (w as usize) + x;
                pixels[i] = if x < mid {
                    [1.0, 0.0, 0.0] // red
                } else {
                    [0.0, 0.0, 1.0] // blue
                };
            }
        }

        let edges = detect_edges(&pixels, w, h, &params);

        // Column just left of the split should be flat (interior, away from border)
        let left_flat = edges[3 * (w as usize) + (mid - 2)];
        let boundary = edges[3 * (w as usize) + (mid - 1)]; // pixel adjacent to split
        let _right_flat = edges[3 * (w as usize) + mid];

        // The boundary pixel (mid-1) is next to mid which is a different color,
        // so it should see a strong gradient.
        assert!(
            boundary > left_flat * 2.0,
            "boundary={} should be > 2× left_flat={}",
            boundary,
            left_flat
        );
        assert!(
            boundary > 0.1,
            "boundary={} should be clearly non-zero",
            boundary
        );
    }

    /// `at()` helper: in-bounds returns Some, out-of-bounds returns None.
    #[test]
    fn test_at_helper() {
        let w: u32 = 4;
        let lab = vec![[0.0; 3]; (4 * 3) as usize];

        // In bounds
        assert!(at(0, 0, w, &lab).is_some());
        assert!(at(3, 2, w, &lab).is_some());

        // Out of bounds
        assert!(at(-1, 0, w, &lab).is_none());
        assert!(at(0, -1, w, &lab).is_none());
        assert!(at(4, 0, w, &lab).is_none());
        assert!(at(0, 3, w, &lab).is_none());
    }
}
