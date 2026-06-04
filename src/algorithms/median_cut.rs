use crate::algorithms::{AlgorithmResult, PaletteEntry};
use crate::colorspace::ColorSpace;
use crate::timing::timed;
use anyhow::{anyhow, Result};

/// Run the Median Cut color quantization algorithm.
///
/// Works in any coordinate space — pixels are converted to `cs`,
/// all splitting happens in that space, and means are converted
/// back to normalized RGB before building the palette.
pub fn run(pixels: &[[f64; 3]], cs: ColorSpace, k: usize) -> Result<AlgorithmResult> {
    let n = pixels.len();
    if n == 0 {
        return Err(anyhow!("Median Cut requires at least one pixel"));
    }

    let effective_k = k.min(n);

    // 1. Convert all pixels to the target color space.
    let coords = cs.convert_batch_to(pixels);

    // 2. Timed: split → means → convert back → build palette.
    let ((palette, dominant), duration) = timed(|| {
        // 2a. Initialize with one bucket containing all pixel indices.
        let mut buckets: Vec<Vec<usize>> = vec![(0..n).collect()];

        // 2b. Repeatedly split the largest bucket until we have k buckets
        //     or no bucket has ≥ 2 pixels.
        while buckets.len() < effective_k {
            let largest_idx = (0..buckets.len())
                .max_by_key(|&i| buckets[i].len())
                .unwrap();

            let largest = &buckets[largest_idx];
            if largest.len() < 2 {
                break;
            }

            // Find the dimension with the widest range of coordinate values.
            let axis = longest_axis(largest, &coords);

            // Sort the indices by pixel value along that axis.
            let mut indices = buckets[largest_idx].clone();
            indices.sort_by(|&a, &b| {
                coords[a][axis]
                    .partial_cmp(&coords[b][axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Split at the median.
            let mid = indices.len() / 2;
            let right = indices.split_off(mid);
            buckets[largest_idx] = indices;
            buckets.push(right);
        }

        // 2c. Compute the mean color for each bucket (in target-space coords).
        let means: Vec<[f64; 3]> = buckets
            .iter()
            .map(|bucket| mean(bucket, &coords))
            .collect();

        // 2d. Convert means back to normalized RGB.
        let rgb_means = cs.convert_batch_from(&means);

        // 2e. Dominant color = mean of the largest bucket.
        let dominant_idx = (0..buckets.len())
            .max_by_key(|&i| buckets[i].len())
            .unwrap();

        let total = n as f64;

        // 2f. Build palette entries.
        let mut palette: Vec<PaletteEntry> = buckets
            .iter()
            .enumerate()
            .map(|(i, bucket)| {
                let rgb_norm = rgb_means[i];
                let proportion = bucket.len() as f64 / total;
                let r = (rgb_norm[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (rgb_norm[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (rgb_norm[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                PaletteEntry {
                    rgb: [r, g, b],
                    hex: format!("#{r:02x}{g:02x}{b:02x}"),
                    lab_l: crate::colorspace::perceptual_lightness(rgb_norm),
                    proportion,
                }
            })
            .collect();

        // 2g. Dominant entry.
        let dominant_rgb = rgb_means[dominant_idx];
        let dr = (dominant_rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let dg = (dominant_rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let db = (dominant_rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        let dominant = PaletteEntry {
            rgb: [dr, dg, db],
            hex: format!("#{dr:02x}{dg:02x}{db:02x}"),
            lab_l: crate::colorspace::perceptual_lightness(dominant_rgb),
            proportion: buckets[dominant_idx].len() as f64 / total,
        };

        // 2h. Sort palette from dark to light (by CIELAB L*).
        palette.sort_by(|a, b| {
            a.lab_l
                .partial_cmp(&b.lab_l)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        (palette, dominant)
    });

    Ok(AlgorithmResult {
        palette,
        dominant,
        duration,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Given a slice of pixel indices and the coordinate data, find which dimension
/// (0, 1, or 2) has the widest range of values.
fn longest_axis(indices: &[usize], coords: &[[f64; 3]]) -> usize {
    let mut ranges = [0.0f64; 3];
    for d in 0..3 {
        let min = indices
            .iter()
            .map(|&i| coords[i][d])
            .fold(f64::INFINITY, f64::min);
        let max = indices
            .iter()
            .map(|&i| coords[i][d])
            .fold(f64::NEG_INFINITY, f64::max);
        ranges[d] = max - min;
    }
    (0..3)
        .max_by(|&a, &b| {
            ranges[a]
                .partial_cmp(&ranges[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

/// Compute the arithmetic mean of pixel coordinates for the given set of indices.
fn mean(indices: &[usize], coords: &[[f64; 3]]) -> [f64; 3] {
    let n = indices.len() as f64;
    if n == 0.0 {
        return [0.0; 3];
    }
    let sum = indices
        .iter()
        .fold([0.0; 3], |acc, &i| {
            [
                acc[0] + coords[i][0],
                acc[1] + coords[i][1],
                acc[2] + coords[i][2],
            ]
        });
    [sum[0] / n, sum[1] / n, sum[2] / n]
}
