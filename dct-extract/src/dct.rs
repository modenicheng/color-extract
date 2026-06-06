use rayon::prelude::*;

// ---------------------------------------------------------------------------
// DCT constants
// ---------------------------------------------------------------------------
const N: usize = 8;        // DCT block size
const THRESHOLD: usize = 4; // u+v < THRESHOLD → low freq; ≥ → high freq
const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// DCT matrix (Type-II) pre‑computation
// ---------------------------------------------------------------------------

/// Pre‑compute the orthonormal DCT Type‑II matrix T (N×N).
/// T[i][j] = α(i) · cos((2j+1)·i·π / 2N)
/// α(0) = √(1/N), α(i>0) = √(2/N)
fn dct_matrix() -> [[f64; N]; N] {
    let mut t = [[0.0; N]; N];
    let inv_sqrt_n = 1.0 / (N as f64).sqrt();
    let sqrt_2_over_n = (2.0 / N as f64).sqrt();
    for i in 0..N {
        let alpha = if i == 0 { inv_sqrt_n } else { sqrt_2_over_n };
        for j in 0..N {
            t[i][j] =
                alpha * ((2.0 * j as f64 + 1.0) * i as f64 * PI / (2.0 * N as f64)).cos();
        }
    }
    t
}

/// Transpose an N×N matrix.
fn transpose(m: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[j][i] = m[i][j];
        }
    }
    out
}

/// 2D DCT Type‑II via separable row‑column approach:
///   F = T × block × Tᵀ
fn dct_2d(block: &[[f64; N]; N], t: &[[f64; N]; N]) -> [[f64; N]; N] {
    let tt = transpose(t);

    // rows: block × Tᵀ
    let mut rows_dct = [[0.0; N]; N];
    for r in 0..N {
        for c in 0..N {
            for k in 0..N {
                rows_dct[r][c] += block[r][k] * tt[k][c];
            }
        }
    }

    // columns: T × rows_dct
    let mut out = [[0.0; N]; N];
    for r in 0..N {
        for c in 0..N {
            for k in 0..N {
                out[r][c] += t[r][k] * rows_dct[k][c];
            }
        }
    }
    out
}

/// High‑frequency ratio from DCT coefficients.
///
///   c = Σ(u+v≥T) F(u,v)² / (Σ(all) F(u,v)² – F(0,0)² + ε)
///
/// Higher c → more texture / edges in the block.
fn high_freq_ratio(coeffs: &[[f64; N]; N]) -> f64 {
    let mut total_ac = 0.0;
    let mut high_freq = 0.0;
    for u in 0..N {
        for v in 0..N {
            if u == 0 && v == 0 {
                continue;
            }
            let e = coeffs[u][v] * coeffs[u][v];
            total_ac += e;
            if u + v >= THRESHOLD {
                high_freq += e;
            }
        }
    }
    high_freq / (total_ac + 1e-10)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute a per‑pixel complexity map where each value is the
/// high‑frequency signal ratio `c` ∈ [0,1] of the 8×8 block centred
/// on that pixel.
///
/// `pixels` – normalized RGB [0,1] row‑major.
pub fn compute_complexity_map(pixels: &[[f64; 3]], width: u32, height: u32) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let offset = (N / 2) as i32; // 4

    // Convert to grayscale luminance Y = 0.299·R + 0.587·G + 0.114·B
    let gray: Vec<f64> = pixels.iter().map(|&[r, g, b]| 0.299 * r + 0.587 * g + 0.114 * b).collect();

    let t = dct_matrix();

    let mut complexity = vec![0.0; w * h];

    complexity
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                // Extract N×N block centred at (x, y) with mirror padding
                let mut block = [[0.0; N]; N];
                for dy in 0..N {
                    for dx in 0..N {
                        let px = (x as i32 + dx as i32 - offset).clamp(0, w as i32 - 1) as usize;
                        let py = (y as i32 + dy as i32 - offset).clamp(0, h as i32 - 1) as usize;
                        block[dy][dx] = gray[py * w + px];
                    }
                }

                let coeffs = dct_2d(&block, &t);
                row[x] = high_freq_ratio(&coeffs);
            }
        });

    complexity
}
