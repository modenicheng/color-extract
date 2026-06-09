// =============================================================================
// DCT 纹理复杂度
// =============================================================================

use rayon::prelude::*;

const DCT_N: usize = 8;

fn dct_matrix() -> [[f64; DCT_N]; DCT_N] {
    let mut t = [[0.0; DCT_N]; DCT_N];
    let inv_sqrt_n = 1.0 / (DCT_N as f64).sqrt();
    let sqrt_2_over_n = (2.0 / DCT_N as f64).sqrt();
    for i in 0..DCT_N {
        let alpha = if i == 0 { inv_sqrt_n } else { sqrt_2_over_n };
        for j in 0..DCT_N {
            t[i][j] = alpha
                * ((2.0 * j as f64 + 1.0) * i as f64 * std::f64::consts::PI / (2.0 * DCT_N as f64))
                    .cos();
        }
    }
    t
}

fn transpose(m: &[[f64; DCT_N]; DCT_N]) -> [[f64; DCT_N]; DCT_N] {
    let mut out = [[0.0; DCT_N]; DCT_N];
    for i in 0..DCT_N {
        for j in 0..DCT_N {
            out[j][i] = m[i][j];
        }
    }
    out
}

fn dct_2d(block: &[[f64; DCT_N]; DCT_N], t: &[[f64; DCT_N]; DCT_N]) -> [[f64; DCT_N]; DCT_N] {
    let tt = transpose(t);
    let mut rows_dct = [[0.0; DCT_N]; DCT_N];
    for r in 0..DCT_N {
        for c in 0..DCT_N {
            for k in 0..DCT_N {
                rows_dct[r][c] += block[r][k] * tt[k][c];
            }
        }
    }
    let mut out = [[0.0; DCT_N]; DCT_N];
    for r in 0..DCT_N {
        for c in 0..DCT_N {
            for k in 0..DCT_N {
                out[r][c] += t[r][k] * rows_dct[k][c];
            }
        }
    }
    out
}

fn high_freq_ratio(coeffs: &[[f64; DCT_N]; DCT_N], threshold: usize) -> f64 {
    let mut total_ac = 0.0;
    let mut high_freq = 0.0;
    for u in 0..DCT_N {
        for v in 0..DCT_N {
            if u == 0 && v == 0 {
                continue;
            }
            let e = coeffs[u][v] * coeffs[u][v];
            total_ac += e;
            if u + v >= threshold {
                high_freq += e;
            }
        }
    }
    high_freq / (total_ac + 1e-10)
}

pub fn compute_dct_complexity(gray: &[f64], w: usize, h: usize) -> Vec<f64> {
    let offset = (DCT_N / 2) as i32;
    let t = dct_matrix();
    let mut map = vec![0.0; w * h];
    map.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut block = [[0.0; DCT_N]; DCT_N];
            for dy in 0..DCT_N {
                for dx in 0..DCT_N {
                    let px = (x as i32 + dx as i32 - offset).clamp(0, w as i32 - 1) as usize;
                    let py = (y as i32 + dy as i32 - offset).clamp(0, h as i32 - 1) as usize;
                    block[dy][dx] = gray[py * w + px];
                }
            }
            let coeffs = dct_2d(&block, &t);
            row[x] = high_freq_ratio(&coeffs, 4);
        }
    });
    map
}
