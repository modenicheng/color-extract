use anyhow::Result;
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array2;
use palette::{FromColor, Lab, Srgb};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Score formula constants  (调参区 —— 改这里！)
// ---------------------------------------------------------------------------
//
// 最终公式:
//   score = p^GAMMA                           ← 占比（<1 抑制大面积，>1 奖励大面积）
//         × (1 + ALPHA_C × c_final)           ← DCT 纹理复杂度奖励
//         × (1 + BETA_C  × C_rel_pos)          ← 相对彩度奖励（图内偏鲜艳加分）
//         × L_weight                            ← 亮度×彩度联动（见下）
//         × (1 + BETA_U  × U_norm)              ← 颜色独特性奖励（与众不同加分）
//         × bg_penalty                          ← 背景惩罚（除法版）
//         × WHITE_GATE (条件触发)               ← 白/浅灰背景额外打压
//
// 亮度联动:
//   L_weight = 1 + BETA_L × L_rel_pos × (0.5 + 0.5 × max(C_rel_pos, U_norm))
//   含义：亮度本身不加分，只有同时具备彩度 OR 独特性时才加分。
//   纯白背景（亮但无色）拿不到亮度奖励。
//
// 背景惩罚:
//   bg_penalty = 1 / (1 + LAMBDA_B × B)
//   LAMBDA_B=0 → 关闭惩罚；越大惩罚越重。一般 0.5~3.0。
//
// 白背景门控 (white gate):
//   触发条件: chroma < 5 且 L* > 85 且 border_connected > 0.4
//   → 仅近中性 + 高亮 + 边界连通三者同时满足才触发
//   → 不会误伤白色衣服、银发等主体色
const SCORE_GAMMA: f64 = 0.5;    // ↑ 占比幂次: 0.5=开根号抑制大块背景
const SCORE_ALPHA_C: f64 = 3.0;  // ↑ 复杂度权重: 纹理丰富的颜色加分更多
const SCORE_BETA_C: f64 = 0.25;  // ↑ 相对彩度权重: 鲜艳色加分
const SCORE_BETA_L: f64 = 0.30;  // ↑ 相对亮度权重: 亮色（有彩度时）加分
const SCORE_BETA_U: f64 = 0.35;  // ↑ 独特性权重: 颜色越独有加分越多
const SCORE_LAMBDA_B: f64 = 1.0;  // ↑ 背景惩罚: 0=关，越大越狠
const SCORE_WHITE_GATE: f64 = 0.15; // ↓ 白背景乘数: 越小打压越狠

// ----- 背景性 B 的组成 (4 项相加 = 1.0) -----
// B = W1×border_connected + W2×edge + W3×spread + W4×smoothness
//
// border_connected: 从图像四边 BFS 连通的像素占比（最关键，区分背景 vs 主体）
// edge:             落在图像四边边缘的像素占比
// spread:           空间分布广度（标准差归一化，大 = 像背景）
// smoothness:       纹理平坦度 (1 − c_abs)，平滑区域更像背景
const BG_W1: f64 = 0.35; // 边界连通 — 最重要
const BG_W2: f64 = 0.35; // 边缘占比
const BG_W3: f64 = 0.10; // 空间扩散
const BG_W4: f64 = 0.20; // 纹理平滑

// ----- 硬编码 clamp 参数 (不常改，但知道在哪) -----
//
// c_abs    = clamp(c_raw / 10.0,   0, 0.6)   — 绝对复杂度截断
// c_final  = 0.6×c_abs + 0.4×percentile_rank  — 复杂度混合比例 (c_final∈[0,1])
// C_rel    = clamp((C−C_med)/C_mad, 0, 3) / 3  — 相对彩度，3 MAD 封顶
// L_rel    = clamp((L−L_med)/L_mad, 0, 3) / 3  — 相对亮度，3 MAD 封顶
//
// White gate 阈值:
//   chroma < 5.0   — 几乎无色
//   L* > 85.0      — 非常亮
//   border_connected > 0.4 — 超 40% 像素从边界连通

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// One colour cluster with its statistics.
#[derive(Debug, Clone)]
pub struct Cluster4D {
    #[allow(dead_code)]
    pub rgb: [u8; 3],
    pub hex: String,
    pub proportion: f64,
    pub avg_complexity: f64,
    pub dominant_score: f64,
    pub lab_l: f64,
    pub lab_a: f64,
    pub lab_b: f64,
    pub chroma: f64,
    pub avg_x: f64,
    pub avg_y: f64,
    // -- background / scoring diagnostics --
    pub edge_ratio: f64,
    pub border_connected_ratio: f64,
    pub spread: f64,
    pub uniqueness: f64,
    pub centre_weight: f64,
    pub backgroundness: f64,
}

/// Result of a clustering run.
#[derive(Debug, Clone)]
pub struct ClusterResult4D {
    pub clusters: Vec<Cluster4D>,
    pub dominant: Cluster4D,
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn lab_to_rgb_norm(lab: [f64; 3]) -> [f64; 3] {
    let lab_c = Lab::new(lab[0] as f32, lab[1] as f32, lab[2] as f32);
    let srgb: Srgb = Srgb::from_color(lab_c);
    [
        (srgb.red as f64).clamp(0.0, 1.0),
        (srgb.green as f64).clamp(0.0, 1.0),
        (srgb.blue as f64).clamp(0.0, 1.0),
    ]
}

fn make_empty_cluster() -> Cluster4D {
    Cluster4D {
        rgb: [0, 0, 0],
        hex: "#000000".into(),
        proportion: 0.0,
        avg_complexity: 0.0,
        dominant_score: 0.0,
        lab_l: 0.0,
        lab_a: 0.0,
        lab_b: 0.0,
        chroma: 0.0,
        avg_x: 0.0,
        avg_y: 0.0,
        edge_ratio: 0.0,
        border_connected_ratio: 0.0,
        spread: 0.0,
        uniqueness: 0.0,
        centre_weight: 0.0,
        backgroundness: 0.0,
    }
}

fn build_cluster(
    rgb_norm: [f64; 3],
    proportion: f64,
    avg_c: f64,
    lab: [f64; 3],
    avg_x: f64,
    avg_y: f64,
) -> Cluster4D {
    let r = (rgb_norm[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgb_norm[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgb_norm[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    Cluster4D {
        rgb: [r, g, b],
        hex: format!("#{r:02x}{g:02x}{b:02x}"),
        proportion,
        avg_complexity: avg_c,
        dominant_score: 0.0, // filled later by score_clusters()
        lab_l: lab[0].clamp(0.0, 100.0),
        lab_a: lab[1],
        lab_b: lab[2],
        chroma,
        avg_x,
        avg_y,
        edge_ratio: 0.0,
        border_connected_ratio: 0.0,
        spread: 0.0,
        uniqueness: 0.0,
        centre_weight: 0.0,
        backgroundness: 0.0,
    }
}

fn sort_by_lightness(clusters: &mut [Cluster4D]) {
    clusters.sort_by(|a, b| a.lab_l.partial_cmp(&b.lab_l).unwrap_or(std::cmp::Ordering::Equal));
}

// ---------------------------------------------------------------------------
// Scoring: compute all advanced metrics and populate dominant_score
// ---------------------------------------------------------------------------

struct ScoreCtx {
    k: usize,
    w: u32,
    h: u32,
    counts: Vec<usize>,
    sum_x: Vec<f64>,
    sum_y: Vec<f64>,
    sum_x2: Vec<f64>,
    sum_y2: Vec<f64>,
    sum_dist_centre: Vec<f64>,     // sum of pixel distances to image centre
    edge_count: Vec<usize>,         // pixels on image border per cluster
    border_connected: Vec<usize>,   // pixels connected to border per cluster
}

// ===========================================================================
//  Statistics helpers — median, MAD, percentile_rank, simple ΔE
//  所有特征用 median+MAD 做稳健归一化，不依赖均值/标准差（抗异常值）
// ===========================================================================

/// 中位数
fn median(vals: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
}

/// 中位数绝对偏差 (Median Absolute Deviation) — 比标准差更稳健
fn median_abs_dev(vals: &[f64], med: f64) -> f64 {
    let devs: Vec<f64> = vals.iter().map(|&v| (v - med).abs()).collect();
    median(&devs)
}

/// 百分位排名: 对每个值返回它在整体中的位置 (0~1)
fn percentile_ranks(vals: &[f64]) -> Vec<f64> {
    let k = vals.len();
    if k <= 1 { return vec![0.5; k]; }
    let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; k];
    for (pos, &(orig_i, _)) in indexed.iter().enumerate() {
        ranks[orig_i] = pos as f64 / (k - 1) as f64;
    }
    ranks
}

/// 简易 CIELAB 欧氏距离 ΔE
fn delta_e(lab1: &[f64; 3], lab2: &[f64; 3]) -> f64 {
    ((lab1[0] - lab2[0]).powi(2)
   + (lab1[1] - lab2[1]).powi(2)
   + (lab1[2] - lab2[2]).powi(2))
    .sqrt()
}

// ---------------------------------------------------------------------------
// Scoring: compute all advanced metrics and populate dominant_score
// ---------------------------------------------------------------------------

fn score_clusters(clusters: &mut [Cluster4D], ctx: &ScoreCtx) {
    let k = ctx.k;
    if k == 0 { return; }
    let eps = 1e-6;

    // ================================================================
    //  1. 空间统计: edge_ratio / border_connected / spread / centre_weight
    // ================================================================
    let w = ctx.w as f64;
    let h = ctx.h as f64;
    let cx = w * 0.5;
    let cy = h * 0.5;
    let max_centre_dist = (cx * cx + cy * cy).sqrt();

    for i in 0..k {
        let n = ctx.counts[i] as f64;
        if n == 0.0 { continue; }
        let c = &mut clusters[i];
        c.edge_ratio = ctx.edge_count[i] as f64 / n;           // 落在图像四边的比例
        c.border_connected_ratio = ctx.border_connected[i] as f64 / n; // BFS 从边界连通的比例
        let var_x = (ctx.sum_x2[i] / n - (ctx.sum_x[i] / n).powi(2)).max(0.0);
        let var_y = (ctx.sum_y2[i] / n - (ctx.sum_y[i] / n).powi(2)).max(0.0);
        c.spread = ((var_x.sqrt() / w) * (var_y.sqrt() / h)).min(1.0); // 空间扩散度
        let avg_dist = ctx.sum_dist_centre[i] / n;
        c.centre_weight = (1.0 - avg_dist / max_centre_dist).max(0.0);  // 靠近中心=1, 边缘=0
    }

    // ================================================================
    //  2. 收集原始特征值
    // ================================================================
    let ls: Vec<f64> = clusters.iter().map(|c| c.lab_l).collect();          // L* 亮度
    let cs: Vec<f64> = clusters.iter().map(|c| c.chroma).collect();         // 色度 C=√(a²+b²)
    let ps: Vec<f64> = clusters.iter().map(|c| c.proportion).collect();     // 占比
    let c_raws: Vec<f64> = clusters.iter().map(|c| c.avg_complexity).collect(); // DCT 复杂度
    let labs: Vec<[f64; 3]> = clusters.iter().map(|c| [c.lab_l, c.lab_a, c.lab_b]).collect();

    // ================================================================
    //  3. 相对彩度 C_rel_pos  (稳健 z-score: median + MAD)
    //     C_z = (C − C_med) / C_mad  →  clamp(0, 3) / 3
    //     只奖励比中位更鲜艳的颜色，更淡的不惩罚
    // ================================================================
    let c_med = median(&cs);
    let c_mad = median_abs_dev(&cs, c_med) + eps;
    let c_rel_pos: Vec<f64> = cs.iter()
        .map(|&ci| ((ci - c_med) / c_mad).clamp(0.0, 3.0) / 3.0)
        .collect();

    // ================================================================
    //  4. 相对亮度 L_rel_pos  (同公式, median + MAD)
    //     只奖励比中位更亮的颜色；暗色不加分也不扣分
    // ================================================================
    let l_med = median(&ls);
    let l_mad = median_abs_dev(&ls, l_med) + eps;
    let l_rel_pos: Vec<f64> = ls.iter()
        .map(|&li| ((li - l_med) / l_mad).clamp(0.0, 3.0) / 3.0)
        .collect();

    // ================================================================
    //  5. 复杂度 c_final  (绝对 + 相对混合)
    //     c_abs  = clamp(c_raw/10, 0, 0.6)   ← 绝对截断
    //     c_rank = 0~1 百分位排名             ← 图内相对
    //     c_final = 0.6×c_abs + 0.4×c_rank   ← 混合
    // ================================================================
    let c_abs: Vec<f64> = c_raws.iter()
        .map(|&cr| (cr / 10.0).clamp(0.0, 0.6))
        .collect();
    let c_rank = percentile_ranks(&c_raws);
    let c_final: Vec<f64> = c_abs.iter().zip(&c_rank)
        .map(|(&a, &r)| 0.6 * a + 0.4 * r)
        .collect();

    // ================================================================
    //  6. 颜色独特性 U_norm  (加权 pairwise ΔE)
    //     U_i = Σ_j p_j × ΔE(Lab_i, Lab_j)
    //     然后除以 max(U) 归一化到 [0,1]
    //     含义: 这个颜色和图中其他颜色有多"不同"
    // ================================================================
    let mut u_raw = vec![0.0f64; k];
    for i in 0..k {
        let mut sum = 0.0;
        for j in 0..k {
            if i == j { continue; }
            sum += ps[j] * delta_e(&labs[i], &labs[j]);  // 加权求和
        }
        u_raw[i] = sum;
    }
    let u_max = u_raw.iter().cloned().fold(0.0f64, f64::max);
    let u_norm: Vec<f64> = if u_max > eps {
        u_raw.iter().map(|&u| u / u_max).collect()
    } else {
        vec![0.0; k]
    };

    // ================================================================
    //  最终分数计算
    // ================================================================
    //
    // 注意：所有颜色特征 (C_rel, L_rel, U_norm) 都是图内相对值，
    //       所以低饱和图里的"相对鲜艳"色同样能拿高分。
    for i in 0..k {
        let n = ctx.counts[i] as f64;
        if n == 0.0 { continue; }
        let c = &mut clusters[i];

        // ---- 背景性 B & 惩罚 ----
        let smoothness = 1.0 - c_abs[i];            // 纹理越平滑越像背景
        c.backgroundness = (BG_W1 * c.border_connected_ratio  // 边界连通
                          + BG_W2 * c.edge_ratio               // 边缘占比
                          + BG_W3 * c.spread                   // 空间扩散
                          + BG_W4 * smoothness)                // 纹理平滑
            .clamp(0.0, 1.0);

        let bg_penalty = 1.0 / (1.0 + SCORE_LAMBDA_B * c.backgroundness);
        // bg_penalty ∈ (0, 1] : B=0→1.0(无惩罚), B=1→1/(1+LAMBDA_B)

        c.uniqueness = u_norm[i];

        // ---- 亮度联动彩度: 亮色+有色才加分 ----
        // L_rel_pos 越大 = 比图内大多数颜色更亮
        // 但乘以 (0.5 + 0.5*max(C_rel,U)) 后，只有同时鲜明/独特才能激活
        let l_weight = 1.0 + SCORE_BETA_L * l_rel_pos[i]
            * (0.5 + 0.5 * c_rel_pos[i].max(u_norm[i]));

        // ---- 拼合最终分数 ----
        c.dominant_score = c.proportion.powf(SCORE_GAMMA)  // 占比
            * (1.0 + SCORE_ALPHA_C * c_final[i])            // 复杂度
            * (1.0 + SCORE_BETA_C  * c_rel_pos[i])          // 相对彩度
            * l_weight                                        // 亮度联动
            * (1.0 + SCORE_BETA_U  * u_norm[i])              // 独特性
            * bg_penalty;                                    // 背景惩罚

        // ---- 白/浅灰背景专用门控 ----
        // 三个条件同时满足才触发, 不会误伤白衣服、银发等主体色
        if c.chroma < 5.0 && c.lab_l > 85.0 && c.border_connected_ratio > 0.4 {
            c.dominant_score *= SCORE_WHITE_GATE;
        }
    }
}

/// BFS from all border pixels to compute border-connected pixel counts.
/// `assignments` – flat row-major cluster id per pixel.
fn compute_border_connected(
    assignments: &[usize],
    w: u32,
    h: u32,
    k: usize,
    _edge_count: &[usize],
) -> Vec<usize> {
    let wu = w as usize;
    let hu = h as usize;
    let n = wu * hu;

    // Collect border pixels (4 edges) and queue them
    let mut queue = VecDeque::new();
    let mut visited = vec![false; n];
    let mut connected = vec![0usize; k];

    // top & bottom rows
    for x in 0..wu {
        for &y in &[0, hu - 1] {
            let idx = y * wu + x;
            if !visited[idx] {
                visited[idx] = true;
                queue.push_back(idx);
            }
        }
    }
    // left & right columns (skip corners already done)
    for y in 1..(hu - 1) {
        for &x in &[0, wu - 1] {
            let idx = y * wu + x;
            if !visited[idx] {
                visited[idx] = true;
                queue.push_back(idx);
            }
        }
    }

    // Multi-source BFS — only traverse same-cluster neighbours
    while let Some(idx) = queue.pop_front() {
        let cl = assignments[idx];
        connected[cl] += 1;

        let x = idx % wu;
        let y = idx / wu;

        // 4-neighbours
        if x > 0 {
            let ni = y * wu + x - 1;
            if !visited[ni] && assignments[ni] == cl {
                visited[ni] = true;
                queue.push_back(ni);
            }
        }
        if x + 1 < wu {
            let ni = y * wu + x + 1;
            if !visited[ni] && assignments[ni] == cl {
                visited[ni] = true;
                queue.push_back(ni);
            }
        }
        if y > 0 {
            let ni = (y - 1) * wu + x;
            if !visited[ni] && assignments[ni] == cl {
                visited[ni] = true;
                queue.push_back(ni);
            }
        }
        if y + 1 < hu {
            let ni = (y + 1) * wu + x;
            if !visited[ni] && assignments[ni] == cl {
                visited[ni] = true;
                queue.push_back(ni);
            }
        }
    }

    connected
}

// ---------------------------------------------------------------------------
// Internal: run KMeans++ on 3D data (baseline, no spatial stats)
// ---------------------------------------------------------------------------

fn run_kmeans_3d(
    data: &[[f64; 3]],
    k: usize,
    rng_seed: u64,
) -> Result<(Vec<Cluster4D>, Cluster4D, Duration)> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok((vec![], make_empty_cluster(), Duration::ZERO));
    }

    let start = std::time::Instant::now();

    let flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let array = Array2::from_shape_vec((n, 3), flat)?;
    let dataset = DatasetBase::from(array);

    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1).tolerance(1e-3).max_n_iterations(50)
        .fit(&dataset)?;

    let assignments = model.predict(&dataset);
    let total = n as f64;
    let mut counts = vec![0usize; k];
    for &cl in assignments.iter() { counts[cl] += 1; }

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k)
        .map(|i| {
            let b = i * 3;
            let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
            let rgb = lab_to_rgb_norm(lab);
            let proportion = counts[i] as f64 / total;
            build_cluster(rgb, proportion, 0.0, lab, 0.0, 0.0)
        })
        .collect();

    let dominant_idx = clusters.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.proportion.partial_cmp(&b.proportion).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok((clusters, dominant, start.elapsed()))
}

pub fn kmeans_baseline(data: &[[f64; 3]], k: usize, rng_seed: u64) -> Result<ClusterResult4D> {
    let (clusters, dominant, duration) = run_kmeans_3d(data, k, rng_seed)?;
    Ok(ClusterResult4D { clusters, dominant, duration })
}

// ---------------------------------------------------------------------------
// K‑Means++ (4D) — with full spatial scoring
// ---------------------------------------------------------------------------

pub fn kmeans_plus_plus(
    data: &[[f64; 4]],
    k: usize,
    rng_seed: u64,
    width: u32,
    height: u32,
) -> Result<ClusterResult4D> {
    cluster_4d_impl(data, k, rng_seed, width, height, false)
}

// ---------------------------------------------------------------------------
// Mini‑Batch K‑Means (4D)
// ---------------------------------------------------------------------------

const BATCH_SIZE: usize = 2048;

pub fn mini_batch_kmeans(
    data: &[[f64; 4]],
    k: usize,
    rng_seed: u64,
    width: u32,
    height: u32,
) -> Result<ClusterResult4D> {
    cluster_4d_impl(data, k, rng_seed, width, height, true)
}

/// Shared 4D clustering: KMeans++ (full data) or MiniBatch (subset → predict all).
fn cluster_4d_impl(
    data: &[[f64; 4]],
    k: usize,
    rng_seed: u64,
    width: u32,
    height: u32,
    minibatch: bool,
) -> Result<ClusterResult4D> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok(ClusterResult4D { clusters: vec![], dominant: make_empty_cluster(), duration: Duration::ZERO });
    }

    let start = std::time::Instant::now();

    // Train on full or batch
    let (model, _train_n) = if minibatch {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
        indices.as_mut_slice().shuffle(&mut rng);
        let batch_n = BATCH_SIZE.min(n);
        let batch: Vec<f64> = indices[..batch_n].iter()
            .flat_map(|&i| [data[i][0], data[i][1], data[i][2], data[i][3]])
            .collect();
        let arr = Array2::from_shape_vec((batch_n, 4), batch)?;
        let ds = DatasetBase::from(arr);
        let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
        let model = KMeans::params_with_rng(k, rng)
            .n_runs(1).tolerance(1e-3).max_n_iterations(50)
            .fit(&ds)?;
        (model, batch_n)
    } else {
        let flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3]]).collect();
        let arr = Array2::from_shape_vec((n, 4), flat)?;
        let ds = DatasetBase::from(arr);
        let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
        let model = KMeans::params_with_rng(k, rng)
            .n_runs(1).tolerance(1e-3).max_n_iterations(50)
            .fit(&ds)?;
        (model, n)
    };

    // Predict all
    let full_flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3]]).collect();
    let full_arr = Array2::from_shape_vec((n, 4), full_flat)?;
    let full_ds = DatasetBase::new(full_arr, ());
    let assignments = model.predict(&full_ds);

    let total = n as f64;
    let mut counts = vec![0usize; k];
    let mut sum_c = vec![0.0f64; k];
    let mut sum_x = vec![0.0f64; k];
    let mut sum_y = vec![0.0f64; k];
    let mut sum_x2 = vec![0.0f64; k];
    let mut sum_y2 = vec![0.0f64; k];
    let mut sum_dist = vec![0.0f64; k];
    let mut edge_count = vec![0usize; k];
    let mut assign_vec = vec![0usize; n];
    let w = width as usize;
    let h = height as usize;
    let cx = w as f64 * 0.5;
    let cy = h as f64 * 0.5;

    for (i, &cl) in assignments.iter().enumerate() {
        let cl = cl;
        counts[cl] += 1;
        sum_c[cl] += data[i][3];
        let px = (i % w) as f64;
        let py = (i / w) as f64;
        sum_x[cl] += px;
        sum_y[cl] += py;
        sum_x2[cl] += px * px;
        sum_y2[cl] += py * py;
        sum_dist[cl] += ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
        assign_vec[i] = cl;
        // edge pixel?
        let x = i % w;
        let y = i / w;
        if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
            edge_count[cl] += 1;
        }
    }

    // Connectivity
    let border_connected = compute_border_connected(&assign_vec, width, height, k, &edge_count);

    // Centroids
    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k).map(|i| {
        let b = i * 4;
        let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
        let rgb = lab_to_rgb_norm(lab);
        let proportion = if counts[i] > 0 { counts[i] as f64 / total } else { 0.0 };
        let avg_c = if counts[i] > 0 { sum_c[i] / counts[i] as f64 } else { 0.0 };
        let avg_x = if counts[i] > 0 { sum_x[i] / counts[i] as f64 } else { 0.0 };
        let avg_y = if counts[i] > 0 { sum_y[i] / counts[i] as f64 } else { 0.0 };
        build_cluster(rgb, proportion, avg_c, lab, avg_x, avg_y)
    }).collect();

    let score_ctx = ScoreCtx {
        k,
        w: width,
        h: height,
        counts,
        sum_x,
        sum_y,
        sum_x2,
        sum_y2,
        sum_dist_centre: sum_dist,
        edge_count,
        border_connected,
    };
    score_clusters(&mut clusters, &score_ctx);

    let dominant_idx = clusters.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.dominant_score.partial_cmp(&b.dominant_score).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok(ClusterResult4D { clusters, dominant, duration: start.elapsed() })
}

// ---------------------------------------------------------------------------
// Internal: run KMeans on minibatch subset (3D, baseline)
// ---------------------------------------------------------------------------

fn run_minibatch_3d(
    data: &[[f64; 3]],
    k: usize,
    rng_seed: u64,
) -> Result<(Vec<Cluster4D>, Cluster4D, Duration)> {
    let n = data.len();
    let k = k.min(n);
    if k == 0 { return Ok((vec![], make_empty_cluster(), Duration::ZERO)); }

    let start = std::time::Instant::now();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
    indices.as_mut_slice().shuffle(&mut rng);
    let batch_n = BATCH_SIZE.min(n);
    let batch_data: Vec<[f64; 3]> = indices[..batch_n].iter().map(|&i| data[i]).collect();

    let flat: Vec<f64> = batch_data.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let array = Array2::from_shape_vec((batch_n, 3), flat)?;
    let ds = DatasetBase::from(array);
    let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
    let model = KMeans::params_with_rng(k, rng)
        .n_runs(1).tolerance(1e-3).max_n_iterations(50)
        .fit(&ds)?;

    let full_flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let full_arr = Array2::from_shape_vec((n, 3), full_flat)?;
    let full_ds = DatasetBase::new(full_arr, ());
    let assignments = model.predict(&full_ds);

    let total = n as f64;
    let mut counts = vec![0usize; k];
    for &cl in assignments.iter() { counts[cl] += 1; }

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k).map(|i| {
        let b = i * 3;
        let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
        let rgb = lab_to_rgb_norm(lab);
        build_cluster(rgb, counts[i] as f64 / total, 0.0, lab, 0.0, 0.0)
    }).collect();

    let dominant_idx = clusters.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.proportion.partial_cmp(&b.proportion).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok((clusters, dominant, start.elapsed()))
}

pub fn minibatch_baseline(data: &[[f64; 3]], k: usize, rng_seed: u64) -> Result<ClusterResult4D> {
    let (clusters, dominant, duration) = run_minibatch_3d(data, k, rng_seed)?;
    Ok(ClusterResult4D { clusters, dominant, duration })
}

// ---------------------------------------------------------------------------
// K‑Means++ (6D)
// ---------------------------------------------------------------------------

pub fn kmeans_plus_plus_6d(
    data: &[[f64; 6]],
    k: usize,
    rng_seed: u64,
    width: u32,
    height: u32,
) -> Result<ClusterResult4D> {
    cluster_6d_impl(data, k, rng_seed, width, height, false)
}

/// Mini‑Batch K‑Means on 6‑D data.
pub fn mini_batch_kmeans_6d(
    data: &[[f64; 6]],
    k: usize,
    rng_seed: u64,
    width: u32,
    height: u32,
) -> Result<ClusterResult4D> {
    cluster_6d_impl(data, k, rng_seed, width, height, true)
}

fn cluster_6d_impl(
    data: &[[f64; 6]],
    k: usize,
    rng_seed: u64,
    width: u32,
    height: u32,
    minibatch: bool,
) -> Result<ClusterResult4D> {
    let n = data.len();
    let k = k.min(n);

    if k == 0 {
        return Ok(ClusterResult4D { clusters: vec![], dominant: make_empty_cluster(), duration: Duration::ZERO });
    }

    let start = std::time::Instant::now();

    let (model, _train_n) = if minibatch {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
        indices.as_mut_slice().shuffle(&mut rng);
        let batch_n = BATCH_SIZE.min(n);
        let batch: Vec<f64> = indices[..batch_n].iter()
            .flat_map(|&i| [data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]])
            .collect();
        let arr = Array2::from_shape_vec((batch_n, 6), batch)?;
        let ds = DatasetBase::from(arr);
        let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
        let model = KMeans::params_with_rng(k, rng)
            .n_runs(1).tolerance(1e-3).max_n_iterations(50)
            .fit(&ds)?;
        (model, batch_n)
    } else {
        let flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
        let arr = Array2::from_shape_vec((n, 6), flat)?;
        let ds = DatasetBase::from(arr);
        let rng = Xoshiro256Plus::seed_from_u64(rng_seed);
        let model = KMeans::params_with_rng(k, rng)
            .n_runs(1).tolerance(1e-3).max_n_iterations(50)
            .fit(&ds)?;
        (model, n)
    };

    let full_flat: Vec<f64> = data.iter().flat_map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
    let full_arr = Array2::from_shape_vec((n, 6), full_flat)?;
    let full_ds = DatasetBase::new(full_arr, ());
    let assignments = model.predict(&full_ds);

    let total = n as f64;
    let mut counts = vec![0usize; k];
    let mut sum_c = vec![0.0f64; k];
    let mut sum_x = vec![0.0f64; k];
    let mut sum_y = vec![0.0f64; k];
    let mut sum_x2 = vec![0.0f64; k];
    let mut sum_y2 = vec![0.0f64; k];
    let mut sum_dist = vec![0.0f64; k];
    let mut edge_count = vec![0usize; k];
    let mut assign_vec = vec![0usize; n];
    let w = width as usize;
    let h = height as usize;
    let cx = w as f64 * 0.5;
    let cy = h as f64 * 0.5;

    for (i, &cl) in assignments.iter().enumerate() {
        let cl = cl;
        counts[cl] += 1;
        sum_c[cl] += data[i][3];
        let px = data[i][4];
        let py = data[i][5];
        sum_x[cl] += px;
        sum_y[cl] += py;
        sum_x2[cl] += px * px;
        sum_y2[cl] += py * py;
        sum_dist[cl] += ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
        assign_vec[i] = cl;
        // edge pixel — derive from original index
        let x = i % w;
        let y = i / w;
        if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
            edge_count[cl] += 1;
        }
    }

    let border_connected = compute_border_connected(&assign_vec, width, height, k, &edge_count);

    let centroid_slice = model.centroids().as_slice().unwrap();
    let mut clusters: Vec<Cluster4D> = (0..k).map(|i| {
        let b = i * 6;
        let lab = [centroid_slice[b], centroid_slice[b + 1], centroid_slice[b + 2]];
        let rgb = lab_to_rgb_norm(lab);
        let proportion = if counts[i] > 0 { counts[i] as f64 / total } else { 0.0 };
        let avg_c = if counts[i] > 0 { sum_c[i] / counts[i] as f64 } else { 0.0 };
        let avg_x = if counts[i] > 0 { sum_x[i] / counts[i] as f64 } else { 0.0 };
        let avg_y = if counts[i] > 0 { sum_y[i] / counts[i] as f64 } else { 0.0 };
        build_cluster(rgb, proportion, avg_c, lab, avg_x, avg_y)
    }).collect();

    let score_ctx = ScoreCtx {
        k,
        w: width,
        h: height,
        counts,
        sum_x,
        sum_y,
        sum_x2,
        sum_y2,
        sum_dist_centre: sum_dist,
        edge_count,
        border_connected,
    };
    score_clusters(&mut clusters, &score_ctx);

    let dominant_idx = clusters.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.dominant_score.partial_cmp(&b.dominant_score).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    let dominant = clusters[dominant_idx].clone();
    sort_by_lightness(&mut clusters);

    Ok(ClusterResult4D { clusters, dominant, duration: start.elapsed() })
}
