mod cluster_4d;
mod dct;
mod html;
pub mod img;

use cluster_4d::ClusterResult4D;
use html::ImageResult;
use palette::{IntoColor, Lab, Srgb};
use std::time::Instant;

fn rgb_to_lab(rgb: [f64; 3]) -> [f64; 3] {
    let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
    let lab: Lab = srgb.into_color();
    [lab.l as f64, lab.a as f64, lab.b as f64]
}

fn main() -> anyhow::Result<()> {
    let max_dim: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "output/results-dct.html".to_string());

    println!("═══ DCT‑Enhanced Color Extraction ═══");
    println!("Loading images from imgs/ (max dim: {max_dim})…");
    let start_total = Instant::now();
    let images = img::load_all("imgs", max_dim)?;
    println!(
        "Loaded {} image(s) in {:?}",
        images.len(),
        start_total.elapsed()
    );

    let k = 10;
    let rng_seed = 42;
    let n_images = images.len();

    // Consume each image → compute DCT → cluster → produce ImageResult
    let image_results: Vec<ImageResult> = images
        .into_iter()
        .enumerate()
        .map(|(idx, img)| {
            println!(
                "\n── [{}/{}] {} ({}×{}, {} px) ──",
                idx + 1,
                n_images,
                img.filename,
                img.width,
                img.height,
                img.width * img.height
            );

            // 1. Compute DCT complexity map
            println!("   Computing DCT complexity map…");
            let c_start = Instant::now();
            let complexity = dct::compute_complexity_map(&img.pixels, img.width, img.height);
            println!("   → done in {:?}", c_start.elapsed());

            // 2. Convert RGB → CIELAB
            let lab_pixels: Vec<[f64; 3]> = img.pixels.iter().map(|&p| rgb_to_lab(p)).collect();

            // 3. Baseline (3D) — no complexity
            println!("   Running K‑Means++ (Baseline)…");
            let kmeans_base = cluster_4d::kmeans_baseline(&lab_pixels, k, rng_seed)
                .expect("KMeans++ baseline failed");
            print_result_summary("KMeans++(base)", &kmeans_base);

            println!("   Running Mini‑Batch K‑Means (Baseline)…");
            let minibatch_base = cluster_4d::minibatch_baseline(&lab_pixels, k, rng_seed)
                .expect("MiniBatch baseline failed");
            print_result_summary("MiniBatch(base)", &minibatch_base);

            // 4. Build 4‑D dataset [L, a, b, c×20]
            // Scale c by 20× so its magnitude competes with LAB dimensions
            let data_4d: Vec<[f64; 4]> = lab_pixels
                .iter()
                .zip(complexity.iter())
                .map(|(&[l, a, b], &c)| [l, a, b, c * 20.0])
                .collect();

            // 5. K‑Means++ (4D)
            println!("   Running K‑Means++ (with c)…");
            let kmeans = cluster_4d::kmeans_plus_plus(&data_4d, k, rng_seed, img.width, img.height)
                .expect("KMeans++ failed");
            print_result_summary("KMeans++(c)", &kmeans);

            // 6. Mini‑Batch K‑Means (4D)
            println!("   Running Mini‑Batch K‑Means (with c)…");
            let minibatch = cluster_4d::mini_batch_kmeans(&data_4d, k, rng_seed, img.width, img.height)
                .expect("MiniBatch failed");
            print_result_summary("MiniBatch(c)", &minibatch);

            // 7. Build 6‑D dataset [L, a, b, c×20, nx×10, ny×10]
            let w = img.width as f64;
            let h = img.height as f64;
            let data_6d: Vec<[f64; 6]> = lab_pixels
                .iter()
                .zip(complexity.iter())
                .enumerate()
                .map(|(i, (&[l, a, b], &c))| {
                    let px = (i as u32 % img.width) as f64 / w * 10.0;
                    let py = (i as u32 / img.width) as f64 / h * 10.0;
                    [l, a, b, c * 20.0, px, py]
                })
                .collect();

            // 8. K‑Means++ (6D) — with coordinates
            println!("   Running K‑Means++ (c + xy)…");
            let kmeans_6d = cluster_4d::kmeans_plus_plus_6d(&data_6d, k, rng_seed, img.width, img.height)
                .expect("KMeans++ 6D failed");
            print_result_summary("KMeans++(c+xy)", &kmeans_6d);

            // 9. Mini‑Batch K‑Means (6D) — with coordinates
            println!("   Running Mini‑Batch K‑Means (c + xy)…");
            let minibatch_6d = cluster_4d::mini_batch_kmeans_6d(&data_6d, k, rng_seed, img.width, img.height)
                .expect("MiniBatch 6D failed");
            print_result_summary("MiniBatch(c+xy)", &minibatch_6d);

            ImageResult {
                img,
                kmeans,
                minibatch,
                kmeans_base,
                minibatch_base,
                kmeans_6d,
                minibatch_6d,
            }
        })
        .collect();

    println!("\nGenerating HTML ({output_path})…");
    html::generate(&image_results, &output_path)?;

    let total_elapsed = start_total.elapsed();
    println!(
        "Done! Total time: {:.2}s. Open {output_path} in your browser.",
        total_elapsed.as_secs_f64()
    );

    Ok(())
}

fn print_result_summary(name: &str, result: &ClusterResult4D) {
    let dominant = &result.dominant;
    println!(
        "   {name} done in {:?} | dominant: {} ({:.1}%) | {} clusters",
        result.duration,
        dominant.hex,
        dominant.proportion * 100.0,
        result.clusters.len()
    );
}
