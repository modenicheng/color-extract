mod algorithms;
mod colorspace;
mod html;
mod img;
mod timing;

use algorithms::{Algorithm, AlgorithmResult};
use colorspace::ColorSpace;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let max_dim: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "output/results.html".to_string());

    println!("Loading images from imgs/ (max dim: {max_dim})...");
    let start_total = Instant::now();
    let images = img::load_all("imgs", max_dim)?;
    println!(
        "Loaded {} image(s) in {:?}",
        images.len(),
        start_total.elapsed()
    );

    let k = 10;
    let rng_seed = 42;

    // Build all tasks: images × algorithms × color spaces
    let tasks: Vec<(usize, Algorithm, ColorSpace, &[[f64; 3]])> = images
        .iter()
        .enumerate()
        .flat_map(|(i, img)| {
            Algorithm::all().into_iter().flat_map(move |algo| {
                ColorSpace::all()
                    .into_iter()
                    .map(move |cs| (i, algo, cs, img.pixels.as_slice()))
            })
        })
        .collect();

    let total = tasks.len();
    println!("Processing {total} combinations in parallel...");
    let proc_start = Instant::now();
    let counter = AtomicUsize::new(0);

    // Process all combinations in parallel
    let results: Vec<(usize, Algorithm, ColorSpace, anyhow::Result<AlgorithmResult>)> = tasks
        .par_iter()
        .map(|&(img_idx, algo, cs, pixels)| {
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            let algo_name = algo.name();
            let cs_name = cs.name();
            let result = algorithms::run_combination(pixels, algo, cs, k, rng_seed);
            match &result {
                Ok(r) => {
                    println!(
                        "[{n:>3}/{total}] {algo_name} + {cs_name} => {}ms | palette: {} colors, dominant: {}",
                        r.duration.as_millis(),
                        r.palette.len(),
                        r.dominant.hex
                    );
                }
                Err(e) => {
                    eprintln!("[ERR] {algo_name} + {cs_name} => {e}");
                }
            }
            (img_idx, algo, cs, result)
        })
        .collect();

    let proc_elapsed = proc_start.elapsed();
    println!(
        "All {total} combinations processed in {:.2}s",
        proc_elapsed.as_secs_f64()
    );

    // Filter out errors
    let success_results: Vec<(usize, Algorithm, ColorSpace, &AlgorithmResult)> = results
        .iter()
        .filter_map(|(i, a, c, r)| r.as_ref().ok().map(|ok| (*i, *a, *c, ok)))
        .collect();

    println!("Generating HTML ({output_path})...");
    html::generate(&images, &success_results, &output_path)?;

    let total_elapsed = start_total.elapsed();
    println!(
        "Done! Total time: {:.2}s. Open {output_path} in your browser.",
        total_elapsed.as_secs_f64()
    );

    Ok(())
}
