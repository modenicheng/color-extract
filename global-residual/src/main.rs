use anyhow::Context;
use image::{GenericImageView, ImageBuffer, ImageReader, Luma, Rgb};
use palette::{Hsl, IntoColor, Lab, Srgb};
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let max_dim: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let out_dir = "output/global-residual";

    println!("═══ Global‑Mean Residual ═══");
    println!(
        "Residual = |pixel − global_mean| per channel, max dim: {max_dim}, output: {out_dir}/"
    );
    println!("Loading images from imgs/ …");
    let start_total = Instant::now();

    std::fs::create_dir_all(out_dir).context("creating output directory")?;

    let dir = Path::new("imgs");
    if !dir.is_dir() {
        anyhow::bail!("'imgs' is not a directory");
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .context("reading imgs directory")?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let ext = e
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            ext == "jpg" || ext == "jpeg" || ext == "png"
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let n_images = entries.len();
    println!("Found {n_images} image(s)");

    for (idx, entry) in entries.iter().enumerate() {
        let path = entry.path();
        let filename = entry.file_name().to_string_lossy().to_string();

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&filename);

        println!("  [{}/{}] {filename}", idx + 1, n_images);

        let img = ImageReader::open(&path)
            .with_context(|| format!("opening {filename}"))?
            .decode()
            .with_context(|| format!("decoding {filename}"))?;

        let (w, h) = img.dimensions();
        let (nw, nh) = fit_dimensions(w, h, max_dim);
        let img = if nw != w || nh != h {
            img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };
        let (fw, fh) = img.dimensions();
        let rgb = img.to_rgb8();
        let total_px = (fw * fh) as f64;

        let start = Instant::now();

        // ── First pass: compute global means ──
        let mut sum_sat = 0.0f64;
        let mut sum_light = 0.0f64;
        let mut sum_l = 0.0f64;
        let mut sum_a = 0.0f64;
        let mut sum_b = 0.0f64;

        // Store per-pixel HSL and LAB values for second pass
        let mut hsl_vals: Vec<Hsl> = Vec::with_capacity((fw * fh) as usize);
        let mut lab_vals: Vec<Lab> = Vec::with_capacity((fw * fh) as usize);

        for pixel in rgb.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let srgb = Srgb::new(r, g, b);
            let hsl: Hsl = srgb.into_color();
            let lab: Lab = srgb.into_color();

            sum_sat += hsl.saturation as f64;
            sum_light += hsl.lightness as f64;
            sum_l += lab.l as f64;
            sum_a += lab.a as f64;
            sum_b += lab.b as f64;

            hsl_vals.push(hsl);
            lab_vals.push(lab);
        }

        let mean_sat = sum_sat / total_px;
        let mean_light = sum_light / total_px;
        let mean_l = sum_l / total_px;
        let mean_a = sum_a / total_px;
        let mean_b = sum_b / total_px;

        // ── Second pass: build residual images ──
        let mut sat_res = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);
        let mut light_res = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);
        let mut lab_res = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(fw, fh);

        let mut idx = 0usize;
        for y in 0..fh {
            for x in 0..fw {
                let hsl = &hsl_vals[idx];
                let lab = &lab_vals[idx];

                // HSL residuals (grayscale)
                let sd = ((hsl.saturation as f64 - mean_sat).abs() * 255.0).round() as u8;
                sat_res.put_pixel(x, y, Luma([sd.min(255)]));

                let ld = ((hsl.lightness as f64 - mean_light).abs() * 255.0).round() as u8;
                light_res.put_pixel(x, y, Luma([ld.min(255)]));

                // LAB residuals mapped to RGB
                // L*: 0..100 → map full range as fraction of its own range
                let lr = ((lab.l as f64 - mean_l).abs() / 100.0 * 255.0).round() as u8;
                // a*: −128..127 → span 255
                let ar = ((lab.a as f64 - mean_a).abs() / 255.0 * 255.0).round() as u8;
                // b*: −128..127 → span 255
                let br = ((lab.b as f64 - mean_b).abs() / 255.0 * 255.0).round() as u8;
                lab_res.put_pixel(x, y, Rgb([lr.min(255), ar.min(255), br.min(255)]));

                idx += 1;
            }
        }

        // ── Save ──
        let sat_path = format!("{out_dir}/{stem}_sat.png");
        sat_res
            .save(&sat_path)
            .context("saving saturation residual")?;

        let light_path = format!("{out_dir}/{stem}_light.png");
        light_res
            .save(&light_path)
            .context("saving lightness residual")?;

        let lab_path = format!("{out_dir}/{stem}_lab.png");
        lab_res.save(&lab_path).context("saving LAB residual")?;

        println!("    → sat residual (μ={mean_sat:.4}):  {sat_path}");
        println!("    → light residual (μ={mean_light:.4}): {light_path}");
        println!(
            "    → LAB residual  (μ_L={mean_l:.2} μ_a={mean_a:.2} μ_b={mean_b:.2}): {lab_path}"
        );
        println!("    → done in {:?}", start.elapsed());
    }

    let total = start_total.elapsed();
    println!(
        "\nDone! {n_images} image(s) processed in {:.2}s. Output in {out_dir}/",
        total.as_secs_f64()
    );
    Ok(())
}

fn fit_dimensions(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (w, h);
    }
    let scale = max_dim as f64 / w.max(h) as f64;
    let nw = (w as f64 * scale) as u32;
    let nh = (h as f64 * scale) as u32;
    (nw.max(1), nh.max(1))
}
