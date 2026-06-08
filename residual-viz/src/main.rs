use anyhow::Context;
use image::{GenericImageView, ImageBuffer, ImageReader, Luma, Rgb};
use palette::{Hsl, IntoColor, Lab, Srgb};
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let sigma: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(25.0);

    let max_dim: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let out_dir = "output/residual-viz";

    println!("═══ Saturation & Lightness Residual Viz ═══");
    println!("Gaussian sigma: {sigma}, max dim: {max_dim}, output: {out_dir}/");
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

        // Strip extension for output filenames
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

        // ── Extract Saturation & Lightness channels from HSL ──
        let start = Instant::now();
        let mut sat_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);
        let mut light_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let srgb = Srgb::new(r, g, b);
            let hsl: Hsl = srgb.into_color();
            sat_img.put_pixel(x, y, Luma([(hsl.saturation.clamp(0.0, 1.0) * 255.0) as u8]));
            light_img.put_pixel(x, y, Luma([(hsl.lightness.clamp(0.0, 1.0) * 255.0) as u8]));
        }

        // ── Gaussian blur ──
        let sat_blurred = image::imageops::blur(&sat_img, sigma);
        let light_blurred = image::imageops::blur(&light_img, sigma);

        // ── Compute |original - blurred| residual ──
        let mut sat_res = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);
        let mut light_res = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);

        for (x, y, _) in rgb.enumerate_pixels() {
            let so = sat_img.get_pixel(x, y)[0];
            let sb = sat_blurred.get_pixel(x, y)[0];
            let sd = (so as i16 - sb as i16).unsigned_abs() as u8;
            sat_res.put_pixel(x, y, Luma([sd]));

            let lo = light_img.get_pixel(x, y)[0];
            let lb = light_blurred.get_pixel(x, y)[0];
            let ld = (lo as i16 - lb as i16).unsigned_abs() as u8;
            light_res.put_pixel(x, y, Luma([ld]));
        }

        // ── Save ──
        let sat_path = format!("{out_dir}/{stem}_sat.png");
        sat_res.save(&sat_path).context("saving saturation residual")?;

        let light_path = format!("{out_dir}/{stem}_light.png");
        light_res.save(&light_path).context("saving lightness residual")?;

        // ── Extract L, a, b channels from CIELAB ──
        let mut l_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);
        let mut a_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);
        let mut b_img = ImageBuffer::<Luma<u8>, Vec<u8>>::new(fw, fh);

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let bl = pixel[2] as f32 / 255.0;
            let srgb = Srgb::new(r, g, bl);
            let lab: Lab = srgb.into_color();
            // L*: 0..100, a*: −128..128, b*: −128..128
            // Normalise each to 0..255 for grayscale image storage
            let l_norm = (lab.l / 100.0).clamp(0.0, 1.0);
            let a_norm = ((lab.a + 128.0) / 256.0).clamp(0.0, 1.0);
            let b_norm = ((lab.b + 128.0) / 256.0).clamp(0.0, 1.0);
            l_img.put_pixel(x, y, Luma([(l_norm * 255.0) as u8]));
            a_img.put_pixel(x, y, Luma([(a_norm * 255.0) as u8]));
            b_img.put_pixel(x, y, Luma([(b_norm * 255.0) as u8]));
        }

        // ── Gaussian blur each LAB channel ──
        let l_blurred = image::imageops::blur(&l_img, sigma);
        let a_blurred = image::imageops::blur(&a_img, sigma);
        let b_blurred = image::imageops::blur(&b_img, sigma);

        // ── Compute |original - blurred| residual for each LAB channel ──
        // Then map L‑residual → R, a‑residual → G, b‑residual → B
        let mut lab_res = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(fw, fh);

        for (x, y, _) in rgb.enumerate_pixels() {
            let lo = l_img.get_pixel(x, y)[0];
            let lb = l_blurred.get_pixel(x, y)[0];
            let lr = (lo as i16 - lb as i16).unsigned_abs() as u8;

            let ao = a_img.get_pixel(x, y)[0];
            let ab = a_blurred.get_pixel(x, y)[0];
            let ar = (ao as i16 - ab as i16).unsigned_abs() as u8;

            let bo = b_img.get_pixel(x, y)[0];
            let bb = b_blurred.get_pixel(x, y)[0];
            let br = (bo as i16 - bb as i16).unsigned_abs() as u8;

            lab_res.put_pixel(x, y, Rgb([lr, ar, br]));
        }

        // ── Save ──
        let sat_path = format!("{out_dir}/{stem}_sat.png");
        sat_res.save(&sat_path).context("saving saturation residual")?;

        let light_path = format!("{out_dir}/{stem}_light.png");
        light_res.save(&light_path).context("saving lightness residual")?;

        let lab_path = format!("{out_dir}/{stem}_lab.png");
        lab_res.save(&lab_path).context("saving LAB residual")?;

        println!("    → sat residual: {sat_path}");
        println!("    → light residual: {light_path}");
        println!("    → LAB residual:  {lab_path}  (R=Lₓ, G=aₓ, B=bₓ)");
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
