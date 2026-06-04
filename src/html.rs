use crate::algorithms::{Algorithm, AlgorithmResult};
use crate::colorspace::ColorSpace;
use crate::img::LoadedImage;
use anyhow::Result;
use std::io::Write;

/// Generate a self-contained HTML file showing all color extraction results.
pub fn generate(
    images: &[LoadedImage],
    all_results: &[(usize, Algorithm, ColorSpace, &AlgorithmResult)],
    output_path: &str,
) -> Result<()> {
    let mut html = String::with_capacity(1_000_000);
    write_html_header(&mut html);
    write_html_body(&mut html, images, all_results);
    write_html_footer(&mut html);

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(html.as_bytes())?;
    Ok(())
}

fn write_html_header(html: &mut String) {
    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Color Extraction Results</title>
<style>
:root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card-hover: #1f1f3a;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6c63ff;
    --border: #2a2a4a;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.5;
    padding: 20px;
}
header {
    text-align: center;
    padding: 40px 20px;
    border-bottom: 2px solid var(--border);
    margin-bottom: 30px;
}
header h1 { font-size: 2rem; color: var(--accent); margin-bottom: 8px; }
header p { color: var(--text-dim); }
.summary-stats {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 15px;
    flex-wrap: wrap;
}
.stat { text-align: center; }
.stat-value { font-size: 2rem; font-weight: bold; color: var(--accent); }
.stat-label { font-size: 0.8rem; color: var(--text-dim); text-transform: uppercase; }
.image-section {
    background: var(--card);
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 30px;
    border: 1px solid var(--border);
}
.image-section h2 {
    font-size: 1.4rem;
    margin-bottom: 5px;
    color: #fff;
}
.image-meta {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-bottom: 15px;
}
.image-preview {
    max-width: 100%;
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
}
.algorithm-section {
    margin-bottom: 25px;
}
.algorithm-section h3 {
    font-size: 1.15rem;
    color: var(--accent);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 15px;
}
.colorspace-group {
    background: var(--card-hover);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 15px;
    border: 1px solid var(--border);
}
.colorspace-group h4 {
    font-size: 1rem;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.runtime-badge {
    background: #2a2a4a;
    color: #aaa;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: normal;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.dominant-section {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 18px;
    padding: 12px;
    background: rgba(108, 99, 255, 0.08);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
}
.dominant-label {
    font-size: 0.8rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    white-space: nowrap;
}
.dominant-swatch {
    width: 80px;
    height: 80px;
    border-radius: 10px;
    border: 2px solid var(--border);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.dominant-info {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.dominant-hex {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.95rem;
    font-weight: 600;
}
.dominant-proportion {
    font-size: 0.75rem;
    color: var(--text-dim);
}
.palette-section { margin-top: 5px; }
.palette-section h5 {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.palette {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.swatch-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    width: 90px;
}
.color-block {
    width: 90px;
    height: 55px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    transition: transform 0.15s;
}
.color-block:hover { transform: scale(1.08); }
.swatch-hex {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.7rem;
    color: #ccc;
}
.swatch-lab {
    font-size: 0.65rem;
    color: var(--text-dim);
}
.swatch-pct {
    font-size: 0.6rem;
    color: var(--text-dim);
}
footer {
    text-align: center;
    padding: 30px;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 30px;
}
</style>
</head>
"#,
    );
}

fn write_html_body(
    html: &mut String,
    images: &[LoadedImage],
    all_results: &[(usize, Algorithm, ColorSpace, &AlgorithmResult)],
) {
    let total_combinations = all_results.len();
    let total_images = images.len();
    let total_algorithms = Algorithm::all().len();
    let total_spaces = ColorSpace::all().len();

    html.push_str("<body>\n");

    // Header
    html.push_str("<header>\n");
    html.push_str("<h1>🎨 Color Extraction Results</h1>\n");
    html.push_str(&format!(
        "<p>{} images × {} algorithms × {} color spaces = {} combinations</p>\n",
        total_images, total_algorithms, total_spaces, total_combinations
    ));

    // Summary stats
    let total_duration: std::time::Duration = all_results
        .iter()
        .map(|(_, _, _, r)| r.duration)
        .sum();
    let avg_duration = total_duration / total_combinations.max(1) as u32;
    html.push_str("<div class=\"summary-stats\">\n");
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{total_images}</div><div class=\"stat-label\">Images</div></div>\n"
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{total_combinations}</div><div class=\"stat-label\">Combinations</div></div>\n"
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{:.1}s</div><div class=\"stat-label\">Total Time</div></div>\n",
        total_duration.as_secs_f64()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{:.0}ms</div><div class=\"stat-label\">Avg per Combo</div></div>\n",
        avg_duration.as_millis()
    ));
    html.push_str("</div>\n");
    html.push_str("</header>\n");

    // For each image
    for (img_idx, img) in images.iter().enumerate() {
        html.push_str("<section class=\"image-section\">\n");
        html.push_str(&format!(
            "<h2>{}</h2>\n",
            html_escape(&img.filename)
        ));
        html.push_str(&format!(
            "<div class=\"image-meta\">Dimensions: {}×{} ({} pixels) | Resized to fit 1024×1024</div>\n",
            img.width,
            img.height,
            img.width * img.height
        ));
        html.push_str(&format!(
            "<img class=\"image-preview\" src=\"{}\" alt=\"{}\" loading=\"lazy\" />\n",
            img.base64_preview,
            html_escape(&img.filename)
        ));

        // For each algorithm
        for algo in Algorithm::all().iter() {
            html.push_str("<section class=\"algorithm-section\">\n");
            html.push_str(&format!("<h3>{}</h3>\n", algo.name()));

            // For each color space
            for cs in ColorSpace::all().iter() {
                // Find matching result
                if let Some((_, _, _, result)) = all_results
                    .iter()
                    .find(|(i, a, c, _)| *i == img_idx && *a == *algo && *c == *cs)
                {
                    write_colorspace_group(html, cs, result);
                }
            }

            html.push_str("</section>\n");
        }

        html.push_str("</section>\n");
    }
}

fn write_colorspace_group(html: &mut String, cs: &ColorSpace, result: &AlgorithmResult) {
    let ms = result.duration.as_millis();
    let time_str = if ms >= 1000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    };

    html.push_str("<div class=\"colorspace-group\">\n");
    html.push_str(&format!(
        "<h4>{} <span class=\"runtime-badge\">{}</span></h4>\n",
        cs.name(),
        time_str
    ));

    // Dominant color
    html.push_str("<div class=\"dominant-section\">\n");
    html.push_str("<span class=\"dominant-label\">Dominant</span>\n");
    html.push_str(&format!(
        "<div class=\"dominant-swatch\" style=\"background:{};\"></div>\n",
        result.dominant.hex
    ));
    html.push_str("<div class=\"dominant-info\">\n");
    html.push_str(&format!(
        "<span class=\"dominant-hex\">{}</span>\n",
        result.dominant.hex
    ));
    html.push_str(&format!(
        "<span class=\"dominant-proportion\">{:.1}% of pixels | L* {:.1}</span>\n",
        result.dominant.proportion * 100.0,
        result.dominant.lab_l
    ));
    html.push_str("</div>\n");
    html.push_str("</div>\n");

    // Palette
    html.push_str("<div class=\"palette-section\">\n");
    html.push_str("<h5>Palette (dark → light)</h5>\n");
    html.push_str("<div class=\"palette\">\n");

    for entry in &result.palette {
        html.push_str("<div class=\"swatch-card\">\n");
        html.push_str(&format!(
            "<div class=\"color-block\" style=\"background:{};\"></div>\n",
            entry.hex
        ));
        html.push_str(&format!(
            "<span class=\"swatch-hex\">{}</span>\n",
            entry.hex
        ));
        html.push_str(&format!(
            "<span class=\"swatch-lab\">L* {:.1}</span>\n",
            entry.lab_l
        ));
        html.push_str(&format!(
            "<span class=\"swatch-pct\">{:.1}%</span>\n",
            entry.proportion * 100.0
        ));
        html.push_str("</div>\n");
    }

    html.push_str("</div>\n");
    html.push_str("</div>\n");

    html.push_str("</div>\n");
}

fn write_html_footer(html: &mut String) {
    html.push_str("<footer>\n");
    html.push_str("<p>Generated by color-extract — Rust + palette + linfa-clustering + rayon</p>\n");
    html.push_str("</footer>\n");
    html.push_str("</body>\n</html>\n");
}

/// Simple HTML entity escaping.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
