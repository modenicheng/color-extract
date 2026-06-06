use crate::cluster_4d::ClusterResult4D;
use crate::img::LoadedImage;
use anyhow::Result;
use std::io::Write;

/// A single image's results from both algorithms, with/without complexity.
pub struct ImageResult {
    pub img: LoadedImage,
    pub kmeans: ClusterResult4D,      // KMeans++ (with c)
    pub minibatch: ClusterResult4D,   // Mini-Batch (with c)
    pub kmeans_base: ClusterResult4D, // KMeans++ (no c) — baseline
    pub minibatch_base: ClusterResult4D, // Mini-Batch (no c) — baseline
    pub kmeans_6d: ClusterResult4D,   // KMeans++ (c + xy)
    pub minibatch_6d: ClusterResult4D, // Mini-Batch (c + xy)
}

/// Generate a self‑contained HTML file.
pub fn generate(results: &[ImageResult], output_path: &str) -> Result<()> {
    let mut html = String::with_capacity(1_000_000);
    write_header(&mut html);
    write_body(&mut html, results);
    write_footer(&mut html);

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(html.as_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// HTML template
// ---------------------------------------------------------------------------

fn write_header(html: &mut String) {
    html.push_str(
        r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DCT‑Enhanced Color Extraction</title>
<style>
:root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card-hover: #1f1f3a;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6c63ff;
    --complexity: #ff9f43;
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
header .subtitle {
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-top: 6px;
}
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
.algo-section {
    margin-bottom: 25px;
}
.algo-section h3 {
    font-size: 1.15rem;
    color: var(--accent);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 15px;
}
.algo-desc {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 12px;
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
.palette-section h4 {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.palette {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
}
.swatch-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    width: 76px;
}
.color-block {
    width: 76px;
    height: 42px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 1px 6px rgba(0,0,0,0.2);
    transition: transform 0.15s;
}
.color-block:hover { transform: scale(1.08); }
.swatch-hex {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.62rem;
    color: #ccc;
}
.swatch-lab {
    font-size: 0.6rem;
    color: var(--text-dim);
}
.swatch-pct {
    font-size: 0.55rem;
    color: var(--text-dim);
}
.complexity-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255, 159, 67, 0.12);
    color: var(--complexity);
    padding: 0 6px;
    border-radius: 8px;
    font-size: 0.58rem;
    font-weight: 600;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.runtime-badge {
    background: #2a2a4a;
    color: #aaa;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: normal;
    font-family: 'SF Mono', 'Fira Code', monospace;
    margin-left: 8px;
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

fn write_body(html: &mut String, results: &[ImageResult]) {
    let total = results.len();

    html.push_str("<body>\n");
    html.push_str("<header>\n");
    html.push_str("<h1>🎨 DCT‑Enhanced Color Extraction</h1>\n");
    html.push_str("<p>K‑Means++ &amp; Mini‑Batch K‑Means — CIELAB ± DCT complexity ± Coordinates (3D / 4D / 6D)</p>\n");
    html.push_str("<div class=\"subtitle\">\n");
    html.push_str("Each pixel is represented in <strong>CIELAB</strong> colour space augmented with a 4<sup>th</sup> dimension <strong>c</strong> = local high‑frequency ratio from DCT,<br>\n");
    html.push_str("and optionally <strong>normalised pixel coordinates (x, y)</strong> as the 5<sup>th</sup> &amp; 6<sup>th</sup> dimensions,<br>\n");
    html.push_str("helping the clustering distinguish regions with similar colour but different textures or spatial positions.\n");
    html.push_str("</div>\n");

    let total_duration: std::time::Duration = results
        .iter()
        .flat_map(|r| [r.kmeans.duration, r.minibatch.duration, r.kmeans_base.duration, r.minibatch_base.duration, r.kmeans_6d.duration, r.minibatch_6d.duration])
        .sum();
    let count = results.len() * 6;
    let avg_duration = if count > 0 { total_duration / count as u32 } else { std::time::Duration::ZERO };

    html.push_str("<div class=\"summary-stats\">\n");
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{total}</div><div class=\"stat-label\">Images</div></div>\n"
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{}</div><div class=\"stat-label\">Clusters per Image</div></div>\n",
        results.first().map(|r| r.kmeans.clusters.len()).unwrap_or(0)
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{:.1}s</div><div class=\"stat-label\">Total Time</div></div>\n",
        total_duration.as_secs_f64()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-value\">{:.0}ms</div><div class=\"stat-label\">Avg per Run</div></div>\n",
        avg_duration.as_millis()
    ));
    html.push_str("</div>\n");
    html.push_str("</header>\n");

    for res in results {
        html.push_str("<section class=\"image-section\">\n");
        html.push_str(&format!(
            "<h2>{}</h2>\n",
            html_escape(&res.img.filename)
        ));
        html.push_str(&format!(
            "<div class=\"image-meta\">{}×{} px ({} pixels)</div>\n",
            res.img.width,
            res.img.height,
            res.img.width * res.img.height
        ));
        html.push_str(&format!(
            "<img class=\"image-preview\" src=\"{}\" alt=\"{}\" loading=\"lazy\" />\n",
            res.img.base64_preview,
            html_escape(&res.img.filename)
        ));

        // Baseline (no c) sections
        write_algo_section(html, "K‑Means++ (Baseline)", "kmeans++ — 仅 CIELAB 3D，无复杂度维度", &res.kmeans_base);
        write_algo_section(html, "Mini‑Batch K‑Means (Baseline)", "minibatch — 仅 CIELAB 3D，无复杂度维度", &res.minibatch_base);

        // K‑Means++ section
        write_algo_section(html, "K‑Means++ (with c)", "kmeans++ — CIELAB 4D，含 DCT 复杂度", &res.kmeans);

        // Mini‑Batch K‑Means section
        write_algo_section(html, "Mini‑Batch K‑Means (with c)", "minibatch — CIELAB 4D，含 DCT 复杂度", &res.minibatch);

        // K‑Means++ 6D section (new)
        write_algo_section(html, "K‑Means++ (c + xy)", "kmeans++ — CIELAB 6D，含 DCT 复杂度 + 像素坐标 (x,y)", &res.kmeans_6d);

        // Mini‑Batch K‑Means 6D section (new)
        write_algo_section(html, "Mini‑Batch K‑Means (c + xy)", "minibatch — CIELAB 6D，含 DCT 复杂度 + 像素坐标 (x,y)", &res.minibatch_6d);

        html.push_str("</section>\n");
    }
}

fn write_algo_section(html: &mut String, name: &str, desc: &str, result: &ClusterResult4D) {
    html.push_str("<section class=\"algo-section\">\n");

    let ms = result.duration.as_millis();
    let time_str = if ms >= 1000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    };

    html.push_str(&format!(
        "<h3>{} <span class=\"runtime-badge\">{}</span></h3>\n",
        name, time_str
    ));
    html.push_str(&format!("<div class=\"algo-desc\">{}</div>\n", desc));

    // Dominant colour — use already-computed dominant_score (softmax(c) if applicable)
    let score = result.dominant.dominant_score;
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
    let coord_info = if result.dominant.avg_x > 0.0 || result.dominant.avg_y > 0.0 {
        format!(" | xy=({:.3},{:.3})", result.dominant.avg_x, result.dominant.avg_y)
    } else {
        String::new()
    };
    html.push_str(&format!(
        "<span class=\"dominant-proportion\">{:.1}% of pixels | L* {:.1} | c={:.3}{}</span>\n",
        result.dominant.proportion * 100.0,
        result.dominant.lab_l,
        result.dominant.avg_complexity,
        coord_info
    ));
    html.push_str(&format!(
        "<span class=\"dominant-proportion\" style=\"color:var(--complexity);font-weight:600;\">score = softmax(p) × softmax(c) = {:.4}</span>\n",
        score
    ));
    html.push_str("</div>\n");
    html.push_str("</div>\n");

    // Full palette
    html.push_str("<div class=\"palette-section\">\n");
    html.push_str("<h4>All Clusters (dark → light)</h4>\n");
    html.push_str("<div class=\"palette\">\n");

    for cluster in &result.clusters {
        html.push_str("<div class=\"swatch-card\">\n");
        html.push_str(&format!(
            "<div class=\"color-block\" style=\"background:{};\"></div>\n",
            cluster.hex
        ));
        html.push_str(&format!(
            "<span class=\"swatch-hex\">{}</span>\n",
            cluster.hex
        ));
        html.push_str(&format!(
            "<span class=\"swatch-lab\">L* {:.1}</span>\n",
            cluster.lab_l
        ));
        html.push_str(&format!(
            "<span class=\"swatch-pct\">{:.1}%</span>\n",
            cluster.proportion * 100.0
        ));
        // Complexity badge
        html.push_str(&format!(
            "<span class=\"complexity-badge\">c={:.3}</span>\n",
            cluster.avg_complexity
        ));
        let coord_str = if cluster.avg_x > 0.0 || cluster.avg_y > 0.0 {
            format!("xy=({:.3},{:.3})", cluster.avg_x, cluster.avg_y)
        } else {
            String::new()
        };
        if !coord_str.is_empty() {
            html.push_str(&format!(
                "<span class=\"complexity-badge\" style=\"background:rgba(0,200,150,0.12);color:#0c9;\">{}</span>\n",
                coord_str
            ));
        }
        html.push_str(&format!(
            "<span class=\"complexity-badge\" style=\"background:rgba(108,99,255,0.1);color:var(--accent);\">s={:.4}</span>\n",
            cluster.dominant_score
        ));
        html.push_str("</div>\n");
    }

    html.push_str("</div>\n");
    html.push_str("</div>\n");

    html.push_str("</section>\n");
}

fn write_footer(html: &mut String) {
    html.push_str("<footer>\n");
    html.push_str("<p>Generated by dct‑extract — Rust + DCT complexity + linfa-clustering</p>\n");
    html.push_str("</footer>\n");
    html.push_str("</body>\n</html>\n");
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
