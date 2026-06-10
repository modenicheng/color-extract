// =============================================================================
// color-segment CLI — Color-based region segmentation visualization
// =============================================================================
//
// 加载 imgs/ 下的动漫/插画图片，执行颜色分割，生成每图独立的 HTML 可视化：
//   - Canvas 1: 原图 + 半透明区域着色叠加 + 区域边界
//   - Canvas 2: 边缘强度热力图
//   - 调色盘面板: 各区域色块 + 面积百分比
//
// 用法:
//   cargo run --release -p color-segment
//   cargo run --release -p color-segment -- <max_dim> <output_stem>
//
// 流程:
//   1. 读 params.yaml → SegmentParams
//   2. 遍历 imgs/ 下所有图片 (jpg/jpeg/png)
//   3. 每张图: 缩放 → color_segment::segment() → HTML 生成
//   4. 输出到 output/color-segment/{stem}_{name}.html

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageReader, RgbImage};
use rayon::prelude::*;
use std::io::Cursor;
use std::time::Instant;

use color_segment::params::SegmentParams;
use color_segment::{SegmentResult, segment};

// =============================================================================
// Base64 — inline implementation (no external crate available)
// =============================================================================

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// 标准 Base64 编码，将字节切片编码为字符串。
fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// 将 DynamicImage 编码为 JPEG base64 data URI。
fn img_to_jpeg_uri(img: &DynamicImage) -> Result<String> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg)
        .with_context(|| "encoding image to JPEG")?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        b64_encode(buf.into_inner().as_slice())
    ))
}

/// 将 DynamicImage 编码为 PNG base64 data URI。
fn img_to_png_uri(img: &DynamicImage) -> Result<String> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .with_context(|| "encoding image to PNG")?;
    Ok(format!(
        "data:image/png;base64,{}",
        b64_encode(buf.into_inner().as_slice())
    ))
}

// =============================================================================
// 可视化叠加图生成 — 区域着色 / 边界 / 边缘热力图
// =============================================================================

/// 生成区域着色叠加图：每个区域的像素以其调色盘颜色填充。
fn gen_colored_overlay(result: &SegmentResult) -> RgbImage {
    let w = result.width;
    let h = result.height;
    let mut img = RgbImage::new(w, h);
    let palette = &result.palette.colors;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let color = match result.labels[idx] {
                Some(rid) if rid < result.regions.len() => {
                    let cid = result.regions[rid].cluster_id;
                    if cid < palette.len() {
                        palette[cid]
                    } else {
                        [0, 0, 0]
                    }
                }
                _ => [0, 0, 0],
            };
            img.put_pixel(x, y, image::Rgb(color));
        }
    }
    img
}

/// 生成区域边界叠加图：边界处为暗色不透明，其余透明。
///
/// 边界定义：像素的 4-邻域中存在不同区域标签（含 None）。
fn gen_boundary_overlay(result: &SegmentResult) -> image::RgbaImage {
    let w = result.width;
    let h = result.height;
    let mut img = image::RgbaImage::new(w, h);
    let labels = &result.labels;

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let cur = labels[idx];

            // 非区域像素不画边界
            if cur.is_none() {
                img.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
                continue;
            }

            let is_boundary = (x > 0 && labels[idx - 1] != cur)
                || (x + 1 < w && labels[idx + 1] != cur)
                || (y > 0 && labels[idx - w as usize] != cur)
                || (y + 1 < h && labels[idx + w as usize] != cur);

            if is_boundary {
                img.put_pixel(x, y, image::Rgba([20, 20, 20, 255]));
            } else {
                img.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            }
        }
    }
    img
}

/// 生成边缘强度灰度热力图。
fn gen_edge_heatmap(result: &SegmentResult) -> image::GrayImage {
    let w = result.width;
    let h = result.height;
    let mut img = image::GrayImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let v = (result.edge_map[idx] * 255.0).round() as u8;
            img.put_pixel(x, y, image::Luma([v]));
        }
    }
    img
}

// =============================================================================
// 切分图 PNG 渲染 — 原图 + 半透明区域着色 + 边界线
// =============================================================================

/// 将分割结果渲染为一张合成 PNG：
///   1. 原图打底
///   2. 每个区域以调色盘颜色 45% 透明度叠加
///   3. 区域边界画 1px 暗色线
fn render_segmented_png(
    original: &RgbImage,
    labels: &[Option<usize>],
    palette: &color_segment::Palette,
    regions: &[color_segment::Region],
    width: u32,
    height: u32,
) -> RgbImage {
    let w = width;
    let h = height;
    let pal = &palette.colors;
    let mut out = RgbImage::new(w, h);

    // 每个像素：原图底色 + 45% 区域色叠加
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let orig = original.get_pixel(x, y).0;

            let over = match labels[idx] {
                Some(rid) if rid < regions.len() => {
                    let cid = regions[rid].cluster_id;
                    if cid < pal.len() { pal[cid] } else { [0, 0, 0] }
                }
                _ => [0, 0, 0],
            };
            // 45% alpha blend: out = over * 0.45 + orig * 0.55
            let r = (over[0] as f32 * 0.45 + orig[0] as f32 * 0.55).round() as u8;
            let g = (over[1] as f32 * 0.45 + orig[1] as f32 * 0.55).round() as u8;
            let b = (over[2] as f32 * 0.45 + orig[2] as f32 * 0.55).round() as u8;
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    // 边界线：相邻像素标签不同 → 画深色线
    let boundary_color = [20u8, 20u8, 20u8];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let cur = labels[idx];
            if cur.is_none() {
                continue;
            }
            let is_boundary = (x > 0 && labels[idx - 1] != cur)
                || (x + 1 < w && labels[idx + 1] != cur)
                || (y > 0 && labels[idx - w as usize] != cur)
                || (y + 1 < h && labels[idx + w as usize] != cur);
            if is_boundary {
                out.put_pixel(x, y, image::Rgb(boundary_color));
            }
        }
    }
    out
}

// =============================================================================
// 图片加载 — 解码、缩放
// =============================================================================

/// 计算保持比例、适配 max_dim 的新尺寸。
fn fit_dimensions(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (w, h);
    }
    let scale = max_dim as f64 / w.max(h) as f64;
    (
        (w as f64 * scale).max(1.0) as u32,
        (h as f64 * scale).max(1.0) as u32,
    )
}

/// 加载并预处理单张图片：Lanczos3 缩放至 max_dim，转换为 RGB8。
fn load_image(path: &std::path::Path, max_dim: u32) -> Result<RgbImage> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let img = ImageReader::open(path)
        .with_context(|| format!("opening {}", filename))?
        .decode()
        .with_context(|| format!("decoding {}", filename))?;

    let (w, h) = img.dimensions();
    let (new_w, new_h) = fit_dimensions(w, h, max_dim);
    let resized = if new_w != w || new_h != h {
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    Ok(resized.to_rgb8())
}

// =============================================================================
// HTML 生成 — 自包含单文件，暗色主题 + Canvas 可视化
// =============================================================================

/// 为单张图片生成完整 HTML。
fn generate_html(result: &SegmentResult, name: &str, orig_uri: &str) -> String {
    let w = result.width;
    let h = result.height;
    let n_regions = result.regions.len();

    // ===== 生成叠加图 =====
    let overlay_img = gen_colored_overlay(result);
    let boundary_img = gen_boundary_overlay(result);
    let edge_img = gen_edge_heatmap(result);

    let overlay_dyn = DynamicImage::ImageRgb8(overlay_img);
    let boundary_dyn = DynamicImage::ImageRgba8(boundary_img);
    let edge_dyn = DynamicImage::ImageLuma8(edge_img);

    let overlay_uri = img_to_png_uri(&overlay_dyn).unwrap_or_default();
    let boundary_uri = img_to_png_uri(&boundary_dyn).unwrap_or_default();
    let edge_uri = img_to_png_uri(&edge_dyn).unwrap_or_default();

    // ===== 区域调色盘 JSON =====
    let palette_json = build_palette_json(result);

    // ===== 组装 HTML =====
    let mut html = String::with_capacity(128_000);
    write_html_head(&mut html, name);
    write_html_body(
        &mut html,
        name,
        w,
        h,
        n_regions,
        orig_uri,
        &overlay_uri,
        &boundary_uri,
        &edge_uri,
        &palette_json,
        result,
    );
    html
}

/// 构建区域调色盘 JSON 数组，每项 { color, area_pct, area }。
fn build_palette_json(result: &SegmentResult) -> String {
    let total_area: usize = result.regions.iter().map(|r| r.area).sum();
    let palette = &result.palette.colors;
    let mut items: Vec<String> = Vec::new();

    for region in &result.regions {
        let cid = region.cluster_id;
        let rgb = if cid < palette.len() {
            palette[cid]
        } else {
            [128, 128, 128]
        };
        let pct = if total_area > 0 {
            region.area as f64 / total_area as f64 * 100.0
        } else {
            0.0
        };
        let hex = format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]);
        items.push(format!(
            r#"{{"color":"{}","area_pct":{:.1},"area":{}}}"#,
            hex, pct, region.area
        ));
    }
    format!("[{}]", items.join(","))
}

// =============================================================================
// HTML 模板片段
// =============================================================================

fn write_html_head(html: &mut String, name: &str) {
    let title = format!("Color Segment — {}", escape_html(name));
    html.push_str(&format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6c63ff;
    --border: #2a2a4a;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    line-height: 1.5;
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
}}
header {{
    text-align: center;
    padding: 32px 20px;
    border-bottom: 2px solid var(--border);
    margin-bottom: 32px;
}}
header h1 {{
    font-size: 1.6rem;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 6px;
}}
header .meta {{
    font-size: 0.8rem;
    color: var(--text-dim);
}}
section {{
    background: var(--card);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
    border: 1px solid var(--border);
}}
section h2 {{
    font-size: 1rem;
    color: var(--accent);
    margin-bottom: 14px;
    font-weight: 500;
}}
canvas {{
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid var(--border);
}}
.palette-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}}
.palette-item {{
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px 8px 8px;
    min-width: 160px;
}}
.swatch {{
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.12);
    flex-shrink: 0;
}}
.swatch-info {{
    font-size: 0.75rem;
    line-height: 1.4;
}}
.swatch-hex {{
    color: var(--text);
    font-weight: 500;
}}
.swatch-pct {{
    color: var(--text-dim);
}}
footer {{
    text-align: center;
    padding: 16px;
    color: var(--text-dim);
    font-size: 0.7rem;
    border-top: 1px solid var(--border);
    margin-top: 16px;
}}
.edge-legend {{
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.edge-legend .bar {{
    width: 120px;
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(to right, #000, #fff);
    border: 1px solid var(--border);
}}
</style>
</head>
"#,
        title = title,
    ));
}

fn write_html_body(
    html: &mut String,
    name: &str,
    w: u32,
    h: u32,
    n_regions: usize,
    orig_uri: &str,
    overlay_uri: &str,
    boundary_uri: &str,
    edge_uri: &str,
    palette_json: &str,
    result: &SegmentResult,
) {
    let escaped_name = escape_html(name);

    // ===== Header =====
    html.push_str("<body>\n");
    html.push_str("<header>\n");
    html.push_str(&format!("<h1>{escaped_name}</h1>\n"));
    html.push_str(&format!(
        "<div class=\"meta\">{w} × {h} — {n} regions</div>\n",
        w = w,
        h = h,
        n = n_regions
    ));
    html.push_str("</header>\n");

    // ===== Canvas 1: Region Overlay =====
    html.push_str("<section>\n");
    html.push_str("<h2>Region Overlay</h2>\n");
    html.push_str(&format!(
        "<canvas id=\"overlay-canvas\" width=\"{w}\" height=\"{h}\"></canvas>\n"
    ));
    html.push_str("</section>\n");

    // ===== Canvas 2: Edge Map =====
    html.push_str("<section>\n");
    html.push_str("<h2>Edge Map</h2>\n");
    html.push_str(&format!(
        "<canvas id=\"edge-canvas\" width=\"{w}\" height=\"{h}\"></canvas>\n"
    ));
    html.push_str(
        "<div class=\"edge-legend\"><span>weak</span><div class=\"bar\"></div><span>strong</span></div>\n",
    );
    html.push_str("</section>\n");

    // ===== Palette =====
    html.push_str("<section>\n");
    html.push_str("<h2>Color Palette</h2>\n");
    write_palette_html(html, result);
    html.push_str("</section>\n");

    // ===== Footer =====
    html.push_str("<footer>Generated by color-segment</footer>\n");

    // ===== Script =====
    html.push_str("<script>\n");
    html.push_str(&format!("const ORIG_URI = \"{orig_uri}\";\n"));
    html.push_str(&format!("const OVERLAY_URI = \"{overlay_uri}\";\n"));
    html.push_str(&format!("const BOUNDARY_URI = \"{boundary_uri}\";\n"));
    html.push_str(&format!("const EDGE_URI = \"{edge_uri}\";\n"));
    html.push_str(&format!("const W = {w};\n"));
    html.push_str(&format!("const H = {h};\n"));
    html.push_str(&format!("const PALETTE = {palette_json};\n"));
    html.push_str(
        r#"
function loadImage(src) {
    return new Promise(function(resolve) {
        var img = new Image();
        img.onload = function() { resolve(img); };
        img.src = src;
    });
}

async function drawOverlay() {
    var c = document.getElementById('overlay-canvas');
    var ctx = c.getContext('2d');

    var orig   = await loadImage(ORIG_URI);
    var overlay = await loadImage(OVERLAY_URI);
    var boundary = await loadImage(BOUNDARY_URI);

    // 原图
    ctx.drawImage(orig, 0, 0);

    // 半透明区域着色
    ctx.globalAlpha = 0.45;
    ctx.drawImage(overlay, 0, 0);

    // 边界线
    ctx.globalAlpha = 1.0;
    ctx.drawImage(boundary, 0, 0);
}

async function drawEdgeMap() {
    var c = document.getElementById('edge-canvas');
    var ctx = c.getContext('2d');
    var edge = await loadImage(EDGE_URI);
    ctx.drawImage(edge, 0, 0);
}

drawOverlay();
drawEdgeMap();
</script>
"#,
    );

    html.push_str("</body>\n</html>\n");
}

/// 生成调色盘 HTML 面板。
fn write_palette_html(html: &mut String, result: &SegmentResult) {
    let total_area: usize = result.regions.iter().map(|r| r.area).sum();
    let palette = &result.palette.colors;
    html.push_str("<div class=\"palette-grid\">\n");

    for region in &result.regions {
        let cid = region.cluster_id;
        let rgb = if cid < palette.len() {
            palette[cid]
        } else {
            [128, 128, 128]
        };
        let pct = if total_area > 0 {
            region.area as f64 / total_area as f64 * 100.0
        } else {
            0.0
        };
        let hex = format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]);

        html.push_str(&format!(
            r#"<div class="palette-item">
    <div class="swatch" style="background:{hex}"></div>
    <div class="swatch-info">
        <div class="swatch-hex">{hex}</div>
        <div class="swatch-pct">{pct:.1}% — {area} px</div>
    </div>
</div>
"#,
            hex = hex,
            pct = pct,
            area = region.area,
        ));
    }

    html.push_str("</div>\n");
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// =============================================================================
// 总览 HTML — 汇总所有分割结果
// =============================================================================

/// 单张图片的处理结果元数据，用于生成总览页面。
struct ImageResult {
    name_stem: String,
    fname: String,
    width: u32,
    height: u32,
    n_regions: usize,
    html_path: String,        // 详情页文件名
    seg_png_path: String,     // 切分图 PNG 文件名
    top_colors: Vec<[u8; 3]>, // 面积最大的几个颜色
}

/// 生成总览 HTML：每张图片一个缩略卡片，链接到详情页。
fn generate_overview_html(
    results: &[ImageResult],
    stem: &str,
    _out_dir: &std::path::Path,
) -> String {
    let mut h = String::with_capacity(32_000);
    h.push_str(&format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Color Segment — Overview ({stem})</title>
<style>
:root {{
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6c63ff;
    --border: #2a2a4a;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    padding: 24px;
}}
header {{
    text-align: center;
    padding: 24px;
    border-bottom: 2px solid var(--border);
    margin-bottom: 28px;
}}
header h1 {{ font-size: 1.4rem; color: var(--accent); margin-bottom: 6px; }}
header .meta {{ font-size: 0.75rem; color: var(--text-dim); }}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
}}
.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    transition: border-color 0.15s;
}}
.card:hover {{ border-color: var(--accent); }}
.card a {{ text-decoration: none; color: inherit; display: block; }}
.card-thumb {{
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
    background: #111;
}}
.card-info {{
    padding: 12px 14px;
}}
.card-name {{
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.card-meta {{
    font-size: 0.65rem;
    color: var(--text-dim);
    display: flex;
    gap: 12px;
    align-items: center;
}}
.card-swatches {{
    display: flex;
    gap: 3px;
    margin-top: 7px;
}}
.card-swatches span {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.1);
    flex-shrink: 0;
}}
footer {{
    text-align: center;
    padding: 18px;
    color: var(--text-dim);
    font-size: 0.65rem;
    border-top: 1px solid var(--border);
    margin-top: 24px;
}}
</style>
</head>
<body>
<header>
<h1>Color Segment Overview</h1>
<div class="meta">{n} images — stem: {stem}</div>
</header>
<div class="grid">
"#,
        stem = stem,
        n = results.len(),
    ));

    for r in results {
        let escaped = escape_html(&r.name_stem);
        let seg_png_rel = format!("{}_{}_seg.png", stem, r.name_stem);
        let html_rel = format!("{}_{}.html", stem, r.name_stem);

        // 前 5 个主色 swatches
        let mut swatches = String::new();
        for c in r.top_colors.iter().take(5) {
            swatches.push_str(&format!(
                r#"<span style="background:#{:02X}{:02X}{:02X}"></span>"#,
                c[0], c[1], c[2]
            ));
        }

        h.push_str(&format!(
            r#"<div class="card">
<a href="{html_rel}">
<img class="card-thumb" src="{seg_png_rel}" alt="{escaped}" loading="lazy">
<div class="card-info">
<div class="card-name">{escaped}</div>
<div class="card-meta">
<span>{w}×{h}</span>
<span>{n} regions</span>
</div>
<div class="card-swatches">{swatches}</div>
</div>
</a>
</div>
"#,
            html_rel = html_rel,
            seg_png_rel = seg_png_rel,
            escaped = escaped,
            w = r.width,
            h = r.height,
            n = r.n_regions,
            swatches = swatches,
        ));
    }

    h.push_str("</div>\n<footer>Generated by color-segment</footer>\n</body>\n</html>\n");
    h
}

// =============================================================================
// main() — CLI 入口
// =============================================================================

fn main() -> Result<()> {
    // ===== 加载参数 YAML =====
    let yaml_str = std::fs::read_to_string("color-segment/params.yaml")
        .with_context(|| "reading color-segment/params.yaml")?;
    let params: SegmentParams =
        serde_yaml::from_str(&yaml_str).with_context(|| "parsing params.yaml")?;

    // ===== CLI 参数解析 =====
    let args: Vec<String> = std::env::args().collect();
    let max_dim: u32 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(params.preprocess_max_dim)
        .max(1);

    let stem = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "segment".to_string());

    println!("color-segment — preprocess_max_dim={max_dim}, stem={stem}");
    println!("rayon threads: {}", rayon::current_num_threads());
    println!(
        "params: max_clusters={}, min_region={}, edge_thr={:.2}, edge_gamma={:.2}, color_merge_delta={:.1}",
        params.max_clusters,
        params.min_region_area,
        params.edge_threshold,
        params.edge_gamma,
        params.color_merge_distance
    );

    // ===== 扫描 imgs/ 目录 =====
    let imgs_dir = std::path::Path::new("imgs");
    if !imgs_dir.is_dir() {
        anyhow::bail!("'imgs/' directory not found — create it and add images");
    }

    let mut img_paths: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(imgs_dir).with_context(|| "reading imgs/ directory")? {
        let entry = entry?;
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext == "jpg" || ext == "jpeg" || ext == "png" {
            img_paths.push(path);
        }
    }

    if img_paths.is_empty() {
        anyhow::bail!("no jpg/jpeg/png images found in imgs/");
    }
    img_paths.sort();

    println!("Found {} image(s) in imgs/", img_paths.len());

    // ===== 确保 output/color-segment/ 存在 =====
    let out_dir = std::path::Path::new("output").join("color-segment");
    if !out_dir.exists() {
        std::fs::create_dir_all(&out_dir)
            .with_context(|| "creating output/color-segment/ directory")?;
    }

    // ===== 并行处理所有图片 =====
    let results: Vec<Result<ImageResult>> = img_paths
        .par_iter()
        .map(|img_path| {
            let fname = img_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            let name_stem = img_path
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            let started = Instant::now();
            println!("[{:?}] processing {}", std::thread::current().id(), fname);

            // 加载图片
            let rgb =
                load_image(img_path, max_dim).with_context(|| format!("loading {}", fname))?;

            // 保留一份原图用于切分图渲染（segment 会 move rgb）
            let orig_for_png = rgb.clone();

            // 原始图 JPEG base64 (在 segment 调用前编码)
            let orig_dyn = DynamicImage::ImageRgb8(rgb.clone());
            let orig_uri =
                img_to_jpeg_uri(&orig_dyn).with_context(|| "encoding original image to JPEG")?;

            // 执行分割 (调用库的 segment())
            let segment_started = Instant::now();
            let result = segment(&rgb, &params).with_context(|| format!("segmenting {}", fname))?;
            let segment_ms = segment_started.elapsed().as_secs_f64() * 1000.0;

            let n_regions = result.regions.len();
            let w = result.width;
            let h = result.height;

            // 渲染切分图 PNG
            let seg_png = render_segmented_png(
                &orig_for_png,
                &result.labels,
                &result.palette,
                &result.regions,
                w,
                h,
            );
            let seg_png_path = out_dir.join(format!("{}_{}_seg.png", stem, name_stem));
            seg_png
                .save(&seg_png_path)
                .with_context(|| format!("saving segmented PNG {}", seg_png_path.display()))?;

            // 生成 HTML
            let html = generate_html(&result, name_stem, &orig_uri);

            // 写入 HTML
            let html_path = out_dir.join(format!("{}_{}.html", stem, name_stem));
            std::fs::write(&html_path, html)
                .with_context(|| format!("writing {}", html_path.display()))?;

            // 收集面积最大的 5 个主色
            let mut indexed: Vec<(usize, &color_segment::Region)> =
                result.regions.iter().enumerate().collect();
            indexed.sort_by_key(|(_, r)| -(r.area as isize));
            let top_colors: Vec<[u8; 3]> = indexed
                .iter()
                .take(5)
                .map(|(_, r)| {
                    let cid = r.cluster_id;
                    if cid < result.palette.colors.len() {
                        result.palette.colors[cid]
                    } else {
                        [128, 128, 128]
                    }
                })
                .collect();

            println!(
                "[{:?}] finished {} → {} regions, segment={:.1}ms total={:.1}ms",
                std::thread::current().id(),
                fname,
                n_regions,
                segment_ms,
                started.elapsed().as_secs_f64() * 1000.0
            );

            Ok(ImageResult {
                name_stem: name_stem.to_string(),
                fname: fname.to_string(),
                width: w,
                height: h,
                n_regions,
                html_path: format!("{}_{}.html", stem, name_stem),
                seg_png_path: format!("{}_{}_seg.png", stem, name_stem),
                top_colors,
            })
        })
        .collect();

    // 统一输出结果
    let mut processed: Vec<ImageResult> = Vec::new();
    for r in results {
        match r {
            Ok(img_res) => {
                println!(
                    "{} → {} regions → saved {} + {}",
                    img_res.fname, img_res.n_regions, img_res.html_path, img_res.seg_png_path
                );
                processed.push(img_res);
            }
            Err(e) => eprintln!("ERROR: {e:#}"),
        }
    }

    // ===== 生成总览 HTML =====
    let overview_html = generate_overview_html(&processed, &stem, &out_dir);
    let overview_path = out_dir.join(format!("{}_overview.html", stem));
    std::fs::write(&overview_path, overview_html)
        .with_context(|| format!("writing overview {}", overview_path.display()))?;
    println!("Overview → {}", overview_path.display());

    println!("Done. {} image(s) processed.", processed.len());
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b64_encode_empty() {
        assert_eq!(b64_encode(b""), "");
    }

    #[test]
    fn test_b64_encode_basic() {
        // "Man" → "TWFu"
        assert_eq!(b64_encode(b"Man"), "TWFu");
        // "Ma" → "TWE="
        assert_eq!(b64_encode(b"Ma"), "TWE=");
        // "M" → "TQ=="
        assert_eq!(b64_encode(b"M"), "TQ==");
    }

    #[test]
    fn test_b64_encode_roundtrip() {
        let data = b"Hello, world! This is a test of the base64 encoder.";
        let encoded = b64_encode(data);
        // Decode and verify
        let decoded = decode_b64(&encoded);
        assert_eq!(decoded, data.to_vec());
    }

    /// Minimal base64 decoder for roundtrip testing.
    fn decode_b64(s: &str) -> Vec<u8> {
        let chars: Vec<u8> = s
            .bytes()
            .filter(|&b| b != b'=')
            .map(|b| B64_CHARS.iter().position(|&c| c == b).unwrap_or(0) as u8)
            .collect();

        let mut out = Vec::new();
        for chunk in chars.chunks(4) {
            if chunk.len() < 2 {
                break;
            }
            let b0 = chunk[0] as u32;
            let b1 = chunk[1] as u32;
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let b3 = if chunk.len() > 3 { chunk[3] as u32 } else { 0 };
            let triple = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;

            out.push(((triple >> 16) & 0xFF) as u8);
            if chunk.len() > 2 {
                out.push(((triple >> 8) & 0xFF) as u8);
            }
            if chunk.len() > 3 {
                out.push((triple & 0xFF) as u8);
            }
        }
        out
    }

    #[test]
    fn test_fit_dimensions_no_resize() {
        assert_eq!(fit_dimensions(100, 100, 512), (100, 100));
    }

    #[test]
    fn test_fit_dimensions_scale_down() {
        let (w, h) = fit_dimensions(1024, 512, 512);
        assert_eq!(w, 512);
        assert_eq!(h, 256);
    }

    #[test]
    fn test_fit_dimensions_min_1() {
        let (w, h) = fit_dimensions(2048, 1, 512);
        assert_eq!(w, 512);
        assert!(h >= 1);
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
    }
}
