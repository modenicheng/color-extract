// =============================================================================
// 保存 PNG + 位图字体标签 + Contact Sheet 拼贴图 + HTML 报告
// =============================================================================

use anyhow::{Context, Result};
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use std::path::Path;

use crate::palette::PaletteEntry;

/// 保存单通道特征图为 PNG
pub fn save_gray_png(data: &[f64], w: u32, h: u32, path: &Path) -> Result<()> {
    let img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let v = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });
    img.save(path).context(format!("save {}", path.display()))
}

/// 保存单通道特征图，并用红点标出按亮度加权的质心。
pub fn save_gray_png_with_centroid(data: &[f64], w: u32, h: u32, path: &Path) -> Result<()> {
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let v = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([v, v, v])
    });
    if let Some((cx, cy)) = weighted_centroid(data, w, h) {
        let radius = ((w.min(h) as f64 / 140.0).round() as i32).clamp(2, 5);
        draw_centroid_marker(&mut img, cx, cy, radius);
    }
    img.save(path).context(format!("save {}", path.display()))
}

/// 保存 RGB PNG
pub fn save_rgb_png(rgb: &[[f64; 3]], w: u32, h: u32, path: &Path) -> Result<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        Rgb([
            (rgb[i][0].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[i][1].clamp(0.0, 1.0) * 255.0) as u8,
            (rgb[i][2].clamp(0.0, 1.0) * 255.0) as u8,
        ])
    });
    img.save(path).context(format!("save {}", path.display()))
}

// ── 8×8 位图字体 ──

fn draw_text_rgb(canvas: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, text: &str, x: i32, y: i32, color: Rgb<u8>) {
    let (cw, ch) = canvas.dimensions();
    for (ci, &ch_byte) in text.as_bytes().iter().enumerate() {
        let idx = ch_byte as usize;
        if idx >= font8x8::legacy::BASIC_LEGACY.len() { continue; }
        let glyph = &font8x8::legacy::BASIC_LEGACY[idx];
        for row in 0..8 {
            let row_data = glyph[row];
            for col in 0..8 {
                if (row_data >> col) & 1 != 0 {
                    let px = x + (ci as i32) * 8 + col as i32;
                    let py = y + row as i32;
                    if px >= 0 && py >= 0 && px < cw as i32 && py < ch as i32 {
                        canvas.put_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }
}

// ── Contact Sheet ──

pub fn make_contact_sheet(
    original: &[[f64; 3]],
    features: &[(&str, &[f64])],
    fused: &[(&str, &[f64])],
    w: u32, h: u32,
    cols: u32, thumb_w: u32,
    palette: &[PaletteEntry],
    timestamp: &str,
    path: &Path,
) -> Result<()> {
    let cell_w = thumb_w.min(w);
    let cell_h = (cell_w as f64 * h as f64 / w as f64).round() as u32;
    let label_h = 16u32;
    let pad = 4u32;
    let step_h = cell_h + label_h + pad;

    // 缩略图列表
    let mut thumbs: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    // 原图
    thumbs.push(make_thumb_rgb(original, w, h, cell_w, cell_h));
    labels.push("Original".to_string());

    // 特征图
    for &(name, feat) in features {
        let thumb = if name == "dct" || name == "spectral" {
            make_thumb_gray_f32_with_centroid(feat, w, h, cell_w, cell_h)
        } else {
            make_thumb_gray_f32(feat, w, h, cell_w, cell_h)
        };
        thumbs.push(thumb);
        labels.push(shorten_name(name));
    }

    // 融合图
    for &(name, f) in fused {
        thumbs.push(make_thumb_gray_f32(f, w, h, cell_w, cell_h));
        labels.push(shorten_name(name));
    }

    // 调色板行（如有）
    if !palette.is_empty() {
        thumbs.push(make_palette_thumb(palette, cell_w, cell_h));
        labels.push("Palette".to_string());
    }

    // 补齐到 cols 的整数倍
    while thumbs.len() % cols as usize != 0 {
        thumbs.push(make_empty_thumb(cell_w, cell_h));
        labels.push(String::new());
    }

    let rows = (thumbs.len() as u32 + cols - 1) / cols;
    let sheet_w = cols * (cell_w + pad) + pad;
    let sheet_h = rows * (step_h + pad) + pad;

    let mut sheet: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(sheet_w, sheet_h);
    for pixel in sheet.pixels_mut() { *pixel = Rgb([255, 255, 255]); }

    for (idx, (thumb, label)) in thumbs.iter().zip(labels.iter()).enumerate() {
        let col = idx as u32 % cols;
        let row = idx as u32 / cols;
        let ox = pad + col * (cell_w + pad);
        let oy = pad + row * (step_h + pad);
        image::imageops::overlay(&mut sheet, thumb, ox as i64, oy as i64);
        if !label.is_empty() {
            let text_w = label.len() as i32 * 8;
            let lx = ox as i32 + (cell_w as i32 - text_w) / 2;
            let ly = oy as i32 + cell_h as i32 + 2;
            draw_text_rgb(&mut sheet, label, lx.max(0), ly, Rgb([40, 40, 40]));
        }
    }

    // 时间戳
    let ts_oy = pad + (rows - 1) * (step_h + pad) + cell_h as u32;
    let ts_text_w = timestamp.len() as i32 * 8;
    draw_text_rgb(&mut sheet, timestamp, (sheet_w as i32 - ts_text_w - 4).max(0), ts_oy as i32 + 2, Rgb([120, 120, 120]));

    sheet.save(path).context(format!("save contact sheet {}", path.display()))?;
    Ok(())
}

// ── HTML 报告 ──

pub fn generate_html_report(
    stem: &str,
    palette: &[PaletteEntry],
    params_yaml: &str,
    path: &Path,
) -> Result<()> {
    let mut html = String::from(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    );
    html.push_str(&format!(
        "<title>Impression Color Extract — {}</title>",
        stem
    ));
    html.push_str(
        "<style>
        body { font-family: sans-serif; max-width: 960px; margin: 2em auto; }
        h1 { color: #333; }
        .palette { display: flex; gap: 4px; margin: 1em 0; flex-wrap: wrap; }
        .swatch { width: 80px; height: 80px; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; padding: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.15); }
        .swatch .hex { background: rgba(255,255,255,0.85); padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-bottom: 4px; font-family: monospace; }
        .swatch .pct { background: rgba(255,255,255,0.85); padding: 2px 6px; border-radius: 4px; font-size: 11px; font-family: monospace; }
        .features { display: flex; flex-wrap: wrap; gap: 8px; margin: 1em 0; }
        .features img { width: 200px; border-radius: 4px; border: 1px solid #ddd; }
        pre { background: #f5f5f5; padding: 1em; border-radius: 6px; overflow-x: auto; font-size: 13px; }
    </style></head><body>");

    html.push_str(&format!("<h1>Impression Color Extract: {}</h1>", stem));

    // 调色板
    html.push_str("<h2>Palette</h2><div class='palette'>");
    for entry in palette {
        html.push_str(&format!(
            "<div class='swatch' style='background:{}'><span class='hex'>{}</span><span class='pct'>{:.1}%</span></div>",
            entry.hex, entry.hex, entry.proportion * 100.0
        ));
    }
    html.push_str("</div>");

    // 配置文件
    html.push_str("<h2>Config</h2><pre>");
    html.push_str(&html_escape(params_yaml));
    html.push_str("</pre></body></html>");

    std::fs::write(path, html).context(format!("write {}", path.display()))?;
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// ── 缩略图辅助 ──

fn shorten_name(name: &str) -> String {
    match name {
        "dct" => "DCT",
        "lab_grad" => "LAB Grad",
        "spectral" => "Spect Res",
        "local_light" => "L-Light",
        "local_sat" => "L-Sat",
        "bg_mask" => "BG Mask",
        "fg_confidence" => "FG Conf",
        "fused" => "Fused",
        other => other,
    }.to_string()
}

fn make_thumb_gray_f32(data: &[f64], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if data.is_empty() { return make_empty_thumb(dst_w, dst_h); }
    let gray: GrayImage = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let v = (data[(y * src_w + x) as usize].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });
    let thumb = image::imageops::resize(&gray, dst_w, dst_h, image::imageops::FilterType::Lanczos3);
    ImageBuffer::from_fn(dst_w, dst_h, |x, y| {
        let Luma([g]) = thumb.get_pixel(x, y);
        Rgb([*g, *g, *g])
    })
}

fn make_thumb_gray_f32_with_centroid(data: &[f64], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut thumb = make_thumb_gray_f32(data, src_w, src_h, dst_w, dst_h);
    if let Some((cx, cy)) = weighted_centroid(data, src_w, src_h) {
        let scale_x = dst_w as f64 / src_w as f64;
        let scale_y = dst_h as f64 / src_h as f64;
        let radius = ((dst_w.min(dst_h) as f64 / 44.0).round() as i32).clamp(2, 4);
        draw_centroid_marker(&mut thumb, cx * scale_x, cy * scale_y, radius);
    }
    thumb
}

fn make_thumb_rgb(data: &[[f64; 3]], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if data.is_empty() { return make_empty_thumb(dst_w, dst_h); }
    let orig: RgbImage = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let i = (y * src_w + x) as usize;
        Rgb([
            (data[i][0].clamp(0.0, 1.0) * 255.0) as u8,
            (data[i][1].clamp(0.0, 1.0) * 255.0) as u8,
            (data[i][2].clamp(0.0, 1.0) * 255.0) as u8,
        ])
    });
    image::imageops::resize(&orig, dst_w, dst_h, image::imageops::FilterType::Lanczos3)
}

fn make_empty_thumb(w: u32, h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    ImageBuffer::from_fn(w, h, |_, _| Rgb([255, 255, 255]))
}

fn make_palette_thumb(palette: &[PaletteEntry], w: u32, h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if palette.is_empty() { return make_empty_thumb(w, h); }
    let n = palette.len() as u32;
    let swatch_w = w / n;
    ImageBuffer::from_fn(w, h, |x, _y| {
        let idx = (x / swatch_w).min(n - 1) as usize;
        let hex = &palette[idx].hex;
        let r = u8::from_str_radix(&hex[1..3], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[3..5], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[5..7], 16).unwrap_or(0);
        Rgb([r, g, b])
    })
}

fn weighted_centroid(data: &[f64], w: u32, h: u32) -> Option<(f64, f64)> {
    if data.len() != (w * h) as usize {
        return None;
    }
    let mut sum_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in 0..h {
        for x in 0..w {
            let v = data[(y * w + x) as usize].clamp(0.0, 1.0);
            sum_w += v;
            sum_x += v * x as f64;
            sum_y += v * y as f64;
        }
    }
    if sum_w <= 1e-12 {
        return None;
    }
    Some((sum_x / sum_w, sum_y / sum_w))
}

fn draw_centroid_marker(canvas: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, cx: f64, cy: f64, radius: i32) {
    let (w, h) = canvas.dimensions();
    let cx = cx.round() as i32;
    let cy = cy.round() as i32;
    let outer = radius + 1;
    for dy in -outer..=outer {
        for dx in -outer..=outer {
            let dist2 = dx * dx + dy * dy;
            if dist2 > outer * outer {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let color = if dist2 <= radius * radius {
                Rgb([255, 0, 0])
            } else {
                Rgb([255, 255, 255])
            };
            canvas.put_pixel(x as u32, y as u32, color);
        }
    }
}
