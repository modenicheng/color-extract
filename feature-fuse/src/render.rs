// =============================================================================
// 保存 PNG + 位图字体标签绘制 + Contact Sheet 拼贴图
// =============================================================================

use anyhow::{Context, Result};
use image::{GrayImage, ImageBuffer, Luma, Rgb};
use std::path::Path;

use crate::params::ContactSheetParams;

/// 保存单通道特征图为 PNG
pub fn save_gray_png(data: &[f64], w: u32, h: u32, path: &Path) -> Result<()> {
    let img: GrayImage = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let v = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });
    img.save(path).context(format!("save {}", path.display()))
}

/// 保存 RGB PNG
pub fn save_rgb_png(rgb: &[[f64; 3]], w: u32, h: u32, path: &Path) -> Result<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |x, y| {
        let i = (y * w + x) as usize;
        let r = (rgb[i][0].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (rgb[i][1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (rgb[i][2].clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([r, g, b])
    });
    img.save(path).context(format!("save {}", path.display()))
}

// ── 8×8 位图字体标签绘制 ──

/// 在 RGB 图像上绘制文本（使用 font8x8 位图字体，8×8 等宽）
fn draw_text_rgb(canvas: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, text: &str, x: i32, y: i32, color: Rgb<u8>) {
    let (cw, ch) = canvas.dimensions();
    for (ci, &ch_byte) in text.as_bytes().iter().enumerate() {
        let idx = ch_byte as usize;
        if idx >= font8x8::legacy::BASIC_LEGACY.len() {
            continue;
        }
        let glyph = &font8x8::legacy::BASIC_LEGACY[idx];
        for row in 0..8 {
            let row_data = glyph[row];
            for col in 0..8 {
                // MSB = leftmost pixel in font8x8
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

/// 创建包含原图、7 张特征图、6 张融合图（3 过滤前 + 3 过滤后）的 contact sheet
/// 每张子图下方带说明标签，融合图区域右下角印时间戳
pub fn make_contact_sheet_full(
    original: &[[f64; 3]],
    features: &[(&str, &[f64])],
    fused: &[(&str, &[f64])],
    w: u32,
    h: u32,
    layout: &ContactSheetParams,
    timestamp: &str,
    path: &Path,
) -> Result<()> {
    let cell_w = layout.thumb_w.min(w);
    let cell_h = (cell_w as f64 * h as f64 / w as f64).round() as u32;

    // sheet 高度每行多出 label_h 用于标签
    let step_h = cell_h + layout.label_h;
    let sheet_w = layout.cols * (cell_w + layout.pad) + layout.pad;
    let sheet_h = layout.rows * (step_h + layout.pad) + layout.pad;

    // 创建空白缩略图（用于补齐对齐）
    let empty_thumb = make_thumb_gray_f32(&[], w, h, cell_w, cell_h);

    // ── 布局规划 ──
    // 第 1 行: 1 原图 + 7 特征 = 8 格（按 cols 填充尾部空图）
    // 第 2 行起: 3 无过滤 + 3 过滤（约束: 3 + n = cols，n >= 0 为补齐空图数）
    let header_count: u32 = 1 + features.len() as u32; // 1 原图 + 7 特征
    let fused_per_row: u32 = 3;                         // 每组（无过滤/过滤）固定 3 张
    let pad_in_fuse: u32 = layout.cols.saturating_sub(fused_per_row);

    // 收集所有缩略图 (RGB) 及其标签
    let mut thumbs: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    // 头部: 原图 + 7 特征
    thumbs.push(make_thumb_rgb(original, w, h, cell_w, cell_h));
    labels.push("Original".to_string());
    for &(_name, feat) in features {
        thumbs.push(make_thumb_gray_f32(feat, w, h, cell_w, cell_h));
    }
    for name in ["DCT", "LAB Grad", "Spect Res", "G-Light", "G-Sat", "L-Light", "L-Sat"].iter() {
        labels.push((*name).to_string());
    }

    // 头部行尾补齐到下一行
    let header_pad = if header_count <= layout.cols {
        layout.cols - header_count
    } else {
        0
    };
    for _ in 0..header_pad {
        thumbs.push(empty_thumb.clone());
        labels.push(String::new());
    }

    // 无过滤的 3 张融合图 + pad_in_fuse 空白图
    for &(_name, f) in fused.iter().take(3) {
        thumbs.push(make_thumb_gray_f32(f, w, h, cell_w, cell_h));
    }
    for name in ["Fuse Add", "Fuse Mult", "Fuse Hybrid"].iter() {
        labels.push((*name).to_string());
    }
    for _ in 0..pad_in_fuse {
        thumbs.push(empty_thumb.clone());
        labels.push(String::new());
    }

    // 若有过滤（fused 至少 6 项），再加一行
    let has_filter = fused.len() >= 6;
    if has_filter {
        for &(_name, f) in fused.iter().skip(3).take(3) {
            thumbs.push(make_thumb_gray_f32(f, w, h, cell_w, cell_h));
        }
        for name in ["Filt Add", "Filt Mult", "Filt Hybrid"].iter() {
            labels.push((*name).to_string());
        }
        for _ in 0..pad_in_fuse {
            thumbs.push(empty_thumb.clone());
            labels.push(String::new());
        }
    }

    let mut sheet: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(sheet_w, sheet_h);
    // 白色背景
    for pixel in sheet.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    for (idx, (thumb, label)) in thumbs.iter().zip(labels.iter()).enumerate() {
        if idx >= (layout.cols * layout.rows) as usize {
            break;
        }
        let col = idx as u32 % layout.cols;
        let row = idx as u32 / layout.cols;
        let ox = layout.pad + col * (cell_w + layout.pad);
        let oy = layout.pad + row * (step_h + layout.pad);

        // overlay thumbnail
        image::imageops::overlay(&mut sheet, thumb, ox as i64, oy as i64);

        // draw label centered below the thumbnail（空标签不绘制）
        if !label.is_empty() {
            let text_w = label.len() as i32 * 8;
            let label_x = ox as i32 + (cell_w as i32 - text_w) / 2;
            let label_y = oy as i32 + cell_h as i32 + 4;
            draw_text_rgb(&mut sheet, label, label_x.max(0), label_y, Rgb([40, 40, 40]));
        }
    }

    // ── 印时间戳（汇总图右下角，与最后一行标签同高度，右对齐）──
    let ts_last_row = layout.rows.saturating_sub(1);
    let ts_oy = layout.pad + ts_last_row * (step_h + layout.pad);
    let ts_text_w = timestamp.len() as i32 * 8;
    let ts_label_x = sheet_w as i32 - ts_text_w - 4; // 右对齐，留 4px 右边距
    let ts_label_y = ts_oy as i32 + cell_h as i32 + 4; // 与最后一行 label 同高度
    draw_text_rgb(&mut sheet, timestamp, ts_label_x.max(0), ts_label_y, Rgb([120, 120, 120]));

    sheet.save(path).context(format!("save contact sheet {}", path.display()))?;
    Ok(())
}

/// 将 f32 [0,1] 特征图转为 RGB 缩略图（灰度映射到 RGB）
fn make_thumb_gray_f32(data: &[f64], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    if data.is_empty() {
        // 空白占位（用于对齐补齐的格子）
        return ImageBuffer::from_fn(dst_w, dst_h, |_, _| Rgb([255, 255, 255]));
    }
    let gray: GrayImage = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let i = (y * src_w + x) as usize;
        let v = (data[i].clamp(0.0, 1.0) * 255.0) as u8;
        Luma([v])
    });
    let thumb = image::imageops::resize(&gray, dst_w, dst_h, image::imageops::FilterType::Lanczos3);
    // 转为 RGB（三通道相同）
    ImageBuffer::from_fn(dst_w, dst_h, |x, y| {
        let Luma([g]) = thumb.get_pixel(x, y);
        Rgb([*g, *g, *g])
    })
}

/// 将 RGB [0,1] 原图转为 RGB 缩略图
fn make_thumb_rgb(data: &[[f64; 3]], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let orig: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let i = (y * src_w + x) as usize;
        let r = (data[i][0].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (data[i][1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (data[i][2].clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([r, g, b])
    });
    image::imageops::resize(&orig, dst_w, dst_h, image::imageops::FilterType::Lanczos3)
}
