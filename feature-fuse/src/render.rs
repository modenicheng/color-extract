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
fn draw_text_rgb(
    canvas: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    text: &str,
    x: i32,
    y: i32,
    color: Rgb<u8>,
) {
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

/// 创建包含原图 + N 张特征图 + 3 融合图 + 3 过滤图的 contact sheet。
///
/// 布局（cols=5 为例）:
///   Row 0: Orig, F1, F2, F3, F4
///   Row 1: F5, F6, F7, F8, F9
///   Row 2: —, FuseAdd, FuseMul, FuseHyb, —
///   Row 3: —, FiltAdd, FiltMul, FiltHyb, —
///
/// 融合图与过滤图在同一列上下对齐，方便对比。
/// 实际显示的列数由 layout.cols 决定，行数不足时自动补齐空白图。
pub fn make_contact_sheet_full(
    original: &[[f64; 3]],
    features: &[(&str, &[f64])],
    fused: &[(&str, &[f64])],
    extra_rgb: &[(&str, &[[f64; 3]])],
    impression_swatch: Option<[f64; 3]>,
    weighted_swatch: Option<[f64; 3]>,
    region_swatch: Option<[f64; 3]>,
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
    let empty_thumb = make_thumb_gray_f32(&[], w, h, cell_w, cell_h);

    let cols = layout.cols;

    // ── 构建缩略图列表 ──
    let mut thumbs: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    // 第 1~2 行: 原图 + 所有特征图
    thumbs.push(make_thumb_rgb(original, w, h, cell_w, cell_h));
    labels.push("Original".to_string());
    for &(name, feat) in features {
        thumbs.push(make_thumb_gray_f32(feat, w, h, cell_w, cell_h));
        labels.push(shorten_feature_name(name));
    }
    // 将原图+特征图补齐到整行
    pad_to_align(&mut thumbs, &mut labels, cols, &empty_thumb);

    // 融合行与过滤行: 每一行在 col=1~3 放置 3 张图，col=0 和 col=cols-1 置空
    // 这样融合与过滤在垂直方向同一列对齐
    if fused.len() >= 3 {
        // 行首空白
        thumbs.push(empty_thumb.clone());
        labels.push(String::new());
        // 3 张无过滤融合图
        for &(name, f) in fused.iter().take(3) {
            thumbs.push(make_thumb_gray_f32(f, w, h, cell_w, cell_h));
            labels.push(fuse_display_name(name));
        }
        // 附加 RGB 复合图（如 Orig×Hyb）放在 FuseHybrid 右侧
        for &(name, comp) in extra_rgb {
            thumbs.push(make_thumb_rgb(comp, w, h, cell_w, cell_h));
            labels.push(fuse_display_name(name));
        }
        // 行尾补齐到 cols 列
        pad_to_align(&mut thumbs, &mut labels, cols, &empty_thumb);
    }

    if fused.len() >= 6 {
        // 行首空白
        thumbs.push(empty_thumb.clone());
        labels.push(String::new());
        // 3 张过滤后融合图
        for &(name, f) in fused.iter().skip(3).take(3) {
            thumbs.push(make_thumb_gray_f32(f, w, h, cell_w, cell_h));
            labels.push(fuse_display_name(name));
        }
        // 印象色色块（Orig×Hyb 正下方）
        if let Some(color) = impression_swatch {
            thumbs.push(make_swatch_thumb(color, cell_w, cell_h));
            labels.push("Imp Color".to_string());
        }
        // 加权聚类预测色（Imp Color 右侧）
        if let Some(color) = weighted_swatch {
            thumbs.push(make_swatch_thumb(color, cell_w, cell_h));
            labels.push("Pred Color".to_string());
        }
        if let Some(color) = region_swatch {
            thumbs.push(make_swatch_thumb(color, cell_w, cell_h));
            labels.push("Region Color".to_string());
        }
        // 行尾补齐到 cols 列
        pad_to_align(&mut thumbs, &mut labels, cols, &empty_thumb);
    }

    // ── 计算实际所需行数 ──
    // 布局: 1~2 行 features + 0~2 行 fuse/filter
    let feat_rows = (1 + features.len() as u32 + cols - 1) / cols;
    let needed_rows = if fused.len() >= 6 {
        feat_rows + 2
    } else if fused.len() >= 3 {
        feat_rows + 1
    } else {
        feat_rows
    };
    let rows = layout.rows.max(needed_rows);

    // 补齐到 rows × cols
    while thumbs.len() < (rows * cols) as usize {
        thumbs.push(empty_thumb.clone());
        labels.push(String::new());
    }

    // ── 渲染 sheet ──
    let sheet_w = cols * (cell_w + layout.pad) + layout.pad;
    let sheet_h = rows * (step_h + layout.pad) + layout.pad;

    let mut sheet: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(sheet_w, sheet_h);
    for pixel in sheet.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    for (idx, (thumb, label)) in thumbs.iter().zip(labels.iter()).enumerate() {
        if idx >= (rows * cols) as usize {
            break;
        }
        let col = idx as u32 % cols;
        let row = idx as u32 / cols;
        let ox = layout.pad + col * (cell_w + layout.pad);
        let oy = layout.pad + row * (step_h + layout.pad);

        image::imageops::overlay(&mut sheet, thumb, ox as i64, oy as i64);

        if !label.is_empty() {
            let text_w = label.len() as i32 * 8;
            let label_x = ox as i32 + (cell_w as i32 - text_w) / 2;
            let label_y = oy as i32 + cell_h as i32 + 4;
            draw_text_rgb(
                &mut sheet,
                label,
                label_x.max(0),
                label_y,
                Rgb([40, 40, 40]),
            );
        }
    }

    // ── 时间戳（最后一行右下角）──
    let ts_last_row = rows.saturating_sub(1);
    let ts_oy = layout.pad + ts_last_row * (step_h + layout.pad);
    let ts_text_w = timestamp.len() as i32 * 8;
    let ts_label_x = sheet_w as i32 - ts_text_w - 4;
    let ts_label_y = ts_oy as i32 + cell_h as i32 + 4;
    draw_text_rgb(
        &mut sheet,
        timestamp,
        ts_label_x.max(0),
        ts_label_y,
        Rgb([120, 120, 120]),
    );

    sheet
        .save(path)
        .context(format!("save contact sheet {}", path.display()))?;
    Ok(())
}

/// 向 thumbs/labels 尾部插入空白图，使当前行恰好铺满 cols 列
fn pad_to_align(
    thumbs: &mut Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    labels: &mut Vec<String>,
    cols: u32,
    empty: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) {
    let rem = cols - (thumbs.len() as u32 % cols);
    if rem < cols {
        for _ in 0..rem {
            thumbs.push(empty.clone());
            labels.push(String::new());
        }
    }
}

/// 将特征图 name 转为 contact sheet 显示用的短标签
fn shorten_feature_name(name: &str) -> String {
    match name {
        "dct_complexity" => "DCT",
        "lab_gradient" => "LAB Grad",
        "spectral_residual" => "Spect Res",
        "global_light_residual" => "G-Light",
        "global_lab_a_residual" => "G-LabA",
        "global_lab_b_residual" => "G-LabB",
        "global_sat_residual" => "G-Sat",
        "local_light_residual" => "L-Light",
        "local_lab_a_residual" => "L-LabA",
        "local_lab_b_residual" => "L-LabB",
        "local_sat_residual" => "L-Sat",
        "background_mask_morph" => "BG Mask",
        "background_fg_confidence" => "BG Conf",
        "subject_prior" => "Subj Prior",
        "abs_light" => "Abs L*",
        "abs_lab_a" => "Abs a*",
        "abs_lab_b" => "Abs b*",
        "abs_sat" => "Abs Sat",
        "segment_foreground" => "Seg FG",
        "bg_candidate" => "BG Cand",
        "bg_barrier" => "BG Barr",
        "bg_mask_before_protect" => "BG Pre",
        "foreground_protect" => "FG Prot",
        "bg_mask_after_protect" => "BG Post",
        "fg_confidence" => "FG Conf",
        "segment_region_id" => "Seg ID",
        "segment_boundary" => "Seg Bound",
        "segment_bg_probability" => "Seg BG",
        "segment_saliency" => "Seg Sal",
        "segment_subject_confidence" => "Seg Subj",
        other => other,
    }
    .to_string()
}

/// 将 fused 图 name 转为 contact sheet 显示用的短标签
fn fuse_display_name(name: &str) -> String {
    match name {
        "fused_add" => "Fuse Add",
        "fused_softmul" => "Fuse Mult",
        "fused_hybrid" => "Fuse Hybrid",
        "fused_original_hybrid" => "Orig×Hyb",
        "fused_original_hybrid_nothreshold" => "Orig×Hyb(NT)",
        "fused_add_filtered" => "Filt Add",
        "fused_softmul_filtered" => "Filt Mult",
        "fused_hybrid_filtered" => "Filt Hybrid",
        "Imp Color" => "Imp Color",
        "Region Color" => "Region Color",
        other => other,
    }
    .to_string()
}

/// 将 f32 [0,1] 特征图转为 RGB 缩略图（灰度映射到 RGB）
fn make_thumb_gray_f32(
    data: &[f64],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
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
fn make_thumb_rgb(
    data: &[[f64; 3]],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let orig: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(src_w, src_h, |x, y| {
        let i = (y * src_w + x) as usize;
        let r = (data[i][0].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (data[i][1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (data[i][2].clamp(0.0, 1.0) * 255.0) as u8;
        Rgb([r, g, b])
    });
    image::imageops::resize(&orig, dst_w, dst_h, image::imageops::FilterType::Lanczos3)
}

/// 渲染纯色色块缩略图（用于显示印象色）
fn make_swatch_thumb(color: [f64; 3], w: u32, h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let r = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
    let g = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
    let b = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
    ImageBuffer::from_fn(w, h, |_, _| Rgb([r, g, b]))
}
