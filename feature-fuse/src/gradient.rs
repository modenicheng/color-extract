// =============================================================================
// LAB Sobel 梯度 (复用 lab-gradient 算法)
// =============================================================================

/// Sobel 梯度幅值：对单通道二维网格计算 |∇f|
fn sobel_magnitude(ch: &[f64], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut mag = vec![0.0; n];

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let i = y * w + x;
            let gx = -1.0 * ch[i - w - 1] + 1.0 * ch[i - w + 1] - 2.0 * ch[i - 1] + 2.0 * ch[i + 1]
                - 1.0 * ch[i + w - 1]
                + 1.0 * ch[i + w + 1];
            let gy = -1.0 * ch[i - w - 1] - 2.0 * ch[i - w] - 1.0 * ch[i - w + 1]
                + 1.0 * ch[i + w - 1]
                + 2.0 * ch[i + w]
                + 1.0 * ch[i + w + 1];
            // 除以 8 归一化到近似每像素 delta
            mag[i] = ((gx * gx + gy * gy).sqrt()) / 8.0;
        }
    }
    mag
}

/// 计算 LAB 梯度融合图：sqrt(gL² + ga² + gb²)
pub fn compute_lab_gradient(
    lab_l: &[f64],
    lab_a: &[f64],
    lab_b: &[f64],
    w: usize,
    h: usize,
) -> Vec<f64> {
    let mag_l = sobel_magnitude(lab_l, w, h);
    let mag_a = sobel_magnitude(lab_a, w, h);
    let mag_b = sobel_magnitude(lab_b, w, h);

    let n = w * h;
    let mut fused = Vec::with_capacity(n);
    for i in 0..n {
        fused.push((mag_l[i] * mag_l[i] + mag_a[i] * mag_a[i] + mag_b[i] * mag_b[i]).sqrt());
    }
    fused
}
