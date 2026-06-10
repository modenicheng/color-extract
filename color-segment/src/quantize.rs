// =============================================================================
// Color palette and pixel types
// =============================================================================

/// Color palette produced by quantization.
#[derive(Debug, Clone)]
pub struct Palette {
    /// sRGB colors in 8-bit [0, 255]
    pub colors: Vec<[u8; 3]>,
    /// Pixel count per color
    pub counts: Vec<usize>,
}

/// Single pixel with original index, CIELAB coordinates, and normalized sRGB.
#[derive(Debug, Clone)]
pub struct Pixel {
    /// Position in the original pixel array
    pub index: usize,
    /// CIELAB L*a*b* coordinates
    pub lab: [f64; 3],
    /// Normalized sRGB [0, 1]
    pub rgb: [f64; 3],
}
