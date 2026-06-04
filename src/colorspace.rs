use palette::{
    cam16::{Cam16, Cam16Jmh, Cam16UcsJab, Parameters},
    hues::Cam16Hue,
    FromColor, Hsl, IntoColor, Lab, Oklab, Srgb, Xyz,
};
use palette::white_point::D65;

/// All supported color spaces for the extraction algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    RGB,
    CIELAB,
    Oklab,
    HSL,
    CAM16,
}

impl ColorSpace {
    pub fn all() -> [Self; 5] {
        [Self::RGB, Self::CIELAB, Self::Oklab, Self::HSL, Self::CAM16]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::RGB => "RGB",
            Self::CIELAB => "CIELAB",
            Self::Oklab => "Oklab",
            Self::HSL => "HSL",
            Self::CAM16 => "CAM16",
        }
    }

    /// Convert a single normalized RGB pixel [r, g, b] (0..1) to target space coords.
    pub fn convert_to(&self, rgb: [f64; 3]) -> [f64; 3] {
        match self {
            Self::RGB => rgb,
            Self::CIELAB => {
                let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
                let lab: Lab = srgb.into_color();
                [lab.l as f64, lab.a as f64, lab.b as f64]
            }
            Self::Oklab => {
                let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
                let oklab: Oklab = srgb.into_color();
                [oklab.l as f64, oklab.a as f64, oklab.b as f64]
            }
            Self::HSL => {
                let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
                let hsl: Hsl = srgb.into_color();
                [hsl.hue.into_positive_degrees() as f64, hsl.saturation as f64, hsl.lightness as f64]
            }
            Self::CAM16 => {
                let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
                let xyz: Xyz<D65, f32> = srgb.into_color();
                let cam16: Cam16<f32> = Cam16::from_xyz(xyz, cam16_params());
                let ucs: Cam16UcsJab<f32> = cam16.into_color();
                [ucs.lightness as f64, ucs.a as f64, ucs.b as f64]
            }
        }
    }

    /// Convert back from target space coords to normalized RGB [r, g, b] (0..1).
    /// Clamps out-of-gamut values to [0, 1].
    pub fn convert_from(&self, coords: [f64; 3]) -> [f64; 3] {
        let rgb = match self {
            Self::RGB => coords,
            Self::CIELAB => {
                let lab = Lab::new(coords[0] as f32, coords[1] as f32, coords[2] as f32);
                let srgb: Srgb = Srgb::from_color(lab);
                [srgb.red as f64, srgb.green as f64, srgb.blue as f64]
            }
            Self::Oklab => {
                let oklab = Oklab::new(coords[0] as f32, coords[1] as f32, coords[2] as f32);
                let srgb: Srgb = Srgb::from_color(oklab);
                [srgb.red as f64, srgb.green as f64, srgb.blue as f64]
            }
            Self::HSL => {
                let hsl = Hsl::new(
                    palette::RgbHue::from_degrees(coords[0] as f32),
                    coords[1] as f32,
                    coords[2] as f32,
                );
                let srgb: Srgb = Srgb::from_color(hsl);
                [srgb.red as f64, srgb.green as f64, srgb.blue as f64]
            }
            Self::CAM16 => {
                cam16ucs_to_rgb(coords[0] as f32, coords[1] as f32, coords[2] as f32)
            }
        };
        [clamp01(rgb[0]), clamp01(rgb[1]), clamp01(rgb[2])]
    }

    /// Batch convert Vec<[f64; 3]> (normalized RGB) to target space coords.
    pub fn convert_batch_to(&self, pixels: &[[f64; 3]]) -> Vec<[f64; 3]> {
        pixels.iter().map(|p| self.convert_to(*p)).collect()
    }

    /// Batch convert from target space back to normalized RGB.
    pub fn convert_batch_from(&self, coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
        coords.iter().map(|c| self.convert_from(*c)).collect()
    }
}

/// CIELAB L* (perceived lightness) for a normalized RGB color.
/// Used for dark-to-light sorting of palette swatches.
pub fn perceptual_lightness(rgb: [f64; 3]) -> f64 {
    let srgb = Srgb::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
    let lab: Lab = srgb.into_color();
    lab.l as f64
}

/// Convert CAM16-UCS J'a'b' coords back to normalized RGB [r, g, b] (0..1).
/// Uses the reverse CAM16-UCS formulas (Li et al. 2017) to recover J, M, h,
/// then constructs Cam16Jmh → Cam16 → XYZ → Srgb.
fn cam16ucs_to_rgb(j_prime: f32, a_prime: f32, b_prime: f32) -> [f64; 3] {
    let c1 = 0.007f32;
    let c2 = 0.0228f32;

    // Reverse CAM16-UCS: J' → J
    let j = j_prime / (1.0 + 100.0 * c1 - c1 * j_prime);

    // Reverse CAM16-UCS: a', b' → M', h → M
    let m_prime = (a_prime * a_prime + b_prime * b_prime).sqrt();
    let m = if m_prime > 0.0 {
        ((c2 * m_prime).exp() - 1.0) / c2
    } else {
        0.0
    };

    let h_rad = f32::atan2(b_prime, a_prime);
    let h_deg = h_rad.to_degrees();
    let h_deg = if h_deg < 0.0 { h_deg + 360.0 } else { h_deg };

    // Build Cam16Jmh and convert back through the palette pipeline
    let jmh = Cam16Jmh::new(j, m, Cam16Hue::from_degrees(h_deg));
    let params = cam16_params();
    let cam16: Cam16<f32> = jmh.into_full(params);
    let xyz: Xyz<D65, f32> = cam16.into_xyz(params);
    let srgb: Srgb = Srgb::from_color(xyz);

    [srgb.red as f64, srgb.green as f64, srgb.blue as f64]
}

/// Create CAM16 parameters for sRGB viewing conditions (40 cd/m²).
fn cam16_params() -> Parameters<palette::cam16::StaticWp<D65>, f32> {
    use std::sync::OnceLock;
    static PARAMS: OnceLock<Parameters<palette::cam16::StaticWp<D65>, f32>> = OnceLock::new();
    *PARAMS.get_or_init(|| {
        let mut p = Parameters::default_static_wp(40.0);
        p.background_luminance = 0.2;
        p
    })
}

fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}
