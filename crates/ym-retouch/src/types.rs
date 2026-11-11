//! The types of `docs/27` section 3.

use crate::error::{Result, RetouchError};

/// Linear AWG4, f32, row-major, no alpha. Nominally 0..1; values above 1 are allowed.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAwg4Image {
    /// Width in pixels.
    pub width_px: u32,
    /// Height in pixels.
    pub height_px: u32,
    /// `width_px * height_px` pixels, rows top to bottom.
    pub data: Vec<[f32; 3]>,
}

impl LinearAwg4Image {
    /// Makes an image, checking that the size matches the data and no side is 0.
    pub fn new(width_px: u32, height_px: u32, data: Vec<[f32; 3]>) -> Result<Self> {
        let image = Self {
            width_px,
            height_px,
            data,
        };
        image.validate()?;
        Ok(image)
    }

    /// An image of one colour.
    pub fn filled(width_px: u32, height_px: u32, pixel: [f32; 3]) -> Result<Self> {
        let len = (width_px as usize).saturating_mul(height_px as usize);
        Self::new(width_px, height_px, vec![pixel; len.min(1 << 31)])
    }

    /// Checks that both sides are at least 1 and the data length is `width_px * height_px`.
    pub fn validate(&self) -> Result<()> {
        check_len("image", self.width_px, self.height_px, self.data.len())
    }
}

/// 8-bit coverage. 0 = outside, 255 = fully inside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    /// Width in pixels.
    pub width_px: u32,
    /// Height in pixels.
    pub height_px: u32,
    /// `width_px * height_px` bytes, rows top to bottom.
    pub data: Vec<u8>,
}

impl Mask {
    /// Makes a mask, checking that the size matches the data and no side is 0.
    pub fn new(width_px: u32, height_px: u32, data: Vec<u8>) -> Result<Self> {
        let mask = Self {
            width_px,
            height_px,
            data,
        };
        mask.validate()?;
        Ok(mask)
    }

    /// A mask of zeros.
    pub fn zeros(width_px: u32, height_px: u32) -> Result<Self> {
        let len = (width_px as usize).saturating_mul(height_px as usize);
        Self::new(width_px, height_px, vec![0; len.min(1 << 31)])
    }

    /// Checks that both sides are at least 1 and the data length is `width_px * height_px`.
    pub fn validate(&self) -> Result<()> {
        check_len("mask", self.width_px, self.height_px, self.data.len())
    }

    /// The size as `(width_px, height_px)`.
    pub fn size_px(&self) -> (u32, u32) {
        (self.width_px, self.height_px)
    }

    /// The byte at `(x_px, y_px)`, or 0 outside the mask.
    pub fn get(&self, x_px: i64, y_px: i64) -> u8 {
        if x_px < 0
            || y_px < 0
            || x_px >= i64::from(self.width_px)
            || y_px >= i64::from(self.height_px)
        {
            return 0;
        }
        self.data
            .get(y_px as usize * self.width_px as usize + x_px as usize)
            .copied()
            .unwrap_or(0)
    }

    /// True when every pixel is 0.
    pub fn is_all_zero(&self) -> bool {
        self.data.iter().all(|&v| v == 0)
    }
}

fn check_len(what: &'static str, width_px: u32, height_px: u32, len: usize) -> Result<()> {
    let expected = (width_px as usize).checked_mul(height_px as usize);
    if width_px == 0 || height_px == 0 || expected != Some(len) {
        return Err(RetouchError::Size {
            what,
            width: width_px,
            height: height_px,
            len,
        });
    }
    Ok(())
}

/// Axis-aligned rectangle in full-resolution image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge in pixels; may be negative.
    pub x_px: i32,
    /// Top edge in pixels; may be negative.
    pub y_px: i32,
    /// Width in pixels.
    pub width_px: u32,
    /// Height in pixels.
    pub height_px: u32,
}

impl Rect {
    /// Makes a rectangle.
    pub fn new(x_px: i32, y_px: i32, width_px: u32, height_px: u32) -> Self {
        Self {
            x_px,
            y_px,
            width_px,
            height_px,
        }
    }

    /// The first column to the right of the rectangle.
    pub fn right_px(&self) -> i64 {
        i64::from(self.x_px) + i64::from(self.width_px)
    }

    /// The first row below the rectangle.
    pub fn bottom_px(&self) -> i64 {
        i64::from(self.y_px) + i64::from(self.height_px)
    }
}

/// A mask over `region` only; outside it the mask is 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedMask {
    /// Where the mask sits, in full-resolution image pixels.
    pub region: Rect,
    /// The mask; its size equals the region's size.
    pub mask: Mask,
}

impl BoundedMask {
    /// Makes a bounded mask, checking that the mask has the region's size.
    pub fn new(region: Rect, mask: Mask) -> Result<Self> {
        let bounded = Self { region, mask };
        bounded.validate()?;
        Ok(bounded)
    }

    /// Checks the mask and that its size equals the region's size.
    pub fn validate(&self) -> Result<()> {
        self.mask.validate()?;
        if self.mask.size_px() != (self.region.width_px, self.region.height_px) {
            return Err(RetouchError::RegionMismatch {
                region: (self.region.width_px, self.region.height_px),
                mask: self.mask.size_px(),
            });
        }
        Ok(())
    }

    /// Places the mask on a zero mask of `size_px`, clipping what falls outside it.
    pub fn to_full(&self, size_px: (u32, u32)) -> Result<Mask> {
        self.validate()?;
        let mut full = Mask::zeros(size_px.0, size_px.1)?;
        let (fw, fh) = (i64::from(size_px.0), i64::from(size_px.1));
        let mw = self.mask.width_px as usize;
        for sy in 0..self.mask.height_px as usize {
            let dy = i64::from(self.region.y_px) + sy as i64;
            if dy < 0 || dy >= fh {
                continue;
            }
            let x0 = i64::from(self.region.x_px).max(0);
            let x1 = self.region.right_px().min(fw);
            if x0 >= x1 {
                continue;
            }
            let src_start = (x0 - i64::from(self.region.x_px)) as usize;
            let len = (x1 - x0) as usize;
            let dst = dy as usize * size_px.0 as usize + x0 as usize;
            full.data[dst..dst + len]
                .copy_from_slice(&self.mask.data[sy * mw + src_start..sy * mw + src_start + len]);
        }
        Ok(full)
    }

    /// A full-size mask as a bounded mask over the bounding box of its non-zero pixels, or `None`
    /// when it is all zeros (`docs/27` section 5, last paragraph).
    pub fn from_full(full: &Mask) -> Option<Self> {
        let region = Rect::new(0, 0, full.width_px, full.height_px);
        Self {
            region,
            mask: full.clone(),
        }
        .crop_to_nonzero()
    }

    /// Crops the region to the bounding box of the non-zero pixels, or `None` when all are zero.
    pub fn crop_to_nonzero(&self) -> Option<Self> {
        let (w, h) = (self.mask.width_px as usize, self.mask.height_px as usize);
        if self.mask.data.len() != w * h {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        for (y, row) in self.mask.data.chunks_exact(w.max(1)).enumerate() {
            let Some(first) = row.iter().position(|&v| v != 0) else {
                continue;
            };
            let last = row.iter().rposition(|&v| v != 0).unwrap_or(first);
            x0 = x0.min(first);
            x1 = x1.max(last + 1);
            y0 = y0.min(y);
            y1 = y1.max(y + 1);
        }
        if x0 >= x1 || y0 >= y1 {
            return None;
        }
        let (nw, nh) = (x1 - x0, y1 - y0);
        let mut data = Vec::with_capacity(nw * nh);
        for y in y0..y1 {
            data.extend_from_slice(&self.mask.data[y * w + x0..y * w + x1]);
        }
        Some(Self {
            region: Rect::new(
                self.region.x_px.saturating_add(x0 as i32),
                self.region.y_px.saturating_add(y0 as i32),
                nw as u32,
                nh as u32,
            ),
            mask: Mask {
                width_px: nw as u32,
                height_px: nh as u32,
                data,
            },
        })
    }
}

/// One face found in the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// The face rectangle in full-resolution image pixels.
    pub rect: Rect,
    /// Face size as a fraction of the longer image side; clamped to 0.05..=1.0 where used.
    pub scale: f32,
}

/// The settings of the skin tools for one face. Every field is clamped to its range on entry
/// ([`SkinSettings::clamped`]); the derived values clamp too.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SkinSettings {
    /// Blemish amount, 0..=100.
    pub blemish: f32,
    /// Dark circles amount, 0..=100.
    pub dark_circles: f32,
    /// Even skin amount, 0..=100.
    pub even_skin: f32,
    /// Texture, -100..=100.
    pub texture: f32,
    /// Blemish removal also covers the neck.
    pub include_neck_blemish: bool,
    /// Even skin also covers the neck.
    pub include_neck_even_skin: bool,
}

/// Clamps to a range; NaN becomes `neutral` (clamped too).
fn clamp_f32(value: f32, min: f32, max: f32, neutral: f32) -> f32 {
    if value.is_nan() {
        neutral.clamp(min, max)
    } else {
        value.clamp(min, max)
    }
}

impl SkinSettings {
    /// The settings with every amount clamped to its range (NaN becomes 0).
    pub fn clamped(&self) -> Self {
        Self {
            blemish: clamp_f32(self.blemish, 0.0, 100.0, 0.0),
            dark_circles: clamp_f32(self.dark_circles, 0.0, 100.0, 0.0),
            even_skin: clamp_f32(self.even_skin, 0.0, 100.0, 0.0),
            texture: clamp_f32(self.texture, -100.0, 100.0, 0.0),
            ..*self
        }
    }

    /// `a = blemish / 100`, in 0..=1.
    pub fn a(&self) -> f64 {
        f64::from(self.clamped().blemish) / 100.0
    }

    /// `dc = dark_circles / 100`, in 0..=1.
    pub fn dc(&self) -> f64 {
        f64::from(self.clamped().dark_circles) / 100.0
    }

    /// `se = even_skin / 100`, in 0..=1.
    pub fn se(&self) -> f64 {
        f64::from(self.clamped().even_skin) / 100.0
    }

    /// `t = (texture + 100) / 200`, in 0..=1.
    pub fn t(&self) -> f64 {
        (f64::from(self.clamped().texture) + 100.0) / 200.0
    }
}

/// Checks `render_scale` (0.0001 <= s <= 1, finite) and returns it as `f64`.
pub fn check_render_scale(render_scale: f32) -> Result<f64> {
    if render_scale.is_finite() && (0.0001..=1.0).contains(&render_scale) {
        Ok(f64::from(render_scale))
    } else {
        Err(RetouchError::RenderScale(render_scale))
    }
}

fn clamp_face_scale(face: &Face) -> f64 {
    f64::from(clamp_f32(face.scale, 0.05, 1.0, 0.05))
}

/// `D`: the face size in working pixels,
/// `max(full_width_px, full_height_px) * render_scale * clamp(face.scale, 0.05, 1)`.
pub fn face_size_px(full_size_px: (u32, u32), render_scale: f32, face: &Face) -> Result<f64> {
    let scale = check_render_scale(render_scale)?;
    Ok(f64::from(full_size_px.0.max(full_size_px.1)) * scale * clamp_face_scale(face))
}

/// `F`: `trunc(face.scale * max(full_width_px, full_height_px))`, the unit of the feather radii.
pub fn face_extent_px(full_size_px: (u32, u32), face: &Face) -> f64 {
    (clamp_face_scale(face) * f64::from(full_size_px.0.max(full_size_px.1))).trunc()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_validated() {
        assert!(Mask::new(0, 4, vec![]).is_err());
        assert!(Mask::new(2, 2, vec![0; 3]).is_err());
        assert!(Mask::new(2, 2, vec![0; 4]).is_ok());
        assert!(LinearAwg4Image::new(2, 1, vec![[0.0; 3]]).is_err());
        assert!(LinearAwg4Image::new(2, 1, vec![[0.0; 3]; 2]).is_ok());
        let bad = BoundedMask {
            region: Rect::new(0, 0, 3, 3),
            mask: Mask {
                width_px: 2,
                height_px: 2,
                data: vec![0; 4],
            },
        };
        assert!(matches!(
            bad.validate(),
            Err(RetouchError::RegionMismatch { .. })
        ));
    }

    #[test]
    fn settings_clamp_and_derive() {
        let s = SkinSettings {
            blemish: 150.0,
            dark_circles: -5.0,
            even_skin: f32::NAN,
            texture: -300.0,
            ..Default::default()
        };
        let c = s.clamped();
        assert_eq!(
            (c.blemish, c.dark_circles, c.even_skin, c.texture),
            (100.0, 0.0, 0.0, -100.0)
        );
        assert_eq!(s.a(), 1.0);
        assert_eq!(s.dc(), 0.0);
        assert_eq!(s.se(), 0.0);
        assert_eq!(s.t(), 0.0);
        let s = SkinSettings {
            blemish: 25.0,
            texture: 100.0,
            even_skin: 50.0,
            ..Default::default()
        };
        assert_eq!((s.a(), s.se(), s.t()), (0.25, 0.5, 1.0));
        assert_eq!(SkinSettings::default().t(), 0.5);
    }

    #[test]
    fn render_scale_and_d() {
        let face = Face {
            rect: Rect::new(0, 0, 10, 10),
            scale: 0.5,
        };
        assert!(face_size_px((100, 50), 0.0, &face).is_err());
        assert!(face_size_px((100, 50), 1.5, &face).is_err());
        assert!(face_size_px((100, 50), f32::NAN, &face).is_err());
        assert!(face_size_px((100, 50), 0.0001, &face).is_ok());
        assert_eq!(
            face_size_px((100, 50), 0.5, &face).unwrap(),
            100.0 * 0.5 * 0.5
        );
        let tiny = Face {
            scale: 0.001,
            ..face
        };
        assert!((face_size_px((100, 50), 1.0, &tiny).unwrap() - 5.0).abs() < 1e-6);
        let huge = Face { scale: 7.0, ..face };
        assert_eq!(face_extent_px((100, 50), &huge), 100.0);
        assert_eq!(face_extent_px((101, 50), &face), 50.0);
    }

    #[test]
    fn bounding_box_crop() {
        let mut m = Mask::zeros(10, 8).unwrap();
        m.data[2 * 10 + 3] = 9;
        m.data[5 * 10 + 6] = 1;
        let b = BoundedMask::from_full(&m).unwrap();
        assert_eq!(b.region, Rect::new(3, 2, 4, 4));
        assert_eq!(b.mask.data[0], 9);
        assert_eq!(*b.mask.data.last().unwrap(), 1);
        assert_eq!(b.to_full((10, 8)).unwrap(), m);
        assert!(BoundedMask::from_full(&Mask::zeros(4, 4).unwrap()).is_none());
        // A region with an offset keeps its offset.
        let shifted = BoundedMask::new(Rect::new(-2, 5, 10, 8), m.clone())
            .unwrap()
            .crop_to_nonzero()
            .unwrap();
        assert_eq!(shifted.region, Rect::new(1, 7, 4, 4));
    }

    #[test]
    fn to_full_clips() {
        let m = Mask::new(4, 4, vec![7; 16]).unwrap();
        let full = BoundedMask::new(Rect::new(-2, -2, 4, 4), m)
            .unwrap()
            .to_full((3, 3))
            .unwrap();
        let ones: Vec<usize> = full
            .data
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == 7)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(ones, vec![0, 1, 3, 4]);
    }
}
