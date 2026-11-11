//! Deterministic mock models for tests (`docs/27` section 4). Enabled by the `mock` feature.
//!
//! Everything is analytic and uses no randomness. Geometry is relative to the face rectangle:
//! the skin is the ellipse inscribed in it, the eyes are two small ellipses, the neck is a
//! rectangle below it.

use crate::error::{Result, RetouchError};
use crate::models::{
    BeautyMarkDetector, BlemishProposer, FaceMesh, FaceParser, FacePart, MeshTable, Models,
};
use crate::types::{BoundedMask, Face, LinearAwg4Image, Mask, Rect};

/// Centres of the two eyes of the mock face, in image pixels.
pub fn eye_centers(rect: &Rect) -> [(f64, f64); 2] {
    let (x, y, w, h) = (
        f64::from(rect.x_px),
        f64::from(rect.y_px),
        f64::from(rect.width_px),
        f64::from(rect.height_px),
    );
    [(x + 0.32 * w, y + 0.40 * h), (x + 0.68 * w, y + 0.40 * h)]
}

/// An ellipse (centre, semi-axes) on `region`: `value(d)` for the normalised distance `d < 1`
/// at the pixel centre, else 0.
fn ellipses(
    region: Rect,
    shapes: &[(f64, f64, f64, f64)],
    value: impl Fn(f64) -> u8,
) -> BoundedMask {
    let (w, h) = (region.width_px.max(1), region.height_px.max(1));
    let mut data = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (
                f64::from(region.x_px) + f64::from(x) + 0.5,
                f64::from(region.y_px) + f64::from(y) + 0.5,
            );
            let mut best = 0u8;
            for &(cx, cy, ax, ay) in shapes {
                let (nx, ny) = ((px - cx) / ax, (py - cy) / ay);
                let d2 = nx * nx + ny * ny;
                if d2 < 1.0 {
                    best = best.max(value(d2.sqrt()));
                }
            }
            data[(y * w + x) as usize] = best;
        }
    }
    BoundedMask {
        region: Rect {
            width_px: w,
            height_px: h,
            ..region
        },
        mask: Mask {
            width_px: w,
            height_px: h,
            data,
        },
    }
}

/// A parser giving an elliptical skin mask inside the face rectangle, two small elliptical eyes,
/// and the neck as a rectangle below the face.
#[derive(Debug, Clone, Copy, Default)]
pub struct MockParser;

impl FaceParser for MockParser {
    fn part_mask(
        &self,
        _image: &LinearAwg4Image,
        face: &Face,
        part: FacePart,
    ) -> Result<Option<BoundedMask>> {
        let r = face.rect;
        let (x, y, w, h) = (
            f64::from(r.x_px),
            f64::from(r.y_px),
            f64::from(r.width_px),
            f64::from(r.height_px),
        );
        Ok(Some(match part {
            FacePart::SkinWithoutEars => {
                ellipses(r, &[(x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0)], |_| 255)
            }
            FacePart::Eyes => {
                let [(lx, ly), (rx, ry)] = eye_centers(&r);
                let (ax, ay) = (0.09 * w, 0.045 * h);
                let region = Rect::new(
                    (lx - ax).floor() as i32,
                    (ly - ay).floor() as i32,
                    ((rx - lx) + 2.0 * ax).ceil() as u32 + 1,
                    (2.0 * ay).ceil() as u32 + 1,
                );
                ellipses(region, &[(lx, ly, ax, ay), (rx, ry, ax, ay)], |_| 255)
            }
            FacePart::NeckSkin => {
                let region = Rect::new(
                    r.x_px + (0.25 * w) as i32,
                    r.y_px + r.height_px as i32,
                    (0.5 * w) as u32,
                    (0.4 * h) as u32,
                );
                BoundedMask {
                    region,
                    mask: Mask {
                        width_px: region.width_px,
                        height_px: region.height_px,
                        data: vec![255; (region.width_px * region.height_px) as usize],
                    },
                }
            }
        }))
    }
}

/// A parser that finds nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoParts;

impl FaceParser for NoParts {
    fn part_mask(
        &self,
        _image: &LinearAwg4Image,
        _face: &Face,
        _part: FacePart,
    ) -> Result<Option<BoundedMask>> {
        Ok(None)
    }
}

/// A mesh that draws analytic tables on the face rectangle: crescent-like under-eye patches (an
/// ellipse below each eye, brightest at its centre), and an inverted-uniformity pattern that is
/// bright on the mouth and falls off to 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct MockMesh;

impl FaceMesh for MockMesh {
    fn render(&self, face: &Face, table: MeshTable, _size_px: (u32, u32)) -> Result<BoundedMask> {
        let r = face.rect;
        let (x, y, w, h) = (
            f64::from(r.x_px),
            f64::from(r.y_px),
            f64::from(r.width_px),
            f64::from(r.height_px),
        );
        let fall = |d: f64| ((1.0 - d * d) * 255.0).round().clamp(0.0, 255.0) as u8;
        Ok(match table {
            MeshTable::UnderEye => {
                let [(lx, ly), (rx, ry)] = eye_centers(&r);
                let (dy, ax, ay) = (0.09 * h, 0.14 * w, 0.045 * h);
                ellipses(r, &[(lx, ly + dy, ax, ay), (rx, ry + dy, ax, ay)], fall)
            }
            MeshTable::InvertedSkinUniformity => {
                ellipses(r, &[(x + 0.5 * w, y + 0.78 * h, 0.22 * w, 0.06 * h)], fall)
            }
        })
    }
}

/// A beauty-mark detector returning fixed boxes.
#[derive(Debug, Clone, Default)]
pub struct MockBeautyMarks {
    /// The boxes returned for every face.
    pub boxes: Vec<Rect>,
}

impl BeautyMarkDetector for MockBeautyMarks {
    fn detect(&self, _image: &LinearAwg4Image, _face: &Face) -> Result<Vec<Rect>> {
        Ok(self.boxes.clone())
    }
}

/// A beauty-mark detector that always fails.
#[derive(Debug, Clone, Copy, Default)]
pub struct FailingBeautyMarks;

impl BeautyMarkDetector for FailingBeautyMarks {
    fn detect(&self, _image: &LinearAwg4Image, _face: &Face) -> Result<Vec<Rect>> {
        Err(RetouchError::Model("mock failure".into()))
    }
}

/// What a [`MockProposer`] does to the centre crop of its input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProposerMode {
    /// Returns the centre crop unchanged.
    Identity,
    /// Multiplies one channel (0 = R, 1 = G, 2 = B) by `factor`.
    Tint {
        /// The channel.
        channel: usize,
        /// The factor.
        factor: f32,
    },
    /// Multiplies every channel by `factor`.
    Scale(f32),
}

/// A proposer that crops the border off its input (border 32) and optionally tints or scales it.
#[derive(Debug, Clone, Copy)]
pub struct MockProposer {
    tile_px: u32,
    mode: ProposerMode,
}

impl MockProposer {
    /// Border consumed on each side.
    pub const BORDER_PX: u32 = 32;

    /// The identity proposer with the given output tile side (at least 1).
    pub fn identity(tile_px: u32) -> Self {
        Self {
            tile_px: tile_px.max(1),
            mode: ProposerMode::Identity,
        }
    }

    /// A proposer that multiplies one channel by `factor`.
    pub fn tint(tile_px: u32, channel: usize, factor: f32) -> Self {
        Self {
            tile_px: tile_px.max(1),
            mode: ProposerMode::Tint { channel, factor },
        }
    }

    /// A proposer that multiplies all channels by `factor`.
    pub fn scale(tile_px: u32, factor: f32) -> Self {
        Self {
            tile_px: tile_px.max(1),
            mode: ProposerMode::Scale(factor),
        }
    }
}

impl BlemishProposer for MockProposer {
    fn output_size_px(&self) -> u32 {
        self.tile_px
    }

    fn border_px(&self) -> u32 {
        Self::BORDER_PX
    }

    fn run(&self, input: &[f32]) -> Result<Vec<f32>> {
        let (tile, border) = (self.tile_px as usize, Self::BORDER_PX as usize);
        let side = tile + 2 * border;
        if input.len() != 3 * side * side {
            return Err(RetouchError::Model(format!(
                "the proposer got {} values, expected {}",
                input.len(),
                3 * side * side
            )));
        }
        let mut out = Vec::with_capacity(3 * tile * tile);
        for c in 0..3 {
            let factor = match self.mode {
                ProposerMode::Identity => 1.0,
                ProposerMode::Tint { channel, factor } => {
                    if channel == c {
                        factor
                    } else {
                        1.0
                    }
                }
                ProposerMode::Scale(factor) => factor,
            };
            for y in 0..tile {
                let start = c * side * side + (y + border) * side + border;
                out.extend(input[start..start + tile].iter().map(|v| v * factor));
            }
        }
        Ok(out)
    }
}

/// The mock models with the given beauty-mark boxes: the mock parser for skin, eyes and neck, the
/// mock mesh, a mock detector and the identity proposer with 64 px tiles.
pub fn models(beauty_marks: Vec<Rect>) -> Models {
    Models {
        parser: Box::new(MockParser),
        neck: Some(Box::new(MockParser)),
        mesh: Box::new(MockMesh),
        beauty_marks: Some(Box::new(MockBeautyMarks {
            boxes: beauty_marks,
        })),
        proposer: Box::new(MockProposer::identity(64)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposer_modes() {
        let (tile, side) = (4usize, 4 + 64);
        let input: Vec<f32> = (0..3 * side * side).map(|i| i as f32).collect();
        let id = MockProposer::identity(4).run(&input).unwrap();
        assert_eq!(id.len(), 3 * 16);
        assert_eq!(id[0], input[32 * side + 32]);
        assert_eq!(id[16], input[side * side + 32 * side + 32]);
        let tinted = MockProposer::tint(4, 1, 1.5).run(&input).unwrap();
        assert_eq!(&tinted[..16], &id[..16]);
        assert_eq!(tinted[16], id[16] * 1.5);
        let scaled = MockProposer::scale(4, 0.5).run(&input).unwrap();
        assert!(scaled.iter().zip(&id).all(|(s, i)| *s == i * 0.5));
        assert!(MockProposer::identity(4).run(&input[1..]).is_err());
        assert_eq!(tile, 4);
    }

    #[test]
    fn mock_geometry_is_inside_the_face() {
        let face = Face {
            rect: Rect::new(10, 10, 60, 80),
            scale: 0.5,
        };
        let img = LinearAwg4Image::filled(100, 100, [0.0; 3]).unwrap();
        let skin = MockParser
            .part_mask(&img, &face, FacePart::SkinWithoutEars)
            .unwrap()
            .unwrap();
        assert_eq!(skin.region, face.rect);
        assert_eq!(skin.mask.get(30, 40), 255);
        assert_eq!(
            skin.mask.get(0, 0),
            0,
            "corner of the rect is outside the ellipse"
        );
        assert!(
            MockMesh
                .render(&face, MeshTable::UnderEye, (100, 100))
                .unwrap()
                .mask
                .data
                .iter()
                .any(|&v| v > 0)
        );
    }
}
