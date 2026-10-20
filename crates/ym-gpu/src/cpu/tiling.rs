//! Tile rectangles, stage halos and the backwards walk that gives a tile's required input
//! rectangle (`docs/05-raw-processing.md` section Tiling and halos).
//!
//! A tile is a *core* rectangle of the output plus a halo of context. Because every stage in
//! [`crate::cpu`] is tile-consistent (see the module documentation there), computing a stage on
//! `core.grow(halo)` and keeping the core gives the whole-frame result. [`required_input_rect`]
//! chains this over a list of stages, last stage first, to find what must be decoded for one tile.

use super::lens::LensGeometry;
use super::sharpen::CaptureSharpen;

/// An axis-aligned pixel rectangle: `[x, x + w) x [y, y + h)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left edge.
    pub x: usize,
    /// Top edge.
    pub y: usize,
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
}

impl Rect {
    /// A rectangle from its corner and size.
    pub const fn new(x: usize, y: usize, w: usize, h: usize) -> Rect {
        Rect { x, y, w, h }
    }

    /// The rectangle `[0, w) x [0, h)`.
    pub const fn whole(w: usize, h: usize) -> Rect {
        Rect { x: 0, y: 0, w, h }
    }

    /// One past the last column.
    pub const fn x1(&self) -> usize {
        self.x + self.w
    }

    /// One past the last row.
    pub const fn y1(&self) -> usize {
        self.y + self.h
    }

    /// Number of pixels.
    pub const fn area(&self) -> usize {
        self.w * self.h
    }

    /// Whether the rectangle holds no pixels.
    pub const fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// The overlap with `other` (empty, at the corner of `self`, when disjoint).
    pub fn intersect(&self, other: &Rect) -> Rect {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.x1().min(other.x1());
        let y1 = self.y1().min(other.y1());
        if x1 <= x0 || y1 <= y0 {
            Rect::new(x0, y0, 0, 0)
        } else {
            Rect::new(x0, y0, x1 - x0, y1 - y0)
        }
    }

    /// Whether `other` lies entirely inside `self`.
    pub fn contains(&self, other: &Rect) -> bool {
        other.x >= self.x && other.y >= self.y && other.x1() <= self.x1() && other.y1() <= self.y1()
    }

    /// Grows by `halo` on each side and clamps to `bounds`.
    pub fn grow(&self, halo: Halo, bounds: Rect) -> Rect {
        let x0 = self.x.saturating_sub(halo.left).max(bounds.x);
        let y0 = self.y.saturating_sub(halo.top).max(bounds.y);
        let x1 = (self.x1() + halo.right).min(bounds.x1());
        let y1 = (self.y1() + halo.bottom).min(bounds.y1());
        Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }

    /// Like [`grow`](Self::grow), then moves the left and top edges down to a multiple of
    /// `align` (counted from the origin of the frame, not of `bounds`) so the Bayer phase and the
    /// R5 superpixel grid are unchanged. The right and bottom edges are not moved (a Bayer tile
    /// may have an odd size).
    pub fn grow_aligned(&self, halo: Halo, bounds: Rect, align: usize) -> Rect {
        let g = self.grow(halo, bounds);
        let x0 = (g.x / align) * align;
        let y0 = (g.y / align) * align;
        Rect::new(x0, y0, g.x1() - x0, g.y1() - y0)
    }

    /// This rectangle's position inside `outer` (which must contain it).
    pub fn relative_to(&self, outer: &Rect) -> Rect {
        Rect::new(self.x - outer.x, self.y - outer.y, self.w, self.h)
    }
}

/// Context needed on each side of a tile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Halo {
    /// Columns needed left of the tile.
    pub left: usize,
    /// Rows needed above the tile.
    pub top: usize,
    /// Columns needed right of the tile.
    pub right: usize,
    /// Rows needed below the tile.
    pub bottom: usize,
}

impl Halo {
    /// No context.
    pub const NONE: Halo = Halo {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };

    /// `n` pixels on every side.
    pub const fn uniform(n: usize) -> Halo {
        Halo {
            left: n,
            top: n,
            right: n,
            bottom: n,
        }
    }

    /// The side-by-side sum of two halos (two stages in sequence).
    pub const fn plus(self, other: Halo) -> Halo {
        Halo {
            left: self.left + other.left,
            top: self.top + other.top,
            right: self.right + other.right,
            bottom: self.bottom + other.bottom,
        }
    }

    /// The largest of the four sides.
    pub const fn max_side(&self) -> usize {
        let a = if self.left > self.right {
            self.left
        } else {
            self.right
        };
        let b = if self.top > self.bottom {
            self.top
        } else {
            self.bottom
        };
        if a > b { a } else { b }
    }
}

/// Splits a `width x height` frame into core rectangles of at most `tile_w x tile_h`, with every
/// internal edge on a multiple of `align` (use 2 for mosaic-domain tiles). Tiles are in raster order.
pub fn tile_grid(
    width: usize,
    height: usize,
    tile_w: usize,
    tile_h: usize,
    align: usize,
) -> Vec<Rect> {
    let align = align.max(1);
    let tw = (tile_w / align).max(1) * align;
    let th = (tile_h / align).max(1) * align;
    let mut tiles = Vec::new();
    let mut y = 0;
    while y < height {
        let h = th.min(height - y);
        let mut x = 0;
        while x < width {
            let w = tw.min(width - x);
            tiles.push(Rect::new(x, y, w, h));
            x += tw;
        }
        y += th;
    }
    tiles
}

/// One stage of the chain, with the parameters that decide its halo.
#[derive(Clone, Copy, Debug)]
pub enum Stage<'a> {
    /// R1 normalise (pointwise).
    Normalize,
    /// R3 vignetting (pointwise).
    Vignette,
    /// R4 gains (pointwise).
    Gains,
    /// R5 highlight reconstruction (needs the global chroma offsets precomputed).
    Highlights,
    /// R5b lateral CA corrected on the mosaic ([`super::lens::correct_mosaic_ca`]), with the
    /// context [`super::lens::mosaic_ca_halo_px`] asks for.
    MosaicCa {
        /// Context needed on each side, in pixels (even).
        halo_px: usize,
    },
    /// R6 classical demosaic with the given method.
    Demosaic(super::demosaic::Method),
    /// R6 bilateral denoise of window radius `radius_px`.
    Denoise {
        /// Half-width of the bilateral window.
        radius_px: usize,
    },
    /// R6's single-frame denoise of `docs/algorithms/05` in place of the bilateral denoise and R7
    /// ([`super::frame_denoise::FrameDenoise`]).
    FrameDenoise {
        /// Context on each side of the output, in pixels; `None` when the method reads the whole frame.
        halo_px: Option<usize>,
    },
    /// False-colour suppression after the single-frame denoise ([`super::false_color`]): R7's colour
    /// guided filter where the luma alternates.
    FalseColor {
        /// Context on each side ([`super::false_color::halo_px`]).
        halo_px: usize,
    },
    /// R7 colour guided filter and detail re-injection.
    Detail {
        /// Box radius of the guided filter (`2 + 6 C / 100`, rounded).
        chroma_radius_px: usize,
        /// Whether the detail term needs a classical demosaic of the same tile.
        needs_classical: bool,
    },
    /// R8 capture sharpening: Richardson-Lucy or diffraction deconvolution, each with its own halo
    /// ([`CaptureSharpen::halo_px`]).
    Sharpen(CaptureSharpen),
    /// R9 distortion and CA resample.
    Lens(&'a LensGeometry),
}

/// Maximum RL halo (`docs/05`): the scheduler never asks for more than this.
pub const SHARPEN_HALO_CAP_PX: usize = 24;

/// The exact dependency radius of `n` Richardson-Lucy iterations with Gaussian PSFs of standard
/// deviation up to `sigma_max_px`, truncated at 3 sigma: each iteration convolves twice (forward
/// and transposed), so the radius is `2 n ceil(3 sigma)`, capped at [`SHARPEN_HALO_CAP_PX`].
pub fn sharpen_halo(iterations: usize, sigma_max_px: f32) -> usize {
    (2 * iterations * (3.0 * sigma_max_px).ceil() as usize).min(SHARPEN_HALO_CAP_PX)
}

impl Stage<'_> {
    /// The symmetric halo of this stage, in pixels of its own input. `Lens` has no fixed halo
    /// (it depends on the tile): see [`Stage::input_rect`].
    pub fn halo_px(&self) -> usize {
        match *self {
            Stage::Normalize | Stage::Vignette | Stage::Gains => 0,
            // Exact need is 2 px (one superpixel) when the chroma offsets are precomputed; the
            // documented 4 px (two superpixels) also covers computing them from the tile.
            Stage::Highlights => 4,
            Stage::MosaicCa { halo_px } => halo_px,
            Stage::Demosaic(m) => m.halo(),
            Stage::Denoise { radius_px } => radius_px,
            // A method that reads the whole frame has no halo: `input_rect` asks for the frame.
            Stage::FrameDenoise { halo_px } => halo_px.unwrap_or(0),
            Stage::FalseColor { halo_px } => halo_px,
            Stage::Detail {
                chroma_radius_px,
                needs_classical,
            } => {
                let guided = 2 * chroma_radius_px;
                if needs_classical {
                    guided.max(super::demosaic::DEFAULT_METHOD.halo())
                } else {
                    guided
                }
            }
            Stage::Sharpen(method) => method.halo_px(),
            Stage::Lens(_) => 0,
        }
    }

    /// Whether the stage's input is the Bayer mosaic (tiles must then start on even coordinates).
    pub fn is_mosaic_domain(&self) -> bool {
        matches!(
            self,
            Stage::Normalize
                | Stage::Vignette
                | Stage::Gains
                | Stage::Highlights
                | Stage::MosaicCa { .. }
                | Stage::Demosaic(_)
        )
    }

    /// The input rectangle needed to produce `out` (clamped to `frame`, the stage's input frame).
    pub fn input_rect(&self, out: Rect, frame: Rect) -> Rect {
        match self {
            Stage::Lens(geo) => geo.source_rect(out).intersect(&frame),
            Stage::FrameDenoise { halo_px: None } => frame,
            _ => {
                let halo = Halo::uniform(self.halo_px());
                if self.is_mosaic_domain() {
                    out.grow_aligned(halo, frame, 2)
                } else {
                    out.grow(halo, frame)
                }
            }
        }
    }
}

/// Walks `stages` (first to last in processing order) backwards from the wanted output rectangle
/// `out` and returns the rectangle of the first stage's input that must be available.
///
/// `frame` is the full input frame (`Rect::whole(width, height)`); every stage's rectangle is
/// clamped to it. The R9 crop shifts coordinates between its output and its source, so for a chain
/// that ends in [`Stage::Lens`] `out` is in the cropped output's coordinates and everything before
/// the lens stage is in source-frame coordinates.
pub fn required_input_rect(stages: &[Stage<'_>], out: Rect, frame: Rect) -> Rect {
    stages
        .iter()
        .rev()
        .fold(out, |rect, stage| stage.input_rect(rect, frame))
}

#[cfg(test)]
mod tests {
    use super::super::sharpen::SharpenParams;
    use super::*;

    #[test]
    fn grow_clamps_and_aligns() {
        let frame = Rect::whole(100, 80);
        let core = Rect::new(31, 17, 20, 20);
        assert_eq!(core.grow(Halo::uniform(8), frame), Rect::new(23, 9, 36, 36));
        let a = core.grow_aligned(Halo::uniform(8), frame, 2);
        assert_eq!((a.x % 2, a.y % 2), (0, 0));
        assert!(a.contains(&core.grow(Halo::uniform(8), frame)));
        assert_eq!(
            Rect::new(0, 0, 10, 10).grow(Halo::uniform(50), frame),
            Rect::new(0, 0, 60, 60)
        );
        assert_eq!(
            Rect::new(90, 70, 10, 10).grow(Halo::uniform(5), frame),
            Rect::new(85, 65, 15, 15)
        );
    }

    #[test]
    fn intersect_and_relative() {
        let a = Rect::new(10, 10, 20, 20);
        let b = Rect::new(20, 25, 30, 30);
        assert_eq!(a.intersect(&b), Rect::new(20, 25, 10, 5));
        assert!(a.intersect(&Rect::new(40, 40, 5, 5)).is_empty());
        assert_eq!(
            Rect::new(12, 13, 4, 4).relative_to(&a),
            Rect::new(2, 3, 4, 4)
        );
    }

    #[test]
    fn tile_grid_covers_frame_exactly_once() {
        let tiles = tile_grid(101, 77, 32, 24, 2);
        let mut seen = vec![0u8; 101 * 77];
        for t in &tiles {
            assert_eq!((t.x % 2, t.y % 2), (0, 0));
            for y in t.y..t.y1() {
                for x in t.x..t.x1() {
                    seen[y * 101 + x] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn sharpen_halo_matches_the_documented_cap() {
        // Auto settings: A = 120 -> n = 4, rho = 0.8 -> 1.25 rho = 1.0 in the corners.
        assert_eq!(sharpen_halo(4, 1.0), 24);
        assert_eq!(sharpen_halo(1, 1.0), 6);
        assert_eq!(sharpen_halo(10, 3.0), SHARPEN_HALO_CAP_PX);
        assert_eq!(sharpen_halo(0, 1.0), 0);
    }

    #[test]
    fn each_capture_sharpening_method_has_its_own_halo() {
        use super::super::sharpen::diffraction::DiffractionParams;
        let rl = |amount| {
            CaptureSharpen::RichardsonLucy(SharpenParams {
                amount,
                radius_px: 0.8,
            })
        };
        // Richardson-Lucy: 2 n ceil(3 sigma_max), capped at 24 px.
        assert_eq!(Stage::Sharpen(rl(120.0)).halo_px(), 24);
        assert_eq!(Stage::Sharpen(rl(60.0)).halo_px(), 12);
        assert_eq!(Stage::Sharpen(rl(300.0)).halo_px(), SHARPEN_HALO_CAP_PX);
        // Diffraction: 10 + R + r_G of the plan (spec 03 section 1.5), not capped by the RL limit.
        let d = |pitch, f| {
            Stage::Sharpen(CaptureSharpen::Diffraction(DiffractionParams::new(
                pitch, f,
            )))
        };
        assert_eq!(d(3.76, 2.8).halo_px(), 13);
        assert_eq!(d(3.76, 8.0).halo_px(), 15);
        assert_eq!(d(3.76, 16.0).halo_px(), 18);
        assert_eq!(d(3.76, 50.0).halo_px(), 22);
        assert_eq!(d(2.4, 45.0).halo_px(), 24);
        // An inactive stage returns its input and needs no context.
        assert_eq!(d(3.76, 0.0).halo_px(), 0);
        assert_eq!(d(0.0, 8.0).halo_px(), 0);
        // The walk back uses the method's halo.
        let frame = Rect::whole(1000, 800);
        let out = Rect::new(400, 300, 100, 100);
        let need = required_input_rect(&[d(3.76, 16.0)], out, frame);
        assert_eq!(need, Rect::new(382, 282, 136, 136));
    }

    #[test]
    fn backwards_walk_accumulates_halos() {
        use super::super::demosaic::Method;
        let frame = Rect::whole(1000, 800);
        let chain = [
            Stage::Normalize,
            Stage::Gains,
            Stage::Highlights,
            Stage::Demosaic(Method::Menon),
            Stage::Denoise { radius_px: 2 },
            Stage::Sharpen(CaptureSharpen::RichardsonLucy(SharpenParams {
                amount: 60.0,
                radius_px: 0.8,
            })),
        ];
        let out = Rect::new(400, 300, 100, 100);
        let need = required_input_rect(&chain, out, frame);
        // 12 (sharpen) + 2 (denoise) + 8 (menon) + 4 (highlights), aligned to even coordinates.
        let total = 12 + 2 + Method::Menon.halo() + 4;
        assert_eq!(need.x, (400 - total) / 2 * 2);
        assert!(need.contains(&out));
        assert_eq!(need.x1(), 500 + total);
    }
}
