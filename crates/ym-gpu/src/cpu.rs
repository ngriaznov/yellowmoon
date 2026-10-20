//! CPU reference implementations of every kernel, used for parity tests and for machines without a usable GPU (`docs/05-raw-processing.md`).
//!
//! This module is the classical raw-processing chain R1 to R9. The WGSL kernels are parity-tested
//! against it, so every function here is deterministic (no thread-count dependence: reductions
//! are accumulated per band in f64 and merged in a fixed order) and tile-friendly.
//!
//! # Conventions
//!
//! * **Storage** is `f32`. Intermediate sums, positions, spline evaluation and anything that
//!   accumulates over many samples use `f64`; per-pixel filters use `f32`. Colour maths on the
//!   CPU reference path is `f64` (`AGENTS.md`), and nothing in this chain is colour maths beyond
//!   the luma weights in [`LUMA`].
//! * **Mosaic** ([`Mosaic`]): one `f32` per photosite, row-major, plus the 2x2 CFA pattern seen
//!   from the buffer's own origin. Values are scene-linear, black subtracted, white at 1.0.
//! * **Image** ([`RgbImage`]): *planar* `f32` (all red samples, then all green, then all blue, each
//!   plane row-major). Planar suits the GPU: a plane is one coalesced storage-buffer range, there
//!   is no `vec3` alignment padding, R9 samples the three planes at different positions (lateral
//!   CA), R7/R8 touch luma or the chroma planes alone, and a plane packs into RGBA textures or
//!   f16 storage with a plain copy.
//! * **Placement** ([`Placement`]): every buffer remembers where its (0, 0) sits in the full
//!   frame, so stages that need absolute geometry (vignetting radius, lens maps, the corner PSF
//!   of R8, per-position black levels, the superpixel grid of R5) give identical results whether
//!   they see the whole frame or a tile cut out of it with [`Mosaic::extract`] /
//!   [`RgbImage::extract`].
//! * **Tiling contract**: for a stage with halo `h`, running it on a tile that includes `h` pixels
//!   of context around the core and keeping the core gives the same pixels as running it on the
//!   whole frame (pointwise stages: bit-exact; the others: within 1e-4, in practice 1e-6).
//!   [`tiling::Stage::halo_px`] gives the halos and [`tiling::required_input_rect`] walks a chain of
//!   stages backwards from a wanted output rectangle; [`chain::Chain`] runs the whole classical
//!   chain on one tile or on the whole frame. Mosaic-domain tiles start on even
//!   coordinates so the CFA phase and the R5 superpixel grid are unchanged.
//! * **Borders**: mosaic stages and filters mirror at the buffer edge without repeating the edge
//!   sample and keep the CFA parity ([`mirror`]); R9 clamps to the edge pixel.
//!
//! # Mapping from `ym_raw::RawImage` (for the integrator)
//!
//! This module does not depend on `ym-raw`. Plain inputs:
//!
//! | `RawImage` field | Here |
//! |---|---|
//! | `data: Plane<u16>` (active area, crop applied) | `&[u16]` plus `width`, `height` for [`normalize::normalize`] |
//! | `layout: Bayer { pattern, .. }` | [`Cfa`] (`Cfa::RGGB`, ...), the pattern at the first pixel of the active area |
//! | `black: [f32; 4]`, `white: [f32; 4]` | [`normalize::Levels`], indexed by CFA *position* `2 * (y & 1) + (x & 1)` of the active area; [`normalize::Levels::from_rggb`] maps values in the R, G1, G2, B order of the SR2 tags onto the positions of any pattern phase |
//! | `qstep: Option<Plane<f16>>` | `Option<&[f32]>` (convert f16 to f32) |
//! | `wb_as_shot: Option<[f32; 4]>` (RGGB levels) | [`gains::g0_from_wb_levels`] |
//! | `lens: LensTags` (`ym_raw::LensTable::raw`: the stored count, then the knots) | [`lens::LensTags::from_stored`], which cuts each table to its stated count (n for tags 0x7032 and 0x7037, 2n for 0x7035: red then blue back to back) and refuses inconsistent ones; an absent tag is an empty vector. `LensTable::knots()` alone is not enough: it keeps any padding after the knots |
//! | `decode.*` switches | [`lens::LensSwitches`] (distortion and CA), [`vignette::VignetteProfile::from_tags`] (vignetting follows `lens_corrections`) |

use std::ops::Range;

pub mod chain;
pub mod demosaic;
pub mod denoise;
pub mod detail;
pub mod filters;
pub mod gains;
pub mod highlights;
pub mod lens;
pub mod magnify;
pub mod metrics;
pub mod noise;
pub mod normalize;
pub mod output_sharpen;
pub mod resample;
pub mod sampler;
pub mod sharpen;
pub mod spline;
pub mod synth;
pub mod tiling;
pub mod vignette;

pub use noise::NoiseDn;
pub use tiling::{Halo, Rect};

/// Luma weights used wherever a stage needs a luminance (R6 denoise, R7, R8): the Rec. 709 / sRGB
/// weights, applied to white-balanced camera RGB. Any positive weights summing to 1 keep the
/// opponent chroma `R - Y`, `B - Y` well defined; these favour green, which carries the least noise
/// after white balance gains.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Errors from the CPU reference stages.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CpuError {
    /// A buffer's length does not match its stated dimensions.
    #[error("buffer holds {got} samples but {width} x {height} needs {expected}")]
    BadLength {
        /// Samples supplied.
        got: usize,
        /// Samples required.
        expected: usize,
        /// Stated width in pixels.
        width: usize,
        /// Stated height in pixels.
        height: usize,
    },
    /// A rectangle does not lie inside the buffer it addresses.
    #[error("rectangle {rect:?} is outside the {width} x {height} buffer")]
    RectOutside {
        /// The offending rectangle.
        rect: Rect,
        /// Buffer width in pixels.
        width: usize,
        /// Buffer height in pixels.
        height: usize,
    },
    /// A lens tag has too few knots to interpolate.
    #[error("lens tag {tag} needs at least {min} knots, got {got}")]
    TooFewKnots {
        /// The tag name.
        tag: &'static str,
        /// Knots supplied.
        got: usize,
        /// Knots required.
        min: usize,
    },
    /// White-balance levels cannot be turned into gains.
    #[error("white balance levels {0:?} are not usable (need positive, finite values)")]
    BadWhiteBalance([f32; 4]),
    /// Two buffers that must have equal dimensions do not.
    #[error("dimension mismatch: {0}")]
    Mismatch(String),
}

/// One colour of a Bayer filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CfaColor {
    /// Red.
    Red = 0,
    /// Green (either of the two greens).
    Green = 1,
    /// Blue.
    Blue = 2,
}

impl CfaColor {
    /// Plane index of this colour in an [`RgbImage`] (red 0, green 1, blue 2).
    pub const fn plane(self) -> usize {
        self as usize
    }
}

/// A 2x2 Bayer pattern as seen from the first pixel of a buffer.
///
/// `pattern[y & 1][x & 1]` is the colour of the filter over photosite `(x, y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cfa {
    pattern: [[CfaColor; 2]; 2],
}

impl Cfa {
    /// Red at (0, 0), blue at (1, 1): all current Sony bodies.
    pub const RGGB: Cfa = Cfa::new([
        [CfaColor::Red, CfaColor::Green],
        [CfaColor::Green, CfaColor::Blue],
    ]);
    /// Green, red / blue, green.
    pub const GRBG: Cfa = Cfa::new([
        [CfaColor::Green, CfaColor::Red],
        [CfaColor::Blue, CfaColor::Green],
    ]);
    /// Green, blue / red, green.
    pub const GBRG: Cfa = Cfa::new([
        [CfaColor::Green, CfaColor::Blue],
        [CfaColor::Red, CfaColor::Green],
    ]);
    /// Blue at (0, 0), red at (1, 1).
    pub const BGGR: Cfa = Cfa::new([
        [CfaColor::Blue, CfaColor::Green],
        [CfaColor::Green, CfaColor::Red],
    ]);
    /// The four phases, in the order RGGB, GRBG, GBRG, BGGR.
    pub const ALL: [Cfa; 4] = [Cfa::RGGB, Cfa::GRBG, Cfa::GBRG, Cfa::BGGR];

    /// Builds a pattern from `[[(0,0), (1,0)], [(0,1), (1,1)]]`.
    pub const fn new(pattern: [[CfaColor; 2]; 2]) -> Cfa {
        Cfa { pattern }
    }

    /// Parses `"RGGB"`, `"GRBG"`, `"GBRG"` or `"BGGR"` (case-insensitive).
    pub fn from_name(name: &str) -> Option<Cfa> {
        match name.to_ascii_uppercase().as_str() {
            "RGGB" => Some(Cfa::RGGB),
            "GRBG" => Some(Cfa::GRBG),
            "GBRG" => Some(Cfa::GBRG),
            "BGGR" => Some(Cfa::BGGR),
            _ => None,
        }
    }

    /// The pattern name in raster order (`"RGGB"` for [`Cfa::RGGB`]).
    pub fn name(&self) -> String {
        let c = |c: CfaColor| match c {
            CfaColor::Red => 'R',
            CfaColor::Green => 'G',
            CfaColor::Blue => 'B',
        };
        [
            self.pattern[0][0],
            self.pattern[0][1],
            self.pattern[1][0],
            self.pattern[1][1],
        ]
        .iter()
        .map(|&x| c(x))
        .collect()
    }

    /// Colour of the filter over photosite `(x, y)`.
    #[inline(always)]
    pub const fn color_at(&self, x: usize, y: usize) -> CfaColor {
        self.pattern[y & 1][x & 1]
    }

    /// The pattern seen from a buffer whose first pixel is `(dx, dy)` of this one.
    pub const fn shifted(&self, dx: usize, dy: usize) -> Cfa {
        Cfa {
            pattern: [
                [
                    self.pattern[dy & 1][dx & 1],
                    self.pattern[dy & 1][(dx + 1) & 1],
                ],
                [
                    self.pattern[(dy + 1) & 1][dx & 1],
                    self.pattern[(dy + 1) & 1][(dx + 1) & 1],
                ],
            ],
        }
    }

    /// Index of photosite `(x, y)` within the 2x2 block, `2 * (y & 1) + (x & 1)`: the order of the
    /// four per-position black and white levels.
    #[inline(always)]
    pub const fn position(x: usize, y: usize) -> usize {
        2 * (y & 1) + (x & 1)
    }

    /// Whether this pattern has red on rows where `row & 1 == 0` (the row parity of the red
    /// photosites).
    pub const fn red_row_parity(&self) -> usize {
        if matches!(self.pattern[0][0], CfaColor::Red)
            || matches!(self.pattern[0][1], CfaColor::Red)
        {
            0
        } else {
            1
        }
    }
}

/// Where a buffer's (0, 0) sits in the full frame, and how big that frame is.
///
/// Stages that depend on absolute geometry read this instead of taking extra arguments, so a tile
/// carries everything needed to give the whole-frame answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    /// Frame x of the buffer's first column.
    pub x0: usize,
    /// Frame y of the buffer's first row.
    pub y0: usize,
    /// Width of the full frame (the active area) in pixels.
    pub frame_width: usize,
    /// Height of the full frame in pixels.
    pub frame_height: usize,
}

impl Placement {
    /// A buffer that is the whole frame.
    pub const fn whole(width: usize, height: usize) -> Placement {
        Placement {
            x0: 0,
            y0: 0,
            frame_width: width,
            frame_height: height,
        }
    }

    /// The placement of a sub-buffer starting at `(dx, dy)` of this one.
    pub const fn offset(&self, dx: usize, dy: usize) -> Placement {
        Placement {
            x0: self.x0 + dx,
            y0: self.y0 + dy,
            ..*self
        }
    }

    /// Frame centre in continuous pixel coordinates (pixel `i` covers `[i, i + 1)`).
    pub fn centre(&self) -> (f64, f64) {
        (
            self.frame_width as f64 * 0.5,
            self.frame_height as f64 * 0.5,
        )
    }

    /// Half the frame diagonal in pixels: the length that normalises the radius `r` of the lens
    /// tags (`r = 1` at the corner of the active area).
    pub fn half_diagonal(&self) -> f64 {
        0.5 * (self.frame_width as f64).hypot(self.frame_height as f64)
    }
}

/// Mirrors an index into `0..n` without repeating the edge sample (`-1 -> 1`, `n -> n - 2`).
///
/// Mirroring about the edge sample keeps the parity of the index, so the Bayer phase survives.
/// For `n == 1` returns 0.
#[inline(always)]
pub fn mirror(i: isize, n: usize) -> usize {
    if (i as usize) < n {
        return i as usize;
    }
    mirror_slow(i, n)
}

#[cold]
#[inline(never)]
fn mirror_slow(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as isize;
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - j;
    }
    j as usize
}

/// Clamps an index into `0..n`.
#[inline(always)]
pub fn clamp_index(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// The linear-light luminance of one pixel with the weights in [`LUMA`].
#[inline(always)]
pub fn luma(rgb: [f32; 3]) -> f32 {
    LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]
}

/// A noise standard deviation in the same linear units as the image: one scalar or one value per
/// pixel (the σ map of `docs/06-noise.md`, in white-balanced camera RGB units).
#[derive(Clone, Copy, Debug)]
pub enum NoiseSigma<'a> {
    /// The same σ everywhere.
    Scalar(f32),
    /// One σ per pixel, row-major, the size of the whole image the stage is run on.
    Plane(&'a [f32]),
}

impl NoiseSigma<'_> {
    /// σ at pixel index `i` (row-major in the image the plane belongs to).
    #[inline(always)]
    pub fn at(&self, i: usize) -> f32 {
        match self {
            NoiseSigma::Scalar(s) => *s,
            NoiseSigma::Plane(p) => p[i],
        }
    }

    /// The σ map expanded to `len` values, each multiplied by `scale`.
    pub fn to_vec(&self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|i| self.at(i) * scale).collect()
    }
}

/// A black-subtracted, normalised mosaic: one `f32` per photosite.
#[derive(Clone, Debug, PartialEq)]
pub struct Mosaic {
    /// Width in photosites.
    pub width: usize,
    /// Height in photosites.
    pub height: usize,
    /// The 2x2 pattern at this buffer's (0, 0).
    pub cfa: Cfa,
    /// Position of this buffer in the full frame.
    pub place: Placement,
    /// Row-major samples, `width * height` of them.
    pub data: Vec<f32>,
}

impl Mosaic {
    /// A zero-filled mosaic that is the whole frame.
    pub fn new(width: usize, height: usize, cfa: Cfa) -> Mosaic {
        Mosaic {
            width,
            height,
            cfa,
            place: Placement::whole(width, height),
            data: vec![0.0; width * height],
        }
    }

    /// Wraps existing samples as a whole-frame mosaic.
    pub fn from_data(
        width: usize,
        height: usize,
        cfa: Cfa,
        data: Vec<f32>,
    ) -> Result<Mosaic, CpuError> {
        if data.len() != width * height {
            return Err(CpuError::BadLength {
                got: data.len(),
                expected: width * height,
                width,
                height,
            });
        }
        Ok(Mosaic {
            width,
            height,
            cfa,
            place: Placement::whole(width, height),
            data,
        })
    }

    /// Sample at `(x, y)`.
    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Row `y`.
    #[inline(always)]
    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    /// Colour of photosite `(x, y)` of this buffer.
    #[inline(always)]
    pub fn color_at(&self, x: usize, y: usize) -> CfaColor {
        self.cfa.color_at(x, y)
    }

    /// Cuts out `rect` (buffer coordinates) as a new mosaic with the CFA phase and placement
    /// adjusted. Panics if `rect` is outside the buffer (callers clamp with [`Rect::grow`]).
    pub fn extract(&self, rect: Rect) -> Mosaic {
        assert!(
            rect.x1() <= self.width && rect.y1() <= self.height,
            "extract rect outside mosaic"
        );
        let mut data = Vec::with_capacity(rect.w * rect.h);
        for y in rect.y..rect.y1() {
            data.extend_from_slice(&self.data[y * self.width + rect.x..y * self.width + rect.x1()]);
        }
        Mosaic {
            width: rect.w,
            height: rect.h,
            cfa: self.cfa.shifted(rect.x, rect.y),
            place: self.place.offset(rect.x, rect.y),
            data,
        }
    }

    /// Copies the sub-rectangle `src_rect` of `tile` (tile coordinates) into this mosaic with its
    /// top-left corner at `(dst_x, dst_y)`.
    pub fn paste(&mut self, tile: &Mosaic, src_rect: Rect, dst_x: usize, dst_y: usize) {
        for dy in 0..src_rect.h {
            let s = (src_rect.y + dy) * tile.width + src_rect.x;
            let d = (dst_y + dy) * self.width + dst_x;
            self.data[d..d + src_rect.w].copy_from_slice(&tile.data[s..s + src_rect.w]);
        }
    }
}

/// Which photosites hold clipped values.
///
/// One byte per photosite: 1 where the raw value reached the white level (minus a margin of four
/// quantisation steps), 0 elsewhere. The mask is made by [`normalize::normalize`] and travels
/// unchanged to R5, R6 and C7. After R5 the flagged values are estimates, not measurements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipMask {
    /// Width in photosites.
    pub width: usize,
    /// Height in photosites.
    pub height: usize,
    /// Row-major flags, 0 or 1.
    pub data: Vec<u8>,
}

impl ClipMask {
    /// An all-clear mask.
    pub fn new(width: usize, height: usize) -> ClipMask {
        ClipMask {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    /// Whether photosite `(x, y)` is clipped.
    #[inline(always)]
    pub fn is_clipped(&self, x: usize, y: usize) -> bool {
        self.data[y * self.width + x] != 0
    }

    /// Number of clipped photosites.
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0).count()
    }

    /// Cuts out `rect`.
    pub fn extract(&self, rect: Rect) -> ClipMask {
        assert!(
            rect.x1() <= self.width && rect.y1() <= self.height,
            "extract rect outside mask"
        );
        let mut data = Vec::with_capacity(rect.w * rect.h);
        for y in rect.y..rect.y1() {
            data.extend_from_slice(&self.data[y * self.width + rect.x..y * self.width + rect.x1()]);
        }
        ClipMask {
            width: rect.w,
            height: rect.h,
            data,
        }
    }

    /// Per-pixel clip flags for a demosaiced image: bit `c` (red 1, green 2, blue 4) of the
    /// result is set where any photosite of colour `c` within the 3x3 neighbourhood is clipped.
    /// This is the form C7 and the R6 network consume. `cfa` is the mosaic's pattern.
    pub fn rgb_flags(&self, cfa: Cfa) -> Vec<u8> {
        let (w, h) = (self.width, self.height);
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut f = 0u8;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let xx = mirror(x as isize + dx, w);
                        let yy = mirror(y as isize + dy, h);
                        if self.data[yy * w + xx] != 0 {
                            f |= 1 << cfa.color_at(xx, yy).plane();
                        }
                    }
                }
                out[y * w + x] = f;
            }
        }
        out
    }
}

/// A demosaiced image: three `f32` planes (red, green, blue), planar and row-major.
///
/// See the module documentation for why the layout is planar.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Position of this buffer in the full frame.
    pub place: Placement,
    /// `3 * width * height` samples: the red plane, then green, then blue.
    pub data: Vec<f32>,
}

impl RgbImage {
    /// A zero-filled image that is the whole frame.
    pub fn new(width: usize, height: usize) -> RgbImage {
        RgbImage {
            width,
            height,
            place: Placement::whole(width, height),
            data: vec![0.0; 3 * width * height],
        }
    }

    /// Wraps existing planar samples as a whole-frame image.
    pub fn from_planar(width: usize, height: usize, data: Vec<f32>) -> Result<RgbImage, CpuError> {
        if data.len() != 3 * width * height {
            return Err(CpuError::BadLength {
                got: data.len(),
                expected: 3 * width * height,
                width,
                height,
            });
        }
        Ok(RgbImage {
            width,
            height,
            place: Placement::whole(width, height),
            data,
        })
    }

    /// Builds an image from interleaved `[r, g, b, r, g, b, ...]` samples.
    pub fn from_interleaved(
        width: usize,
        height: usize,
        rgb: &[f32],
    ) -> Result<RgbImage, CpuError> {
        if rgb.len() != 3 * width * height {
            return Err(CpuError::BadLength {
                got: rgb.len(),
                expected: 3 * width * height,
                width,
                height,
            });
        }
        let n = width * height;
        let mut img = RgbImage::new(width, height);
        for i in 0..n {
            for c in 0..3 {
                img.data[c * n + i] = rgb[3 * i + c];
            }
        }
        Ok(img)
    }

    /// Samples per plane.
    #[inline(always)]
    pub fn plane_len(&self) -> usize {
        self.width * self.height
    }

    /// Plane `c` (0 red, 1 green, 2 blue).
    #[inline(always)]
    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.plane_len();
        &self.data[c * n..(c + 1) * n]
    }

    /// Mutable plane `c`.
    #[inline(always)]
    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let n = self.plane_len();
        &mut self.data[c * n..(c + 1) * n]
    }

    /// The three planes at once, mutably.
    pub fn planes_mut(&mut self) -> [&mut [f32]; 3] {
        let n = self.plane_len();
        let (r, rest) = self.data.split_at_mut(n);
        let (g, b) = rest.split_at_mut(n);
        [r, g, b]
    }

    /// The three planes at once.
    pub fn planes(&self) -> [&[f32]; 3] {
        [self.plane(0), self.plane(1), self.plane(2)]
    }

    /// Pixel `(x, y)` as `[r, g, b]`.
    #[inline(always)]
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let n = self.plane_len();
        let i = y * self.width + x;
        [self.data[i], self.data[n + i], self.data[2 * n + i]]
    }

    /// Sets pixel `(x, y)`.
    #[inline(always)]
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [f32; 3]) {
        let n = self.plane_len();
        let i = y * self.width + x;
        self.data[i] = rgb[0];
        self.data[n + i] = rgb[1];
        self.data[2 * n + i] = rgb[2];
    }

    /// The luminance plane with the weights in [`LUMA`].
    pub fn luma_plane(&self) -> Vec<f32> {
        let [r, g, b] = self.planes();
        r.iter()
            .zip(g)
            .zip(b)
            .map(|((&r, &g), &b)| LUMA[0] * r + LUMA[1] * g + LUMA[2] * b)
            .collect()
    }

    /// Interleaved `[r, g, b, ...]` copy of the samples.
    pub fn to_interleaved(&self) -> Vec<f32> {
        let n = self.plane_len();
        let mut out = vec![0.0; 3 * n];
        for i in 0..n {
            for c in 0..3 {
                out[3 * i + c] = self.data[c * n + i];
            }
        }
        out
    }

    /// Cuts out `rect` (buffer coordinates) with the placement adjusted. Panics if outside.
    pub fn extract(&self, rect: Rect) -> RgbImage {
        assert!(
            rect.x1() <= self.width && rect.y1() <= self.height,
            "extract rect outside image"
        );
        let mut out = RgbImage::new(rect.w, rect.h);
        out.place = self.place.offset(rect.x, rect.y);
        for c in 0..3 {
            let src = self.plane(c);
            let dst = out.plane_mut(c);
            for y in 0..rect.h {
                let s = (rect.y + y) * self.width + rect.x;
                dst[y * rect.w..(y + 1) * rect.w].copy_from_slice(&src[s..s + rect.w]);
            }
        }
        out
    }

    /// Copies the sub-rectangle `src_rect` of `tile` (tile coordinates) into this image with its
    /// top-left corner at `(dst_x, dst_y)`.
    pub fn paste(&mut self, tile: &RgbImage, src_rect: Rect, dst_x: usize, dst_y: usize) {
        for c in 0..3 {
            let src = tile.plane(c);
            let width = self.width;
            let dst = self.plane_mut(c);
            for dy in 0..src_rect.h {
                let s = (src_rect.y + dy) * tile.width + src_rect.x;
                let d = (dst_y + dy) * width + dst_x;
                dst[d..d + src_rect.w].copy_from_slice(&src[s..s + src_rect.w]);
            }
        }
    }

    /// Largest absolute difference to `other` over all samples (dimensions must match).
    ///
    /// A NaN is not hidden: a sample that is NaN in one image and not in the other is an infinite
    /// difference (so `diff < tolerance` fails, as it must when one image has a hole the other
    /// has not). A NaN at the same place in both, and equal infinities, count as no difference:
    /// the two images agree there.
    pub fn max_abs_diff(&self, other: &RgbImage) -> f32 {
        assert_eq!(
            (self.width, self.height),
            (other.width, other.height),
            "dimension mismatch"
        );
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| sample_diff(a, b))
            .fold(0.0, f32::max)
    }
}

/// The absolute difference of two samples for [`RgbImage::max_abs_diff`]: 0 for equal values
/// (infinities of one sign and NaN with NaN included), infinity when exactly one is NaN.
fn sample_diff(a: f32, b: f32) -> f32 {
    if a == b || (a.is_nan() && b.is_nan()) {
        0.0
    } else if a.is_nan() || b.is_nan() {
        f32::INFINITY
    } else {
        (a - b).abs()
    }
}

/// Splits `0..n` into at most `parts` contiguous ranges of near-equal size (used by the banded
/// parallel runners). The split depends only on `n` and `parts`, never on the thread count.
pub fn split_range(n: usize, parts: usize) -> Vec<Range<usize>> {
    let parts = parts.clamp(1, n.max(1));
    let base = n / parts;
    let extra = n % parts;
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    for p in 0..parts {
        let len = base + usize::from(p < extra);
        out.push(start..start + len);
        start += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_keeps_parity_and_bounds() {
        for n in [2usize, 3, 5, 8] {
            for i in -20isize..40 {
                let j = mirror(i, n);
                assert!(j < n);
                if n > 1 {
                    assert_eq!(j & 1, (i.rem_euclid(2)) as usize, "parity of {i} in {n}");
                }
            }
        }
        assert_eq!(mirror(-1, 5), 1);
        assert_eq!(mirror(5, 5), 3);
        assert_eq!(mirror(0, 1), 0);
    }

    #[test]
    fn cfa_shift_matches_color_at() {
        for cfa in Cfa::ALL {
            for dx in 0..3 {
                for dy in 0..3 {
                    let s = cfa.shifted(dx, dy);
                    for x in 0..4 {
                        for y in 0..4 {
                            assert_eq!(s.color_at(x, y), cfa.color_at(x + dx, y + dy));
                        }
                    }
                }
            }
        }
        assert_eq!(Cfa::from_name("rggb"), Some(Cfa::RGGB));
        assert_eq!(Cfa::GBRG.name(), "GBRG");
        assert_eq!(Cfa::BGGR.red_row_parity(), 1);
        assert_eq!(Cfa::GRBG.red_row_parity(), 0);
    }

    #[test]
    fn extract_and_paste_round_trip() {
        let mut img = RgbImage::new(7, 5);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = i as f32;
        }
        let r = Rect::new(2, 1, 4, 3);
        let tile = img.extract(r);
        assert_eq!(tile.place.x0, 2);
        assert_eq!(tile.pixel(0, 0), img.pixel(2, 1));
        let mut back = RgbImage::new(7, 5);
        back.paste(&tile, Rect::new(0, 0, 4, 3), 2, 1);
        assert_eq!(back.pixel(5, 3), img.pixel(5, 3));
        assert_eq!(
            RgbImage::from_interleaved(7, 5, &img.to_interleaved()).unwrap(),
            img
        );
    }

    #[test]
    fn max_abs_diff_sees_a_nan_that_the_other_image_does_not_have() {
        let mut a = RgbImage::new(4, 3);
        let mut b = RgbImage::new(4, 3);
        for (i, (x, y)) in a.data.iter_mut().zip(b.data.iter_mut()).enumerate() {
            *x = i as f32 * 0.25;
            *y = *x;
        }
        assert_eq!(a.max_abs_diff(&b), 0.0);
        // Ordinary differences are the largest absolute one.
        b.data[7] += 0.5;
        b.data[9] -= 0.75;
        assert_eq!(a.max_abs_diff(&b), 0.75);
        // A NaN in one image only: infinite, whichever image has it, wherever it is, whatever else differs.
        for at in [0, 7, a.data.len() - 1] {
            let mut with_nan = a.clone();
            with_nan.data[at] = f32::NAN;
            assert_eq!(with_nan.max_abs_diff(&a), f32::INFINITY, "NaN at {at}");
            assert_eq!(a.max_abs_diff(&with_nan), f32::INFINITY, "NaN at {at}");
            assert_eq!(with_nan.max_abs_diff(&b), f32::INFINITY, "NaN at {at}");
        }
        // NaN at different places in the two images: both positions mismatch.
        let (mut c, mut d) = (a.clone(), a.clone());
        c.data[1] = f32::NAN;
        d.data[2] = f32::NAN;
        assert_eq!(c.max_abs_diff(&d), f32::INFINITY);
        // NaN at the same place in both: the images agree there, so only other samples count.
        d.data[2] = a.data[2];
        d.data[1] = f32::NAN;
        assert_eq!(c.max_abs_diff(&d), 0.0);
        d.data[5] += 0.125;
        assert_eq!(c.max_abs_diff(&d), 0.125);
        // The tolerance check callers write (`diff < tolerance`) fails on a NaN mismatch.
        let mut e = a.clone();
        e.data[3] = f32::NAN;
        assert!(e.max_abs_diff(&a) >= 1e-6);
    }

    #[test]
    fn max_abs_diff_treats_infinities_like_numbers() {
        let mut a = RgbImage::new(2, 2);
        let mut b = RgbImage::new(2, 2);
        a.data[0] = f32::INFINITY;
        b.data[0] = f32::INFINITY;
        assert_eq!(a.max_abs_diff(&b), 0.0);
        b.data[0] = f32::NEG_INFINITY;
        assert_eq!(a.max_abs_diff(&b), f32::INFINITY);
        b.data[0] = 1.0;
        assert_eq!(a.max_abs_diff(&b), f32::INFINITY);
    }

    #[test]
    fn split_range_covers_exactly() {
        let parts = split_range(10, 4);
        assert_eq!(parts.iter().map(|r| r.len()).sum::<usize>(), 10);
        assert_eq!(parts[0].start, 0);
        assert_eq!(parts.last().unwrap().end, 10);
    }
}
