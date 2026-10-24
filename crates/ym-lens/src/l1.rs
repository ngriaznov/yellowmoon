//! L1: bokeh reshaping and the bokeh detector that feeds it (`docs/21` section 4, L1, and section 7).
//!
//! L1 never adds bokeh. It reshapes the blur the photo already holds: out-of-focus areas are stretched into
//! taller ovals across the anamorphic axis (Anamorphic) and a little along the tangent (Edges times the lens's
//! swirl), and bright balls are redrawn as ellipses with the energy they had. Pixels in focus are never
//! changed, and where the detector is unsure nothing happens.
//!
//! * [`BokehPlanes`] is the per-photo data, made once by [`LensPlanes::build`](crate::LensPlanes::build) on the
//!   lens plane: the blur map and its confidence, and the ball list. The detector is classical image analysis
//!   (no depth model, no learned model): [`detector`], [`balls`], [`spread`], [`disc`].
//! * [`apply`] is the per-tile pass ([`tile`], [`draw`]); [`halo`] is its reach.
//!
//! With nothing to reshape [`halo`] is 0 and [`apply`] hands back its input.

mod balls;
mod detector;
mod disc;
mod draw;
mod ladder;
mod spread;
mod tile;

use crate::consts::B_CAP;
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::LensPlan;
use crate::planes::{LensPlanes, ReshapeReason, ReshapeReport};

/// A bright ball: a defocused light found on the lens plane (`docs/21` section 4, L1, step 2).
///
/// Lengths are in half-diagonals of the lens plane's frame and positions are fractions of the frame, so the
/// ball is the same at every scale.
#[derive(Clone, Debug, PartialEq)]
pub struct Ball {
    /// Centre as a fraction of the frame, `x` then `y`, in the buffer's orientation.
    pub x: f64,
    /// See [`Ball::x`].
    pub y: f64,
    /// Radius in half-diagonals.
    pub radius: f64,
    /// Radius of the disc that is replaced by the ring's fit before the stretch, in half-diagonals.
    pub take_out_radius: f64,
    /// Per channel, the light above the ring's fit, as a fraction of the frame's area times linear value (so
    /// the sum of the ball's pixels over a view of `n` pixels is `energy * n`).
    pub energy: [f64; 3],
    /// Per channel, the ring's plane fit: the value at the centre and the slopes along `x` and `y` per
    /// half-diagonal.
    pub fit: [[f64; 3]; 3],
    /// The 90th percentile of the ball's luminance.
    pub peak: f64,
    /// True when the edges around the ball agree with its size.
    pub supported: bool,
}

/// One level of the blur map's reductions: the mean of weight times blur, and of weight.
#[derive(Clone, Debug)]
pub(crate) struct MapLevel {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) a: Vec<f32>,
    pub(crate) c: Vec<f32>,
}

/// The per-photo data of L1: the blur map and the ball list (`docs/21` section 3).
#[derive(Clone, Debug)]
pub struct BokehPlanes {
    w: usize,
    h: usize,
    reduce: usize,
    b: Vec<f32>,
    conf: Vec<f32>,
    balls: Vec<Ball>,
    active: bool,
    edges: usize,
    fraction: f64,
    b_max: f64,
    noise_v: f64,
    levels: Vec<MapLevel>,
    /// Summed-area table (`(w + 1) x (h + 1)`) of the cells with a blur.
    nonzero: Vec<u32>,
}

impl BokehPlanes {
    /// Runs the detector on the lens plane.
    pub(crate) fn build(
        _plan: &LensPlan,
        lens_plane: &ImageRef,
        focus: Option<(f64, f64)>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Self, LensError> {
        let d = detector::detect(lens_plane, focus, cancel)?;
        let fraction = detector::blurred_fraction(&d);
        let active = detector::enough(&d);
        let (w, h) = (d.w, d.h);
        let b_max = d
            .b
            .iter()
            .fold(0.0f64, |m, v| m.max(f64::from(*v)))
            .min(B_CAP);
        let balls: Vec<Ball> = d
            .balls
            .iter()
            .map(|b| Ball {
                x: b.cx / w as f64,
                y: b.cy / h as f64,
                radius: b.r / d.hd,
                take_out_radius: b.r_out / d.hd,
                energy: [
                    b.energy[0] / (w * h) as f64,
                    b.energy[1] / (w * h) as f64,
                    b.energy[2] / (w * h) as f64,
                ],
                fit: [
                    [b.fit[0][0], b.fit[0][1] * d.hd, b.fit[0][2] * d.hd],
                    [b.fit[1][0], b.fit[1][1] * d.hd, b.fit[1][2] * d.hd],
                    [b.fit[2][0], b.fit[2][1] * d.hd, b.fit[2][2] * d.hd],
                ],
                peak: b.peak,
                supported: b.supported,
            })
            .collect();
        let mut planes = BokehPlanes {
            w,
            h,
            reduce: d.reduce,
            b: d.b,
            conf: d.conf,
            balls,
            active,
            edges: d.edges,
            fraction,
            b_max,
            noise_v: d.noise_v,
            levels: Vec::new(),
            nonzero: Vec::new(),
        };
        if active {
            planes.build_levels();
        }
        Ok(planes)
    }

    /// The mean-pooled reductions of the blur map for tiles at coarser scales, and the table of blurred cells.
    fn build_levels(&mut self) {
        let (w, h) = (self.w, self.h);
        // Weight of a cell: its confidence where it is out of focus or confidently in focus, else 0.
        let weight = |i: usize| -> f32 {
            let (b, c) = (self.b[i], self.conf[i]);
            if b > 0.0 || f64::from(c) >= 0.3 { c } else { 0.0 }
        };
        let mut a = Vec::with_capacity(w * h);
        let mut c = Vec::with_capacity(w * h);
        for i in 0..w * h {
            let wt = weight(i);
            a.push(wt * self.b[i]);
            c.push(wt);
        }
        let mut levels = vec![MapLevel { w, h, a, c }];
        while levels.len() < 6 {
            let last = levels.last().expect("one level");
            if last.w < 2 || last.h < 2 {
                break;
            }
            let (w2, h2) = (last.w.div_ceil(2), last.h.div_ceil(2));
            let mut a2 = vec![0.0f32; w2 * h2];
            let mut c2 = vec![0.0f32; w2 * h2];
            for y in 0..h2 {
                for x in 0..w2 {
                    let (mut sa, mut sc, mut n) = (0.0f32, 0.0f32, 0.0f32);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (sx, sy) = (2 * x + dx, 2 * y + dy);
                            if sx < last.w && sy < last.h {
                                sa += last.a[sy * last.w + sx];
                                sc += last.c[sy * last.w + sx];
                                n += 1.0;
                            }
                        }
                    }
                    a2[y * w2 + x] = sa / n;
                    c2[y * w2 + x] = sc / n;
                }
            }
            levels.push(MapLevel {
                w: w2,
                h: h2,
                a: a2,
                c: c2,
            });
        }
        self.levels = levels;
        // Summed-area table of the cells with a blur.
        let stride = w + 1;
        let mut sat = vec![0u32; stride * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                row += u32::from(self.b[y * w + x] > 0.0);
                sat[(y + 1) * stride + x + 1] = sat[y * stride + x + 1] + row;
            }
        }
        self.nonzero = sat;
    }

    /// What the reshaping does for this photo.
    pub(crate) fn report(&self) -> ReshapeReport {
        if self.active {
            ReshapeReport {
                reshaped: true,
                reason: None,
            }
        } else {
            ReshapeReport {
                reshaped: false,
                reason: Some(ReshapeReason::LittleBlur),
            }
        }
    }

    /// Size of the maps in cells: the lens plane, or its half when the plane is large.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// How many pixels of the lens plane one cell of the maps covers per side (1 or 2).
    #[must_use]
    pub fn reduce(&self) -> usize {
        self.reduce
    }

    /// The blur radius `b` of every cell, in half-diagonals, row-major; 0 where nothing is reshaped (in
    /// focus, unsure, or below `b_min`).
    #[must_use]
    pub fn blur(&self) -> &[f32] {
        &self.b
    }

    /// The confidence `c` (0 to 1) of every cell's value, `b` or 0.
    #[must_use]
    pub fn confidence(&self) -> &[f32] {
        &self.conf
    }

    /// The balls, brightest first.
    #[must_use]
    pub fn balls(&self) -> &[Ball] {
        &self.balls
    }

    /// True when the photo has enough out-of-focus background to reshape.
    #[must_use]
    pub fn active(&self) -> bool {
        self.active
    }

    /// Edges measured (readings of the blur at edges, plus the perimeters of the balls).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges
    }

    /// The fraction of the frame at `b >= 2 b_min` with confidence.
    #[must_use]
    pub fn blurred_fraction(&self) -> f64 {
        self.fraction
    }

    /// The largest blur of the map, held at `b_cap`, in half-diagonals.
    #[must_use]
    pub fn max_blur(&self) -> f64 {
        self.b_max
    }

    /// Noise of the square-root luminance per cell.
    pub(crate) fn noise_v(&self) -> f64 {
        self.noise_v
    }

    /// Number of reduction levels (0 when the planes are inactive).
    pub(crate) fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Level `i` of the reductions.
    pub(crate) fn level(&self, i: usize) -> &MapLevel {
        &self.levels[i.min(self.levels.len() - 1)]
    }

    /// Cells with a blur in the inclusive cell ranges `[i0, i1] x [j0, j1]` (clamped to the maps).
    pub(crate) fn count_blurred(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> u32 {
        if self.nonzero.is_empty() || i0 > i1 || j0 > j1 || i0 >= self.w || j0 >= self.h {
            return 0;
        }
        let (i1, j1) = (i1.min(self.w - 1), j1.min(self.h - 1));
        let s = self.w + 1;
        let t = &self.nonzero;
        t[(j1 + 1) * s + i1 + 1] + t[j0 * s + i0] - t[j0 * s + i1 + 1] - t[(j1 + 1) * s + i0]
    }

    /// Bytes the planes hold.
    pub(crate) fn bytes(&self) -> usize {
        let levels: usize = self.levels.iter().map(|l| 4 * (l.a.len() + l.c.len())).sum();
        std::mem::size_of::<Self>()
            + 4 * (self.b.len() + self.conf.len() + self.nonzero.len())
            + levels
            + self.balls.len() * std::mem::size_of::<Ball>()
    }
}

/// A picture a stage hands to the next: the caller's input, borrowed, or a new buffer over a rectangle.
pub(crate) enum Stage<'a> {
    /// The stage changed nothing.
    Same(ImageRef<'a>),
    /// The stage's result over `rect` of a `frame` pixels frame.
    New {
        rect: PixelRect,
        frame: (usize, usize),
        data: Vec<f32>,
    },
}

impl Stage<'_> {
    /// The picture.
    pub(crate) fn image(&self) -> ImageRef<'_> {
        match self {
            Stage::Same(i) => *i,
            Stage::New { rect, frame, data } => ImageRef {
                width: rect.w,
                height: rect.h,
                x0: rect.x,
                y0: rect.y,
                frame_width: frame.0,
                frame_height: frame.1,
                data,
            },
        }
    }
}

/// The error for planes built without the blur map that a plan now needs.
fn missing_map() -> LensError {
    LensError::Frame(
        "the planes were built without the blur map; build them again for this lens setting (LensPlanes::serves)"
            .into(),
    )
}

/// The pass of a plan over a frame, or `None` when L1 leaves the picture alone.
fn pass<'a>(
    plan: &'a LensPlan,
    planes: &'a LensPlanes,
    k: usize,
    fw: usize,
    fh: usize,
) -> Result<Option<tile::Pass<'a>>, LensError> {
    if !plan.needs_blur_map() {
        return Ok(None);
    }
    let Some(bp) = planes.bokeh() else {
        return Err(missing_map());
    };
    Ok(tile::Pass::new(plan, bp, k, fw, fh))
}

/// Context L1 needs around a rectangle of a `fw x fh` frame at scale `k`.
pub(crate) fn halo(plan: &LensPlan, planes: &LensPlanes, k: usize, fw: usize, fh: usize) -> usize {
    // A plan whose planes lack the map has no halo here; `apply` reports the mismatch.
    match pass(plan, planes, k, fw, fh) {
        Ok(Some(p)) => p.halo(),
        _ => 0,
    }
}

/// L1 over `want` (L2's receptive field). `input` covers `want` grown by [`halo`].
pub(crate) fn apply<'a>(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef<'a>,
    want: PixelRect,
) -> Result<Stage<'a>, LensError> {
    let (fw, fh) = (input.frame_width, input.frame_height);
    let Some(p) = pass(plan, planes, k, fw, fh)? else {
        return Ok(Stage::Same(*input));
    };
    let data = p.run(input, want)?;
    Ok(Stage::New {
        rect: want,
        frame: (fw, fh),
        data,
    })
}
