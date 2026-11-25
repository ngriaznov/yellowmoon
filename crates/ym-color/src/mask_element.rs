//! The soft masks of the Reference Grade element grades (`docs/34-reference-grade.md` section 4a.3, RG-04).
//!
//! A scene ([`ElementScene`]) says for every pixel of the Input's picture which of the seven elements it belongs to (sky,
//! vegetation, water, ground, built, skin, subject; `ym_ml::scene` makes it, `ym-core` keeps it per photo). At any render size the
//! scene's bits are carried to the output pixel by nearest neighbour through the pixel's place in the photo
//! (`⌊place · size⌋`, which for a frame with no geometry is the centre-mapped `⌊(x + ½) / W · S⌋` of the spec), and each element's
//! binary mask is made soft by a **guided filter** (He, Sun and Tang) whose guide is the **L\*/100 of the Input's picture** at the
//! render's grid ([`crate::display::ColorChain::scene_guide_pixel`]):
//!
//! * the radius is `r = max(1, round(0.02 · the photo's short side))` output pixels ([`element_radius`]), ε = 1e-3 ([`ELEMENT_EPS`]);
//! * the window means are **box means cut at the frame** (a mean over the pixels that exist);
//! * `a = cov / (var + ε)`, `b = mean_p − a · mean_I`, `q = mean(a) · I + mean(b)`, held to [0, 1].
//!
//! # Arithmetic, and why a tile equals the frame
//!
//! The values are `f32` and the **box sums are `f64`**: every box mean is [`crate::mask_raster::box_mean`], the blocked fold of
//! `docs/algorithms/06` section 5.2 (van Herk, Gil and Werman): a window's sum is a suffix scan of one block joined to a prefix scan
//! of the next, with blocks that start at *absolute* multiples of `2r + 1`, divided by the number of samples that exist. The scans
//! run in a fixed order and restart at every block boundary, so a window's value is made from the samples of the window only and in
//! an order that depends on their absolute positions alone: **it does not depend on how the plane was cut**. The filter reads two
//! windows deep (`mean(a)` of an `a` that is itself a window mean), so a tile is made over itself grown by [`element_halo`] =
//! `2 r + 1` pixels, cut to the frame, and equals the whole frame **bit for bit** at any tile size. The rest is per pixel: `f32`
//! products and sums in a fixed order, no fused multiply-add. The result is stored as a u16 code like every raster plane
//! ([`crate::mask_raster::quantise`]), so that the device (RG-08) reads the same codes.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::hue::ELEMENT_COUNT;
use crate::mask::{PhotoFrame, RasterPlanes};
use crate::mask_raster::{Area, PLANE_ONE, PositionAt, box_mean, par_rows, quantise};

/// The guided filter's radius as a share of the photo's short side (`docs/34` section 4a.3, step 2).
pub const ELEMENT_RADIUS_SHARE: f64 = 0.02;
/// The guided filter's regularisation ε, on a guide in [0, 1].
pub const ELEMENT_EPS: f64 = 1e-3;
/// A fit makes the masks of every element that covers at least this share of the picture (`docs/34` section 4a.3, step 5).
pub const FIT_MIN_SHARE: f64 = 0.02;

/// The radius of the guided filter for a render whose photo has `u_px` output pixels along its long edge and the aspect `aspect`
/// (the `PhotoFrame`'s `(W, H) / L`): `max(1, round(0.02 · short side))` output pixels, the short side being `u_px · min(aspect)`.
#[must_use]
pub fn element_radius(u_px: f64, aspect: [f64; 2]) -> usize {
    let short = u_px * aspect[0].min(aspect[1]);
    let r = (ELEMENT_RADIUS_SHARE * short).round();
    if r.is_nan() || r < 1.0 { 1 } else { r as usize }
}

/// The receptive field of the soft masks in output pixels: a tile is made over itself grown by this much. The filter's two
/// nested windows reach `2 r`; one more pixel keeps the box at the tile's edge whole.
#[must_use]
pub fn element_halo(radius: usize) -> usize {
    2 * radius + 1
}

/// What the element masks of a photo are made from: for every pixel of a `width x height` grid over the photo frame, a byte whose
/// bit `e` says that the pixel belongs to element `e` (0 sky, 1 vegetation, 2 water, 3 ground, 4 built, 5 skin, 6 subject).
#[derive(Clone, PartialEq, Eq)]
pub struct ElementScene {
    /// Names the scene: the photo's scene key (the segmenter's input and the rules' version), hashed. Equal keys are equal scenes.
    pub key: u64,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` bytes, rows top to bottom.
    pub bits: Vec<u8>,
}

impl std::fmt::Debug for ElementScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementScene")
            .field("key", &self.key)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl ElementScene {
    /// The radius of the guided filter at a render with `u_px` output pixels along the photo's long edge: [`element_radius`] with
    /// the proportions of the scene's grid, which keeps the photo's up to the rounding of its sides to whole pixels. The scene,
    /// not the render, gives the aspect, so that the schedule (which knows `u_px` only) and the planes agree on the radius, and so
    /// does every tile.
    #[must_use]
    pub fn radius(&self, u_px: f64) -> usize {
        let long = self.width.max(self.height).max(1) as f64;
        element_radius(u_px, [self.width as f64 / long, self.height as f64 / long])
    }

    /// The bits at the place `pos` (fractions of the photo frame): the pixel of the grid that contains it, nearest neighbour.
    #[must_use]
    pub fn bits_at(&self, pos: [f64; 2]) -> u8 {
        if self.width == 0 || self.height == 0 {
            return 0;
        }
        let x = ((pos[0] * self.width as f64).floor().max(0.0) as usize).min(self.width - 1);
        let y = ((pos[1] * self.height as f64).floor().max(0.0) as usize).min(self.height - 1);
        self.bits[y * self.width + x]
    }

    /// The share of the grid's pixels that belong to element `e`.
    #[must_use]
    pub fn share(&self, e: usize) -> f64 {
        if self.bits.is_empty() || e >= ELEMENT_COUNT {
            return 0.0;
        }
        self.bits.iter().filter(|&&b| b >> e & 1 != 0).count() as f64 / self.bits.len() as f64
    }

    /// Which elements cover at least `min_share` of the picture and at least one pixel.
    #[must_use]
    pub fn covering(&self, min_share: f64) -> [bool; ELEMENT_COUNT] {
        let mut out = [false; ELEMENT_COUNT];
        for (e, o) in out.iter_mut().enumerate() {
            let s = self.share(e);
            *o = s > 0.0 && s >= min_share;
        }
        out
    }
}

/// The live elements of one Reference Grade node of the plan, with the scene they are made from.
#[derive(Debug, Clone)]
pub struct ElementRow {
    /// The node's id.
    pub node: u32,
    /// Which elements are live (their effective grade is not the identity): only these make a mask.
    pub live: [bool; ELEMENT_COUNT],
    /// The photo's scene.
    pub scene: Arc<ElementScene>,
}

/// What a render knows for the element planes of one window.
pub struct ElementJob<'a> {
    /// The photo's aspect and the size of one output pixel in units of its long edge.
    pub photo: PhotoFrame,
    /// The window the planes are computed over: the tile grown by [`element_halo`], cut to the frame.
    pub grown: Area,
    /// The whole output image: window sums are clipped to it.
    pub frame: Area,
    /// The part of `grown` the planes are returned for (the tile).
    pub window: Area,
    /// Where each pixel of `grown` is in the photo (`None` outside the photo: no element there).
    pub position: PositionAt<'a>,
    /// The guide: L\*/100 of the Input's picture at every pixel of `grown`, row-major, in [0, 1].
    pub guide: &'a [f32],
}

impl ElementJob<'_> {
    /// `u_px`: the output pixels along the photo's long edge.
    #[must_use]
    pub fn u_px(&self) -> f64 {
        1.0 / self.photo.pixel
    }
}

/// An `f32` rounded value held in an `f64` (exactly).
fn widen(v: f32) -> f64 {
    f64::from(v)
}

/// A window mean of `src` as `f32`: the box sums are `f64` ([`box_mean`]), the mean is rounded once.
fn mean32(src: &[f64], g: &Area, frame: &Area, r: usize) -> Vec<f32> {
    box_mean(src, g, frame, r)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// The guide's own statistics for a window and a radius, shared by every element: its mean and the mean of its square.
struct GuideStats {
    mean: Vec<f32>,
    mean_sq: Vec<f32>,
}

impl GuideStats {
    fn new(guide: &[f32], g: &Area, frame: &Area, r: usize) -> Self {
        let wide: Vec<f64> = guide.iter().map(|&v| widen(v)).collect();
        let sq: Vec<f64> = guide.iter().map(|&v| widen(v * v)).collect();
        Self {
            mean: mean32(&wide, g, frame, r),
            mean_sq: mean32(&sq, g, frame, r),
        }
    }
}

/// The guided filter of the binary mask `p` (0 or 1 per pixel of `g`) with `guide`, held to [0, 1]: the section's formulas in `f32`
/// with `f64` box sums, in one fixed order.
fn guided_f32(
    p: &[f32],
    guide: &[f32],
    stats: &GuideStats,
    g: &Area,
    frame: &Area,
    r: usize,
) -> Vec<f32> {
    let eps = ELEMENT_EPS as f32;
    let wide_p: Vec<f64> = p.iter().map(|&v| widen(v)).collect();
    let pg: Vec<f64> = p.iter().zip(guide).map(|(&a, &b)| widen(a * b)).collect();
    let mean_p = mean32(&wide_p, g, frame, r);
    let mean_pg = mean32(&pg, g, frame, r);
    let n = p.len();
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    par_rows(&mut a, g.w.max(1), |y, row| {
        for (x, d) in row.iter_mut().enumerate() {
            let i = y * g.w + x;
            let var = stats.mean_sq[i] - stats.mean[i] * stats.mean[i];
            let cov = mean_pg[i] - stats.mean[i] * mean_p[i];
            *d = widen(cov / (var + eps));
        }
    });
    par_rows(&mut b, g.w.max(1), |y, row| {
        for (x, d) in row.iter_mut().enumerate() {
            let i = y * g.w + x;
            let ai = a[i] as f32;
            *d = widen(mean_p[i] - ai * stats.mean[i]);
        }
    });
    let abar = mean32(&a, g, frame, r);
    let bbar = mean32(&b, g, frame, r);
    (0..n)
        .map(|i| {
            let q = abar[i] * guide[i] + bbar[i];
            if q.is_nan() { 0.0 } else { q.clamp(0.0, 1.0) }
        })
        .collect()
}

/// The soft masks of `wanted` elements over `job.window` as `f32` in [0, 1] (row-major, the window's size), `None` for an element
/// that is not wanted or has no pixel in the area the filter looks at (its mask is 0 there exactly: with an empty binary mask the
/// filter's `a` and `b` are 0).
///
/// # Panics
///
/// When `job.window` is not inside `job.grown`, or the guide is not the size of `job.grown`: a mistake of the caller.
#[must_use]
pub fn soft_masks(
    scene: &ElementScene,
    wanted: &[bool; ELEMENT_COUNT],
    job: &ElementJob<'_>,
) -> [Option<Vec<f32>>; ELEMENT_COUNT] {
    assert!(
        job.grown.contains(&job.window),
        "the window of an element plane is inside the area it is computed over"
    );
    let g = job.grown;
    assert_eq!(
        job.guide.len(),
        g.w * g.h,
        "one guide value for every pixel"
    );
    let mut out: [Option<Vec<f32>>; ELEMENT_COUNT] = Default::default();
    if !wanted.iter().any(|&w| w) || g.is_empty() {
        return out;
    }
    // The scene's bits at every pixel of the area (0 outside the photo).
    let mut bits = vec![0u8; g.w * g.h];
    for y in 0..g.h {
        for x in 0..g.w {
            if let Some(pos) = (job.position)(x, y) {
                bits[y * g.w + x] = scene.bits_at(pos);
            }
        }
    }
    let r = scene.radius(job.u_px());
    let mut stats: Option<GuideStats> = None;
    for e in 0..ELEMENT_COUNT {
        if !wanted[e] || !bits.iter().any(|&b| b >> e & 1 != 0) {
            continue;
        }
        let stats = stats.get_or_insert_with(|| GuideStats::new(job.guide, &g, &job.frame, r));
        let p: Vec<f32> = bits
            .iter()
            .map(|&b| if b >> e & 1 != 0 { 1.0 } else { 0.0 })
            .collect();
        let q = guided_f32(&p, job.guide, stats, &g, &job.frame, r);
        let w = job.window;
        let mut cut = Vec::with_capacity(w.w * w.h);
        for y in 0..w.h {
            let row = (w.y - g.y) as usize + y;
            let col = (w.x - g.x) as usize;
            cut.extend_from_slice(&q[row * g.w + col..row * g.w + col + w.w]);
        }
        out[e] = Some(cut);
    }
    out
}

/// The soft masks of a whole picture (`width x height`, a fit's analysis picture), for the elements in `wanted`: `guide` is the
/// L\*/100 of the picture, row-major. Each plane is `f32` in [0, 1] over the picture; `None` for an element that is not wanted or has
/// no pixel. The radius follows the picture's short side. This is the call of the fit (RG-06); at the sizes of a render the same
/// function is [`soft_masks`] over a window.
///
/// # Panics
///
/// When `guide` is not `width * height` values.
#[must_use]
pub fn soft_masks_of_picture(
    scene: &ElementScene,
    wanted: &[bool; ELEMENT_COUNT],
    guide: &[f32],
    width: usize,
    height: usize,
) -> [Option<Vec<f32>>; ELEMENT_COUNT] {
    assert_eq!(guide.len(), width * height);
    let whole = Area::whole(width, height);
    let position = move |x: usize, y: usize| {
        Some([
            (x as f64 + 0.5) / width as f64,
            (y as f64 + 0.5) / height as f64,
        ])
    };
    let job = ElementJob {
        photo: PhotoFrame::new(width as f64, height as f64, 1.0),
        grown: whole,
        frame: whole,
        window: whole,
        position: &position,
        guide,
    };
    soft_masks(scene, wanted, &job)
}

/// The element planes of a window for the Reference Grade nodes of a chain: for every node with a live element, a u16 plane
/// (row-major, the window's size) for each live element that has pixels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ElementPlanes {
    /// The window's width.
    pub width: usize,
    /// The window's height.
    pub height: usize,
    /// For every node, the planes of its seven elements; an empty plane is 0 everywhere.
    pub nodes: BTreeMap<u32, [Vec<u16>; ELEMENT_COUNT]>,
}

impl ElementPlanes {
    /// Whether no node has a plane.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// These planes (of a window) placed in an `extent_w x extent_h` extent with the window's first pixel at `at`, 0 around them.
    ///
    /// # Panics
    ///
    /// When the window does not fit in the extent at `at`: a mistake of the caller.
    #[must_use]
    pub fn placed(&self, at: (usize, usize), extent_w: usize, extent_h: usize) -> ElementPlanes {
        assert!(
            at.0 + self.width <= extent_w && at.1 + self.height <= extent_h,
            "a window of planes lies inside the extent they are placed in"
        );
        if (self.width, self.height) == (extent_w, extent_h) {
            return self.clone();
        }
        let nodes = self
            .nodes
            .iter()
            .map(|(&id, planes)| {
                let placed: [Vec<u16>; ELEMENT_COUNT] = std::array::from_fn(|e| {
                    if planes[e].is_empty() {
                        return Vec::new();
                    }
                    let mut out = vec![0u16; extent_w * extent_h];
                    for y in 0..self.height {
                        let dst = (at.1 + y) * extent_w + at.0;
                        out[dst..dst + self.width]
                            .copy_from_slice(&planes[e][y * self.width..(y + 1) * self.width]);
                    }
                    out
                });
                (id, placed)
            })
            .collect();
        ElementPlanes {
            width: extent_w,
            height: extent_h,
            nodes,
        }
    }

    /// The value in [0, 1] of element `e` of node `node` at `(x, y)` of the window: 0 for a plane that is not there.
    #[must_use]
    pub fn value(&self, node: u32, e: usize, x: usize, y: usize) -> f64 {
        match self.nodes.get(&node).and_then(|p| p.get(e)) {
            Some(p) if !p.is_empty() && x < self.width && y < self.height => {
                f64::from(p[y * self.width + x]) / PLANE_ONE
            }
            _ => 0.0,
        }
    }
}

impl RasterPlanes for ElementPlanes {
    fn value(&self, _plane: usize, _x: usize, _y: usize) -> f64 {
        0.0
    }

    fn element(&self, node: u32, element: usize, x: usize, y: usize) -> f64 {
        ElementPlanes::value(self, node, element, x, y)
    }
}

/// The planes of every row over `job.window`, quantised to u16 codes. A row with no live element is left out.
#[must_use]
pub fn element_planes(rows: &[ElementRow], job: &ElementJob<'_>) -> ElementPlanes {
    let mut planes = ElementPlanes {
        width: job.window.w,
        height: job.window.h,
        nodes: BTreeMap::new(),
    };
    for row in rows {
        if !row.live.iter().any(|&l| l) {
            continue;
        }
        let soft = soft_masks(&row.scene, &row.live, job);
        let codes: [Vec<u16>; ELEMENT_COUNT] = std::array::from_fn(|e| {
            soft[e].as_ref().map_or_else(Vec::new, |q| {
                q.iter().map(|&v| quantise(f64::from(v))).collect()
            })
        });
        // A node whose live elements have no pixel anywhere in view makes no plane at all.
        planes.nodes.insert(row.node, codes);
    }
    planes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference guided filter, slow and plain: every window summed directly in ascending order in `f64`, every step in `f64`.
    fn slow_guided(p: &[f64], guide: &[f64], w: usize, h: usize, r: usize, eps: f64) -> Vec<f64> {
        let mean = |src: &dyn Fn(usize) -> f64| -> Vec<f64> {
            let mut out = vec![0.0; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
                    let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
                    let mut s = 0.0;
                    let mut n = 0.0;
                    for yy in y0..=y1 {
                        for xx in x0..=x1 {
                            s += src(yy * w + xx);
                            n += 1.0;
                        }
                    }
                    out[y * w + x] = s / n;
                }
            }
            out
        };
        let mi = mean(&|i| guide[i]);
        let mp = mean(&|i| p[i]);
        let mii = mean(&|i| guide[i] * guide[i]);
        let mip = mean(&|i| guide[i] * p[i]);
        let a: Vec<f64> = (0..w * h)
            .map(|i| (mip[i] - mi[i] * mp[i]) / (mii[i] - mi[i] * mi[i] + eps))
            .collect();
        let b: Vec<f64> = (0..w * h).map(|i| mp[i] - a[i] * mi[i]).collect();
        let abar = mean(&|i| a[i]);
        let bbar = mean(&|i| b[i]);
        (0..w * h)
            .map(|i| (abar[i] * guide[i] + bbar[i]).clamp(0.0, 1.0))
            .collect()
    }

    /// A seeded picture: smooth ramps and some noise, as an L*/100 guide.
    fn guide_of(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64 / w as f64, (i / w) as f64 / h as f64);
                let edge = if x + 0.3 * y > 0.6 { 0.35 } else { 0.0 };
                (0.2 + 0.4 * x + edge + 0.03 * next()).clamp(0.0, 1.0) as f32
            })
            .collect()
    }

    fn scene_of(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> ElementScene {
        ElementScene {
            key: 1,
            width: w,
            height: h,
            bits: (0..w * h).map(|i| f(i % w, i / w)).collect(),
        }
    }

    fn whole_job<'a>(
        w: usize,
        h: usize,
        guide: &'a [f32],
        position: PositionAt<'a>,
    ) -> ElementJob<'a> {
        let whole = Area::whole(w, h);
        ElementJob {
            photo: PhotoFrame::new(w as f64, h as f64, 1.0),
            grown: whole,
            frame: whole,
            window: whole,
            position,
            guide,
        }
    }

    #[test]
    fn the_radius_is_two_percent_of_the_short_side_and_at_least_one() {
        let aspect = |w: f64, h: f64| [w / w.max(h), h / w.max(h)];
        assert_eq!(element_radius(1024.0, aspect(1024.0, 683.0)), 14); // 0.02 * 683 = 13.7
        assert_eq!(element_radius(6000.0, aspect(6000.0, 4000.0)), 80);
        assert_eq!(element_radius(160.0, aspect(160.0, 107.0)), 2);
        assert_eq!(
            element_radius(40.0, aspect(40.0, 30.0)),
            1,
            "a tiny picture still has a window"
        );
        assert_eq!(element_radius(0.0, [1.0, 1.0]), 1);
        assert_eq!(element_halo(14), 29);
    }

    #[test]
    fn the_scene_is_read_by_nearest_neighbour_through_the_place() {
        let s = scene_of(4, 2, |x, y| (x + 4 * y) as u8);
        assert_eq!(s.bits_at([0.0, 0.0]), 0);
        assert_eq!(s.bits_at([0.26, 0.0]), 1);
        assert_eq!(s.bits_at([0.99, 0.99]), 7);
        assert_eq!(s.bits_at([1.0, 1.0]), 7, "held to the grid");
        assert_eq!(s.bits_at([-0.2, 0.6]), 4);
        // The centre-mapped rule of the spec: output pixel x of a W wide frame takes source index floor((x + 1/2) / W * S).
        let (wide, src) = (10usize, 4usize);
        for x in 0..wide {
            let pos = (x as f64 + 0.5) / wide as f64;
            assert_eq!(
                (pos * src as f64).floor() as usize,
                ((x as f64 + 0.5) / wide as f64 * src as f64).floor() as usize
            );
        }
        assert!((s.share(0) - 0.5).abs() < 1e-12);
    }

    /// The guided filter against the slow reference on a picture with an edge in the guide: the bound (1e-5; measured 1.4e-6 and 2.2e-7)
    /// is the f32 arithmetic's.
    #[test]
    fn the_guided_filter_agrees_with_a_slow_reference_within_the_f32_bound() {
        let (w, h) = (73, 59);
        let guide = guide_of(w, h, 3);
        let scene = scene_of(w, h, |x, y| {
            u8::from(x + y > 70 || (20..30).contains(&x) && y < 25) << 2
        });
        let pos =
            |x: usize, y: usize| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64]);
        let job = whole_job(w, h, &guide, &pos);
        let r = scene.radius(job.u_px());
        assert_eq!(r, 1, "the radius of a 59 px picture");
        // Use a larger radius too, through a bigger picture of the same content.
        let mut wanted = [false; ELEMENT_COUNT];
        wanted[2] = true;
        let got = soft_masks(&scene, &wanted, &job)[2]
            .clone()
            .expect("a plane");
        let p: Vec<f64> = scene.bits.iter().map(|&b| f64::from(b >> 2 & 1)).collect();
        let g64: Vec<f64> = guide.iter().map(|&v| f64::from(v)).collect();
        let want = slow_guided(&p, &g64, w, h, r, ELEMENT_EPS);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        println!("the slow reference, r = {r}: worst difference {worst:e}");
        assert!(worst < 1e-5, "worst {worst}");
        // A bigger radius (a 400 x 300 picture: r = 6) against the reference.
        let (w, h) = (400, 300);
        let guide = guide_of(w, h, 5);
        let scene = scene_of(w, h, |x, y| {
            u8::from((x as f64 - 200.0).hypot(y as f64 - 150.0) < 90.0) << 2
        });
        let pos =
            |x: usize, y: usize| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64]);
        let job = whole_job(w, h, &guide, &pos);
        let r = scene.radius(job.u_px());
        assert_eq!(r, 6);
        let got = soft_masks(&scene, &wanted, &job)[2]
            .clone()
            .expect("a plane");
        let p: Vec<f64> = scene.bits.iter().map(|&b| f64::from(b >> 2 & 1)).collect();
        let g64: Vec<f64> = guide.iter().map(|&v| f64::from(v)).collect();
        let want = slow_guided(&p, &g64, w, h, r, ELEMENT_EPS);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        println!("the slow reference, r = {r}: worst difference {worst:e}");
        assert!(worst < 1e-5, "worst {worst} at r = {r}");
    }

    #[test]
    fn known_inputs_a_step_edge_a_flat_guide_and_an_empty_mask() {
        let (w, h) = (128, 128);
        let pos =
            |x: usize, y: usize| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64]);
        let mut wanted = [false; ELEMENT_COUNT];
        wanted[0] = true;
        // A mask with a vertical step edge at x = 64 and a guide with the same step: the guide pulls the mask's edge onto itself, so
        // the soft mask stays a step (0 well left of the edge, 1 well right of it, within a code).
        let scene = scene_of(w, h, |x, _| u8::from(x >= 64));
        let guide: Vec<f32> = (0..w * h)
            .map(|i| if i % w >= 64 { 0.8 } else { 0.2 })
            .collect();
        let job = whole_job(w, h, &guide, &pos);
        let q = soft_masks(&scene, &wanted, &job)[0]
            .clone()
            .expect("a plane");
        assert!(q[10 * w + 10] < 1e-6 && q[10 * w + 120] > 1.0 - 1e-6);
        assert!(
            q[10 * w + 63] < 0.05 && q[10 * w + 64] > 0.95,
            "{} {}",
            q[10 * w + 63],
            q[10 * w + 64]
        );
        // The same mask with a flat guide: there is no covariance, so a is 0 and q is the mean of the window means of the mask: the
        // edge becomes a ramp of 4 r + 1 pixels (two boxes), symmetric about the edge (the two pixels beside it sum to 1) and 0 and 1
        // beyond it.
        let flat = vec![0.5f32; w * h];
        let job = whole_job(w, h, &flat, &pos);
        let r = scene.radius(job.u_px());
        let q = soft_masks(&scene, &wanted, &job)[0]
            .clone()
            .expect("a plane");
        assert!(q[10 * w + 63] < 0.5 && q[10 * w + 64] > 0.5);
        assert!((q[10 * w + 63] + q[10 * w + 64] - 1.0).abs() < 1e-3);
        assert!(q[10 * w + 64 - 2 * r - 1] < 1e-4 && q[10 * w + 64 + 2 * r] > 1.0 - 1e-4);
        // An empty mask is 0 everywhere, and has no plane at all.
        let empty = scene_of(w, h, |_, _| 0);
        let job = whole_job(w, h, &guide, &pos);
        assert!(
            soft_masks(&empty, &wanted, &job)
                .iter()
                .all(Option::is_none)
        );
        // An element that is not wanted makes no plane; a full mask is 1 everywhere.
        let full = scene_of(w, h, |_, _| 1);
        let q = soft_masks(&full, &wanted, &job);
        assert!(
            q[0].as_ref()
                .expect("a plane")
                .iter()
                .all(|&v| v > 1.0 - 1e-6)
        );
        assert!(q[1..].iter().all(Option::is_none));
        // The values are held to [0, 1].
        let noisy = guide_of(w, h, 11);
        let job = whole_job(w, h, &noisy, &pos);
        let ring = scene_of(w, h, |x, y| u8::from((x / 3 + y / 5) % 2 == 0));
        let q = soft_masks(&ring, &wanted, &job)[0]
            .clone()
            .expect("a plane");
        assert!(q.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    /// The planes of a tile (the tile grown by the halo, cut to the frame) equal the planes of the whole frame bit for bit, for tile
    /// sizes 97, 160 and 256 and for the bits' own tile edges that cut a block of the fold.
    #[test]
    fn a_tile_equals_the_whole_frame_bit_for_bit_at_any_tile_size() {
        let (w, h) = (520, 390);
        let guide = guide_of(w, h, 21);
        let scene = scene_of(w, h, |x, y| {
            let blob = (x as f64 - 260.0).hypot(y as f64 - 200.0) < 120.0;
            u8::from((x + 2 * y) % 97 < 40) | (u8::from(blob) << 2) | (u8::from(x > 300) << 6)
        });
        let pos =
            |x: usize, y: usize| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64]);
        let frame = Area::whole(w, h);
        let photo = PhotoFrame::new(w as f64, h as f64, 1.0);
        let r = element_radius(1.0 / photo.pixel, photo.aspect);
        assert_eq!(r, 8);
        assert_eq!(scene.radius(1.0 / photo.pixel), 8);
        let rows = [ElementRow {
            node: 5,
            live: [true, false, true, false, false, false, true],
            scene: Arc::new(scene),
        }];
        let whole = {
            let job = ElementJob {
                photo,
                grown: frame,
                frame,
                window: frame,
                position: &pos,
                guide: &guide,
            };
            element_planes(&rows, &job)
        };
        assert!(
            whole.nodes[&5][0].iter().any(|&c| c > 0 && c < 65535),
            "a soft edge exists"
        );
        for tile in [97usize, 160, 256] {
            let mut stitched: [Vec<u16>; ELEMENT_COUNT] =
                std::array::from_fn(|_| vec![0u16; w * h]);
            for ty in (0..h).step_by(tile) {
                for tx in (0..w).step_by(tile) {
                    let window = Area {
                        x: tx as i64,
                        y: ty as i64,
                        w: tile.min(w - tx),
                        h: tile.min(h - ty),
                    };
                    let grown = window.grown(element_halo(r), &frame);
                    let g: Vec<f32> = (0..grown.h)
                        .flat_map(|y| {
                            let row = (grown.y as usize + y) * w + grown.x as usize;
                            guide[row..row + grown.w].iter().copied()
                        })
                        .collect();
                    let off = (grown.x as usize, grown.y as usize);
                    let tile_pos = move |x: usize, y: usize| {
                        Some([
                            ((x + off.0) as f64 + 0.5) / w as f64,
                            ((y + off.1) as f64 + 0.5) / h as f64,
                        ])
                    };
                    let job = ElementJob {
                        photo,
                        grown,
                        frame,
                        window,
                        position: &tile_pos,
                        guide: &g,
                    };
                    let planes = element_planes(&rows, &job);
                    for (dst, src) in stitched.iter_mut().zip(&planes.nodes[&5]) {
                        if src.is_empty() {
                            continue;
                        }
                        for y in 0..window.h {
                            for x in 0..window.w {
                                dst[(ty + y) * w + tx + x] = src[y * window.w + x];
                            }
                        }
                    }
                }
            }
            for e in [0usize, 2, 6] {
                assert!(
                    stitched[e] == whole.nodes[&5][e],
                    "tile {tile}: element {e} differs from the whole frame"
                );
            }
            // An element that is not live has no plane at either size.
            assert!(whole.nodes[&5][1].is_empty());
        }
        // A halo one box short differs: the test sees what it guards.
        let window = Area {
            x: 160,
            y: 130,
            w: 160,
            h: 130,
        };
        let short = window.grown(element_halo(r) - 2 * r, &frame);
        let g: Vec<f32> = (0..short.h)
            .flat_map(|y| {
                let row = (short.y as usize + y) * w + short.x as usize;
                guide[row..row + short.w].iter().copied()
            })
            .collect();
        let off = (short.x as usize, short.y as usize);
        let tile_pos = move |x: usize, y: usize| {
            Some([
                ((x + off.0) as f64 + 0.5) / w as f64,
                ((y + off.1) as f64 + 0.5) / h as f64,
            ])
        };
        let job = ElementJob {
            photo,
            grown: short,
            frame,
            window,
            position: &tile_pos,
            guide: &g,
        };
        let planes = element_planes(&rows, &job);
        let mut differs = false;
        for y in 0..window.h {
            for x in 0..window.w {
                differs |= planes.nodes[&5][0][y * window.w + x]
                    != whole.nodes[&5][0][(window.y as usize + y) * w + window.x as usize + x];
            }
        }
        assert!(differs, "a halo of one pixel should not be enough");
    }

    #[test]
    fn the_same_bits_on_two_runs_and_the_planes_read_back_as_codes() {
        let (w, h) = (96, 64);
        let guide = guide_of(w, h, 8);
        let scene = Arc::new(scene_of(w, h, |x, _| u8::from(x > 40) << 3));
        let pos =
            |x: usize, y: usize| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64]);
        let job = whole_job(w, h, &guide, &pos);
        let rows = [ElementRow {
            node: 2,
            live: [false, false, false, true, false, false, false],
            scene,
        }];
        let a = element_planes(&rows, &job);
        let b = element_planes(&rows, &job);
        assert_eq!(a, b);
        let v = RasterPlanes::element(&a, 2, 3, 80, 10);
        assert!(v > 0.9);
        assert_eq!(RasterPlanes::element(&a, 2, 3, 2, 10), 0.0);
        assert_eq!(
            RasterPlanes::element(&a, 2, 0, 80, 10),
            0.0,
            "an element that is not live"
        );
        assert_eq!(RasterPlanes::element(&a, 9, 3, 80, 10), 0.0, "another node");
        // Placed in a larger extent, with 0 around.
        let placed = a.placed((5, 7), w + 10, h + 20);
        assert_eq!(placed.value(2, 3, 85, 17), v);
        assert_eq!(placed.value(2, 3, 2, 2), 0.0);
        // The whole-picture call of the fit gives the f32 planes the quantised ones come from.
        let mut wanted = [false; ELEMENT_COUNT];
        wanted[3] = true;
        let fit = soft_masks_of_picture(&rows[0].scene, &wanted, &guide, w, h);
        let plane = fit[3].as_ref().expect("a plane");
        assert!(
            plane
                .iter()
                .zip(&a.nodes[&2][3])
                .all(|(&f, &c)| quantise(f64::from(f)) == c)
        );
    }
}
