//! Bright balls: out-of-focus lights, found and measured by their size (`docs/21` section 4, L1 step 2, and
//! section 7, step 4).
//!
//! A defocused light is a bright blob with a rim: sharp-edged on a good lens, soft-edged on most, sometimes
//! clipped, sometimes dim next to the colour of its ground, sometimes overlapping a neighbour. The edge measure
//! of the blur map reads a sharp rim as sharp, so the balls are found directly, and are what gives the
//! anamorphic look its strongest cue (oval highlights).
//!
//! 1. **Candidates.** A scale-space search for bright blobs: differences of Gaussians of the square root of the
//!    brightest channel (the colour of a light counts, not only its luminance), at half-octave steps from 1 to
//!    22.6 px, scale-normalised, maximum in space and scale.
//! 2. **Measure.** Each candidate is measured on the linear plane: the plateau's level `T` (75th percentile
//!    within half the seed radius), the ground `B` (median of a ring around), and the region above
//!    `B + 0.4 (T - B)` (flood fill from the centre). Its area gives the radius `r`, its second moments the
//!    elongation and how well it fills its ellipse. The centre and radius are measured again from the region.
//! 3. **Shape tests.** A ball is a compact blob: at least 0.12 brighter than its ground in square-root units
//!    (about a fifth of a stop at mid grey, but the colour counts) and 1.25 times brighter in light, not
//!    elongated beyond 1.6 (cat's-eye lemons, overlaps), filling at least 80 % of its moment ellipse, not
//!    leaking out of the search window (a texture, not a disc), and between the smallest and largest radius. A
//!    plateau that is not flat does not matter: soft balls, balls with a bright rim, balls with a dim side are
//!    all balls.
//! 4. **Context.** The edges around a ball (their weight and mean log radius come in as `ev_w` and `ev_s`, from
//!    the classical readings or from any other blur map) say whether the ground is out of focus: a ball whose
//!    ground is much sharper than the ball is a lamp in focus and is dropped; a ball over a ground blurred
//!    about as much (within 1.8 times smaller to 3 times larger) is supported. A ball with no edges around
//!    (flat ground) counts only when it is big enough not to be a point light in focus and the scene has at
//!    least three such balls: a lone round light (the moon, a lamp) is more likely sharp than blurred.
//! 5. **Order and cap.** Brightest first, of two closer than 1.2 radii the dimmer is dropped, at most
//!    [`MAX_BALLS`].
//!
//! Then each ball gets the plane fit of its ring, the energy above it, and the radius of the disc that is
//! taken out: 1.15 `r` plus a margin, or, for a small ball, where its own light has died away into its ground (the
//! light that stays in the picture is stretched with the rest of the blur and would stand beside the redrawn oval
//! as a pillar).

use crate::consts::MAX_BALLS;
use crate::error::LensError;

use super::ladder::{blur, blur_wide};

/// A ball on the detector's working grid, in pixels of that grid.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct BallPx {
    /// Centre, continuous pixel coordinates (pixel `i` covers `[i, i + 1)`).
    pub cx: f64,
    pub cy: f64,
    /// Radius of the region above 40 % of the way from the ground to the plateau.
    pub r: f64,
    /// Radius of the disc that is taken out (1.15 `r` plus a margin, or the soft tail's reach).
    pub r_out: f64,
    /// Per channel, the light above the ring's plane fit, summed over the disc.
    pub energy: [f64; 3],
    /// Per channel, the ring's plane fit `a + b dx + c dy` (`dx`, `dy` in pixels from the centre).
    pub fit: [[f64; 3]; 3],
    /// The 90th percentile of the disc's luminance.
    pub peak: f64,
    /// True when the edges around the ball agree with its size.
    pub supported: bool,
    /// Width of the rim (80 % to 20 % of the way down) over the radius.
    pub softness: f64,
    /// True when the ball passes the strict rules (a ring with at most a twentieth of it bright, a rim no wider than
    /// 0.9 radii, no sharp ground around it): only such balls count towards "the scene is one of lights".
    pub clear: bool,
}

/// Everything the search reads.
pub(super) struct BallInput<'a> {
    pub w: usize,
    pub h: usize,
    /// The working grid's red, green and blue planes (linear).
    pub rgb: [&'a [f32]; 3],
    /// Its luminance (linear).
    pub y: &'a [f32],
    /// Noise of the square-root luminance, per pixel.
    pub noise_v: f64,
    /// Largest ball radius, in pixels.
    pub r_max_px: f64,
    /// The blur evidence around every pixel: its weight and weight times log radius (in pixels of this grid).
    /// The classical detector's sparse edge readings are one source; a dense blur map with a confidence
    /// (`ev_w = confidence`, `ev_s = confidence * ln(radius)`) is another, and both read the same way.
    pub ev_w: &'a [f32],
    pub ev_s: &'a [f32],
    /// Pixels of the subject anchor (the camera's focus point): no ball starts there.
    pub anchor: &'a [bool],
    /// Pixels that read sharp: at a strong sharp edge (the classical detector's own flags), or where a dense
    /// blur map says in focus.
    pub sharp: &'a [bool],
    /// The most a ball's ring may hold of those pixels, as a share of the ring: [`MAX_SHARP_DENSITY`] for the
    /// thin lines of edges, much more for the area of a dense map.
    pub max_sharp_density: f64,
    /// True when `sharp` is the area of a dense blur map (a sharp subject on one side of a ball is then no reason
    /// to drop it: the median over the ring's sectors counts); false for the classical detector's thin edge flags
    /// (a single sharp edge in the ring of a light is a gap in leaves, not a subject: the whole ring counts).
    pub sector_rule: bool,
    /// The pixels a learned blur map calls blurred, when there is one (the classical detector has none). A ball with
    /// a wide rim is a smooth blob, and what tells a defocused light from a smooth bright thing in focus (a bud, a
    /// cloud, white fur) is that the map reads the blob itself as blurred ([`SOFT_BLURRED_SHARE`]).
    pub blurred: Option<&'a [bool]>,
}

/// Margin of the take-out disc beyond 1.15 `r`, in working-grid pixels.
pub(super) const TAKE_OUT_MARGIN_PX: f64 = 1.5;
/// The take-out disc reaches at least 1.15 `r`, at most this many radii (plus the margin), and where the ball's
/// own light has died away in between.
const TAKE_OUT_MIN_RADII: f64 = 1.15;
const TAKE_OUT_MAX_RADII: f64 = 2.0;
/// The ball's light has died away where the median of what is above the ground, in a ring half a pixel wide, is
/// within the allowance: the smaller of [`TAIL_FRACTION`] of the ball's height above the ground and
/// [`TAIL_OF_GROUND`] times the ground (a very bright ball over a dark ground must not leave a tail as bright as
/// the ground), but at least [`TAIL_NOISE_UNITS`] noise units of the ground (the plane's noise must not hold the
/// disc open).
const TAIL_FRACTION: f64 = 0.03;
const TAIL_OF_GROUND: f64 = 0.15;
const TAIL_NOISE_UNITS: f64 = 3.0;
/// Standard deviations (px) of the scale-space levels: half-octave steps.
const SCALES: [f64; 10] = [
    1.0, 1.414, 2.0, 2.828, 4.0, 5.657, 8.0, 11.314, 16.0, 22.627,
];
/// The seed radius of a candidate found at scale `s` is this times `s`.
const SEED_PER_SCALE: f64 = 1.7;
/// A response of the differences of Gaussians (square-root units) below this is no blob.
const RESPONSE_MIN: f64 = 0.012;
/// ... nor is a response below this many noise units.
const RESPONSE_NOISE: f64 = 3.0;
/// Candidate cap.
const MAX_CANDIDATES: usize = 1500;
/// Smallest ball radius, in pixels of the working grid.
const R_MIN_PX: f64 = 3.2;
/// ... and in half-diagonals (the larger of the two applies).
const R_MIN_HD: f64 = 0.0025;
/// A ball with no edges around it (flat ground) must be at least this large, in pixels of the working grid: a
/// smaller light is more likely a point light in focus.
const UNSUPPORTED_R_MIN_PX: f64 = 5.0;
/// Plateau over ground in square-root units (the brightest channel).
const MIN_CONTRAST: f64 = 0.12;
/// On a plane whose noise (square-root units) is above this the edges around a ball are too noisy to tell it from a
/// patch of a texture (the ball is left out unless the ground is flat). (The lens
/// planes of real photos are binned and denoised: 0.0015 to 0.003.)
const NOISY_PLANE: f64 = 0.008;
/// A ball whose rim is wider than this fraction of its radius needs the blur around it to say so (edges or a blur
/// map): a smooth bright patch of a sharp subject (white fur, a cloud) is no ball.
const SOFT_NEEDS_EVIDENCE: f64 = 0.9;
/// A ball with a wide rim (softness above [`SOFT_NEEDS_EVIDENCE`]) must lie where a learned blur map reads blurred:
/// at least this share of the pixels of its disc and rim (to 1.15 radii) must be called blurred. The blur around it
/// is not enough (a bud in a blurred garden has that), the blob itself has to be out of focus.
const SOFT_BLURRED_SHARE: f64 = 0.5;
/// A ball whose rim is wider than this fraction of its radius is a smooth blob, not a ball. (A soft ball must also
/// stand out by [`SOFT_CONTRAST`] per unit of softness, which is what keeps the dim soft blobs out: this bound
/// only holds back the ones that are all rim.)
const MAX_SOFTNESS: f64 = 1.3;
/// The contrast a ball needs grows with the softness of its rim above the softness any ball of its size has by this
/// much per unit.
const SOFT_CONTRAST: f64 = 0.3;
/// A sharp rim is this wide, in pixels of the working grid, once the optics' blur and the pixel grid have had
/// their say: the rim of a small ball measures soft whatever it is (1.5 px over a radius of 4.5 px is 0.33), and
/// only the softness above that makes the ball a smooth blob. At least [`NATIVE_SOFTNESS_MIN`] of the radius.
const NATIVE_RIM_PX: f64 = 1.5;
const NATIVE_SOFTNESS_MIN: f64 = 0.2;
/// The ring around a ball is calm where it stays below this fraction of the way from the ground to the plateau.
const RING_CALM: f64 = 0.5;
/// Up to this share of the ring may be above that level: a light beside a bright thing (the curl of a hook, its
/// neighbour in a cluster of lights) is still a light, while a patch of a texture has neighbours of its own brightness
/// all around it. A ball that needs more than [`RING_BRIGHT_CLEAR`] of it (the strict rule, a twentieth) is not a clear
/// ball ([`BallPx::clear`]), and a large one (over [`SMALL_BALL_RADIUS_HD`]) that needs more than that is no ball at
/// all: what shares the ring of a large light is most likely a neighbour it overlaps in a field of discs, which is
/// not found as balls.
const RING_BRIGHT_MAX: f64 = 0.15;
const RING_BRIGHT_CLEAR: f64 = 0.05;
/// A ball of at most this radius, as a fraction of the working grid's half-diagonal, is a small one. Only a small
/// ball's take-out disc is taken beyond 1.15 `r` to where its light has died away: the glow of a large soft ball is
/// not told from the field it lies in, and a disc as large as that would show as a flat disc.
const SMALL_BALL_RADIUS_HD: f64 = 0.025;
/// Plateau over ground in light.
const MIN_RATIO: f64 = 2.5;
/// Plateau over ground, in units of the plateau's noise.
const MIN_SN: f64 = 6.0;
/// The level of the ball's region, as a fraction of the way from the ground to the plateau.
const LEVEL: f64 = 0.4;
/// The most elongated a ball may be (ratio of the region's principal axes).
const MAX_ELONGATION: f64 = 1.6;
/// The region must fill at least this fraction of the ellipse that has its second moments.
const MIN_FILL: f64 = 0.8;
/// The search window is this many seed radii either side of the centre.
const WINDOW_RADII: f64 = 3.2;
/// A ball whose ring has more than this share of its pixels at a strong sharp edge sits in a sharp scene (a gap
/// in the leaves, a seed, a catch-light): it is a light in focus, not a defocused one. The share is that of the
/// sectors the ring is cut into: see [`RING_SECTORS`].
pub(super) const MAX_SHARP_DENSITY: f64 = 0.05;
/// The ring around a ball (1.5 to 3 radii) is cut into this many sectors, and what is sharp around the ball is the
/// share that at least five eighths of them reach (the lower median): a light in focus has a sharp ground all
/// around it, a defocused light beside a sharp subject (a hat, a hand) has it on one side.
const RING_SECTORS: usize = 8;
/// A sector with fewer pixels than this (outside the frame, or inside the footprints of other balls) does not count.
const MIN_SECTOR_PX: f64 = 6.0;
/// Edge evidence (sum of weights in the ring around a ball) needed to hold a ball to the edges' radius.
const EVIDENCE_MIN: f64 = 1.5;
/// A ball whose ground reads sharper than this radius, in pixels of the working grid, is a light in focus. (The
/// edge readings around a defocused light on foliage are mostly of the fine texture, a pixel or two wide, so
/// only a ground that reads as sharp as the optics allow counts against a ball.)
const GROUND_SHARP_PX: f64 = 1.0;
/// A ball whose ground is this much blurrier than the ball is not a ball of that ground.
const BLURRIER_THAN: f64 = 3.0;
/// A scene needs this many balls, or balls the edges confirm.
const MIN_CLUSTER: usize = 3;
/// The ring of a ball needs this many pixels for a plane fit.
const RING_FIT_MIN_SAMPLES: usize = 12;
/// Of two balls closer than this many radii the dimmer is dropped.
const MIN_SEPARATION: f64 = 1.2;

#[derive(Clone, Copy, Debug)]
struct Candidate {
    score: f32,
    x: usize,
    y: usize,
    r0: f64,
}

#[derive(Clone, Debug)]
struct Measured {
    cx: f64,
    cy: f64,
    r: f64,
    /// The plateau's brightest-channel level and the ground's.
    top: f64,
    ground: f64,
    /// 90th percentile of the luminance of the disc (the brightness order).
    peak: f64,
    /// Width of the rim (from 80 % to 20 % of the way from the ground to the plateau, in the mean radial
    /// profile) over the radius: about 0.2 for a sharp-rimmed disc, over 0.5 for a soft blob.
    softness: f64,
    /// The radius at which the ball's light has died away into the ground (median radial profile), when it has
    /// within [`TAKE_OUT_MAX_RADII`] radii (a ball on a bright region or among others never gets there) and the ball
    /// is a small one ([`SMALL_BALL_RADIUS_HD`]).
    tail: Option<f64>,
    /// The share of the ring (1.6 to 2.5 radii) that is above half way from the ground to the plateau.
    ring_bright: f64,
}

impl Measured {
    /// The radius of the ball's rim zone, 1.15 `r` plus the margin: its own sharp edge, which says nothing about the
    /// blur around it, so the edge evidence skips it.
    fn r_rim(&self) -> f64 {
        TAKE_OUT_MIN_RADII * self.r + TAKE_OUT_MARGIN_PX
    }

    /// The radius of the disc that is taken out: 1.15 `r` plus the margin, or the tail's end when that is farther,
    /// at most [`TAKE_OUT_MAX_RADII`] radii.
    fn r_out(&self) -> f64 {
        let base = TAKE_OUT_MIN_RADII * self.r + TAKE_OUT_MARGIN_PX;
        match self.tail {
            Some(tail) => base
                .max(tail + 1.0)
                .min(TAKE_OUT_MAX_RADII * self.r + TAKE_OUT_MARGIN_PX),
            None => base,
        }
    }
}

fn median(values: &mut [f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, v, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    f64::from(*v)
}

fn percentile(values: &mut [f32], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    let (_, v, _) = values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    f64::from(*v)
}

/// Visits every pixel whose centre lies within `[r_lo, r_hi]` of `(cx, cy)`, as `(x, y, dx, dy, d)`.
fn for_each_in_annulus(
    w: usize,
    h: usize,
    cx: f64,
    cy: f64,
    r_lo: f64,
    r_hi: f64,
    mut f: impl FnMut(usize, usize, f64, f64, f64),
) {
    let x0 = (cx - r_hi).floor().max(0.0) as usize;
    let y0 = (cy - r_hi).floor().max(0.0) as usize;
    let x1 = ((cx + r_hi).ceil() as usize + 1).min(w);
    let y1 = ((cy + r_hi).ceil() as usize + 1).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d >= r_lo && d <= r_hi {
                f(x, y, dx, dy, d);
            }
        }
    }
}

/// The brightest channel of every pixel (linear), and its square root.
fn brightest_channel(inp: &BallInput) -> (Vec<f32>, Vec<f32>) {
    let n = inp.w * inp.h;
    let mut v = Vec::with_capacity(n);
    let mut root = Vec::with_capacity(n);
    for i in 0..n {
        let m = inp.rgb[0][i].max(inp.rgb[1][i]).max(inp.rgb[2][i]).max(0.0);
        v.push(m);
        root.push(m.sqrt());
    }
    (v, root)
}

/// The scale-space maxima of the differences of Gaussians of `root`, strongest response first.
fn candidates(
    inp: &BallInput,
    root: &[f32],
    cancel: &dyn Fn() -> bool,
) -> Result<Vec<Candidate>, LensError> {
    let (w, h) = (inp.w, inp.h);
    // The levels, each made from the one before (Gaussians add in quadrature), and their differences: a bright
    // blob is brighter in the finer level, so the response is positive.
    let mut dogs: Vec<Vec<f32>> = Vec::with_capacity(SCALES.len() - 1);
    let mut prev = blur_wide(root, w, h, SCALES[0]);
    for m in 1..SCALES.len() {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let step = (SCALES[m] * SCALES[m] - SCALES[m - 1] * SCALES[m - 1]).sqrt();
        let next = blur_wide(&prev, w, h, step);
        let d: Vec<f32> = prev.iter().zip(&next).map(|(a, b)| a - b).collect();
        dogs.push(d);
        prev = next;
    }
    let theta = RESPONSE_MIN.max(RESPONSE_NOISE * inp.noise_v.max(0.0015)) as f32;
    let mut out = Vec::new();
    for mi in 0..dogs.len() {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let r0 = SEED_PER_SCALE * SCALES[mi];
        if r0 < 0.7 * R_MIN_PX || r0 > 1.5 * inp.r_max_px {
            continue;
        }
        let d = &dogs[mi];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let v = d[y * w + x];
                if v <= theta {
                    continue;
                }
                let mut is_max = true;
                'nb: for (nj, other) in dogs
                    .iter()
                    .enumerate()
                    .take((mi + 1).min(dogs.len() - 1) + 1)
                    .skip(mi.saturating_sub(1))
                {
                    for dy in 0..3usize {
                        for dx in 0..3usize {
                            if nj == mi && dx == 1 && dy == 1 {
                                continue;
                            }
                            let (nx, ny) = (x + dx - 1, y + dy - 1);
                            let u = other[ny * w + nx];
                            // Ties go to the neighbour that comes first in (level, row, column) order.
                            let before = (nj, ny, nx) < (mi, y, x);
                            if u > v || (before && u == v) {
                                is_max = false;
                                break 'nb;
                            }
                        }
                    }
                }
                if is_max {
                    out.push(Candidate { score: v, x, y, r0 });
                }
            }
        }
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.y.cmp(&b.y))
            .then(a.x.cmp(&b.x))
    });
    out.truncate(MAX_CANDIDATES);
    Ok(out)
}

/// Scratch space of the flood fill, reused between candidates.
struct Scratch {
    seen: Vec<bool>,
    stack: Vec<(usize, usize)>,
    values: Vec<f32>,
}

/// Measures one candidate, or rejects it.
fn measure(
    inp: &BallInput,
    vmax: &[f32],
    smooth: &[f32],
    c: &Candidate,
    sc: &mut Scratch,
) -> Option<Measured> {
    let (w, h) = (inp.w, inp.h);
    let (mut cx, mut cy) = (c.x as f64 + 0.5, c.y as f64 + 0.5);
    let mut r = c.r0;
    let (mut top, mut ground, mut ring_bright) = (0.0, 0.0, 0.0);
    let (mut elong, mut fill) = (1.0, 1.0);
    for _ in 0..3 {
        // The plateau and the ground.
        sc.values.clear();
        for_each_in_annulus(w, h, cx, cy, 0.0, (0.5 * r).max(1.2), |x, y, _, _, _| {
            sc.values.push(vmax[y * w + x]);
        });
        if sc.values.len() < 3 {
            return None;
        }
        top = percentile(&mut sc.values, 0.75);
        sc.values.clear();
        for_each_in_annulus(
            w,
            h,
            cx,
            cy,
            1.6 * r + 1.0,
            2.5 * r + 2.0,
            |x, y, _, _, _| {
                sc.values.push(vmax[y * w + x]);
            },
        );
        if sc.values.len() < 8 {
            return None;
        }
        ground = median(&mut sc.values);
        if top < MIN_RATIO * ground {
            return None;
        }
        // A light stands alone: nothing around it is nearly as bright (a patch of a texture has neighbours of
        // its own brightness).
        let ring_hi = ground + RING_CALM * (top - ground);
        ring_bright = sc
            .values
            .iter()
            .filter(|v| f64::from(**v) > ring_hi)
            .count() as f64
            / sc.values.len() as f64;
        let level = (ground + LEVEL * (top - ground)) as f32;
        // The region above the level that holds the centre, in a window around it.
        let win = (WINDOW_RADII * r).ceil() as i64 + 2;
        let (ix, iy) = (cx.floor() as i64, cy.floor() as i64);
        let (x0, x1) = ((ix - win).max(0), (ix + win + 1).min(w as i64));
        let (y0, y1) = ((iy - win).max(0), (iy + win + 1).min(h as i64));
        let (ww, wh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let at = |x: i64, y: i64| smooth[y as usize * w + x as usize];
        // The seed: the pixel under the centre, or the nearest pixel above the level within 2 px.
        let mut seed: Option<(i64, i64)> = None;
        let mut best = f64::INFINITY;
        for yy in (iy - 2).max(y0)..=(iy + 2).min(y1 - 1) {
            for xx in (ix - 2).max(x0)..=(ix + 2).min(x1 - 1) {
                if at(xx, yy) > level {
                    let d = (xx as f64 + 0.5 - cx).powi(2) + (yy as f64 + 0.5 - cy).powi(2);
                    if d < best {
                        best = d;
                        seed = Some((xx, yy));
                    }
                }
            }
        }
        let (sx, sy) = seed?;
        sc.seen.clear();
        sc.seen.resize(ww * wh, false);
        sc.stack.clear();
        sc.stack.push((sx as usize, sy as usize));
        sc.seen[(sy - y0) as usize * ww + (sx - x0) as usize] = true;
        let (mut n, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
        while let Some((px, py)) = sc.stack.pop() {
            let (fx, fy) = (px as f64 + 0.5, py as f64 + 0.5);
            n += 1.0;
            mx += fx;
            my += fy;
            sxx += fx * fx;
            sxy += fx * fy;
            syy += fy * fy;
            for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (px as i64 + dx, py as i64 + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                if nx < x0 || ny < y0 || nx >= x1 || ny >= y1 {
                    // The region leaves the window: a texture, not a compact disc of this size.
                    if at(nx, ny) > level {
                        return None;
                    }
                    continue;
                }
                let li = (ny - y0) as usize * ww + (nx - x0) as usize;
                if sc.seen[li] {
                    continue;
                }
                sc.seen[li] = true;
                if at(nx, ny) > level {
                    sc.stack.push((nx as usize, ny as usize));
                }
            }
            if n > 40_000.0 {
                return None;
            }
        }
        if n < 6.0 {
            return None;
        }
        let (mean_x, mean_y) = (mx / n, my / n);
        // Central second moments; a pixel's own extent adds 1/12 to each variance.
        let vxx = sxx / n - mean_x * mean_x + 1.0 / 12.0;
        let vyy = syy / n - mean_y * mean_y + 1.0 / 12.0;
        let vxy = sxy / n - mean_x * mean_y;
        let tr = vxx + vyy;
        let det = vxx * vyy - vxy * vxy;
        let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
        let (l1, l2) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(1e-12));
        elong = (l1 / l2).sqrt();
        fill = n / (4.0 * std::f64::consts::PI * (l1 * l2).sqrt());
        cx = mean_x;
        cy = mean_y;
        r = (n / std::f64::consts::PI).sqrt();
    }
    let contrast = top.sqrt() - ground.sqrt();
    if contrast < MIN_CONTRAST || ring_bright > RING_BRIGHT_MAX {
        return None;
    }
    if top - ground < MIN_SN * 2.0 * top.max(1e-6).sqrt() * inp.noise_v {
        return None;
    }
    if elong > MAX_ELONGATION || fill < MIN_FILL {
        return None;
    }
    let r_lo = R_MIN_PX.max(R_MIN_HD * hd_of(inp));
    if r < r_lo || r > inp.r_max_px {
        return None;
    }
    // The brightness order: the 90th percentile of the luminance inside the ball.
    sc.values.clear();
    for_each_in_annulus(w, h, cx, cy, 0.0, 0.9 * r, |x, y, _, _, _| {
        sc.values.push(inp.y[y * w + x]);
    });
    if sc.values.len() < 4 {
        return None;
    }
    let peak = percentile(&mut sc.values, 0.9);
    let (softness, tail) = rim_profile(inp, vmax, cx, cy, r, top, ground);
    Some(Measured {
        cx,
        cy,
        r,
        top,
        ground,
        peak,
        softness,
        tail,
        ring_bright,
    })
}

/// The rim width over the radius (from the mean radial profile in rings half a pixel wide), and the radius at which
/// the ball's light has died away (from the median profile, which a neighbour covering part of a ring does not
/// lift; see [`TAIL_FRACTION`]).
fn rim_profile(
    inp: &BallInput,
    vmax: &[f32],
    cx: f64,
    cy: f64,
    r: f64,
    top: f64,
    ground: f64,
) -> (f64, Option<f64>) {
    let (w, h) = (inp.w, inp.h);
    let limit = 2.5 * r + 1.0;
    let bins = (2.0 * limit).ceil() as usize + 1;
    let mut sum = vec![0.0f64; bins];
    let mut cnt = vec![0.0f64; bins];
    let mut rings: Vec<Vec<f32>> = vec![Vec::new(); bins];
    for_each_in_annulus(w, h, cx, cy, 0.0, limit, |x, y, _, _, d| {
        let i = ((2.0 * d) as usize).min(bins - 1);
        sum[i] += f64::from(vmax[y * w + x]);
        cnt[i] += 1.0;
        rings[i].push(vmax[y * w + x]);
    });
    let span = (top - ground).max(1e-9);
    let (mut d80, mut d20) = (None, None);
    for i in 0..bins {
        if cnt[i] == 0.0 {
            continue;
        }
        let e = (sum[i] / cnt[i] - ground) / span;
        let d = 0.5 * (i as f64 + 0.5);
        if d80.is_none() && e < 0.8 {
            d80 = Some(d);
        }
        if d80.is_some() && d20.is_none() && e < 0.2 {
            d20 = Some(d);
        }
    }
    let softness = match (d80, d20) {
        (Some(a), Some(b)) => (b - a).max(0.0) / r,
        _ => 1.0,
    };
    // The tail: the first ring beyond the ball's radius whose median is within the allowance of the ground.
    let noise_lin = 2.0 * ground.max(1e-6).sqrt() * inp.noise_v.max(0.0015);
    let allowance = (TAIL_NOISE_UNITS * noise_lin)
        .max((TAIL_FRACTION * (top - ground)).min(TAIL_OF_GROUND * ground));
    let reach = TAKE_OUT_MAX_RADII * r + TAKE_OUT_MARGIN_PX;
    let mut tail = None;
    for (i, ring) in rings.iter_mut().enumerate() {
        if r > SMALL_BALL_RADIUS_HD * hd_of(inp) {
            break;
        }
        let d = 0.5 * (i as f64 + 0.5);
        if d <= r || ring.is_empty() {
            continue;
        }
        if d > reach {
            break;
        }
        if median(ring) - ground <= allowance {
            tail = Some(d);
            break;
        }
    }
    (softness, tail)
}

/// The half-diagonal of the working grid in pixels.
fn hd_of(inp: &BallInput) -> f64 {
    0.5 * ((inp.w * inp.w + inp.h * inp.h) as f64).sqrt()
}

/// Solves the 3 x 3 system `m x = v` by Gaussian elimination with partial pivoting; `None` when singular.
#[allow(clippy::needless_range_loop)] // indices of a 3 x 3 matrix read better as indices
fn solve3(mut m: [[f64; 3]; 3], mut v: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        v.swap(col, piv);
        for row in col + 1..3 {
            let f = m[row][col] / m[col][col];
            for k in col..3 {
                m[row][k] -= f * m[col][k];
            }
            v[row] -= f * v[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut s = v[row];
        for k in row + 1..3 {
            s -= m[row][k] * x[k];
        }
        x[row] = s / m[row][row];
    }
    Some(x)
}

/// Least-squares plane `a + b dx + c dy` through `samples = (dx, dy, value)`.
#[allow(clippy::needless_range_loop)] // indices of a 3 x 3 matrix read better as indices
fn fit_plane(samples: &[(f64, f64, f64)]) -> Option<[f64; 3]> {
    let mut m = [[0.0f64; 3]; 3];
    let mut v = [0.0f64; 3];
    for &(dx, dy, z) in samples {
        let row = [1.0, dx, dy];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += row[i] * row[j];
            }
            v[i] += row[i] * z;
        }
    }
    solve3(m, v)
}

/// The ring's plane fit per channel (robust: samples far from the first luminance fit are dropped).
fn ring_fit(inp: &BallInput, ball: &Measured, r_out: f64, owner: &[i32]) -> [[f64; 3]; 3] {
    let (w, h) = (inp.w, inp.h);
    // (pixel, dx, dy): the ring's pixels that no ball owns, and that do not read sharp (a hat or a hand beside the
    // light is not its ground); with too few of those, all of them.
    let mut samples: Vec<(usize, f64, f64)> = Vec::new();
    let mut with_sharp: Vec<(usize, f64, f64)> = Vec::new();
    for_each_in_annulus(
        w,
        h,
        ball.cx,
        ball.cy,
        r_out + 1.0,
        r_out + 0.6 * ball.r + 2.0,
        |x, y, dx, dy, _| {
            let i = y * w + x;
            if owner[i] < 0 {
                with_sharp.push((i, dx, dy));
                if !inp.sharp[i] && !inp.anchor[i] {
                    samples.push((i, dx, dy));
                }
            }
        },
    );
    if samples.len() < RING_FIT_MIN_SAMPLES {
        samples = with_sharp;
    }
    let channel = |c: usize, keep: &[bool]| -> Option<[f64; 3]> {
        let pts: Vec<(f64, f64, f64)> = samples
            .iter()
            .zip(keep)
            .filter(|(_, k)| **k)
            .map(|(s, _)| (s.1, s.2, f64::from(inp.rgb[c][s.0])))
            .collect();
        if pts.len() < RING_FIT_MIN_SAMPLES {
            return None;
        }
        fit_plane(&pts)
    };
    let mut keep = vec![true; samples.len()];
    for _ in 0..2 {
        let lum: Vec<(f64, f64, f64)> = samples
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(s, _)| (s.1, s.2, f64::from(inp.y[s.0])))
            .collect();
        if lum.len() < RING_FIT_MIN_SAMPLES {
            break;
        }
        let Some(p) = fit_plane(&lum) else { break };
        let mut res: Vec<f32> = samples
            .iter()
            .map(|s| (f64::from(inp.y[s.0]) - (p[0] + p[1] * s.1 + p[2] * s.2)).abs() as f32)
            .collect();
        let mad = median(&mut res);
        let limit = 3.0 * 1.4826 * mad + 1e-6;
        for (i, s) in samples.iter().enumerate() {
            let r = (f64::from(inp.y[s.0]) - (p[0] + p[1] * s.1 + p[2] * s.2)).abs();
            keep[i] = r <= limit;
        }
    }
    let mut out = [[0.0f64; 3]; 3];
    for (c, o) in out.iter_mut().enumerate() {
        *o = channel(c, &keep).unwrap_or_else(|| {
            // Too few ring samples: a flat level, the median of what there is.
            let mut vals: Vec<f32> = samples.iter().map(|s| inp.rgb[c][s.0]).collect();
            [median(&mut vals), 0.0, 0.0]
        });
    }
    out
}

/// The share of the ring around a ball (1.5 to 3 radii, outside the footprints in `claim`) that reads sharp. With a
/// dense blur map ([`BallInput::sector_rule`]) it is the lower median, over the ring's [`RING_SECTORS`] sectors, of
/// each sector's share (so that at least five eighths of the sectors are that sharp or sharper; sectors with hardly
/// a pixel do not count); with the classical detector's edge flags it is the share of the whole ring.
fn ring_sharpness(inp: &BallInput, cx: f64, cy: f64, r: f64, claim: &[bool], sectors: bool) -> f64 {
    let w = inp.w;
    let mut n = [0.0f64; RING_SECTORS];
    let mut sharp = [0.0f64; RING_SECTORS];
    for_each_in_annulus(inp.w, inp.h, cx, cy, 1.5 * r, 3.0 * r, |x, y, dx, dy, _| {
        let i = y * w + x;
        if claim[i] {
            return;
        }
        let turn = (libm::atan2(dy, dx) + std::f64::consts::PI) / std::f64::consts::TAU;
        let k = ((turn * RING_SECTORS as f64) as usize).min(RING_SECTORS - 1);
        n[k] += 1.0;
        sharp[k] += f64::from(u8::from(inp.sharp[i]));
    });
    let (total, flagged) = (n.iter().sum::<f64>(), sharp.iter().sum::<f64>());
    // The sectors that have pixels (a ball at the frame's edge has some outside it); with fewer than four the whole
    // ring counts.
    let mut share: Vec<f64> = (0..RING_SECTORS)
        .filter(|&k| n[k] >= MIN_SECTOR_PX)
        .map(|k| sharp[k] / n[k])
        .collect();
    if !sectors || share.len() < RING_SECTORS / 2 {
        return flagged / total.max(1.0);
    }
    share.sort_by(f64::total_cmp);
    // Five eighths of the sectors that count are at least this sharp.
    share[(share.len() - 1) * 3 / 8]
}

/// Finds the balls of a plane.
pub(super) fn find_balls(
    inp: &BallInput,
    cancel: &dyn Fn() -> bool,
) -> Result<Vec<BallPx>, LensError> {
    let (w, h) = (inp.w, inp.h);
    if w < 16 || h < 16 {
        return Ok(Vec::new());
    }
    let (vmax, root) = brightest_channel(inp);
    let cands = candidates(inp, &root, cancel)?;
    drop(root);
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // Phase A: measure and shape-test every candidate (on a lightly smoothed copy of the plane).
    let smooth = blur(&vmax, w, h, 0.7);
    let mut sc = Scratch {
        seen: Vec::new(),
        stack: Vec::new(),
        values: Vec::new(),
    };
    let mut refined: Vec<Measured> = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        if i % 64 == 0 && cancel() {
            return Err(LensError::Cancelled);
        }
        if inp.anchor[c.y * w + c.x] {
            continue;
        }
        if let Some(m) = measure(inp, &vmax, &smooth, c, &mut sc) {
            refined.push(m);
        }
    }
    drop(smooth);
    // Brightest first, ties by position.
    refined.sort_by(|a, b| {
        b.peak
            .total_cmp(&a.peak)
            .then(a.cy.total_cmp(&b.cy))
            .then(a.cx.total_cmp(&b.cx))
    });
    // A region is found from every candidate inside it: one measured ball stands for all of them.
    let mut unique: Vec<Measured> = Vec::new();
    for m in refined {
        if unique.iter().any(|u| {
            let d = ((u.cx - m.cx).powi(2) + (u.cy - m.cy).powi(2)).sqrt();
            d < 0.6 * u.r.max(m.r)
        }) {
            continue;
        }
        unique.push(m);
    }
    let refined = unique;
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // The footprints of every candidate that passed the shape tests (the edge evidence skips them: their rims
    // are sharp edges that say nothing about the blur around).
    let mut claim = vec![false; w * h];
    for b in &refined {
        let r_out = b.r_rim();
        for_each_in_annulus(w, h, b.cx, b.cy, 0.0, 1.4 * r_out, |x, y, _, _, _| {
            claim[y * w + x] = true;
        });
    }
    // Phase B: overlap and context, brightest first.
    let mut accepted: Vec<(Measured, bool, bool)> = Vec::new();
    'cand: for b in &refined {
        for (a, _, _) in &accepted {
            let d = ((a.cx - b.cx).powi(2) + (a.cy - b.cy).powi(2)).sqrt();
            if d < MIN_SEPARATION * a.r.max(b.r) {
                continue 'cand;
            }
        }
        let (mut ew, mut es) = (0.0f64, 0.0f64);
        for_each_in_annulus(w, h, b.cx, b.cy, 1.5 * b.r, 3.0 * b.r, |x, y, _, _, _| {
            let i = y * w + x;
            if !claim[i] {
                ew += f64::from(inp.ev_w[i]);
                es += f64::from(inp.ev_s[i]);
            }
        });
        let sharp_density = ring_sharpness(inp, b.cx, b.cy, b.r, &claim, inp.sector_rule);
        let supported = ew >= EVIDENCE_MIN;
        if b.ring_bright > RING_BRIGHT_CLEAR && b.r > SMALL_BALL_RADIUS_HD * hd_of(inp) {
            continue;
        }
        if sharp_density > inp.max_sharp_density {
            continue;
        }
        // A soft blob must stand out more: a dim, soft highlight (a nose, a bead of water) is no ball.
        let contrast = b.top.sqrt() - b.ground.sqrt();
        // (The softness every ball of this size has is not held against it.)
        let soft_excess = (b.softness - (NATIVE_RIM_PX / b.r).max(NATIVE_SOFTNESS_MIN)).max(0.0);
        if b.softness > MAX_SOFTNESS
            || (b.softness > SOFT_NEEDS_EVIDENCE && !supported)
            || contrast < MIN_CONTRAST + SOFT_CONTRAST * soft_excess
        {
            continue;
        }
        if b.softness > SOFT_NEEDS_EVIDENCE
            && let Some(blurred) = inp.blurred
        {
            let (mut n, mut blurry) = (0.0f64, 0.0f64);
            for_each_in_annulus(
                w,
                h,
                b.cx,
                b.cy,
                0.0,
                TAKE_OUT_MIN_RADII * b.r,
                |x, y, _, _, _| {
                    n += 1.0;
                    blurry += f64::from(u8::from(blurred[y * w + x]));
                },
            );
            if blurry < SOFT_BLURRED_SHARE * n {
                continue;
            }
        }
        if supported && inp.noise_v > NOISY_PLANE {
            // The edges around it are too noisy to read: no way to tell a light from a patch of a texture.
            continue;
        }
        if supported {
            // The edges' mean log radius against the ball's: a ground much sharper than the ball is a light in
            // focus; a ground much blurrier is not this ball's ground.
            let ground = libm::exp(es / ew);
            if ground < GROUND_SHARP_PX || ground > BLURRIER_THAN * b.r {
                continue;
            }
        } else if b.r < UNSUPPORTED_R_MIN_PX {
            continue;
        }
        // A clear ball passes the strict rules too, and counts towards a scene of lights.
        let clear = b.ring_bright <= RING_BRIGHT_CLEAR
            && b.softness <= SOFT_NEEDS_EVIDENCE
            && contrast >= MIN_CONTRAST + SOFT_CONTRAST * b.softness
            && (!inp.sector_rule
                || ring_sharpness(inp, b.cx, b.cy, b.r, &claim, false) <= inp.max_sharp_density);
        accepted.push((b.clone(), supported, clear));
        if accepted.len() >= MAX_BALLS {
            break;
        }
    }
    // Verification: the edges of the other candidates were left out of the evidence above (their rims say nothing
    // about the blur), which in a dense texture leaves no evidence at all. Now only the balls accepted are left
    // out: a patch of a sharp texture that was taken for a ball finds the texture's edges around it and goes.
    for _ in 0..2 {
        let mut claim2 = vec![false; w * h];
        for (b, _, _) in &accepted {
            let r_out = b.r_rim();
            for_each_in_annulus(w, h, b.cx, b.cy, 0.0, 1.4 * r_out, |x, y, _, _, _| {
                claim2[y * w + x] = true;
            });
        }
        let before = accepted.len();
        accepted.retain(|(b, _, _)| {
            ring_sharpness(inp, b.cx, b.cy, b.r, &claim2, inp.sector_rule) <= inp.max_sharp_density
        });
        if accepted.len() == before {
            break;
        }
    }
    // A scene needs a cluster of balls, or balls the edges confirm.
    if accepted.iter().filter(|a| a.2).count() < MIN_CLUSTER {
        accepted.retain(|(_, s, _)| *s);
    }
    if cancel() {
        return Err(LensError::Cancelled);
    }
    // The take-out discs. (A soft ball's tail is left in the picture: it is gathered with the rest of the
    // blur and reads as the glow of the ball. Taking it out would leave a dark box around the redrawn oval.)
    // The disc reaches no farther than the nearest pixel that reads sharp (a subject, a strand, the focus point's
    // anchor) beyond the ball's rim zone: light is not taken out across a sharp thing.
    let r_outs: Vec<f64> = accepted
        .iter()
        .map(|(b, _, _)| {
            let (lo, hi) = (b.r_rim(), b.r_out());
            if hi <= lo {
                return lo;
            }
            let mut nearest = hi;
            for_each_in_annulus(w, h, b.cx, b.cy, lo, hi, |x, y, _, _, d| {
                let i = y * w + x;
                if (inp.sharp[i] || inp.anchor[i]) && !claim[i] && d < nearest {
                    nearest = d;
                }
            });
            (nearest - 1.0).max(lo)
        })
        .collect();
    // Ownership of the discs (the brightest ball owns a pixel two discs share).
    let mut owner = vec![-1i32; w * h];
    for (i, (b, _, _)) in accepted.iter().enumerate() {
        for_each_in_annulus(w, h, b.cx, b.cy, 0.0, r_outs[i], |x, y, _, _, _| {
            let o = &mut owner[y * w + x];
            if *o < 0 {
                *o = i as i32;
            }
        });
    }
    let mut out = Vec::with_capacity(accepted.len());
    for (i, (b, supported, clear)) in accepted.iter().enumerate() {
        let r_out = r_outs[i];
        let fit = ring_fit(inp, b, r_out, &owner);
        let mut energy = [0.0f64; 3];
        for_each_in_annulus(w, h, b.cx, b.cy, 0.0, r_out, |x, y, dx, dy, _| {
            if owner[y * w + x] == i as i32 {
                for c in 0..3 {
                    let base = fit[c][0] + fit[c][1] * dx + fit[c][2] * dy;
                    energy[c] += f64::from(inp.rgb[c][y * w + x]) - base;
                }
            }
        });
        for e in &mut energy {
            *e = e.max(0.0);
        }
        out.push(BallPx {
            cx: b.cx,
            cy: b.cy,
            r: b.r,
            r_out,
            energy,
            fit,
            peak: b.peak,
            supported: *supported,
            softness: b.softness,
            clear: *clear,
        });
    }
    Ok(out)
}
