//! The bright balls stretched (`docs/21` section 4, L1, step 5).
//!
//! A ball is not drawn again from its energy: its own light is stretched. For each ball the light of the picture
//! above the plane fit of its ring (`light = max(picture - fit, 0)`, per channel) is read back at the position the
//! pixel has in the ball before the stretch, `d' = d / q` along the long axis and `d` across, over the disc the ball
//! was taken out of (`r_out`, its glow included), and added to the stretched picture with an alpha of one over the
//! disc and a ramp of a pixel at its edge. The edge softness, the colour, the glow and the clipped core of the ball
//! carry over because they are the ball's own pixels, and nothing but the ball's light is taken out and put back, so
//! no ring is left. The surface brightness of the ball is kept, not its energy: a clipped ball (whose energy is
//! unknown) stays as bright as it was over its larger area.
//!
//! The long axis is the sum of the stretch across the anamorphic axis, `(q - 1) r`, and the stretch along the
//! tangent, `(q_t - 1) r` with `q_t = sqrt(1 + 4 s^2 / 3)` (the aspect of a disc blurred along a box line of
//! half-length `s r`, whose second moment the gather matches; `s = A (sigma + a H / (k r))` for the lens's swirl
//! `sigma` and astigmatism `a` and the Anamorphic amount `A`, [`tangent_half_length`]) held to
//! [`TANGENT_ASPECT_MAX`], the sign of the second chosen to add. The lens's cat's-eye cuts the stretched patch with
//! a soft mask: the patch's ellipse intersected with a copy of it shifted radially (by `A` times the lens's
//! offset). The swirl and the cat's-eye belong to the Anamorphic slider, not to Edges (`docs/21` section 4, L1,
//! *As built (swirl with Anamorphic)*). The lens's rim profile (`bokeh.edge`) modulates the light, relative to its
//! mean over the disc (so that a flat profile changes nothing), across the ball and not beyond it.
//!
//! Everything is a function of the ball and the pixel's absolute position, so tiles agree.

use crate::consts::{
    BALL_GLOW_FADE, PROFILE_SCALE, RIM_FADE_RADII, RIM_SAMPLES, RIM_SOFT_FROM, RIM_SOFT_TO,
    TANGENT_ASPECT_MAX, TANGENT_REACH_MAX,
};

use super::Ball;

/// The largest distance between the two copies of a cat's-eye ball, in the ellipse's own semi-axes.
const MAX_CAT_OFFSET: f64 = 1.8;
/// The soft edge of the cat's-eye mask is at least this many pixels wide, and half the glow's width at most.
const MASK_EDGE_MIN_PX: f64 = 1.5;
/// How many earlier balls overlapping a ball's disc are remembered to give each pixel of a shared disc to one ball.
pub(super) const OWNERS: usize = 3;

/// A ball's stretched patch in the pixels of the view being drawn.
#[derive(Clone, Debug)]
pub(super) struct Shape {
    /// Centre in pixel coordinates (pixel `i` covers `[i, i + 1)`).
    pub cx: f64,
    pub cy: f64,
    /// Semi-axes of the stretched patch: `a` along `m`, `b` across (`b` is the disc's radius, `a = f b`).
    pub a: f64,
    pub b: f64,
    /// The ratio of the patch's long axis to its short one: the stretch.
    pub f: f64,
    /// Unit vector along the long axis.
    pub m: (f64, f64),
    /// Half the cat's-eye offset between the two copies (a vector in pixels, in the patch's scale); zero without one.
    pub half_off: (f64, f64),
    /// The cat's-eye offset in the ellipse's normalised frame, in units of the semi-axes.
    pub delta: f64,
    /// Width in pixels of the soft edge of the cat's-eye mask.
    pub edge: f64,
    /// Radius, in pixels, of the disc taken out of the picture and read back.
    pub r_out: f64,
    /// The ball's own radius in pixels.
    pub r_ball: f64,
    /// How much of the lens's rim profile this ball takes, 0 to 1: all for a crisp disc, none for a soft glow
    /// ([`RIM_SOFT_FROM`], [`RIM_SOFT_TO`]).
    pub rim_k: f64,
    /// Half the ball's plateau (linear luminance): a pixel in the ball at least this bright is the ball's own light,
    /// however sharp it reads (a glass ornament's highlight at its centre).
    pub core: f64,
    /// Bounding radius of everything drawn, in pixels.
    pub reach: f64,
    /// The ball's plane fit: value at the centre and slopes per pixel, per channel.
    pub fit: [[f64; 3]; 3],
    /// The earlier balls (brighter ones, which own the pixels two discs share) whose disc overlaps this ball's: the
    /// offset of their centre from this ball's, their disc's radius (0 for an empty place) and the width of their
    /// weight's fall ([`Shape::fade`]).
    pub earlier: [(f64, f64, f64, f64); OWNERS],
}

/// What one tile pass needs to shape the balls.
#[derive(Clone, Copy, Debug)]
pub(super) struct ShapeParams<'a> {
    /// The frame at this scale and its half-diagonal in pixels.
    pub fw: f64,
    pub fh: f64,
    pub hk: f64,
    /// Bokeh height over width.
    pub q: f64,
    /// The amount of the lens's own swirl and cat's-eye, 0 to 1: Anamorphic / 100.
    pub amount: f64,
    /// True when the anamorphic axis is the buffer's x axis (so the cross direction is y).
    pub axis_x: bool,
    /// Lens field radii, swirl, astigmatism and cat's-eye profiles (9 entries each).
    pub radii: &'a [f64],
    pub swirl: &'a [f64],
    pub astigmatism: &'a [f64],
    pub cats_eye: &'a [f64],
    /// The lens's rim profile (16 samples).
    pub rim: &'a [f64],
    /// Mean of the rim profile over a disc (area-weighted).
    pub rim_mean: f64,
}

/// Linear interpolation of a profile given at `radii`, held at the last value beyond the last radius.
pub(super) fn profile_at(radii: &[f64], values: &[f64], rho: f64) -> f64 {
    if rho <= radii[0] {
        return values[0];
    }
    for i in 1..radii.len() {
        if rho <= radii[i] {
            let t = (rho - radii[i - 1]) / (radii[i] - radii[i - 1]);
            return values[i - 1] + t * (values[i] - values[i - 1]);
        }
    }
    values[values.len() - 1]
}

/// The rim profile at normalised radius `s` (0 the centre, 1 the edge), linear between its 16 samples.
#[inline]
pub(super) fn rim_at(rim: &[f64], s: f64) -> f64 {
    let x = s.clamp(0.0, 1.0) * (RIM_SAMPLES - 1) as f64;
    let i = (x as usize).min(RIM_SAMPLES - 2);
    let t = x - i as f64;
    rim[i] + t * (rim[i + 1] - rim[i])
}

/// The area-weighted mean of the rim profile over a disc: `integral prof(s) 2 s ds`, exact for the piecewise
/// linear profile (two-point Gauss on every segment).
pub(super) fn rim_disc_mean(rim: &[f64]) -> f64 {
    let n = RIM_SAMPLES - 1;
    let g = 0.5 / 3f64.sqrt();
    let mut sum = 0.0;
    for i in 0..n {
        let (s0, s1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
        let mid = 0.5 * (s0 + s1);
        let half = 0.5 * (s1 - s0);
        for sign in [-1.0, 1.0] {
            let s = mid + sign * 2.0 * g * half;
            let p = rim[i] + (rim[i + 1] - rim[i]) * (s - s0) / (s1 - s0);
            sum += half * 2.0 * s * p;
        }
    }
    sum
}

/// The factor the lens's rim profile puts on the ball's light at `s` radii from its centre: the profile over its
/// mean across the ball, fading to 1 over [`RIM_FADE_RADII`] radii beyond its edge, by `strength` (0 to 1: how much
/// of the profile this ball takes, [`Shape::rim_k`]).
#[inline]
pub(super) fn rim_gain(rim: &[f64], mean: f64, s: f64, strength: f64) -> f64 {
    let g = 1.0 + strength * (rim_at(rim, s) / mean - 1.0);
    let t = ((s - 1.0) / RIM_FADE_RADII).clamp(0.0, 1.0);
    let fade = t * t * (3.0 - 2.0 * t);
    g + fade * (1.0 - g)
}

/// The share of the lens's rim profile a ball whose rim is `softness` radii wide takes: 1 for a crisp disc, 0 for
/// a soft glow.
#[inline]
pub(super) fn rim_strength(softness: f64) -> f64 {
    let t = ((softness - RIM_SOFT_FROM) / (RIM_SOFT_TO - RIM_SOFT_FROM)).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// The aspect of a disc blurred along a line of box half-length `sigma e` radii: its second moment over the
/// disc's, `sqrt(1 + 4 (sigma e)^2 / 3)`, which is 2 at `sigma e = 1.5`.
#[inline]
pub(super) fn tangential_aspect(sigma_e: f64) -> f64 {
    (1.0 + 4.0 / 3.0 * sigma_e * sigma_e).sqrt()
}

/// The half-length, in pixels, of the biweight line of the tangential gather at a pixel of blur `blur_px` where
/// the lens has the swirl `sigma` and the astigmatism `astig` (half-diagonals): `A (k sigma b + a H)` with `A` the
/// amount (Anamorphic / 100), `k` = [`PROFILE_SCALE`] and `H` = `hk`, the half-diagonal in pixels, held to
/// [`TANGENT_REACH_MAX`] of it (`docs/21` section 4, L1, *As built (field swirl)*). The first term grows with the
/// blur, the second does not depend on it: the tangential and sagittal focal surfaces part with the field radius
/// whatever the defocus.
#[inline]
pub(super) fn tangent_half_length(
    amount: f64,
    sigma: f64,
    astig: f64,
    blur_px: f64,
    hk: f64,
) -> f64 {
    (amount * (PROFILE_SCALE * sigma * blur_px + astig * hk)).min(TANGENT_REACH_MAX * hk)
}

/// The patch of `ball` for this view, or `None` when the lens leaves the ball as it is.
pub(super) fn shape_of(ball: &Ball, p: &ShapeParams) -> Option<Shape> {
    let r_px = ball.radius * p.hk;
    if r_px < 0.25 {
        return None;
    }
    let cx = ball.x * p.fw;
    let cy = ball.y * p.fh;
    // Field position of the ball: radial unit vector and field radius.
    let (ux, uy) = (cx - 0.5 * p.fw, cy - 0.5 * p.fh);
    let dist = (ux * ux + uy * uy).sqrt();
    let rho = dist / p.hk;
    let (rx, ry) = if dist > 1e-9 {
        (ux / dist, uy / dist)
    } else {
        (1.0, 0.0)
    };
    let (tx, ty) = (-ry, rx);
    let sigma = profile_at(p.radii, p.swirl, rho);
    let astig = profile_at(p.radii, p.astigmatism, rho);
    let cat = profile_at(p.radii, p.cats_eye, rho);
    // The stretch across the anamorphic axis and the swirl along the tangent: the aspect of a disc blurred along
    // the gather's line (the box line of its spread), a lemon of at most `TANGENT_ASPECT_MAX` to one.
    let (yx, yy) = if p.axis_x { (0.0, 1.0) } else { (1.0, 0.0) };
    let v = (p.q - 1.0) * r_px;
    let box_half = tangent_half_length(p.amount, sigma, astig, r_px, p.hk) / PROFILE_SCALE;
    let q_t = tangential_aspect(box_half / r_px).min(TANGENT_ASPECT_MAX);
    let t = (q_t - 1.0) * r_px;
    let (vx, vy) = (v * yx, v * yy);
    let mut sgn = 1.0;
    if vx * tx + vy * ty < 0.0 {
        sgn = -1.0;
    }
    let (sx, sy) = (vx + sgn * t * tx, vy + sgn * t * ty);
    let s_len = (sx * sx + sy * sy).sqrt();
    let off = cat * p.amount * r_px;
    let stretched = s_len > 0.02 * r_px;
    let cut = off > 0.02 * r_px;
    if !stretched && !cut {
        return None;
    }
    let r_out = ball.take_out_radius * p.hk;
    // The disc read back is the take-out disc, `kappa` times the ball: the patch is the ball's shape scaled by it.
    let kappa = r_out / r_px;
    let f = (r_px + s_len) / r_px;
    let (a, b) = (f * r_out, r_out);
    let m = if stretched {
        (sx / s_len, sy / s_len)
    } else {
        (1.0, 0.0)
    };
    // The cat's-eye copies sit half the offset either side of the centre along the radial direction.
    let (half_off, delta) = if cut {
        let h = (0.5 * off * rx * kappa, 0.5 * off * ry * kappa);
        // The offset vector in the normalised frame (long axis along m, short axis across).
        let n = (-m.1, m.0);
        let full = (2.0 * h.0, 2.0 * h.1);
        let ex = (full.0 * m.0 + full.1 * m.1) / a;
        let ey = (full.0 * n.0 + full.1 * n.1) / b;
        let len = (ex * ex + ey * ey).sqrt();
        // Two copies further apart than this leave almost nothing: hold the cut at 1.8 semi-axes.
        let keep = (MAX_CAT_OFFSET / len.max(1e-12)).min(1.0);
        ((h.0 * keep, h.1 * keep), len * keep)
    } else {
        ((0.0, 0.0), 0.0)
    };
    let edge = MASK_EDGE_MIN_PX.max(0.5 * (r_out - r_px));
    Some(Shape {
        cx,
        cy,
        a,
        b,
        f,
        m,
        half_off,
        delta,
        edge,
        r_out,
        r_ball: r_px,
        rim_k: rim_strength(ball.softness),
        core: 0.5 * ball.peak,
        reach: a + (half_off.0 * half_off.0 + half_off.1 * half_off.1).sqrt() + 0.5 * edge + 1.5,
        fit: [
            fit_px(&ball.fit[0], p.hk),
            fit_px(&ball.fit[1], p.hk),
            fit_px(&ball.fit[2], p.hk),
        ],
        earlier: [(0.0, 0.0, 0.0, 0.0); OWNERS],
    })
}

/// Fills in which earlier balls overlap each ball's disc (`shapes` in list order): the light of a place two discs
/// share is divided between them by the earlier one's weight (the earlier is the brighter, and keeps the part
/// nearer its centre).
pub(super) fn assign_owners(shapes: &mut [Shape]) {
    for i in 0..shapes.len() {
        let mut found = 0;
        let mut earlier = [(0.0, 0.0, 0.0, 0.0); OWNERS];
        for j in 0..i {
            let (dx, dy) = (shapes[j].cx - shapes[i].cx, shapes[j].cy - shapes[i].cy);
            if dx.hypot(dy) < shapes[i].r_out + shapes[j].r_out && found < OWNERS {
                earlier[found] = (dx, dy, shapes[j].r_out, shapes[j].fade());
                found += 1;
            }
        }
        shapes[i].earlier = earlier;
    }
}

/// The smoothstep of `t` between 0 and 1: a weight falling from 1 to 0 as `t` goes from 1 to 0.
#[inline]
fn fall(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A plane fit given per half-diagonal, as value at the centre and slope per pixel.
fn fit_px(f: &[f64; 3], hk: f64) -> [f64; 3] {
    [f[0], f[1] / hk, f[2] / hk]
}

/// Where a pixel of the stretched picture reads the ball's light, and how much of it counts.
#[derive(Clone, Copy, Debug)]
pub(super) struct Source {
    /// The offset from the ball's centre of the position the pixel has in the ball before the stretch.
    pub dx: f64,
    pub dy: f64,
    /// The weight of the disc's edge: 1 inside, a ramp of one pixel at the edge.
    pub alpha: f64,
}

impl Shape {
    /// The position in the ball that the pixel centre `(px, py)` of the stretched picture reads, or `None` outside the
    /// ball's disc or where an earlier ball has all of the light.
    #[inline]
    pub(super) fn source(&self, px: f64, py: f64) -> Option<Source> {
        let (dx, dy) = (px - self.cx, py - self.cy);
        let u = (dx * self.m.0 + dy * self.m.1) / self.f;
        let v = -dx * self.m.1 + dy * self.m.0;
        let alpha = self.weight(u.hypot(v));
        if alpha <= 0.0 {
            return None;
        }
        let (sx, sy) = (u * self.m.0 - v * self.m.1, u * self.m.1 + v * self.m.0);
        // The light of a place an earlier ball's disc covers is shared: that ball has its weight there, this one the
        // rest (no edge where the discs meet, and no light drawn twice).
        let mut share = alpha;
        for &(ox, oy, r, fade) in &self.earlier {
            if r > 0.0 {
                share *= 1.0 - fall((r + 0.5 - (sx - ox).hypot(sy - oy)) / fade);
            }
        }
        if share <= 0.0 {
            return None;
        }
        Some(Source {
            dx: sx,
            dy: sy,
            alpha: share,
        })
    }

    /// The weight of the ball's light and of its take-out at `d` pixels from its centre: 1 over the ball, falling to 0 at
    /// the take-out radius (half a pixel beyond it) over [`BALL_GLOW_FADE`] of the glow's width, at least a pixel.
    #[inline]
    pub(super) fn weight(&self, d: f64) -> f64 {
        fall((self.r_out + 0.5 - d) / self.fade())
    }

    /// The width in pixels over which [`Shape::weight`] falls from 1 to 0: [`BALL_GLOW_FADE`] of the glow's width, at
    /// least a pixel, and half a pixel more.
    #[inline]
    pub(super) fn fade(&self) -> f64 {
        (BALL_GLOW_FADE * (self.r_out - self.r_ball)).max(1.0) + 0.5
    }

    /// The ball's plane fit of channel `c` at the offset `(dx, dy)` from its centre, as the take-out reads it.
    #[inline]
    pub(super) fn fit_at(&self, c: usize, dx: f64, dy: f64) -> f64 {
        let f = &self.fit[c];
        (f[0] + f[1] * dx + f[2] * dy).max(0.0)
    }

    /// Coverage (0 to 1; a soft edge `edge` pixels wide) of the patch's cat's-eye mask and the normalised radius at the
    /// pixel centre `(px, py)`: the patch's ellipse, or the intersection of two copies of it. Without a cat's-eye it is
    /// the ellipse alone, which is where the patch can be.
    #[inline]
    pub(super) fn coverage(&self, px: f64, py: f64) -> (f64, f64) {
        let ell = |ox: f64, oy: f64| -> (f64, f64) {
            let (dx, dy) = (px - self.cx - ox, py - self.cy - oy);
            let u = dx * self.m.0 + dy * self.m.1;
            let v = -dx * self.m.1 + dy * self.m.0;
            let g = (u / self.a) * (u / self.a) + (v / self.b) * (v / self.b);
            if g < 0.0625 && self.edge <= 1.0 {
                // Deep inside: the signed distance is at least a quarter of the smaller semi-axis.
                return (-self.b * 0.5, g.sqrt());
            }
            let gx = 2.0 * u / (self.a * self.a);
            let gy = 2.0 * v / (self.b * self.b);
            let grad = (gx * gx + gy * gy).sqrt().max(1e-12);
            ((g - 1.0) / grad, g.sqrt())
        };
        let cover = |sd: f64| -> f64 {
            let t = (0.5 - sd / self.edge).clamp(0.0, 1.0);
            if self.edge > 1.0 {
                t * t * (3.0 - 2.0 * t)
            } else {
                t
            }
        };
        if self.delta == 0.0 {
            let (sd, rn) = ell(0.0, 0.0);
            return (cover(sd), rn);
        }
        let (sd1, r1) = ell(-self.half_off.0, -self.half_off.1);
        let (sd2, r2) = ell(self.half_off.0, self.half_off.1);
        let sd = sd1.max(sd2);
        (cover(sd), r1.max(r2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_disc_mean_of_a_flat_rim_is_one_and_of_a_linear_one_is_two_thirds() {
        let flat = vec![1.0; RIM_SAMPLES];
        assert!((rim_disc_mean(&flat) - 1.0).abs() < 1e-12);
        let ramp: Vec<f64> = (0..RIM_SAMPLES)
            .map(|i| i as f64 / (RIM_SAMPLES - 1) as f64)
            .collect();
        // integral s * 2 s ds = 2/3
        assert!((rim_disc_mean(&ramp) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_flat_rim_profile_changes_nothing_and_the_gain_fades_out_beyond_the_edge() {
        let flat = vec![1.0; RIM_SAMPLES];
        for s in [0.0, 0.5, 1.0, 1.2, 2.0] {
            assert!((rim_gain(&flat, 1.0, s, 1.0) - 1.0).abs() < 1e-12);
        }
        let mut bright_edge = vec![1.0; RIM_SAMPLES];
        bright_edge[RIM_SAMPLES - 1] = 3.0;
        let mean = rim_disc_mean(&bright_edge);
        assert!(rim_gain(&bright_edge, mean, 1.0, 1.0) > 2.5);
        assert!((rim_gain(&bright_edge, mean, 1.0 + RIM_FADE_RADII, 1.0) - 1.0).abs() < 1e-12);
        assert!((rim_gain(&bright_edge, mean, 3.0, 1.0) - 1.0).abs() < 1e-12);
        // A soft ball takes none of the profile, a ball in between a part.
        assert!((rim_gain(&bright_edge, mean, 1.0, 0.0) - 1.0).abs() < 1e-12);
        let half = rim_gain(&bright_edge, mean, 1.0, 0.5);
        assert!(half > 1.5 && half < rim_gain(&bright_edge, mean, 1.0, 1.0));
        assert_eq!(rim_strength(0.0), 1.0);
        assert_eq!(rim_strength(RIM_SOFT_FROM), 1.0);
        assert_eq!(rim_strength(RIM_SOFT_TO), 0.0);
        assert_eq!(rim_strength(1.3), 0.0);
    }

    #[test]
    fn the_tangential_aspect_is_two_at_one_and_a_half_radii_and_one_at_none() {
        assert!((tangential_aspect(0.0) - 1.0).abs() < 1e-12);
        assert!((tangential_aspect(1.5) - 2.0).abs() < 1e-12);
        assert!(tangential_aspect(0.5) > 1.0 && tangential_aspect(0.5) < tangential_aspect(1.0));
    }

    /// The profiles of a lens for the shape tests: nine radii, a swirl, an astigmatism, no cat's-eye and a flat rim.
    struct Tables {
        radii: Vec<f64>,
        swirl: Vec<f64>,
        astigmatism: Vec<f64>,
        cats_eye: Vec<f64>,
        rim: Vec<f64>,
    }

    fn tables(swirl: [f64; 9], astigmatism: [f64; 9]) -> Tables {
        Tables {
            radii: (0..9).map(|i| i as f64 / 8.0).collect(),
            swirl: swirl.to_vec(),
            astigmatism: astigmatism.to_vec(),
            cats_eye: vec![0.0; 9],
            rim: vec![1.0; RIM_SAMPLES],
        }
    }

    /// The half-diagonal of the 1500 x 1000 frame of these tests, in pixels.
    fn half_diagonal() -> f64 {
        750.0f64.hypot(500.0)
    }

    fn params(t: &Tables, amount: f64) -> ShapeParams<'_> {
        ShapeParams {
            fw: 1500.0,
            fh: 1000.0,
            hk: half_diagonal(),
            q: 1.0,
            amount,
            axis_x: true,
            radii: &t.radii,
            swirl: &t.swirl,
            astigmatism: &t.astigmatism,
            cats_eye: &t.cats_eye,
            rim: &t.rim,
            rim_mean: 1.0,
        }
    }

    /// A ball of `r_px` pixels at `rho` half-diagonals from the centre of the 1500 x 1000 frame, on the diagonal.
    fn ball_on_diagonal(rho: f64, r_px: f64) -> Ball {
        let hk = half_diagonal();
        let (ux, uy) = (1500.0 / (2.0 * hk), 1000.0 / (2.0 * hk));
        Ball {
            x: 0.5 + rho * hk * ux / 1500.0,
            y: 0.5 + rho * hk * uy / 1000.0,
            radius: r_px / hk,
            take_out_radius: 1.5 * r_px / hk,
            energy: [1.0; 3],
            fit: [[0.0; 3]; 3],
            peak: 1.0,
            supported: true,
            softness: 0.2,
            clear: true,
        }
    }

    #[test]
    fn the_tangent_line_is_the_swirl_plus_the_astigmatism_held_to_the_reach() {
        let hk = 900.0;
        // Nothing without either term; the swirl grows with the blur, the astigmatism does not.
        assert_eq!(tangent_half_length(1.0, 0.0, 0.0, 30.0, hk), 0.0);
        let swirl = tangent_half_length(1.0, 1.2, 0.0, 20.0, hk);
        assert!(
            (swirl - PROFILE_SCALE * 1.2 * 20.0).abs() < 1e-12,
            "{swirl}"
        );
        let flat = tangent_half_length(1.0, 0.0, 0.03, 20.0, hk);
        assert!((flat - 27.0).abs() < 1e-12, "{flat}");
        assert_eq!(flat, tangent_half_length(1.0, 0.0, 0.03, 5.0, hk));
        // Together they add, and the amount scales both.
        let both = tangent_half_length(1.0, 1.2, 0.03, 20.0, hk);
        assert!((both - (swirl + flat)).abs() < 1e-12);
        let half = tangent_half_length(0.5, 1.2, 0.03, 20.0, hk);
        assert!((half - 0.5 * both).abs() < 1e-12);
        // Never longer than the reach.
        let cap = TANGENT_REACH_MAX * hk;
        assert_eq!(tangent_half_length(1.0, 2.9, 0.03, 27.0, hk), cap);
        assert!(tangent_half_length(1.0, 0.1, 0.01, 10.0, hk) < cap);
    }

    #[test]
    fn a_small_ball_of_the_outer_field_is_a_lemon_by_the_formula_and_never_a_line() {
        let mut astig = [0.0; 9];
        for (i, a) in astig.iter_mut().enumerate().skip(3) {
            *a = 0.03 * ((i as f64 / 8.0 - 0.2) / 0.8).powf(1.64);
        }
        let t = tables([0.0; 9], astig);
        let p = params(&t, 1.0);
        let hk = p.hk;
        for (rho, r_px) in [(0.9, 6.0), (0.9, 12.0), (0.9, 24.0), (0.6, 10.0)] {
            let s = shape_of(&ball_on_diagonal(rho, r_px), &p).expect("a lemon");
            // `a(rho)` read between the nine radii as the gather reads it.
            let a = profile_at(&t.radii, &t.astigmatism, rho);
            let box_half = a * hk / PROFILE_SCALE;
            let want = tangential_aspect(box_half / r_px).min(TANGENT_ASPECT_MAX);
            assert!(
                (s.f / want - 1.0).abs() < 1e-9,
                "rho {rho}, r {r_px}: stretch {} against {want}",
                s.f
            );
            // The long axis is the tangent: perpendicular to the radius (the diagonal here).
            let (dx, dy) = (1500.0 / (2.0 * hk), 1000.0 / (2.0 * hk));
            assert!((s.m.0 * dx + s.m.1 * dy).abs() < 1e-6, "{:?}", s.m);
        }
        // A tiny ball would be a line: held at the cap. One at the centre stays round.
        let tiny = shape_of(&ball_on_diagonal(0.95, 1.0), &p).expect("a lemon");
        assert!((tiny.f - TANGENT_ASPECT_MAX).abs() < 1e-9, "{}", tiny.f);
        assert!(shape_of(&ball_on_diagonal(0.1, 12.0), &p).is_none());
        // An amount of 0 (Anamorphic 0) leaves every ball round.
        assert!(shape_of(&ball_on_diagonal(0.9, 8.0), &params(&t, 0.0)).is_none());
    }

    fn patch(f: f64, earlier: [(f64, f64, f64, f64); OWNERS]) -> Shape {
        Shape {
            cx: 100.0,
            cy: 100.0,
            a: f * 20.0,
            b: 20.0,
            f,
            m: (0.0, 1.0),
            half_off: (0.0, 0.0),
            delta: 0.0,
            edge: 1.5,
            r_out: 20.0,
            r_ball: 14.0,
            rim_k: 1.0,
            core: 0.0,
            reach: f * 20.0 + 3.0,
            fit: [[0.0; 3]; 3],
            earlier,
        }
    }

    #[test]
    fn a_pixel_reads_the_ball_where_it_was_before_the_stretch() {
        // Stretched twice along y: the pixel 20 px above the centre reads the ball 10 px above it, the one across stays.
        let s = patch(2.0, [(0.0, 0.0, 0.0, 0.0); OWNERS]);
        let at = s.source(100.0 + 7.0, 100.0 - 20.0).expect("inside");
        assert!(
            (at.dx - 7.0).abs() < 1e-12 && (at.dy + 10.0).abs() < 1e-12,
            "{at:?}"
        );
        assert!((at.alpha - 1.0).abs() < 1e-12);
        // The disc's edge is a ramp of a pixel, and beyond it nothing is read: 2 x 20 = 40 px along the axis.
        assert!(
            s.source(100.0, 100.0 - 40.2)
                .is_some_and(|a| a.alpha > 0.0 && a.alpha < 1.0)
        );
        assert!(s.source(100.0, 100.0 - 41.0).is_none());
        assert!(s.source(100.0 + 21.0, 100.0).is_none());
    }

    #[test]
    fn the_light_two_discs_share_is_divided_between_them_without_an_edge() {
        // The earlier ball's disc (radius 20, centre 15 px to the right) overlaps this one's (radius 20).
        let mut earlier = [(0.0, 0.0, 0.0, 0.0); OWNERS];
        let other = patch(1.0, [(0.0, 0.0, 0.0, 0.0); OWNERS]);
        earlier[0] = (15.0, 0.0, 20.0, other.fade());
        let s = patch(1.0, earlier);
        let alone = patch(1.0, [(0.0, 0.0, 0.0, 0.0); OWNERS]);
        // Away from the earlier ball's disc nothing changes; deep inside it the later ball has none.
        let at = |s: &Shape, x: f64| s.source(100.0 + x, 100.0).map_or(0.0, |a| a.alpha);
        assert!((at(&s, -10.0) - at(&alone, -10.0)).abs() < 1e-12);
        assert!(at(&s, 10.0) < 1e-9, "deep inside the earlier ball's disc");
        // The two shares add up to this ball's weight alone, and fall step by step, never jump.
        let mut prev = at(&s, -10.0);
        let mut x = -10.0;
        while x < 18.0 {
            x += 0.25;
            let a = at(&s, x);
            assert!(a <= prev + 1e-12 && prev - a < 0.2, "x {x}: {prev} -> {a}");
            let earlier_weight = other.weight((x - 15.0f64).abs());
            if (at(&alone, x) - 1.0).abs() < 1e-12 {
                assert!((a - (1.0 - earlier_weight)).abs() < 1e-12, "x {x}");
            }
            prev = a;
        }
    }
}
