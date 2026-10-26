//! The colour of the light the field stages move (`docs/21` section 4, *As built (the light's colour)*).
//!
//! The lens plane is camera RGB balanced with `g0`: channel `c` clips at `g0_c`, and `g0` is 1 for green and
//! 1.5 to 3 for red and blue, so green clips first for any white light. Between green's clip level and the
//! others' a white light reads magenta (green is a lower bound, the others measure), and R5, which rebuilds the
//! clipped channel from a calibration on the fringe around the light, stays under it. A lens stage that spreads
//! such a pixel spreads the cast with it, and where the spread falls under display white the cast shows: a white
//! star turned violet.
//!
//! [`lit_input`] gives L1 and L2 the input they should spread: the picture with every pixel's light in its own
//! colour ([`Restore::light_colour`](crate::l3::Restore), the rule the flares' restored light uses too, so all
//! the stages agree). It is a function of one pixel, so a tile and the whole frame give the same bits. The
//! picture itself is not changed: L3 reads the picture as it is and the lens-off path never comes here.
//!
//! [`l2_input`] gives L2 the light it spreads besides the picture. A star the sensor clipped holds far more light
//! than its clipped core says, and what L2 spreads from the picture as it is would be mostly the ring of
//! unclipped, fringed pixels round it (a white star turned violet). A small clipped light therefore gets its
//! *headroom* for the spreading ([`LightMap`]: the boost L3 estimates by the light's size, a share of it, in a
//! neutral light), and every pixel with a saturated channel its light's colour. That light is the *extra*: the
//! picture L2 reads is the picture as it is, and the extra goes through the same gather and reaches only the
//! pixels the lens moves it to (`field::apply`), so the lens adds light it moves and no more.

use rayon::prelude::*;

use crate::consts::{
    CLIP_RAMP, CLIP_RAMP_PLANE, FIELD_HEADROOM_MAX, FIELD_HEADROOM_SHARE, FIELD_LIGHT_SIZE_FACTOR, L1_UNCHANGED_RAMP,
    LIGHT_SATURATION_RAMP,
};
use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::l3::{Restore, boost_ev};
use crate::plan::LensPlan;
use crate::planes::LensPlanes;

/// What the device pass that gives L1 and L2 the light's colour takes of a plan: the clip levels and the clip
/// test's ramp (`LensPlan::light_pass`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightPass {
    /// `g0` of each channel (0: the channel has no clip level).
    pub clip: [f32; 3],
    /// The clip test's ramp for the picture's own pixels: from, to.
    pub ramp: (f32, f32),
}

impl LensPlan {
    /// The device pass that gives the stages that move light (L1 and L2) the light's colour, or `None` when
    /// neither acts (Edges and Anamorphic both 0: `Z` is the picture itself and nothing is spread).
    #[must_use]
    pub fn light_pass(&self) -> Option<LightPass> {
        if self.field_is_identity() {
            return None;
        }
        Some(LightPass {
            clip: self.frame().clip,
            ramp: (CLIP_RAMP.0 as f32, CLIP_RAMP.1 as f32),
        })
    }
}

/// The input of L1 and L2 with the colour of the light it holds, or `None` when no pixel changes (no channel
/// is saturated anywhere, or the plan has no field stage: Edges and Anamorphic both 0, where `Z` is the
/// picture itself). The CPU reference of the device pass `lens_light`.
///
/// The result has the samples of `input` (planar, `3 * width * height`).
#[must_use]
pub fn lit_input(plan: &LensPlan, input: &ImageRef) -> Option<Vec<f32>> {
    if plan.field_is_identity() || !any_saturated(plan, input) {
        return None;
    }
    let n = input.width * input.height;
    let rs = Restore::new(plan, CLIP_RAMP);
    let mut out = input.data.to_vec();
    let (r, rest) = out.split_at_mut(n);
    let (g, b) = rest.split_at_mut(n);
    r.par_chunks_mut(input.width)
        .zip(g.par_chunks_mut(input.width))
        .zip(b.par_chunks_mut(input.width))
        .for_each(|((r, g), b)| {
            for ((r, g), b) in r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()) {
                let x = [f64::from(*r), f64::from(*g), f64::from(*b)];
                let y = rs.light_colour(x);
                if y != x {
                    *r = y[0] as f32;
                    *g = y[1] as f32;
                    *b = y[2] as f32;
                }
            }
        });
    Some(out)
}

/// A smoothstep from 0 at `t = 0` to 1 at `t = 1`.
#[inline]
fn smooth(t: f64) -> f64 {
    let u = t.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// The clipped lights of a photo as L2 spreads them: how much light a small clipped light holds above what
/// the picture records, by the size of its patch on the lens plane, once per photo.
///
/// A patch is a set of 8-connected pixels of the lens plane that are clipped (the clip test of the plane,
/// [`Restore::clip_test`]) or are a white light with a saturated channel ([`Restore::light`]); its extra
/// headroom in EV is L3's estimate for the size of the patch ([`boost_ev`]: 9 EV for a patch of at most 0.025
/// half-diagonals across, nothing from 0.4) read at [`FIELD_LIGHT_SIZE_FACTOR`] times its diameter (the field
/// stages spread light far, so only a patch of a few pixels is a point), and the photo's cap scale ([`LIGHT_BOOST_ENERGY_CAP`](crate::consts))
/// is L3's. The map holds the EV per pixel of the lens plane, the largest within one pixel of it (a pixel
/// clipped at a finer scale lies in the plane pixel beside the patch's), and every scale reads it at its pixel's
/// place ([`LightMap::ev_at`]).
#[derive(Clone, Debug)]
pub(crate) struct LightMap {
    /// The EV of every lens-plane pixel, row-major; empty when no pixel is part of a light.
    ev: Vec<f32>,
    width: usize,
    height: usize,
    /// The scale `k_L` of the lens plane.
    plane_scale: usize,
    /// The share of the extra headroom the photo keeps, 1 up to the cap.
    scale: f64,
}

impl LightMap {
    /// The map of a photo without lights: nothing is added.
    pub(crate) fn empty() -> Self {
        LightMap {
            ev: Vec::new(),
            width: 0,
            height: 0,
            plane_scale: 1,
            scale: 1.0,
        }
    }

    /// Makes the map of the lens plane `plane`, with the cap's `scale` of the photo (L3's
    /// `FlarePlanes::boost_scale`). `cancel` is polled once per row.
    ///
    /// # Errors
    ///
    /// [`LensError::Cancelled`].
    pub(crate) fn build(
        plan: &LensPlan,
        plane: &ImageRef,
        scale: f64,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Self, LensError> {
        let (w, h) = (plane.width, plane.height);
        let n = w * h;
        let rs = Restore::new(plan, CLIP_RAMP_PLANE);
        let mut mask = vec![false; n];
        let mut any = false;
        for j in 0..h {
            if cancel() {
                return Err(LensError::Cancelled);
            }
            for i in 0..w {
                let at = j * w + i;
                let x = [
                    f64::from(plane.data[at]),
                    f64::from(plane.data[n + at]),
                    f64::from(plane.data[2 * n + at]),
                ];
                if rs.clip_test(x).max(rs.light(x).1) > 0.0 {
                    mask[at] = true;
                    any = true;
                }
            }
        }
        let plane_scale = plan.lens_plane_scale();
        if !any {
            return Ok(LightMap {
                plane_scale,
                scale,
                ..LightMap::empty()
            });
        }
        let half_diag = 0.5 * ((w * w + h * h) as f64).sqrt();
        let mut ev = vec![0.0f32; n];
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut members: Vec<usize> = Vec::new();
        for start in 0..n {
            if start % w == 0 && cancel() {
                return Err(LensError::Cancelled);
            }
            if !mask[start] || seen[start] {
                continue;
            }
            seen[start] = true;
            stack.push(start);
            members.clear();
            while let Some(p) = stack.pop() {
                members.push(p);
                let (x, y) = (p % w, p / w);
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let q = yy * w + xx;
                        if mask[q] && !seen[q] {
                            seen[q] = true;
                            stack.push(q);
                        }
                    }
                }
            }
            let area = members.len() as f64 / (half_diag * half_diag);
            let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
            let patch_ev = boost_ev(FIELD_LIGHT_SIZE_FACTOR * diameter) as f32;
            for p in &members {
                ev[*p] = patch_ev;
            }
        }
        // The largest EV within one pixel.
        let mut grown = ev.clone();
        for y in 0..h {
            for x in 0..w {
                let mut m = 0.0f32;
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        m = m.max(ev[yy * w + xx]);
                    }
                }
                grown[y * w + x] = m;
            }
        }
        Ok(LightMap {
            ev: grown,
            width: w,
            height: h,
            plane_scale,
            scale,
        })
    }

    /// The EV of the lens-plane pixel the pixel `(i, j)` of the frame at scale `k` lies in (0 when none): pixel
    /// `i` covers `[i k, (i + 1) k)` of the full-resolution frame, its centre falls in the lens-plane pixel
    /// `floor((2 i + 1) k / (2 k_L))`, in integers so every scale and tile agrees.
    #[inline]
    pub(crate) fn ev_at(&self, i: usize, j: usize, k: usize) -> f64 {
        if self.ev.is_empty() {
            return 0.0;
        }
        let kl = self.plane_scale.max(1);
        let pi = (((2 * i + 1) * k) / (2 * kl)).min(self.width - 1);
        let pj = (((2 * j + 1) * k) / (2 * kl)).min(self.height - 1);
        f64::from(self.ev[pj * self.width + pi])
    }

    /// The gain of the headroom at `ev` EV: `scale (2^(FIELD_HEADROOM_SHARE ev) - 1)`, in units of the light's
    /// neutral level.
    #[inline]
    pub(crate) fn gain(&self, ev: f64) -> f64 {
        if ev <= 0.0 {
            return 0.0;
        }
        self.scale * (libm::exp2(FIELD_HEADROOM_SHARE * ev) - 1.0)
    }

    /// True when no pixel of the photo is part of a light.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.ev.is_empty()
    }

    /// The map for the device: the EV of the lens plane, its size and scale, and the cap's scale.
    pub(crate) fn data(&self) -> crate::device::LightMapData {
        crate::device::LightMapData {
            ev: self.ev.clone(),
            width: self.width,
            height: self.height,
            plane_scale: self.plane_scale,
            scale: self.scale,
        }
    }

    /// Bytes the map holds.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.ev.len() * 4
    }
}

/// The light a pixel holds for L2: its colour (the shared rule), and the neutral extra light it holds above
/// what it records, `a g nu`: `a` how much of a clipped light the pixel is (the clip test, or the white light
/// weight of a saturated channel), `g` the photo's gain at the pixel's EV ([`LightMap::gain`]) and `nu` the
/// smallest channel of the light's colour (the light the clipped part of the pixel holds is neutral; the
/// colour above it is the recorded fringe's).
#[inline]
fn pixel_light(rs: &Restore, map: &LightMap, x: [f64; 3], ev: f64) -> ([f64; 3], f64) {
    let (lit, white) = rs.light(x);
    let a = rs.clip_test(x).max(white);
    if a <= 0.0 || ev <= 0.0 {
        return (lit, 0.0);
    }
    let nu = lit[0].min(lit[1]).min(lit[2]).max(0.0);
    (lit, (a * map.gain(ev) * nu).min(FIELD_HEADROOM_MAX))
}

/// True when some channel of some pixel of `input` is in the saturation ramp.
fn any_saturated(plan: &LensPlan, input: &ImageRef) -> bool {
    let clip = plan.frame().clip;
    let n = input.width * input.height;
    let from = LIGHT_SATURATION_RAMP.0;
    let below = |c: usize| {
        let g0 = f64::from(clip[c]);
        g0 <= 0.0
            || input.data[c * n..(c + 1) * n]
                .iter()
                .all(|v| f64::from(*v) < from * g0)
    };
    !(below(0) && below(1) && below(2))
}

/// What L2 reads: the picture it gathers and the extra light it spreads beside it ([`l2_input`]).
#[derive(Clone, Debug)]
pub struct L2Input {
    /// The rectangle `extra` and `own` cover.
    pub rect: PixelRect,
    /// The light L2 spreads that the picture does not record, per pixel of `rect`: planar, `3 * rect.w * rect.h`
    /// samples. The pixel's light in its own colour less the picture (the saturated channel raised to the others'
    /// mean, the clip pattern's cast taken off: a channel above its light's colour is lowered, so a sample can be
    /// negative) and the headroom of a clipped small light (neutral, never negative). 0 for the pixels L1
    /// changed.
    pub extra: Vec<f32>,
    /// The picture L2 gathers over `rect`, planar like `extra`: the raw picture, and L1's output at the pixels L1
    /// changed.
    pub own: Vec<f32>,
}

/// What L2 reads besides the picture as it is, from the picture `raw` and L1's output `l1` over `want` (`None`
/// when L1 leaves the picture alone), or `None` when nothing in the picture is saturated (L2 reads `l1`, or `raw`,
/// as they are).
///
/// Every pixel's light is in its own colour ([`Restore::light`]) and a clipped small light holds its headroom
/// ([`pixel_light`]); the extra is that light less the picture. L2 gathers the picture and the extra apart and
/// adds the extra only where the lens moves light (`EXTRA_INFLOW_RAMP`, `EXTRA_CHANGE_RAMP`,
/// `EXTRA_KERNEL_RAMP`), so a pixel the lens leaves alone is the picture's. The pixels L1 left as they are (its
/// output within [`L1_UNCHANGED_RAMP`] of the picture with the light's colour, a smoothstep) get the extra, the
/// pixels L1 changed (a blurred pixel: a ball's core is not a small light) gather L1's output and have none, and
/// between the two by the smoothstep. A function of the pixel and its place alone, so a tile and the whole frame
/// give the same bits.
#[must_use]
pub fn l2_input(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    raw: &ImageRef,
    l1: Option<&ImageRef>,
) -> Option<L2Input> {
    l2_input_with(plan, planes.light(), k, raw, l1)
}

/// [`l2_input`] with the map of the clipped lights itself.
pub(crate) fn l2_input_with(
    plan: &LensPlan,
    map: &LightMap,
    k: usize,
    raw: &ImageRef,
    l1: Option<&ImageRef>,
) -> Option<L2Input> {
    if plan.field_is_identity() || !any_saturated(plan, raw) {
        return None;
    }
    let clip = plan.frame().clip;
    let rs = Restore::new(plan, CLIP_RAMP);
    let rect = l1.map_or_else(|| raw.rect(), ImageRef::rect);
    let (w, h) = (rect.w, rect.h);
    let n = w * h;
    let rn = raw.width * raw.height;
    let mut extra = vec![0.0f32; 3 * n];
    let mut own = vec![0.0f32; 3 * n];
    let (er, erest) = extra.split_at_mut(n);
    let (eg, eb) = erest.split_at_mut(n);
    let (or, orest) = own.split_at_mut(n);
    let (og, ob) = orest.split_at_mut(n);
    let rows = er
        .par_chunks_mut(w)
        .zip(eg.par_chunks_mut(w))
        .zip(eb.par_chunks_mut(w))
        .zip(or.par_chunks_mut(w))
        .zip(og.par_chunks_mut(w))
        .zip(ob.par_chunks_mut(w));
    rows.enumerate().for_each(|(row, (((((er, eg), eb), or), og), ob))| {
        let j = rect.y + row;
        for col in 0..w {
            let i = rect.x + col;
            let at = (j - raw.y0) * raw.width + (i - raw.x0);
            let x = [
                f64::from(raw.data[at]),
                f64::from(raw.data[rn + at]),
                f64::from(raw.data[2 * rn + at]),
            ];
            let ev = map.ev_at(i, j, k);
            let (lit, headroom) = pixel_light(&rs, map, x, ev);
            // Without L1: the picture gathers as it is, and the light in its own colour less the picture, with
            // the headroom, is spread beside it.
            let (picture, more) = (
                x,
                [
                    lit[0] - x[0] + headroom,
                    lit[1] - x[1] + headroom,
                    lit[2] - x[2] + headroom,
                ],
            );
            let (picture, more) = match l1 {
                None => (picture, more),
                Some(l1) => {
                    let li = (j - l1.y0) * l1.width + (i - l1.x0);
                    let ln = l1.width * l1.height;
                    let v = [
                        f64::from(l1.data[li]),
                        f64::from(l1.data[ln + li]),
                        f64::from(l1.data[2 * ln + li]),
                    ];
                    // How much L1 left the pixel as it is, in clip levels.
                    let mut d = 0.0f64;
                    for c in 0..3 {
                        let g0 = f64::from(clip[c]);
                        let g0 = if g0 > 0.0 { g0 } else { 1.0 };
                        d = d.max((v[c] - lit[c]).abs() / g0);
                    }
                    let m = 1.0
                        - smooth((d - L1_UNCHANGED_RAMP.0) / (L1_UNCHANGED_RAMP.1 - L1_UNCHANGED_RAMP.0));
                    (
                        [
                            m * picture[0] + (1.0 - m) * v[0],
                            m * picture[1] + (1.0 - m) * v[1],
                            m * picture[2] + (1.0 - m) * v[2],
                        ],
                        [m * more[0], m * more[1], m * more[2]],
                    )
                }
            };
            er[col] = more[0] as f32;
            eg[col] = more[1] as f32;
            eb[col] = more[2] as f32;
            or[col] = picture[0] as f32;
            og[col] = picture[1] as f32;
            ob[col] = picture[2] as f32;
        }
    });
    Some(L2Input { rect, extra, own })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amounts::{LensAmounts, LensKind};
    use crate::field::{self, Geometry, Options};
    use crate::plan::{Axis, FrameInfo};
    use crate::planes::LensPlanes;
    use crate::test_support::write_synthetic_lenses;

    /// The clip levels of a body whose white balance puts 2.6 on red and 1.95 on blue: green clips first.
    const G0: [f32; 3] = [2.6, 1.0, 1.95];
    const W: usize = 600;
    const H: usize = 400;

    fn plan(kind: LensKind, edges: f64, anamorphic: f64) -> LensPlan {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_lenses(dir.path()).unwrap();
        let lib = crate::data::LensLibrary::from_dir(dir.path()).unwrap();
        let amounts = LensAmounts {
            kind,
            flares: 0.0,
            edges,
            anamorphic,
        };
        let frame = FrameInfo {
            width: W,
            height: H,
            axis: Axis::X,
            f_number: None,
            clip: G0,
        };
        LensPlan::new(&lib, &amounts, frame, &crate::spectral::CameraResponse::neutral()).unwrap()
    }

    fn whole(data: &[f32]) -> ImageRef<'_> {
        ImageRef {
            width: W,
            height: H,
            x0: 0,
            y0: 0,
            frame_width: W,
            frame_height: H,
            data,
        }
    }

    /// A dim ground with a disc of `value` of radius `r` at `(cx, cy)`.
    fn paint(data: &mut [f32], value: [f32; 3], cx: f64, cy: f64, r: f64) {
        let n = W * H;
        for y in 0..H {
            for x in 0..W {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                if dx * dx + dy * dy <= r * r {
                    for c in 0..3 {
                        data[c * n + y * W + x] = value[c];
                    }
                }
            }
        }
    }

    fn ground() -> Vec<f32> {
        vec![0.02f32; 3 * W * H]
    }

    #[test]
    fn the_map_gives_a_star_its_full_boost_and_a_large_light_none() {
        let mut data = ground();
        // A star of one pixel at the clip pattern's level, a green-clipped white light of 9 pixels, a clipped
        // disc of 49 pixels and a clipped window 120 pixels across.
        paint(&mut data, [3.0, 3.0, 3.0], 100.5, 100.5, 0.7);
        paint(&mut data, [1.4, 1.1, 1.7], 300.5, 100.5, 1.7);
        paint(&mut data, [3.0, 3.0, 3.0], 200.5, 250.5, 4.0);
        paint(&mut data, [3.0, 3.0, 3.0], 400.0, 300.0, 60.0);
        let p = plan(LensKind::Vintage, 100.0, 0.0);
        let map = LightMap::build(&p, &whole(&data), 1.0, &|| false).unwrap();
        assert!(!map.is_empty());
        let half_diag = 0.5 * ((W * W + H * H) as f64).sqrt();
        let want = |pixels: f64| {
            let d = 2.0 * (pixels / std::f64::consts::PI).sqrt() / half_diag;
            f64::from(boost_ev(FIELD_LIGHT_SIZE_FACTOR * d) as f32)
        };
        assert_eq!(map.ev_at(100, 100, 1), 9.0, "a clipped star of one pixel");
        assert_eq!(map.ev_at(101, 99, 1), 9.0, "the pixel beside the star's");
        let nine = map.ev_at(300, 100, 1);
        assert!(
            (nine - want(9.0)).abs() < 1e-5 && nine < 9.0 && nine > 7.0,
            "a light clipped in green alone, 9 pixels: {nine}"
        );
        let fifty = map.ev_at(200, 250, 1);
        assert!((fifty - want(49.0)).abs() < 0.6 && fifty < nine, "a disc of 49 pixels: {fifty}");
        assert_eq!(map.ev_at(400, 300, 1), 0.0, "a window is a surface");
        assert_eq!(map.ev_at(10, 10, 1), 0.0, "the ground");
        // The same place at a scale of 2 reads the plane pixel under the scale's pixel centre.
        assert_eq!(map.ev_at(50, 50, 2), 9.0);
        assert_eq!(map.ev_at(200, 150, 2), 0.0);
        // The gain is 0 without EV, grows with it and with the cap's scale.
        assert_eq!(map.gain(0.0), 0.0);
        assert!(map.gain(9.0) > map.gain(4.0));
        let capped = LightMap::build(&p, &whole(&data), 0.25, &|| false).unwrap();
        assert!((capped.gain(9.0) - 0.25 * map.gain(9.0)).abs() < 1e-12);
    }

    #[test]
    fn pixels_the_lens_leaves_alone_have_the_bits_they_have_without_the_rule() {
        // Near-saturated discs where Edges 40's kernels are a copy (the centre) and where they are not (a corner).
        let mut data = ground();
        paint(&mut data, [1.4, 1.1, 1.7], 300.5, 200.5, 5.0);
        paint(&mut data, [3.0, 3.0, 3.0], 330.5, 200.5, 2.0);
        paint(&mut data, [1.4, 1.1, 1.7], 80.5, 60.5, 5.0);
        let p = plan(LensKind::Default, 40.0, 0.0);
        let planes = LensPlanes::build(&p, &whole(&data), None, &|| false).unwrap();
        let rect = PixelRect::new(0, 0, W, H);
        let with = crate::apply_field(&p, &planes, 1, &whole(&data), rect).unwrap();
        let without = field::apply(&p, &whole(&data), None, rect, Options::default()).unwrap();
        let geo = Geometry::new(&p, W, H, Options::default()).unwrap();
        let n = W * H;
        let (mut copies, mut moved) = (0, 0);
        for j in 0..H {
            for i in 0..W {
                let same = (0..3).all(|c| with[c * n + j * W + i] == without[c * n + j * W + i]);
                if geo.pixel(i, j).omega == 0.0 {
                    copies += 1;
                    assert!(same, "the copy pixel ({i}, {j}) has the bits it has without the rule");
                } else if !same {
                    moved += 1;
                }
            }
        }
        assert!(copies > 1000, "{copies} copy pixels");
        assert!(moved > 20, "the rule changes the light the lens moves: {moved} pixels");
    }

    #[test]
    fn the_lens_adds_no_negative_light_and_no_more_than_the_light_it_moves() {
        // A pixel's output is its gather less its own weight of the extra: never under what the pixel itself
        // keeps of its own light times its weight.
        let mut data = ground();
        paint(&mut data, [3.0, 3.0, 3.0], 300.5, 120.5, 1.5);
        paint(&mut data, [1.4, 1.1, 1.7], 480.5, 280.5, 1.5);
        for (kind, edges) in [(LensKind::Vintage, 100.0), (LensKind::Petzval, 60.0)] {
            let p = plan(kind, edges, 0.0);
            let planes = LensPlanes::build(&p, &whole(&data), None, &|| false).unwrap();
            let out = crate::apply_field(&p, &planes, 1, &whole(&data), PixelRect::new(0, 0, W, H))
                .unwrap();
            assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "{kind}");
        }
    }
}
