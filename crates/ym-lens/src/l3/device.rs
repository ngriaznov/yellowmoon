//! L3 for the device (`crate::device`): the data the flare kernel reads, made from the CPU reference's own
//! structures ([`Items`], [`GlowProfiles`], [`FlarePlanes`]) so the two read one source.
//!
//! Three lifetimes:
//!
//! * per photo ([`FlarePlaneData`], from the planes): the tail plane and the spike tables, as `f32`;
//! * per lens and camera ([`FlareLensData`]): the rim curve, the aperture, the tables of the normal tail;
//! * per plan shape and scale ([`FlareScaleData`]): the glow core's taps, the streak's profile and the
//!   drawn items with the bins that cull them. It does not depend on Flares, which only scales the sum
//!   (`F` and `1 - a` are in [`FlareMix`]), so a Flares move uploads nothing.

use crate::consts::{CLIP_RAMP, LENS_HEADROOM_EV, RIM_SAMPLES, SPIKE_SIZE};
use crate::device::{
    FlareLensData, FlarePlaneData, FlareRestore, FlareScaleData, MAX_APERTURE_NORMALS,
};
use crate::l3::geometry::{Frame, Span};
use crate::l3::items::Items;
use crate::l3::profile::{RimCurve, normal_tail};
use crate::l3::terms::Terms;
use crate::plan::{Axis, LensPlan};
use crate::planes::LensPlanes;

/// Side of a bin in pixels: the item lists are culled per bin of the frame's absolute pixel grid.
pub(crate) const BIN_PX: usize = 32;

/// Vectors of the header of [`FlareScaleData::vectors`]: the streak's profile (3) and its cut table (17).
pub(crate) const HEADER: usize = 20;

/// The tail plane and spike tables of a photo.
pub(crate) fn plane_data(planes: &LensPlanes) -> Option<FlarePlaneData> {
    let flare = planes.flare();
    flare.glow()?;
    let (_, tw, th) = flare.tail(0);
    let mut tail = Vec::with_capacity(3 * tw * th);
    for c in 0..3 {
        tail.extend(flare.tail(c).0.iter().map(|v| *v as f32));
    }
    let mut spikes = Vec::with_capacity(3 * SPIKE_SIZE * SPIKE_SIZE);
    for c in 0..3 {
        spikes.extend(flare.spikes()[c].iter().map(|v| *v as f32));
    }
    Some(FlarePlaneData {
        tail_size: (tw, th),
        tail,
        spikes,
    })
}

/// The lens's rim, aperture and the normal tail's tables.
pub(crate) fn lens_data(plan: &LensPlan) -> FlareLensData {
    let lens = plan.lens();
    let mut tab: Vec<[f32; 4]> = Vec::new();
    let rim = RimCurve::new(&lens.rim);
    let (y, m) = rim.device_tables();
    for table in [y, m] {
        let mut padded = [0.0f32; RIM_SAMPLES];
        for (d, v) in padded.iter_mut().zip(table) {
            *d = *v as f32;
        }
        tab.extend(padded.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]));
    }
    // The aperture: the count of normals and the circle's weight, then the normals, two to a vector.
    let shape = super::items::Shape::aperture(&lens.flare.aperture);
    let (normals, rho) = shape.device_shape();
    let n = normals.len().min(MAX_APERTURE_NORMALS);
    tab.push([n as f32, rho as f32, 0.0, 0.0]);
    let mut flat = [0.0f32; 2 * MAX_APERTURE_NORMALS];
    for (i, nm) in normals.iter().take(n).enumerate() {
        flat[2 * i] = nm[0] as f32;
        flat[2 * i + 1] = nm[1] as f32;
    }
    tab.extend(flat.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]));
    tab.extend(normal_tail().device_nodes());
    FlareLensData {
        key: format!(
            "flare:{}:v{}:{}",
            plan.kind(),
            lens.info.version,
            plan.camera().key()
        ),
        tab,
    }
}

/// The scale-dependent data of a plan for a frame of `width x height` pixels, or `None` when the flares add
/// no light (Flares 0) or the planes hold no highlight data.
pub(crate) fn scale_data(
    plan: &LensPlan,
    planes: &LensPlanes,
    width: usize,
    height: usize,
) -> Option<FlareScaleData> {
    let terms = Terms::new(plan);
    if terms.a <= 0.0 {
        return None;
    }
    let flare = planes.flare();
    let glow = flare.glow()?;
    let frame = Frame::new(width, height, plan.frame().axis);
    let whole = Span {
        x0: 0,
        x1: width,
        y0: 0,
        y1: height,
    };
    let items = Items::build(plan, &terms, frame, flare.sources(), flare.spikes(), &whole);
    let taps = glow.core_taps(frame.half_diag);
    let core_radius = if taps.is_empty() { 0 } else { taps.radius };

    let mut vectors: Vec<[f32; 4]> = Vec::new();
    // The streak's profile: (core, power, length, flat), (flat mass, half mass, 0, 0), the cut table.
    match items.along() {
        Some(a) => {
            let (c, cut) = a.device_constants();
            vectors.push([c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]);
            vectors.push([c[4] as f32, c[5] as f32, 0.0, 0.0]);
            vectors.push([0.0; 4]);
            let mut padded = vec![0.0f32; 68];
            for (d, v) in padded.iter_mut().zip(cut) {
                *d = *v as f32;
            }
            vectors.extend(padded.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]));
        }
        None => vectors.extend(std::iter::repeat_n([0.0f32; 4], HEADER)),
    }
    debug_assert_eq!(vectors.len(), HEADER);
    let core_offset = vectors.len();
    if !taps.is_empty() {
        vectors.extend(taps.lanes.iter().copied());
    }
    let record_offset = vectors.len();
    let records = items.device_records();
    let count = records.len() / 6;
    vectors.extend(records);

    // Bins: the items whose pixel rectangle meets each bin of the absolute grid, in list order.
    let (nbx, nby) = (width.div_ceil(BIN_PX), height.div_ceil(BIN_PX));
    let spans = items.spans();
    let mut counts = vec![0u32; nbx * nby];
    let bins_of = |s: &Span| -> (usize, usize, usize, usize) {
        (
            s.x0 / BIN_PX,
            (s.x1.saturating_sub(1)) / BIN_PX,
            s.y0 / BIN_PX,
            (s.y1.saturating_sub(1)) / BIN_PX,
        )
    };
    for s in &spans {
        if s.is_empty() {
            continue;
        }
        let (bx0, bx1, by0, by1) = bins_of(s);
        for by in by0..=by1.min(nby - 1) {
            for bx in bx0..=bx1.min(nbx - 1) {
                counts[by * nbx + bx] += 1;
            }
        }
    }
    let mut start = vec![0u32; nbx * nby + 1];
    for i in 0..nbx * nby {
        start[i + 1] = start[i] + counts[i];
    }
    let mut fill = start.clone();
    let mut bin_items = vec![0u32; start[nbx * nby] as usize];
    for (k, s) in spans.iter().enumerate() {
        if s.is_empty() {
            continue;
        }
        let (bx0, bx1, by0, by1) = bins_of(s);
        for by in by0..=by1.min(nby - 1) {
            for bx in bx0..=bx1.min(nbx - 1) {
                let b = by * nbx + bx;
                bin_items[fill[b] as usize] = k as u32;
                fill[b] += 1;
            }
        }
    }
    let mut bins = start;
    bins.extend(bin_items);

    let c = plan.frame().clip;
    Some(FlareScaleData {
        width,
        height,
        half_diag: frame.half_diag,
        axis_y: plan.frame().axis == Axis::Y,
        core_radius,
        vectors,
        core_offset,
        record_offset,
        records: count,
        bins,
        bins_x: nbx,
        bins_y: nby,
        bin_px: BIN_PX,
        spike_radius: items.spike_radius(),
        restore: FlareRestore {
            clip: c,
            ramp: (CLIP_RAMP.0 as f32, CLIP_RAMP.1 as f32),
            boost: (libm::exp2(LENS_HEADROOM_EV) - 1.0) as f32,
        },
        has_streak: items.along().is_some(),
    })
}

/// `(F, 1 - a)` of a plan: the weight of the flare terms and the share of the picture's light that stays.
pub(crate) fn mix(plan: &LensPlan) -> (f64, f64) {
    let t = Terms::new(plan);
    (t.f, 1.0 - t.a)
}
