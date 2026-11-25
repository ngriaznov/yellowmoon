//! The measures of a picture and its scene, `X.*` of `docs/34` section 4b.4.
//!
//! Measured in CIELAB with the D65 white on the picture's sRGB display codes, clamped to 0 to 1. The photo is measured on the 8192
//! sample pixels of section 5.1 rendered with the node at its identity, the reference on every pixel of its analysis picture.
//! This is the proof of concept's `measure` with the round-15 flags fixed on (the cast off the whites and the skin, Oklab for the
//! purple band), written again under the house rules: `f64`, a fixed order, no I/O.

use std::collections::BTreeMap;

use crate::consts::{REC709, Vec3, linalg};
use crate::deltae::{d65_white_xyz, xyz_to_lab};
use crate::encode::srgb_decode;

use super::{ELEMENT_NAMES, SECTORS};

/// The segmenter's element numbers of a pixel (`ym_ml::eomt::Element`).
pub mod class {
    /// No segment claims the pixel.
    pub const UNLABELLED: u8 = 0;
    /// Sky.
    pub const SKY: u8 = 1;
    /// Vegetation.
    pub const VEGETATION: u8 = 2;
    /// Water.
    pub const WATER: u8 = 3;
    /// Ground.
    pub const GROUND: u8 = 4;
    /// Architecture (with [`INTERIOR`], the grade's `built`).
    pub const ARCHITECTURE: u8 = 5;
    /// Interior.
    pub const INTERIOR: u8 = 6;
    /// Person.
    pub const PERSON: u8 = 7;
    /// Every other thing class.
    pub const OTHER_THING: u8 = 8;
    /// Every other stuff class.
    pub const OTHER_STUFF: u8 = 9;
}

/// The hue below which an element or a sector pixel has no hue: its C\* is at most this (`docs/34` section 4b.4).
pub const HUE_MIN_CHROMA: f64 = 5.0;
/// The share of the picture an element needs to be present in the walls (`docs/34` section 5.4).
pub const PRESENT_SHARE: f64 = 0.02;
/// The cast is measured on pixels with L\* in this range (`docs/34` section 4b.4).
const CAST_L: (f64, f64) = (15.0, 80.0);
/// ... when they are at least this share of the picture.
const CAST_MIN_SHARE: f64 = 0.10;
/// A neutral pixel of a tercile has C\* at most the larger of this and the 15 % rank of its group.
const NEUTRAL_CHROMA_MAX: f64 = 8.0;
/// The rank of the neutral group's chroma that bounds the near-neutral pixels.
const NEUTRAL_MIN_FRACTION: f64 = 0.15;
/// The purple band of the Oklab hue, in degrees (`docs/34` section 4b.4, `magenta_share`).
pub const PURPLE_BAND: (f64, f64) = (300.0, 355.0);
/// Crushed black: L\* below this (`docs/34` section 5.1).
pub const CRUSH_L: f64 = 2.5;
/// Clipped white: an encoded channel at or above this (`docs/34` section 5.1).
pub const CLIP_HI_ENCODED: f64 = 0.995;

/// What the labels say about the pixels of a picture.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PixelScene {
    /// The segmenter's element number of each pixel ([`class`]).
    pub class: Vec<u8>,
    /// Whether each pixel is skin.
    pub skin: Vec<bool>,
    /// Whether each pixel is the subject.
    pub subject: Vec<bool>,
}

impl PixelScene {
    /// The scene of `n` pixels with no label, no skin and no subject.
    #[must_use]
    pub fn blank(n: usize) -> Self {
        Self {
            class: vec![class::UNLABELLED; n],
            skin: vec![false; n],
            subject: vec![false; n],
        }
    }

    /// The number of pixels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.class.len()
    }

    /// Whether there are no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.class.is_empty()
    }

    /// Whether pixel `i` belongs to the grade's element `k` (sky, vegetation, water, ground, built, skin).
    #[must_use]
    pub fn in_element(&self, i: usize, k: usize) -> bool {
        let e = self.class[i];
        match k {
            0 => e == class::SKY,
            1 => e == class::VEGETATION,
            2 => e == class::WATER,
            3 => e == class::GROUND,
            4 => e == class::ARCHITECTURE || e == class::INTERIOR,
            _ => self.skin[i],
        }
    }

    /// The pixels at `indices`, in that order.
    #[must_use]
    pub fn gather(&self, indices: &[usize]) -> Self {
        Self {
            class: indices.iter().map(|&i| self.class[i]).collect(),
            skin: indices.iter().map(|&i| self.skin[i]).collect(),
            subject: indices.iter().map(|&i| self.subject[i]).collect(),
        }
    }
}

/// One measured pixel.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Px {
    /// CIELAB with the D65 white.
    pub lab: [f64; 3],
    /// Oklab (a, b), for the purple tests.
    pub ok: [f64; 2],
    /// Any encoded channel at 0.995 or more.
    pub hi: bool,
    /// L\* below 2.5.
    pub lo25: bool,
}

/// Oklab (a, b) of a linear sRGB colour (Ottosson's matrices).
#[must_use]
pub fn oklab_ab(c: Vec3) -> [f64; 2] {
    let l = (0.412_221_470_8 * c[0] + 0.536_332_536_3 * c[1] + 0.051_445_992_9 * c[2]).cbrt();
    let m = (0.211_903_498_2 * c[0] + 0.680_699_545_1 * c[1] + 0.107_396_956_6 * c[2]).cbrt();
    let s = (0.088_302_461_9 * c[0] + 0.281_718_837_6 * c[1] + 0.629_978_700_5 * c[2]).cbrt();
    [
        1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    ]
}

/// The hue, in degrees 0 to 360, of an Oklab (a, b) vector.
#[must_use]
pub fn oklab_hue(ab: [f64; 2]) -> f64 {
    ab[1].atan2(ab[0]).to_degrees().rem_euclid(360.0)
}

/// Wraps an angle difference into (-180, 180].
#[must_use]
pub fn wrap_deg(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

/// The converter from display colour to measured pixels.
#[derive(Debug, Clone)]
pub struct LabConv {
    m: [[f64; 3]; 3],
    white: Vec3,
    hi: f64,
}

impl Default for LabConv {
    fn default() -> Self {
        Self::new()
    }
}

impl LabConv {
    /// The Rec.709 primaries and the D65 white.
    #[must_use]
    pub fn new() -> Self {
        Self {
            m: REC709.rgb_to_xyz(),
            white: d65_white_xyz(),
            hi: srgb_decode(CLIP_HI_ENCODED),
        }
    }

    /// A pixel of linear sRGB display light (held to 0 to 1 here, as an 8-bit file holds it; a NaN reads as 0).
    #[must_use]
    pub fn px(&self, lin: Vec3) -> Px {
        let c = lin.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) });
        let xyz = linalg::mul_vec(&self.m, c);
        let lab = xyz_to_lab(xyz, self.white);
        Px {
            lab,
            lo25: lab[0] < CRUSH_L,
            ok: oklab_ab(c),
            hi: c.iter().any(|&v| v >= self.hi),
        }
    }

    /// A pixel of sRGB display codes (0 to 1): decoded to linear light, then [`LabConv::px`].
    #[must_use]
    pub fn px_of_codes(&self, codes: Vec3) -> Px {
        self.px(codes.map(|c| srgb_decode(if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) })))
    }
}

/// The median of `v`, taken at `len / 2` of the sorted values (0 for an empty list).
fn median_of(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The circular mean, in degrees 0 to 360, of the unit vectors with sums `(sx, sy)`.
#[must_use]
pub fn circ_mean_deg(sx: f64, sy: f64) -> f64 {
    sy.atan2(sx).to_degrees().rem_euclid(360.0)
}

/// One element's statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ElementMeasure {
    /// The element's share of the picture.
    pub share: f64,
    /// Its median L\*.
    pub l: f64,
    /// The circular mean hue of its pixels with C\* above 5, in degrees 0 to 360 (0 if none).
    pub h: f64,
    /// Its median C\*.
    pub c: f64,
    /// `ln(max(C, 1))`.
    pub logc: f64,
    /// The number of pixels it holds.
    pub n: usize,
    /// Whether any pixel has a hue.
    pub has_hue: bool,
    /// The mean (a\*, b\*) of every member pixel: its angle is the group's hue and its length the group's chroma in the walls.
    pub mv_a: f64,
    /// See `mv_a`.
    pub mv_b: f64,
}

impl ElementMeasure {
    /// The mean-vector hue (degrees, 0 to 360) and chroma.
    #[must_use]
    pub fn mv(&self) -> (f64, f64) {
        (
            circ_mean_deg(self.mv_a, self.mv_b),
            self.mv_a.hypot(self.mv_b),
        )
    }
}

/// One hue sector's statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SectorMeasure {
    /// The share of the picture.
    pub share: f64,
    /// `ln(max(median C*, 1))`.
    pub logc: f64,
    /// The circular mean hue.
    pub hue: f64,
    /// The number of pixels.
    pub n: usize,
}

/// What the network and the objective read from a picture.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Measure {
    /// L\* at 2 % of the pixels' ranks.
    pub p2: f64,
    /// L\* at 50 %.
    pub p50: f64,
    /// L\* at 98 %.
    pub p98: f64,
    /// `(p75 - p25) / max(p98 - p2, 1)`.
    pub iqr_rel: f64,
    /// The overall chroma, in log terms.
    pub logsat: f64,
    /// The mean (a\*, b\*) of the near-neutral, neutral-capable pixels of the three L\* terciles (shadows, mids, highlights).
    pub tint: [[f64; 2]; 3],
    /// The six elements: sky, vegetation, water, ground, built, skin.
    pub el: [ElementMeasure; 6],
    /// Whether the subject and the background each hold enough pixels.
    pub subj: bool,
    /// The subject's median L\* minus the background's.
    pub sep: f64,
    /// The six hue sectors.
    pub sec: [SectorMeasure; SECTORS],
    /// The share of pixels with an encoded channel at 0.995 or more.
    pub hi: f64,
    /// The share of pixels with L\* below 2.5.
    pub lo25: f64,
    /// The frame's cast: the mean a\* and b\* over the pixels that are not skin and have L\* 15 to 80.
    pub cast_a: f64,
    /// See `cast_a`.
    pub cast_b: f64,
    /// The length of the cast over the mean C\*: 0 when colours cancel, 1 when every pixel has one hue.
    pub conc: f64,
    /// The share of the picture with C\* above 8 and Oklab hue 300 to 355 degrees.
    pub magenta_share: f64,
    /// The clothing group: person minus skin.
    pub clothing: ElementMeasure,
}

/// Measures a picture (`docs/34` section 4b.4). `excl` are the elements left out of the sectors (skin always is).
///
/// # Panics
///
/// When `px` is empty or `scene` has another length: a mistake of the caller.
#[must_use]
pub fn measure(px: &[Px], scene: &PixelScene, excl: &[bool; 6]) -> Measure {
    use class::{ARCHITECTURE, INTERIOR, OTHER_STUFF, OTHER_THING, PERSON, SKY, UNLABELLED};
    let n = px.len();
    assert!(n > 0 && scene.len() == n, "a picture and its scene");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| px[a].lab[0].total_cmp(&px[b].lab[0]).then(a.cmp(&b)));
    let l_at = |q: f64| {
        let pos = q / 100.0 * (n - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        let (a, b) = (px[idx[lo]].lab[0], px[idx[hi]].lab[0]);
        a + (b - a) * (pos - lo as f64)
    };
    let (p2, p50, p98) = (l_at(2.0), l_at(50.0), l_at(98.0));
    let iqr_rel = (l_at(75.0) - l_at(25.0)) / (p98 - p2).max(1.0);
    let mut band = vec![0u8; n];
    for (rank, &i) in idx.iter().enumerate() {
        band[i] = ((rank * 3 / n).min(2)) as u8;
    }
    let mut el_l: [Vec<f64>; 6] = Default::default();
    let mut el_c: [Vec<f64>; 6] = Default::default();
    let mut el_v = [[0.0f64; 2]; 6];
    let mut el_hn = [0usize; 6];
    let mut el_ab = [[0.0f64; 2]; 6];
    let (mut pool_a, mut pool_b): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut capable: [Vec<(f64, f64, f64)>; 3] = Default::default();
    let (mut subj, mut bg): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut sec_c: [Vec<f64>; SECTORS] = Default::default();
    let mut sec_v = [[0.0f64; 2]; SECTORS];
    let (mut hi, mut lo25) = (0usize, 0usize);
    for (i, p) in px.iter().enumerate() {
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        let e = scene.class[i];
        for k in 0..6 {
            if scene.in_element(i, k) {
                el_ab[k][0] += a;
                el_ab[k][1] += b;
                el_l[k].push(p.lab[0]);
                el_c[k].push(c);
                if c > HUE_MIN_CHROMA {
                    el_v[k][0] += a / c;
                    el_v[k][1] += b / c;
                    el_hn[k] += 1;
                }
            }
        }
        if matches!(e, 2 | 3 | 4 | 5 | 6 | 8 | 9) {
            pool_a.push(c);
        }
        if e != SKY && e != PERSON {
            pool_b.push(c);
        }
        if matches!(
            e,
            UNLABELLED | 4 | ARCHITECTURE | INTERIOR | OTHER_THING | OTHER_STUFF
        ) || (e == PERSON && !scene.skin[i])
        {
            capable[usize::from(band[i])].push((c, a, b));
        }
        if scene.subject[i] {
            subj.push(p.lab[0]);
        } else if e != UNLABELLED {
            bg.push(p.lab[0]);
        }
        if c > HUE_MIN_CHROMA
            && !scene.skin[i]
            && !(0..5).any(|k| excl[k] && scene.in_element(i, k))
        {
            let h = b.atan2(a).to_degrees();
            let s = ((((h + 30.0).rem_euclid(360.0)) / 60.0) as usize) % SECTORS;
            sec_c[s].push(c);
            sec_v[s][0] += a / c;
            sec_v[s][1] += b / c;
        }
        hi += usize::from(p.hi);
        lo25 += usize::from(p.lo25);
    }
    let nf = n as f64;
    // The cast: pixels that are not skin and have L* 15 to 80 when they are at least a tenth of the picture; else every pixel that
    // is not skin; else every pixel.
    let mid: Vec<usize> = (0..n)
        .filter(|&i| !scene.skin[i] && (CAST_L.0..=CAST_L.1).contains(&px[i].lab[0]))
        .collect();
    let cast_set: Vec<usize> = if mid.len() as f64 >= CAST_MIN_SHARE * nf {
        mid
    } else {
        (0..n).filter(|&i| !scene.skin[i]).collect()
    };
    let cast_set = if cast_set.is_empty() {
        (0..n).collect()
    } else {
        cast_set
    };
    let (sum_a, sum_b, sum_c) = cast_set.iter().fold((0.0, 0.0, 0.0), |s, &i| {
        let p = &px[i];
        (
            s.0 + p.lab[1],
            s.1 + p.lab[2],
            s.2 + p.lab[1].hypot(p.lab[2]),
        )
    });
    let cn = cast_set.len() as f64;
    let (cast_a, cast_b) = (sum_a / cn, sum_b / cn);
    let conc = cast_a.hypot(cast_b) / (sum_c / cn).max(1.0);
    let mut m = Measure {
        cast_a,
        cast_b,
        conc,
        p2,
        p50,
        p98,
        iqr_rel,
        hi: hi as f64 / nf,
        lo25: lo25 as f64 / nf,
        ..Measure::default()
    };
    let pool = if pool_a.len() as f64 / nf >= 0.01 {
        &mut pool_a
    } else {
        &mut pool_b
    };
    m.logsat = median_of(pool).max(1.0).ln();
    for b in 0..3 {
        let mut cs: Vec<f64> = capable[b].iter().map(|p| p.0).collect();
        cs.sort_unstable_by(f64::total_cmp);
        let floor_c = cs
            .get((cs.len() as f64 * NEUTRAL_MIN_FRACTION) as usize)
            .copied()
            .unwrap_or(0.0);
        let limit = NEUTRAL_CHROMA_MAX.max(floor_c);
        let (mut sa, mut sb, mut k) = (0.0, 0.0, 0usize);
        for &(c, a, bb) in &capable[b] {
            if c <= limit {
                sa += a;
                sb += bb;
                k += 1;
            }
        }
        if k > 0 {
            m.tint[b] = [sa / k as f64, sb / k as f64];
        }
    }
    // The magenta share and the clothing group (person minus skin).
    let mut mag = 0usize;
    let (mut cl_l, mut cl_c) = (Vec::new(), Vec::new());
    let (mut cl_x, mut cl_y, mut cl_hn) = (0.0, 0.0, 0usize);
    let (mut cl_sa, mut cl_sb) = (0.0, 0.0);
    for (i, p) in px.iter().enumerate() {
        let (a, b) = (p.lab[1], p.lab[2]);
        let c = a.hypot(b);
        if c > 8.0 && (PURPLE_BAND.0..=PURPLE_BAND.1).contains(&oklab_hue(p.ok)) {
            mag += 1;
        }
        if scene.class[i] == PERSON && !scene.skin[i] {
            cl_sa += a;
            cl_sb += b;
            cl_l.push(p.lab[0]);
            cl_c.push(c);
            if c > HUE_MIN_CHROMA {
                cl_x += a / c;
                cl_y += b / c;
                cl_hn += 1;
            }
        }
    }
    m.magenta_share = mag as f64 / nf;
    if !cl_l.is_empty() {
        let c = median_of(&mut cl_c);
        m.clothing = ElementMeasure {
            share: cl_l.len() as f64 / nf,
            l: median_of(&mut cl_l),
            h: if cl_hn > 0 {
                circ_mean_deg(cl_x, cl_y)
            } else {
                0.0
            },
            c,
            logc: c.max(1.0).ln(),
            n: cl_hn,
            has_hue: cl_hn > 0,
            mv_a: cl_sa / cl_c.len() as f64,
            mv_b: cl_sb / cl_c.len() as f64,
        };
    }
    for k in 0..6 {
        let cnt = el_l[k].len();
        if cnt == 0 {
            continue;
        }
        let c = median_of(&mut el_c[k]);
        m.el[k] = ElementMeasure {
            share: cnt as f64 / nf,
            l: median_of(&mut el_l[k]),
            h: if el_hn[k] > 0 {
                circ_mean_deg(el_v[k][0], el_v[k][1])
            } else {
                0.0
            },
            c,
            logc: c.max(1.0).ln(),
            n: cnt,
            has_hue: el_hn[k] > 0,
            mv_a: el_ab[k][0] / cnt as f64,
            mv_b: el_ab[k][1] / cnt as f64,
        };
    }
    let enough = |v: &Vec<f64>| v.len() as f64 >= (0.01 * nf).max(30.0);
    m.subj = enough(&subj) && enough(&bg);
    if m.subj {
        m.sep = median_of(&mut subj) - median_of(&mut bg);
    }
    for j in 0..SECTORS {
        let cnt = sec_c[j].len();
        if cnt > 0 {
            m.sec[j] = SectorMeasure {
                share: cnt as f64 / nf,
                logc: median_of(&mut sec_c[j]).max(1.0).ln(),
                hue: circ_mean_deg(sec_v[j][0], sec_v[j][1]),
                n: cnt,
            };
        }
    }
    m
}

/// Whether an element of both pictures is left out of the sectors: both hold it at 2 % or more of the picture (sky, vegetation,
/// water, ground, built).
#[must_use]
pub fn sector_exclusions(p: &Measure, r: &Measure) -> [bool; 6] {
    let mut excl = [false; 6];
    for (k, flag) in excl.iter_mut().enumerate().take(5) {
        *flag = p.el[k].share >= PRESENT_SHARE && r.el[k].share >= PRESENT_SHARE;
    }
    excl
}

/// The network's input values from the photo's and the reference's measures, in the order of
/// [`Network::inputs`](super::network::Network::inputs).
#[must_use]
pub fn network_inputs(p: &Measure, r: &Measure) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    for (x, q) in [("P", p), ("R", r)] {
        m.insert(format!("{x}.p2"), q.p2);
        m.insert(format!("{x}.p50"), q.p50);
        m.insert(format!("{x}.p98"), q.p98);
        m.insert(format!("{x}.iqr_rel"), q.iqr_rel);
        m.insert(format!("{x}.logsat"), q.logsat);
        m.insert(format!("{x}.subj"), if q.subj { 1.0 } else { 0.0 });
        m.insert(format!("{x}.sep"), q.sep);
        m.insert(format!("{x}.cast_a"), q.cast_a);
        m.insert(format!("{x}.cast_b"), q.cast_b);
        m.insert(format!("{x}.conc"), q.conc);
        m.insert(format!("{x}.magenta_share"), q.magenta_share);
        for (b, name) in ["s", "m", "h"].iter().enumerate() {
            m.insert(format!("{x}.tint_{name}_a"), q.tint[b][0]);
            m.insert(format!("{x}.tint_{name}_b"), q.tint[b][1]);
        }
        for (k, e) in ELEMENT_NAMES.iter().enumerate() {
            let em = &q.el[k];
            m.insert(format!("{x}.{e}.share"), em.share);
            m.insert(format!("{x}.{e}.L"), em.l);
            m.insert(format!("{x}.{e}.h"), em.h);
            m.insert(format!("{x}.{e}.C"), em.c);
            m.insert(format!("{x}.{e}.logC"), em.logc);
        }
        for j in 0..SECTORS {
            m.insert(format!("{x}.s{j}.share"), q.sec[j].share);
            m.insert(format!("{x}.s{j}.logC"), q.sec[j].logc);
        }
    }
    for (k, e) in ELEMENT_NAMES.iter().enumerate() {
        let d = if p.el[k].has_hue && r.el[k].has_hue {
            wrap_deg(r.el[k].h - p.el[k].h)
        } else {
            0.0
        };
        m.insert(format!("D.{e}.dh"), d);
    }
    for j in 0..SECTORS {
        let d = if p.sec[j].n > 0 && r.sec[j].n > 0 {
            wrap_deg(r.sec[j].hue - p.sec[j].hue)
        } else {
            0.0
        };
        m.insert(format!("D.s{j}.dh"), d);
    }
    m
}
