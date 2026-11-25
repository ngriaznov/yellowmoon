//! The objective of the Reference Grade fit: the look terms and the walls of `docs/34` sections 5.3 and 5.4.
//!
//! This is round 15's objective: the proof of concept's v14 objective with its principles P2 to P5 on and P1 off, every constant
//! taken from its code. [`Fixed`] holds everything that is fixed for one photo and one reference (their measures, the network's
//! answer, the groups of the photo's own sample pixels); [`CandidateMeasure`] is what is measured on a candidate's picture;
//! [`terms`] turns the pair into [`Terms`], and [`Terms::total`] and [`Terms::look_only_total`] sum them.

use super::measure::{
    Measure, PRESENT_SHARE, PURPLE_BAND, PixelScene, Px, circ_mean_deg, measure, network_inputs,
    oklab_hue, sector_exclusions, wrap_deg,
};
use super::network::{Network, NetworkError, NetworkErrorKind, NetworkValues};
use super::{ELEMENT_NAMES, SECTORS};

/// A candidate the chain refused gets this clip term and is worse than anything.
pub const REFUSED_CLIP: f64 = 1e6;
/// The weight of the clip term in the full objective.
pub const CLIP_WEIGHT: f64 = 3.0;
/// The regulariser's weight (global and local halves each).
pub const REG_WEIGHT: f64 = 0.5;
/// The slack of the clip term, as a share of the picture (`docs/34` section 5.4).
pub const CLIP_SLACK: f64 = 0.005;
/// The scale of the clip term.
pub const CLIP_SCALE: f64 = 0.01;

/// The weight of each element's terms (sky, vegetation, water, ground, built, skin).
pub const ELEMENT_WEIGHT: [f64; 6] = [1.5, 1.2, 1.0, 1.0, 0.6, 2.0];
/// The hue cap of each element, in degrees (`docs/34` section 4a.1).
pub const ELEMENT_CAP_DEG: [f64; 6] = [40.0, 35.0, 35.0, 30.0, 25.0, 15.0];
/// The weight of each tint band's terms (shadows, mids, highlights).
const TINT_BAND_WEIGHT: [f64; 3] = [1.0, 0.7, 1.0];
/// A group counts in a wall only when its mean vector is at least this long (CIELAB a\*, b\*).
const MIN_MEAN_VECTOR: f64 = 2.5;
/// The most share of the photo's pixels the whites' group and the dark colours' group need to exist.
const WHITES_MIN_SHARE: f64 = 0.02;
/// The whites of the walls: L\* at least this and C\* below [`WHITES_C_MAX`] (the hue wall's group).
const WHITES_L_MIN: f64 = 80.0;
const WHITES_C_MAX: f64 = 10.0;
/// The whites of the chroma cap and the direction: C\* below this, not skin.
const WHITES_PIXEL_C_MAX: f64 = 8.0;
/// The dark colours: L\* below this, C\* at least [`DARK_C_MIN`], not skin.
const DARK_L_MAX: f64 = 35.0;
const DARK_C_MIN: f64 = 2.0;
/// The whites' hue wall applies when the mean vector is at least this long.
const WHITES_HUE_MIN_C: f64 = 4.0;
/// A sector group needs this share of the photo (`docs/34` section 5.4).
const SECTOR_GROUP_MIN_SHARE: f64 = 0.04;
/// The hue-drift wall: weight and scale (degrees); an element with its hue gate open may reach its cap plus this.
const DRIFT_WEIGHT: f64 = 2.0;
const DRIFT_SCALE_DEG: f64 = 5.0;
const DRIFT_EXTRA_DEG: f64 = 5.0;
/// The memory-colour wall: weight and scale (degrees).
const MEMORY: (f64, f64) = (6.0, 2.0);
/// The whites' chroma cap: weight and scale.
const WHITES_CAP: (f64, f64) = (3.0, 2.0);
/// The whites' direction: weight and scale.
const WHITES_DIRECTION: (f64, f64) = (3.0, 2.0);
/// The purple wall of a group: weight and scale (degrees).
const PURPLE_TURN: (f64, f64) = (3.0, 5.0);
/// How far from the band a group's approach counts (degrees), and its weight.
const PURPLE_REACH_DEG: f64 = 15.0;
const PURPLE_APPROACH_WEIGHT: f64 = 1.0;
/// The purple wall's pixel guard: a pixel counts as purple from this C\*, and a photo pixel under [`PURPLE_FROM_C`] is a grey.
const PURPLE_MIN_C: f64 = 4.0;
const PURPLE_FROM_C: f64 = 2.0;
/// The pixel guard: weight and scale.
const PURPLE_PIXELS: (f64, f64) = (6.0, 0.01);
/// The skin walls' weight, the scale of the hue and chroma excess of the absolute wall, the scale of the drift wall and of the
/// chroma wall.
const SKIN_WEIGHT: f64 = 12.0;
const SKIN_ABSOLUTE_SCALE: f64 = 3.0;
const SKIN_DRIFT_SCALE: f64 = 3.0;
const SKIN_CHROMA_SCALE: f64 = 0.05;
/// Skin's range of hue (degrees) and chroma in the absolute wall.
const SKIN_HUE: (f64, f64) = (30.0, 70.0);
const SKIN_CHROMA: (f64, f64) = (12.0, 38.0);
/// The skin chroma wall needs this share of the photo.
const SKIN_CHROMA_MIN_SHARE: f64 = 0.01;
/// The sky chroma wall: factor, weight and scale.
const SKY_FACTOR: f64 = 1.3;
const SKY_WEIGHT: f64 = 2.0;
const SKY_SCALE: f64 = 2.0;
/// The content turn budget: weight and scale, and the chroma a strong colour has from.
const TURN: (f64, f64) = (6.0, 0.01);
const TURN_MIN_C: f64 = 8.0;
/// An element's gate is open above this.
pub const GATE_OPEN: f64 = 0.5;

// The look terms: (scale, dead band, weight) of `docs/34` section 5.3.
const FLOOR: (f64, f64, f64) = (2.0, 1.0, 1.0);
const CEILING: (f64, f64, f64) = (2.0, 1.0, 1.0);
const CONTRAST: (f64, f64, f64) = (0.03, 0.01, 1.0);
const ANCHOR: (f64, f64, f64) = (1.0, 5.0, 4.0);
const TINT: (f64, f64) = (3.0, 1.0);
const SATURATION: (f64, f64, f64) = (0.05, 0.03, 1.0);
const SECTOR_HUE: (f64, f64, f64) = (8.0, 2.0, 0.8);
const SECTOR_CHROMA: (f64, f64, f64) = (0.08, 0.03, 0.8);
const ELEMENT_HUE: (f64, f64) = (10.0, 3.0);
const ELEMENT_CHROMA: (f64, f64) = (0.10, 0.03);
const ELEMENT_EXPOSURE: (f64, f64) = (3.0, 2.0);
const SUBJECT: (f64, f64, f64) = (3.0, 2.0, 1.0);
/// The frame cast: (scale, dead band, weight).
const FRAME_CAST: (f64, f64, f64) = (2.0, 1.0, 1.2);

const PG_CLOTHING: usize = 6;
const PG_SECTOR: usize = 7;
const PG_WHITES: usize = PG_SECTOR + SECTORS;
const PG_DARK: usize = PG_WHITES + 1;
const PG_COUNT: usize = PG_DARK + 1;

fn sq(x: f64) -> f64 {
    x * x
}

/// A difference with a dead band: 0 inside `band`, the excess outside.
fn dead(d: f64, band: f64) -> f64 {
    (d.abs() - band).max(0.0)
}

/// `term(err; scale, dead) = ((max(|err| - dead, 0)) / scale)^2` of `docs/34` section 5.3.
#[must_use]
pub fn term(err: f64, scale: f64, dead_band: f64) -> f64 {
    sq(dead(err, dead_band) / scale)
}

/// The network's outputs the objective reads, by name, found once.
#[derive(Debug, Clone, Default)]
pub struct Targets {
    /// `T.p2`.
    pub p2: f64,
    /// `T.p98`.
    pub p98: f64,
    /// `T.iqr_rel`.
    pub iqr_rel: f64,
    /// `T.p50`.
    pub p50: f64,
    /// `T.logsat`.
    pub logsat: f64,
    /// `T.tint_{s,m,h}_{a,b}`.
    pub tint: [[f64; 2]; 3],
    /// `T.cast_a`, `T.cast_b`.
    pub cast: [f64; 2],
    /// `T.wall_deg`.
    pub wall_deg: f64,
    /// `T.hl_cmax`.
    pub hl_cmax: f64,
    /// `T.white_a`, `T.white_b`.
    pub white: [f64; 2],
    /// `T.white_tol`.
    pub white_tol: f64,
    /// `T.sep`.
    pub sep: f64,
    /// `T.s{j}.dh`.
    pub sector_dh: [f64; SECTORS],
    /// `T.s{j}.dlogC`.
    pub sector_dlogc: [f64; SECTORS],
    /// `T.{e}.h`.
    pub el_h: [f64; 6],
    /// `T.{e}.logC`.
    pub el_logc: [f64; 6],
    /// `T.{e}.off`.
    pub el_off: [f64; 6],
    /// `T.purple_ok`.
    pub purple_ok: f64,
    /// `T.purple_max_share`.
    pub purple_max_share: f64,
    /// `T.purple_rot_deg`.
    pub purple_rot_deg: f64,
    /// `T.skin_drift`.
    pub skin_drift: f64,
    /// `T.skin_dlogC_max`.
    pub skin_dlogc_max: f64,
    /// `T.mem.sky_deg`, `T.mem.vegetation_deg`.
    pub mem_deg: [f64; 2],
    /// `T.turn_deg`.
    pub turn_deg: f64,
    /// `T.turn_share`.
    pub turn_share: f64,
    /// `T.turn_cref`.
    pub turn_cref: f64,
    /// `T.turn_cap`.
    pub turn_cap: f64,
    /// `T.turn_full`.
    pub turn_full: f64,
    /// `T.clip_extra`.
    pub clip_extra: f64,
    /// `G.unify`.
    pub unify: f64,
    /// `G.subj`.
    pub subj: f64,
    /// `G.s{j}`.
    pub sector_gate: [f64; SECTORS],
    /// `G.{e}.matched`.
    pub matched: [f64; 6],
    /// `G.{e}.hue`.
    pub hue_gate: [f64; 6],
    /// `{e}.e_gate`: the exposure gate of each element (matched, and no key mismatch).
    pub exposure_gate: [f64; 6],
}

impl Targets {
    /// Reads every output the objective needs from the network's values, and refuses a network that lacks one.
    ///
    /// # Errors
    ///
    /// [`NetworkErrorKind::MissingOutput`] for the first output the file does not define.
    pub fn read(net: &Network, v: &NetworkValues) -> Result<Self, NetworkError> {
        let names = net.names();
        let get = |name: String| -> Result<f64, NetworkError> {
            if names.contains(name.as_str()) {
                Ok(v.get(&name))
            } else {
                Err(NetworkError {
                    line: 0,
                    name: Some(name.clone()),
                    kind: NetworkErrorKind::MissingOutput(name),
                })
            }
        };
        let mut t = Targets {
            p2: get("T.p2".into())?,
            p98: get("T.p98".into())?,
            iqr_rel: get("T.iqr_rel".into())?,
            p50: get("T.p50".into())?,
            logsat: get("T.logsat".into())?,
            wall_deg: get("T.wall_deg".into())?,
            hl_cmax: get("T.hl_cmax".into())?,
            white_tol: get("T.white_tol".into())?,
            sep: get("T.sep".into())?,
            purple_ok: get("T.purple_ok".into())?,
            purple_max_share: get("T.purple_max_share".into())?,
            purple_rot_deg: get("T.purple_rot_deg".into())?,
            skin_drift: get("T.skin_drift".into())?,
            skin_dlogc_max: get("T.skin_dlogC_max".into())?,
            turn_deg: get("T.turn_deg".into())?,
            turn_share: get("T.turn_share".into())?,
            turn_cref: get("T.turn_cref".into())?,
            turn_cap: get("T.turn_cap".into())?,
            turn_full: get("T.turn_full".into())?,
            clip_extra: get("T.clip_extra".into())?,
            unify: get("G.unify".into())?,
            subj: get("G.subj".into())?,
            ..Targets::default()
        };
        t.cast = [get("T.cast_a".into())?, get("T.cast_b".into())?];
        t.white = [get("T.white_a".into())?, get("T.white_b".into())?];
        t.mem_deg = [
            get("T.mem.sky_deg".into())?,
            get("T.mem.vegetation_deg".into())?,
        ];
        for (b, name) in ["s", "m", "h"].iter().enumerate() {
            for (c, cn) in ["a", "b"].iter().enumerate() {
                t.tint[b][c] = get(format!("T.tint_{name}_{cn}"))?;
            }
        }
        for j in 0..SECTORS {
            t.sector_dh[j] = get(format!("T.s{j}.dh"))?;
            t.sector_dlogc[j] = get(format!("T.s{j}.dlogC"))?;
            t.sector_gate[j] = get(format!("G.s{j}"))?;
        }
        for (k, e) in ELEMENT_NAMES.iter().enumerate() {
            t.el_h[k] = get(format!("T.{e}.h"))?;
            t.el_logc[k] = get(format!("T.{e}.logC"))?;
            t.el_off[k] = get(format!("T.{e}.off"))?;
            t.matched[k] = get(format!("G.{e}.matched"))?;
            t.hue_gate[k] = get(format!("G.{e}.hue"))?;
            t.exposure_gate[k] = get(format!("{e}.e_gate"))?;
        }
        Ok(t)
    }
}

/// Everything fixed for one pair: the photo's and the reference's measures, the network's answer and the groups of the photo's own
/// sample pixels the walls look at.
#[derive(Debug, Clone)]
pub struct Fixed {
    /// The elements both pictures hold, left out of the sectors.
    pub excl: [bool; 6],
    /// The photo's measures at the identity.
    pub p: Measure,
    /// The reference's measures.
    pub r: Measure,
    /// The network's outputs.
    pub t: Targets,
    /// The network's neuron values (every neuron, in file order), kept for reports.
    pub net_values: NetworkValues,
    /// Whether the three skin walls are on (`docs/34` section 5.6).
    pub skin_protection: bool,
    sector: Vec<u8>,
    p_hue: Vec<f64>,
    p_sec_logc: [f64; SECTORS],
    hl_scope: Vec<bool>,
    p_hl_ab: [f64; 2],
    p_sec_ab: [[f64; 2]; SECTORS],
    dark_scope: Vec<bool>,
    p_dark_ab: [f64; 2],
    hl4_scope: Vec<bool>,
    p_okh: Vec<f64>,
    p_cstar: Vec<f64>,
    not_skin: Vec<bool>,
    pg: Vec<Vec<u32>>,
    p_pg_ab: Vec<[f64; 2]>,
    sample_len: usize,
}

/// The mean (a\*, b\*) over the pixels in the scope (0, 0 for none).
fn mean_ab(px: &[Px], scope: &[bool]) -> [f64; 2] {
    let (mut sa, mut sb, mut n) = (0.0, 0.0, 0usize);
    for (q, _) in px.iter().zip(scope).filter(|&(_, &s)| s) {
        sa += q.lab[1];
        sb += q.lab[2];
        n += 1;
    }
    if n == 0 {
        [0.0, 0.0]
    } else {
        [sa / n as f64, sb / n as f64]
    }
}

/// The mean Oklab (a, b) of `px` over each group's pixels.
fn pg_means(px: &[Px], pg: &[Vec<u32>]) -> Vec<[f64; 2]> {
    pg.iter()
        .map(|g| {
            if g.is_empty() {
                return [0.0, 0.0];
            }
            let (mut a, mut b) = (0.0, 0.0);
            for &i in g {
                a += px[i as usize].ok[0];
                b += px[i as usize].ok[1];
            }
            [a / g.len() as f64, b / g.len() as f64]
        })
        .collect()
}

impl Fixed {
    /// Makes the fixed part from the photo's sample at the identity (`p_px`, `scene`) and the reference's pixels (`r_px`, `r_scene`).
    ///
    /// # Errors
    ///
    /// [`NetworkError`] when the network lacks an output the objective reads.
    ///
    /// # Panics
    ///
    /// When a picture is empty or its scene has another length.
    pub fn new(
        net: &Network,
        p_px: &[Px],
        scene: &PixelScene,
        r_px: &[Px],
        r_scene: &PixelScene,
        skin_protection: bool,
    ) -> Result<Self, NetworkError> {
        let p0 = measure(p_px, scene, &[false; 6]);
        let r0 = measure(r_px, r_scene, &[false; 6]);
        let excl = sector_exclusions(&p0, &r0);
        let p = measure(p_px, scene, &excl);
        let r = measure(r_px, r_scene, &excl);
        let values = network_inputs(&p, &r);
        let inputs: Vec<f64> = net.inputs().iter().map(|n| values[n]).collect();
        let net_values = net.eval(&inputs);
        let t = Targets::read(net, &net_values)?;
        // P's sectors, on the same pixels every candidate is measured on.
        let n = p_px.len();
        let mut sector = vec![255u8; n];
        let mut p_hue = vec![0.0; n];
        let mut cs: [Vec<f64>; SECTORS] = Default::default();
        let mut sec_sum = [[0.0f64; 2]; SECTORS];
        let mut sec_n = [0usize; SECTORS];
        for (i, q) in p_px.iter().enumerate() {
            let (a, b) = (q.lab[1], q.lab[2]);
            let c = a.hypot(b);
            p_hue[i] = b.atan2(a).to_degrees();
            if c > super::measure::HUE_MIN_CHROMA
                && !scene.skin[i]
                && !(0..5).any(|k| excl[k] && scene.in_element(i, k))
            {
                let s = ((((p_hue[i] + 30.0).rem_euclid(360.0)) / 60.0) as usize) % SECTORS;
                sector[i] = s as u8;
                cs[s].push(c);
                sec_sum[s][0] += a;
                sec_sum[s][1] += b;
                sec_n[s] += 1;
            }
        }
        let mut p_sec_logc = [0.0; SECTORS];
        for j in 0..SECTORS {
            cs[j].sort_unstable_by(f64::total_cmp);
            let med = if cs[j].is_empty() {
                0.0
            } else {
                cs[j][cs[j].len() / 2]
            };
            p_sec_logc[j] = med.max(1.0).ln();
        }
        let hl_scope: Vec<bool> = p_px
            .iter()
            .map(|q| q.lab[0] >= WHITES_L_MIN && q.lab[1].hypot(q.lab[2]) < WHITES_C_MAX)
            .collect();
        let p_hl_ab = mean_ab(p_px, &hl_scope);
        let mut p_sec_ab = [[0.0; 2]; SECTORS];
        for j in 0..SECTORS {
            if sec_n[j] > 0 {
                p_sec_ab[j] = [
                    sec_sum[j][0] / sec_n[j] as f64,
                    sec_sum[j][1] / sec_n[j] as f64,
                ];
            }
        }
        let dark_scope: Vec<bool> = p_px
            .iter()
            .enumerate()
            .map(|(i, q)| {
                !scene.skin[i] && q.lab[0] < DARK_L_MAX && q.lab[1].hypot(q.lab[2]) >= DARK_C_MIN
            })
            .collect();
        let p_dark_ab = mean_ab(p_px, &dark_scope);
        let p_okh: Vec<f64> = p_px.iter().map(|q| oklab_hue(q.ok)).collect();
        let p_cstar: Vec<f64> = p_px.iter().map(|q| q.lab[1].hypot(q.lab[2])).collect();
        let not_skin: Vec<bool> = (0..n).map(|i| !scene.skin[i]).collect();
        let mut pg: Vec<Vec<u32>> = vec![Vec::new(); PG_COUNT];
        for i in 0..n {
            for k in 0..6 {
                if scene.in_element(i, k) {
                    pg[k].push(i as u32);
                }
            }
            if scene.class[i] == super::measure::class::PERSON && !scene.skin[i] {
                pg[PG_CLOTHING].push(i as u32);
            }
            if sector[i] != 255 {
                pg[PG_SECTOR + usize::from(sector[i])].push(i as u32);
            }
            if hl_scope[i] {
                pg[PG_WHITES].push(i as u32);
            }
            if dark_scope[i] {
                pg[PG_DARK].push(i as u32);
            }
        }
        let p_pg_ab = pg_means(p_px, &pg);
        let hl4_scope: Vec<bool> = p_px
            .iter()
            .enumerate()
            .map(|(i, q)| {
                !scene.skin[i]
                    && q.lab[0] >= WHITES_L_MIN
                    && q.lab[1].hypot(q.lab[2]) < WHITES_PIXEL_C_MAX
            })
            .collect();
        Ok(Self {
            excl,
            p,
            r,
            t,
            net_values,
            skin_protection,
            sector,
            p_hue,
            p_sec_logc,
            hl_scope,
            p_hl_ab,
            p_sec_ab,
            dark_scope,
            p_dark_ab,
            hl4_scope,
            p_okh,
            p_cstar,
            not_skin,
            pg,
            p_pg_ab,
            sample_len: n,
        })
    }
}

/// The Oklab hue turn, in degrees, a colour of CIELAB chroma `c_star` may make before it counts as content turning:
/// `clamp(asin(min(1, cref * sin(turn_deg) / c_star)), turn_deg, cap)`. A pale colour moved sideways reads as a tint, so it may
/// turn up to `cap`; from `c_star = cref` up it may turn only `turn_deg`. A `c_star` of 0 gives `cap` (`docs/34` section 5.4, P2).
#[must_use]
pub fn allowed_turn_deg(c_star: f64, turn_deg: f64, cref: f64, cap: f64) -> f64 {
    let ratio = if c_star > 0.0 {
        (cref * turn_deg.to_radians().sin() / c_star).min(1.0)
    } else {
        1.0
    };
    ratio
        .asin()
        .to_degrees()
        .max(turn_deg)
        .min(cap.max(turn_deg))
}

/// The weight with which one pixel counts against the content turn budget. A hue turn of `dh_deg` degrees (an absolute value) that
/// does not pass `allowed_deg` weighs 0; one of `full_deg` or more weighs 1; in between the weight grows linearly. If `full_deg` is
/// not past `allowed_deg` there is no ramp and the weight is the step: 1 past `allowed_deg`, else 0.
#[must_use]
pub fn turn_weight(dh_deg: f64, allowed_deg: f64, full_deg: f64) -> f64 {
    if full_deg <= allowed_deg {
        return if dh_deg > allowed_deg { 1.0 } else { 0.0 };
    }
    ((dh_deg - allowed_deg) / (full_deg - allowed_deg)).clamp(0.0, 1.0)
}

/// The share of the sample that turned its hue further than it may: for every non-skin pixel whose photo C\* is at least 8, the
/// [`turn_weight`] of its Oklab hue turn against [`allowed_turn_deg`] of its C\*, summed and divided by **all** the sample's pixels.
#[must_use]
pub fn content_turn_share(
    px: &[Px],
    p_cstar: &[f64],
    p_okh: &[f64],
    not_skin: &[bool],
    (turn_deg, cref, cap, full): (f64, f64, f64, f64),
) -> f64 {
    if px.is_empty() {
        return 0.0;
    }
    let sum: f64 = px
        .iter()
        .enumerate()
        .filter(|&(i, _)| not_skin[i] && p_cstar[i] >= TURN_MIN_C)
        .map(|(i, q)| {
            let dh = wrap_deg(oklab_hue(q.ok) - p_okh[i]).abs();
            turn_weight(dh, allowed_turn_deg(p_cstar[i], turn_deg, cref, cap), full)
        })
        .sum();
    sum / px.len() as f64
}

/// The share of a frame a grade may clip or crush: the photo's own, plus at most `extra` where the reference has more.
#[must_use]
pub fn clip_allowance(photo: f64, reference: f64, extra: f64) -> f64 {
    photo.max(reference.min(photo + extra))
}

/// One side of the clip term: `((o - a - slack)+ / scale)^2`.
#[must_use]
pub fn clip_over(o: f64, a: f64) -> f64 {
    sq((o - a - CLIP_SLACK).max(0.0) / CLIP_SCALE)
}

/// What is measured on a candidate's picture, beyond [`Measure`].
#[derive(Debug, Clone, Default)]
pub struct CandidateMeasure {
    /// The candidate's measures.
    pub m: Measure,
    sec_shift: [f64; SECTORS],
    sec_dlogc: [f64; SECTORS],
    np_share: f64,
    pg_ab: Vec<[f64; 2]>,
    hl4_c: f64,
    hl4_d: f64,
    turn_share: f64,
}

/// The mean per-pixel C\* over the whites in `scope`, and the mean per-pixel distance of their (a\*, b\*) to the target. Opposite
/// tints do not cancel in either.
fn whites_stats(px: &[Px], scope: &[bool], white: [f64; 2]) -> (f64, f64) {
    let (mut c, mut d, mut n) = (0.0, 0.0, 0usize);
    for (q, _) in px.iter().zip(scope).filter(|&(_, &s)| s) {
        c += q.lab[1].hypot(q.lab[2]);
        d += (q.lab[1] - white[0]).hypot(q.lab[2] - white[1]);
        n += 1;
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        (c / n as f64, d / n as f64)
    }
}

/// Measures a candidate's sample (`px`, with the photo's `scene`) against the pair's fixed part.
///
/// # Panics
///
/// When `px` is not the sample's length.
#[must_use]
pub fn candidate_measure(px: &[Px], scene: &PixelScene, f: &Fixed) -> CandidateMeasure {
    assert_eq!(px.len(), f.sample_len, "the candidate's sample");
    let m = measure(px, scene, &f.excl);
    let mut shift = [0.0f64; SECTORS];
    let mut cnt = [0usize; SECTORS];
    let mut cs: [Vec<f64>; SECTORS] = Default::default();
    for (i, q) in px.iter().enumerate() {
        let s = f.sector[i];
        if s == 255 {
            continue;
        }
        let s = usize::from(s);
        let h = q.lab[2].atan2(q.lab[1]).to_degrees();
        shift[s] += wrap_deg(h - f.p_hue[i]);
        cnt[s] += 1;
        cs[s].push(q.lab[1].hypot(q.lab[2]));
    }
    let mut sec_shift = [0.0; SECTORS];
    let mut sec_dlogc = [0.0; SECTORS];
    for j in 0..SECTORS {
        if cnt[j] > 0 {
            sec_shift[j] = shift[j] / cnt[j] as f64;
            cs[j].sort_unstable_by(f64::total_cmp);
            let med = cs[j][cs[j].len() / 2];
            sec_dlogc[j] = med.max(1.0).ln() - f.p_sec_logc[j];
        }
    }
    // The share of the sample's non-skin pixels that are newly purple: Oklab hue in the band, C* at least 4, and either a grey in the
    // photo (C* under 2) or a hue that turned by at least `T.purple_rot_deg`.
    let rot = f.t.purple_rot_deg;
    let np = px
        .iter()
        .enumerate()
        .filter(|&(i, q)| {
            if !f.not_skin[i] || q.lab[1].hypot(q.lab[2]) < PURPLE_MIN_C {
                return false;
            }
            let h = oklab_hue(q.ok);
            (PURPLE_BAND.0..=PURPLE_BAND.1).contains(&h)
                && (f.p_cstar[i] < PURPLE_FROM_C || wrap_deg(h - f.p_okh[i]).abs() >= rot)
        })
        .count();
    let np_share = np as f64 / px.len() as f64;
    let pg_ab = pg_means(px, &f.pg);
    let (hl4_c, hl4_d) = whites_stats(px, &f.hl4_scope, f.t.white);
    let turn_share = content_turn_share(
        px,
        &f.p_cstar,
        &f.p_okh,
        &f.not_skin,
        (f.t.turn_deg, f.t.turn_cref, f.t.turn_cap, f.t.turn_full),
    );
    CandidateMeasure {
        m,
        sec_shift,
        sec_dlogc,
        np_share,
        pg_ab,
        hl4_c,
        hl4_d,
        turn_share,
    }
}

/// The objective's terms, unweighted (each is 0 for a perfect match), with the regulariser and the clip term.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Terms {
    /// Floor: the 2 % rank of L\*.
    pub floor: f64,
    /// Ceiling: the 98 % rank.
    pub ceiling: f64,
    /// Contrast: the relative inter-quartile range.
    pub contrast: f64,
    /// The photo's own median.
    pub anchor: f64,
    /// The tint of the near-neutrals, six channels.
    pub tint: f64,
    /// The overall saturation.
    pub saturation: f64,
    /// The sectors' hue.
    pub sector_hue: f64,
    /// The sectors' chroma.
    pub sector_chroma: f64,
    /// The elements' hue.
    pub element_hue: f64,
    /// The elements' chroma.
    pub element_chroma: f64,
    /// The elements' exposure.
    pub element_exposure: f64,
    /// The subject's separation.
    pub subject: f64,
    /// The frame's cast (unify).
    pub frame_cast: f64,
    /// The hue-drift wall.
    pub guard_hue: f64,
    /// The skin's absolute wall.
    pub guard_skin: f64,
    /// The sky's chroma wall.
    pub guard_sky: f64,
    /// The whites' chroma cap and hue wall.
    pub guard_whites: f64,
    /// The purple wall of the groups.
    pub guard_purple: f64,
    /// The whites' direction wall.
    pub guard_white_direction: f64,
    /// The skin's drift wall.
    pub guard_skin_drift: f64,
    /// The memory-colour wall.
    pub guard_memory: f64,
    /// The pixel guard of the purple wall.
    pub guard_purple_pixels: f64,
    /// The skin's chroma wall.
    pub guard_skin_chroma: f64,
    /// The content turn budget.
    pub guard_turn: f64,
    /// The share of the sample that is newly purple (a diagnostic).
    pub purple_share: f64,
    /// The skin's log chroma change (a diagnostic).
    pub skin_dlogc: f64,
    /// The global half of the regulariser: the mean of the squares of the 30 global numbers.
    pub reg_global: f64,
    /// The local half: the mean of the squares of the 19 local numbers.
    pub reg_local: f64,
    /// The clip term.
    pub clip: f64,
}

impl Terms {
    /// The full objective: the look terms, the walls, the regulariser and 3.0 times the clip term (`docs/34` section 5.3).
    #[must_use]
    pub fn total(&self) -> f64 {
        (self.floor + self.ceiling + self.anchor)
            + self.contrast
            + (self.tint + self.frame_cast)
            + (self.saturation + self.sector_chroma)
            + self.sector_hue
            + (self.element_hue + self.element_chroma)
            + (self.element_exposure + self.subject)
            + self.guard_hue
            + self.guard_skin
            + self.guard_sky
            + self.guard_whites
            + self.guard_purple
            + self.guard_white_direction
            + self.guard_skin_drift
            + self.guard_memory
            + self.guard_purple_pixels
            + self.guard_skin_chroma
            + self.guard_turn
            + self.common()
    }

    /// The regulariser and the clip term, which both objectives have.
    fn common(&self) -> f64 {
        REG_WEIGHT * self.reg_global + REG_WEIGHT * self.reg_local + CLIP_WEIGHT * self.clip
    }

    /// The look-only objective: the full one **without the walls and without the clip term** (the regulariser stays). A candidate the
    /// chain refused keeps its full total, so it stays worse than anything.
    #[must_use]
    pub fn look_only_total(&self) -> f64 {
        if self.clip >= REFUSED_CLIP {
            return self.total();
        }
        let mut t = *self;
        t.clip = 0.0;
        for v in [
            &mut t.guard_hue,
            &mut t.guard_skin,
            &mut t.guard_sky,
            &mut t.guard_whites,
            &mut t.guard_purple,
            &mut t.guard_white_direction,
            &mut t.guard_skin_drift,
            &mut t.guard_memory,
            &mut t.guard_purple_pixels,
            &mut t.guard_skin_chroma,
            &mut t.guard_turn,
        ] {
            *v = 0.0;
        }
        t.total()
    }

    /// The sum of the walls and of the clip term at its weight: what [`Terms::look_only_total`] leaves out.
    #[must_use]
    pub fn walls_and_clip(&self) -> f64 {
        self.guard_hue
            + self.guard_skin
            + self.guard_sky
            + self.guard_whites
            + self.guard_purple
            + self.guard_white_direction
            + self.guard_skin_drift
            + self.guard_memory
            + self.guard_purple_pixels
            + self.guard_skin_chroma
            + self.guard_turn
            + CLIP_WEIGHT * self.clip
    }

    /// The objective's named parts as they enter [`Terms::total`], for a report.
    #[must_use]
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("floor", self.floor),
            ("ceiling", self.ceiling),
            ("anchor", self.anchor),
            ("contrast", self.contrast),
            ("tint", self.tint),
            ("frame_cast", self.frame_cast),
            ("saturation", self.saturation),
            ("sector_chroma", self.sector_chroma),
            ("sector_hue", self.sector_hue),
            ("element_hue", self.element_hue),
            ("element_chroma", self.element_chroma),
            ("element_exposure", self.element_exposure),
            ("subject", self.subject),
            ("guard_hue", self.guard_hue),
            ("guard_skin", self.guard_skin),
            ("guard_sky", self.guard_sky),
            ("guard_whites", self.guard_whites),
            ("guard_purple", self.guard_purple),
            ("guard_white_direction", self.guard_white_direction),
            ("guard_skin_drift", self.guard_skin_drift),
            ("guard_memory", self.guard_memory),
            ("guard_purple_pixels", self.guard_purple_pixels),
            ("guard_skin_chroma", self.guard_skin_chroma),
            ("guard_turn", self.guard_turn),
            ("regulariser_global", REG_WEIGHT * self.reg_global),
            ("regulariser_local", REG_WEIGHT * self.reg_local),
            ("clip", CLIP_WEIGHT * self.clip),
        ]
    }
}

/// The clip term of a candidate (`docs/34` section 5.4, P5).
#[must_use]
pub fn clip_term(f: &Fixed, m: &Measure) -> f64 {
    let extra = f.t.clip_extra;
    clip_over(m.hi, clip_allowance(f.p.hi, f.r.hi, extra))
        + clip_over(m.lo25, clip_allowance(f.p.lo25, f.r.lo25, extra))
}

/// The terms of a candidate: the look terms of section 5.3 and the walls of section 5.4. The regulariser and the clip term are set
/// by the caller.
#[must_use]
pub fn terms(c: &CandidateMeasure, f: &Fixed) -> Terms {
    let o = &f.t;
    let m = &c.m;
    let mut t = Terms {
        purple_share: c.np_share,
        ..Terms::default()
    };
    {
        let (pe, me) = (&f.p.el[5], &m.el[5]);
        if pe.share >= SKIN_CHROMA_MIN_SHARE && me.n > 0 {
            t.skin_dlogc = (me.mv().1.max(1.0) / pe.mv().1.max(1.0)).ln();
        }
    }
    t.floor = FLOOR.2 * term(m.p2 - o.p2, FLOOR.0, FLOOR.1);
    t.ceiling = CEILING.2 * term(m.p98 - o.p98, CEILING.0, CEILING.1);
    t.contrast = CONTRAST.2 * term(m.iqr_rel - o.iqr_rel, CONTRAST.0, CONTRAST.1);
    t.anchor = ANCHOR.2 * term(m.p50 - o.p50, ANCHOR.0, ANCHOR.1);
    for b in 0..3 {
        for ch in 0..2 {
            t.tint += TINT_BAND_WEIGHT[b] * term(m.tint[b][ch] - o.tint[b][ch], TINT.0, TINT.1);
        }
    }
    t.saturation = SATURATION.2 * term(m.logsat - o.logsat, SATURATION.0, SATURATION.1);
    for j in 0..SECTORS {
        let g = o.sector_gate[j];
        if g <= 0.0 {
            continue;
        }
        let w = g * f.p.sec[j].share.sqrt();
        t.sector_hue +=
            SECTOR_HUE.2 * w * term(c.sec_shift[j] - o.sector_dh[j], SECTOR_HUE.0, SECTOR_HUE.1);
        t.sector_chroma += SECTOR_CHROMA.2
            * w
            * term(
                c.sec_dlogc[j] - o.sector_dlogc[j],
                SECTOR_CHROMA.0,
                SECTOR_CHROMA.1,
            );
    }
    for k in 0..6 {
        let (gm, gh) = (o.matched[k], o.hue_gate[k]);
        let w = ELEMENT_WEIGHT[k] * f.p.el[k].share.sqrt();
        let em = &m.el[k];
        if gh > 0.0 && em.has_hue {
            t.element_hue +=
                w * gh * term(wrap_deg(em.h - o.el_h[k]), ELEMENT_HUE.0, ELEMENT_HUE.1);
        }
        if gm > 0.0 {
            t.element_chroma +=
                w * gm * term(em.logc - o.el_logc[k], ELEMENT_CHROMA.0, ELEMENT_CHROMA.1);
            t.element_exposure += w
                * gm
                * term(
                    (em.l - m.p50) - o.el_off[k],
                    ELEMENT_EXPOSURE.0,
                    ELEMENT_EXPOSURE.1,
                );
        }
    }
    let g = o.unify;
    if g > 0.0 {
        let e = |got: f64, target: f64| sq(dead(got - target, FRAME_CAST.1) / FRAME_CAST.0);
        t.frame_cast = FRAME_CAST.2 * g * (e(m.cast_a, o.cast[0]) + e(m.cast_b, o.cast[1]));
    }
    let gs = o.subj;
    if gs > 0.0 {
        t.subject = SUBJECT.2 * gs * term(m.sep - o.sep, SUBJECT.0, SUBJECT.1);
    }
    walls(&mut t, c, f);
    // The pixel guard of the purple wall.
    if c.np_share > o.purple_max_share {
        t.guard_purple_pixels =
            PURPLE_PIXELS.0 * sq((c.np_share - o.purple_max_share) / PURPLE_PIXELS.1);
    }
    t
}

/// The hue-drift wall of one group: `2 * ((|d| - limit)+ / 5)^2`.
fn drift(d: f64, limit: f64) -> f64 {
    if d.abs() > limit {
        DRIFT_WEIGHT * sq((d.abs() - limit) / DRIFT_SCALE_DEG)
    } else {
        0.0
    }
}

/// The walls: every group's hue is that of its mean Oklab (a, b) vector, and its chroma threshold that of its mean CIELAB vector.
fn walls(t: &mut Terms, c: &CandidateMeasure, f: &Fixed) {
    let o = &f.t;
    let m = &c.m;
    let wall = o.wall_deg;
    let vec_hc = |ab: [f64; 2]| (circ_mean_deg(ab[0], ab[1]), ab[0].hypot(ab[1]));
    let hues = |gid: usize| (oklab_hue(f.p_pg_ab[gid]), oklab_hue(c.pg_ab[gid]));
    // The groups that count in the purple wall: their mean-vector chroma and their number.
    let mut groups: Vec<(f64, usize)> = Vec::new();
    let n = f.sample_len as f64;
    for (k, memory) in (0..6).zip([Some(0), Some(1), None, None, None, None]) {
        let (pe, me) = (&f.p.el[k], &m.el[k]);
        if pe.share < PRESENT_SHARE || me.n == 0 {
            continue;
        }
        let (_, pc) = pe.mv();
        if pc < MIN_MEAN_VECTOR {
            continue;
        }
        let (ph, ch) = hues(k);
        let limit = if o.hue_gate[k] > GATE_OPEN {
            wall.max(ELEMENT_CAP_DEG[k] + DRIFT_EXTRA_DEG)
        } else {
            wall
        };
        t.guard_hue += drift(wrap_deg(ch - ph), limit);
        groups.push((pc, k));
        // The memory-colour wall: sky and foliage hold their hue within the network's limit.
        if let Some(i) = memory {
            let excess = (wrap_deg(ch - ph).abs() - o.mem_deg[i]).max(0.0);
            t.guard_memory += MEMORY.0 * sq(excess / MEMORY.1);
        }
    }
    let (pcl, mcl) = (&f.p.clothing, &m.clothing);
    if pcl.share >= PRESENT_SHARE && mcl.n > 0 {
        let (_, pc) = pcl.mv();
        if pc >= MIN_MEAN_VECTOR {
            let (ph, ch) = hues(PG_CLOTHING);
            t.guard_hue += drift(wrap_deg(ch - ph), wall);
            groups.push((pc, PG_CLOTHING));
        }
    }
    for j in 0..SECTORS {
        if f.p.sec[j].share >= SECTOR_GROUP_MIN_SHARE && f.p.sec[j].n > 0 {
            let (_, pc) = vec_hc(f.p_sec_ab[j]);
            if pc >= MIN_MEAN_VECTOR {
                let (ph, ch) = hues(PG_SECTOR + j);
                t.guard_hue += drift(wrap_deg(ch - ph), wall);
                groups.push((pc, PG_SECTOR + j));
            }
        }
    }
    let n_hl = f.hl_scope.iter().filter(|&&s| s).count() as f64;
    if n_hl >= WHITES_MIN_SHARE * n {
        let (_, pc) = vec_hc(f.p_hl_ab);
        // The cap is on the mean per-pixel chroma of the whites.
        t.guard_whites = WHITES_CAP.0 * sq((c.hl4_c - o.hl_cmax).max(0.0) / WHITES_CAP.1);
        if pc >= WHITES_HUE_MIN_C {
            let (ph, ch) = hues(PG_WHITES);
            t.guard_whites += drift(wrap_deg(ch - ph), wall);
        }
        if pc >= MIN_MEAN_VECTOR {
            groups.push((pc, PG_WHITES));
        }
        // The whites' direction: the mean per-pixel distance to the reference-given tint.
        let d = c.hl4_d;
        if d > o.white_tol {
            t.guard_white_direction =
                WHITES_DIRECTION.0 * sq((d - o.white_tol) / WHITES_DIRECTION.1);
        }
    }
    let n_dark = f.dark_scope.iter().filter(|&&s| s).count() as f64;
    if n_dark >= WHITES_MIN_SHARE * n {
        let (_, pc) = vec_hc(f.p_dark_ab);
        if pc >= MIN_MEAN_VECTOR {
            let (ph, ch) = hues(PG_DARK);
            t.guard_hue += drift(wrap_deg(ch - ph), wall);
            groups.push((pc, PG_DARK));
        }
    }
    // The purple wall, with its approach term: a turn counts wherever it ends in the band; one that is heading for it counts less.
    let ok = 1.0 - o.purple_ok;
    let (lo, hi) = PURPLE_BAND;
    let rot = o.purple_rot_deg;
    let in_band = |h: f64| (lo..=hi).contains(&h.rem_euclid(360.0));
    let dist = |h: f64| {
        if in_band(h) {
            0.0
        } else {
            wrap_deg(h - lo).abs().min(wrap_deg(h - hi).abs())
        }
    };
    for (pc, gid) in groups {
        if pc < MIN_MEAN_VECTOR {
            continue;
        }
        let (ph, ch) = hues(gid);
        if in_band(ch) {
            let excess = (wrap_deg(ch - ph).abs() - rot).max(0.0);
            t.guard_purple += PURPLE_TURN.0 * sq(excess / PURPLE_TURN.1);
            continue;
        }
        if in_band(ph) {
            continue;
        }
        let ch = ch.rem_euclid(360.0);
        if dist(ch) < PURPLE_REACH_DEG && dist(ch) < dist(ph) {
            t.guard_purple +=
                PURPLE_APPROACH_WEIGHT * sq((PURPLE_REACH_DEG - dist(ch)) / DRIFT_SCALE_DEG) * ok;
        }
    }
    // Skin: the absolute wall and the drift wall, on the mean vector, and the chroma wall (the three of Skin protection).
    let (pe, me) = (&f.p.el[5], &m.el[5]);
    if pe.share >= PRESENT_SHARE && me.n > 0 {
        let ((_, pc), (h, chroma)) = (pe.mv(), me.mv());
        let eh = (SKIN_HUE.0 - h).max(h - SKIN_HUE.1).max(0.0);
        let ec = (SKIN_CHROMA.0 - chroma)
            .max(chroma - SKIN_CHROMA.1)
            .max(0.0);
        if f.skin_protection {
            t.guard_skin =
                SKIN_WEIGHT * (sq(eh / SKIN_ABSOLUTE_SCALE) + sq(ec / SKIN_ABSOLUTE_SCALE));
        }
        if pc >= MIN_MEAN_VECTOR && f.skin_protection {
            let (ph, h) = hues(5);
            let excess = (wrap_deg(h - ph).abs() - o.skin_drift).max(0.0);
            t.guard_skin_drift = SKIN_WEIGHT * sq(excess / SKIN_DRIFT_SCALE);
        }
        if pe.share >= SKIN_CHROMA_MIN_SHARE && f.skin_protection {
            let excess = (t.skin_dlogc - o.skin_dlogc_max).max(0.0);
            t.guard_skin_chroma = SKIN_WEIGHT * sq(excess / SKIN_CHROMA_SCALE);
        }
    }
    // The content turn budget.
    if c.turn_share > o.turn_share {
        t.guard_turn = TURN.0 * sq((c.turn_share - o.turn_share) / TURN.1);
    }
    // The sky chroma wall, on the mean-vector chroma.
    if f.p.el[0].share >= PRESENT_SHARE {
        let limit = (SKY_FACTOR * f.p.el[0].mv().1).max(o.el_logc[0].exp());
        let excess = (m.el[0].mv().1 - limit).max(0.0);
        t.guard_sky = SKY_WEIGHT * sq(excess / SKY_SCALE);
    }
}
