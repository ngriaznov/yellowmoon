//! The synthetic protocol of `docs/27` S-12: skin patches in ten tones with known lesions, moles,
//! freckles and hair-like lines, run through the real entry points (`prepare_skin`,
//! `retouch_skin`) with the classical proposer and detector, and scored against gates.
//!
//! The research (`research/reports/Skin retouching models and data.md`, "Masked synthetic metrics
//! and a blind A/B replace PSNR") asks for: the share of inserted optical density removed, no
//! change outside every mask, mean skin colour moving by no more than |dL*| 0.5 and |da*|, |db*| 1,
//! mole survival of at least 90 % at the default amount, and a pore-scale band-pass energy ratio of
//! at least 0.95. Those numbers are used where they apply; the others are reasoned and written at
//! the gates below.
//!
//! Everything here is seeded: the same code gives the same numbers on every run and every machine.

use rayon::prelude::*;

use super::grey::Grey;
use super::synth::{Rng, clean_skin_rough, clean_skin_tone};
use super::{ClassicalBeautyMarks, ClassicalProposer};
use crate::mock::MockMesh;
use crate::models::{FaceMesh, FaceParser, FacePart, MeshTable, Models};
use crate::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, RetouchCache, SkinSettings, prepare_skin,
    retouch_skin,
};

/// A lesion counts as visible, and is scored for removal, from this optical density (the sum over
/// the channels of `-ln(lesion / skin)` at its core): about a 10 % drop in each channel. Fainter
/// ones are noise-level for a detector that must leave 3 % skin texture alone.
pub const VISIBLE_OD: f64 = 0.40;

/// Side of a patch: the patch is one face of 450 px, the size of the blemish map.
pub const N: usize = 450;

/// How many skin tones the protocol runs (Monk 1 to 10 in spirit).
pub const TONES: usize = 10;

/// The linear AWG4 colour of the mean skin of tone `t` (0 = very light, 9 = very dark).
///
/// Luminance `Y` falls geometrically from 0.55 to 0.06 (scene-referred, mid grey 0.18, so L* from
/// about 79 to 28); the red-to-green ratio rises from 1.35 to 1.70 and the blue-to-green ratio
/// falls from 0.78 to 0.63, because melanin absorbs blue most. Reasoned values, not measurements.
pub fn tone_rgb(t: usize) -> [f64; 3] {
    let u = t as f64 / (TONES - 1) as f64;
    let y = 0.55 * (0.06f64 / 0.55).powf(u);
    let rg = 1.35 + 0.35 * u;
    let bg = 0.78 - 0.15 * u;
    // AWG4 luminance row (ym_color::gamut): Y = 0.2545 R + 0.7815 G - 0.0360 B.
    let g = y / (0.2545 * rg + 0.7815 - 0.0360 * bg);
    [g * rg, g, g * bg]
}

/// What an item is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Haemoglobin papule.
    Papule,
    /// Papule with a pale core.
    Pustule,
    /// Flat brown post-acne mark.
    Brown,
    /// Dark brown mole, to keep.
    Mole,
    /// Faint small freckle, to keep (partly).
    Freckle,
    /// Thin dark hair-like line, to keep.
    Hair,
}

impl Class {
    /// A lesion the node should remove.
    pub fn is_lesion(self) -> bool {
        matches!(self, Class::Papule | Class::Pustule | Class::Brown)
    }
    /// Full-strength factors per channel (linear light) on the mean skin.
    fn factors(self) -> [f64; 3] {
        match self {
            Class::Papule | Class::Pustule => [0.97, 0.72, 0.80],
            Class::Brown => [0.82, 0.68, 0.60],
            Class::Mole => [0.50, 0.38, 0.30],
            Class::Freckle => [0.88, 0.80, 0.72],
            Class::Hair => [0.62, 0.58, 0.55],
        }
    }
    /// How much of the contrast survives on tone `t`: haemoglobin lesions fade on deep skin
    /// (melanin hides the redness; to 0.6), melanin ones fade a little (to 0.8). Reasoned.
    fn tone_scale(self, t: usize) -> f64 {
        let u = t as f64 / (TONES - 1) as f64;
        match self {
            Class::Papule | Class::Pustule => 1.0 - 0.4 * u,
            Class::Brown => 1.0 - 0.1 * u,
            Class::Mole | Class::Freckle => 1.0 - 0.2 * u,
            Class::Hair => 1.0,
        }
    }
}

/// One thing painted on a patch.
#[derive(Debug, Clone)]
pub struct Item {
    /// Its class.
    pub class: Class,
    /// Centre in pixels (the midpoint for a hair).
    pub cx: f64,
    /// Centre in pixels.
    pub cy: f64,
    /// Core radius in pixels (for a hair: half its length, for exclusion only).
    pub radius: f64,
    /// Contrast multiplier, 0..1.
    pub amp: f64,
    /// Pixels of the footprint: index, weight 0..1 and distance over radius (hair: 0).
    pub foot: Vec<(usize, f64, f64)>,
}

/// Soft edge: 1 inside 0.75 R, 0 beyond 1.25 R (the shape of `synth::Lesion`).
fn soft(dist: f64, r: f64) -> f64 {
    let t = ((1.25 * r - dist) / (0.5 * r)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn disc_item(class: Class, cx: f64, cy: f64, radius: f64, amp: f64) -> Item {
    let mut foot = Vec::new();
    let r = (1.3 * radius).ceil() as i64 + 1;
    for y in (cy as i64 - r).max(0)..(cy as i64 + r + 1).min(N as i64) {
        for x in (cx as i64 - r).max(0)..(cx as i64 + r + 1).min(N as i64) {
            let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            let w = soft(d, radius);
            if w > 0.0 {
                foot.push((y as usize * N + x as usize, w, d / radius));
            }
        }
    }
    Item {
        class,
        cx,
        cy,
        radius,
        amp,
        foot,
    }
}

/// A curved hair: a quadratic Bezier of `len` px through the centre, 1.2 px wide with a soft edge.
fn hair_item(rng: &mut Rng, cx: f64, cy: f64, len: f64) -> Item {
    let a = rng.range(0.0, std::f64::consts::PI);
    let bend = rng.range(-0.25, 0.25) * len;
    let (dx, dy) = (a.cos(), a.sin());
    let p0 = (cx - 0.5 * len * dx, cy - 0.5 * len * dy);
    let p2 = (cx + 0.5 * len * dx, cy + 0.5 * len * dy);
    let p1 = (cx - bend * dy, cy + bend * dx);
    let pts: Vec<(f64, f64)> = (0..=(2.0 * len) as usize)
        .map(|i| {
            let t = i as f64 / (2.0 * len);
            let (u, v) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t));
            (
                u * p0.0 + v * p1.0 + t * t * p2.0,
                u * p0.1 + v * p1.1 + t * t * p2.1,
            )
        })
        .collect();
    let mut best: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &(px, py) in &pts {
        for y in (py as i64 - 2).max(0)..(py as i64 + 3).min(N as i64) {
            for x in (px as i64 - 2).max(0)..(px as i64 + 3).min(N as i64) {
                let d = ((x as f64 + 0.5 - px).powi(2) + (y as f64 + 0.5 - py).powi(2)).sqrt();
                let w = ((1.3 - d) / 0.8).clamp(0.0, 1.0);
                if w > 0.0 {
                    let e = best.entry(y as usize * N + x as usize).or_insert(0.0);
                    *e = e.max(w);
                }
            }
        }
    }
    Item {
        class: Class::Hair,
        cx,
        cy,
        radius: 0.5 * len,
        amp: 1.0,
        foot: best.into_iter().map(|(i, w)| (i, w, 0.0)).collect(),
    }
}

/// Which patch of a tone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    /// Papules, pustules and brown marks of several sizes and contrasts, plus a mole beside a papule.
    Lesions,
    /// Moles, freckles and hair-like lines.
    Keepers,
    /// Skin only, with a broad flush (a cheek's redness, 2 % of the face wide: not a blemish).
    Clean,
    /// Skin only, with twice the texture (3 %) and 100 deep pores: nothing here is a blemish.
    Rough,
    /// Skin with two dark under-eye crescents where the mock mesh puts its under-eye zones.
    Circles,
}

/// A patch: the clean skin, the same skin with the items, and the items.
pub struct Patch {
    /// Which scene.
    pub scene: Scene,
    /// Skin without items.
    pub clean: LinearAwg4Image,
    /// Skin with items.
    pub img: LinearAwg4Image,
    /// The items.
    pub items: Vec<Item>,
    /// Pixels farther than `2.4 R + 8` from every item and 60 px inside the borders.
    pub far: Vec<bool>,
    /// The blotches of the clean scene as a relative luminance change per pixel (else empty).
    pub blotch: Vec<f64>,
}

fn place(rng: &mut Rng, placed: &[Item], radius: f64, margin: f64, gap: f64) -> (f64, f64) {
    for _ in 0..4000 {
        let (x, y) = (
            rng.range(margin, N as f64 - margin),
            rng.range(margin, N as f64 - margin),
        );
        if placed
            .iter()
            .all(|o| ((o.cx - x).powi(2) + (o.cy - y).powi(2)).sqrt() >= gap + o.radius + radius)
        {
            return (x, y);
        }
    }
    panic!("no room for an item")
}

/// Builds the patch of tone `t` and `scene`. The seed depends on the tone and the scene only.
pub fn patch(t: usize, scene: Scene) -> Patch {
    let seed = 9000 + 17 * t as u64 + scene as u64;
    let mut clean_px = match scene {
        Scene::Rough => clean_skin_rough(N, N, seed, tone_rgb(t), 2.0, 100, 0.2),
        _ => clean_skin_tone(N, N, seed, tone_rgb(t)),
    };
    let mut blotch = Vec::new();
    if scene == Scene::Circles {
        // The mock mesh's zones are ellipses of 0.14 W by 0.045 H, 0.09 H below eyes at (0.32, 0.40) and
        // (0.68, 0.40). The dark crescent is the upper part of each zone (0.11 W by 0.025 H, 9 px
        // higher): 12 % less light, more so in blue (a bluish-brown tint). The tool moves the zone to
        // its own average, so the lighter lower part of the zone is what the crescent is lifted toward.
        for (i, p) in clean_px.iter_mut().enumerate() {
            let (x, y) = ((i % N) as f64, (i / N) as f64);
            let f: f64 = [0.32, 0.68]
                .iter()
                .map(|&ex| {
                    let (dx, dy) = (
                        (x - ex * N as f64) / (0.11 * N as f64),
                        (y - 0.49 * N as f64 + 9.0) / (0.025 * N as f64),
                    );
                    (-(dx * dx + dy * dy) * 1.5).exp()
                })
                .fold(0.0, f64::max);
            blotch.push(-0.12 * f);
            for (c, k) in [0.8, 1.0, 1.2].iter().enumerate() {
                p[c] *= 1.0 - 0.12 * f * k;
            }
        }
    }
    if scene == Scene::Clean {
        // Blotches: six Gaussian patches of 8 to 14 px sigma (2 to 3 % of the face), 5 to 10 % darker in all
        // channels (uneven light and tone: what Even skin is for).
        let mut brng = Rng(seed ^ 0x5151);
        let blobs: Vec<(f64, f64, f64, f64)> = (0..6)
            .map(|k| {
                (
                    brng.range(90.0, 360.0),
                    brng.range(90.0, 360.0),
                    brng.range(8.0, 14.0),
                    -brng.range(0.05, 0.10) + 0.0 * k as f64,
                )
            })
            .collect();
        for (i, p) in clean_px.iter_mut().enumerate() {
            let (x, y) = ((i % N) as f64, (i / N) as f64);
            let f: f64 = blobs
                .iter()
                .map(|&(bx, by, sg, amp)| {
                    amp * (-((x - bx).powi(2) + (y - by).powi(2)) / (2.0 * sg * sg)).exp()
                })
                .sum();
            blotch.push(f);
            for c in p.iter_mut() {
                *c *= 1.0 + f;
            }
        }
        // A flush: Gaussian, sigma 28 px (6 % of the face), 10 % less green and blue.
        for (i, p) in clean_px.iter_mut().enumerate() {
            let d2 = ((i % N) as f64 - 300.0).powi(2) + ((i / N) as f64 - 170.0).powi(2);
            let f = 1.0 - 0.10 * (-d2 / (2.0 * 28.0 * 28.0)).exp();
            p[1] *= f;
            p[2] *= f;
        }
    }
    let mut rng = Rng(seed ^ 0xA5A5);
    let mut items: Vec<Item> = Vec::new();
    let (margin, gap) = (62.0, 30.0);
    match scene {
        Scene::Lesions => {
            // (class, radius px, amp): radii 1.6 to 7.8 px are diameters of 0.7 to 3.5 % of the face, the band
            // of `docs/27` section 6.
            let list = [
                (Class::Papule, 1.6, 1.0),
                (Class::Papule, 2.2, 1.0),
                (Class::Papule, 3.2, 0.8),
                (Class::Papule, 4.6, 1.0),
                (Class::Papule, 6.5, 0.6),
                (Class::Pustule, 3.0, 1.0),
                (Class::Pustule, 4.5, 0.8),
                (Class::Brown, 2.2, 1.0),
                (Class::Brown, 3.5, 0.7),
                (Class::Brown, 5.0, 0.9),
                (Class::Brown, 7.8, 0.8),
            ];
            for (class, r, amp) in list {
                let (x, y) = place(&mut rng, &items, r, margin, gap);
                items.push(disc_item(class, x, y, r, amp * class.tone_scale(t)));
            }
            // A mole 42 px from a papule: the papule goes, the mole stays.
            let (px, py) = (items[3].cx, items[3].cy);
            let mut pos = None;
            for k in 0..16 {
                let a = k as f64 * 0.39;
                let (x, y) = (px + 42.0 * a.cos(), py + 42.0 * a.sin());
                if x > 55.0
                    && y > 55.0
                    && x < N as f64 - 55.0
                    && y < N as f64 - 55.0
                    && items.iter().all(|o| {
                        ((o.cx - x).powi(2) + (o.cy - y).powi(2)).sqrt() >= 30.0 + o.radius + 5.5
                    })
                {
                    pos = Some((x, y));
                    break;
                }
            }
            let (x, y) = pos.unwrap_or_else(|| place(&mut rng, &items, 5.5, margin, gap));
            items.push(disc_item(Class::Mole, x, y, 5.5, Class::Mole.tone_scale(t)));
        }
        Scene::Keepers => {
            for (r, amp) in [(4.0, 1.0), (6.0, 0.9), (8.0, 1.0)] {
                let (x, y) = place(&mut rng, &items, r, margin, gap);
                items.push(disc_item(
                    Class::Mole,
                    x,
                    y,
                    r,
                    amp * Class::Mole.tone_scale(t),
                ));
            }
            for k in 0..6 {
                let r = 1.3 + 0.14 * k as f64;
                let (x, y) = place(&mut rng, &items, r, margin, gap);
                let amp = 0.35 + 0.03 * k as f64;
                items.push(disc_item(
                    Class::Freckle,
                    x,
                    y,
                    r,
                    amp * Class::Freckle.tone_scale(t),
                ));
            }
            for len in [60.0, 75.0, 90.0] {
                let (x, y) = place(&mut rng, &items, 0.5 * len, margin + 30.0, 4.0);
                items.push(hair_item(&mut rng, x, y, len));
            }
        }
        Scene::Clean | Scene::Rough | Scene::Circles => {}
    }
    let mut px = clean_px.clone();
    for it in &items {
        let f = it.class.factors();
        for &(i, w, d) in &it.foot {
            for c in 0..3 {
                let mut factor = 1.0 - w * it.amp * (1.0 - f[c]);
                if it.class == Class::Pustule && d < 0.5 && c > 0 {
                    // The pale core: a lighter centre inside the red ring.
                    let core = ((0.5 - d) / 0.2).clamp(0.0, 1.0);
                    factor *= 1.0 + 0.2 * core;
                }
                px[i][c] *= factor;
            }
        }
    }
    let to_img = |v: &[[f64; 3]]| LinearAwg4Image {
        width_px: N as u32,
        height_px: N as u32,
        data: v
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect(),
    };
    let mut far = vec![false; N * N];
    for y in 60..N - 60 {
        far[y * N + 60..y * N + N - 60].fill(true);
    }
    for o in &items {
        if o.class == Class::Hair {
            for &(i, _, _) in &o.foot {
                let (fx, fy) = ((i % N) as i64, (i / N) as i64);
                for y in (fy - 12).max(0)..=(fy + 12).min(N as i64 - 1) {
                    for x in (fx - 12).max(0)..=(fx + 12).min(N as i64 - 1) {
                        far[y as usize * N + x as usize] = false;
                    }
                }
            }
        } else {
            let reach = 2.4 * o.radius + 8.0;
            let r = reach.ceil() as i64;
            for y in (o.cy as i64 - r).max(0)..=(o.cy as i64 + r).min(N as i64 - 1) {
                for x in (o.cx as i64 - r).max(0)..=(o.cx as i64 + r).min(N as i64 - 1) {
                    if (o.cx - x as f64).hypot(o.cy - y as f64) <= reach {
                        far[y as usize * N + x as usize] = false;
                    }
                }
            }
        }
    }
    Patch {
        scene,
        clean: to_img(&clean_px),
        img: to_img(&px),
        items,
        far,
        blotch,
    }
}

// ---------------------------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------------------------

/// A parser that finds skin everywhere in the face rectangle (and no eyes or mouth).
struct FlatSkin;

impl FaceParser for FlatSkin {
    fn part_mask(
        &self,
        _image: &LinearAwg4Image,
        face: &Face,
        part: FacePart,
    ) -> crate::Result<Option<BoundedMask>> {
        Ok(match part {
            FacePart::SkinWithoutEars => Some(BoundedMask::new(
                face.rect,
                Mask::new(
                    face.rect.width_px,
                    face.rect.height_px,
                    vec![255; (face.rect.width_px * face.rect.height_px) as usize],
                )?,
            )?),
            _ => None,
        })
    }
}

/// A mesh with both tables empty: no under-eye zone and no feature to keep Even skin off, so the
/// patches (which have neither) are scored on the tools themselves and not on where a mock draws
/// its zones. Dark circles are scored on their own with [`MockMesh`].
struct QuietMesh;

impl FaceMesh for QuietMesh {
    fn render(
        &self,
        _image: &LinearAwg4Image,
        face: &Face,
        _table: MeshTable,
        _size_px: (u32, u32),
    ) -> crate::Result<BoundedMask> {
        BoundedMask::new(
            face.rect,
            Mask::new(
                face.rect.width_px,
                face.rect.height_px,
                vec![0; (face.rect.width_px * face.rect.height_px) as usize],
            )?,
        )
    }
}

/// The models of the protocol: flat skin, an empty mesh (or the mock mesh when `mock_mesh`), the
/// classical detector and proposer.
pub fn models(mock_mesh: bool) -> Models {
    Models {
        parser: Box::new(FlatSkin),
        mesh: if mock_mesh {
            Box::new(MockMesh)
        } else {
            Box::new(QuietMesh)
        },
        beauty_marks: Some(Box::new(ClassicalBeautyMarks)),
        proposer: Box::new(ClassicalProposer::new(N as f64)),
    }
}

/// The face of a patch: the whole patch.
pub fn face() -> Face {
    Face {
        rect: Rect::new(0, 0, N as u32, N as u32),
        scale: 1.0,
    }
}

/// A patch with its masks and blemish map prepared once.
pub struct Prepared {
    /// The patch.
    pub patch: Patch,
    models: Models,
    cache: RetouchCache,
}

impl Prepared {
    /// Prepares `patch`.
    pub fn new(patch: Patch) -> Self {
        Self::with_mesh(patch, false)
    }

    /// Prepares `patch` with the mock mesh's tables (under-eye zones, a mouth to keep off).
    pub fn with_mesh(patch: Patch, mock_mesh: bool) -> Self {
        let models = models(mock_mesh);
        let mut cache = RetouchCache::new(1, 1);
        let faces = [(face(), SkinSettings::default())];
        prepare_skin(&patch.img, &faces, &models, &mut cache).unwrap();
        Self {
            patch,
            models,
            cache,
        }
    }

    /// The patch retouched with `s`.
    pub fn run(&mut self, s: SkinSettings) -> LinearAwg4Image {
        self.run_on(&self.patch.img.clone(), s)
    }

    /// An image of the same size (another grade of the patch) retouched with `s`, using the maps
    /// prepared on the patch.
    pub fn run_on(&mut self, image: &LinearAwg4Image, s: SkinSettings) -> LinearAwg4Image {
        retouch_skin(
            image,
            (N as u32, N as u32),
            &[(face(), s)],
            1.0,
            &self.models,
            &mut self.cache,
            None,
        )
        .unwrap()
    }
}

// ---------------------------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------------------------

fn luma(p: [f32; 3]) -> f64 {
    // Rec. 709 luma of the channels: only used as a scalar for texture and halos.
    0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
}

fn mean_of(img: &LinearAwg4Image, idx: &[usize]) -> [f64; 3] {
    let mut s = [0.0; 3];
    for &i in idx {
        for (acc, v) in s.iter_mut().zip(img.data[i]) {
            *acc += f64::from(v);
        }
    }
    s.map(|v| v / idx.len().max(1) as f64)
}

/// Optical density of `img` against `clean` over `idx`: the sum over channels of
/// `-ln(mean(img) / mean(clean))`. Positive where `img` is darker.
pub fn optical_density(img: &LinearAwg4Image, clean: &LinearAwg4Image, idx: &[usize]) -> f64 {
    let (a, b) = (mean_of(img, idx), mean_of(clean, idx));
    (0..3).map(|c| -(a[c] / b[c]).ln()).sum()
}

/// Sigma of the high-pass on clean skin: the 2 px grain and the pores of the synthetic skin are
/// below it, the skin's gradient is far above it.
pub const SKIN_BAND_SIGMA_PX: f64 = 3.0;

/// Sigma of the high-pass inside lesions: the fill keeps only the detail finer than the Gaussian
/// of the proposer (1.35 px at 450 px), so the band is finer than on clean skin.
pub const LESION_BAND_SIGMA_PX: f64 = 1.5;

/// High-pass of the log luminance: the value minus its Gaussian of sigma `sigma_px`.
pub fn high_pass(img: &LinearAwg4Image, sigma_px: f64) -> Vec<f64> {
    let l: Vec<f64> = img.data.iter().map(|&p| luma(p).max(1e-6).ln()).collect();
    let blur = Grey::new(N, N, l.clone()).gaussian(sigma_px);
    l.iter().zip(&blur.data).map(|(a, b)| a - b).collect()
}

fn energy(v: &[f64], idx: &[usize]) -> f64 {
    idx.iter().map(|&i| v[i] * v[i]).sum()
}

fn rms(v: &[f64], idx: &[usize]) -> f64 {
    (idx.iter().map(|&i| v[i] * v[i]).sum::<f64>() / idx.len().max(1) as f64).sqrt()
}

/// CIELAB (D65 white at Y = 1) of a linear AWG4 pixel.
pub fn lab(p: [f32; 3]) -> [f64; 3] {
    let m = ym_color::gamut::matrices().awg4_to_xyz;
    let rgb = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
    let xyz = [0, 1, 2].map(|r| (0..3).map(|c| m[r][c] * rgb[c]).sum::<f64>());
    ym_color::deltae::xyz_to_lab(xyz, ym_color::deltae::d65_white_xyz())
}

/// What the protocol measures on one lesion-like item for one setting.
#[derive(Debug, Clone, Copy)]
pub struct ItemStat {
    /// Class.
    pub class: Class,
    /// Radius.
    pub radius: f64,
    /// Optical density of the input at the item's core.
    pub od_in: f64,
    /// Optical density of the output.
    pub od_out: f64,
    /// Energy (sum of squares) of the fine high-pass over the item's plateau, of the output and
    /// of the clean skin; pooled over items by the report (hair: both 0).
    pub texture_energy: (f64, f64),
    /// Mean `ln(out / clean)` of the luminance just outside the item (a halo is positive), and the
    /// number of pixels it is the mean of.
    pub halo: f64,
    /// Pixels in the halo ring.
    pub halo_n: usize,
    /// Mean `|ln(out / in)|` of the luminance beyond the item's reach.
    pub leak: f64,
}

impl ItemStat {
    /// Output optical density over input optical density: 0 is removed, 1 is untouched,
    /// negative is an overshoot.
    pub fn residual(&self) -> f64 {
        self.od_out / self.od_in
    }
}

/// What the protocol measures on the clean skin of a patch for one setting.
#[derive(Debug, Clone, Copy, Default)]
pub struct CleanStat {
    /// Mean dL*.
    pub dl: f64,
    /// Mean da*.
    pub da: f64,
    /// Mean db*.
    pub db: f64,
    /// Mean CIEDE2000.
    pub de_mean: f64,
    /// 99th percentile of CIEDE2000.
    pub de_p99: f64,
    /// RMS high-pass of the output over that of the input.
    pub texture: f64,
    /// Largest absolute change of any channel in relative terms.
    pub max_rel: f64,
}

/// Measures the far pixels of a patch.
pub fn clean_stat(p: &Patch, out: &LinearAwg4Image) -> CleanStat {
    let idx: Vec<usize> = (0..N * N).filter(|&i| p.far[i]).collect();
    let (mut dl, mut da, mut db) = (0.0, 0.0, 0.0);
    let mut des = Vec::with_capacity(idx.len());
    let mut max_rel = 0.0f64;
    for &i in &idx {
        let (a, b) = (lab(p.img.data[i]), lab(out.data[i]));
        dl += b[0] - a[0];
        da += b[1] - a[1];
        db += b[2] - a[2];
        des.push(ym_color::deltae::delta_e_2000(a, b));
        for c in 0..3 {
            let (x, y) = (f64::from(p.img.data[i][c]), f64::from(out.data[i][c]));
            max_rel = max_rel.max(((y - x) / x).abs());
        }
    }
    let n = idx.len() as f64;
    des.sort_by(f64::total_cmp);
    CleanStat {
        dl: dl / n,
        da: da / n,
        db: db / n,
        de_mean: des.iter().sum::<f64>() / n,
        de_p99: des[((0.99 * n) as usize).min(des.len() - 1)],
        texture: rms(&high_pass(out, SKIN_BAND_SIGMA_PX), &idx)
            / rms(&high_pass(&p.img, SKIN_BAND_SIGMA_PX), &idx),
        max_rel,
    }
}

/// The share of pixels where some channel of `out` differs from `img` by more than `limit`
/// (relative).
pub fn moved_share(img: &LinearAwg4Image, out: &LinearAwg4Image, limit: f64) -> f64 {
    let n = img
        .data
        .iter()
        .zip(&out.data)
        .filter(|(a, b)| {
            (0..3).any(|c| ((f64::from(b[c]) - f64::from(a[c])) / f64::from(a[c])).abs() > limit)
        })
        .count();
    n as f64 / img.data.len() as f64
}

/// Measures every item of a patch.
///
/// `halo` is the mean of `ln(out / clean)` of the luminance over the ring `1.25 R .. 2 R` just
/// outside the item, where the input equals the clean skin: a fill that overshoots leaves a bright
/// ring there. `leak` is the mean of `|ln(out / in)|` over `2 R .. 2.4 R + 8` (the pixels beyond
/// every mask, where nothing should change). The texture is measured on the plateau, `0.7 R`.
pub fn item_stats(p: &Patch, out: &LinearAwg4Image) -> Vec<ItemStat> {
    let hp_clean = high_pass(&p.clean, LESION_BAND_SIGMA_PX);
    let hp_out = high_pass(out, LESION_BAND_SIGMA_PX);
    let ln = |im: &LinearAwg4Image, i: usize| luma(im.data[i]).max(1e-6).ln();
    p.items
        .iter()
        .map(|it| {
            let core: Vec<usize> = it
                .foot
                .iter()
                .filter(|&&(_, w, _)| w >= if it.class == Class::Hair { 0.9 } else { 0.5 })
                .map(|&(i, _, _)| i)
                .collect();
            let dist = |i: usize| ((i % N) as f64 - it.cx).hypot((i / N) as f64 - it.cy);
            let pick = |lo: f64, hi: f64| -> Vec<usize> {
                if it.class == Class::Hair {
                    return Vec::new();
                }
                let r = hi.ceil() as i64;
                let mut v = Vec::new();
                for y in (it.cy as i64 - r).max(0)..=(it.cy as i64 + r).min(N as i64 - 1) {
                    for x in (it.cx as i64 - r).max(0)..=(it.cx as i64 + r).min(N as i64 - 1) {
                        let i = y as usize * N + x as usize;
                        let d = dist(i);
                        if d >= lo && d <= hi && p.near_other(i, it) {
                            v.push(i);
                        }
                    }
                }
                v
            };
            let mean = |idx: &[usize], f: &dyn Fn(usize) -> f64| {
                idx.iter().map(|&i| f(i)).sum::<f64>() / idx.len().max(1) as f64
            };
            let plateau = pick(0.0, 0.7 * it.radius);
            let near = pick(1.25 * it.radius, 2.0 * it.radius);
            let far = pick(2.0 * it.radius, 2.4 * it.radius + 8.0);
            ItemStat {
                class: it.class,
                radius: it.radius,
                od_in: optical_density(&p.img, &p.clean, &core),
                od_out: optical_density(out, &p.clean, &core),
                texture_energy: (energy(&hp_out, &plateau), energy(&hp_clean, &plateau)),
                halo: mean(&near, &|i| ln(out, i) - ln(&p.clean, i)),
                halo_n: near.len(),
                leak: mean(&far, &|i| (ln(out, i) - ln(&p.img, i)).abs()),
            }
        })
        .collect()
}

impl Patch {
    /// True when pixel `i` is not close to any item other than `me` (so its change is `me`'s).
    fn near_other(&self, i: usize, me: &Item) -> bool {
        let (x, y) = ((i % N) as f64, (i / N) as f64);
        self.items.iter().all(|o| {
            std::ptr::eq(o, me)
                || if o.class == Class::Hair {
                    o.foot.iter().all(|&(j, _, _)| {
                        ((j % N) as f64 - x).abs().max(((j / N) as f64 - y).abs()) > 10.0
                    })
                } else {
                    (o.cx - x).hypot(o.cy - y) > 2.4 * o.radius + 8.0
                }
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------------------------

/// The settings the protocol runs: (name, settings).
pub fn settings() -> Vec<(&'static str, SkinSettings)> {
    let b = |blemish: f32| SkinSettings {
        blemish,
        ..Default::default()
    };
    vec![
        ("b30", b(30.0)),
        ("b60", b(60.0)),
        ("b100", b(100.0)),
        (
            "dc100",
            SkinSettings {
                dark_circles: 100.0,
                ..Default::default()
            },
        ),
        (
            "se100",
            SkinSettings {
                even_skin: 100.0,
                ..Default::default()
            },
        ),
        (
            "default",
            SkinSettings {
                blemish: 30.0,
                dark_circles: 30.0,
                even_skin: 25.0,
                ..Default::default()
            },
        ),
        (
            "strong",
            SkinSettings {
                blemish: 80.0,
                dark_circles: 80.0,
                even_skin: 80.0,
                ..Default::default()
            },
        ),
    ]
}

/// One row of the report: one tone, one setting.
#[derive(Debug, Clone, Default)]
pub struct Row {
    /// Tone index.
    pub tone: usize,
    /// Setting name.
    pub setting: &'static str,
    /// Residual (output OD over input OD) per lesion (papule, pustule, brown).
    pub lesion_residuals: Vec<f64>,
    /// Residual of the same lesions in the three smaller-contrast band: amp 1.0 only.
    pub strong_residuals: Vec<f64>,
    /// Residual of every mole: 1 = kept.
    pub mole_residuals: Vec<f64>,
    /// Residual of every freckle.
    pub freckle_residuals: Vec<f64>,
    /// Residual of every hair line.
    pub hair_residuals: Vec<f64>,
    /// Fine-band texture kept on the plateau of the lesions of 4 px radius and more (smaller
    /// ones have no plateau at this band), pooled: output RMS over clean-skin RMS.
    pub lesion_texture: f64,
    /// Largest halo (|mean| over the near ring) of any lesion.
    pub halo_max: f64,
    /// Mean halo over the rings of all lesions pooled (signed: a systematic ring).
    pub halo_pooled: f64,
    /// Largest leak over the lesions, moles and freckles.
    pub leak_max: f64,
    /// The share of the depth of the dark blotches that Even skin lifted (0 nothing, 1 all).
    pub blotch_lifted: f64,
    /// The share of the depth of the dark circles that Dark circles lifted.
    pub circle_lifted: f64,
    /// Share of the pixels of the rough skin that moved by more than 1 % in some channel.
    pub rough_moved: f64,
    /// Clean skin of the lesion patch, of the keepers patch, of the clean patch and of the rough one.
    pub clean: [CleanStat; 4],
}

/// Runs the whole protocol on every tone.
pub fn run_protocol() -> Vec<Row> {
    run_protocol_on(&(0..TONES).collect::<Vec<_>>(), &[])
}

/// Runs the protocol on some tones and the named settings (all when `only` is empty).
pub fn run_protocol_on(tones: &[usize], only: &[&str]) -> Vec<Row> {
    let scenes = [
        Scene::Lesions,
        Scene::Keepers,
        Scene::Clean,
        Scene::Rough,
        Scene::Circles,
    ];
    let list: Vec<_> = settings()
        .into_iter()
        .filter(|(n, _)| only.is_empty() || only.contains(n))
        .collect();
    // One tone per task; the rows come back in tone order whatever the thread count.
    let per_tone: Vec<Vec<Row>> = tones
        .par_iter()
        .map(|&t| run_tone(t, &scenes, &list))
        .collect();
    per_tone.into_iter().flatten().collect()
}

fn run_tone(t: usize, scenes: &[Scene], list: &[(&'static str, SkinSettings)]) -> Vec<Row> {
    let mut rows = Vec::new();
    {
        let mut prepared: Vec<Prepared> = scenes
            .iter()
            .map(|&s| Prepared::with_mesh(patch(t, s), s == Scene::Circles))
            .collect();
        for (name, s) in list {
            let mut row = Row {
                tone: t,
                setting: name,
                ..Default::default()
            };
            let (mut e_out, mut e_ref) = (0.0, 0.0);
            let (mut halo_sum, mut halo_n) = (0.0, 0usize);
            for (k, prep) in prepared.iter_mut().enumerate() {
                let out = prep.run(*s);
                if prep.patch.scene == Scene::Circles {
                    let (mut got, mut all) = (0.0, 0.0);
                    for (i, &f) in prep
                        .patch
                        .blotch
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| **f < -0.04)
                    {
                        got += luma(out.data[i]).ln() - luma(prep.patch.img.data[i]).ln();
                        all += -(1.0 + f).ln();
                    }
                    row.circle_lifted = got / all;
                    continue;
                }
                row.clean[k] = clean_stat(&prep.patch, &out);
                if prep.patch.scene == Scene::Clean {
                    let (mut got, mut all) = (0.0, 0.0);
                    for (i, &f) in prep
                        .patch
                        .blotch
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| **f < -0.02)
                    {
                        let dl = luma(out.data[i]).ln() - luma(prep.patch.img.data[i]).ln();
                        got += dl;
                        all += -(1.0 + f).ln();
                    }
                    row.blotch_lifted = got / all;
                }
                if prep.patch.scene == Scene::Rough {
                    row.rough_moved = moved_share(&prep.patch.img, &out, 0.01);
                }
                for st in item_stats(&prep.patch, &out) {
                    row.leak_max = row.leak_max.max(st.leak);
                    if st.class.is_lesion() {
                        row.halo_max = row.halo_max.max(st.halo.abs());
                        halo_sum += st.halo * st.halo_n as f64;
                        halo_n += st.halo_n;
                    }
                    match st.class {
                        Class::Papule | Class::Pustule | Class::Brown => {
                            if st.od_in >= VISIBLE_OD {
                                row.lesion_residuals.push(st.residual());
                            }
                            if st.radius >= 4.0 {
                                e_out += st.texture_energy.0;
                                e_ref += st.texture_energy.1;
                            }
                            if st.od_in > 0.0 {
                                row.strong_residuals.push(st.residual());
                            }
                        }
                        Class::Mole => row.mole_residuals.push(st.residual()),
                        Class::Freckle => row.freckle_residuals.push(st.residual()),
                        Class::Hair => row.hair_residuals.push(st.residual()),
                    }
                }
            }
            row.lesion_texture = (e_out / e_ref).sqrt();
            row.halo_pooled = halo_sum / halo_n.max(1) as f64;
            rows.push(row);
        }
    }
    rows
}

// ---------------------------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------------------------

/// One gate: a worst value over the tones against a limit.
#[derive(Debug, Clone)]
pub struct Gate {
    /// What it checks.
    pub name: &'static str,
    /// The setting it applies to.
    pub setting: &'static str,
    /// `true`: the value must stay at or under `limit`; `false`: at or over.
    pub upper: bool,
    /// The limit.
    pub limit: f64,
    /// The worst value over the tones.
    pub worst: f64,
    /// The tone that has it.
    pub tone: usize,
}

impl Gate {
    /// True when the worst value is inside the limit.
    pub fn passes(&self) -> bool {
        if self.upper {
            self.worst <= self.limit
        } else {
            self.worst >= self.limit
        }
    }
    /// The distance to the limit as a share of the limit (positive: room to spare).
    pub fn margin(&self) -> f64 {
        let d = if self.upper {
            self.limit - self.worst
        } else {
            self.worst - self.limit
        };
        d / self.limit.abs().max(1e-9)
    }
}

/// The rows of one setting.
fn of<'a>(rows: &'a [Row], setting: &str) -> impl Iterator<Item = &'a Row> {
    let setting = setting.to_string();
    rows.iter().filter(move |r| r.setting == setting)
}

/// Evaluates every gate on the rows of a full protocol run.
///
/// The limits come from the research where it gives a number (mole survival 0.9 at the default
/// amount; mean |dL*| 0.5 and |da*|, |db*| 1 on clean skin; band-pass energy ratio 0.95) and are
/// reasoned otherwise, each with its reason beside it.
pub fn gates(rows: &[Row]) -> Vec<Gate> {
    let mut out = Vec::new();
    // `pick` takes the worst of `value(row)` over the rows of `setting`.
    let mut add = |name: &'static str,
                   setting: &'static str,
                   upper: bool,
                   limit: f64,
                   value: &dyn Fn(&Row) -> f64| {
        let mut worst = if upper { f64::MIN } else { f64::MAX };
        let mut tone = 0;
        for r in of(rows, setting) {
            let v = value(r);
            if (upper && v > worst) || (!upper && v < worst) {
                worst = v;
                tone = r.tone;
            }
        }
        out.push(Gate {
            name,
            setting,
            upper,
            limit,
            worst,
            tone,
        });
    };
    let max_abs = |v: &[f64]| v.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    let min_of = |v: &[f64]| v.iter().copied().fold(f64::MAX, f64::min);
    let clean_max =
        |r: &Row, f: &dyn Fn(&CleanStat) -> f64| r.clean.iter().map(f).fold(0.0, f64::max);

    // Removed. Residual = output optical density over input optical density at the lesion's core.
    // At 100 a lesion must go: 0.30 leaves a spot at under a third of its contrast, which is under
    // the texture of the skin at these sizes. An overshoot counts like a residue.
    add(
        "removed: worst |residual| of any lesion",
        "b100",
        true,
        0.30,
        &|r| max_abs(&r.lesion_residuals),
    );
    add(
        "removed: worst |residual| of any lesion",
        "b60",
        true,
        0.45,
        &|r| max_abs(&r.lesion_residuals),
    );
    // The default is meant to be partial ("low values fix only strong blemishes"): the median
    // lesion loses at least half of its density, and no lesion grows.
    add("removed: median lesion residual", "b30", true, 0.50, &|r| {
        median(&r.lesion_residuals)
    });
    add(
        "removed: worst residual (no lesion grows)",
        "b30",
        true,
        0.85,
        &|r| r.lesion_residuals.iter().copied().fold(f64::MIN, f64::max),
    );
    // Strongest first, monotonic in the amount: a bigger amount never leaves more of a lesion.
    add(
        "monotonic: b60 residual minus b30 residual, worst lesion",
        "b60",
        true,
        0.08,
        &|r| {
            let before = of(rows, "b30")
                .find(|q| q.tone == r.tone)
                .map(|q| q.lesion_residuals.clone())
                .unwrap_or_default();
            r.lesion_residuals
                .iter()
                .zip(&before)
                .map(|(a, b)| a - b)
                .fold(f64::MIN, f64::max)
        },
    );
    add(
        "monotonic: b100 residual minus b60 residual, worst lesion",
        "b100",
        true,
        0.08,
        &|r| {
            let before = of(rows, "b60")
                .find(|q| q.tone == r.tone)
                .map(|q| q.lesion_residuals.clone())
                .unwrap_or_default();
            r.lesion_residuals
                .iter()
                .zip(&before)
                .map(|(a, b)| a - b)
                .fold(f64::MIN, f64::max)
        },
    );
    // Texture kept. Clean skin: the research asks for 0.95 of the band-pass energy at the default.
    add(
        "texture kept on clean skin (band sigma 3 px)",
        "default",
        false,
        0.95,
        &|r| r.clean.iter().map(|c| c.texture).fold(f64::MAX, f64::min),
    );
    add(
        "texture kept on clean skin (band sigma 3 px)",
        "se100",
        false,
        0.88,
        &|r| r.clean.iter().map(|c| c.texture).fold(f64::MAX, f64::min),
    );
    add(
        "texture kept on clean skin (band sigma 3 px)",
        "strong",
        false,
        0.88,
        &|r| r.clean.iter().map(|c| c.texture).fold(f64::MAX, f64::min),
    );
    // Inside a removed lesion of 4 px radius and more the fill must not be plastic.
    add(
        "texture kept inside lesions of R >= 4 px",
        "b100",
        false,
        0.60,
        &|r| r.lesion_texture,
    );
    // Clean skin unchanged. Blemish alone changes nothing farther than the lesions' reach.
    for setting in ["b30", "b60", "b100"] {
        add(
            "clean skin: largest relative change of any pixel (not rough)",
            setting,
            true,
            0.0,
            &|r| r.clean[..3].iter().map(|c| c.max_rel).fold(0.0, f64::max),
        );
    }
    // Even skin lifts dark blotches (5 to 10 % deep, 2 to 3 % of the face wide) toward their
    // surroundings, and by design only so far: the regional average it moves toward is itself
    // dragged down by the blotch, and the blend rolls off brightening. At 100 it must lift a
    // fifth of the depth; at the default 25 a twentieth.
    add(
        "even skin lifts dark blotches: share of depth lifted",
        "default",
        false,
        0.05,
        &|r| r.blotch_lifted,
    );
    add(
        "even skin lifts dark blotches: share of depth lifted",
        "se100",
        false,
        0.20,
        &|r| r.blotch_lifted,
    );
    // Dark circles move the under-eye zone to its own average colour, so a crescent that fills
    // half of the zone can lose at most about half of its depth (and the blend rolls off
    // brightening): at the default a tenth of it, at 100 a sixth. These catch a tool that stops
    // working; the radius of the average (0.077 D, tried up to 0.13 D) gains little on them.
    add(
        "dark circles lift the crescents: share of depth lifted",
        "default",
        false,
        0.10,
        &|r| r.circle_lifted,
    );
    add(
        "dark circles lift the crescents: share of depth lifted",
        "dc100",
        false,
        0.15,
        &|r| r.circle_lifted,
    );
    // Rough skin (3 % texture, 100 deep pores) is not a blemish: at most 1 % of its pixels may
    // move by more than 1 %, at the strongest Blemish.
    add(
        "rough skin: share of pixels moved by more than 1 %",
        "b100",
        true,
        0.01,
        &|r| r.rough_moved,
    );
    add(
        "rough skin: share of pixels moved by more than 1 %",
        "b30",
        true,
        0.003,
        &|r| r.rough_moved,
    );
    // The research: mean |dL*| 0.5 and |da*|, |db*| 1 (the 50 % acceptability thresholds are
    // dL* 0.864, da* 2.037, db* 2.517); a just-noticeable difference is dE 1.
    for (setting, de) in [("default", 1.0), ("strong", 2.0)] {
        add("clean skin: |mean dL*|", setting, true, 0.5, &|r| {
            clean_max(r, &|c| c.dl.abs())
        });
        add("clean skin: |mean da*|", setting, true, 1.0, &|r| {
            clean_max(r, &|c| c.da.abs())
        });
        add("clean skin: |mean db*|", setting, true, 1.0, &|r| {
            clean_max(r, &|c| c.db.abs())
        });
        add(
            "clean skin: dE2000 99th percentile",
            setting,
            true,
            de,
            &|r| clean_max(r, &|c| c.de_p99),
        );
    }
    // Moles kept: 0.9 of the density at the default amount (the research); Even skin at 80 may
    // lighten a mole a little more.
    for (setting, limit) in [
        ("b30", 0.9),
        ("b100", 0.9),
        ("default", 0.9),
        ("strong", 0.8),
    ] {
        add(
            "moles kept: smallest residual of any mole",
            setting,
            false,
            limit,
            &|r| min_of(&r.mole_residuals),
        );
    }
    // Hair-like lines and freckles are left alone by Blemish and lightened at most a little by
    // Even skin.
    add(
        "hair kept: smallest residual of any line",
        "b100",
        false,
        0.95,
        &|r| min_of(&r.hair_residuals),
    );
    add(
        "hair kept: smallest residual of any line",
        "default",
        false,
        0.90,
        &|r| min_of(&r.hair_residuals),
    );
    add(
        "freckles kept: smallest residual of any freckle",
        "b100",
        false,
        0.95,
        &|r| min_of(&r.freckle_residuals),
    );
    add(
        "freckles kept: smallest residual of any freckle",
        "default",
        false,
        0.85,
        &|r| min_of(&r.freckle_residuals),
    );
    // No halo: the mean of ln(out / clean) in the ring just outside the lesions. Pooled over all
    // lesions of a tone it must be under 0.5 % (a systematic ring is seen at about 1 %); a single
    // lesion of 2 px radius has 20 ring pixels of 1.5 % texture, so its own mean is noise-limited
    // to about 2.5 %.
    for (setting, limit) in [
        ("b60", 0.005),
        ("b100", 0.005),
        ("default", 0.008),
        ("strong", 0.012),
    ] {
        add(
            "no halo: |pooled ring mean| just outside the lesions",
            setting,
            true,
            limit,
            &|r| r.halo_pooled.abs(),
        );
    }
    for (setting, limit) in [
        ("b60", 0.025),
        ("b100", 0.025),
        ("default", 0.03),
        ("strong", 0.035),
    ] {
        add(
            "no halo: largest |ring mean| of any one lesion",
            setting,
            true,
            limit,
            &|r| r.halo_max,
        );
    }
    // No leak beyond a lesion's reach (between 2 R and 2.4 R + 8 px).
    for (setting, limit) in [
        ("b30", 0.003),
        ("b100", 0.003),
        ("default", 0.006),
        ("strong", 0.015),
    ] {
        add(
            "no leak: largest mean |change| beyond a lesion's reach",
            setting,
            true,
            limit,
            &|r| r.leak_max,
        );
    }
    out
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() { 0.0 } else { s[s.len() / 2] }
}

/// The gates as a table, one line each.
pub fn gate_table(gs: &[Gate]) -> String {
    let mut s = String::new();
    for g in gs {
        s += &format!(
            "{} {:8} {:62} {} {:7.4} (limit {:7.4}, tone {}, margin {:+.0} %)\n",
            if g.passes() { "pass" } else { "FAIL" },
            g.setting,
            g.name,
            if g.upper { "<=" } else { ">=" },
            g.worst,
            g.limit,
            g.tone,
            100.0 * g.margin().clamp(-9.99, 9.99),
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(v: &[f64]) -> String {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        format!("{:6.2}..{:6.2}", s[0], s[s.len() - 1])
    }

    #[test]
    #[ignore = "prints every lesion"]
    fn print_the_lesions() {
        for t in std::env::var("YM_TONES")
            .map(|v| {
                v.split(',')
                    .filter_map(|x| x.parse().ok())
                    .collect::<Vec<usize>>()
            })
            .unwrap_or_else(|_| vec![6])
        {
            let mut prep = Prepared::new(patch(t, Scene::Lesions));
            for (name, s) in settings().into_iter().filter(|(n, _)| *n == "b100") {
                let out = prep.run(s);
                println!("tone {t} {name} clean {:?}", clean_stat(&prep.patch, &out));
                for (it, st) in prep.patch.items.iter().zip(item_stats(&prep.patch, &out)) {
                    println!(
                        "tone {t} {name:5} {:?} R {:.1} amp {:.2} od_in {:.3} residual {:+.2} tex {:.2} halo {:+.4}",
                        st.class,
                        st.radius,
                        it.amp,
                        st.od_in,
                        st.residual(),
                        (st.texture_energy.0 / st.texture_energy.1).sqrt(),
                        st.halo
                    );
                }
            }
        }
    }

    /// The gates of S-12 on the whole protocol: every tone, every setting.
    #[test]
    fn synthetic_gates_pass_in_every_skin_tone() {
        let rows = run_protocol();
        let gs = gates(&rows);
        let table = gate_table(&gs);
        println!("{table}");
        let failed: Vec<_> = gs.iter().filter(|g| !g.passes()).collect();
        assert!(failed.is_empty(), "gates failed:\n{table}");
    }

    /// The protocol is deterministic: the same patch gives the same bytes, and 1 and 4 threads agree.
    #[test]
    fn protocol_is_deterministic_with_one_and_four_threads() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| Prepared::new(patch(4, Scene::Lesions)).run(settings()[4].1))
        };
        assert_eq!(run(1).data, run(4).data);
    }

    /// Dark circles at 100 on clean skin with the mock mesh's under-eye zones: they lift the zone
    /// toward its surroundings and darken almost nothing (the research: "dark circles that darken
    /// almost no pixels"), in every tone.
    #[test]
    fn dark_circles_darken_almost_nothing() {
        let mut worst = 0.0f64;
        for t in 0..TONES {
            let mut prep = Prepared::with_mesh(patch(t, Scene::Rough), true);
            let out = prep.run(SkinSettings {
                dark_circles: 100.0,
                ..Default::default()
            });
            let darker = prep
                .patch
                .img
                .data
                .iter()
                .zip(&out.data)
                .filter(|(a, b)| luma(**b) < 0.98 * luma(**a))
                .count() as f64
                / (N * N) as f64;
            worst = worst.max(darker);
        }
        println!("dark circles 100: worst share of pixels darkened by more than 2 %: {worst:.5}");
        assert!(worst <= 0.005, "{worst}");
    }

    #[test]
    #[ignore = "prints the gate table"]
    fn print_the_gates() {
        let tones: Vec<usize> = std::env::var("YM_TONES")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| (0..TONES).collect());
        let only = std::env::var("YM_SETTINGS").unwrap_or_default();
        let only: Vec<&str> = only.split(',').filter(|x| !x.is_empty()).collect();
        let rows = run_protocol_on(&tones, &only);
        println!("{}", gate_table(&gates(&rows)));
    }

    #[test]
    #[ignore = "prints the protocol table"]
    fn print_the_protocol() {
        let tones: Vec<usize> = std::env::var("YM_TONES")
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|_| (0..TONES).collect());
        let only = std::env::var("YM_SETTINGS").unwrap_or_default();
        let only: Vec<&str> = only.split(',').filter(|x| !x.is_empty()).collect();
        for r in run_protocol_on(&tones, &only) {
            println!(
                "tone {} {:7} lesion {} mole {} freckle {} hair {} tex {:.2} halo {:.4} leak {:.4} | clean dL {:+.3} da {:+.3} db {:+.3} dE {:.3}/{:.3} tex {:.3}/{:.3}/{:.3}",
                r.tone,
                r.setting,
                fmt(&r.lesion_residuals),
                fmt(&r.mole_residuals),
                fmt(&r.freckle_residuals),
                fmt(&r.hair_residuals),
                r.lesion_texture,
                r.halo_max,
                r.leak_max,
                r.clean[2].dl,
                r.clean[2].da,
                r.clean[2].db,
                r.clean[2].de_mean,
                r.clean[2].de_p99,
                r.clean[0].texture,
                r.clean[1].texture,
                r.clean[2].texture,
            );
        }
    }

    /// Mean green channel against the clean skin by distance from the centre of each smallest lesion.
    #[test]
    #[ignore = "prints radial profiles"]
    fn print_radial_profiles() {
        let t: usize = std::env::var("YM_TONE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let mut prep = Prepared::new(patch(t, Scene::Lesions));
        let out = prep.run(settings()[2].1);
        for it in prep.patch.items.iter().filter(|it| it.class.is_lesion()) {
            let mut line = format!("tone {t} {:?} R {:.1}:", it.class, it.radius);
            for j in 0..12 {
                let (lo, hi) = (j as f64 * 0.2 * it.radius, (j + 1) as f64 * 0.2 * it.radius);
                let idx: Vec<usize> = (0..N * N)
                    .filter(|&q| {
                        let d = ((q % N) as f64 - it.cx).hypot((q / N) as f64 - it.cy);
                        d >= lo && d < hi
                    })
                    .collect();
                if idx.is_empty() {
                    continue;
                }
                let g = |im: &LinearAwg4Image| {
                    idx.iter().map(|&q| f64::from(im.data[q][1])).sum::<f64>() / idx.len() as f64
                };
                let (c, i, o) = (g(&prep.patch.clean), g(&prep.patch.img), g(&out));
                line += &format!(
                    " {:.1}R {:+.2}/{:+.2}",
                    (lo + hi) / 2.0 / it.radius,
                    i / c - 1.0,
                    o / c - 1.0
                );
            }
            println!("{line}");
        }
    }

    #[test]
    #[ignore = "prints the quiet scenes"]
    fn print_quiet_scenes() {
        for t in [0, 5, 9] {
            for scene in [Scene::Keepers, Scene::Clean, Scene::Rough] {
                let mut prep = Prepared::new(patch(t, scene));
                for (name, s) in settings().into_iter().filter(|(n, _)| *n == "b100") {
                    let out = prep.run(s);
                    let c = clean_stat(&prep.patch, &out);
                    let changed = (0..N * N)
                        .filter(|&i| prep.patch.img.data[i] != out.data[i])
                        .count();
                    let share = |lim: f64| {
                        (0..N * N)
                            .filter(|&i| {
                                (0..3).any(|k| {
                                    let (a, b) = (
                                        f64::from(prep.patch.img.data[i][k]),
                                        f64::from(out.data[i][k]),
                                    );
                                    ((b - a) / a).abs() > lim
                                })
                            })
                            .count() as f64
                            / (N * N) as f64
                    };
                    println!(
                        "tone {t} {scene:?} {name}: max_rel {:.4} changed px {changed}, share over 1 % {:.4}, over 3 % {:.4}",
                        c.max_rel,
                        share(0.01),
                        share(0.03)
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "prints the keepers"]
    fn print_keepers() {
        for t in [3, 7] {
            for scene in [Scene::Lesions, Scene::Keepers] {
                let mut prep = Prepared::new(patch(t, scene));
                for (name, s) in settings()
                    .into_iter()
                    .filter(|(n, _)| *n == "default" || *n == "strong")
                {
                    let out = prep.run(s);
                    for (it, st) in prep.patch.items.iter().zip(item_stats(&prep.patch, &out)) {
                        if !st.class.is_lesion() {
                            println!(
                                "tone {t} {scene:?} {name} {:?} R {:.1} amp {:.2} od_in {:.2} residual {:.3}",
                                st.class,
                                st.radius,
                                it.amp,
                                st.od_in,
                                st.residual()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "prints the circles"]
    fn print_circles() {
        let mut prep = Prepared::with_mesh(patch(5, Scene::Circles), true);
        for dc in [0.0, 30.0, 60.0, 100.0] {
            let out = prep.run(SkinSettings {
                dark_circles: dc,
                ..Default::default()
            });
            let (mut got, mut all) = (0.0, 0.0);
            for (i, &f) in prep
                .patch
                .blotch
                .iter()
                .enumerate()
                .filter(|(_, f)| **f < -0.04)
            {
                got += luma(out.data[i]).ln() - luma(prep.patch.img.data[i]).ln();
                all += -(1.0 + f).ln();
            }
            println!("dc {dc}: lifted {:.4}", got / all);
        }
    }

    /// The per-tone table of the S-12 note in `docs/27`, as markdown.
    #[test]
    #[ignore = "prints the per-tone table"]
    fn print_the_table() {
        let rows = run_protocol();
        let get = |t: usize, s: &str| rows.iter().find(|r| r.tone == t && r.setting == s).unwrap();
        println!(
            "| tone | Y | lesions left at 100 (worst) | lesions left at 30 (median) | ring at 100 (worst lesion) | rough skin moved at 100 | mole kept at 25/30/25 | texture kept at 25/30/25 | blotch lifted at Even 100 | Even 80 texture |"
        );
        println!("|---|---|---|---|---|---|---|---|---|---|");
        for t in 0..TONES {
            let (b100, b30, d, se, st) = (
                get(t, "b100"),
                get(t, "b30"),
                get(t, "default"),
                get(t, "se100"),
                get(t, "strong"),
            );
            let max_abs = |v: &[f64]| v.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            let min = |v: &[f64]| v.iter().copied().fold(f64::MAX, f64::min);
            println!(
                "| {} | {:.2} | {:.2} | {:.2} | {:+.3} | {:.2} % | {:.2} | {:.3} | {:.2} | {:.3} |",
                t + 1,
                {
                    let c = tone_rgb(t);
                    0.2545 * c[0] + 0.7815 * c[1] - 0.0360 * c[2]
                },
                max_abs(&b100.lesion_residuals),
                median(&b30.lesion_residuals),
                b100.halo_max,
                100.0 * b100.rough_moved,
                min(&d.mole_residuals),
                d.clean.iter().map(|c| c.texture).fold(f64::MAX, f64::min),
                se.blotch_lifted,
                st.clean.iter().map(|c| c.texture).fold(f64::MAX, f64::min),
            );
        }
    }
}
