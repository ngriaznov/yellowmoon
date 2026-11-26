//! The harms H1 to H6 and the look-transfer metrics T1 to T3 of one fit (`docs/34` section 10.1).
//!
//! They are measured on **every pixel** of the fit-size renders of the photo as it is and of the fit, in display sRGB (CIELAB for C\*
//! and L\*, Oklab for hue), and on the reference's own analysis picture. They are checks, not objective terms: their definitions and
//! thresholds differ from the walls' on purpose, so a fit cannot pass them just by satisfying its own walls. The definitions, the
//! constants, the seed and the order of every operation are those of the proof of concept's harness, so a run of the product and a
//! run of round 15 compare number for number.

use ym_color::refgrade::measure::Px;

/// The element numbers of the label step (`ym_color::refgrade::measure::class`) the harms read.
const SKY: u8 = 1;
const VEGETATION: u8 = 2;

/// The ids of the six harms, in order.
pub const HARMS: [&str; 6] = ["H1", "H2", "H3", "H4", "H5", "H6"];
/// Their names, for the tables.
pub const HARM_NAMES: [&str; 6] = [
    "purple turn",
    "skin colour",
    "memory colour",
    "tinted whites",
    "clipping",
    "colour reversal",
];
/// The Oklab purple band of the checks, in degrees.
const BAND: (f64, f64) = (300.0, 355.0);
/// The least mean-vector C\* of the photo at which a hue shift is tested (H2, H3).
const HUE_MIN_CHROMA: f64 = 4.0;
/// Seeded pixel samples per picture for the sliced Wasserstein distance.
pub const SW_SAMPLES: usize = 4096;
/// Fixed directions of the sliced Wasserstein distance.
pub const SW_DIRECTIONS: usize = 32;
/// The seed of the pixel samples.
pub const SW_SEED: u64 = 0x5EED_2026_1124;

/// SplitMix64: a seeded generator with no state beyond one word.
#[derive(Debug, Clone)]
pub struct Rng(pub u64);

impl Rng {
    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The hue, in degrees 0 to 360, of an Oklab (a, b) vector.
#[must_use]
pub fn oklab_hue(ab: [f64; 2]) -> f64 {
    ab[1].atan2(ab[0]).to_degrees().rem_euclid(360.0)
}

/// Wraps an angle difference to -180 to 180 degrees.
#[must_use]
pub fn wrap_deg(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

fn cstar(q: &Px) -> f64 {
    q.lab[1].hypot(q.lab[2])
}

fn in_band(h: f64) -> bool {
    (BAND.0..=BAND.1).contains(&h)
}

/// What a scene says about the pixels of a picture: each pixel's element number and whether it is skin.
#[derive(Debug, Clone, Copy)]
pub struct SceneRef<'a> {
    /// The element of each pixel (`ym_color::refgrade::measure::class`).
    pub class: &'a [u8],
    /// Whether each pixel is skin.
    pub skin: &'a [bool],
}

/// One picture, measured, with its scene when it has one.
#[derive(Debug, Clone, Copy)]
pub struct Pic<'a> {
    /// Every pixel, row-major.
    pub px: &'a [Px],
    /// Its scene.
    pub scene: Option<SceneRef<'a>>,
}

impl Pic<'_> {
    fn skin(&self, i: usize) -> bool {
        self.scene.is_some_and(|s| s.skin[i])
    }

    fn elem(&self, i: usize) -> usize {
        self.scene.map_or(0, |s| usize::from(s.class[i]))
    }

    fn share(&self, f: impl Fn(usize, &Px) -> bool) -> f64 {
        let n = self.px.iter().enumerate().filter(|&(i, q)| f(i, q)).count();
        n as f64 / self.px.len().max(1) as f64
    }

    /// The Oklab mean-vector hue and the CIELAB mean-vector chroma over the pixels `f` allows, with their count.
    fn mean_vec(&self, f: impl Fn(usize, &Px) -> bool) -> Option<(f64, f64, usize)> {
        let (mut oa, mut ob, mut la, mut lb, mut n) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for (i, q) in self.px.iter().enumerate() {
            if f(i, q) {
                oa += q.ok[0];
                ob += q.ok[1];
                la += q.lab[1];
                lb += q.lab[2];
                n += 1;
            }
        }
        if n == 0 {
            return None;
        }
        let nf = n as f64;
        Some((oklab_hue([oa / nf, ob / nf]), (la / nf).hypot(lb / nf), n))
    }
}

/// The median taken at `len / 2` of the sorted values (0 for an empty list).
fn median_of(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The harm values and flags of one fit: `(value, flag)` for H1 to H6, `None` where the check does not apply.
///
/// `p0` is the photo as it is, `fit` the photo with the fit, `refp` the reference; `p0` and `fit` share one scene.
#[must_use]
pub fn harms(p0: &Pic<'_>, fit: &Pic<'_>, refp: &Pic<'_>) -> [(Option<f64>, bool); 6] {
    let mut out = [(None, false); 6];
    // H1: purple turn.
    let h1 = fit.share(|i, q| {
        !p0.skin(i) && cstar(q) >= 5.0 && in_band(oklab_hue(q.ok)) && {
            let p = &p0.px[i];
            cstar(p) < 2.0 || wrap_deg(oklab_hue(q.ok) - oklab_hue(p.ok)).abs() >= 15.0
        }
    });
    let ref_purple = refp.share(|_, q| cstar(q) > 8.0 && in_band(oklab_hue(q.ok)));
    out[0] = (Some(h1), h1 > 0.02 && ref_purple < 0.05);
    // H2: skin colour.
    let skin_n = p0.share(|i, _| p0.skin(i));
    if skin_n >= 0.01
        && let (Some(a), Some(b)) = (
            p0.mean_vec(|i, _| p0.skin(i)),
            fit.mean_vec(|i, _| p0.skin(i)),
        )
    {
        let ratio = b.1 / a.1.max(1e-9);
        // Hue is only meaningful on a coloured area: below a mean-vector C* of 4 the hue shift is not tested.
        let dh = (a.1 >= HUE_MIN_CHROMA).then(|| wrap_deg(b.0 - a.0).abs());
        out[1] = (
            dh,
            dh.is_some_and(|d| d > 10.0) || !(0.55..=1.25).contains(&ratio),
        );
    }
    // H3: memory colour (the larger hue move of sky and vegetation).
    let mut h3: Option<f64> = None;
    for e in [SKY, VEGETATION] {
        let e = usize::from(e);
        let ps = p0.share(|i, _| p0.elem(i) == e);
        let rs = refp.share(|i, _| refp.elem(i) == e);
        if ps >= 0.02
            && rs < 0.02
            && let (Some(a), Some(b)) = (
                p0.mean_vec(|i, _| p0.elem(i) == e),
                fit.mean_vec(|i, _| p0.elem(i) == e),
            )
        {
            // Not sky when dark (median L* under 20); no hue test on a near-grey area.
            let mut ls: Vec<f64> = (0..p0.px.len())
                .filter(|&i| p0.elem(i) == e)
                .map(|i| p0.px[i].lab[0])
                .collect();
            if e == usize::from(SKY) && median_of(&mut ls) < 20.0 {
                continue;
            }
            if a.1 >= HUE_MIN_CHROMA {
                let dh = wrap_deg(b.0 - a.0).abs();
                h3 = Some(h3.map_or(dh, |v| v.max(dh)));
            }
        }
    }
    out[2] = (h3, h3.is_some_and(|v| v > 15.0));
    // H4: tinted whites.
    let white = |_: usize, q: &Px| q.lab[0] >= 80.0 && cstar(q) < 8.0;
    let white_share = p0.share(white);
    if white_share >= 0.01 {
        let mean_c = |pic: &Pic<'_>, scope: &dyn Fn(usize) -> bool| {
            let (mut s, mut k) = (0.0, 0usize);
            for (i, q) in pic.px.iter().enumerate() {
                if scope(i) {
                    s += cstar(q);
                    k += 1;
                }
            }
            if k == 0 { 0.0 } else { s / k as f64 }
        };
        let fit_c = mean_c(fit, &|i| white(i, &p0.px[i]));
        let ref_c = mean_c(refp, &|i| white(i, &refp.px[i]));
        out[3] = (Some(fit_c), fit_c > 8.0f64.max(ref_c + 4.0));
    }
    // H5: clipping.
    let clip = |pic: &Pic<'_>| {
        (
            pic.share(|_, q| q.lab[0] > 98.0),
            pic.share(|_, q| q.lab[0] < 2.0),
        )
    };
    let (f, p, r) = (clip(fit), clip(p0), clip(refp));
    let ex = (f.0 - p.0.max(r.0)).max(f.1 - p.1.max(r.1));
    out[4] = (Some(ex), ex > 0.03);
    // H6: colour reversal.
    let h6 = fit.share(|i, q| {
        let p = &p0.px[i];
        !p0.skin(i) && cstar(p) >= 10.0 && wrap_deg(oklab_hue(q.ok) - oklab_hue(p.ok)).abs() > 60.0
    });
    out[5] = (Some(h6), h6 > 0.03);
    out
}

/// `SW_SAMPLES` seeded samples of (a\*, b\*) of a picture.
#[must_use]
pub fn sample_ab(px: &[Px]) -> Vec<[f64; 2]> {
    let mut rng = Rng(SW_SEED);
    (0..SW_SAMPLES)
        .map(|_| {
            let q = &px[(rng.next_u64() % px.len() as u64) as usize];
            [q.lab[1], q.lab[2]]
        })
        .collect()
}

/// The sliced Wasserstein-1 distance of two equal-size samples over `SW_DIRECTIONS` fixed directions.
#[must_use]
pub fn sliced_w1(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let mut total = 0.0;
    for k in 0..SW_DIRECTIONS {
        let th = std::f64::consts::PI * k as f64 / SW_DIRECTIONS as f64;
        let (c, s) = (th.cos(), th.sin());
        let mut pa: Vec<f64> = a.iter().map(|p| p[0] * c + p[1] * s).collect();
        let mut pb: Vec<f64> = b.iter().map(|p| p[0] * c + p[1] * s).collect();
        pa.sort_unstable_by(f64::total_cmp);
        pb.sort_unstable_by(f64::total_cmp);
        total += pa.iter().zip(&pb).map(|(x, y)| (x - y).abs()).sum::<f64>() / pa.len() as f64;
    }
    total / SW_DIRECTIONS as f64
}

/// The key-preserving tone shape of L\*: `(p98 - p2, (p50 - p2) / (p98 - p2))`.
#[must_use]
pub fn tone_shape(px: &[Px]) -> [f64; 2] {
    let mut l: Vec<f64> = px.iter().map(|q| q.lab[0]).collect();
    l.sort_unstable_by(f64::total_cmp);
    let at = |q: f64| l[((l.len() - 1) as f64 * q).round() as usize];
    let (p2, p50, p98) = (at(0.02), at(0.5), at(0.98));
    let span = (p98 - p2).max(1e-9);
    [span, (p50 - p2) / span]
}

/// The L1 distance of two tone shapes, the span scaled by 1/50.
#[must_use]
pub fn tone_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).abs() / 50.0 + (a[1] - b[1]).abs()
}

/// The cast: the mean (a\*, b\*) of the non-skin pixels with 15 <= L\* <= 80.
#[must_use]
pub fn cast(pic: &Pic<'_>) -> [f64; 2] {
    let (mut a, mut b, mut n) = (0.0, 0.0, 0usize);
    for (i, q) in pic.px.iter().enumerate() {
        if !pic.skin(i) && (15.0..=80.0).contains(&q.lab[0]) {
            a += q.lab[1];
            b += q.lab[2];
            n += 1;
        }
    }
    if n == 0 {
        [0.0, 0.0]
    } else {
        [a / n as f64, b / n as f64]
    }
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The three transfer metrics; `None` where the denominator is too small (under 1 for T1 and T3, under 0.01 for T2).
///
/// `p0_sample` is `sample_ab` of the photo as it is, made once per photo.
#[must_use]
pub fn transfers(
    p0: &Pic<'_>,
    fit: &Pic<'_>,
    refp: &Pic<'_>,
    p0_sample: &[[f64; 2]],
) -> [Option<f64>; 3] {
    let ref_sample = sample_ab(refp.px);
    let fit_sample = sample_ab(fit.px);
    let d0 = sliced_w1(p0_sample, &ref_sample);
    let t1 = (d0 >= 1.0).then(|| 1.0 - sliced_w1(&fit_sample, &ref_sample) / d0);
    let rt = tone_shape(refp.px);
    let e0 = tone_dist(tone_shape(p0.px), rt);
    let t2 = (e0 >= 0.01).then(|| 1.0 - tone_dist(tone_shape(fit.px), rt) / e0);
    let rc = cast(refp);
    let c0 = dist2(cast(p0), rc);
    let t3 = (c0 >= 1.0).then(|| 1.0 - dist2(cast(fit), rc) / c0);
    [t1, t2, t3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_color::refgrade::measure::LabConv;

    /// A picture of `n` pixels of one linear sRGB colour.
    fn flat(lin: [f64; 3], n: usize) -> Vec<Px> {
        let conv = LabConv::new();
        vec![conv.px(lin); n]
    }

    fn pic(px: &[Px]) -> Pic<'_> {
        Pic { px, scene: None }
    }

    #[test]
    fn rng_is_splitmix64_with_the_published_first_outputs() {
        // The reference values of SplitMix64 for seed 0.
        let mut r = Rng(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn hue_and_wrap_by_hand() {
        assert!((oklab_hue([1.0, 0.0])).abs() < 1e-12);
        assert!((oklab_hue([0.0, 1.0]) - 90.0).abs() < 1e-12);
        assert!((oklab_hue([-1.0, -1e-300]) - 180.0).abs() < 1e-9);
        assert!((oklab_hue([1.0, -1.0]) - 315.0).abs() < 1e-12);
        assert!((wrap_deg(350.0 - 10.0) + 20.0).abs() < 1e-12);
        assert!((wrap_deg(190.0) + 170.0).abs() < 1e-12);
        assert!((wrap_deg(-190.0) - 170.0).abs() < 1e-12);
    }

    #[test]
    fn a_fit_that_does_nothing_has_no_harm_and_no_transfer() {
        // The photo is a grey ramp with a red patch; the fit is the photo.
        let conv = LabConv::new();
        let mut p: Vec<Px> = (0..1000)
            .map(|i| {
                let v = (i as f64 + 0.5) / 1000.0;
                conv.px([v, v, v])
            })
            .collect();
        for q in p.iter_mut().take(200) {
            *q = conv.px([0.6, 0.1, 0.1]);
        }
        let r = flat([0.2, 0.3, 0.5], 1000);
        let h = harms(&pic(&p), &pic(&p), &pic(&r));
        assert!(h.iter().all(|x| !x.1), "{h:?}");
        assert_eq!(h[0].0, Some(0.0));
        assert_eq!(h[5].0, Some(0.0));
        assert_eq!(h[4].0, Some(0.0));
        let t = transfers(&pic(&p), &pic(&p), &pic(&r), &sample_ab(&p));
        // The fit is the photo: 1 - d/d = 0 on every metric that applies.
        for v in t.into_iter().flatten() {
            assert!(v.abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn a_fit_that_is_the_reference_transfers_everything() {
        let conv = LabConv::new();
        let p: Vec<Px> = (0..1000)
            .map(|i| {
                let v = (i as f64 + 0.5) / 1000.0;
                conv.px([v, v * 0.8, v * 0.6])
            })
            .collect();
        let r: Vec<Px> = (0..1000)
            .map(|i| {
                let v = (i as f64 + 0.5) / 1000.0;
                conv.px([v * 0.5, v, v * 0.7])
            })
            .collect();
        let t = transfers(&pic(&p), &pic(&r), &pic(&r), &sample_ab(&p));
        for v in t {
            assert!((v.expect("applies") - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn a_fit_further_from_the_reference_scores_below_zero() {
        let p = flat([0.5, 0.4, 0.35], 600);
        let r = flat([0.4, 0.5, 0.4], 600);
        let away = flat([0.6, 0.3, 0.3], 600);
        let t = transfers(&pic(&p), &pic(&away), &pic(&r), &sample_ab(&p));
        assert!(t[0].expect("t1") < 0.0);
        assert!(t[2].expect("t3") < 0.0);
    }

    #[test]
    fn a_denominator_under_the_limit_skips_the_metric() {
        // The photo is the reference: every denominator is 0.
        let p = flat([0.3, 0.4, 0.5], 500);
        let t = transfers(&pic(&p), &pic(&p), &pic(&p), &sample_ab(&p));
        assert_eq!(t, [None, None, None]);
        // The same colour with a different L only: no colour or cast to transfer, the tone shape of two flat pictures is equal.
        let q = flat([0.12, 0.16, 0.2], 500);
        let t = transfers(&pic(&p), &pic(&p), &pic(&q), &sample_ab(&p));
        assert!(t[0].is_some() && t[2].is_some());
    }

    #[test]
    fn the_tone_shape_keeps_the_key_out() {
        // A ramp of L*: the shape is (span, relative median), the same for a ramp lifted by 20.
        let mk = |lift: f64| -> Vec<Px> {
            (0..1001)
                .map(|i| Px {
                    lab: [lift + 50.0 * i as f64 / 1000.0, 0.0, 0.0],
                    ..Px::default()
                })
                .collect()
        };
        let (a, b) = (tone_shape(&mk(0.0)), tone_shape(&mk(20.0)));
        assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        // p2 = 1, p98 = 49 on the unlifted ramp.
        assert!((a[0] - 48.0).abs() < 1e-9);
        assert!((a[1] - 0.5).abs() < 1e-9);
        assert!((tone_dist([48.0, 0.5], [38.0, 0.4]) - (10.0 / 50.0 + 0.1)).abs() < 1e-12);
    }

    #[test]
    fn the_cast_leaves_out_skin_and_the_extreme_lightnesses() {
        let px = vec![
            Px {
                lab: [50.0, 10.0, -4.0],
                ..Px::default()
            },
            Px {
                lab: [60.0, 20.0, 4.0],
                ..Px::default()
            },
            // Too dark, too light and skin: none counts.
            Px {
                lab: [10.0, 90.0, 90.0],
                ..Px::default()
            },
            Px {
                lab: [95.0, -90.0, 90.0],
                ..Px::default()
            },
            Px {
                lab: [50.0, 70.0, 70.0],
                ..Px::default()
            },
        ];
        let skin = [false, false, false, false, true];
        let class = [0u8; 5];
        let c = cast(&Pic {
            px: &px,
            scene: Some(SceneRef {
                class: &class,
                skin: &skin,
            }),
        });
        assert!((c[0] - 15.0).abs() < 1e-12 && c[1].abs() < 1e-12);
        // Without a scene nothing is skin.
        let c2 = cast(&pic(&px));
        assert!(c2[0] > 15.0);
    }

    #[test]
    fn h1_flags_a_grey_turned_purple_unless_the_reference_is_purple() {
        let conv = LabConv::new();
        let n = 1000;
        let grey = flat([0.2, 0.2, 0.2], n);
        // Oklab hue about 330: a purple.
        let purple = flat([0.3, 0.1, 0.3], n);
        let ph = oklab_hue(purple[0].ok);
        assert!((300.0..=355.0).contains(&ph), "{ph}");
        let teal = flat([0.1, 0.3, 0.3], n);
        let h = harms(&pic(&grey), &pic(&purple), &pic(&teal));
        assert!(h[0].1 && h[0].0.expect("value") > 0.99);
        // A purple reference (share of C* > 8 purple at 100 %) is a licence.
        let h = harms(&pic(&grey), &pic(&purple), &pic(&purple));
        assert!(!h[0].1);
        // 1.5 % of the frame is under the 2 % line.
        let mut some = grey.clone();
        for q in some.iter_mut().take(15) {
            *q = conv.px([0.3, 0.1, 0.3]);
        }
        assert!(!harms(&pic(&grey), &pic(&some), &pic(&teal))[0].1);
        for q in some.iter_mut().take(25) {
            *q = conv.px([0.3, 0.1, 0.3]);
        }
        assert!(harms(&pic(&grey), &pic(&some), &pic(&teal))[0].1);
        // A purple that was already purple (same hue, chroma at least 2) is not a turn.
        assert!(!harms(&pic(&purple), &pic(&purple), &pic(&teal))[0].1);
    }

    /// A scene of `n` pixels with the given class and skin everywhere.
    struct Fixed {
        class: Vec<u8>,
        skin: Vec<bool>,
    }

    impl Fixed {
        fn new(n: usize, class: u8, skin: bool) -> Self {
            Self {
                class: vec![class; n],
                skin: vec![skin; n],
            }
        }

        fn pic<'a>(&'a self, px: &'a [Px]) -> Pic<'a> {
            Pic {
                px,
                scene: Some(SceneRef {
                    class: &self.class,
                    skin: &self.skin,
                }),
            }
        }
    }

    #[test]
    fn h2_tests_the_skin_hue_and_the_chroma_ratio() {
        let n = 400;
        let sc = Fixed::new(n, 7, true);
        let skin = flat([0.45, 0.28, 0.2], n);
        let (a_hue, a_c) = {
            let px = &skin[0];
            (oklab_hue(px.ok), px.lab[1].hypot(px.lab[2]))
        };
        assert!(a_c >= 4.0);
        let refp = flat([0.3, 0.3, 0.3], n);
        // The same skin: nothing.
        assert!(!harms(&sc.pic(&skin), &sc.pic(&skin), &pic(&refp))[1].1);
        // A hue turned by 20 degrees: flagged, the value is the shift.
        let green = flat([0.4, 0.33, 0.2], n);
        let dh = wrap_deg(oklab_hue(green[0].ok) - a_hue).abs();
        assert!(dh > 10.0, "{dh}");
        let h = harms(&sc.pic(&skin), &sc.pic(&green), &pic(&refp));
        assert!(h[1].1);
        assert!((h[1].0.expect("value") - dh).abs() < 1e-9);
        // Washed out to a third of the chroma (ratio under 0.55), the hue about the same.
        let pale = flat([0.40, 0.33, 0.29], n);
        let h = harms(&sc.pic(&skin), &sc.pic(&pale), &pic(&refp));
        let ratio = (pale[0].lab[1].hypot(pale[0].lab[2])) / a_c;
        assert!(ratio < 0.55 || h[1].0.is_some_and(|d| d > 10.0), "{ratio}");
        assert!(h[1].1);
        // Under 1 % skin the check does not apply.
        let mut none = Fixed::new(n, 7, false);
        for s in none.skin.iter_mut().take(3) {
            *s = true;
        }
        let h = harms(&none.pic(&skin), &none.pic(&green), &pic(&refp));
        assert_eq!(h[1], (None, false));
        // A pale skin (mean-vector C* under 4): the hue is not tested, so no value; the chroma ratio of an unchanged skin is 1.
        let pale_skin = flat([0.30, 0.285, 0.27], n);
        assert!(oklab_hue(pale_skin[0].ok).is_finite());
        assert!(pale_skin[0].lab[1].hypot(pale_skin[0].lab[2]) < 4.0);
        let h = harms(&sc.pic(&pale_skin), &sc.pic(&pale_skin), &pic(&refp));
        assert_eq!(h[1], (None, false));
    }

    #[test]
    fn h3_flags_a_sky_that_changes_hue_when_the_reference_has_none() {
        let n = 400;
        let sc = Fixed::new(n, 1, false);
        let sky = flat([0.15, 0.25, 0.6], n);
        let teal = flat([0.1, 0.45, 0.45], n);
        let refp = Fixed::new(n, 4, false);
        let ref_px = flat([0.3, 0.3, 0.3], n);
        let h = harms(&sc.pic(&sky), &sc.pic(&teal), &refp.pic(&ref_px));
        assert!(h[2].0.is_some_and(|v| v > 15.0));
        assert!(h[2].1);
        // The reference has sky itself (2 % or more): allowed.
        let ref_sky = Fixed::new(n, 1, false);
        let h = harms(&sc.pic(&sky), &sc.pic(&teal), &ref_sky.pic(&ref_px));
        assert_eq!(h[2], (None, false));
        // A dark sky (median L* under 20) is not tested.
        let night = flat([0.004, 0.006, 0.02], n);
        assert!(night[0].lab[0] < 20.0);
        let night2 = flat([0.003, 0.01, 0.01], n);
        let h = harms(&sc.pic(&night), &sc.pic(&night2), &refp.pic(&ref_px));
        assert_eq!(h[2], (None, false));
        // The same sky in vegetation (class 2) is tested whatever its lightness.
        let veg = Fixed::new(n, 2, false);
        let leaf = flat([0.05, 0.003, 0.02], n);
        let leaf2 = flat([0.003, 0.05, 0.02], n);
        let h = harms(&veg.pic(&leaf), &veg.pic(&leaf2), &refp.pic(&ref_px));
        assert!(h[2].1);
    }

    #[test]
    fn h4_flags_whites_tinted_beyond_the_reference() {
        let n = 400;
        let white = flat([0.9, 0.9, 0.9], n);
        let warm = flat([1.0, 0.8, 0.45], n);
        assert!(cstar(&warm[0]) > 12.0);
        let neutral_ref = flat([0.5, 0.5, 0.5], n);
        let h = harms(&pic(&white), &pic(&warm), &pic(&neutral_ref));
        assert!(h[3].1 && h[3].0.expect("value") > 12.0);
        // A reference with whites of its own (L* 80 or more, C* under 8) whose mean C* is about 6: the fit may go to 6 + 4 = 10.
        let ref_whites = flat([0.95, 0.9, 0.8], n);
        let rc = cstar(&ref_whites[0]);
        assert!(
            ref_whites[0].lab[0] >= 80.0 && (5.0..8.0).contains(&rc),
            "{rc}"
        );
        let tint = |lin: [f64; 3]| flat(lin, n);
        let nine = tint([0.92, 0.86, 0.72]);
        let c9 = cstar(&nine[0]);
        assert!(c9 > 8.0 && c9 < rc + 4.0, "{c9} {rc}");
        assert!(!harms(&pic(&white), &pic(&nine), &pic(&ref_whites))[3].1);
        // The same fit against a neutral reference (C* 0 on its whites): the line is 8.
        assert!(harms(&pic(&white), &pic(&nine), &pic(&neutral_ref))[3].1);
        // A reference with no whites of its own has a mean of 0: the line is 8 again.
        assert!(harms(&pic(&white), &pic(&warm), &pic(&warm))[3].1);
        // No whites in the photo: not tested.
        let dark = flat([0.05, 0.05, 0.05], n);
        assert_eq!(
            harms(&pic(&dark), &pic(&warm), &pic(&neutral_ref))[3],
            (None, false)
        );
        // A chroma of at most 8 passes.
        let slight = flat([0.9, 0.88, 0.84], n);
        assert!(!harms(&pic(&white), &pic(&slight), &pic(&neutral_ref))[3].1);
    }

    #[test]
    fn h5_flags_clipping_beyond_the_larger_of_photo_and_reference() {
        let conv = LabConv::new();
        let n = 1000;
        let mut p = flat([0.4, 0.4, 0.4], n);
        let r = flat([0.4, 0.4, 0.4], n);
        // The fit blows 4 % of the frame to white.
        let mut f = p.clone();
        for q in f.iter_mut().take(40) {
            *q = conv.px([1.0, 1.0, 1.0]);
        }
        let h = harms(&pic(&p), &pic(&f), &pic(&r));
        assert!(h[4].1 && (h[4].0.expect("value") - 0.04).abs() < 1e-12);
        // 3 % exactly is not over the line.
        let mut f = p.clone();
        for q in f.iter_mut().take(30) {
            *q = conv.px([1.0, 1.0, 1.0]);
        }
        assert!(!harms(&pic(&p), &pic(&f), &pic(&r))[4].1);
        // The photo already had them: the excess is against the photo's share.
        for q in p.iter_mut().take(40) {
            *q = conv.px([1.0, 1.0, 1.0]);
        }
        let mut f = p.clone();
        for q in f.iter_mut().skip(40).take(60) {
            *q = conv.px([1.0, 1.0, 1.0]);
        }
        let h = harms(&pic(&p), &pic(&f), &pic(&r));
        assert!((h[4].0.expect("value") - 0.06).abs() < 1e-12);
        assert!(h[4].1);
        // Blacks count the same way.
        let mut f = r.clone();
        for q in f.iter_mut().take(50) {
            *q = conv.px([0.0, 0.0, 0.0]);
        }
        assert!(harms(&pic(&r), &pic(&f), &pic(&r))[4].1);
    }

    #[test]
    fn h6_flags_a_colour_reversal_on_coloured_non_skin_pixels() {
        let n = 1000;
        let red = flat([0.5, 0.05, 0.05], n);
        let cyan = flat([0.02, 0.4, 0.4], n);
        let r = flat([0.3, 0.3, 0.3], n);
        let h = harms(&pic(&red), &pic(&cyan), &pic(&r));
        assert!(h[5].1 && h[5].0.expect("value") > 0.99);
        // The same turn on skin is H2's business, not H6's.
        let sc = Fixed::new(n, 7, true);
        assert_eq!(
            harms(&sc.pic(&red), &sc.pic(&cyan), &pic(&r))[5].0,
            Some(0.0)
        );
        // Grey pixels (C* under 10) have no hue to reverse.
        let grey = flat([0.2, 0.2, 0.2], n);
        assert!(!harms(&pic(&grey), &pic(&cyan), &pic(&r))[5].1);
        // 20 degrees of turn is not a reversal.
        let orange = flat([0.5, 0.12, 0.04], n);
        assert!(!harms(&pic(&red), &pic(&orange), &pic(&r))[5].1);
    }

    #[test]
    fn the_sample_is_seeded_and_the_distance_is_symmetric_and_zero_on_itself() {
        let conv = LabConv::new();
        let p: Vec<Px> = (0..777)
            .map(|i| {
                let v = (i as f64 + 0.5) / 777.0;
                conv.px([v, v * v, 0.3])
            })
            .collect();
        let (a, b) = (sample_ab(&p), sample_ab(&p));
        assert_eq!(a, b);
        assert_eq!(a.len(), SW_SAMPLES);
        assert!(sliced_w1(&a, &b).abs() < 1e-15);
        let q: Vec<Px> = (0..777).map(|_| conv.px([0.1, 0.5, 0.2])).collect();
        let c = sample_ab(&q);
        assert!((sliced_w1(&a, &c) - sliced_w1(&c, &a)).abs() < 1e-12);
        assert!(sliced_w1(&a, &c) > 1.0);
    }
}
