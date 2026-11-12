//! Tests of the classical proposer and detector on synthetic skin.

use super::synth::{Kind, Lesion, Rng, clean_skin, random_lesions, skin_image, with_lesions};
use super::{ClassicalBeautyMarks, ClassicalProposer};
use crate::blemish::encode_channel;
use crate::mock::{MockBeautyMarks, MockMesh, MockParser};
use crate::models::{BeautyMarkDetector, BlemishProposer, Models};
use crate::{Face, LinearAwg4Image, Rect, RetouchCache, SkinSettings, prepare_skin, retouch_skin};

const N: usize = 450;

/// Encodes like `docs/27` 6.1 step 2; `mean` fixes the normalisation (None: the image's own).
fn encode(img: &LinearAwg4Image, mean: Option<f64>) -> (Vec<f32>, f64) {
    let mean = mean.unwrap_or_else(|| {
        let s: f64 = img
            .data
            .iter()
            .map(|p| {
                (encode_channel(p[0], 1.0)
                    + 2.0 * encode_channel(p[1], 1.0)
                    + encode_channel(p[2], 1.0))
                    / 4.0
            })
            .sum();
        s / img.data.len() as f64
    });
    let n = 0.5 / mean;
    let mut out = Vec::with_capacity(3 * img.data.len());
    for c in 0..3 {
        out.extend(
            img.data
                .iter()
                .map(|p| (encode_channel(p[c], 1.0) * n) as f32),
        );
    }
    (out, mean)
}

/// The proposer over a whole planar image: tiles with replicated edges, as `docs/27` 6.1 step 3.
fn run_tiled(p: &dyn BlemishProposer, planar: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (t, b) = (p.output_size_px() as usize, p.border_px() as usize);
    let side = t + 2 * b;
    let origins = |len: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..len).step_by(t).collect();
        if len > t {
            *v.last_mut().unwrap() = len - t;
            v.dedup();
        }
        v
    };
    let mut out = vec![0.0f32; 3 * w * h];
    for &ty in &origins(h) {
        for &tx in &origins(w) {
            let mut input = vec![0.0f32; 3 * side * side];
            for c in 0..3 {
                for y in 0..side {
                    let sy = (ty as i64 + y as i64 - b as i64).clamp(0, h as i64 - 1) as usize;
                    for x in 0..side {
                        let sx = (tx as i64 + x as i64 - b as i64).clamp(0, w as i64 - 1) as usize;
                        input[c * side * side + y * side + x] = planar[c * w * h + sy * w + sx];
                    }
                }
            }
            let o = p.run(&input).unwrap();
            for c in 0..3 {
                for y in 0..t.min(h - ty) {
                    for x in 0..t.min(w - tx) {
                        out[c * w * h + (ty + y) * w + tx + x] = o[c * t * t + y * t + x];
                    }
                }
            }
        }
    }
    out
}

fn disc_mean(plane: &[f32], w: usize, cx: f64, cy: f64, r: f64) -> f64 {
    let (mut s, mut n) = (0.0, 0.0);
    for y in (cy - r).floor() as usize..=(cy + r).ceil() as usize {
        for x in (cx - r).floor() as usize..=(cx + r).ceil() as usize {
            if (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2) <= r * r {
                s += f64::from(plane[y * w + x]);
                n += 1.0;
            }
        }
    }
    s / n
}

fn lesions_for_proposer() -> Vec<Lesion> {
    vec![
        Lesion::new(Kind::Papule, 110.0, 110.0, 6.0),
        Lesion::new(Kind::Papule, 330.0, 120.0, 4.0),
        Lesion::new(Kind::Papule, 150.0, 330.0, 7.0),
        Lesion::new(Kind::Brown, 300.0, 310.0, 2.5),
        Lesion::new(Kind::Brown, 220.0, 90.0, 5.0),
        Lesion::new(Kind::Papule, 370.0, 360.0, 7.0),
    ]
}

fn residuals(seed: u64, lesions: &[Lesion]) -> (Vec<f32>, Vec<f32>, Vec<(f64, f64)>) {
    let clean = skin_image(N, N, seed, &[]);
    let img = skin_image(N, N, seed, lesions);
    let (inp, mean) = encode(&img, None);
    let (cl, _) = encode(&clean, Some(mean));
    let p = ClassicalProposer::with_tile_px(N as f64, N as u32);
    let out = run_tiled(&p, &inp, N, N);
    let plane = N * N;
    let mut res = Vec::new();
    for l in lesions {
        // The green channel always darkens; red for the brown spots only.
        let chans: &[usize] = if l.kind == Kind::Papule {
            &[1]
        } else {
            &[0, 1, 2]
        };
        for &c in chans {
            let s = c * plane..(c + 1) * plane;
            let r = 0.7 * l.radius;
            let (i, o, k) = (
                disc_mean(&inp[s.clone()], N, l.cx, l.cy, r),
                disc_mean(&out[s.clone()], N, l.cx, l.cy, r),
                disc_mean(&cl[s], N, l.cx, l.cy, r),
            );
            res.push(((i - k) / k, (o - k) / (i - k)));
        }
    }
    (inp, out, res)
}

#[test]
fn proposer_removes_red_and_dark_spots() {
    let (_, _, res) = residuals(11, &lesions_for_proposer());
    for (drop, resid) in &res {
        println!("drop {drop:.3} residual {resid:.3}");
        assert!(*drop < -0.03, "the lesion is visible in the input: {drop}");
        assert!(
            resid.abs() < 0.3,
            "residual contrast {resid} for a drop of {drop}"
        );
    }
    let worst = res.iter().map(|r| r.1.abs()).fold(0.0, f64::max);
    println!("proposer worst residual contrast fraction: {worst:.3}");
}

#[test]
fn clean_skin_is_untouched_and_far_pixels_are_identical() {
    // A patch without lesions: the whole output equals the input.
    for seed in 1u64..=8 {
        let img = skin_image(N, N, seed, &[]);
        let (inp, _) = encode(&img, None);
        let out = run_tiled(&ClassicalProposer::new(N as f64), &inp, N, N);
        assert_eq!(out, inp, "seed {seed}");
    }
    // With lesions, everything far from them is bit-identical.
    let lesions = lesions_for_proposer();
    let (inp, out, _) = residuals(11, &lesions);
    let mut changed = 0;
    for y in 0..N {
        for x in 0..N {
            let far = lesions.iter().all(|l| {
                ((x as f64 - l.cx).powi(2) + (y as f64 - l.cy).powi(2)).sqrt()
                    > 2.0 * l.radius + 8.0
            });
            for c in 0..3 {
                let i = c * N * N + y * N + x;
                if far {
                    assert_eq!(out[i], inp[i], "pixel ({x}, {y}) channel {c}");
                } else if out[i] != inp[i] {
                    changed += 1;
                }
            }
        }
    }
    assert!(changed > 0);
}

#[test]
fn texture_is_kept_inside_the_spots() {
    let l = Lesion::new(Kind::Papule, 225.0, 225.0, 9.0);
    let (inp, out, _) = residuals(5, &[l]);
    let hp = |v: &[f32]| -> f64 {
        // Std of the pixel minus its 4-neighbour mean, inside the disc of radius R.
        let (mut s, mut n) = (0.0, 0.0);
        for y in 200..250usize {
            for x in 200..250usize {
                if (x as f64 - l.cx).powi(2) + (y as f64 - l.cy).powi(2) <= 81.0 {
                    let i = N * N + y * N + x; // green
                    let m = (v[i - 1] + v[i + 1] + v[i - N] + v[i + N]) / 4.0;
                    s += f64::from(v[i] - m).powi(2);
                    n += 1.0;
                }
            }
        }
        (s / n).sqrt()
    };
    let (ti, to) = (hp(&inp), hp(&out));
    println!("high-pass std inside a spot: in {ti:.5}, out {to:.5}");
    assert!(to > 0.3 * ti, "texture inside the spot: in {ti}, out {to}");
}

#[test]
fn proposer_is_deterministic_with_1_and_4_threads() {
    let img = skin_image(N, N, 11, &lesions_for_proposer());
    let (inp, _) = encode(&img, None);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run_tiled(&ClassicalProposer::with_tile_px(N as f64, 256), &inp, N, N))
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn tiling_does_not_change_the_result() {
    let img = skin_image(N, N, 11, &lesions_for_proposer());
    let (inp, _) = encode(&img, None);
    let a = run_tiled(&ClassicalProposer::with_tile_px(N as f64, 128), &inp, N, N);
    let b = run_tiled(
        &ClassicalProposer::with_tile_px(N as f64, N as u32),
        &inp,
        N,
        N,
    );
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    println!("tile 128 against one tile, max abs difference: {max:e}");
    assert!(max < 1e-3, "{max}");
}

fn models(boxes: Vec<Rect>, classical: bool) -> Models {
    Models {
        parser: Box::new(MockParser),
        mesh: Box::new(MockMesh),
        beauty_marks: Some(if classical {
            Box::new(ClassicalBeautyMarks)
        } else {
            Box::new(MockBeautyMarks { boxes })
        }),
        proposer: Box::new(ClassicalProposer::new(N as f64)),
    }
}

fn face() -> Face {
    Face {
        rect: Rect::new(0, 0, N as u32, N as u32),
        scale: 1.0,
    }
}

fn pipeline(img: &LinearAwg4Image, amount: f32, m: &Models) -> LinearAwg4Image {
    let faces = [(
        face(),
        SkinSettings {
            blemish: amount,
            ..Default::default()
        },
    )];
    let mut cache = RetouchCache::new(1, 1);
    prepare_skin(img, &faces, m, &mut cache).unwrap();
    retouch_skin(img, (N as u32, N as u32), &faces, 1.0, m, &mut cache, None).unwrap()
}

fn lin_disc(img: &LinearAwg4Image, c: usize, l: &Lesion, k: f64) -> f64 {
    let (mut s, mut n) = (0.0, 0.0);
    let r = k * l.radius;
    for y in (l.cy - r).floor() as usize..=(l.cy + r).ceil() as usize {
        for x in (l.cx - r).floor() as usize..=(l.cx + r).ceil() as usize {
            if (x as f64 - l.cx).powi(2) + (y as f64 - l.cy).powi(2) <= r * r {
                s += f64::from(img.data[y * N + x][c]);
                n += 1.0;
            }
        }
    }
    s / n
}

#[test]
fn pipeline_fades_lesions_and_leaves_clean_skin() {
    let clean = skin_image(N, N, 21, &[]);
    let m = models(Vec::new(), false);
    assert_eq!(
        pipeline(&clean, 60.0, &m).data,
        clean.data,
        "clean skin is bit-identical"
    );
    let lesions = [
        Lesion::new(Kind::Papule, 170.0, 190.0, 6.0),
        Lesion::new(Kind::Brown, 280.0, 250.0, 4.0),
    ];
    let img = skin_image(N, N, 21, &lesions);
    let out = pipeline(&img, 60.0, &m);
    for l in &lesions {
        let rel =
            |im: &LinearAwg4Image| lin_disc(im, 1, l, 0.7) / lin_disc(&clean, 1, l, 0.7) - 1.0;
        let (before, after) = (rel(&img), rel(&out));
        println!(
            "pipeline Blemish 60, {:?}: contrast {before:.3} -> {after:.3}",
            l.kind
        );
        assert!(
            after.abs() < 0.4 * before.abs(),
            "{:?}: {before} -> {after}",
            l.kind
        );
    }
}

/// `docs/27` decision 13: the models run on the analysis picture (neutral) and the blemish gain map is applied to the node's input,
/// which has another grade. A gain is a ratio, so an exposure change leaves it exact; a tone curve changes the ratio a little. The
/// map of the neutral picture applied to a picture with +2 EV is the neutral result times 4 (to rounding), and applied to a picture
/// through a strong power curve (0.7) it over-corrects to a contrast of the other sign and about half the size.
#[test]
fn a_map_of_one_grade_applies_to_another() {
    let lesion = Lesion::new(Kind::Papule, 170.0, 190.0, 6.0);
    let clean = skin_image(N, N, 21, &[]);
    let img = skin_image(N, N, 21, &[lesion]);
    let m = models(Vec::new(), false);
    let faces = [(
        face(),
        SkinSettings {
            blemish: 60.0,
            ..Default::default()
        },
    )];
    let mut cache = RetouchCache::new(1, 1);
    prepare_skin(&img, &faces, &m, &mut cache).unwrap();
    let apply = |image: &LinearAwg4Image, cache: &mut RetouchCache| {
        retouch_skin(image, (N as u32, N as u32), &faces, 1.0, &m, cache, None).unwrap()
    };
    let graded = |f: &dyn Fn(f32) -> f32| {
        LinearAwg4Image::new(
            N as u32,
            N as u32,
            img.data.iter().map(|p| p.map(f)).collect(),
        )
        .unwrap()
    };
    let neutral = apply(&img, &mut cache);
    // Exposure: +2 EV is a multiplication by 4, and the gain does not care.
    let bright = apply(&graded(&|v| v * 4.0), &mut cache);
    let worst = bright
        .data
        .iter()
        .zip(&neutral.data)
        .flat_map(|(b, n)| (0..3).map(move |c| (b[c] - 4.0 * n[c]).abs() / (4.0 * n[c]).max(1e-6)))
        .fold(0.0f32, f32::max);
    println!(
        "map of the neutral picture on a picture at +2 EV: worst relative difference {worst:e}"
    );
    assert!(worst < 1e-5, "{worst}");
    // A power curve (the display's own tone is about this far from linear light), against the map of its own picture.
    let curve = graded(&|v| v.powf(0.7));
    let rel = |im: &LinearAwg4Image, reference: &LinearAwg4Image| {
        lin_disc(im, 1, &lesion, 0.7) / lin_disc(reference, 1, &lesion, 0.7) - 1.0
    };
    let curve_clean = LinearAwg4Image::new(
        N as u32,
        N as u32,
        clean.data.iter().map(|p| p.map(|v| v.powf(0.7))).collect(),
    )
    .unwrap();
    let before = rel(&curve, &curve_clean);
    let neutral_map = apply(&curve, &mut cache);
    let mut own_cache = RetouchCache::new(2, 2);
    prepare_skin(&curve, &faces, &m, &mut own_cache).unwrap();
    let own_map = apply(&curve, &mut own_cache);
    let (a, b) = (rel(&neutral_map, &curve_clean), rel(&own_map, &curve_clean));
    println!(
        "papule contrast on a power-curve picture: {before:.3}; after the neutral map {a:.3}, after its own map {b:.3}"
    );
    // A tone curve this strong (the gain is applied in the curve's own space, where the spot's ratio is the curve's power of
    // the neutral one) makes the correction too big, not too small: it is within the original contrast. The curves of
    // a corrector upstream are gentler (`docs/27` decision 13).
    assert!(a.abs() < 0.6 * before.abs(), "{before} -> {a}");
}

/// The upstream curves of the S-12 measurement: power curves and an S-curve (a smoothstep in the log2 of the value around
/// mid grey, 8 stops wide: a slope of 1.5 at mid grey).
fn upstream_curves() -> Vec<(&'static str, Box<dyn Fn(f32) -> f32>)> {
    let power = |p: f32| -> Box<dyn Fn(f32) -> f32> { Box::new(move |v| v.powf(p)) };
    let s_curve: Box<dyn Fn(f32) -> f32> = Box::new(|v| {
        let u = (((f64::from(v) / 0.18).log2() + 4.0).clamp(0.0, 8.0)) / 8.0;
        (0.18 * 2f64.powf(8.0 * (3.0 * u * u - 2.0 * u * u * u) - 4.0)) as f32
    });
    vec![
        ("power 0.5", power(0.5)),
        ("power 0.7", power(0.7)),
        ("power 1.4", power(1.4)),
        ("power 2.0", power(2.0)),
        ("S-curve", s_curve),
    ]
}

/// `docs/27` decision 13 and S-12: the blemish gain map is made on the analysis picture and applied to a picture that has a
/// curve upstream. Applying the gain through the relation between the two pictures (`ToneRelation`) puts the papule back to
/// what the picture's own map gives, for power curves and an S-curve, and an exposure change is untouched (bit for bit).
#[test]
fn a_tone_relation_corrects_the_gain_under_an_upstream_curve() {
    let lesions = [
        Lesion::new(Kind::Papule, 170.0, 190.0, 6.0),
        Lesion::new(Kind::Brown, 280.0, 250.0, 4.0),
    ];
    let clean = skin_image(N, N, 21, &[]);
    let img = skin_image(N, N, 21, &lesions);
    let m = models(Vec::new(), false);
    let faces = [(
        face(),
        SkinSettings {
            blemish: 100.0,
            ..Default::default()
        },
    )];
    let mut cache = RetouchCache::new(1, 1);
    prepare_skin(&img, &faces, &m, &mut cache).unwrap();
    let grade = |im: &LinearAwg4Image, f: &dyn Fn(f32) -> f32| {
        LinearAwg4Image::new(
            N as u32,
            N as u32,
            im.data.iter().map(|p| p.map(f)).collect(),
        )
        .unwrap()
    };
    let place = crate::Placement {
        origin_px: (0, 0),
        working_px: (N as u32, N as u32),
    };
    let contrast = |im: &LinearAwg4Image, reference: &LinearAwg4Image| -> Vec<f64> {
        lesions
            .iter()
            .map(|l| lin_disc(im, 1, l, 0.7) / lin_disc(reference, 1, l, 0.7) - 1.0)
            .collect()
    };
    // Exposure: the relation is plain and the result is the plain gain, bit for bit.
    let bright = grade(&img, &|v| v * 4.0);
    let rel = crate::ToneRelation::estimate(&img.data, &bright.data);
    assert!(rel.is_plain());
    let region = |image: &LinearAwg4Image, tone: Option<&crate::ToneRelation>| {
        crate::retouch_skin_region_toned(
            image,
            place,
            (N as u32, N as u32),
            &faces,
            1.0,
            &cache,
            None,
            tone,
        )
        .unwrap()
    };
    assert_eq!(region(&bright, Some(&rel)).data, region(&bright, None).data);
    let mut table = String::new();
    for (name, f) in upstream_curves() {
        let graded = grade(&img, f.as_ref());
        let graded_clean = grade(&clean, f.as_ref());
        let before = contrast(&graded, &graded_clean);
        let plain = contrast(&region(&graded, None), &graded_clean);
        let rel = crate::ToneRelation::estimate(&img.data, &graded.data);
        let toned = contrast(&region(&graded, Some(&rel)), &graded_clean);
        let mut own_cache = RetouchCache::new(2, 2);
        prepare_skin(&graded, &faces, &m, &mut own_cache).unwrap();
        let own_image = retouch_skin(
            &graded,
            (N as u32, N as u32),
            &faces,
            1.0,
            &m,
            &mut own_cache,
            None,
        )
        .unwrap();
        let own = contrast(&own_image, &graded_clean);
        table += &format!(
            "{name:10} papule: in {:+.3}, plain map {:+.3}, with the relation {:+.3}, own map {:+.3}; brown: in {:+.3}, plain {:+.3}, relation {:+.3}, own {:+.3}\n",
            before[0], plain[0], toned[0], own[0], before[1], plain[1], toned[1], own[1]
        );
        println!("{table}");
        for k in 0..2 {
            assert!(
                toned[k].abs() <= 0.5 * plain[k].abs().max(0.02) + 0.02,
                "{name} lesion {k}: plain {} relation {}",
                plain[k],
                toned[k]
            );
            assert!(toned[k].abs() < 0.3 * before[k].abs(), "{name} lesion {k}");
        }
    }
    println!("{table}");
}

#[test]
fn pipeline_keeps_a_mole_and_removes_a_papule_at_blemish_100() {
    let mole = Lesion::new(Kind::Mole, 215.0, 225.0, 6.5);
    let papule = Lesion::new(Kind::Papule, 255.0, 225.0, 6.0);
    let clean = skin_image(N, N, 33, &[]);
    let img = skin_image(N, N, 33, &[mole, papule]);
    let m = models(Vec::new(), true);
    let out = pipeline(&img, 100.0, &m);
    let rel = |im: &LinearAwg4Image, l: &Lesion| {
        lin_disc(im, 1, l, 0.7) / lin_disc(&clean, 1, l, 0.7) - 1.0
    };
    let (m0, m1) = (rel(&img, &mole), rel(&out, &mole));
    let (p0, p1) = (rel(&img, &papule), rel(&out, &papule));
    println!("pipeline Blemish 100: mole {m0:.3} -> {m1:.3}, papule {p0:.3} -> {p1:.3}");
    assert!(m1 < 0.9 * m0, "the mole keeps its contrast");
    assert!(p1.abs() < 0.3 * p0.abs(), "the papule is removed");
    // Without the detector the mole goes too (the proposer fixes every dark spot).
    let bare = pipeline(&img, 100.0, &models(Vec::new(), false));
    assert!(rel(&bare, &mole).abs() < 0.5 * m0.abs());
}

// ---------------------------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------------------------

fn patches() -> Vec<(LinearAwg4Image, Vec<Lesion>)> {
    (0..8u64)
        .map(|seed| {
            let mut rng = Rng(1000 + seed);
            let ls = random_lesions(
                &mut rng,
                N,
                N,
                (4, 4, 3),
                [(3.0, 8.0), (1.5, 3.0), (3.5, 8.0)],
                40.0,
                45.0,
            );
            (skin_image(N, N, 77 + seed, &ls), ls)
        })
        .collect()
}

#[test]
fn detector_finds_moles_and_not_papules() {
    let (mut tp, mut moles, mut boxes) = (0, 0, 0);
    let mut fp_kinds = Vec::new();
    for (img, ls) in patches() {
        let found = ClassicalBeautyMarks.detect(&img, &face()).unwrap();
        assert!(found.len() <= 10);
        let contains = |r: &Rect, l: &Lesion| {
            l.cx >= f64::from(r.x_px)
                && l.cx <= r.right_px() as f64
                && l.cy >= f64::from(r.y_px)
                && l.cy <= r.bottom_px() as f64
        };
        for l in ls.iter().filter(|l| l.kind == Kind::Mole) {
            moles += 1;
            if found.iter().any(|r| contains(r, l)) {
                tp += 1;
            }
        }
        for r in &found {
            boxes += 1;
            if !ls.iter().any(|l| l.kind == Kind::Mole && contains(r, l)) {
                fp_kinds.push(
                    ls.iter()
                        .filter(|l| contains(r, l))
                        .map(|l| l.kind)
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    let recall = tp as f64 / moles as f64;
    let precision = (boxes - fp_kinds.len()) as f64 / boxes.max(1) as f64;
    println!(
        "detector: recall {recall:.3} ({tp}/{moles}), precision {precision:.3}, false boxes {fp_kinds:?}"
    );
    assert!(recall >= 0.9, "recall {recall}");
    assert!(precision >= 0.9, "precision {precision}");
}

#[test]
fn detector_caps_the_count_and_pads_the_boxes() {
    // 16 moles: the cap is 10.
    let mut ls = Vec::new();
    for i in 0..16 {
        ls.push(Lesion::new(
            Kind::Mole,
            50.0 + 80.0 * (i % 5) as f64,
            50.0 + 90.0 * (i / 5) as f64,
            5.0 + (i % 4) as f64,
        ));
    }
    let img = skin_image(N, N, 9, &ls);
    let found = ClassicalBeautyMarks.detect(&img, &face()).unwrap();
    assert_eq!(found.len(), 10);
    // One mole: the box is the extent plus about 30 %.
    let one = Lesion::new(Kind::Mole, 200.0, 200.0, 6.0);
    let found = ClassicalBeautyMarks
        .detect(&skin_image(N, N, 9, &[one]), &face())
        .unwrap();
    assert_eq!(found.len(), 1);
    let r = found[0];
    println!(
        "mole of core radius 6: box {}x{} at ({}, {})",
        r.width_px, r.height_px, r.x_px, r.y_px
    );
    assert!(r.width_px >= 13 && r.width_px <= 24 && r.height_px >= 13 && r.height_px <= 24);
    // The inscribed ellipse of the box covers the mole core.
    let (ex, ey) = (
        f64::from(r.x_px) + f64::from(r.width_px) / 2.0,
        f64::from(r.y_px) + f64::from(r.height_px) / 2.0,
    );
    assert!((ex - 200.0).abs() < 2.0 && (ey - 200.0).abs() < 2.0);
    // Boxes stay inside the face rectangle for a face smaller than the image.
    let small = Face {
        rect: Rect::new(100, 100, 200, 200),
        scale: 0.4,
    };
    for b in ClassicalBeautyMarks
        .detect(
            &skin_image(N, N, 9, &[one, Lesion::new(Kind::Mole, 50.0, 50.0, 6.0)]),
            &small,
        )
        .unwrap()
    {
        assert!(b.x_px >= 95 && b.right_px() <= 305, "{b:?}");
    }
}

#[test]
fn run_time_per_450_px_face() {
    let img = skin_image(N, N, 11, &lesions_for_proposer());
    let (inp, _) = encode(&img, None);
    let p = ClassicalProposer::new(N as f64);
    let t = std::time::Instant::now();
    let _ = run_tiled(&p, &inp, N, N);
    let tp = t.elapsed();
    let t = std::time::Instant::now();
    let _ = ClassicalBeautyMarks.detect(&img, &face()).unwrap();
    println!(
        "time per 450 px face: proposer {tp:?}, detector {:?}",
        t.elapsed()
    );
    // Unused import guard for the generator in release-less builds.
    let _ = with_lesions(8, 8, &clean_skin(8, 8, 1), &[]);
}
