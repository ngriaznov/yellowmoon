//! RA-03b: the CA-aware green refinement of the adaptive ratio demosaic
//! (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 3.6, `INTEGRATION.md` section 6).
//!
//! * **The vector.** `ca_weighting` (a 32 x 32 tile at frame (4800, 3330) of a 6048 x 4024 frame)
//!   through the whole-piece routine and the banded one, within its tolerance (1e-5).
//! * **The curves from the lens.** `LensGeometry::ca_curves` reproduces the curves the vector
//!   carries from the same 16 + 16 CA knots and 16 distortion knots, and follows the R9 model: at
//!   the source radius of green, red is displaced by exactly `shift_R(r)`. No CA tag, CA switched
//!   off, or a distortion table that folds over gives no curves.
//! * **The path itself.** Banded equals whole, tiles equal the whole frame and crops at the four
//!   CFA phases give the identical interior, all bit for bit, with a placement inside a big frame
//!   so that the misregistration varies across the tile. The placement moves the CA field, a
//!   constant misregistration does not depend on it, and a centre that is not finite is refused.
//! * **The chain.** `Chain::run_tile` with a lens that has CA runs the CA path for the adaptive
//!   ratio demosaic and no other method; without CA (no tag, switched off) it runs the plain path.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_gpu::cpu::chain::{Chain, RawFrame};
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Error, Params};
use ym_gpu::cpu::demosaic::{CaCurveError, CaCurves, DemosaicInputs, Method, RadialCurve};
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::synth::{self, Rng};
use ym_gpu::cpu::tiling::{Halo, Rect, tile_grid};
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, Placement, RgbImage, demosaic, gains, resample};
use ym_noise::model::NoiseModel;

const VECTOR: &str = "tests/cpu_data/adaptive_ratio_demosaic/ca_weighting.json";

/// The model of the vectors' "low" noise: SNR at 18 % grey of 39.5 dB, shadow matching off.
fn low_noise() -> NoiseDn {
    NoiseDn::from_model(&NoiseModel {
        a: 2e-5,
        b: 1e-8,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: 1.0 / 65535.0,
    })
}

/// A noisy model (23.9 dB at 18 % grey): shadow matching on, halo 13.
fn high_noise() -> NoiseDn {
    NoiseDn::from_model(&NoiseModel {
        a: 5e-4,
        b: 4e-5,
        delta: [0.0; 4],
        sigma_row: 1e-3,
        q: 0.0,
    })
}

fn bits(img: &RgbImage) -> Vec<u32> {
    img.data.iter().map(|v| v.to_bits()).collect()
}

/// Largest absolute difference of two same-sized images; NaN when either holds a NaN.
fn image_diff(a: &RgbImage, b: &RgbImage) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "dimensions");
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| f64::from(*x - *y).abs())
        .fold(0.0, |acc, v| {
            if acc.is_nan() || v.is_nan() {
                f64::NAN
            } else {
                acc.max(v)
            }
        })
}

// ---------------------------------------------------------------------------------------------
// The vector
// ---------------------------------------------------------------------------------------------

fn vector() -> Value {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join(VECTOR);
    serde_json::from_str(&std::fs::read_to_string(path).expect("vector file")).expect("json")
}

fn doubles(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn ints(v: &Value) -> Vec<i16> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| i16::try_from(x.as_i64().expect("integer")).expect("i16 knot"))
        .collect()
}

fn vector_curve(v: &Value) -> RadialCurve {
    RadialCurve::new(doubles(&v["radius_px"]), doubles(&v["shift_px"])).expect("curve")
}

/// The vector decoded: the mosaic with its placement, the inputs and the expected output.
struct CaVector {
    json: Value,
    mosaic: Mosaic,
    gains: [f32; 3],
    noise: NoiseDn,
    curves: CaCurves,
    expected: Vec<f32>,
}

fn ca_vector() -> CaVector {
    let json = vector();
    let params = &json["params"];
    let input = &json["input"];
    let (w, h) = (
        input["width"].as_u64().expect("width") as usize,
        input["height"].as_u64().expect("height") as usize,
    );
    let cfa = Cfa::from_name(input["cfa"].as_str().expect("cfa")).expect("pattern");
    let data: Vec<f32> = doubles(&input["data"]).iter().map(|v| *v as f32).collect();
    let mut mosaic = Mosaic::from_data(w, h, cfa, data).expect("mosaic");
    let ca = &params["ca"];
    let knots = &ca["lens_knots"];
    mosaic.place = Placement {
        x0: ca["origin_x"].as_u64().expect("origin_x") as usize,
        y0: ca["origin_y"].as_u64().expect("origin_y") as usize,
        frame_width: knots["frame_width"].as_u64().expect("frame width") as usize,
        frame_height: knots["frame_height"].as_u64().expect("frame height") as usize,
    };
    let g = doubles(&params["gains"]);
    let n = &params["noise"];
    let noise = NoiseDn::from_model(&NoiseModel {
        a: n["a"].as_f64().expect("a"),
        b: n["b"].as_f64().expect("b"),
        delta: [0.0; 4],
        sigma_row: n["sigma_row"].as_f64().expect("sigma_row"),
        q: n["q"].as_f64().expect("q"),
    });
    let curves = CaCurves::new(
        ca["centre_x"].as_f64().expect("centre_x"),
        ca["centre_y"].as_f64().expect("centre_y"),
        vector_curve(&ca["red"]),
        vector_curve(&ca["blue"]),
    )
    .expect("curves");
    CaVector {
        expected: doubles(&json["expected"]["data"])
            .iter()
            .map(|v| *v as f32)
            .collect(),
        mosaic,
        gains: [g[0] as f32, g[1] as f32, g[2] as f32],
        noise,
        curves,
        json,
    }
}

#[test]
fn the_ca_vector_matches_the_oracle() {
    let v = ca_vector();
    let tolerance = v.json["tolerance"].as_f64().expect("tolerance");
    let inputs = DemosaicInputs {
        gains: v.gains,
        noise: v.noise,
        ca: Some(&v.curves),
    };
    let params = Params::default();
    let whole = adaptive_ratio::demosaic_whole(&v.mosaic, &inputs, &params).expect("whole");
    let banded = adaptive_ratio::demosaic(&v.mosaic, &inputs, &params).expect("banded");
    assert_eq!(bits(&whole), bits(&banded), "banded differs from whole");
    assert_eq!(whole.place, v.mosaic.place, "the output keeps the placement");
    let worst = whole
        .data
        .iter()
        .zip(&v.expected)
        .map(|(a, b)| f64::from(*a - *b).abs())
        .fold(0.0, f64::max);
    println!("ca_weighting: max |rust - oracle| = {worst:.3e} (tolerance {tolerance:.0e})");
    assert!(whole.data.iter().all(|x| x.is_finite()));
    assert!(worst <= tolerance, "{worst} > {tolerance}");
    // The plain path must give a visibly different answer on this tile, or the vector would not
    // exercise the CA path at all (the misregistration here is 0.68 to 0.72 px).
    let plain = adaptive_ratio::demosaic_whole(
        &v.mosaic,
        &DemosaicInputs {
            ca: None,
            ..inputs
        },
        &params,
    )
    .expect("plain");
    let d = image_diff(&plain, &whole);
    println!("ca_weighting: plain path differs from the CA path by up to {d:.3e}");
    assert!(d > 1e-3, "the CA path changes nothing on the vector: {d}");
    // The derived values of the vector, as the oracle printed them.
    let want = &v.json["params"]["derived"];
    let derived = adaptive_ratio::Derived::new(v.gains, v.noise, &params).expect("derived");
    assert!((derived.offset1_dn / want["K"].as_f64().expect("K") - 1.0).abs() < 1e-9);
    assert_eq!(derived.halo_px as u64, want["halo"].as_u64().expect("halo"));
}

// ---------------------------------------------------------------------------------------------
// The curves from the R9 lens model
// ---------------------------------------------------------------------------------------------

/// The lens tags of the vector's `lens_knots`: 16 CA knots per colour (red first) and 16
/// distortion knots, on the 6048 x 4024 frame.
fn vector_tags() -> (LensTags, usize, usize) {
    let json = vector();
    let k = &json["params"]["ca"]["lens_knots"];
    let mut ca = ints(&k["ca_red"]);
    ca.extend(ints(&k["ca_blue"]));
    let tags = LensTags {
        vignetting: vec![],
        ca,
        distortion: ints(&k["distortion"]),
    };
    (
        tags,
        k["frame_width"].as_u64().expect("w") as usize,
        k["frame_height"].as_u64().expect("h") as usize,
    )
}

#[test]
fn the_curves_from_the_lens_knots_match_the_vectors_curves() {
    let v = ca_vector();
    let (tags, fw, fh) = vector_tags();
    let geo = LensGeometry::new(fw, fh, &tags, LensSwitches::default()).expect("lens");
    let got = geo.ca_curves().expect("curves").expect("the lens has CA");
    assert_eq!(
        (got.centre_x, got.centre_y),
        (v.curves.centre_x, v.curves.centre_y)
    );
    let mut worst = 0.0f64;
    for (name, g, w) in [
        ("red", &got.red, &v.curves.red),
        ("blue", &got.blue, &v.curves.blue),
    ] {
        assert_eq!(g.len(), w.len(), "{name}: knot count");
        for (a, b) in g
            .radius_px()
            .iter()
            .zip(w.radius_px())
            .chain(g.shift_px().iter().zip(w.shift_px()))
        {
            // The vector prints 17 significant digits; serde_json may read one ulp off.
            worst = worst.max((a - b).abs() / b.abs().max(1.0));
        }
    }
    println!(
        "curves from the lens knots vs the vector's: {} knots, worst relative difference {worst:.2e}",
        got.red.len()
    );
    assert!(worst < 1e-13, "{worst}");
    // The curve spacing: 8 px of output radius, from 0 to past the corner of the frame.
    let hd = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    // (The knots are in source radius: this lens corrects a pincushion, F < 1, so the last one is
    // at about F * (hd + 16), short of hd; the curve extends linearly beyond it, in a part of the
    // mosaic that R9 never samples.)
    assert_eq!(got.red.len(), ((hd + 16.0) / 8.0).ceil() as usize + 1);
    assert_eq!(got.red.radius_px()[0], 0.0);
    assert_eq!(got.red.radius_px(), got.blue.radius_px());
    // And what comes out is usable as pass-2 input: |shift_R - shift_B| is about 0.7 px at the
    // tile of the vector.
    let ca = got.misregistration_px(
        ((4800.0 + 16.0 - 3024.0f64).powi(2) + (3330.0 + 16.0 - 2012.0f64).powi(2)).sqrt(),
    );
    assert!((0.6..0.8).contains(&ca), "{ca}");
}

#[test]
fn the_curves_follow_the_r9_model() {
    // Without distortion shift_c(r) = r (K_c(r) - 1). With it, R9 samples green at r F(r) and red at
    // r F(r) K_c(r): at the source radius of green the red plane is displaced by shift_R there.
    let (fw, fh) = (3000usize, 2000usize);
    let tags = LensTags {
        vignetting: vec![],
        ca: vec![0, 600, 2200, 4800, 8200, 12600, 0, -400, -1800, -4200, -7600, -12000],
        distortion: vec![0, 40, 160, 380, 700, 1100, 1600],
    };
    for (name, switches) in [
        ("with distortion", LensSwitches::default()),
        (
            "distortion off",
            LensSwitches {
                lens_corrections: false,
                chromatic_aberration: true,
            },
        ),
    ] {
        let geo = LensGeometry::new(fw, fh, &tags, switches).expect("lens");
        let curves = geo.ca_curves().expect("curves").expect("CA present");
        let place = Placement::whole(fw, fh);
        let (cx, cy) = place.centre();
        let mut rng = Rng::new(11);
        let mut worst: f64 = 0.0;
        for _ in 0..400 {
            let (px, py) = (
                f64::from(rng.range(0.0, fw as f32)),
                f64::from(rng.range(0.0, fh as f32)),
            );
            let [sr, sg, sb] = geo.source_positions(px, py);
            let rad = |s: (f64, f64)| (s.0 - cx).hypot(s.1 - cy);
            // The curves are functions of the source (mosaic) radius of green.
            let r_g = rad(sg);
            worst = worst
                .max(((rad(sr) - r_g) - curves.red.shift_at(r_g)).abs())
                .max(((rad(sb) - r_g) - curves.blue.shift_at(r_g)).abs());
        }
        println!("{name}: R9 radial displacement vs the curves, worst {worst:.2e} px");
        // Linear interpolation of 8 px knots of a smooth function: well below a thousandth of a pixel.
        assert!(worst < 2e-3, "{name}: {worst}");
        if name == "distortion off" {
            // The closed form r (K - 1) at the knots of the curve.
            for (r, s) in curves
                .red
                .radius_px()
                .iter()
                .zip(curves.red.shift_px())
                .skip(1)
                .step_by(37)
            {
                let k = (s / r) + 1.0;
                assert!(k > 1.0 - 1e-3 && k < 1.02, "K = {k} at {r}");
            }
        }
    }
}

#[test]
fn no_ca_no_curves_and_a_folded_distortion_table_is_refused() {
    let (fw, fh) = (600usize, 400usize);
    let ca = vec![0, 300, 900, 1800, 0, -300, -900, -1800];
    let distortion = vec![0, 60, 200, 450, 800];
    let only = |ca: Vec<i16>, distortion: Vec<i16>| LensTags {
        vignetting: vec![],
        ca,
        distortion,
    };
    // No 0x7035 table: none; distortion alone is not CA.
    let geo = LensGeometry::new(fw, fh, &only(vec![], distortion.clone()), LensSwitches::default())
        .expect("lens");
    assert_eq!(geo.ca_curves(), Ok(None));
    assert_eq!(LensGeometry::identity(fw, fh).ca_curves(), Ok(None));
    // A neutral table (every knot zero: R9 moves nothing) is no CA either; one knot is enough.
    let neutral = only(vec![0; 8], distortion.clone());
    let geo = LensGeometry::new(fw, fh, &neutral, LensSwitches::default()).expect("lens");
    assert_eq!(geo.ca_curves(), Ok(None));
    let nearly = only(vec![0, 0, 0, 1, 0, 0, 0, 0], distortion.clone());
    let geo = LensGeometry::new(fw, fh, &nearly, LensSwitches::default()).expect("lens");
    assert!(geo.ca_curves().expect("ok").is_some());
    // `decode.chromatic_aberration = false` takes the CA out of the geometry.
    let off = LensSwitches {
        lens_corrections: true,
        chromatic_aberration: false,
    };
    let geo = LensGeometry::new(fw, fh, &only(ca.clone(), distortion.clone()), off).expect("lens");
    assert_eq!(geo.ca_curves(), Ok(None));
    // `decode.lens_corrections = false` only drops the distortion: the CA curves stay, with F = 1.
    let no_dist = LensSwitches {
        lens_corrections: false,
        chromatic_aberration: true,
    };
    let geo = LensGeometry::new(fw, fh, &only(ca.clone(), distortion), no_dist).expect("lens");
    let c = geo.ca_curves().expect("ok").expect("curves");
    let hd = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    // Without distortion the knot at output radius j * 8 px is at the source radius j * 8 px.
    for (j, r) in c.red.radius_px().iter().enumerate().take(20) {
        assert!((r - 8.0 * j as f64).abs() < 1e-9 * hd, "knot {j} at {r}");
    }
    // A distortion table that makes the source radius fall (F drops faster than 1 / r grows).
    let folded = only(ca, vec![0, -9000, -16000, -16300, -16380, -16384]);
    let geo = LensGeometry::new(fw, fh, &folded, LensSwitches::default()).expect("lens");
    assert!(matches!(
        geo.ca_curves(),
        Err(CaCurveError::NotIncreasing { .. })
    ));
}

// ---------------------------------------------------------------------------------------------
// The path itself
// ---------------------------------------------------------------------------------------------

/// Two curves in a big frame: red and blue displaced in opposite directions, growing with the
/// radius, so that |shift_R - shift_B| runs from 0 at the centre to about `corner_px` at the
/// corner, with a bend (it is not linear) so that both the relaxation of pass 1 and every regime
/// of pass 2 occur across a tile.
fn strong_curves(fw: usize, fh: usize, corner_px: f64) -> CaCurves {
    let hd = 0.5 * ((fw * fw + fh * fh) as f64).sqrt();
    let knots = |sign: f64| -> RadialCurve {
        let r: Vec<f64> = (0..=24).map(|j| hd * f64::from(j) / 20.0).collect();
        let s: Vec<f64> = r
            .iter()
            .map(|&x| {
                let t = x / hd;
                sign * 0.5 * corner_px * (0.6 * t + 0.4 * t * t)
            })
            .collect();
        RadialCurve::new(r, s).expect("curve")
    };
    CaCurves::new(fw as f64 / 2.0, fh as f64 / 2.0, knots(1.0), knots(-1.0)).expect("curves")
}

/// A textured scene on the sub-rectangle of a big frame that starts at (`x0`, `y0`), with the
/// placement set.
fn placed_scene(w: usize, h: usize, cfa: Cfa, seed: u64, level: f32, frame: (usize, usize), at: (usize, usize)) -> Mosaic {
    let mut m = synth::mosaic_from_rgb(&synth::natural(w, h, seed), cfa);
    for v in &mut m.data {
        *v *= level;
    }
    m.place = Placement {
        x0: at.0,
        y0: at.1,
        frame_width: frame.0,
        frame_height: frame.1,
    };
    m
}

fn inputs<'a>(ca: Option<&'a CaCurves>, noise: NoiseDn) -> DemosaicInputs<'a> {
    DemosaicInputs {
        gains: [1.9, 1.0, 1.4],
        noise,
        ca,
    }
}

#[test]
fn banded_equals_whole_and_the_ca_path_changes_the_result() {
    // 300 rows: three bands of 128 rows with seams in the middle of the CA field.
    let frame = (900usize, 700usize);
    let curves = strong_curves(frame.0, frame.1, 3.0);
    for (cfa, noise) in [(Cfa::RGGB, low_noise()), (Cfa::BGGR, high_noise())] {
        let m = placed_scene(61, 300, cfa, 3, 0.4, frame, (700, 300));
        let params = Params::default();
        let with = inputs(Some(&curves), noise);
        let whole = adaptive_ratio::demosaic_whole(&m, &with, &params).expect("whole");
        let banded = adaptive_ratio::demosaic(&m, &with, &params).expect("banded");
        assert_eq!(bits(&whole), bits(&banded), "{}", cfa.name());
        assert!(whole.data.iter().all(|v| v.is_finite()));
        let plain = adaptive_ratio::demosaic_whole(&m, &inputs(None, noise), &params).expect("plain");
        let d = image_diff(&plain, &whole);
        println!("{}: CA path vs plain, max |diff| {d:.3e}", cfa.name());
        assert!(d > 1e-3, "{}: the CA path did nothing ({d})", cfa.name());
    }
}

#[test]
fn tiles_with_the_declared_halo_equal_the_whole_frame_with_ca() {
    // Odd frame size, non-RGGB phases, a placement deep inside a big frame; the limit is 1e-4, the
    // oracle's plain path measures 0 and so must this.
    let big = (2400usize, 1600usize);
    let (fw, fh) = (77usize, 61usize);
    let frame_rect = Rect::whole(fw, fh);
    let curves = strong_curves(big.0, big.1, 2.5);
    for (name, cfa, noise, level) in [
        ("plain GBRG", Cfa::GBRG, low_noise(), 0.4),
        ("shadow matching RGGB", Cfa::RGGB, high_noise(), 0.05),
        ("plain GRBG", Cfa::GRBG, low_noise(), 0.01),
    ] {
        // Near the corner of the big frame, where the misregistration is largest, and at the centre.
        for at in [(2250usize, 1450usize), (1160, 770)] {
            let m = placed_scene(fw, fh, cfa, 1234, level, big, at);
            let inp = inputs(Some(&curves), noise);
            let params = Params::default();
            let whole = adaptive_ratio::demosaic_whole(&m, &inp, &params).expect("whole");
            for halo in [adaptive_ratio::HALO_PLAIN_PX, Method::AdaptiveRatio.halo()] {
                if halo < adaptive_ratio::Derived::new(inp.gains, noise, &params).expect("d").halo_px
                {
                    continue;
                }
                let mut tiled = RgbImage::new(fw, fh);
                for core in tile_grid(fw, fh, 24, 20, 2) {
                    let ext = core.grow_aligned(Halo::uniform(halo), frame_rect, 2);
                    let res = adaptive_ratio::demosaic_whole(&m.extract(ext), &inp, &params)
                        .expect("tile");
                    tiled.paste(&res, core.relative_to(&ext), core.x, core.y);
                }
                let d = image_diff(&whole, &tiled);
                println!("tiling {name} at {at:?}: halo {halo}: max |diff| {d:.3e}");
                assert_eq!(d, 0.0, "{name} at {at:?} halo {halo}");
            }
        }
    }
}

#[test]
fn crops_at_every_phase_give_the_identical_interior_with_ca() {
    let big = (2400usize, 1600usize);
    let (fw, fh) = (64usize, 56usize);
    let curves = strong_curves(big.0, big.1, 2.5);
    for (name, noise, level) in [("plain", low_noise(), 0.4), ("matching", high_noise(), 0.05)] {
        let frame = placed_scene(fw, fh, Cfa::RGGB, 77, level, big, (1900, 1200));
        let inp = inputs(Some(&curves), noise);
        let params = Params::default();
        let whole = adaptive_ratio::demosaic_whole(&frame, &inp, &params).expect("whole");
        let halo = adaptive_ratio::Derived::new(inp.gains, noise, &params)
            .expect("derived")
            .halo_px;
        let mut compared = 0;
        for oy in 0..2 {
            for ox in 0..2 {
                // `extract` moves the placement with the crop, so the CA field stays put.
                let crop = frame.extract(Rect::new(ox, oy, fw - 3, fh - 3));
                let part = adaptive_ratio::demosaic_whole(&crop, &inp, &params).expect("crop");
                for c in 0..3 {
                    for y in halo..crop.height - halo {
                        for x in halo..crop.width - halo {
                            let a = part.plane(c)[y * crop.width + x];
                            let b = whole.plane(c)[(y + oy) * fw + x + ox];
                            assert_eq!(a.to_bits(), b.to_bits(), "{name} crop ({ox},{oy}) c{c} ({x},{y})");
                            compared += 1;
                        }
                    }
                }
            }
        }
        println!("equivariance with CA, {name}: {compared} samples identical");
    }
}

#[test]
fn the_placement_sets_the_position_of_the_ca_field() {
    let big = (2400usize, 1600usize);
    let (w, h) = (48usize, 48usize);
    let params = Params::default();
    // A misregistration that is the same everywhere does not depend on where the tile sits ...
    let flat = |c: f64| {
        let red = RadialCurve::from_knots(&[(0.0, c), (3000.0, c)]).expect("red");
        let blue = RadialCurve::from_knots(&[(0.0, 0.0), (3000.0, 0.0)]).expect("blue");
        CaCurves::new(1200.0, 800.0, red, blue).expect("curves")
    };
    for c in [0.05, 0.4, 1.7] {
        let curves = flat(c);
        let a = placed_scene(w, h, Cfa::RGGB, 5, 0.4, big, (100, 100));
        let mut b = a.clone();
        b.place.x0 = 2000;
        b.place.y0 = 1400;
        let inp = inputs(Some(&curves), low_noise());
        let ra = adaptive_ratio::demosaic_whole(&a, &inp, &params).expect("a");
        let rb = adaptive_ratio::demosaic_whole(&b, &inp, &params).expect("b");
        assert_eq!(bits(&ra), bits(&rb), "constant misregistration {c}");
    }
    // ... a misregistration that grows with the radius does: a tile at the centre has none, one at
    // the corner has a lot, and the results differ.
    let curves = strong_curves(big.0, big.1, 3.0);
    let centre = placed_scene(w, h, Cfa::RGGB, 5, 0.4, big, (1176, 776));
    let mut corner = centre.clone();
    corner.place.x0 = 2300;
    corner.place.y0 = 1500;
    let inp = inputs(Some(&curves), low_noise());
    let r_centre = adaptive_ratio::demosaic_whole(&centre, &inp, &params).expect("centre");
    let r_corner = adaptive_ratio::demosaic_whole(&corner, &inp, &params).expect("corner");
    let plain = adaptive_ratio::demosaic_whole(&centre, &inputs(None, low_noise()), &params)
        .expect("plain");
    let near = image_diff(&r_centre, &plain);
    let far = image_diff(&r_corner, &plain);
    println!("CA path vs plain: centre tile {near:.3e}, corner tile {far:.3e}");
    assert!(far > 5.0 * near, "centre {near}, corner {far}");
}

#[test]
fn a_centre_that_is_not_finite_is_refused() {
    let m = placed_scene(16, 16, Cfa::RGGB, 1, 0.4, (400, 300), (0, 0));
    let mut curves = strong_curves(400, 300, 1.0);
    for (x, y) in [(f64::NAN, 150.0), (200.0, f64::INFINITY)] {
        curves.centre_x = x;
        curves.centre_y = y;
        let inp = inputs(Some(&curves), low_noise());
        for run in [adaptive_ratio::demosaic, adaptive_ratio::demosaic_whole] {
            assert!(matches!(
                run(&m, &inp, &Params::default()),
                Err(Error::BadCurve { .. })
            ));
        }
        // The infallible entry point falls back to Menon, like for every refused input.
        let fallback = demosaic::demosaic_with(&m, Method::AdaptiveRatio, &inp);
        assert_eq!(bits(&fallback), bits(&demosaic::demosaic(&m, Method::Menon)));
    }
}

#[test]
fn demosaic_with_runs_the_ca_path_for_the_adaptive_ratio_method_only() {
    let big = (1800usize, 1200usize);
    let curves = strong_curves(big.0, big.1, 3.0);
    let m = placed_scene(70, 200, Cfa::RGGB, 21, 0.4, big, (1600, 900));
    let with = inputs(Some(&curves), low_noise());
    let without = inputs(None, low_noise());
    let a = demosaic::demosaic_with(&m, Method::AdaptiveRatio, &with);
    let b = demosaic::demosaic_whole_with(&m, Method::AdaptiveRatio, &with);
    assert_eq!(bits(&a), bits(&b), "banded and whole agree through demosaic_with");
    assert!(image_diff(&a, &demosaic::demosaic_with(&m, Method::AdaptiveRatio, &without)) > 1e-3);
    // Every other method ignores the curves.
    for method in [Method::Bilinear, Method::Menon, Method::Lmmse, Method::LogGuided] {
        assert_eq!(
            bits(&demosaic::demosaic_with(&m, method, &with)),
            bits(&demosaic::demosaic_with(&m, method, &without)),
            "{}",
            method.name()
        );
    }
}

#[test]
fn random_input_with_ca_gives_finite_output_for_every_size_and_phase() {
    let big = (1000usize, 800usize);
    let mut rng = Rng::new(404);
    let mut runs = 0;
    for (w, h) in [(2usize, 2usize), (2, 5), (3, 3), (5, 2), (7, 5), (13, 11), (30, 17)] {
        for cfa in Cfa::ALL {
            for cfg in 0..3 {
                let data: Vec<f32> = (0..w * h)
                    .map(|_| {
                        let u = rng.range(0.0, 1.0);
                        if u < 0.2 {
                            -0.05 * rng.range(0.0, 1.0)
                        } else if u < 0.3 {
                            4.0 * rng.range(0.0, 1.0)
                        } else {
                            rng.range(0.0, 1.0)
                        }
                    })
                    .collect();
                let mut m = Mosaic::from_data(w, h, cfa, data).expect("mosaic");
                m.place = Placement {
                    x0: (rng.range(0.0, 900.0)) as usize,
                    y0: (rng.range(0.0, 700.0)) as usize,
                    frame_width: big.0,
                    frame_height: big.1,
                };
                // Misregistration from nothing to far above the 1 px clamp of pass 2.
                let curves = strong_curves(big.0, big.1, [0.0, 1.0, 12.0][cfg]);
                let noise = if cfg == 1 { high_noise() } else { low_noise() };
                let params = Params::default();
                let inp = inputs(Some(&curves), noise);
                let out = adaptive_ratio::demosaic(&m, &inp, &params)
                    .unwrap_or_else(|e| panic!("{w}x{h} {} cfg {cfg}: {e}", cfa.name()));
                assert!(
                    out.data.iter().all(|v| v.is_finite()),
                    "{w}x{h} {} cfg {cfg}: not finite",
                    cfa.name()
                );
                let whole = adaptive_ratio::demosaic_whole(&m, &inp, &params).expect("whole");
                assert_eq!(bits(&out), bits(&whole), "{w}x{h} {}", cfa.name());
                runs += 1;
            }
        }
    }
    assert_eq!(runs, 7 * 4 * 3);
}

// ---------------------------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------------------------

/// CA strong enough to matter on a small frame: +-0.9 % of the radius at the corner of the knots,
/// about 1.4 px at the corner of a 200 x 140 frame.
fn strong_tags(ca: bool) -> LensTags {
    LensTags {
        vignetting: vec![],
        ca: if ca {
            vec![0, 5000, 14000, 30000, 0, -4000, -12000, -26000]
        } else {
            vec![]
        },
        distortion: vec![0, 30, 100, 220, 400, 650],
    }
}

fn levels() -> Levels {
    Levels {
        black: [512.0, 514.0, 513.0, 511.0],
        white: [16000.0; 4],
    }
}

fn raw_of(w: usize, h: usize, cfa: Cfa, seed: u64) -> Vec<u16> {
    let m = synth::mosaic_from_rgb(&synth::natural(w, h, seed), cfa);
    let mut rng = Rng::new(seed + 7);
    m.data
        .iter()
        .map(|&v| {
            let x = 512.0 + (v * 0.8 + 0.004 * rng.gaussian()) * 15488.0;
            x.clamp(0.0, 16000.0) as u16
        })
        .collect()
}

const G0: [f32; 3] = [2.0, 1.0, 1.6];

/// R1, R4, R6 and the lens resample only.
fn chain_with(method: Method, lens: Option<LensGeometry>, noise: NoiseDn) -> Chain {
    Chain {
        levels: levels(),
        vignette: None,
        gains: G0,
        highlight_recovery: false,
        method,
        denoise: None,
        noise_reduction: NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        },
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise: Some(noise),
        sharpen: None,
        lens,
    }
}

/// The chain by hand over the whole frame: R1, R4, `demosaic_whole_with` and the lens resample.
fn by_hand(
    raw: &RawFrame<'_>,
    method: Method,
    geo: &LensGeometry,
    ca: Option<&CaCurves>,
    noise: NoiseDn,
) -> RgbImage {
    let (mut m, _) = normalize::normalize(raw.data, raw.width, raw.height, raw.cfa, &levels(), None)
        .expect("normalize");
    gains::apply_gains(&mut m, G0);
    let img = demosaic::demosaic_whole_with(
        &m,
        method,
        &DemosaicInputs {
            gains: G0,
            noise,
            ca,
        },
    );
    let (ow, oh) = geo.output_size();
    resample::resample_lens(&img, geo, Rect::whole(ow, oh)).expect("resample")
}

fn run_whole(chain: &Chain, raw: &RawFrame<'_>) -> RgbImage {
    let prepared = chain.prepare(raw).expect("prepare");
    let (ow, oh) = chain.output_size(raw.width, raw.height);
    chain
        .run_tile(raw, &prepared, Rect::whole(ow, oh))
        .expect("run")
}

#[test]
fn the_chain_runs_the_ca_path_when_the_lens_has_ca_and_only_then() {
    let (w, h) = (200usize, 140usize);
    for cfa in [Cfa::RGGB, Cfa::GBRG] {
        let data = raw_of(w, h, cfa, 17);
        let raw = RawFrame {
            width: w,
            height: h,
            cfa,
            data: &data,
            qstep: None,
        };
        let noise = low_noise();
        let geo = |tags: &LensTags, switches: LensSwitches| {
            LensGeometry::new(w, h, tags, switches)
                .expect("lens")
                .with_auto_crop()
        };
        let on = geo(&strong_tags(true), LensSwitches::default());
        let curves = on.ca_curves().expect("ok").expect("this lens has CA");
        // The corner misregistration is more than a pixel, or the test would not mean much.
        let hd = (((w * w + h * h) as f64).sqrt()) / 2.0;
        assert!(curves.misregistration_px(hd) > 1.0, "{}", curves.misregistration_px(hd));

        // CA tag and switch on: the CA path.
        let chain = chain_with(Method::AdaptiveRatio, Some(on.clone()), noise);
        let got = run_whole(&chain, &raw);
        let want = by_hand(&raw, Method::AdaptiveRatio, &on, Some(&curves), noise);
        assert_eq!(bits(&got), bits(&want), "{}: the CA path", cfa.name());
        let plain = by_hand(&raw, Method::AdaptiveRatio, &on, None, noise);
        assert!(
            image_diff(&got, &plain) > 1e-3,
            "{}: the chain ran the plain path",
            cfa.name()
        );

        // A lens with distortion only: no CA, the plain path.
        let no_tag = geo(&strong_tags(false), LensSwitches::default());
        let got = run_whole(&chain_with(Method::AdaptiveRatio, Some(no_tag.clone()), noise), &raw);
        let want = by_hand(&raw, Method::AdaptiveRatio, &no_tag, None, noise);
        assert_eq!(bits(&got), bits(&want), "{}: no CA tag", cfa.name());

        // A neutral table (every knot zero) is no CA: R9 moves nothing, the demosaic stays plain.
        let neutral = geo(
            &LensTags {
                ca: vec![0; 8],
                ..strong_tags(false)
            },
            LensSwitches::default(),
        );
        let got = run_whole(&chain_with(Method::AdaptiveRatio, Some(neutral.clone()), noise), &raw);
        let want = by_hand(&raw, Method::AdaptiveRatio, &neutral, None, noise);
        assert_eq!(bits(&got), bits(&want), "{}: neutral CA table", cfa.name());

        // `decode.chromatic_aberration = false`: the geometry has no CA, so neither has the demosaic.
        let off = geo(
            &strong_tags(true),
            LensSwitches {
                lens_corrections: true,
                chromatic_aberration: false,
            },
        );
        let got = run_whole(&chain_with(Method::AdaptiveRatio, Some(off.clone()), noise), &raw);
        let want = by_hand(&raw, Method::AdaptiveRatio, &off, None, noise);
        assert_eq!(bits(&got), bits(&want), "{}: CA switched off", cfa.name());

        // No lens stage at all: the plain path (and nothing to resample).
        let got = run_whole(&chain_with(Method::AdaptiveRatio, None, noise), &raw);
        assert_eq!((got.width, got.height), (w, h));

        // Another method with the same CA lens ignores the curves.
        let chain = chain_with(Method::Menon, Some(on.clone()), noise);
        let got = run_whole(&chain, &raw);
        let want = by_hand(&raw, Method::Menon, &on, None, noise);
        assert_eq!(bits(&got), bits(&want), "{}: Menon", cfa.name());
    }
}

#[test]
fn tiles_of_the_chain_equal_the_whole_frame_with_the_ca_path() {
    // The curves belong to the frame: every tile runs the same ones, at its own placement.
    let (w, h) = (200usize, 140usize);
    for cfa in Cfa::ALL {
        let data = raw_of(w, h, cfa, 29);
        let raw = RawFrame {
            width: w,
            height: h,
            cfa,
            data: &data,
            qstep: None,
        };
        let geo = LensGeometry::new(w, h, &strong_tags(true), LensSwitches::default())
            .expect("lens")
            .with_auto_crop();
        for noise in [low_noise(), high_noise()] {
            let chain = chain_with(Method::AdaptiveRatio, Some(geo.clone()), noise);
            let prepared = chain.prepare(&raw).expect("prepare");
            let (ow, oh) = chain.output_size(w, h);
            let whole = chain
                .run_tile(&raw, &prepared, Rect::whole(ow, oh))
                .expect("whole");
            let mut tiled = RgbImage::new(ow, oh);
            for core in tile_grid(ow, oh, 61, 47, 1) {
                let t = chain.run_tile(&raw, &prepared, core).expect("tile");
                tiled.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
            }
            let d = image_diff(&whole, &tiled);
            println!("chain tiles with CA, {}: max |diff| {d:.3e}", cfa.name());
            assert!(d <= 1e-4, "{}: {d}", cfa.name());
        }
    }
}
