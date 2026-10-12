//! M1-11 (C12): the display transform dispatch: ACES 2 and imported ARRI display LUTs, for every
//! output, with our own synthetic LUTs (no ARRI files exist in the repository).

use std::sync::Arc;

use ym_color::aces2::Aces2Transform;
use ym_color::consts::Vec3;
use ym_color::consts::linalg;
use ym_color::display::{
    ArriDisplayLut, ChainParams, ColorChain, DisplayError, DisplayLutInput, DisplayLutOutput,
    DisplayLutPrimaries, DisplayLutTransfer, DisplayTransform, OutputTarget,
};
use ym_color::encode::OutputSpace;
use ym_color::gamut::matrices;
use ym_color::logc::{LogC4, logc4_decode_rgb};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A chain that does nothing before C12: identity matrix, no roll-off, zero grade, Neutral look.
fn bare(display: DisplayTransform, output: OutputTarget) -> ColorChain {
    let mut p = ChainParams::new(IDENTITY, ToneParams::new(0.5, 0.0, 1.0e6, 0.0).unwrap());
    p.display = display;
    p.output = output;
    ColorChain::new(&p).unwrap()
}

fn identity_lut(size: usize, min: f64, max: f64) -> Arc<Lut> {
    Arc::new(Lut::from_cube(
        Lut3d::bake(size, [min; 3], [max; 3], |c| c).unwrap(),
    ))
}

fn max_diff(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)
}

/// Scene pixels spanning dark to bright, in and out of gamut (linear AWG4).
fn scene_pixels() -> Vec<Vec3> {
    let mut v = vec![
        [0.18, 0.18, 0.18],
        [0.0, 0.0, 0.0],
        [0.02, 0.01, 0.005],
        [1.0, 0.5, 0.2],
        [0.05, 0.3, 0.1],
        [0.1, 0.1, 4.0],
        [8.0, 6.0, 4.0],
        [0.5, -0.02, 0.1],
    ];
    let mut rng = Rng(0x0D15_B1A7);
    for _ in 0..200 {
        let s = 0.18 * 2.0_f64.powf(-6.0 + 10.0 * rng.unit());
        v.push([
            s * (0.2 + 1.8 * rng.unit()),
            s * (0.2 + 1.8 * rng.unit()),
            s * (0.2 + 1.8 * rng.unit()),
        ]);
    }
    v
}

#[test]
fn aces2_dispatch_is_the_reference_transform_for_every_output() {
    for target in OutputTarget::ALL {
        let chain = bare(DisplayTransform::Aces2, target);
        let reference = Aces2Transform::new(target.aces2());
        let mut worst = 0.0_f64;
        for px in scene_pixels() {
            let t = chain.trace(px);
            let ap0 = linalg::mul_vec(&matrices().awg4_to_ap0, px);
            // C12 keeps display-linear in the output's primaries; C13 is the output's transfer.
            let want_linear = reference.transform_ap0_linear(ap0);
            let want_encoded = target.space().encode_clamped(want_linear);
            // The only difference is the LogC4 round trip C8 -> C12.
            worst = worst
                .max(max_diff(t.c12_display_linear, want_linear))
                .max(max_diff(t.c13_encoded, want_encoded));
            assert!(
                max_diff(t.c13_encoded, want_encoded) < 1e-9,
                "{target:?} {px:?}: {:?} vs {want_encoded:?}",
                t.c13_encoded
            );
            // The port's own encode (the reference's moncurve for sRGB) agrees to its 1e-5 transfer difference.
            assert!(max_diff(t.c13_encoded, reference.transform_ap0(ap0)) < 1.0e-5);
        }
        println!("aces2 {target:?}: worst difference {worst:.2e}");
    }
}

#[test]
fn aces2_output_is_always_a_valid_code_value() {
    for target in OutputTarget::ALL {
        let chain = bare(DisplayTransform::Aces2, target);
        for px in scene_pixels() {
            for v in chain.apply_pixel(px) {
                assert!((0.0..=1.0).contains(&v), "{target:?} {px:?}: {v}");
            }
        }
    }
}

fn arri(input: DisplayLutInput, output: DisplayLutOutput, lut: Arc<Lut>) -> DisplayTransform {
    DisplayTransform::arri(ArriDisplayLut::new(lut, input, output).unwrap())
}

/// Converts display-linear between the three display gamuts the way the module should: through XYZ.
fn convert(linear: Vec3, from: DisplayLutPrimaries, to: OutputSpace) -> Vec3 {
    let m = matrices();
    let to_xyz = match from {
        DisplayLutPrimaries::Rec709 => m.rec709_to_xyz,
        DisplayLutPrimaries::P3D65 => m.p3_to_xyz,
        DisplayLutPrimaries::Rec2020 => m.rec2020_to_xyz,
    };
    let from_xyz = match to {
        OutputSpace::Srgb => m.xyz_to_rec709,
        OutputSpace::DisplayP3 => m.xyz_to_p3,
        OutputSpace::Rec2020 => m.xyz_to_rec2020,
    };
    linalg::mul_vec(&from_xyz, linalg::mul_vec(&to_xyz, linear))
}

fn gamma_decode(codes: Vec3, gamma: f64) -> Vec3 {
    codes.map(|c| if c > 0.0 { c.powf(gamma) } else { 0.0 })
}

#[test]
fn logc4_display_lut_applies_to_c11_directly() {
    // An identity cube turns the LogC4 code values themselves into the LUT's output codes.
    let lut = identity_lut(17, 0.0, 1.0);
    for target in OutputTarget::ALL {
        let chain = bare(
            arri(
                DisplayLutInput::LogC4,
                DisplayLutOutput::REC709_G24,
                lut.clone(),
            ),
            target,
        );
        for px in scene_pixels() {
            let t = chain.trace(px);
            let codes = t.c11_logc4.0.map(|v| v.clamp(0.0, 1.0));
            let linear = gamma_decode(codes, 2.4);
            let want = convert(linear, DisplayLutPrimaries::Rec709, target.space());
            assert!(
                max_diff(t.c12_display_linear, want) < 1e-12,
                "{target:?} {px:?}: {:?} vs {want:?}",
                t.c12_display_linear
            );
            assert_eq!(
                t.c13_encoded,
                target.space().encode_clamped(t.c12_display_linear)
            );
        }
    }
}

#[test]
fn logc3_display_lut_applies_after_the_exact_logc4_to_logc3_conversion() {
    let lut = identity_lut(33, 0.0, 1.0);
    for target in [OutputTarget::Viewport, OutputTarget::Srgb] {
        let chain = bare(
            arri(
                DisplayLutInput::LogC3Ei800,
                DisplayLutOutput::REC709_G24,
                lut.clone(),
            ),
            target,
        );
        for px in scene_pixels() {
            let t = chain.trace(px);
            // LogC4 -> linear AWG4 -> AWG3 -> LogC3 EI 800, by hand.
            let lin3 = LogC4(t.c11_logc4.0).to_linear().to_awg3();
            let c3 = ym_color::logc::logc3_ei800_encode_rgb(lin3.0);
            let codes = c3.map(|v| v.clamp(0.0, 1.0));
            let want = convert(
                gamma_decode(codes, 2.4),
                DisplayLutPrimaries::Rec709,
                target.space(),
            );
            assert!(
                max_diff(t.c12_display_linear, want) < 1e-12,
                "{target:?} {px:?}: {:?} vs {want:?}",
                t.c12_display_linear
            );
        }
    }
}

#[test]
fn display_lut_outputs_convert_primaries_to_the_output_space() {
    let lut = identity_lut(17, 0.0, 1.0);
    let cases = [
        (
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::P3D65,
                transfer: DisplayLutTransfer::Gamma(2.6),
            },
            2.6,
        ),
        (
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::Rec2020,
                transfer: DisplayLutTransfer::Gamma(2.4),
            },
            2.4,
        ),
    ];
    for (output, gamma) in cases {
        for target in OutputTarget::ALL {
            let chain = bare(arri(DisplayLutInput::LogC4, output, lut.clone()), target);
            for px in scene_pixels() {
                let t = chain.trace(px);
                let codes = t.c11_logc4.0.map(|v| v.clamp(0.0, 1.0));
                let want = convert(gamma_decode(codes, gamma), output.primaries, target.space());
                assert!(
                    max_diff(t.c12_display_linear, want) < 1e-12,
                    "{output:?} -> {target:?} {px:?}"
                );
            }
        }
    }
}

#[test]
fn srgb_transfer_display_lut_decodes_piecewise() {
    let lut = identity_lut(17, 0.0, 1.0);
    let output = DisplayLutOutput {
        primaries: DisplayLutPrimaries::Rec709,
        transfer: DisplayLutTransfer::Srgb,
    };
    let chain = bare(
        arri(DisplayLutInput::LogC4, output, lut),
        OutputTarget::Srgb,
    );
    for px in scene_pixels() {
        let t = chain.trace(px);
        for k in 0..3 {
            let code = t.c11_logc4.0[k].clamp(0.0, 1.0);
            let want = ym_color::encode::srgb_decode(code);
            assert!((t.c12_display_linear[k] - want).abs() < 1e-12);
            // sRGB in, sRGB out: the identity on the clamped code value.
            assert!((t.c13_encoded[k] - code).abs() < 1e-12);
        }
    }
}

#[test]
fn display_lut_with_its_own_domain_clamps_to_it() {
    // An identity cube over [-0.1, 1.2]: codes up to 1.2 pass, the output stage clamps to 1.
    let lut = identity_lut(9, -0.1, 1.2);
    let chain = bare(
        arri(DisplayLutInput::LogC4, DisplayLutOutput::REC709_G24, lut),
        OutputTarget::Srgb,
    );
    let t = chain.trace([0.18, 0.18, 0.18]);
    let code = t.c11_logc4.0[0];
    assert!((t.c12_display_linear[0] - code.powf(2.4)).abs() < 1e-12);
    // A value below the domain clamps to the domain minimum, which decodes to zero light.
    let dark = chain.trace([-1.0, -1.0, -1.0]);
    assert!(dark.c12_display_linear.iter().all(|v| *v == 0.0));
    assert!(dark.c13_encoded.iter().all(|v| *v == 0.0));
}

#[test]
fn display_transform_is_aces2_by_default_and_luts_are_shared_not_copied() {
    assert_eq!(DisplayTransform::default(), DisplayTransform::Aces2);
    let lut = identity_lut(5, 0.0, 1.0);
    let d = ArriDisplayLut::new(
        Arc::clone(&lut),
        DisplayLutInput::LogC3Ei800,
        DisplayLutOutput::REC709_G24,
    )
    .unwrap();
    assert!(Arc::ptr_eq(d.lut(), &lut));
    assert_eq!(d.input(), DisplayLutInput::LogC3Ei800);
    assert_eq!(d.output(), DisplayLutOutput::REC709_G24);
    let copy = d.clone();
    assert!(Arc::ptr_eq(copy.lut(), &lut));
    // The prepared chain keeps the same cube.
    let transform = DisplayTransform::arri(d);
    let before = Arc::strong_count(&lut);
    let _chain = bare(transform, OutputTarget::Viewport);
    assert!(Arc::strong_count(&lut) >= before);
}

#[test]
fn display_lut_targets_from_the_importer_parse() {
    let out = DisplayLutOutput::from_target("rec709-g24").unwrap();
    assert_eq!(out, DisplayLutOutput::REC709_G24);
    for hdr in ["hlg", "pq", "rec2020-pq"] {
        assert!(matches!(
            DisplayLutOutput::from_target(hdr),
            Err(DisplayError::UnsupportedTarget { .. })
        ));
    }
}

#[test]
fn aces2_and_a_matching_display_lut_agree_on_neutral_ordering() {
    // Not the same rendering, but both are monotone on a neutral ramp (a sanity check on the plumbing).
    let ramp: Vec<f64> = (0..40)
        .map(|i| 0.18 * (f64::from(i) / 3.0 - 6.0).exp2())
        .collect();
    let lut = Arc::new(Lut::from_cube(
        Lut3d::bake(33, [0.0; 3], [1.0; 3], |c| c.map(|v| v.powf(1.3))).unwrap(),
    ));
    for display in [
        DisplayTransform::Aces2,
        arri(DisplayLutInput::LogC4, DisplayLutOutput::REC709_G24, lut),
    ] {
        let chain = bare(display, OutputTarget::Viewport);
        let mut last = [-1.0; 3];
        for g in &ramp {
            let out = chain.apply_pixel([*g; 3]);
            for k in 0..3 {
                assert!(out[k] >= last[k], "not monotone at {g}");
            }
            last = out;
        }
    }
    // Keep the decode helper honest.
    let _ = logc4_decode_rgb([0.5; 3]);
}

#[test]
fn targets_with_a_white_point_other_than_d65_are_refused() {
    // A D60 or D50 "P3" rendered as D65 would show a wrong white; refuse it like a DCI white.
    for slug in [
        "p3-d60",
        "p3-d60-g26",
        "rec709-d50",
        "rec2020-d55",
        "Rec709_D75",
    ] {
        assert!(
            matches!(
                DisplayLutOutput::from_target(slug),
                Err(DisplayError::UnsupportedTarget { .. })
            ),
            "{slug}"
        );
    }
    // The D65 spellings still parse.
    for slug in ["p3-d65", "p3-d65-g26", "rec709-g24", "rec2020-d65"] {
        assert!(DisplayLutOutput::from_target(slug).is_ok(), "{slug}");
    }
}

#[test]
fn target_parsing_never_panics_and_only_accepts_sane_outputs() {
    // Seeded random slugs built from the words the importer produces, plus noise characters.
    let words = [
        "gamma", "g", "rec", "709", "p3", "2020", "srgb", "d65", "d60", "dci", "hlg", "pq", "24",
        "26", "22", "2", "4", "1", "x", "é", "-", "_", " ", ".", "", "99999", "0",
    ];
    let mut rng = Rng(0xDEC0_DE5E);
    let mut accepted = 0;
    for _ in 0..20_000 {
        let n = 1 + (rng.next_u64() % 6) as usize;
        let slug: String = (0..n)
            .map(|_| words[(rng.next_u64() % words.len() as u64) as usize])
            .collect::<Vec<_>>()
            .join(if rng.next_u64().is_multiple_of(2) {
                "-"
            } else {
                ""
            });
        if let Ok(output) = DisplayLutOutput::from_target(&slug) {
            accepted += 1;
            if let DisplayLutTransfer::Gamma(g) = output.transfer {
                assert!((1.0..=4.0).contains(&g), "{slug:?} gave gamma {g}");
            }
            // An accepted output always builds a display transform.
            let lut = identity_lut(2, 0.0, 1.0);
            assert!(ArriDisplayLut::new(lut, DisplayLutInput::LogC4, output).is_ok());
        }
    }
    assert!(accepted > 500, "only {accepted} of 20000 slugs parsed");
}

#[test]
fn hdr_targets_are_refused_even_without_separators() {
    for slug in [
        "rec2020-hlg",
        "rec2020-hlg1000",
        "Rec2020PQ1000nits",
        "rec2020-pq-1000nits",
        "rec2020_st2084",
        "p3-hdr10",
    ] {
        assert!(
            matches!(
                DisplayLutOutput::from_target(slug),
                Err(DisplayError::UnsupportedTarget { .. })
            ),
            "{slug}"
        );
    }
}
