//! M1 (C7 to C13): the CPU reference chain `ym_color::display::ColorChain`.
//!
//! * stage order: every stage changed alone moves only itself and what follows it;
//! * the Auto-everything identity facts (Neutral look, identity curve, zero grade);
//! * log-to-log looks (LogC4, LogC3 with an identity LUT) and the Rec.709-baked look blend;
//! * the DNG stop point and the Before view;
//! * encode and dither determinism, thread counts, layouts and tiling;
//! * a neutral ramp through ACES 2 (monotone, neutral stays neutral).
//!
//! Display LUT (C12) specifics live in `display_transform.rs`. All LUTs here are our own synthetic
//! cubes. Pixel values are fixed or seeded; nothing depends on the machine.

use std::sync::Arc;

use ym_color::aces2::{Aces2Output, Aces2Transform};
use ym_color::cdl::Cdl;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::{self, Mat3};
use ym_color::curve::{Curve, CurveSet};
use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::display::{
    BeforePoint, ChainParams, ColorChain, DisplayError, Frame, Layout, LookSetting, OutputTarget,
    View, aces2_input_matrix,
};
use ym_color::encode::{self, OutputSpace};
use ym_color::gamut::{LinearAwg4, matrices};
use ym_color::grade::{ColorBalance, Grade, GradeParams, grade_luma};
use ym_color::logc::{LogC4, logc4_decode_rgb, logc4_encode_rgb};
use ym_color::look::{
    LookInput, LookTransform, blend_display_linear, blend_logc4, rec709_gamma24_decode,
};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

// ---- helpers -----------------------------------------------------------------------------------

/// SplitMix64: a tiny deterministic generator.
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

/// A white-preserving matrix standing in for the fused C1 to C6 (rows sum to 1), with +0.25 EV.
fn fused_matrix() -> Mat3 {
    let gain = 0.25_f64.exp2();
    let m = [
        [0.92, 0.10, -0.02],
        [0.04, 0.97, -0.01],
        [-0.03, 0.08, 0.95],
    ];
    m.map(|row| row.map(|v| v * gain))
}

fn tone() -> ToneParams {
    ToneParams::new(0.5, 0.5, 16.0, 0.002).unwrap()
}

/// Representative camera pixels: greys over many stops, saturated colours, near black, highlights
/// into the roll-off, and a slightly negative sample.
fn pixels() -> Vec<Vec3> {
    let mut v = vec![
        [0.18, 0.18, 0.18],
        [0.5, 0.3, 0.2],
        [0.02, 0.03, 0.08],
        [0.9, 0.8, 0.1],
        [0.001, 0.001, 0.001],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.5, 1.0],
        [12.0, 9.0, 6.0],
        [0.3, 0.05, 0.4],
        [0.05, 0.4, 0.1],
        [-0.004, 0.01, 0.02],
    ];
    let mut rng = Rng(0x00C0_FFEE);
    for _ in 0..40 {
        let scale = 0.18 * 2.0_f64.powf(-7.0 + 11.0 * rng.unit());
        v.push([
            scale * (0.3 + 1.4 * rng.unit()),
            scale * (0.3 + 1.4 * rng.unit()),
            scale * (0.3 + 1.4 * rng.unit()),
        ]);
    }
    v
}

fn max_diff(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)
}

fn assert_close(a: Vec3, b: Vec3, tol: f64, what: &str) {
    assert!(max_diff(a, b) <= tol, "{what}: {a:?} vs {b:?} (tol {tol})");
}

fn cube_lut(size: usize, f: impl FnMut(Vec3) -> Vec3) -> Arc<Lut> {
    Arc::new(Lut::from_cube(
        Lut3d::bake(size, [0.0; 3], [1.0; 3], f).unwrap(),
    ))
}

/// A smooth, non-trivial per-pixel function standing in for a creative LUT (our own).
fn tweak(c: Vec3) -> Vec3 {
    [
        (c[0] * 1.05 + 0.02).min(1.0) * 0.98 + 0.01 * c[1],
        c[1] * 0.97 + 0.03 * c[2] * c[2],
        (c[2] + 0.04 * (c[0] - c[1])).clamp(0.0, 1.0),
    ]
}

fn logc4_look() -> Arc<LookTransform> {
    let cdl = Cdl::new([1.1, 1.0, 0.9], [0.02, 0.0, -0.02], [1.0, 1.1, 1.0], 1.1).unwrap();
    Arc::new(
        LookTransform::new(
            LookInput::LogC4,
            Some(cdl),
            Some((*cube_lut(17, tweak)).clone()),
        )
        .unwrap(),
    )
}

fn other_logc4_look() -> Arc<LookTransform> {
    let cdl = Cdl::new([0.9, 1.0, 1.1], [-0.02, 0.01, 0.0], [1.2, 1.0, 1.0], 0.8).unwrap();
    Arc::new(LookTransform::new(LookInput::LogC4, Some(cdl), None).unwrap())
}

fn grade_params() -> GradeParams {
    GradeParams {
        contrast: 20.0,
        brightness: 10.0,
        shadows: 15.0,
        highlights: -20.0,
        black: 5.0,
        white: -5.0,
        balance_shadows: ColorBalance { a: 0.2, b: -0.1 },
        balance_midtones: ColorBalance { a: 0.0, b: 0.1 },
        balance_highlights: ColorBalance { a: -0.1, b: 0.0 },
        saturation: 12.0,
    }
}

fn curves() -> CurveSet {
    CurveSet {
        luma: Curve::new(&[(0.0, 0.0), (0.3, 0.26), (0.7, 0.78), (1.0, 1.0)]).unwrap(),
        r: Curve::new(&[(0.0, 0.02), (0.5, 0.52), (1.0, 1.0)]).unwrap(),
        g: Curve::identity(),
        b: Curve::new(&[(0.0, 0.0), (0.4, 0.36), (1.0, 0.97)]).unwrap(),
    }
}

/// Everything switched on: grade, a LogC4 look at 80 %, curves, ACES 2, viewport.
fn rich_params() -> ChainParams {
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.grade = grade_params();
    p.look = Some(LookSetting::new(logc4_look(), 80.0));
    p.curves = curves();
    p
}

fn chain(p: &ChainParams) -> ColorChain {
    ColorChain::new(p).unwrap()
}

/// Independent composition of C12 and C13 for an ACES 2 output, from the finished modules:
/// the port's display-linear output, then C13's transfer from `encode` (IEC 61966-2-1 sRGB; the
/// port's own `transform_ap0` uses the reference's continuous "moncurve" instead, which differs by
/// up to 9.8e-6, see `c13_srgb_differs_from_the_aces2_references_moncurve_by_under_1e5`).
fn aces2_of_linear_awg4(linear: Vec3, output: Aces2Output) -> Vec3 {
    let ap0 = linalg::mul_vec(&aces2_input_matrix(), linear);
    let display_linear = Aces2Transform::new(output).transform_ap0_linear(ap0);
    OutputTarget::from(output)
        .space()
        .encode_clamped(display_linear)
}

// ---- stage order -------------------------------------------------------------------------------

#[test]
fn trace_equals_an_independent_composition_of_the_stage_modules() {
    let p = rich_params();
    let c = chain(&p);
    let grade = Grade::new(&p.grade);
    let baked = p.curves.bake();
    let look = logc4_look();
    let aces = Aces2Transform::new(Aces2Output::Viewport);
    for px in pixels() {
        let t = c.trace(px);
        let c6 = linalg::mul_vec(&p.matrix, px);
        let c7 = p.tone.apply_rgb(c6);
        let lin = LinearAwg4(c7);
        let c8 = lin.to_logc4();
        let c9 = grade.apply(c8, grade_luma(lin));
        let c10 = blend_logc4(c9, look.eval_logc4(c9).unwrap(), 80.0);
        let c11 = baked.apply(c10);
        let ap0 = linalg::mul_vec(&aces2_input_matrix(), logc4_decode_rgb(c11.0));
        let c12 = aces.transform_ap0_linear(ap0);
        let c13 = OutputSpace::DisplayP3.encode_clamped(c12);
        assert_close(t.c6_linear_awg4, c6, 1e-14, "c6");
        assert_close(t.c7_linear_awg4, c7, 1e-14, "c7");
        assert_close(t.c8_logc4.0, c8.0, 1e-14, "c8");
        assert_close(t.c9_logc4.0, c9.0, 1e-14, "c9");
        assert_close(t.c10_logc4.0, c10.0, 1e-14, "c10");
        assert_close(t.c11_logc4.0, c11.0, 1e-14, "c11");
        assert_close(t.c12_display_linear, c12, 1e-14, "c12");
        assert_close(t.c13_encoded, c13, 1e-14, "c13");
        assert_eq!(c.apply_pixel(px), t.c13_encoded);
        assert_eq!(c.apply_to_display_linear(px), t.c12_display_linear);
        assert_eq!(c.apply_logc4_pixel(px), t.c11_logc4);
    }
}

/// Which of the trace's stages are bit-identical between two chains, as eight flags (C6 to C13).
fn same_stages(a: &ColorChain, b: &ColorChain, px: Vec3) -> [bool; 8] {
    let (x, y) = (a.trace(px), b.trace(px));
    [
        x.c6_linear_awg4 == y.c6_linear_awg4,
        x.c7_linear_awg4 == y.c7_linear_awg4,
        x.c8_logc4 == y.c8_logc4,
        x.c9_logc4 == y.c9_logc4,
        x.c10_logc4 == y.c10_logc4,
        x.c11_logc4 == y.c11_logc4,
        x.c12_display_linear == y.c12_display_linear,
        x.c13_encoded == y.c13_encoded,
    ]
}

/// `first_changed` is the index (C6 = 0 ... C13 = 7) of the first stage whose value must differ;
/// everything before it must be bit-identical, everything from it on must differ.
fn check_toggle(name: &str, change: impl FnOnce(&mut ChainParams), first_changed: usize) {
    let base = rich_params();
    let mut other = base.clone();
    change(&mut other);
    let (a, b) = (chain(&base), chain(&other));
    // A mid-tone, moderately saturated pixel, and a bright one for the roll-off.
    for px in [[0.4, 0.25, 0.15], [9.0, 7.0, 5.0], [0.06, 0.05, 0.09]] {
        let same = same_stages(&a, &b, px);
        for (i, same) in same.iter().enumerate() {
            if i < first_changed {
                assert!(same, "{name}: stage {i} should be unchanged for {px:?}");
            }
        }
        assert!(
            !same[first_changed],
            "{name}: stage {first_changed} should change for {px:?}"
        );
    }
}

#[test]
fn each_stage_changed_alone_moves_only_itself_and_what_follows() {
    // Index: C6 = 0, C7 = 1, C8 = 2, C9 = 3, C10 = 4, C11 = 5, C12 = 6, C13 = 7.
    check_toggle("matrix (C1-C6)", |p| p.matrix[0][0] *= 1.1, 0);
    check_toggle(
        "tone (C7)",
        |p| p.tone = ToneParams::new(1.0, 0.9, 16.0, 0.01).unwrap(),
        1,
    );
    check_toggle("grade (C9)", |p| p.grade.contrast = -30.0, 3);
    check_toggle(
        "look (C10)",
        |p| p.look = Some(LookSetting::new(other_logc4_look(), 80.0)),
        4,
    );
    check_toggle(
        "look intensity (C10)",
        |p| p.look = Some(LookSetting::new(logc4_look(), 30.0)),
        4,
    );
    check_toggle(
        "curve (C11)",
        |p| {
            p.curves.luma = Curve::new(&[(0.0, 0.0), (0.5, 0.45), (1.0, 1.0)]).unwrap();
        },
        5,
    );
    check_toggle(
        "output primaries (C12)",
        |p| p.output = OutputTarget::Srgb,
        6,
    );
}

#[test]
fn display_transform_change_leaves_every_logc4_stage_alone() {
    use ym_color::display::{ArriDisplayLut, DisplayLutInput, DisplayLutOutput, DisplayTransform};
    let lut = ArriDisplayLut::new(
        cube_lut(17, tweak),
        DisplayLutInput::LogC4,
        DisplayLutOutput::REC709_G24,
    )
    .unwrap();
    check_toggle(
        "display transform (C12)",
        |p| p.display = DisplayTransform::arri(lut),
        6,
    );
}

#[test]
fn dither_changes_only_the_quantised_bytes() {
    let mut off = rich_params();
    off.dither = false;
    let (on, off) = (chain(&rich_params()), chain(&off));
    let mut differs = false;
    for px in pixels() {
        assert_eq!(on.trace(px), off.trace(px));
        assert_eq!(on.apply_pixel(px), off.apply_pixel(px));
        let lin = on.apply_to_display_linear(px);
        let (a, b) = (on.encode_u8(lin, 5, 9), off.encode_u8(lin, 5, 9));
        differs |= a != b;
        for k in 0..3 {
            assert!(i32::from(a[k]).abs_diff(i32::from(b[k])) <= 1);
        }
    }
    assert!(
        differs,
        "dither never changed a byte over {} pixels",
        pixels().len()
    );
}

#[test]
fn the_curve_runs_before_a_look_that_includes_display_rendering() {
    // C9 -> C11 -> look: the look must see the curved value, never the other way round.
    let look = rec709_baked_look();
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.grade = grade_params();
    p.curves = curves();
    p.look = Some(LookSetting::new(Arc::clone(&look), 100.0));
    let c = chain(&p);
    let mut no_look = p.clone();
    no_look.look = None;
    let reference = chain(&no_look);
    let mut moved = 0;
    for px in pixels() {
        let t = c.trace(px);
        let r = reference.trace(px);
        // The look has not run at C10; C11 is the curve of C9.
        assert_eq!(t.c10_logc4, t.c9_logc4);
        assert_eq!(t.c11_logc4, r.c11_logc4);
        let rec709 = look.eval_display_rec709(t.c11_logc4).unwrap();
        let want = OutputSpace::DisplayP3.from_linear_rec709(rec709);
        assert_eq!(t.c12_display_linear, want);
        // The opposite order gives a different picture.
        let swapped = look
            .eval_display_rec709(LogC4(t.c9_logc4.0))
            .map(|v| OutputSpace::DisplayP3.from_linear_rec709(v))
            .unwrap();
        if max_diff(swapped, want) > 1e-4 {
            moved += 1;
        }
    }
    assert!(
        moved > 20,
        "the curve barely changed the look input ({moved})"
    );
}

// ---- identity facts ----------------------------------------------------------------------------

#[test]
fn auto_everything_is_c12_of_c8_of_c7_of_m_x_for_every_output() {
    for target in OutputTarget::ALL {
        let mut p = ChainParams::new(fused_matrix(), tone());
        p.output = target;
        let c = chain(&p);
        for px in pixels() {
            let c7 = tone().apply_rgb(linalg::mul_vec(&fused_matrix(), px));
            let linear_again = logc4_decode_rgb(logc4_encode_rgb(c7));
            let want = aces2_of_linear_awg4(linear_again, target.aces2());
            assert_close(c.apply_pixel(px), want, 1e-9, &format!("{target:?} {px:?}"));
            // The same through Aces2Transform on the exact linear value: only the log round trip differs.
            let direct = aces2_of_linear_awg4(c7, target.aces2());
            assert_close(c.apply_pixel(px), direct, 1e-8, "direct");
        }
    }
}

#[test]
fn neutral_look_identity_look_and_zero_intensity_change_nothing() {
    let base = chain(&ChainParams::new(fused_matrix(), tone()));
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.look = Some(LookSetting::new(Arc::new(LookTransform::neutral()), 100.0));
    let neutral = chain(&p);
    // A real look at 0 % is skipped entirely: bit-exact.
    p.look = Some(LookSetting::new(logc4_look(), 0.0));
    let zero = chain(&p);
    // An identity cube changes nothing visible (tetrahedral on a linear table is exact).
    let id =
        LookTransform::new(LookInput::LogC4, None, Some((*cube_lut(9, |c| c)).clone())).unwrap();
    p.look = Some(LookSetting::new(Arc::new(id), 100.0));
    let identity_cube = chain(&p);
    for px in pixels() {
        assert_eq!(neutral.trace(px), base.trace(px));
        assert_eq!(zero.trace(px), base.trace(px));
        assert_close(
            identity_cube.apply_pixel(px),
            base.apply_pixel(px),
            1e-9,
            "identity cube",
        );
    }
}

#[test]
fn zero_grade_and_identity_curves_are_exact_pass_throughs() {
    let c = chain(&ChainParams::new(fused_matrix(), tone()));
    for px in pixels() {
        let t = c.trace(px);
        assert_eq!(t.c8_logc4, t.c9_logc4);
        assert_eq!(t.c9_logc4, t.c10_logc4);
        assert_eq!(t.c10_logc4, t.c11_logc4);
    }
}

#[test]
fn logc3_look_with_an_identity_lut_round_trips() {
    let id = Arc::new(
        LookTransform::new(
            LookInput::LogC3Ei800,
            None,
            Some((*cube_lut(33, |c| c)).clone()),
        )
        .unwrap(),
    );
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.grade = GradeParams {
        contrast: 10.0,
        ..GradeParams::default()
    };
    let neutral = chain(&p);
    for intensity in [100.0, 55.0] {
        p.look = Some(LookSetting::new(Arc::clone(&id), intensity));
        let with_look = chain(&p);
        let mut checked = 0;
        for px in pixels() {
            let t = neutral.trace(px);
            // In-gamut, in-domain pixels: LogC3 codes inside [0, 1] and no AWG3 channel below zero.
            let c3 = t.c9_logc4.to_logc3_ei800().0;
            if c3.iter().any(|v| !(0.0..=1.0).contains(v)) {
                continue;
            }
            checked += 1;
            let l = with_look.trace(px);
            assert_close(
                l.c10_logc4.0,
                t.c9_logc4.0,
                1e-9,
                "logc3 identity round trip",
            );
            assert_close(
                with_look.apply_pixel(px),
                neutral.apply_pixel(px),
                1e-8,
                "output",
            );
        }
        assert!(
            checked >= 30,
            "only {checked} pixels were inside the LogC3 domain"
        );
    }
}

#[test]
fn logc3_look_with_a_real_lut_matches_the_documented_path() {
    let lut = cube_lut(17, tweak);
    let look =
        Arc::new(LookTransform::new(LookInput::LogC3Ei800, None, Some((*lut).clone())).unwrap());
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.look = Some(LookSetting::new(look, 70.0));
    let c = chain(&p);
    for px in pixels() {
        let t = c.trace(px);
        // LogC4 -> linear AWG4 -> AWG3 -> LogC3 EI 800 -> LUT -> linear AWG3 -> AWG4 -> LogC4, blended in LogC4.
        let in3 = t.c9_logc4.to_logc3_ei800();
        let out4 = ym_color::logc::LogC3Ei800(lut.sample(in3.0)).to_logc4();
        let want = blend_logc4(t.c9_logc4, out4, 70.0);
        assert_close(t.c10_logc4.0, want.0, 1e-12, "logc3 look");
    }
}

// ---- LogC3-to-Rec.709 looks ----------------------------------------------------------------------

/// A synthetic "LogC3 to Rec.709" LUT of our own: a smooth tone map from LogC3 code values to
/// Rec.709 gamma 2.4 code values, with a small colour twist.
fn rec709_baked_look() -> Arc<LookTransform> {
    let lut = cube_lut(33, |c| {
        let t = tweak(c);
        [t[0].powf(1.1), t[1] * 0.95 + 0.02, t[2].powf(0.9)]
    });
    Arc::new(LookTransform::new(LookInput::LogC3ToRec709, None, Some((*lut).clone())).unwrap())
}

#[test]
fn rec709_baked_look_at_zero_is_our_c12_exactly() {
    let look = rec709_baked_look();
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.grade = grade_params();
    p.curves = curves();
    let reference = chain(&p);
    p.look = Some(LookSetting::new(look, 0.0));
    let zero = chain(&p);
    assert!(zero.look_includes_display_rendering());
    for px in pixels() {
        assert_eq!(zero.trace(px), reference.trace(px));
    }
}

#[test]
fn rec709_baked_look_at_100_replaces_c12_for_every_output() {
    let look = rec709_baked_look();
    for target in OutputTarget::ALL {
        let mut p = ChainParams::new(fused_matrix(), tone());
        p.output = target;
        p.curves = curves();
        p.look = Some(LookSetting::new(Arc::clone(&look), 100.0));
        let c = chain(&p);
        for px in pixels() {
            let t = c.trace(px);
            let rec709 = look.eval_display_rec709(t.c11_logc4).unwrap();
            let want = target.space().from_linear_rec709(rec709);
            assert_eq!(t.c12_display_linear, want, "{target:?}");
            assert_eq!(t.c13_encoded, target.space().encode_clamped(want));
            // The same LUT output computed by hand: LogC4 -> LogC3 -> LUT -> gamma 2.4.
            let codes = look_lut_codes(&look, t.c11_logc4);
            let by_hand = rec709_gamma24_decode(codes);
            assert_eq!(rec709, by_hand);
        }
    }
}

fn look_lut_codes(look: &LookTransform, v: LogC4) -> Vec3 {
    look.lut().unwrap().sample(v.to_logc3_ei800().0)
}

#[test]
fn rec709_baked_look_blends_between_c12_and_the_lut_in_display_linear() {
    let look = rec709_baked_look();
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.curves = curves();
    let own = chain(&p);
    p.look = Some(LookSetting::new(Arc::clone(&look), 100.0));
    let full = chain(&p);
    for intensity in [25.0, 50.0, 90.0] {
        p.look = Some(LookSetting::new(Arc::clone(&look), intensity));
        let mixed = chain(&p);
        for px in pixels() {
            let a = own.trace(px).c12_display_linear;
            let b = full.trace(px).c12_display_linear;
            let want = blend_display_linear(a, b, intensity);
            assert_eq!(mixed.trace(px).c12_display_linear, want);
            let k = intensity / 100.0;
            for ch in 0..3 {
                let by_formula = a[ch] + k * (b[ch] - a[ch]);
                assert!((want[ch] - by_formula).abs() < 1e-15);
            }
        }
    }
}

#[test]
fn rec709_baked_look_blends_with_an_arri_display_lut_as_c12_too() {
    use ym_color::display::{ArriDisplayLut, DisplayLutInput, DisplayLutOutput, DisplayTransform};
    let arri = ArriDisplayLut::new(
        cube_lut(17, |c| [c[0].powf(1.05), c[1], c[2] * 0.97]),
        DisplayLutInput::LogC4,
        DisplayLutOutput::REC709_G24,
    )
    .unwrap();
    let look = rec709_baked_look();
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.display = DisplayTransform::arri(arri);
    let own = chain(&p);
    p.look = Some(LookSetting::new(Arc::clone(&look), 40.0));
    let mixed = chain(&p);
    p.look = Some(LookSetting::new(look, 100.0));
    let full = chain(&p);
    for px in pixels() {
        let want = blend_display_linear(
            own.trace(px).c12_display_linear,
            full.trace(px).c12_display_linear,
            40.0,
        );
        assert_eq!(mixed.trace(px).c12_display_linear, want);
    }
}

#[test]
fn rec709_baked_look_cannot_go_into_a_dng() {
    let mut p = ChainParams::new(fused_matrix(), tone());
    p.look = Some(LookSetting::new(rec709_baked_look(), 100.0));
    let c = chain(&p);
    assert!(!c.dng_available());
    assert_eq!(
        c.apply_dng_pixel([0.2, 0.2, 0.2]),
        Err(DisplayError::DngUnavailable)
    );
    let data = vec![0.2_f32; 12];
    let src = Frame::interleaved(2, 2, &data).unwrap();
    let mut out = vec![0.0_f32; 12];
    assert_eq!(
        c.render_dng_linear_awg4_f32(&src, Layout::Interleaved, &mut out, 1),
        Err(DisplayError::DngUnavailable)
    );
}

// ---- the DNG stop point --------------------------------------------------------------------------

#[test]
fn dng_stop_point_applies_grade_look_and_curve_then_decodes_to_linear_awg4() {
    let p = rich_params();
    let c = chain(&p);
    assert!(c.dng_available());
    for px in pixels() {
        let t = c.trace(px);
        let dng = c.apply_dng_pixel(px).unwrap();
        // Decoded C11 output, nothing after it.
        assert_close(dng, t.c11_logc4.to_linear().0, 1e-15, "dng");
        // Re-encoding gives C11 back.
        assert_close(logc4_encode_rgb(dng), t.c11_logc4.0, 1e-9, "re-encode");
    }
    // Look and curve really are in it: without them the values differ.
    let plain = chain(&ChainParams::new(fused_matrix(), tone()));
    let moved = pixels()
        .into_iter()
        .filter(|px| {
            let (a, b) = (
                c.apply_dng_pixel(*px).unwrap(),
                plain.apply_dng_pixel(*px).unwrap(),
            );
            max_diff(logc4_encode_rgb(a), logc4_encode_rgb(b)) > 1e-3
        })
        .count();
    assert!(
        moved > 40,
        "grade, look and curve changed only {moved} pixels"
    );
}

#[test]
fn dng_stop_point_of_a_neutral_chain_is_the_tone_mapped_linear_awg4() {
    // Identity grade, Neutral look, identity curves: the DNG holds C7's output, up to the log round trip.
    let c = chain(&ChainParams::new(fused_matrix(), tone()));
    for px in pixels() {
        let want = tone().apply_rgb(linalg::mul_vec(&fused_matrix(), px));
        let got = c.apply_dng_pixel(px).unwrap();
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() <= 1e-12 + 1e-11 * want[k].abs(),
                "{px:?}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn dng_buffer_render_matches_the_pixel_function() {
    let c = chain(&rich_params());
    let px = pixels();
    let (w, h) = (px.len(), 1);
    let data: Vec<f32> = px.iter().flat_map(|p| p.map(|v| v as f32)).collect();
    let src = Frame::interleaved(w, h, &data).unwrap();
    let mut out = vec![0.0_f32; 3 * w];
    c.render_dng_linear_awg4_f32(&src, Layout::Interleaved, &mut out, 1)
        .unwrap();
    for (i, p) in px.iter().enumerate() {
        let f32_in = p.map(|v| f64::from(v as f32));
        let want = c.apply_dng_pixel(f32_in).unwrap();
        for k in 0..3 {
            assert_eq!(out[3 * i + k], want[k] as f32);
        }
    }
}

// ---- the Before view -----------------------------------------------------------------------------

#[test]
fn before_view_is_the_clamped_logc4_signal_used_as_code_values() {
    let p = rich_params();
    let c = chain(&p);
    let mut after = p.clone();
    after.before = BeforePoint::AfterGrade;
    let after = chain(&after);
    let mut differs = 0;
    for px in pixels() {
        let t = c.trace(px);
        let want8 = t.c8_logc4.0.map(|v| v.clamp(0.0, 1.0));
        let want9 = t.c9_logc4.0.map(|v| v.clamp(0.0, 1.0));
        assert_eq!(c.apply_before_pixel(px), want8);
        assert_eq!(after.apply_before_pixel(px), want9);
        assert_eq!(c.apply_view_pixel(View::Before, px), want8);
        assert_eq!(c.apply_view_pixel(View::Final, px), c.apply_pixel(px));
        if max_diff(want8, want9) > 1e-3 {
            differs += 1;
        }
    }
    assert!(differs > 20);
    // Mid grey sits at the LogC4 code of 0.18 in every channel when the matrix is neutral.
    let neutral = chain(&ChainParams::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ToneParams::new(0.5, 0.0, 16.0, 0.0).unwrap(),
    ));
    let g = neutral.apply_before_pixel([0.18, 0.18, 0.18]);
    for v in g {
        assert!((v - 0.278_395_836_5).abs() < 1e-9);
    }
}

#[test]
fn before_view_ignores_look_curve_and_display() {
    let rich = chain(&rich_params());
    let mut bare = rich_params();
    bare.look = None;
    bare.curves = CurveSet::default();
    bare.output = OutputTarget::Rec2020;
    let bare = chain(&bare);
    for px in pixels() {
        assert_eq!(rich.apply_before_pixel(px), bare.apply_before_pixel(px));
    }
}

// ---- encode, dither, buffers ---------------------------------------------------------------------

fn test_image(w: usize, h: usize) -> Vec<f32> {
    // Interleaved, deterministic: a gradient with seeded texture.
    let mut rng = Rng(0x5EED_1234);
    let mut v = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let t = (x + y) as f64 / (w + h) as f64;
            let scale = 0.01 * 2.0_f64.powf(9.0 * t);
            for k in 0..3 {
                v.push((scale * (0.5 + rng.unit() + 0.2 * k as f64)) as f32);
            }
        }
    }
    v
}

fn to_planar(interleaved: &[f32]) -> Vec<f32> {
    let n = interleaved.len() / 3;
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = interleaved[3 * i + c];
        }
    }
    out
}

#[test]
fn buffer_renders_equal_the_pixel_function_in_both_layouts() {
    let (w, h) = (37, 23);
    let inter = test_image(w, h);
    let planar = to_planar(&inter);
    let c = chain(&rich_params());
    let a = Frame::interleaved(w, h, &inter).unwrap();
    let b = Frame::planar(w, h, &planar).unwrap();
    let mut out_i = vec![0.0_f32; 3 * w * h];
    let mut out_p = vec![0.0_f32; 3 * w * h];
    // Same input, all four layout combinations.
    c.render_encoded_f32(View::Final, &a, Layout::Interleaved, &mut out_i, 1)
        .unwrap();
    c.render_encoded_f32(View::Final, &b, Layout::Planar, &mut out_p, 1)
        .unwrap();
    for i in 0..w * h {
        let px = [inter[3 * i], inter[3 * i + 1], inter[3 * i + 2]].map(f64::from);
        let want = c.apply_pixel(px).map(|v| v as f32);
        for k in 0..3 {
            assert_eq!(out_i[3 * i + k], want[k], "interleaved {i}");
            assert_eq!(out_p[k * w * h + i], want[k], "planar {i}");
        }
    }
    let mut cross = vec![0.0_f32; 3 * w * h];
    c.render_encoded_f32(View::Final, &b, Layout::Interleaved, &mut cross, 2)
        .unwrap();
    assert_eq!(cross, out_i);
    c.render_encoded_f32(View::Final, &a, Layout::Planar, &mut cross, 3)
        .unwrap();
    assert_eq!(cross, out_p);
}

#[test]
fn output_is_identical_for_any_thread_count() {
    let (w, h) = (53, 41);
    let inter = test_image(w, h);
    let src = Frame::interleaved(w, h, &inter).unwrap();
    let c = chain(&rich_params());
    let mut reference8 = vec![0_u8; 3 * w * h];
    c.render_u8(View::Final, &src, (0, 0), &mut reference8, 1)
        .unwrap();
    let mut reference_f = vec![0.0_f32; 3 * w * h];
    c.render_display_linear_f32(&src, Layout::Planar, &mut reference_f, 1)
        .unwrap();
    for threads in [0, 2, 3, 4, 7, 64] {
        let mut out8 = vec![0_u8; 3 * w * h];
        c.render_u8(View::Final, &src, (0, 0), &mut out8, threads)
            .unwrap();
        assert_eq!(out8, reference8, "u8, {threads} threads");
        let mut out_f = vec![0.0_f32; 3 * w * h];
        c.render_display_linear_f32(&src, Layout::Planar, &mut out_f, threads)
            .unwrap();
        assert_eq!(out_f, reference_f, "f32, {threads} threads");
    }
    // Repeated runs agree too.
    let mut again = vec![0_u8; 3 * w * h];
    c.render_u8(View::Final, &src, (0, 0), &mut again, 4)
        .unwrap();
    assert_eq!(again, reference8);
}

#[test]
fn tiles_dither_like_the_whole_image() {
    let (w, h) = (96, 80);
    let inter = test_image(w, h);
    let src = Frame::interleaved(w, h, &inter).unwrap();
    let c = chain(&rich_params());
    let mut whole = vec![0_u8; 3 * w * h];
    c.render_u8(View::Final, &src, (0, 0), &mut whole, 0)
        .unwrap();
    // Cut out a tile that does not start on a multiple of the 64 px noise period.
    let (x0, y0, tw, th) = (21, 13, 50, 41);
    let mut tile_px = Vec::with_capacity(3 * tw * th);
    for y in y0..y0 + th {
        tile_px.extend_from_slice(&inter[3 * (y * w + x0)..3 * (y * w + x0 + tw)]);
    }
    let tile = Frame::interleaved(tw, th, &tile_px).unwrap();
    let mut tile_out = vec![0_u8; 3 * tw * th];
    c.render_u8(View::Final, &tile, (x0, y0), &mut tile_out, 0)
        .unwrap();
    for y in 0..th {
        let a = &tile_out[3 * y * tw..3 * (y + 1) * tw];
        let b = &whole[3 * ((y0 + y) * w + x0)..3 * ((y0 + y) * w + x0 + tw)];
        assert_eq!(a, b, "row {y}");
    }
}

#[test]
fn dither_stays_within_one_code_of_plain_rounding_and_matches_the_encode_module() {
    let (w, h) = (64, 64);
    let inter = test_image(w, h);
    let src = Frame::interleaved(w, h, &inter).unwrap();
    let dithered = chain(&rich_params());
    let mut plain_params = rich_params();
    plain_params.dither = false;
    let plain = chain(&plain_params);
    let mut a = vec![0_u8; 3 * w * h];
    let mut b = vec![0_u8; 3 * w * h];
    dithered
        .render_u8(View::Final, &src, (0, 0), &mut a, 0)
        .unwrap();
    plain
        .render_u8(View::Final, &src, (0, 0), &mut b, 0)
        .unwrap();
    let mut different = 0;
    for i in 0..a.len() {
        assert!(a[i].abs_diff(b[i]) <= 1);
        different += usize::from(a[i] != b[i]);
    }
    assert!(different > 100, "dither changed only {different} samples");
    // The chain's 8-bit encode is the encode module's.
    let lin = dithered.apply_to_display_linear([0.2, 0.3, 0.1]);
    for (x, y) in [(0, 0), (5, 9), (63, 63), (130, 200)] {
        assert_eq!(
            dithered.encode_u8(lin, x, y),
            encode::encode_8bit_dithered(lin, OutputSpace::DisplayP3, x, y)
        );
    }
}

#[test]
fn sixteen_and_ten_bit_outputs_are_not_dithered_and_match_encode_quantized() {
    let (w, h) = (20, 11);
    let inter = test_image(w, h);
    let src = Frame::interleaved(w, h, &inter).unwrap();
    let c = chain(&rich_params());
    for bits in [10_u32, 16] {
        let mut out = vec![0_u16; 3 * w * h];
        c.render_u16(View::Final, bits, &src, &mut out, 2).unwrap();
        for i in 0..w * h {
            let px = [inter[3 * i], inter[3 * i + 1], inter[3 * i + 2]].map(f64::from);
            let lin = c.apply_to_display_linear(px);
            let want = encode::encode_quantized(lin, OutputSpace::DisplayP3, bits);
            assert_eq!([out[3 * i], out[3 * i + 1], out[3 * i + 2]], want);
        }
    }
}

#[test]
fn export_path_resizes_in_display_linear_then_encodes() {
    // render_display_linear_f32 -> (a resize would go here) -> encode_display_linear_u8 equals render_u8.
    let (w, h) = (31, 17);
    let inter = test_image(w, h);
    let src = Frame::interleaved(w, h, &inter).unwrap();
    let c = chain(&rich_params());
    let mut linear = vec![0.0_f32; 3 * w * h];
    c.render_display_linear_f32(&src, Layout::Interleaved, &mut linear, 0)
        .unwrap();
    let lin_frame = Frame::interleaved(w, h, &linear).unwrap();
    let mut via_linear = vec![0_u8; 3 * w * h];
    c.encode_display_linear_u8(&lin_frame, (0, 0), &mut via_linear, 0)
        .unwrap();
    let mut direct = vec![0_u8; 3 * w * h];
    c.render_u8(View::Final, &src, (0, 0), &mut direct, 0)
        .unwrap();
    // f32 storage of the display-linear values can move a sample across a rounding boundary.
    let mut off_by_one = 0;
    for i in 0..direct.len() {
        let d = direct[i].abs_diff(via_linear[i]);
        assert!(d <= 1, "sample {i}: {} vs {}", direct[i], via_linear[i]);
        off_by_one += usize::from(d == 1);
    }
    assert!(
        off_by_one * 50 < direct.len(),
        "too many f32 storage flips: {off_by_one}"
    );
}

#[test]
fn buffer_errors_are_typed() {
    let c = chain(&ChainParams::new(fused_matrix(), tone()));
    let data = vec![0.1_f32; 3 * 4 * 3];
    assert!(matches!(
        Frame::planar(4, 4, &data),
        Err(DisplayError::BadBuffer {
            got: 36,
            expected: 48,
            ..
        })
    ));
    assert!(matches!(
        Frame::interleaved(usize::MAX, 2, &data),
        Err(DisplayError::TooLarge { .. })
    ));
    let src = Frame::interleaved(4, 3, &data).unwrap();
    let mut short = vec![0_u8; 5];
    assert!(matches!(
        c.render_u8(View::Final, &src, (0, 0), &mut short, 1),
        Err(DisplayError::BadBuffer {
            got: 5,
            expected: 36,
            ..
        })
    ));
    // Empty images are fine.
    let empty = Frame::interleaved(0, 7, &[]).unwrap();
    c.render_u8(View::Final, &empty, (0, 0), &mut [], 4)
        .unwrap();
}

#[test]
fn a_non_finite_matrix_is_refused_and_non_finite_pixels_are_defined() {
    let mut bad = fused_matrix();
    bad[1][2] = f64::NAN;
    assert_eq!(
        ColorChain::new(&ChainParams::new(bad, tone())).err(),
        Some(DisplayError::NonFiniteMatrix)
    );
    let c = chain(&rich_params());
    for px in [
        [f64::NAN, 0.2, 0.2],
        [f64::INFINITY, 0.2, 0.2],
        [f64::NEG_INFINITY, f64::NAN, 1e300],
    ] {
        for v in c.apply_pixel(px) {
            assert!((0.0..=1.0).contains(&v), "{px:?} gave {v}");
        }
        for v in c.apply_before_pixel(px) {
            assert!((0.0..=1.0).contains(&v));
        }
    }
    // NaN counts as zero.
    assert_eq!(c.trace([f64::NAN, 0.2, 0.2]), c.trace([0.0, 0.2, 0.2]));
}

#[test]
fn chain_is_send_sync_and_cheap_to_share() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ColorChain>();
    assert_send_sync::<ChainParams>();
}

// ---- ACES 2 neutral ramp (Delta E 2000 smoke test) -------------------------------------------------

fn lab_of_display_p3(encoded: Vec3) -> Vec3 {
    let linear = encoded.map(|v| OutputSpace::DisplayP3.decode_channel(v));
    let xyz = linalg::mul_vec(&matrices().p3_to_xyz, linear);
    xyz_to_lab(xyz, d65_white_xyz())
}

#[test]
fn neutral_ramp_through_aces2_is_monotone_and_stays_neutral() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tone_off = ToneParams::new(0.5, 0.0, 1.0e6, 0.0).unwrap();
    let c = chain(&ChainParams::new(identity, tone_off));
    // From -8 to +7 stops around grey in third-stop steps.
    let steps: Vec<f64> = (0..=45).map(|i| -8.0 + f64::from(i) / 3.0).collect();
    let mut previous: Option<Vec3> = None;
    let mut worst_chroma = 0.0_f64;
    let mut worst_de_to_neutral = 0.0_f64;
    let mut smallest_step = f64::INFINITY;
    let mut l_at_grey = f64::NAN;
    let mut top_l = 0.0;
    for stops in &steps {
        let g = 0.18 * stops.exp2();
        let out = c.apply_pixel([g, g, g]);
        let lab = lab_of_display_p3(out);
        let chroma = lab[1].hypot(lab[2]);
        worst_chroma = worst_chroma.max(chroma);
        worst_de_to_neutral = worst_de_to_neutral.max(delta_e_2000(lab, [lab[0], 0.0, 0.0]));
        // R = G = B for a neutral input.
        assert!(max_diff([out[0]; 3], out) < 1e-9, "stops {stops}: {out:?}");
        if let Some(prev) = previous {
            // Non-decreasing code values; strictly increasing until the display clips.
            for k in 0..3 {
                assert!(
                    out[k] >= prev[k],
                    "not monotone at {stops} stops: {prev:?} -> {out:?}"
                );
            }
            if out[0] < 0.999 && prev[0] > 0.0 {
                let de = delta_e_2000(lab_of_display_p3(prev), lab);
                assert!(de > 0.0, "flat spot at {stops} stops");
                smallest_step = smallest_step.min(de);
            }
        }
        if stops.abs() < 1e-9 {
            l_at_grey = lab[0];
        }
        top_l = lab[0];
        previous = Some(out);
    }
    println!(
        "neutral ramp: worst chroma {worst_chroma:.4}, worst dE00 to neutral {worst_de_to_neutral:.4}, \
         smallest step dE00 {smallest_step:.4}, L* at grey {l_at_grey:.2}, L* at +7 stops {top_l:.2}"
    );
    assert!(worst_chroma < 0.05, "neutral drifted: C* {worst_chroma}");
    assert!(worst_de_to_neutral < 0.05);
    assert!(smallest_step > 0.0);
    // ACES 2 puts mid grey at a plausible display lightness and highlights at (almost) white.
    assert!((30.0..55.0).contains(&l_at_grey), "L* at grey {l_at_grey}");
    assert!(top_l > 97.0, "L* at +7 stops {top_l}");
    // Black stays black.
    assert!(c.apply_pixel([0.0, 0.0, 0.0]).iter().all(|v| *v < 1e-3));
}

#[test]
fn aces2_white_point_matches_the_encoding_white() {
    // A D65-neutral AWG4 white lands on the display's D65 white: equal channels, near 1.
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let c = chain(&ChainParams::new(
        identity,
        ToneParams::new(0.5, 0.0, 1.0e6, 0.0).unwrap(),
    ));
    let white = c.apply_pixel([16.0, 16.0, 16.0]);
    assert!(
        white
            .iter()
            .all(|v| (*v - white[0]).abs() < 1e-9 && *v > 0.97)
    );
}

#[test]
fn c13_srgb_differs_from_the_aces2_references_moncurve_by_under_1e5() {
    // The chain encodes with IEC 61966-2-1 (docs/02, C13); the ACES 2 port's own display encode uses the
    // reference's continuous moncurve (slope 12.9232 below 0.0030399). Bound the difference, which peaks
    // near linear 0.00304 at 9.8e-6 (0.0025 of an 8-bit code, 0.64 of a 16-bit code).
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tone_off = ToneParams::new(0.5, 0.0, 1.0e6, 0.0).unwrap();
    for target in OutputTarget::ALL {
        let mut p = ChainParams::new(identity, tone_off);
        p.output = target;
        let c = chain(&p);
        let reference = Aces2Transform::new(target.aces2());
        let mut worst = 0.0_f64;
        for stops in -140..=30 {
            let g = 0.18 * (f64::from(stops) / 10.0).exp2();
            let ap0 = linalg::mul_vec(&aces2_input_matrix(), [g, g * 0.9, g * 0.7]);
            worst = worst.max(max_diff(
                c.apply_pixel([g, g * 0.9, g * 0.7]),
                reference.transform_ap0(ap0),
            ));
        }
        println!("{target:?}: chain vs transform_ap0 worst {worst:.2e}");
        assert!(worst < 1.0e-5, "{target:?}: {worst}");
    }
}

#[test]
fn c12_alone_matches_the_trace_and_params_rebuild_a_chain_for_another_output() {
    let p = rich_params();
    let c = chain(&p);
    let mut rebuilt = c.params().clone();
    rebuilt.output = OutputTarget::Srgb;
    let srgb = chain(&rebuilt);
    for px in pixels() {
        let t = c.trace(px);
        assert_eq!(
            c.display_linear_from_logc4(t.c11_logc4),
            t.c12_display_linear
        );
        // Only C12 onwards differ between the viewport chain and its sRGB rebuild.
        let u = srgb.trace(px);
        assert_eq!(t.c11_logc4, u.c11_logc4);
    }
    // The look is shared by the rebuild, not copied.
    let (a, b) = (
        c.params().look.as_ref().unwrap(),
        srgb.params().look.as_ref().unwrap(),
    );
    assert!(Arc::ptr_eq(&a.look, &b.look));
    assert_eq!(srgb.output(), OutputTarget::Srgb);
}

#[test]
fn tiny_frames_and_oversubscribed_threads_work() {
    let c = chain(&rich_params());
    for (w, h) in [(1, 1), (1, 9), (9, 1), (2, 2)] {
        let data = test_image(w, h);
        let src = Frame::interleaved(w, h, &data).unwrap();
        let mut one = vec![0_u8; 3 * w * h];
        let mut many = vec![0_u8; 3 * w * h];
        c.render_u8(View::Final, &src, (3, 4), &mut one, 1).unwrap();
        c.render_u8(View::Final, &src, (3, 4), &mut many, 16)
            .unwrap();
        assert_eq!(one, many, "{w} x {h}");
    }
}

// ---- review additions ------------------------------------------------------------------------------

#[test]
fn band_splitting_is_exact_for_odd_shapes_layouts_and_thread_counts() {
    // Shapes that put the band boundaries in awkward places: one row, one column, widths and
    // heights either side of the 64 row band limit and of the 64 px noise period.
    let c = chain(&rich_params());
    for (w, h) in [
        (1, 1),
        (1, 130),
        (130, 1),
        (3, 2),
        (63, 65),
        (64, 64),
        (65, 63),
        (7, 131),
    ] {
        let inter = test_image(w, h);
        let planar = to_planar(&inter);
        let (fi, fp) = (
            Frame::interleaved(w, h, &inter).unwrap(),
            Frame::planar(w, h, &planar).unwrap(),
        );
        let n = w * h;
        let mut want8 = vec![0_u8; 3 * n];
        c.render_u8(View::Final, &fi, (5, 7), &mut want8, 1)
            .unwrap();
        let mut want_f = vec![0.0_f32; 3 * n];
        c.render_encoded_f32(View::Final, &fi, Layout::Planar, &mut want_f, 1)
            .unwrap();
        for threads in [0, 2, 3, 5, 17, usize::MAX] {
            for src in [&fi, &fp] {
                let mut got8 = vec![0_u8; 3 * n];
                c.render_u8(View::Final, src, (5, 7), &mut got8, threads)
                    .unwrap();
                assert_eq!(got8, want8, "u8 {w}x{h}, {threads} threads");
                let mut got_planar = vec![0.0_f32; 3 * n];
                c.render_encoded_f32(View::Final, src, Layout::Planar, &mut got_planar, threads)
                    .unwrap();
                assert_eq!(got_planar, want_f, "planar {w}x{h}, {threads} threads");
                let mut got_inter = vec![0.0_f32; 3 * n];
                c.render_encoded_f32(
                    View::Final,
                    src,
                    Layout::Interleaved,
                    &mut got_inter,
                    threads,
                )
                .unwrap();
                for i in 0..n {
                    for k in 0..3 {
                        assert_eq!(got_inter[3 * i + k], want_f[k * n + i], "{w}x{h} {i}");
                    }
                }
            }
        }
    }
}

#[test]
fn a_tile_origin_near_the_end_of_usize_wraps_instead_of_panicking() {
    // The origin only picks a noise texel. `origin + x` used to overflow (a panic in debug builds)
    // for an origin within the image width of `usize::MAX`.
    let (w, h) = (9, 6);
    let inter = test_image(w, h);
    let src = Frame::interleaved(w, h, &inter).unwrap();
    let c = chain(&rich_params());
    let origin = (usize::MAX - 3, usize::MAX - 1);
    let mut out = vec![0_u8; 3 * w * h];
    c.render_u8(View::Final, &src, origin, &mut out, 2).unwrap();
    // Pixel (x, y) uses texel (origin + (x, y)) mod 2^64, which the pixel function agrees with.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let px = [inter[3 * i], inter[3 * i + 1], inter[3 * i + 2]].map(f64::from);
            let want = c.quantize_u8(
                c.apply_pixel(px),
                origin.0.wrapping_add(x),
                origin.1.wrapping_add(y),
            );
            assert_eq!(out[3 * i..3 * i + 3], want, "pixel ({x}, {y})");
        }
    }
    let mut again = vec![0_u8; 3 * w * h];
    let lin = vec![0.2_f32; 3 * w * h];
    c.encode_display_linear_u8(
        &Frame::interleaved(w, h, &lin).unwrap(),
        origin,
        &mut again,
        0,
    )
    .unwrap();
    assert!(again.iter().any(|v| *v != 0));
}

#[test]
fn debug_output_stays_small_for_big_inputs() {
    // A derived `Debug` printed every LUT entry (a 17-point cube is 14,739 numbers; a 65-point one
    // 824,000) and every sample of a frame.
    let c = chain(&rich_params());
    let text = format!("{c:?} {:?}", c.params());
    assert!(
        text.len() < 6_000,
        "ColorChain Debug is {} bytes",
        text.len()
    );
    assert!(text.contains("intensity_pct"), "{text}");
    let data = vec![0.5_f32; 3 * 256 * 256];
    let frame = Frame::planar(256, 256, &data).unwrap();
    let text = format!("{frame:?}");
    assert!(text.len() < 200 && text.contains("256"), "{text}");
    let lut = ym_color::display::ArriDisplayLut::new(
        cube_lut(33, tweak),
        ym_color::display::DisplayLutInput::LogC4,
        ym_color::display::DisplayLutOutput::REC709_G24,
    )
    .unwrap();
    let text = format!("{lut:?}");
    assert!(text.len() < 400 && text.contains("33"), "{text}");
}
