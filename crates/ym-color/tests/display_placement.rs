//! C12 step 1a (`docs/02-colour-pipeline.md`): the display placement, `2^DISPLAY_PLACEMENT_EV` on the
//! scene-linear AP0 triple before the ACES 2 transform, on the ACES 2 path only.
//!
//! * Neutral look and ACES 2 put scene 0.18 at the display-linear Y that OpenColorIO's built-in ACES 2.0
//!   SDR transform gives for `0.18 * 2^k`, and the placement is exactly `2^k` (scene `0.18 * 2^-k` lands
//!   where the unplaced transform puts 0.18).
//! * The placement equals an exposure offset of `+k` in front of the chain on the Neutral look (C7 is
//!   homogeneous away from the low-light toe).
//! * An imported ARRI display LUT and a LogC3-to-Rec.709 look keep ARRI's own brightness: their C12 is
//!   bit for bit what the LUT gives, with no placement in it.
//!
//! OpenColorIO's values (2.6.0, float32, `pip install opencolorio==2.6.0`):
//!
//! ```python
//! import PyOpenColorIO as ocio, numpy as np
//! t = ocio.BuiltinTransform()
//! t.setStyle("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0")   # or ...-P3-D65_2.0
//! g = ocio.GroupTransform(); g.appendTransform(t)
//! proc = ocio.Config.CreateRaw().getProcessor(g).getDefaultCPUProcessor()
//! px = np.array([[s, s, s]], dtype=np.float32); proc.applyRGB(px)    # s = 0.18, 0.18 * 2**0.85
//! # px[0][1] is the display Y (1.0 = 100 nit): 0.099999 and 0.189253 (the same for both limiting gamuts)
//! ```

use std::sync::Arc;

use ym_color::aces2::Aces2Transform;
use ym_color::consts::Vec3;
use ym_color::consts::linalg::{self, Mat3};
use ym_color::display::{
    ArriDisplayLut, ChainParams, ColorChain, DISPLAY_PLACEMENT_EV, DisplayLutInput,
    DisplayLutOutput, DisplayTransform, LookSetting, OutputTarget, aces2_input_matrix,
};
use ym_color::encode::OutputSpace;
use ym_color::gamut::matrices;
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_color::tone::ToneParams;

/// OpenColorIO's display Y of neutral scene 0.18 through ACES 2.0 SDR 100 nit, no placement.
const OCIO_Y_AT_GREY: f64 = 0.099_999;
/// OpenColorIO's display Y of neutral scene `0.18 * 2^0.85` (the value the placement targets).
const OCIO_Y_AT_PLACED_GREY: f64 = 0.189_253;
/// The tolerance of the two values above.
const Y_TOLERANCE: f64 = 5.0e-4;

const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn gain() -> f64 {
    DISPLAY_PLACEMENT_EV.exp2()
}

/// C7 off: no highlight range, no cap, no toe.
fn tone_off() -> ToneParams {
    ToneParams::IDENTITY
}

fn chain_with(matrix: Mat3, tone: ToneParams, output: OutputTarget) -> ColorChain {
    let mut p = ChainParams::new(matrix, tone);
    p.output = output;
    ColorChain::new(&p).unwrap()
}

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

/// Scene pixels from 8 stops below to 4 stops above grey, mostly chromatic (linear AWG4).
fn scene_pixels(seed: u64, n: usize) -> Vec<Vec3> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let s = 0.18 * 2.0_f64.powf(-8.0 + 12.0 * rng.unit());
            [
                s * (0.3 + 1.4 * rng.unit()),
                s * (0.3 + 1.4 * rng.unit()),
                s * (0.3 + 1.4 * rng.unit()),
            ]
        })
        .collect()
}

fn max_diff(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)
}

#[test]
fn the_placement_is_the_documented_constant_folded_into_the_matrix() {
    assert_eq!(DISPLAY_PLACEMENT_EV, 0.85);
    let (plain, placed) = (matrices().awg4_to_ap0, aces2_input_matrix());
    for r in 0..3 {
        for c in 0..3 {
            assert_eq!(placed[r][c], plain[r][c] * DISPLAY_PLACEMENT_EV.exp2());
        }
    }
    // The gamut matrix itself is untouched (the DNG path, the ARRI tables and the tools use it).
    for s in linalg::row_sums(&plain) {
        assert!((s - 1.0).abs() < 1e-15, "{s}");
    }
    // The placed rows sum to 2^k: scene white `[1, 1, 1]` goes to AP0 `[2^k; 3]`.
    for s in linalg::row_sums(&placed) {
        assert!((s - gain()).abs() < 1e-14, "{s}");
    }
}

#[test]
fn neutral_look_puts_scene_grey_at_the_opencolorio_display_y_of_the_placed_grey() {
    for target in OutputTarget::ALL {
        let chain = chain_with(IDENTITY, tone_off(), target);
        // Scene 0.18 through the Neutral look and ACES 2 with the placement.
        let linear = chain.trace([0.18; 3]).c12_display_linear;
        for (ch, y) in linear.iter().enumerate() {
            assert!(
                (y - OCIO_Y_AT_PLACED_GREY).abs() < Y_TOLERANCE,
                "{target:?} channel {ch}: display Y {y} against OpenColorIO's {OCIO_Y_AT_PLACED_GREY}"
            );
        }
        // The placement is exactly 2^k: scene 0.18 / 2^k is where the plain transform puts 0.18.
        let back = chain.trace([0.18 / gain(); 3]).c12_display_linear;
        for (ch, y) in back.iter().enumerate() {
            assert!(
                (y - OCIO_Y_AT_GREY).abs() < Y_TOLERANCE,
                "{target:?} channel {ch}: display Y {y} against OpenColorIO's {OCIO_Y_AT_GREY}"
            );
        }
    }
    // What it means in code values and video level (sRGB export, 8 bit; video % at gamma 2.4).
    let srgb = chain_with(IDENTITY, tone_off(), OutputTarget::Srgb);
    let code = srgb.apply_pixel([0.18; 3])[1] * 255.0;
    assert!(
        (code - 120.4).abs() < 0.3,
        "mid grey is at sRGB code {code} (a camera JPEG: 118)"
    );
    let video_pct = 100.0 * srgb.trace([0.18; 3]).c12_display_linear[1].powf(1.0 / 2.4);
    assert!((video_pct - 50.0).abs() < 0.1, "{video_pct} % video");
}

#[test]
fn the_placement_equals_an_exposure_offset_of_plus_k_on_the_neutral_look() {
    for target in OutputTarget::ALL {
        let reference = Aces2Transform::new(target.aces2());
        // C7 off: exact to rounding for every pixel, near-black included.
        let placed = chain_with(IDENTITY, tone_off(), target);
        for px in scene_pixels(0x5EED_0001, 400) {
            // The unplaced transform of the scene exposed +k stops: no placement in it.
            let exposed = px.map(|v| v * gain());
            let ap0 = linalg::mul_vec(&matrices().awg4_to_ap0, exposed);
            let want = reference.transform_ap0_linear(ap0);
            let got = placed.trace(px).c12_display_linear;
            assert!(
                max_diff(got, want) < 1e-9,
                "{target:?} {px:?}: {got:?} against {want:?}"
            );
        }
        // C7 on (the highlight range at an Auto-like clip, no toe): the placement comes after C7, so
        // the placed chain equals the unplaced transform of the C7 output exposed +k. The range is
        // not homogeneous (its knee floor and the ALEXA cap are fixed stops over grey), so the
        // placement is not the same as more exposure in front of C7.
        let tone = ToneParams::new(0.5, 0.0, 0.18 * 3.58_f64.exp2(), 0.0).unwrap();
        let placed = chain_with(IDENTITY, tone, target);
        let mut ranged = 0;
        for px in scene_pixels(0x5EED_0002, 400) {
            let c7 = tone.apply_rgb(px);
            ranged += usize::from(c7 != px);
            let c7_exposed = c7.map(|v| v * gain());
            let ap0 = linalg::mul_vec(&matrices().awg4_to_ap0, c7_exposed);
            let want = reference.transform_ap0_linear(ap0);
            let got = placed.trace(px).c12_display_linear;
            assert!(
                max_diff(got, want) < 1e-8,
                "{target:?} {px:?}: {got:?} against {want:?}"
            );
        }
        assert!(ranged > 20, "only {ranged} pixels reach the range");
    }
}

/// An identity cube over `[0, 1]` as an imported LogC4 to Rec.709 gamma 2.4 display LUT.
fn arri_display(input: DisplayLutInput) -> DisplayTransform {
    let lut = Arc::new(Lut::from_cube(
        Lut3d::bake(17, [0.0; 3], [1.0; 3], |c| c).unwrap(),
    ));
    DisplayTransform::arri(ArriDisplayLut::new(lut, input, DisplayLutOutput::REC709_G24).unwrap())
}

#[test]
fn an_imported_arri_display_lut_gets_no_placement() {
    for input in [DisplayLutInput::LogC4, DisplayLutInput::LogC3Ei800] {
        for target in OutputTarget::ALL {
            let mut p = ChainParams::new(IDENTITY, tone_off());
            p.display = arri_display(input);
            p.output = target;
            let chain = ColorChain::new(&p).unwrap();
            for px in scene_pixels(0x5EED_0003, 200) {
                let t = chain.trace(px);
                // By hand, from the finished modules: the LUT's codes, decoded with gamma 2.4, then the
                // primaries matrix of the output. The same arithmetic as before the placement existed,
                // so equal bit for bit; an exposure of 2^k in front would change every value.
                let codes = match input {
                    DisplayLutInput::LogC4 => t.c11_logc4.0,
                    DisplayLutInput::LogC3Ei800 => t.c11_logc4.to_logc3_ei800().0,
                }
                .map(|v| v.clamp(0.0, 1.0));
                let linear = codes.map(|c| if c > 0.0 { c.powf(2.4) } else { 0.0 });
                let want = match target.space() {
                    OutputSpace::Srgb => linear,
                    OutputSpace::DisplayP3 => linalg::mul_vec(&matrices().rec709_to_p3, linear),
                    OutputSpace::Rec2020 => linalg::mul_vec(
                        &linalg::mul(&matrices().xyz_to_rec2020, &matrices().rec709_to_xyz),
                        linear,
                    ),
                };
                assert!(
                    max_diff(t.c12_display_linear, want) < 1e-12,
                    "{input:?} {target:?} {px:?}: {:?} against {want:?}",
                    t.c12_display_linear
                );
                // The picture is the LUT's, not 2^k brighter.
                let brighter = want.map(|v| v * gain());
                if want[1] > 1e-3 && want[1] < 0.5 {
                    assert!(max_diff(t.c12_display_linear, brighter) > 1e-3);
                }
            }
        }
    }
}

/// A synthetic "LogC3 to Rec.709" look of our own (never an ARRI file): a gentle tone map.
fn rec709_look() -> Arc<LookTransform> {
    let lut = Lut3d::bake(17, [0.0; 3], [1.0; 3], |c| {
        [c[0].powf(1.1), c[1] * 0.95 + 0.02, c[2].powf(0.9)]
    })
    .unwrap();
    Arc::new(LookTransform::new(LookInput::LogC3ToRec709, None, Some(Lut::from_cube(lut))).unwrap())
}

#[test]
fn a_logc3_to_rec709_look_keeps_its_own_brightness_and_the_blend_starts_from_the_placed_c12() {
    let look = rec709_look();
    for target in OutputTarget::ALL {
        let mut p = ChainParams::new(IDENTITY, tone_off());
        p.output = target;
        let own = ColorChain::new(&p).unwrap();
        p.look = Some(LookSetting::new(Arc::clone(&look), 100.0));
        let full = ColorChain::new(&p).unwrap();
        p.look = Some(LookSetting::new(Arc::clone(&look), 0.0));
        let zero = ColorChain::new(&p).unwrap();
        let reference = Aces2Transform::new(target.aces2());
        for px in scene_pixels(0x5EED_0004, 100) {
            // 100 %: the look's LUT, decoded and converted, with nothing of the placement in it.
            let t = full.trace(px);
            let rec709 = look.eval_display_rec709(t.c11_logc4).unwrap();
            assert_eq!(
                t.c12_display_linear,
                target.space().from_linear_rec709(rec709)
            );
            // 0 %: our C12, which is the placed ACES 2 path.
            let ap0 = linalg::mul_vec(&matrices().awg4_to_ap0, px.map(|v| v * gain()));
            let want = reference.transform_ap0_linear(ap0);
            assert!(max_diff(zero.trace(px).c12_display_linear, want) < 1e-9);
            assert_eq!(
                zero.trace(px).c12_display_linear,
                own.trace(px).c12_display_linear
            );
        }
        // The slider therefore moves the brightness: mid grey differs between 0 and 100 %.
        let grey = [0.18; 3];
        let (a, b) = (
            zero.trace(grey).c12_display_linear[1],
            full.trace(grey).c12_display_linear[1],
        );
        assert!(a > 0.0 && b > 0.0 && (a / b - 1.0).abs() > 0.01, "{a} {b}");
        // Looks still see mid grey at LogC4 0.2784 as ever: the placement is after C10 and C11.
        let c11 = full.trace(grey).c11_logc4;
        assert!((c11.0[1] - 0.2783958365).abs() < 1e-9);
    }
}
