//! C12: the chain's display transform is the cine rendering, for every output, with nothing in front of it.

use ym_color::consts::Vec3;
use ym_color::display::cine::CineRendering;
use ym_color::display::{ChainParams, ColorChain, OutputTarget};
use ym_color::gamut::LinearAwg4;
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

/// A chain that does nothing before C12: identity matrix, C7 off, zero grade, Neutral look.
fn bare(output: OutputTarget) -> ColorChain {
    let mut p = ChainParams::new(IDENTITY, ToneParams::IDENTITY);
    p.output = output;
    ColorChain::new(&p).unwrap()
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
fn the_chain_is_the_cine_rendering_for_every_output() {
    for target in OutputTarget::ALL {
        let chain = bare(target);
        let reference = CineRendering::new(target);
        let mut worst = 0.0_f64;
        for px in scene_pixels() {
            let t = chain.trace(px);
            // C12 keeps display-linear in the output's primaries; C13 is the output's transfer.
            let want_linear = reference.forward(LinearAwg4(px));
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
        }
        println!("cine {target:?}: worst difference {worst:.2e}");
    }
}

#[test]
fn output_is_always_a_valid_code_value() {
    for target in OutputTarget::ALL {
        let chain = bare(target);
        for px in scene_pixels() {
            for v in chain.apply_pixel(px) {
                assert!((0.0..=1.0).contains(&v), "{target:?} {px:?}: {v}");
            }
        }
    }
}

#[test]
fn nothing_stands_in_front_of_the_rendering() {
    // No display placement, no white lift, no red steps: middle grey through Neutral lands where the
    // rendering puts it (display-linear 0.1109 at 0.18 scene), within the LogC4 round trip.
    let chain = bare(OutputTarget::Srgb);
    let v = chain.apply_to_display_linear([0.18; 3]);
    for c in v {
        assert!((c - 0.1109).abs() < 5e-4, "{v:?}");
    }
}

#[test]
fn the_wider_outputs_use_their_own_primaries() {
    // A saturated green keeps more chroma in Display P3 than in Rec.709 and the chains differ only there.
    let px = [0.02, 0.4, 0.05];
    let srgb = bare(OutputTarget::Srgb).apply_to_display_linear(px);
    let p3 = bare(OutputTarget::DisplayP3).apply_to_display_linear(px);
    let viewport = bare(OutputTarget::Viewport).apply_to_display_linear(px);
    assert_eq!(p3, viewport);
    assert!(max_diff(srgb, p3) > 1e-3, "{srgb:?} {p3:?}");
}

#[test]
fn rendering_is_monotone_on_a_neutral_ramp() {
    // A sanity check on the plumbing.
    let ramp: Vec<f64> = (0..40)
        .map(|i| 0.18 * (f64::from(i) / 3.0 - 6.0).exp2())
        .collect();
    let chain = bare(OutputTarget::Viewport);
    let mut last = [-1.0; 3];
    for g in &ramp {
        let out = chain.apply_pixel([*g; 3]);
        for k in 0..3 {
            assert!(out[k] >= last[k], "not monotone at {g}");
        }
        last = out;
    }
}
