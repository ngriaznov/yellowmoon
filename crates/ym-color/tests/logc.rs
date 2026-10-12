//! M1-01: LogC4 and LogC3 (EI 800) spot values, round trips and the exact LogC3 <-> LogC4 conversion.

use proptest::prelude::*;
use ym_color::consts::{LOGC3_EI800, LOGC4, LogC4Constants};
use ym_color::gamut::LinearAwg4;
use ym_color::logc::{
    LogC3Ei800, LogC4, logc3_ei800_decode, logc3_ei800_encode, logc4_decode, logc4_encode,
};

#[test]
fn logc4_constants_match_docs() {
    // Printed in docs/02 section LogC4.
    assert!((LOGC4.a - 2231.8263090676883).abs() < 1e-9);
    assert!((LOGC4.b - 0.9071358748778103).abs() < 1e-15);
    assert!((LOGC4.c - 0.09286412512218964).abs() < 1e-16);
    assert!((LOGC4.s - 0.1135972086105891).abs() < 1e-16);
    assert!((LOGC4.t - (-0.01805699611991131)).abs() < 1e-16);
    // And the formulas evaluate to the same numbers.
    let f = LogC4Constants::from_formulas();
    assert!((f.s - LOGC4.s).abs() < 1e-16 && (f.t - LOGC4.t).abs() < 1e-16);
}

#[test]
fn logc4_spot_values() {
    assert!((logc4_encode(0.18) - 0.2783958365).abs() < 5e-11);
    assert!((logc4_encode(0.0) - 0.0928641251).abs() < 5e-11);
    // decode(1.0) = 469.8 within the printed precision (colour-science prints 469.8 too).
    assert!(
        (logc4_decode(1.0) - 469.8).abs() < 0.05,
        "{}",
        logc4_decode(1.0)
    );
    // colour-science 0.4.7 log_encoding_ARRILogC4 at the same inputs (pasted from a local run).
    let cs = [
        (0.0, 0.09286412512218964),
        (0.005, 0.10788865969709344),
        (0.18, 0.2783958365482653),
        (1.0, 0.42751936483530617),
        (10.0, 0.6403898462614473),
        (400.0, 0.9849654314501207),
        (-0.01, 0.052778017073297455),
    ];
    for (x, want) in cs {
        assert!(
            (logc4_encode(x) - want).abs() < 1e-14,
            "x={x}: {} vs {want}",
            logc4_encode(x)
        );
    }
    // 11.35 stops above 0.18: log2(469.8 / 0.18) = 11.35.
    assert!(((logc4_decode(1.0) / 0.18).log2() - 11.35).abs() < 0.01);
}

#[test]
fn logc4_stop_is_b_over_14() {
    // One stop in the log segment is b / 14 code values = 0.06480 (docs/02 section C9).
    let s = LOGC4.b / 14.0;
    assert!((s - 0.06480).abs() < 5e-6);
    // Far above the toe, doubling the input adds one stop.
    let d = logc4_encode(2000.0) - logc4_encode(1000.0);
    assert!((d - s).abs() < 2e-4);
}

#[test]
fn logc3_spot_values() {
    assert!((logc3_ei800_encode(0.18) - 400.0 / 1023.0).abs() < 2e-8);
    assert!((logc3_ei800_encode(0.18) - 0.3910068).abs() < 5e-8);
    // colour-science 0.4.7 log_encoding_ARRILogC3 (EI 800, SUP 3.x) pasted from a local run.
    let cs = [
        (0.0, 0.092809),
        (0.18, 0.39100683203408376),
        (1.0, 0.5706315581204173),
        (10.0, 0.8169171588006766),
        (0.010591, 0.149657834105),
        (0.02, 0.19104968410606715),
    ];
    for (x, want) in cs {
        assert!((logc3_ei800_encode(x) - want).abs() < 1e-14, "x={x}");
    }
    for (e, want) in [(0.5, 0.513383396023284), (0.9, 21.69348589542808)] {
        assert!(
            (logc3_ei800_decode(e) - want).abs() < 1e-11 * want.abs().max(1.0),
            "e={e}"
        );
    }
}

#[test]
fn logc4_round_trip_over_the_documented_range() {
    // docs/02 Tests: 1e-12 relative (1e-14 absolute near zero) over [-0.02, 400].
    let n = 200_000;
    let mut worst_rel = 0.0_f64;
    for k in 0..=n {
        let x = -0.02 + 400.02 * (k as f64 / n as f64);
        let y = logc4_decode(logc4_encode(x));
        let err = (y - x).abs();
        assert!(
            err <= 1e-14 + 1e-12 * x.abs(),
            "x={x}: round trip {y}, err {err:e}"
        );
        worst_rel = worst_rel.max(err / x.abs().max(1e-300));
    }
    // A log-spaced sweep as well, since a linear sweep barely visits the toe and the grey region.
    for k in 0..=n {
        let x = 1e-6 * (4.0e8_f64).powf(k as f64 / n as f64);
        let y = logc4_decode(logc4_encode(x));
        assert!((y - x).abs() <= 1e-14 + 1e-12 * x, "x={x}");
    }
    // Negative side below the toe: the linear segment is exactly invertible.
    for k in 0..1000 {
        let x = -0.02 - 5.0 * (k as f64 / 1000.0);
        assert!((logc4_decode(logc4_encode(x)) - x).abs() < 1e-14 + 1e-12 * x.abs());
    }
    let _ = worst_rel;
}

#[test]
fn logc3_round_trip_within_1e_7() {
    // docs/02 Tests: 1e-7 absolute. The published EI 800 constants leave a 2.5e-7 step at `cut`.
    let n = 200_000;
    let mut worst = 0.0_f64;
    for k in 0..=n {
        let x = -0.02 + 200.02 * (k as f64 / n as f64);
        let y = logc3_ei800_decode(logc3_ei800_encode(x));
        worst = worst.max((y - x).abs());
    }
    for k in 0..=n {
        let x = 1e-5 * (2.0e7_f64).powf(k as f64 / n as f64);
        let y = logc3_ei800_decode(logc3_ei800_encode(x));
        worst = worst.max((y - x).abs());
    }
    assert!(worst < 1e-7, "worst LogC3 round trip error {worst:e}");
    println!("LogC3 EI800 worst linear-domain round-trip error: {worst:e}");
}

#[test]
fn logc3_step_at_cut_is_as_documented() {
    // The two segments disagree by ~2.5e-7 code values at `cut` (published constants are rounded).
    let c = LOGC3_EI800;
    let log_side = c.c * (c.a * c.cut + c.b).log10() + c.d;
    let lin_side = c.e * c.cut + c.f;
    let step = (log_side - lin_side).abs();
    println!("LogC3 EI800 step at cut: {step:e}");
    assert!(step < 3e-7 && step > 1e-8, "{step:e}");
}

#[test]
fn logc3_code_value_round_trip() {
    // Code values round trip too (decode then encode), away from the step at cut.
    for k in 0..=1000 {
        let t = 0.1 + 0.9 * (k as f64 / 1000.0);
        let y = logc3_ei800_encode(logc3_ei800_decode(t));
        assert!((y - t).abs() < 1e-7, "t={t}");
    }
}

#[test]
fn logc4_to_logc3_conversion_is_exact_for_greys_and_round_trips() {
    // Grey stays grey (both gamuts share the D65 white).
    for v in [0.1, 0.2784, 0.5, 0.8] {
        let out = LogC4([v, v, v]).to_logc3_ei800().0;
        assert!((out[0] - out[1]).abs() < 1e-12 && (out[1] - out[2]).abs() < 1e-12);
        // The same linear value, encoded directly.
        let lin = logc4_decode(v);
        assert!((out[0] - logc3_ei800_encode(lin)).abs() < 1e-12);
    }
    // 18% grey: LogC4 0.2784 to LogC3 0.3910.
    let g = LogC4([0.2783958365482653; 3]).to_logc3_ei800().0;
    assert!((g[0] - 400.0 / 1023.0).abs() < 2e-8);
    // Colour round trip. LogC3 is limited by its 2.5e-7 step, so compare in LogC4 with that bound.
    let c = LogC4([0.45, 0.30, 0.20]);
    let back = c.to_logc3_ei800().to_logc4().0;
    for (b, orig) in back.iter().zip(c.0) {
        assert!((b - orig).abs() < 1e-9, "{back:?}");
    }
    let l3 = LogC3Ei800([0.5, 0.4, 0.3]);
    let l3b = l3.to_logc4().to_logc3_ei800().0;
    for (b, orig) in l3b.iter().zip(l3.0) {
        assert!((b - orig).abs() < 1e-9);
    }
}

#[test]
fn negative_linear_passes_through_unclipped() {
    for x in [-0.001, -0.01, -0.5, -10.0] {
        assert!(logc4_encode(x) < logc4_encode(0.0));
        assert!((logc4_decode(logc4_encode(x)) - x).abs() < 1e-12);
    }
    let lin = LinearAwg4([-0.05, 0.18, 0.4]);
    let back = lin.to_logc4().to_linear().0;
    for (b, orig) in back.iter().zip(lin.0) {
        assert!((b - orig).abs() < 1e-12);
    }
}

proptest! {
    #[test]
    fn logc4_is_monotonic(a in -0.5_f64..500.0, b in -0.5_f64..500.0) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(logc4_encode(lo) <= logc4_encode(hi));
    }

    #[test]
    fn logc4_round_trip_prop(x in -0.02_f64..400.0) {
        let y = logc4_decode(logc4_encode(x));
        prop_assert!((y - x).abs() <= 1e-14 + 1e-12 * x.abs());
    }
}
