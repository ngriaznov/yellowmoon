//! The Film node's own Rec.2020 output (`docs/24-film.md` section 3.9c, F-08b): the print is scanned and returned in the
//! Rec.2020 gamut instead of through Display P3's. A print density is the same colour in any primaries, so the baked scan
//! of one output is the other's within the tables' interpolation, and the return stage reaches the same scene colour from
//! the same display colour.
mod bake_common;
use bake_common::*;
use ym_color::gamut::matrices;
use ym_film::Output;
use ym_film::bake::{BakeSettings, DRAFT_SIZE};

fn xyz(m: &[[f64; 3]; 3], rgb: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| (0..3).map(|j| m[i][j] * rgb[j]).sum())
}

#[test]
fn the_scan_in_rec2020_is_the_same_colour_as_in_display_p3() {
    let m = matrices();
    let mut worst = 0.0_f64;
    for stock in [EIGHT[0], EIGHT[2], EIGHT[7]] {
        let mut scans = Vec::new();
        for out in [Output::DisplayP3, Output::Rec2020] {
            let chain = chain(stock, out);
            let mut bs = BakeSettings::of(&chain);
            bs.t3a_size = DRAFT_SIZE;
            bs.t3b_size = 65;
            scans.push(baker(2).bake(&chain, &bs).expect("bake"));
        }
        for d in [
            [0.35, 0.4, 0.45],
            [0.8, 0.9, 1.0],
            [1.2, 1.1, 1.3],
            [0.6, 0.9, 1.4],
        ] {
            let p3 = xyz(&m.p3_to_xyz, scans[0].scan_display(d));
            let r2 = xyz(&m.rec2020_to_xyz, scans[1].scan_display(d));
            for i in 0..3 {
                worst = worst.max((p3[i] - r2[i]).abs() / p3[1].abs().max(0.02));
            }
        }
    }
    assert!(worst < 5e-3, "the scan differs between outputs: {worst:.3e}");
}

#[test]
fn the_return_stage_of_rec2020_reaches_a_finite_scene_colour_and_keeps_a_neutral_neutral() {
    let chain = chain(EIGHT[0], Output::Rec2020);
    let mut bs = BakeSettings::of(&chain);
    bs.t3a_size = DRAFT_SIZE;
    bs.t3b_size = 65;
    let baked = baker(2).bake(&chain, &bs).expect("bake");
    assert_eq!(baked.output(), Output::Rec2020);
    for grey in [0.02, 0.1, 0.18, 0.5, 0.9] {
        let back = baked.return_to_logc4([grey, grey, grey]);
        assert!(back.iter().all(|v| v.is_finite()), "{grey}: {back:?}");
        let spread = back.iter().fold(0.0_f64, |a, v| a.max((v - back[1]).abs()));
        assert!(spread < 1e-3, "{grey}: a neutral comes back with a cast {spread:.2e}");
    }
}
