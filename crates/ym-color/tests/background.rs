//! Background's source plane (`docs/algorithms/06` section 5.4, N-15): `S0 = S((P - (1 - level) + w / 2) / w)`
//! with `w = max(softness, BLUR_RAMP_MIN)` on the cells of a blur map, rounded once to a u16 code, against an
//! independent evaluation on known maps; the two ends of Level and Softness, the minimum ramp, a map of an evenly
//! sharp photo; and the cache that keeps Level and Softness (and nothing else) in the key.

use std::sync::Arc;

use ym_color::mask_raster::{
    BLUR_RAMP_MIN, BackgroundSource, BlurMap, RasterSource, background_plane, background_s0,
};

/// The smoothstep written out here, so the test does not lean on the code it checks.
fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The formula of section 5.4 in f64, as a code.
fn reference(p: f64, level: f64, softness: f64) -> u16 {
    let w = softness.max(0.05);
    let s = smooth((p - (1.0 - level) + 0.5 * w) / w);
    (s * 65535.0 + 0.5).floor() as u16
}

fn map(key: u64, p: Vec<f32>, width: usize) -> Arc<BlurMap> {
    let height = p.len() / width;
    Arc::new(BlurMap {
        key,
        width,
        height,
        p,
    })
}

#[test]
fn the_formula_matches_an_independent_evaluation_over_the_whole_range_of_p_level_and_softness() {
    let ps: Vec<f32> = (0..=200).map(|i| i as f32 / 200.0).collect();
    let mut worst = 0u32;
    for level in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        for softness in [0.0, 0.02, 0.05, 0.1, 0.5, 0.8, 1.0] {
            let got = background_s0(&ps, level, softness);
            for (p, code) in ps.iter().zip(&got) {
                let want = reference(f64::from(*p), level, softness);
                worst = worst.max(u32::from(code.abs_diff(want)));
            }
        }
    }
    // f32 cells against f64: at most one code.
    assert!(worst <= 1, "{worst} codes");
    eprintln!("[background] the formula against f64: largest difference {worst} code(s)");
}

#[test]
fn the_default_puts_the_edge_where_the_maps_own_decision_falls_and_the_ramp_is_the_softness() {
    // Level 0.5, Softness 0.5: the ramp spans P = 0.25 to 0.75, half way at P = 0.5.
    let p = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0];
    let got = background_s0(&p, 0.5, 0.5);
    assert_eq!(got[0], 0);
    assert_eq!(got[1], 0);
    assert_eq!(got[3], 32768, "S(0.5) is half");
    assert_eq!(got[5], 65535);
    assert_eq!(got[6], 65535);
    // The smoothstep is odd about the middle: one quarter and three quarters of the way sum to one.
    assert!((i32::from(got[2]) + i32::from(got[4]) - 65535).abs() <= 1);
    assert!(got[1] <= got[2] && got[2] <= got[3] && got[3] <= got[4] && got[4] <= got[5]);
}

#[test]
fn a_higher_level_lets_more_count_and_a_lower_one_only_what_is_strongly_out_of_focus() {
    let p: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
    let at = |level: f64| background_s0(&p, level, 0.5);
    let (low, mid, high) = (at(0.2), at(0.5), at(0.8));
    for i in 0..p.len() {
        assert!(low[i] <= mid[i] && mid[i] <= high[i], "P = {}", p[i]);
    }
    // Level 0.2 puts the edge (half way up the ramp) at P = 0.8: a cell at P = 0.5 is not background, one at P = 1
    // nearly is (the ramp ends a quarter above the edge, at 1.05).
    assert_eq!(low[50], 0);
    assert!(low[100] > 60_000, "{}", low[100]);
    // Level 0.8 puts it at P = 0.2: a cell at P = 0.5 is background, one at P = 0 nearly is not.
    assert_eq!(high[50], 65535);
    assert!(high[0] < 2_500, "{}", high[0]);
}

#[test]
fn the_ends_of_level() {
    let p: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
    // Level 0: the edge is at P = 1, so only a certain cell reaches half and nothing is background below the ramp.
    let zero = background_s0(&p, 0.0, 0.5);
    assert_eq!(zero[75], 0, "P = 0.75 is the bottom of the ramp");
    assert_eq!(zero[100], 32768, "P = 1 is half way");
    assert!(zero.iter().all(|c| *c <= 32768));
    // Level 1: the edge is at P = 0 - the formula gives half at a cell with no blur at all (and the whole mask from
    // P = w / 2), so Level 100 is "everything counts, the blurred part fully".
    let one = background_s0(&p, 1.0, 0.5);
    assert_eq!(one[0], 32768);
    assert_eq!(one[25], 65535);
    assert!(one.iter().all(|c| *c >= 32768));
}

#[test]
fn the_ends_of_softness_and_the_minimum_ramp() {
    let p: Vec<f32> = (0..=1000).map(|i| i as f32 / 1000.0).collect();
    // Softness 0 is read as the minimum ramp, 0.05, so the edge is never a step.
    assert_eq!(BLUR_RAMP_MIN, 0.05);
    assert_eq!(
        background_s0(&p, 0.5, 0.0),
        background_s0(&p, 0.5, BLUR_RAMP_MIN)
    );
    let narrow = background_s0(&p, 0.5, 0.0);
    // The ramp of 0.05 about P = 0.5: 0 at 0.475, 1 at 0.525, and the cells in between take every step of the way
    // (a ramp, not a step).
    assert_eq!(narrow[474], 0);
    assert_eq!(narrow[526], 65535);
    let between = narrow[475..=525]
        .iter()
        .filter(|c| **c > 0 && **c < 65535)
        .count();
    assert!(between >= 48, "{between} cells on the ramp");
    // Softness 1: the ramp is the whole range of P about the edge (P = 0 to 1 at Level 0.5).
    let wide = background_s0(&p, 0.5, 1.0);
    assert!(wide.windows(2).all(|w| w[0] <= w[1]), "monotone in P");
    assert_eq!(wide[500], 32768);
    assert_eq!(wide[0], 0, "the bottom of the ramp is P = 0");
    assert_eq!(wide[1000], 65535, "and its top is P = 1");
    // A wider ramp is gentler: a cell a little above the edge is less background at Softness 1 than at 0.1.
    assert!(background_s0(&[0.6], 0.5, 1.0)[0] < background_s0(&[0.6], 0.5, 0.1)[0]);
}

#[test]
fn an_evenly_sharp_photo_gives_an_empty_mask() {
    // The map of a photo with no out-of-focus area: P is 0 in every cell. At the default Level and Softness, and at
    // every Level up to 1 - w / 2, the plane is all zero.
    let sharp = vec![0.0f32; 512 * 342];
    assert!(background_s0(&sharp, 0.5, 0.5).iter().all(|c| *c == 0));
    for level in [0.0, 0.25, 0.5, 0.7, 0.75] {
        assert!(
            background_s0(&sharp, level, 0.5).iter().all(|c| *c == 0),
            "level {level}"
        );
    }
    // A map of noise-level probabilities (a faint 0.03) is empty too.
    let faint = vec![0.03f32; 64];
    assert!(background_s0(&faint, 0.5, 0.5).iter().all(|c| *c == 0));
    // And a map that is out of focus everywhere is the whole photo.
    let blurred = vec![1.0f32; 64];
    assert!(
        background_s0(&blurred, 0.5, 0.5)
            .iter()
            .all(|c| *c == 65535)
    );
}

#[test]
fn nan_and_values_outside_the_range_do_not_break_the_plane() {
    let p = [f32::NAN, -3.0, 7.0, f32::INFINITY, f32::NEG_INFINITY];
    let got = background_s0(&p, 0.5, 0.5);
    assert_eq!(got[0], 0, "NaN reads as no background");
    assert_eq!(got[1], 0);
    assert_eq!(got[2], 65535);
    assert_eq!(got[3], 65535);
    assert_eq!(got[4], 0);
    // A level or softness out of range or NaN is held to the range.
    assert_eq!(
        background_s0(&[0.5], f64::NAN, 0.5),
        background_s0(&[0.5], 0.0, 0.5)
    );
    assert_eq!(
        background_s0(&[0.5], 7.0, f64::NAN),
        background_s0(&[0.5], 1.0, 0.0)
    );
}

#[test]
fn the_plane_is_the_maps_cells_cut_once_and_the_codes_are_the_formulas() {
    let cells: Vec<f32> = (0..12).map(|i| i as f32 / 11.0).collect();
    let m = map(9_001, cells.clone(), 4);
    let src = BackgroundSource {
        map: Arc::clone(&m),
        level: 0.6,
        softness: 0.3,
    };
    let plane = background_plane(&src);
    assert_eq!((plane.width, plane.height), (4, 3));
    assert_eq!(plane.data, background_s0(&cells, 0.6, 0.3));
    assert_eq!(plane.key, src.key());
    // The source of a row gives the same plane, whatever the photo's aspect (the map's cells cover the photo frame).
    let source = RasterSource::Background(Arc::new(src));
    assert!(Arc::ptr_eq(&source.plane([1.0, 0.6]), &plane));
    assert!(Arc::ptr_eq(&source.plane([1.0, 1.0]), &plane));
    assert_eq!(source.key(), plane.key);
}

#[test]
fn level_and_softness_make_a_new_plane_from_the_cached_map_and_nothing_else_does() {
    let m = map(9_002, (0..64).map(|i| i as f32 / 63.0).collect(), 8);
    let source = |level: f64, softness: f64| BackgroundSource {
        map: Arc::clone(&m),
        level,
        softness,
    };
    let base = background_plane(&source(0.5, 0.5));
    // The same map, Level and Softness are the same plane (the cache), Arc for Arc.
    assert!(Arc::ptr_eq(&base, &background_plane(&source(0.5, 0.5))));
    // A new source value made from the same numbers has the same key (Feather, Grow and Refine edges are not in it).
    assert_eq!(source(0.5, 0.5).key(), source(0.5, 0.5).key());
    // Level and Softness each make another plane, with another key.
    let level = background_plane(&source(0.6, 0.5));
    let softness = background_plane(&source(0.5, 0.6));
    assert!(!Arc::ptr_eq(&base, &level) && !Arc::ptr_eq(&base, &softness));
    assert_ne!(base.key, level.key);
    assert_ne!(base.key, softness.key);
    assert_ne!(level.key, softness.key);
    assert_ne!(base.data, level.data);
    assert_ne!(base.data, softness.data);
    // The old planes are still cached.
    assert!(Arc::ptr_eq(&base, &background_plane(&source(0.5, 0.5))));
    // Another map (another photo, other raw parameters) never reads this map's planes, with the same numbers.
    let other = Arc::new(BlurMap {
        key: 9_003,
        ..(*m).clone()
    });
    let again = background_plane(&BackgroundSource {
        map: other,
        level: 0.5,
        softness: 0.5,
    });
    assert!(!Arc::ptr_eq(&base, &again));
    assert_ne!(base.key, again.key);
    assert_eq!(base.data, again.data, "the same cells give the same codes");
}

#[test]
fn a_map_with_no_cells_or_the_wrong_number_of_them_is_an_empty_plane() {
    let broken = Arc::new(BlurMap {
        key: 9_004,
        width: 4,
        height: 4,
        p: vec![1.0; 3],
    });
    let plane = background_plane(&BackgroundSource {
        map: broken,
        level: 0.5,
        softness: 0.5,
    });
    assert_eq!((plane.width, plane.height), (4, 4));
    assert!(plane.data.iter().all(|c| *c == 0));
}
