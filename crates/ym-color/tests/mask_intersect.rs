//! Mask Intersect on the CPU reference (`docs/23-nodes.md` section 4.2, `docs/algorithms/06-masks.md` section 3; MI-03): the
//! rows of a node whose `intersect` is on multiply. The rule is written out again here, in the shape of the maths file, and the
//! module is held to it.

use ym_color::logc::LogC4;
use ym_color::mask::{MaskKind, MaskRow, NodeMasks, PhotoFrame, PhotoSpot, PixelAt};

fn spot(x: f64, y: f64) -> PixelAt<'static> {
    Some(PhotoSpot::new([x, y], PhotoFrame::new(1000.0, 1000.0, 1.0)))
}

fn grey() -> LogC4 {
    LogC4([0.3, 0.3, 0.3])
}

fn linear(center: [f64; 2], angle_deg: f64, feather: f64) -> MaskRow {
    MaskRow::new(MaskKind::Linear {
        center,
        angle_deg,
        feather,
    })
}

fn radial(center: [f64; 2], radii: [f64; 2], feather: f64) -> MaskRow {
    MaskRow::new(MaskKind::Radial {
        center,
        radii,
        angle_deg: 0.0,
        feather,
    })
}

fn colour(center: [f64; 2], width: f64, softness: f64) -> MaskRow {
    MaskRow::new(MaskKind::ColorRange {
        center,
        width,
        softness,
    })
}

fn brightness(low: f64, high: f64, softness: f64) -> MaskRow {
    MaskRow::new(MaskKind::BrightnessRange {
        low,
        high,
        softness,
    })
}

/// A row that selects nothing: a raster row with no plane (an Object row with no point is one).
fn empty() -> MaskRow {
    MaskRow::new(MaskKind::Raster { plane: usize::MAX })
}

/// A row whose `s` is 1 over the photo's middle (a Linear line along the bottom edge: the whole photo above it).
fn whole(opacity: f64) -> MaskRow {
    MaskRow {
        opacity,
        ..linear([0.5, 1.0], 0.0, 0.0)
    }
}

/// A row whose `a` is exactly 0 there: `whole` with its own Invert.
fn none() -> MaskRow {
    MaskRow {
        invert: true,
        ..whole(1.0)
    }
}

fn sub(row: MaskRow) -> MaskRow {
    MaskRow {
        subtract: true,
        ..row
    }
}

fn run(rows: &[MaskRow], intersect: bool, invert: bool, key: Option<f64>, at: (f64, f64)) -> f64 {
    NodeMasks {
        rows: rows.to_vec(),
        invert,
        intersect,
    }
    .prepare()
    .mask(key, grey(), 0.3, spot(at.0, at.1))
}

fn mid(rows: &[MaskRow], intersect: bool) -> f64 {
    run(rows, intersect, false, None, (0.5, 0.5))
}

/// The rule of the maths file, written out: `f` of a row from its own `a` (its Invert applied, no Opacity yet).
fn factor(a: f64, o: f64, subtract: bool) -> f64 {
    if subtract {
        1.0 - o * a
    } else {
        1.0 - o * (1.0 - a)
    }
}

fn kinds() -> Vec<(&'static str, MaskRow)> {
    vec![
        ("linear", linear([0.5, 0.5], 20.0, 0.3)),
        ("radial", radial([0.45, 0.5], [0.3, 0.2], 0.4)),
        ("colour range", colour([0.0, 0.0], 0.05, 0.5)),
        ("brightness range", brightness(0.2, 0.6, 0.4)),
        ("raster without a plane", empty()),
    ]
}

const PLACES: [(f64, f64); 5] = [
    (0.5, 0.5),
    (0.2, 0.3),
    (0.9, 0.1),
    (0.45, 0.8),
    (0.05, 0.95),
];

#[test]
fn one_row_at_full_opacity_is_the_same_mask_in_both_rules_bit_for_bit() {
    for (name, row) in kinds() {
        for invert in [false, true] {
            for subtract in [false, true] {
                let r = MaskRow {
                    invert,
                    subtract,
                    ..row.clone()
                };
                for at in PLACES {
                    let add = run(&[r.clone()], false, false, None, at);
                    let int = run(&[r.clone()], true, false, None, at);
                    assert_eq!(
                        add.to_bits(),
                        int.to_bits(),
                        "{name} invert {invert} subtract {subtract} at {at:?}: {add} against {int}"
                    );
                }
            }
        }
    }
}

#[test]
fn an_opacity_of_zero_is_neutral_for_an_add_row_and_for_a_subtract_row() {
    let base = [whole(1.0), radial([0.5, 0.5], [0.4, 0.4], 0.5)];
    let want = mid(&base, true);
    for zero in [whole(0.0), radial([0.2, 0.2], [0.1, 0.1], 0.1), none()] {
        let mut rows = base.to_vec();
        rows.push(MaskRow {
            opacity: 0.0,
            ..zero.clone()
        });
        assert_eq!(
            mid(&rows, true).to_bits(),
            want.to_bits(),
            "an Add row at 0"
        );
        rows.pop();
        rows.push(MaskRow {
            opacity: 0.0,
            ..sub(zero)
        });
        assert_eq!(
            mid(&rows, true).to_bits(),
            want.to_bits(),
            "a Subtract row at 0"
        );
    }
    // Alone, a row at 0 % is the whole photo (the neutral value of an intersection is 1).
    assert_eq!(mid(&[whole(0.0)], true), 1.0);
    assert_eq!(mid(&[sub(whole(0.0))], true), 1.0);
    assert_eq!(
        mid(
            &[MaskRow {
                opacity: 0.0,
                ..empty()
            }],
            true
        ),
        1.0
    );
    // In Add, the same row adds nothing: the mask is empty.
    assert_eq!(mid(&[whole(0.0)], false), 0.0);
}

#[test]
fn opacity_scales_a_row_towards_one() {
    // An Add row at half opacity is 1 - 0.5 (1 - a); a Subtract row 1 - 0.5 a.
    let half = MaskRow {
        opacity: 0.5,
        ..radial([0.5, 0.5], [0.3, 0.3], 0.4)
    };
    for at in PLACES {
        let a = run(
            &[radial([0.5, 0.5], [0.3, 0.3], 0.4)],
            false,
            false,
            None,
            at,
        );
        assert!(
            (run(&[half.clone()], true, false, None, at) - factor(a, 0.5, false)).abs() < 1e-15
        );
        assert!(
            (run(&[sub(half.clone())], true, false, None, at) - factor(a, 0.5, true)).abs() < 1e-15
        );
    }
}

#[test]
fn an_empty_row_empties_the_mask_and_a_row_that_is_off_does_not() {
    // A row that selects nothing is present: a = 0, so f = 0 and the product is 0 wherever the others are.
    let rows = [whole(1.0), empty(), radial([0.5, 0.5], [0.4, 0.4], 0.3)];
    for at in PLACES {
        assert_eq!(run(&rows, true, false, None, at), 0.0, "{at:?}");
    }
    // The same rows in Add keep what the others add.
    assert!(mid(&rows, false) > 0.9);
    // With its own Invert the empty row is a = 1 and narrows nothing.
    let inverted = [
        whole(1.0),
        MaskRow {
            invert: true,
            ..empty()
        },
    ];
    assert_eq!(mid(&inverted, true), 1.0);
    // A row that is off is not in the list at all (the node's rows are the enabled ones): the others stand.
    assert_eq!(mid(&[whole(1.0)], true), 1.0);
    // A lower opacity lets the rest through by 1 - o.
    let faint = [
        whole(1.0),
        MaskRow {
            opacity: 0.25,
            ..empty()
        },
    ];
    assert!((mid(&faint, true) - 0.75).abs() < 1e-15);
}

#[test]
fn a_subtract_row_cuts_out_exactly_its_coverage() {
    let a = whole(1.0);
    let b = radial([0.5, 0.5], [0.2, 0.2], 0.3);
    for at in PLACES {
        let sb = run(&[b.clone()], false, false, None, at);
        let m = run(&[a.clone(), sub(b.clone())], true, false, None, at);
        assert!((m - (1.0 - sb)).abs() < 1e-15, "{at:?}");
    }
    // The subtract is not special when it comes first: the product does not care about the order.
    let first = run(&[sub(b.clone()), a.clone()], true, false, None, (0.5, 0.5));
    let last = run(&[a, sub(b)], true, false, None, (0.5, 0.5));
    assert!((first - last).abs() < 1e-15);
    // A lone subtract is the whole photo minus its coverage, as in Add; at 100 % it is the same bits.
    assert_eq!(mid(&[sub(whole(1.0))], true), 0.0);
    assert_eq!(mid(&[sub(none())], true), 1.0);
}

#[test]
fn with_every_row_0_or_1_intersect_is_and_and_add_is_or() {
    for bits in 0..8u32 {
        let rows: Vec<MaskRow> = (0..3)
            .map(|i| {
                if bits >> i & 1 == 1 {
                    whole(1.0)
                } else {
                    none()
                }
            })
            .collect();
        let ones = bits.count_ones();
        assert_eq!(
            mid(&rows, true),
            if ones == 3 { 1.0 } else { 0.0 },
            "{bits:03b}"
        );
        assert_eq!(
            mid(&rows, false),
            if ones >= 1 { 1.0 } else { 0.0 },
            "{bits:03b}"
        );
        // Each Subtract row is the complement.
        let cut: Vec<MaskRow> = rows.iter().cloned().map(sub).collect();
        assert_eq!(
            mid(&cut, true),
            if ones == 0 { 1.0 } else { 0.0 },
            "{bits:03b}"
        );
    }
}

#[test]
fn the_list_order_does_not_matter_up_to_rounding() {
    let rows = [
        radial([0.4, 0.5], [0.3, 0.2], 0.4),
        MaskRow {
            opacity: 0.7,
            ..linear([0.5, 0.5], 30.0, 0.5)
        },
        sub(MaskRow {
            opacity: 0.4,
            ..radial([0.6, 0.4], [0.2, 0.2], 0.3)
        }),
        brightness(0.1, 0.9, 0.5),
        colour([0.0, 0.0], 0.05, 0.5),
    ];
    let orders: [[usize; 5]; 4] = [
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
        [2, 0, 4, 1, 3],
        [1, 4, 0, 3, 2],
    ];
    for at in PLACES {
        let reference = run(&rows, true, false, None, at);
        for order in orders {
            let permuted: Vec<MaskRow> = order.iter().map(|&i| rows[i].clone()).collect();
            let m = run(&permuted, true, false, None, at);
            assert!(
                (m - reference).abs() < 1e-15,
                "{order:?} at {at:?}: {m} against {reference}"
            );
        }
    }
}

#[test]
fn the_node_invert_comes_after_the_product_and_no_rows_is_the_whole_photo() {
    let rows = [
        radial([0.5, 0.5], [0.3, 0.3], 0.4),
        linear([0.5, 0.5], 0.0, 0.4),
    ];
    for at in PLACES {
        let m = run(&rows, true, false, None, at);
        let inverted = run(&rows, true, true, None, at);
        assert!((inverted - (1.0 - m)).abs() < 1e-15);
    }
    assert_eq!(run(&[], true, false, None, (0.5, 0.5)), 1.0);
    assert_eq!(run(&[], true, true, None, (0.5, 0.5)), 0.0);
    // A key with no rows is the key, in both rules.
    assert_eq!(run(&[], true, false, Some(0.37), (0.5, 0.5)), 0.37);
}

#[test]
fn the_key_input_is_an_implicit_first_region_and_ranges_multiply_into_k() {
    let r = radial([0.5, 0.5], [0.3, 0.3], 0.4);
    for at in PLACES {
        let a = run(&[r.clone()], true, false, None, at);
        let m = run(&[r.clone()], true, false, Some(0.6), at);
        assert!((m - 0.6 * a).abs() < 1e-15);
    }
    // A key and a Subtract row: key (1 - a).
    let cut = run(&[sub(whole(0.5))], true, false, Some(0.8), (0.5, 0.5));
    assert!((cut - 0.8 * 0.5).abs() < 1e-15);
    // Two ranges and a region: R K1 K2.
    let b = brightness(0.2, 0.6, 0.4);
    let c = colour([0.0, 0.0], 0.05, 0.5);
    for at in PLACES {
        let r_ = run(&[r.clone()], true, false, None, at);
        let k1 = run(&[b.clone()], true, false, None, at);
        let k2 = run(&[c.clone()], true, false, None, at);
        let all = run(&[r.clone(), b.clone(), c.clone()], true, false, None, at);
        assert!((all - r_ * k1 * k2).abs() < 1e-15);
    }
}

#[test]
fn a_node_that_intersects_blends_by_the_same_rule_as_one_that_adds() {
    let rows = vec![whole(1.0)];
    let int = NodeMasks {
        rows: rows.clone(),
        invert: false,
        intersect: true,
    }
    .prepare();
    let add = NodeMasks {
        rows,
        invert: false,
        intersect: false,
    }
    .prepare();
    assert!(int.intersect() && !add.intersect());
    assert_eq!(int.blends(false), add.blends(false));
    assert!(
        !NodeMasks {
            rows: vec![],
            invert: false,
            intersect: true
        }
        .prepare()
        .blends(false)
    );
}
