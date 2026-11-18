//! The reach of the Blur node (`docs/33` section 3.8, BL-04): the table's numbers, the reach against what the exact kernels read,
//! and the oracle on a window of the reach equals the whole frame bit for bit (tile invariance of the framework).

use ym_color::blur::{
    self, BlurKind, BlurParams, Frame, aperture_table, cell_factor, gauss_weights, line_table,
    reach_px,
};
use ym_color::consts::Vec3;

fn p(kind: BlurKind) -> BlurParams {
    BlurParams {
        kind,
        respect_mask_edges: true,
    }
}

#[test]
fn the_cell_factor_and_the_reach_table_of_the_spec() {
    // 32 at 6000 px and 64 at 8600 px.
    assert_eq!(cell_factor(6000.0), 32);
    assert_eq!(cell_factor(8600.0), 64);
    assert_eq!(cell_factor(100.0), 1);
    assert_eq!(cell_factor(f64::NAN), 1);
    // Amount 100 on 6000 px: Soft 480 + 3 x 32, Lens 240 + 96, Motion 240 + 1 + 96 (the table, section 3.8).
    assert_eq!(
        reach_px(&p(BlurKind::Soft { radius: 100.0 }), 6000, 4000),
        576
    );
    assert_eq!(
        reach_px(
            &p(BlurKind::Lens {
                radius: 100.0,
                blades: 6,
                rotation_deg: 0.0,
                highlights: 0.0
            }),
            6000,
            4000
        ),
        336
    );
    assert_eq!(
        reach_px(
            &p(BlurKind::Motion {
                length: 100.0,
                angle_deg: 0.0
            }),
            6000,
            4000
        ),
        337
    );
    // Zoom with the centre in a corner: 0.05 x the diagonal (7211) + 32.
    let z = reach_px(
        &p(BlurKind::Zoom {
            amount: 100.0,
            center: [0.0, 0.0],
        }),
        6000,
        4000,
    );
    assert_eq!(z, (0.05 * (6000f64).hypot(4000.0)).ceil() as usize + 32);
    assert!((392..=394).contains(&z), "{z}");
    // 8600 px: Soft 688 + 192 = 880.
    assert_eq!(
        reach_px(&p(BlurKind::Soft { radius: 100.0 }), 8600, 5733),
        880
    );
    // A node at its identity reads nothing, and a number that is not finite reads nothing.
    assert_eq!(reach_px(&p(BlurKind::Soft { radius: 0.0 }), 6000, 4000), 0);
    assert_eq!(
        reach_px(&p(BlurKind::Soft { radius: f64::NAN }), 6000, 4000),
        0
    );
}

#[test]
fn the_reach_holds_what_the_exact_kernels_read() {
    for (w, h) in [(97usize, 61usize), (640, 420), (3000, 2000), (6000, 4000)] {
        let u = w.max(h) as f64;
        for amount in [0.5, 1.0, 7.0, 33.0, 100.0] {
            let soft = reach_px(&p(BlurKind::Soft { radius: amount }), w, h);
            let half = gauss_weights(blur::radius_px(amount, u) / 2.0).len() / 2;
            assert!(
                soft >= half,
                "soft {soft} against {half} at {w} x {h}, {amount}"
            );
            for blades in [0, 3, 6, 12] {
                let lens = reach_px(
                    &p(BlurKind::Lens {
                        radius: amount,
                        blades,
                        rotation_deg: 17.0,
                        highlights: 0.0,
                    }),
                    w,
                    h,
                );
                let half = aperture_table(blur::radius_px(amount, u), blades, 17.0).half as usize;
                assert!(lens >= half, "lens {lens} against {half}");
            }
            let motion = reach_px(
                &p(BlurKind::Motion {
                    length: amount,
                    angle_deg: 33.0,
                }),
                w,
                h,
            );
            let half = line_table(blur::length_px(amount, u), 33.0).half as usize;
            assert!(motion >= half, "motion {motion} against {half}");
        }
    }
}

/// Which pixels a window of the reach is exact for: the oracle over a window grown by the reach equals the oracle over the frame
/// on the pixels of the window's core, bit for bit, for each type.
#[test]
fn a_window_grown_by_the_reach_is_the_frame_bit_for_bit() {
    let (w, h) = (90usize, 70usize);
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let v: Vec<Vec3> = (0..w * h)
        .map(|_| [next() * 0.8, next() * 0.8, next() * 0.8])
        .collect();
    let m: Vec<f64> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if x < 20 {
                0.0
            } else if x < 40 {
                (x - 20) as f64 / 20.0
            } else if y > 30 {
                1.0
            } else {
                0.3
            }
        })
        .collect();
    let kinds = [
        BlurKind::Soft { radius: 60.0 },
        BlurKind::Lens {
            radius: 70.0,
            blades: 6,
            rotation_deg: 10.0,
            highlights: 50.0,
        },
        BlurKind::Lens {
            radius: 70.0,
            blades: 0,
            rotation_deg: 0.0,
            highlights: 0.0,
        },
        BlurKind::Motion {
            length: 80.0,
            angle_deg: 25.0,
        },
        BlurKind::Zoom {
            amount: 90.0,
            center: [0.3, 0.6],
        },
    ];
    for kind in kinds {
        for respect in [true, false] {
            let params = BlurParams {
                kind,
                respect_mask_edges: respect,
            };
            let whole = blur::apply(&v, &m, w, h, &params, &Frame::whole(w, h)).expect("whole");
            let reach = reach_px(&params, w, h);
            for core in [
                (10usize, 8usize, 25usize, 19usize),
                (0, 0, 31, 31),
                (55, 40, 35, 30),
            ] {
                let (cx, cy, cw, ch) = core;
                let (x0, y0) = (cx.saturating_sub(reach), cy.saturating_sub(reach));
                let (x1, y1) = ((cx + cw + reach).min(w), (cy + ch + reach).min(h));
                let (ww, wh) = (x1 - x0, y1 - y0);
                let sub_v: Vec<Vec3> = (y0..y1)
                    .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                    .map(|(x, y)| v[y * w + x])
                    .collect();
                let sub_m: Vec<f64> = (y0..y1)
                    .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                    .map(|(x, y)| m[y * w + x])
                    .collect();
                let frame = Frame {
                    full_w: w,
                    full_h: h,
                    x0,
                    y0,
                };
                let got = blur::apply(&sub_v, &sub_m, ww, wh, &params, &frame).expect("window");
                for y in cy..(cy + ch).min(h) {
                    for x in cx..(cx + cw).min(w) {
                        let a = got[(y - y0) * ww + (x - x0)];
                        let b = whole[y * w + x];
                        assert!(
                            a.map(f64::to_bits) == b.map(f64::to_bits),
                            "{kind:?} respect {respect} core {core:?} at ({x}, {y}): {a:?} against {b:?}, reach {reach}"
                        );
                    }
                }
            }
        }
    }
}
