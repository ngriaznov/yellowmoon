//! `apply_fast_rows` (BL-09b): the rows of a window made alone are the same bits as those rows of `apply_fast`, for every
//! type, on a window that is a part of its frame; and the cost of the CPU run per megapixel, before and after (`--ignored`).

use std::time::Instant;

use ym_color::blur::{self, BlurKind, BlurParams, Frame, apply_fast, apply_fast_rows, reach_px};
use ym_color::consts::Vec3;

fn noise(x: usize, y: usize) -> f64 {
    let h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    let h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (h >> 40) as f64 / (1u64 << 24) as f64
}

fn picture(x0: usize, y0: usize, w: usize, h: usize) -> (Vec<Vec3>, Vec<f64>) {
    let mut v = Vec::with_capacity(w * h);
    let mut m = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x0 + x, y0 + y);
            v.push([
                0.2 + 0.4 * noise(fx, fy),
                0.25 + 0.3 * noise(fy + 7, fx + 3),
                0.3 + 0.2 * noise(fx * 3 + 1, fy * 5),
            ]);
            // a mask with a soft edge, zero on some pixels
            m.push(if fx % 29 == 0 {
                0.0
            } else {
                (fx % 7) as f64 / 6.0
            });
        }
    }
    (v, m)
}

fn kinds() -> Vec<BlurKind> {
    vec![
        BlurKind::Soft { radius: 5.0 },
        BlurKind::Soft { radius: 60.0 },
        BlurKind::Lens {
            radius: 10.0,
            blades: 0,
            rotation_deg: 0.0,
            highlights: 30.0,
        },
        BlurKind::Lens {
            radius: 70.0,
            blades: 6,
            rotation_deg: 15.0,
            highlights: 60.0,
        },
        BlurKind::Motion {
            length: 40.0,
            angle_deg: 30.0,
        },
        BlurKind::Zoom {
            amount: 25.0,
            center: [0.3, 0.6],
        },
        BlurKind::Zoom {
            amount: 100.0,
            center: [0.0, 1.0],
        },
    ]
}

#[test]
fn the_rows_made_alone_are_the_bits_of_those_rows_of_the_whole_window() {
    let (fw, fh) = (640usize, 480usize);
    let (x0, y0, w, h) = (37usize, 51usize, 301usize, 223usize);
    let (v, m) = picture(x0, y0, w, h);
    let frame = Frame {
        full_w: fw,
        full_h: fh,
        x0,
        y0,
    };
    for kind in kinds() {
        for respect in [true, false] {
            let p = BlurParams {
                kind,
                respect_mask_edges: respect,
            };
            let whole = apply_fast(&v, &m, w, h, &p, &frame).expect("the whole window");
            for rows in [0..h, 0..1, 5..6, 40..131, h - 3..h, 17..17] {
                let part =
                    apply_fast_rows(&v, &m, w, h, &p, &frame, rows.clone()).expect("the rows");
                let want = &whole[rows.start * w..rows.end * w];
                assert!(
                    part.len() == want.len()
                        && part
                            .iter()
                            .zip(want)
                            .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits)),
                    "{kind:?} respect {respect} rows {rows:?}"
                );
            }
            assert!(apply_fast_rows(&v, &m, w, h, &p, &frame, 10..h + 1).is_err());
        }
    }
    // an amount of 0 is the input, rows of it
    let p = BlurParams {
        kind: BlurKind::Soft { radius: 0.0 },
        respect_mask_edges: true,
    };
    let part = apply_fast_rows(&v, &m, w, h, &p, &frame, 3..9).expect("identity rows");
    assert_eq!(part, v[3 * w..9 * w].to_vec());
}

/// The cost of one band of a 1.0 MP made area (one thread: the band is the whole area) of the 6000 x 4000 frame, in the shape
/// `blur_plane` runs it: the window is the made area grown by the reach, cut to the frame; before = `apply_fast` over the
/// window, then the made rows kept; after = `apply_fast_rows` over the made rows.
#[test]
#[ignore = "a speed measurement: cargo test -p ym-color --test blur_rows -- --ignored --nocapture"]
fn the_cost_per_megapixel_before_and_after() {
    let (fw, fh) = (6000usize, 4000usize);
    let (mx, my, mw, mh) = (2500usize, 1500usize, 1000usize, 1000usize);
    let mp = (mw * mh) as f64 / 1e6;
    let cases: Vec<(&str, BlurKind)> = vec![
        (
            "zoom 20",
            BlurKind::Zoom {
                amount: 20.0,
                center: [0.5, 0.5],
            },
        ),
        (
            "zoom 100",
            BlurKind::Zoom {
                amount: 100.0,
                center: [0.5, 0.5],
            },
        ),
        ("soft 100", BlurKind::Soft { radius: 100.0 }),
        (
            "lens 100",
            BlurKind::Lens {
                radius: 100.0,
                blades: 6,
                rotation_deg: 15.0,
                highlights: 60.0,
            },
        ),
        (
            "lens 20",
            BlurKind::Lens {
                radius: 20.0,
                blades: 6,
                rotation_deg: 15.0,
                highlights: 60.0,
            },
        ),
        (
            "motion 100",
            BlurKind::Motion {
                length: 100.0,
                angle_deg: 30.0,
            },
        ),
    ];
    for (name, kind) in cases {
        let p = BlurParams {
            kind,
            respect_mask_edges: true,
        };
        let reach = reach_px(&p, fw, fh);
        let (wx0, wy0) = (mx.saturating_sub(reach), my.saturating_sub(reach));
        let (wx1, wy1) = ((mx + mw + reach).min(fw), (my + mh + reach).min(fh));
        let (ww, wh) = (wx1 - wx0, wy1 - wy0);
        let (v, m) = picture(wx0, wy0, ww, wh);
        let frame = Frame {
            full_w: fw,
            full_h: fh,
            x0: wx0,
            y0: wy0,
        };
        let (r0, r1) = (my - wy0, my - wy0 + mh);
        let (c0, c1) = (mx - wx0, mx - wx0 + mw);
        let t = Instant::now();
        let whole = apply_fast(&v, &m, ww, wh, &p, &frame).expect("before");
        let before = t.elapsed().as_secs_f64();
        let mut kept = Vec::with_capacity(mw * mh);
        for r in r0..r1 {
            kept.extend_from_slice(&whole[r * ww + c0..r * ww + c1]);
        }
        let t = Instant::now();
        let rows = apply_fast_rows(&v, &m, ww, wh, &p, &frame, r0..r1).expect("after");
        let after = t.elapsed().as_secs_f64();
        let mut got = Vec::with_capacity(mw * mh);
        for r in 0..mh {
            got.extend_from_slice(&rows[r * ww + c0..r * ww + c1]);
        }
        assert!(
            got.iter()
                .zip(&kept)
                .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits)),
            "{name}: not the same bits"
        );
        println!(
            "{name:12} reach {reach:4} px, window {ww}x{wh} ({:.2} MP): before {:.0} ms/MP, after {:.0} ms/MP ({:.1}x)",
            (ww * wh) as f64 / 1e6,
            before * 1e3 / mp,
            after * 1e3 / mp,
            before / after
        );
    }
    let _ = blur::cell_factor(6000.0);
}
