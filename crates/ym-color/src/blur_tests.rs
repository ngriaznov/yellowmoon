// Tests of `blur.rs` (included into its `tests` module): the property gates of `docs/33` section 8 on the oracle.

include!("blur_py_fixture.rs");

fn hash01(i: usize) -> f64 {
    ((i.wrapping_mul(2_654_435_761).wrapping_add(12345)) % 4_294_967_296) as f64 / 4_294_967_296.0
}

/// The fixture's scene (also `blur_fixture.py`): hashed values in 0.02 to 2, a 3 x 3 light of 300.
fn scene(w: usize, h: usize) -> Vec<Vec3> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut p = [0.0; 3];
            for (c, o) in p.iter_mut().enumerate() {
                let i = (y * 131 + x * 31 + c * 17 + 7) * 3 + c;
                *o = 0.02 + 1.98 * hash01(i);
            }
            if (20..23).contains(&y) && (30..33).contains(&x) {
                p = [300.0; 3];
            }
            v.push(p);
        }
    }
    v
}

/// The fixture's mask: 0 left of x = 20, a ramp to x = 44, 1 beyond; the bottom rows at half.
fn ramp_mask(w: usize, h: usize) -> Vec<f64> {
    let mut m = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut v = ((x as f64 - 20.0) / 24.0).clamp(0.0, 1.0) * if y < 40 { 1.0 } else { 0.5 };
            if x < 20 {
                v = 0.0;
            }
            if x >= 44 {
                v = 1.0;
            }
            m.push(v);
        }
    }
    m
}

fn p(kind: BlurKind) -> BlurParams {
    BlurParams {
        kind,
        respect_mask_edges: true,
    }
}

/// One of each type at full amount (and variants), the cases of the property tests.
fn kinds() -> Vec<(&'static str, BlurKind)> {
    vec![
        ("soft", BlurKind::Soft { radius: 100.0 }),
        (
            "lens hex",
            BlurKind::Lens {
                radius: 100.0,
                blades: 6,
                rotation_deg: 10.0,
                highlights: 50.0,
            },
        ),
        (
            "lens round",
            BlurKind::Lens {
                radius: 100.0,
                blades: 0,
                rotation_deg: 0.0,
                highlights: 0.0,
            },
        ),
        (
            "lens 5",
            BlurKind::Lens {
                radius: 100.0,
                blades: 5,
                rotation_deg: 20.0,
                highlights: 100.0,
            },
        ),
        (
            "motion 30",
            BlurKind::Motion {
                length: 100.0,
                angle_deg: 30.0,
            },
        ),
        (
            "motion 0",
            BlurKind::Motion {
                length: 100.0,
                angle_deg: 0.0,
            },
        ),
        (
            "motion 90",
            BlurKind::Motion {
                length: 100.0,
                angle_deg: 90.0,
            },
        ),
        (
            "zoom",
            BlurKind::Zoom {
                amount: 100.0,
                center: [0.3, 0.6],
            },
        ),
    ]
}

/// The same type with its amount at 0.
fn zeroed(kind: BlurKind) -> BlurKind {
    match kind {
        BlurKind::Soft { .. } => BlurKind::Soft { radius: 0.0 },
        BlurKind::Lens {
            blades,
            rotation_deg,
            highlights,
            ..
        } => BlurKind::Lens {
            radius: 0.0,
            blades,
            rotation_deg,
            highlights,
        },
        BlurKind::Motion { angle_deg, .. } => BlurKind::Motion {
            length: 0.0,
            angle_deg,
        },
        BlurKind::Zoom { center, .. } => BlurKind::Zoom {
            amount: 0.0,
            center,
        },
    }
}

const W: usize = 64;
const H: usize = 48;

fn run_lin(x: &[Vec3], m: &[f64], w: usize, h: usize, params: &BlurParams) -> Vec<Vec3> {
    apply_linear(x, m, w, h, params, &Frame::whole(w, h)).unwrap()
}

fn max_diff(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(p, q)| (0..3).map(move |c| (p[c] - q[c]).abs()))
        .fold(0.0, f64::max)
}

// ------------------------------------------------------------------------------------------------ identity

#[test]
fn identity_at_amount_zero_is_bit_exact() {
    let x = scene(W, H);
    let m = ramp_mask(W, H);
    for (name, k) in kinds() {
        let params = p(zeroed(k));
        let a = run_lin(&x, &m, W, H, &params);
        assert!(a.iter().zip(&x).all(|(u, v)| u == v), "{name}: linear");
        let enc: Vec<Vec3> = x.iter().map(|q| logc4_encode_rgb(*q)).collect();
        let b = apply(&enc, &m, W, H, &params, &Frame::whole(W, H)).unwrap();
        assert!(b.iter().zip(&enc).all(|(u, v)| u == v), "{name}: LogC4");
        assert!(is_identity(&params));
    }
}

#[test]
fn mask_zero_everywhere_returns_the_input_bits() {
    let x = scene(W, H);
    let enc: Vec<Vec3> = x.iter().map(|q| logc4_encode_rgb(*q)).collect();
    let zero = vec![0.0; W * H];
    for (name, k) in kinds() {
        for respect in [true, false] {
            let params = BlurParams {
                kind: k,
                respect_mask_edges: respect,
            };
            let a = run_lin(&x, &zero, W, H, &params);
            assert!(a.iter().zip(&x).all(|(u, v)| u == v), "{name} lin");
            let b = apply(&enc, &zero, W, H, &params, &Frame::whole(W, H)).unwrap();
            assert!(b.iter().zip(&enc).all(|(u, v)| u == v), "{name} log");
        }
    }
}

#[test]
fn black_huge_negative_and_subnormal_stay_finite() {
    let m = ramp_mask(W, H);
    let pictures: Vec<(&str, Vec<Vec3>)> = vec![
        ("black", vec![[0.0; 3]; W * H]),
        ("huge", vec![[1e6; 3]; W * H]),
        ("negative", scene(W, H).iter().map(|q| [-q[0], q[1], -0.5 * q[2]]).collect()),
        ("subnormal", vec![[1e-310; 3]; W * H]),
    ];
    for (name, k) in kinds() {
        for (pn, pic) in &pictures {
            for mask in [&m, &vec![0.0; W * H]] {
                for respect in [true, false] {
                    let params = BlurParams {
                        kind: k,
                        respect_mask_edges: respect,
                    };
                    let o = run_lin(pic, mask, W, H, &params);
                    assert!(o.iter().all(|q| q.iter().all(|v| v.is_finite())), "{name} {pn}");
                }
            }
        }
    }
    // A black picture stays black.
    for (name, k) in kinds() {
        let o = run_lin(&pictures[0].1, &m, W, H, &p(k));
        assert!(o.iter().all(|q| q.iter().all(|v| v.abs() < 1e-12)), "{name}");
    }
    // A bright picture with a zero mask is the input exactly.
    let o = run_lin(&pictures[1].1, &vec![0.0; W * H], W, H, &p(kinds()[0].1));
    assert!(o.iter().all(|q| *q == [1e6; 3]));
}

#[test]
fn flat_field_is_unchanged_to_the_edge_and_corner() {
    let flat = vec![[0.37, 0.2, 0.9]; W * H];
    let m = ramp_mask(W, H);
    let ones = vec![1.0; W * H];
    for (name, k) in kinds() {
        for mask in [&m, &ones] {
            for respect in [true, false] {
                let params = BlurParams {
                    kind: k,
                    respect_mask_edges: respect,
                };
                let o = run_lin(&flat, mask, W, H, &params);
                let d = max_diff(&o, &flat);
                assert!(d < 1e-9, "{name} respect {respect}: {d:e}");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ the mask

/// A hard mask (0 left of the edge, 1 right) and a picture made wild in the columns just left of the edge.
fn mask_case(vertical_edge: bool) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
    let x = scene(W, H);
    if !vertical_edge {
        // A horizontal edge: wild rows 30 to 37, the mask 1 from row 38 down (for a line along y).
        let mut y = x.clone();
        for r in 30..38 {
            for c in 0..W {
                y[r * W + c] = [1e4 * (0.5 + hash01(r * W + c)); 3];
            }
        }
        let m = (0..W * H).map(|i| if i / W >= 38 { 1.0 } else { 0.0 }).collect();
        return (x, y, m);
    }
    let mut y = x.clone();
    for r in 0..H {
        for c in 50..58 {
            y[r * W + c] = [1e4 * (0.5 + hash01(r * W + c)); 3];
        }
    }
    let m: Vec<f64> = (0..W * H).map(|i| if i % W >= 58 { 1.0 } else { 0.0 }).collect();
    (x, y, m)
}

#[test]
fn nothing_from_outside_the_mask_reaches_inside_and_the_test_bites() {
    for (name, k) in kinds() {
        let (x, y, m) = mask_case(name != "motion 90");
        let on = p(k);
        let (a, b) = (run_lin(&x, &m, W, H, &on), run_lin(&y, &m, W, H, &on));
        let mut inside = 0.0f64;
        let mut off_inside = 0.0f64;
        let off = BlurParams {
            respect_mask_edges: false,
            ..on
        };
        let (c, d) = (run_lin(&x, &m, W, H, &off), run_lin(&y, &m, W, H, &off));
        for i in (0..W * H).filter(|i| m[*i] > 0.5) {
            for ch in 0..3 {
                inside = inside.max((a[i][ch] - b[i][ch]).abs());
                off_inside = off_inside.max((c[i][ch] - d[i][ch]).abs());
            }
        }
        assert!(inside < 1e-9, "{name}: respect on moved by {inside:e}");
        assert!(off_inside > 1e-3, "{name}: respect off did not bleed ({off_inside:e})");
    }
}

#[test]
fn a_mask_at_half_weighs_half_and_the_blend_is_linear_in_m() {
    let x = scene(W, H);
    for (name, k) in kinds() {
        let params = p(k);
        let one = run_lin(&x, &vec![1.0; W * H], W, H, &params);
        let half = run_lin(&x, &vec![0.5; W * H], W, H, &params);
        let want: Vec<Vec3> = x
            .iter()
            .zip(&one)
            .map(|(a, b)| [0, 1, 2].map(|c| a[c] + 0.5 * (b[c] - a[c])))
            .collect();
        assert!(max_diff(&half, &want) < 1e-9, "{name}");
    }
}

#[test]
fn the_result_is_continuous_in_m() {
    let x = scene(W, H);
    for (name, k) in kinds() {
        let params = p(k);
        // A sweep of a uniform mask: the result moves by no more than the step times the range.
        let a = run_lin(&x, &vec![0.4; W * H], W, H, &params);
        let b = run_lin(&x, &vec![0.4 + 1e-6; W * H], W, H, &params);
        let range = max_diff(&run_lin(&x, &vec![1.0; W * H], W, H, &params), &x);
        assert!(max_diff(&a, &b) <= 1e-6 * range * 1.001 + 1e-12, "{name}");
        // One pixel with a tiny mask and no other weight in reach: a tiny change, never a jump.
        for tiny in [1e-300, 1e-100, 1e-15, 1e-9, 1e-6] {
            let mut m = vec![0.0; W * H];
            let c = 25 * W + 20;
            m[c] = tiny;
            let o = run_lin(&x, &m, W, H, &params);
            assert!(o.iter().all(|q| q.iter().all(|v| v.is_finite())), "{name} {tiny}");
            let d = (0..3).map(|ch| (o[c][ch] - x[c][ch]).abs()).fold(0.0, f64::max);
            assert!(d <= tiny * 600.0 + 1e-12, "{name} {tiny:e}: {d:e}");
        }
    }
}

#[test]
fn a_mask_with_opacity_counts_in_the_weights() {
    // Opacity 0.5 of a hard mask is a mask of 0.5 inside: still no light from where it is 0.
    let (x, y, m) = mask_case(true);
    let m: Vec<f64> = m.iter().map(|v| v * 0.5).collect();
    let on = p(kinds()[1].1);
    let (a, b) = (run_lin(&x, &m, W, H, &on), run_lin(&y, &m, W, H, &on));
    let d = (0..W * H)
        .filter(|i| m[*i] > 0.0)
        .map(|i| (0..3).map(|c| (a[i][c] - b[i][c]).abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    assert!(d < 1e-9);
}

#[test]
fn log_wrapper_has_the_exact_branches() {
    let x = scene(W, H);
    let enc: Vec<Vec3> = x.iter().map(|q| logc4_encode_rgb(*q)).collect();
    let dec: Vec<Vec3> = enc.iter().map(|q| logc4_decode_rgb(*q)).collect();
    let m = ramp_mask(W, H);
    for (name, k) in kinds() {
        let params = p(k);
        let out = apply(&enc, &m, W, H, &params, &Frame::whole(W, H)).unwrap();
        let (n, d) = sums(&dec, &m, W, H, &params, &Frame::whole(W, H)).unwrap();
        for i in 0..W * H {
            if m[i] == 0.0 {
                assert_eq!(out[i], enc[i], "{name} m=0 keeps the bits");
            } else if m[i] == 1.0 {
                let q = d[i] + EPS;
                let want = logc4_encode_rgb([n[i][0] / q, n[i][1] / q, n[i][2] / q]);
                assert_eq!(out[i], want, "{name} m=1");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ properties

#[test]
fn mirror_symmetry() {
    let x = scene(W, H);
    let m = ramp_mask(W, H);
    let flip_h = |v: &[Vec3]| -> Vec<Vec3> {
        (0..W * H).map(|i| v[(i / W) * W + (W - 1 - i % W)]).collect()
    };
    let flip_v = |v: &[Vec3]| -> Vec<Vec3> {
        (0..W * H).map(|i| v[(H - 1 - i / W) * W + i % W]).collect()
    };
    let flip_hm = |v: &[f64]| -> Vec<f64> { (0..W * H).map(|i| v[(i / W) * W + (W - 1 - i % W)]).collect() };
    let flip_vm = |v: &[f64]| -> Vec<f64> { (0..W * H).map(|i| v[(H - 1 - i / W) * W + i % W]).collect() };
    for (name, k) in kinds() {
        // Soft, round Lens (no Highlights needed: the weight is a function of the pixel), Motion at 0 and 90.
        let (horizontal, vertical) = match name {
            "soft" | "lens round" => (true, true),
            "motion 0" => (true, false),
            "motion 90" => (false, true),
            _ => continue,
        };
        let a = run_lin(&x, &m, W, H, &p(k));
        if horizontal {
            let b = flip_h(&run_lin(&flip_h(&x), &flip_hm(&m), W, H, &p(k)));
            assert!(max_diff(&a, &b) < 1e-9, "{name} horizontal");
        }
        if vertical {
            let b = flip_v(&run_lin(&flip_v(&x), &flip_vm(&m), W, H, &p(k)));
            assert!(max_diff(&a, &b) < 1e-9, "{name} vertical");
        }
    }
}

#[test]
fn energy_of_a_point_in_the_interior() {
    let (w, h) = (128, 96);
    let mut x = vec![[0.0; 3]; w * h];
    for r in 0..h {
        for c in 0..w {
            let d2 = ((r as f64 - 48.0).powi(2) + (c as f64 - 64.0).powi(2)) / (2.0 * 9.0);
            x[r * w + c] = [1000.0 * (-d2).exp(); 3];
        }
    }
    let total: f64 = x.iter().map(|q| q[0]).sum();
    let ones = vec![1.0; w * h];
    let cases = [
        ("soft", BlurKind::Soft { radius: 100.0 }, 1e-9),
        (
            "lens",
            BlurKind::Lens {
                radius: 100.0,
                blades: 0,
                rotation_deg: 0.0,
                highlights: 0.0,
            },
            1e-9,
        ),
        (
            "motion",
            BlurKind::Motion {
                length: 100.0,
                angle_deg: 45.0,
            },
            1e-9,
        ),
        (
            "zoom",
            BlurKind::Zoom {
                amount: 100.0,
                center: [0.5, 0.5],
            },
            2e-2,
        ),
    ];
    for (name, k, bound) in cases {
        let o = run_lin(&x, &ones, w, h, &p(k));
        let e: f64 = o.iter().map(|q| q[0]).sum();
        let err = (e / total - 1.0).abs();
        assert!(err < bound, "{name}: {err:e}");
        eprintln!("energy {name}: {err:e}");
    }
}

#[test]
fn repeat_render_is_identical() {
    let x = scene(W, H);
    let m = ramp_mask(W, H);
    for (name, k) in kinds() {
        let a = run_lin(&x, &m, W, H, &p(k));
        let b = run_lin(&x, &m, W, H, &p(k));
        assert!(a == b, "{name}");
    }
}

#[test]
fn a_window_with_its_halo_equals_the_frame_bit_for_bit() {
    let (fw, fh) = (150usize, 110usize);
    let x = scene(fw, fh);
    let m: Vec<f64> = (0..fw * fh)
        .map(|i| (((i % fw) as f64 / fw as f64) * 1.4 - 0.2).clamp(0.0, 1.0))
        .collect();
    let margin = 16usize;
    for (name, k) in kinds() {
        // Smaller amounts, so the reach fits the margin (long edge 150: R = 6 x 0.04 x 150 / 100 at 40).
        let k = match k {
            BlurKind::Soft { .. } => BlurKind::Soft { radius: 40.0 },
            BlurKind::Lens {
                blades,
                rotation_deg,
                highlights,
                ..
            } => BlurKind::Lens {
                radius: 40.0,
                blades,
                rotation_deg,
                highlights,
            },
            BlurKind::Motion { angle_deg, .. } => BlurKind::Motion {
                length: 40.0,
                angle_deg,
            },
            z => z,
        };
        let params = p(k);
        let whole = apply_linear(&x, &m, fw, fh, &params, &Frame::whole(fw, fh)).unwrap();
        for (tw, th) in [(97usize, 61usize), (160, 110), (64, 64)] {
            let mut ty = 0;
            while ty < fh {
                let mut tx = 0;
                while tx < fw {
                    let (x1, y1) = ((tx + tw).min(fw), (ty + th).min(fh));
                    let (wx0, wy0) = (tx.saturating_sub(margin), ty.saturating_sub(margin));
                    let (wx1, wy1) = ((x1 + margin).min(fw), (y1 + margin).min(fh));
                    let (ww, wh) = (wx1 - wx0, wy1 - wy0);
                    let mut wx = Vec::with_capacity(ww * wh);
                    let mut wm = Vec::with_capacity(ww * wh);
                    for r in wy0..wy1 {
                        for c in wx0..wx1 {
                            wx.push(x[r * fw + c]);
                            wm.push(m[r * fw + c]);
                        }
                    }
                    let frame = Frame {
                        full_w: fw,
                        full_h: fh,
                        x0: wx0,
                        y0: wy0,
                    };
                    let part = apply_linear(&wx, &wm, ww, wh, &params, &frame).unwrap();
                    for r in ty..y1 {
                        for c in tx..x1 {
                            assert_eq!(
                                part[(r - wy0) * ww + (c - wx0)],
                                whole[r * fw + c],
                                "{name} tile {tw}x{th} at ({tx}, {ty}), pixel ({c}, {r})"
                            );
                        }
                    }
                    tx += tw;
                }
                ty += th;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ formulas

#[test]
fn erf_matches_known_values() {
    for (x, want) in [
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (3.0, 0.999_977_909_503_001_4),
        (-1.0, -0.842_700_792_949_714_9),
    ] {
        assert!((erf(x) - want).abs() < 1e-12, "erf({x})");
    }
}

#[test]
fn gaussian_weights_follow_the_pixel_integral() {
    for sigma in [0.4, 1.0, 2.56, 7.3] {
        let g = gauss_weights(sigma);
        let half = (SIGMA_CUT * sigma).ceil() as i64;
        assert_eq!(g.len() as i64, 2 * half + 1);
        let phi = |v: f64| 0.5 * (1.0 + erf(v / (sigma * std::f64::consts::SQRT_2)));
        let raw: Vec<f64> = (-half..=half)
            .map(|i| phi(i as f64 + 0.5) - phi(i as f64 - 0.5))
            .collect();
        let total: f64 = raw.iter().sum();
        assert!((total - 1.0).abs() < 6.5e-5, "the cut leaves 6e-5 of the mass");
        for (a, b) in g.iter().zip(&raw) {
            assert!((a - b / total).abs() < 1e-12);
        }
        assert!((g.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for i in 0..g.len() {
            assert!((g[i] - g[g.len() - 1 - i]).abs() < 1e-15);
        }
    }
    assert_eq!(gauss_weights(0.0), vec![1.0]);
    let tiny = gauss_weights(1e-3);
    assert!((tiny[tiny.len() / 2] - 1.0).abs() < 1e-12, "sigma to 0 tends to the identity");
}

#[test]
fn aperture_coverage_matches_the_analytic_areas() {
    // A square aligned with the pixels: exact (coverage 1 on 11 x 11 pixels, 0 elsewhere).
    let r = 5.5 * 2f64.sqrt();
    let (half, cov) = aperture_coverage(r, 4, -45.0);
    let side = 2 * half + 1;
    for dy in -half..=half {
        for dx in -half..=half {
            let want = if dx.abs() <= 5 && dy.abs() <= 5 { 1.0 } else { 0.0 };
            assert!((cov[((dy + half) * side + dx + half) as usize] - want).abs() < 1e-12, "({dx}, {dy})");
        }
    }
    let t = aperture_table(r, 4, -45.0);
    assert!((t.at(5, -5) - 1.0 / 121.0).abs() < 1e-12 && t.at(6, 0) == 0.0);
    // A disc and a hexagon against their areas: the coverage is exact (BL-07b), so to rounding.
    let area = |blades: u32, r: f64, rot: f64| aperture_coverage(r, blades, rot).1.iter().sum::<f64>();
    let disc = area(0, 12.0, 0.0);
    let hex = area(6, 12.0, 7.0);
    let e_disc = (disc / (std::f64::consts::PI * 144.0) - 1.0).abs();
    let e_hex = (hex / (1.5 * 3f64.sqrt() * 144.0) - 1.0).abs();
    eprintln!("coverage area error: disc {e_disc:e}, hexagon {e_hex:e}");
    assert!(e_disc < 1e-9 && e_hex < 1e-9, "{e_disc:e} {e_hex:e}");
    // A disc is symmetric under the pixel grid's mirrors.
    let t = aperture_table(6.3, 0, 0.0);
    assert_eq!(t.at(3, -2), t.at(-3, 2));
    assert_eq!(t.at(3, -2), t.at(2, 3));
}

#[test]
fn line_weights_are_the_length_inside_each_pixel() {
    let t = line_table(4.0, 0.0);
    for (dx, want) in [(-3, 0.0), (-2, 0.125), (-1, 0.25), (0, 0.25), (1, 0.25), (2, 0.125), (3, 0.0)] {
        assert!((t.at(dx, 0) - want).abs() < 1e-12, "L = 4 at {dx}");
        assert!(t.at(dx, 1).abs() < 1e-12);
    }
    let t = line_table(5.0, 0.0);
    for dx in -2..=2 {
        assert!((t.at(dx, 0) - 0.2).abs() < 1e-12);
    }
    assert_eq!(t.at(3, 0), 0.0);
    // 45 degrees counter-clockwise on screen: up and to the right is y negative.
    let t = line_table(3.0 * 2f64.sqrt(), 45.0);
    for k in -1..=1 {
        assert!((t.at(k, -k) - 1.0 / 3.0).abs() < 1e-12, "diagonal {k}");
    }
    assert!(t.at(1, 1) < 1e-12);
    // The total of an unnormalised table is the length: the table is normalised to 1.
    for (l, a) in [(7.3, 33.0), (20.0, 190.0)] {
        let t = line_table(l, a);
        assert!((t.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn highlight_weight_at_two_three_and_four_stops() {
    let at = |ev: f64, h: f64| highlight_weight([0.18 * 2f64.powf(ev); 3], h);
    assert!((at(2.0, 100.0) - 1.0).abs() < 1e-9);
    assert!((at(3.0, 100.0) - 8.5).abs() < 1e-9);
    assert!((at(4.0, 100.0) - 16.0).abs() < 1e-9);
    assert!((at(6.0, 100.0) - 16.0).abs() < 1e-9);
    assert!((at(0.0, 100.0) - 1.0).abs() < 1e-12);
    assert_eq!(at(5.0, 0.0), 1.0);
    assert!((at(3.0, 50.0) - 4.75).abs() < 1e-9);
    assert!(highlight_weight([0.0; 3], 100.0).is_finite());
}

#[test]
fn units_follow_the_spec() {
    assert!((radius_px(100.0, 6000.0) - 240.0).abs() < 1e-12);
    assert!((radius_px(50.0, 8600.0) - 172.0).abs() < 1e-12);
    assert!((length_px(100.0, 6000.0) - 480.0).abs() < 1e-12);
    assert!((zoom_span(80.0) - 0.08).abs() < 1e-15);
    assert_eq!(zoom_taps(1.0), 64);
    assert_eq!(zoom_taps(100.0), 200);
}

#[test]
fn zoom_matches_the_identity_at_the_centre_and_a_flat_field_exactly() {
    let x = scene(W, H);
    let ones = vec![1.0; W * H];
    let k = BlurKind::Zoom {
        amount: 100.0,
        center: [0.5, 0.5],
    };
    // At the centre the line is as short as a pixel's own offset: the pixel is nearly unchanged.
    let o = run_lin(&x, &ones, W, H, &p(k));
    let c = 24 * W + 32;
    assert!(o[c].iter().zip(&x[c]).all(|(a, b)| (a - b).abs() < 0.5), "centre");
    let flat = vec![[0.5; 3]; W * H];
    let o = run_lin(&flat, &ones, W, H, &p(k));
    assert!(max_diff(&o, &flat) < 1e-12);
}

#[test]
fn errors_are_reported_not_panicked() {
    let x = scene(W, H);
    let m = ramp_mask(W, H);
    let k = p(kinds()[0].1);
    assert!(matches!(
        apply_linear(&x[1..], &m, W, H, &k, &Frame::whole(W, H)),
        Err(BlurError::Size { .. })
    ));
    assert!(matches!(
        apply_linear(&x, &m, W, H, &k, &Frame::whole(W - 1, H)),
        Err(BlurError::Window { .. })
    ));
    let bad = p(BlurKind::Soft { radius: f64::NAN });
    assert_eq!(
        apply_linear(&x, &m, W, H, &bad, &Frame::whole(W, H)),
        Err(BlurError::NotFinite("radius"))
    );
    let msg = BlurError::NotFinite("radius").to_string();
    assert!(msg.contains("radius"));
}

// ------------------------------------------------------------------------------------------------ the sketch

#[test]
fn agrees_with_the_numpy_sketch() {
    let x = scene(W, H);
    let m = ramp_mask(W, H);
    let cases: [BlurParams; 8] = [
        p(BlurKind::Soft { radius: 100.0 }),
        BlurParams {
            kind: BlurKind::Soft { radius: 100.0 },
            respect_mask_edges: false,
        },
        p(BlurKind::Lens {
            radius: 100.0,
            blades: 0,
            rotation_deg: 0.0,
            highlights: 0.0,
        }),
        p(BlurKind::Lens {
            radius: 100.0,
            blades: 6,
            rotation_deg: 10.0,
            highlights: 50.0,
        }),
        p(BlurKind::Lens {
            radius: 100.0,
            blades: 5,
            rotation_deg: 20.0,
            highlights: 0.0,
        }),
        p(BlurKind::Motion {
            length: 100.0,
            angle_deg: 30.0,
        }),
        p(BlurKind::Motion {
            length: 100.0,
            angle_deg: 0.0,
        }),
        p(BlurKind::Zoom {
            amount: 100.0,
            center: [0.3, 0.6],
        }),
    ];
    // Soft, Lens and Zoom are the same maths (1e-9 absolute on values up to 300, the sketch's FFT); Motion differs by the
    // sketch's quadrature of the line (512 samples a pixel) against the exact clip.
    for (i, sum, pix) in PY_FIXTURE.iter() {
        let o = run_lin(&x, &m, W, H, &cases[*i]);
        let motion = matches!(cases[*i].kind, BlurKind::Motion { .. });
        // The sketch's aperture coverage is the exact pixel area, as the oracle's (`docs/33` 11 item 81), so Lens is as
        // tight as Soft and Zoom.
        let tol = if motion { 2e-3 } else { 1e-9 };
        let mut worst = 0.0f64;
        for (y, xx, want) in pix {
            for c in 0..3 {
                let got = o[y * W + xx][c];
                let rel = (got - want[c]).abs() / want[c].abs().max(1.0);
                worst = worst.max(rel);
            }
        }
        let total: f64 = o.iter().map(|q| q[0] + q[1] + q[2]).sum();
        let sum_rel = (total - sum).abs() / sum;
        eprintln!("sketch case {i}: pixel {worst:e}, sum {sum_rel:e}");
        assert!(worst < tol, "case {i}: {worst:e}");
        assert!(
            sum_rel < if motion { 2e-3 } else { 1e-9 },
            "case {i} sum: {sum_rel:e}"
        );
    }
}
