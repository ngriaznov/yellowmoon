//! The per-pixel return of the cine rendering's inverse (`docs/02` C12 *Per-pixel return*, F-04d): the core table,
//! a fixed number of Newton steps and the nearest-reachable clamp.

use ym_color::display::OutputTarget;
use ym_color::display::cine::{CineRendering, CoreDims, CoreTable, D_MIN, scene_from_coordinates};
use ym_color::gamut::LinearAwg4;

fn colours(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut s = seed;
    let mut u = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1_u64 << 53) as f64
    };
    (0..n)
        .map(|_| [0, 1, 2].map(|_| 0.18 * 2.0_f64.powf(-8.0 + 13.0 * u())))
        .collect()
}

/// Display colours anywhere in the unit cube (most are beyond what the rendering produces).
fn display_cube(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut s = seed;
    let mut u = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1_u64 << 53) as f64
    };
    (0..n).map(|_| [0, 1, 2].map(|_| u())).collect()
}

#[test]
fn it_undoes_the_forward_map_on_colours_the_rendering_makes() {
    let table = CoreTable::shared();
    for target in [OutputTarget::DisplayP3, OutputTarget::Srgb] {
        let r = CineRendering::new(target);
        let (mut worst, mut kept) = (0.0_f64, 0);
        let all = colours(4000, 7);
        for x in &all {
            let d = r.forward(LinearAwg4(*x));
            let held = d.map(|v| v.clamp(D_MIN, 1.0));
            let q = r.return_coordinates(table, d);
            if q.chroma_fraction < 1.0 {
                continue;
            }
            kept += 1;
            let back = r.forward(scene_from_coordinates(q.coordinates));
            for k in 0..3 {
                worst = worst.max((back[k] - held[k]).abs());
            }
        }
        assert!(kept > all.len() * 9 / 10, "{target:?}: {kept}");
        assert!(worst < 1e-6, "{target:?}: {worst:e}");
    }
}

#[test]
fn it_gives_the_continuation_s_pre_image_on_colours_the_rendering_makes() {
    let table = CoreTable::shared();
    for target in [OutputTarget::DisplayP3, OutputTarget::Srgb] {
        let r = CineRendering::new(target);
        let mut worst = 0.0_f64;
        let mut compared = 0;
        for x in colours(1500, 11) {
            let d = r.forward(LinearAwg4(x));
            let (a, b) = (
                r.return_coordinates(table, d),
                r.inverse_coordinates_reference(d),
            );
            if a.chroma_fraction < 1.0 || b.chroma_fraction < 1.0 {
                continue;
            }
            compared += 1;
            for k in 0..3 {
                worst = worst.max((a.coordinates[k] - b.coordinates[k]).abs());
            }
        }
        assert!(compared > 1200, "{target:?}: {compared}");
        // The two are the same sheet; the Jacobian is large near white, where a pre-image coordinate moves
        // faster than the display colour.
        assert!(worst < 2e-3, "{target:?}: {worst:e}");
    }
}

#[test]
fn it_is_a_pure_function_and_finite_for_any_display_colour() {
    let table = CoreTable::shared();
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let mut cases = display_cube(3000, 5);
    cases.extend([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [f64::NAN, 0.5, 0.5],
        [-3.0, 0.2, 7.0],
        [1e-9, 1e-9, 1.0],
    ]);
    for d in cases {
        let a = r.return_coordinates(table, d);
        assert_eq!(a, r.return_coordinates(table, d));
        assert!(a.coordinates.iter().all(|v| v.is_finite()), "{d:?}");
        assert!(a.chroma_fraction > 0.0 && a.chroma_fraction <= 1.0, "{d:?}");
        let scene = scene_from_coordinates(a.coordinates).0;
        assert!(scene.iter().all(|v| v.is_finite()), "{d:?}");
    }
}

#[test]
fn a_colour_beyond_the_boundary_takes_the_pre_image_of_the_boundary() {
    // Two display colours on one ray of constant luminance and hue, both beyond the reachable length, have the
    // same pre-image (the nearest reachable colour), and the answer is a colour the rendering makes.
    let table = CoreTable::shared();
    let r = CineRendering::new(OutputTarget::Srgb);
    let mut beyond = 0;
    for d in display_cube(2000, 21) {
        let q = r.return_coordinates(table, d);
        if q.chroma_fraction >= 1.0 {
            continue;
        }
        beyond += 1;
        let near = d.map(|v| (v * 0.999 + 0.0005).clamp(0.0, 1.0));
        let q2 = r.return_coordinates(table, near);
        let back = r.forward(scene_from_coordinates(q.coordinates));
        assert!(back.iter().all(|v| (0.0..=1.0).contains(v)), "{back:?}");
        if q2.chroma_fraction < 1.0 {
            // Neighbouring unreachable colours land near each other.
            let jump = (0..3)
                .map(|k| (q.coordinates[k] - q2.coordinates[k]).abs())
                .fold(0.0, f64::max);
            assert!(jump < 0.5, "{d:?}: {jump}");
        }
    }
    assert!(beyond > 100, "{beyond}");
}

#[test]
fn the_pre_image_moves_continuously_along_a_line_of_display_colours() {
    let table = CoreTable::shared();
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let a = r.forward(LinearAwg4([0.3, 0.15, 0.05]));
    let b = r.forward(LinearAwg4([0.05, 0.3, 0.2]));
    let mut prev: Option<[f64; 3]> = None;
    for i in 0..=400 {
        let t = f64::from(i) / 400.0;
        let d = [0, 1, 2].map(|k| a[k] * (1.0 - t) + b[k] * t);
        let q = r.return_coordinates(table, d).coordinates;
        if let Some(p) = prev {
            for k in 0..3 {
                assert!((q[k] - p[k]).abs() < 0.2, "step {i} coordinate {k}");
            }
        }
        prev = Some(q);
    }
}

#[test]
fn the_table_bytes_do_not_depend_on_the_thread_count() {
    let dims = CoreDims {
        ns: 12,
        nt: 16,
        nk: 8,
    };
    let one = CoreTable::build(dims, 1);
    let eight = CoreTable::build(dims, 8);
    let again = CoreTable::build(dims, 3);
    assert!(one == eight && one == again);
    assert_eq!(one.data().len(), 12 * 16 * 8);
    assert_eq!(one.ln_reach().len(), 12 * 16);
    assert!(
        one.ln_reach()
            .iter()
            .all(|r| r.is_finite() && *r <= 4.0_f32.ln()),
        "reachable lengths"
    );
    assert!(one.data().iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn zero_steps_is_the_table_alone_and_more_steps_cut_the_residual() {
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let base = CoreTable::shared().clone();
    let err = |steps: usize| {
        let t = base.clone().with_steps(steps);
        let mut worst = 0.0_f64;
        for x in colours(500, 3) {
            let d = r.forward(LinearAwg4(x));
            let c = r.display_coordinates(d);
            // Not within 2 % of the boundary of the reachable set, where the table's length is only that accurate.
            if c.ca2.hypot(c.cb2) > 0.98 * t.reachable_length(c) {
                continue;
            }
            let q = r.return_coordinates(&t, d);
            let held = d.map(|v| v.clamp(D_MIN, 1.0));
            let back = r.forward(scene_from_coordinates(q.coordinates));
            for k in 0..3 {
                worst = worst.max((back[k] - held[k]).abs());
            }
        }
        worst
    };
    let (e0, e1, e2, e5) = (err(0), err(1), err(2), err(5));
    assert!(e1 < e0 && e2 < e1 && e5 < e2, "{e0:e} {e1:e} {e2:e} {e5:e}");
    assert!(e5 < 1e-6, "{e5:e}");
}

#[test]
#[ignore = "diagnostic"]
fn worst_round_trips() {
    let table = CoreTable::shared();
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let mut rows = Vec::new();
    for x in colours(4000, 7) {
        let d = r.forward(LinearAwg4(x));
        let held = d.map(|v| v.clamp(D_MIN, 1.0));
        let q = r.return_coordinates(table, d);
        if q.chroma_fraction < 1.0 {
            continue;
        }
        let back = r.forward(scene_from_coordinates(q.coordinates));
        let e = (0..3)
            .map(|k| (back[k] - held[k]).abs())
            .fold(0.0, f64::max);
        rows.push((e, x, d));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let n_bad = rows.iter().filter(|r| r.0 > 1e-6).count();
    eprintln!("{} of {} above 1e-6", n_bad, rows.len());
    for (e, x, d) in rows.iter().take(12) {
        let c = r.display_coordinates(*d);
        let reference = r.inverse_coordinates_reference(*d);
        let q = r.return_coordinates(table, *d);
        let q0 = r.return_coordinates(&table.clone().with_steps(0), *d);
        eprintln!(
            "err {e:.2e} scene {x:.4?} display {d:.5?} y2 {:.4e} ca2 {:.3} cb2 {:.3}\n   ref {:.4?} (frac {:.3})  mine {:.4?}  start {:.4?}",
            c.y2,
            c.ca2,
            c.cb2,
            reference.coordinates,
            reference.chroma_fraction,
            q.coordinates,
            q0.coordinates
        );
        eprintln!(
            "   rho {:.4} reach {:.4}",
            c.ca2.hypot(c.cb2),
            table.reachable_length(c)
        );
    }
}

#[test]
#[ignore = "diagnostic"]
fn sweep_round_trips() {
    let r = CineRendering::new(OutputTarget::DisplayP3);
    for (ns, nt, nk) in [(40, 64, 24), (64, 96, 32), (96, 128, 48)] {
        let base = CoreTable::build(CoreDims { ns, nt, nk }, 4);
        for steps in [3, 5, 8, 12] {
            let t = base.clone().with_steps(steps);
            let (mut bad, mut n, mut worst) = (0, 0, 0.0_f64);
            let mut frac_missed = 0;
            let start = std::time::Instant::now();
            for x in colours(4000, 7) {
                let d = r.forward(LinearAwg4(x));
                let held = d.map(|v| v.clamp(D_MIN, 1.0));
                let q = r.return_coordinates(&t, d);
                if q.chroma_fraction < 1.0 {
                    frac_missed += 1;
                    continue;
                }
                let back = r.forward(scene_from_coordinates(q.coordinates));
                let e = (0..3)
                    .map(|k| (back[k] - held[k]).abs())
                    .fold(0.0, f64::max);
                n += 1;
                bad += usize::from(e > 1e-6);
                worst = worst.max(e);
            }
            eprintln!(
                "dims {ns},{nt},{nk} steps {steps}: {bad} of {n} above 1e-6, worst {worst:.2e}, {frac_missed} beyond; {:.2} us/colour",
                start.elapsed().as_secs_f64() * 1e6 / 4000.0
            );
        }
    }
}

#[test]
#[ignore = "diagnostic"]
fn beyond_vs_reference() {
    let table = CoreTable::shared();
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let (mut both, mut only_mine, mut only_ref, mut total) = (0, 0, 0, 0);
    let mut shown = 0;
    for x in colours(4000, 7) {
        let d = r.forward(LinearAwg4(x));
        let q = r.return_coordinates(table, d);
        let reference = r.inverse_coordinates_reference(d);
        total += 1;
        match (q.chroma_fraction < 1.0, reference.chroma_fraction < 1.0) {
            (true, true) => both += 1,
            (true, false) => {
                only_mine += 1;
                if shown < 8 {
                    shown += 1;
                    let c = r.display_coordinates(d);
                    eprintln!(
                        "only mine beyond: scene {x:.4?} display {d:.5?} y2 {:.3e} rho {:.4} frac {:.4}  ref q {:.4?}",
                        c.y2,
                        c.ca2.hypot(c.cb2),
                        q.chroma_fraction,
                        reference.coordinates
                    );
                }
            }
            (false, true) => only_ref += 1,
            _ => {}
        }
    }
    eprintln!(
        "of {total}: beyond in both {both}, only mine {only_mine}, only reference {only_ref}"
    );
}

#[test]
#[ignore = "diagnostic"]
fn sweep_variant_a_colours() {
    for target in [OutputTarget::DisplayP3, OutputTarget::Srgb] {
        let r = CineRendering::new(target);
        // The colours of bake_variant_a's round-trip test.
        let mut s = 11_u64;
        let mut u = move || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
        };
        let displays: Vec<[f64; 3]> = (0..50_000)
            .filter_map(|_| {
                let scene = [0, 1, 2].map(|_| 0.002 * (u() * 9.0).exp2() * (0.4 + 1.2 * u()));
                let d = r.forward(LinearAwg4(scene));
                (!d.iter().any(|v| *v < 1e-4 || *v > 0.9999)).then_some(d)
            })
            .collect();
        for (ns, nt, nk) in [(40, 64, 24), (64, 96, 32)] {
            let base = CoreTable::build(CoreDims { ns, nt, nk }, 4);
            for steps in [3, 4, 5, 6, 8] {
                let t = base.clone().with_steps(steps);
                let (mut bad, mut n, mut worst, mut beyond) = (0, 0, 0.0_f64, 0);
                for d in &displays {
                    let q = r.return_coordinates(&t, *d);
                    if q.chroma_fraction < 1.0 {
                        beyond += 1;
                        continue;
                    }
                    let back = r.forward(scene_from_coordinates(q.coordinates));
                    let e = (0..3).map(|k| (back[k] - d[k]).abs()).fold(0.0, f64::max);
                    n += 1;
                    bad += usize::from(e > 1e-6);
                    worst = worst.max(e);
                }
                eprintln!(
                    "{target:?} dims {ns},{nt},{nk} steps {steps}: {bad} of {n} above 1e-6, worst {worst:.2e}, {beyond} beyond"
                );
                if steps == 5 && ns == 40 {
                    for d in &displays {
                        let q = r.return_coordinates(&t, *d);
                        if q.chroma_fraction < 1.0 {
                            continue;
                        }
                        let back = r.forward(scene_from_coordinates(q.coordinates));
                        let e = (0..3).map(|k| (back[k] - d[k]).abs()).fold(0.0, f64::max);
                        if e > 1e-6 {
                            let c = r.display_coordinates(*d);
                            let rf = r.inverse_coordinates_reference(*d);
                            eprintln!(
                                "  bad {e:.2e}: display {d:.5?} y2 {:.4e} rho {:.4} reach {:.4}; ref {:.4?} frac {:.4} mine {:.4?} start {:.4?}",
                                c.y2,
                                c.ca2.hypot(c.cb2),
                                t.reachable_length(c),
                                rf.coordinates,
                                rf.chroma_fraction,
                                q.coordinates,
                                r.return_coordinates(&t.clone().with_steps(0), *d)
                                    .coordinates
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "diagnostic"]
fn unreachable_agreement() {
    let table = CoreTable::shared();
    for target in [OutputTarget::DisplayP3, OutputTarget::Srgb] {
        let r = CineRendering::new(target);
        let (mut n, mut both_reach, mut both_beyond, mut mine_only, mut ref_only) = (0, 0, 0, 0, 0);
        let (mut big, mut worst_frac) = (0, 0.0_f64);
        for d in display_cube(20_000, 99) {
            let (a, b) = (
                r.return_coordinates(table, d),
                r.inverse_coordinates_reference(d),
            );
            n += 1;
            match (a.chroma_fraction < 1.0, b.chroma_fraction < 1.0) {
                (false, false) => both_reach += 1,
                (true, true) => {
                    both_beyond += 1;
                    let df = (a.chroma_fraction - b.chroma_fraction).abs();
                    worst_frac = worst_frac.max(df);
                    big += usize::from(df > 0.05);
                }
                (true, false) => mine_only += 1,
                (false, true) => ref_only += 1,
            }
        }
        eprintln!(
            "{target:?}: {n} display colours: both reach {both_reach}, both beyond {both_beyond} (fraction differs by more than 0.05 for {big}, worst {worst_frac:.3}), only mine beyond {mine_only}, only the reference beyond {ref_only}"
        );
    }
}
