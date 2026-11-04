//! The continuous inverse of the cine rendering (`docs/02` C12, *Inverse*, *Continuous pre-image*).

use ym_color::display::OutputTarget;
use ym_color::display::cine::{CineRendering, scene_from_coordinates};
use ym_color::gamut::LinearAwg4;

fn colours(n: usize) -> Vec<[f64; 3]> {
    let mut s = 7_u64;
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

#[test]
fn it_undoes_the_forward_map_on_colours_the_rendering_makes() {
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let mut worst = 0.0_f64;
    let mut kept = 0;
    for x in colours(3000) {
        let d = r.forward(LinearAwg4(x));
        let held = d.map(|v| v.clamp(1e-6, 1.0));
        let inv = r.inverse_continuous(d);
        if inv.chroma_fraction < 1.0 - 1e-9 {
            continue;
        }
        kept += 1;
        let back = r.forward(inv.scene);
        for k in 0..3 {
            worst = worst.max((back[k] - held[k]).abs());
        }
    }
    assert!(kept > 2500, "{kept}");
    assert!(worst < 2e-6, "{worst}");
}

#[test]
fn it_agrees_with_the_plain_inverse_on_ordinary_colours() {
    let r = CineRendering::new(OutputTarget::Srgb);
    for x in colours(2000) {
        let d = r.forward(LinearAwg4(x));
        let (a, b) = (r.inverse(d), r.inverse_continuous(d));
        if a.chroma_fraction == 1.0 && b.chroma_fraction == 1.0 {
            let (fa, fb) = (r.forward(a.scene), r.forward(b.scene));
            for k in 0..3 {
                assert!((fa[k] - fb[k]).abs() < 2e-6);
            }
        }
    }
}

#[test]
fn the_coordinates_are_a_pure_function_and_the_finish_is_finite_between_them() {
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let (a, b) = (
        r.forward(LinearAwg4([0.5, 0.2, 0.02])),
        r.forward(LinearAwg4([0.1, 0.4, 0.3])),
    );
    let (ca, cb) = (r.inverse_coordinates(a), r.inverse_coordinates(b));
    assert_eq!(ca, r.inverse_coordinates(a));
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let q = [0, 1, 2].map(|k| ca.coordinates[k] * (1.0 - t) + cb.coordinates[k] * t);
        assert!(scene_from_coordinates(q).0.iter().all(|v| v.is_finite()));
    }
    // Beyond the part of the limiter that can be undone, or non-finite: still finite.
    for q in [[0.0, 5.0, 5.0], [-40.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]] {
        assert!(scene_from_coordinates(q).0.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn the_pre_image_moves_continuously_along_a_line_of_display_colours() {
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let a = r.forward(LinearAwg4([0.3, 0.15, 0.05]));
    let b = r.forward(LinearAwg4([0.05, 0.3, 0.2]));
    let mut prev: Option<[f64; 3]> = None;
    for i in 0..=400 {
        let t = f64::from(i) / 400.0;
        let d = [0, 1, 2].map(|k| a[k] * (1.0 - t) + b[k] * t);
        let q = r.inverse_coordinates(d).coordinates;
        if let Some(p) = prev {
            for k in 0..3 {
                assert!((q[k] - p[k]).abs() < 0.2, "step {i} coordinate {k}");
            }
        }
        prev = Some(q);
    }
}

#[test]
fn the_fast_path_returns_the_continuation_s_pre_image_on_colours_the_rendering_makes() {
    // `inverse_coordinates` takes a short iteration from a table of the connected sheet and falls back to the
    // continuation; on the colours the rendering makes the two give the same coordinates (docs/24 section 3.8a
    // measures the lattice of a print's densities, where a few nodes differ).
    for target in [OutputTarget::Srgb, OutputTarget::DisplayP3] {
        let r = CineRendering::new(target);
        let mut worst = 0.0_f64;
        for x in colours(3000) {
            let d = r.forward(LinearAwg4(x));
            let (a, b) = (r.inverse_coordinates(d), r.inverse_coordinates_reference(d));
            assert!((a.chroma_fraction - b.chroma_fraction).abs() < 1e-6);
            for k in 0..3 {
                worst = worst.max((a.coordinates[k] - b.coordinates[k]).abs());
            }
        }
        assert!(worst < 1e-4, "{target:?}: {worst:e}");
    }
}

#[test]
fn a_start_from_a_neighbour_gives_the_same_pre_image_or_the_plain_answer() {
    let r = CineRendering::new(OutputTarget::DisplayP3);
    let mut accepted = 0;
    for x in colours(1000) {
        let d = r.forward(LinearAwg4(x));
        let near = d.map(|v| (v * 1.003).min(1.0));
        let start = r.inverse_coordinates(near);
        let a = r.inverse_coordinates_from(d, &start);
        let b = r.inverse_coordinates(d);
        let worst = (0..3)
            .map(|k| (a.coordinates[k] - b.coordinates[k]).abs())
            .fold(0.0, f64::max);
        accepted += usize::from(worst < 1e-4);
        // A start that is not accepted gives exactly the plain answer.
        assert!(worst < 1e-4 || a == b, "{worst:e}");
    }
    assert!(accepted > 900, "{accepted}");
}
