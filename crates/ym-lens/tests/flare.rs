//! L3, the highlight optics (`docs/21` section 4, L3, and the tests of section 14).
//!
//! The energy tests use lenses edited from the synthetic folder so that the parts can be told apart: each part
//! on its own, a round aperture where the exact energy is checked, and flares that stay inside the frame.

mod common;

use std::fs;
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{apply_tile, camera, pattern, tiles, whole};
use serde_json::{Value, json};
use ym_lens::consts::GLOW_BOOST_SHARE;
use ym_lens::test_support::{coating, write_sums, write_synthetic_lenses};
use ym_lens::{
    Axis, FrameInfo, LensAmounts, LensError, LensKind, LensLibrary, LensPlan, LensPlanes,
    PixelRect, apply, light_boost_ev,
};

// ---------------------------------------------------------------------------------------------------------
// Lenses

fn ghost_json(m: f64, n: f64, s: f64, clip: f64, fraction: f64, nm: f64) -> Value {
    json!({ "m": m, "n": n, "s": s, "clip_radius": clip, "fraction": fraction, "soft": 0.1,
            "coating": coating(nm) })
}

/// A lens folder: the synthetic one with `lens_edit` applied to every `lens.json` and `att_edit` to
/// `anamorphic.json`.
fn library_with(lens_edit: &dyn Fn(&mut Value), att_edit: &dyn Fn(&mut Value)) -> LensLibrary {
    let dir = tempfile::tempdir().expect("temp dir");
    write_synthetic_lenses(dir.path()).expect("write the synthetic lenses");
    for kind in LensKind::ALL {
        let path = dir.path().join(kind.as_str()).join("lens.json");
        let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        lens_edit(&mut v);
        fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }
    let path = dir.path().join("anamorphic.json");
    let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    att_edit(&mut v);
    fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    write_sums(dir.path()).unwrap();
    LensLibrary::from_dir(dir.path()).expect("the edited lenses load")
}

/// Which parts of the flare a test lens has.
#[derive(Clone, Copy)]
struct Parts {
    glow: bool,
    veil: bool,
    ghosts: bool,
    burst: bool,
    streak: bool,
    att_ghosts: bool,
    /// Blades of the aperture (0: round).
    blades: u32,
}

impl Parts {
    const NONE: Parts = Parts {
        glow: false,
        veil: false,
        ghosts: false,
        burst: false,
        streak: false,
        att_ghosts: false,
        blades: 0,
    };
    const ALL: Parts = Parts {
        glow: true,
        veil: true,
        ghosts: true,
        burst: true,
        streak: true,
        att_ghosts: true,
        blades: 0,
    };
}

/// A lens whose flares stay inside a 480 x 320 frame (reaches of 0.3 half-diagonals) and have the parts of
/// `p`. Ghosts are not cut by their clipping circles.
fn parts_library(p: Parts) -> LensLibrary {
    let on = |b: bool, v: f64| if b { v } else { 0.0 };
    library_with(
        &|lens| {
            lens["flare"]["glow"] = json!({ "fraction": on(p.glow, 0.15), "core_radius": 0.0015,
                "r0": 0.01, "power": 2.0, "reach": 0.3,
                "chroma": [1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8] });
            lens["flare"]["veil"] = json!({ "fraction": on(p.veil, 0.10), "radius": 0.3 });
            lens["flare"]["aperture"] =
                json!({ "blades": p.blades, "rotation_deg": 10.0, "roundness": 0.5 });
            lens["flare"]["ghosts"] = if p.ghosts {
                json!([
                    ghost_json(-0.5, 0.3, 0.04, 3.0, 0.10, 450.0),
                    ghost_json(0.7, -0.2, 0.06, 3.0, 0.08, 600.0),
                    ghost_json(1.4, 0.0, 0.03, 3.0, 0.06, 520.0),
                ])
            } else {
                json!([])
            };
            lens["flare"]["starburst"] =
                json!({ "fraction": on(p.burst, 0.10), "radius": 0.06, "f_number_full": 11.0 });
        },
        &|att| {
            att["streak"] = json!({ "fraction": on(p.streak, 0.20), "reach": 0.3, "core": 0.002,
                "power": 1.5, "tint": coating(470.0) });
            att["ghosts"] = if p.att_ghosts {
                json!([
                    ghost_json(-0.4, 0.2, 0.05, 3.0, 0.06, 470.0),
                    ghost_json(0.9, 0.1, 0.04, 3.0, 0.04, 560.0),
                ])
            } else {
                json!([])
            };
        },
    )
}

// ---------------------------------------------------------------------------------------------------------
// Plans and scenes

#[allow(clippy::too_many_arguments)]
fn plan_of(
    lib: &LensLibrary,
    flares: f64,
    edges: f64,
    anamorphic: f64,
    (w, h): (usize, usize),
    axis: Axis,
    f_number: Option<f64>,
    clip: f32,
) -> LensPlan {
    let amounts = LensAmounts {
        kind: LensKind::Classic,
        flares,
        edges,
        anamorphic,
    };
    let frame = FrameInfo {
        width: w,
        height: h,
        axis,
        f_number,
        clip: [clip; 3],
    };
    LensPlan::new(lib, &amounts, frame, camera()).expect("plan")
}

/// A planar scene of black with lights.
struct Scene {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

impl Scene {
    fn new(w: usize, h: usize) -> Self {
        Scene {
            w,
            h,
            data: vec![0.0; 3 * w * h],
        }
    }

    fn fill(&mut self, x: usize, y: usize, lw: usize, lh: usize, rgb: [f32; 3]) {
        for (c, v) in rgb.iter().enumerate() {
            for yy in y..y + lh {
                for xx in x..x + lw {
                    self.data[c * self.w * self.h + yy * self.w + xx] = *v;
                }
            }
        }
    }

    /// The scene with x and y swapped (for the anamorphic axis along y).
    fn transposed(&self) -> Scene {
        let mut t = Scene::new(self.h, self.w);
        for c in 0..3 {
            for y in 0..self.h {
                for x in 0..self.w {
                    t.data[c * self.w * self.h + x * self.h + y] =
                        self.data[c * self.w * self.h + y * self.w + x];
                }
            }
        }
        t
    }

    fn rect(&self) -> PixelRect {
        PixelRect::new(0, 0, self.w, self.h)
    }
}

/// The standard 480 x 320 scene of the energy tests: a textured patch of mid tones, a white clipped 3 x 3 light
/// (A), a coloured unclipped pixel (B) and a white clipped pixel (C). Everything lies well inside the frame.
fn energy_scene() -> Scene {
    let mut s = Scene::new(480, 320);
    // A textured patch of mid tones (every pixel differs, so the output's f32 rounding averages out).
    for c in 0..3 {
        for y in 140..180 {
            for x in 150..200 {
                s.data[c * 480 * 320 + y * 480 + x] = 0.4 * pattern(x, y, c);
            }
        }
    }
    s.fill(250, 170, 3, 3, [1.0; 3]);
    s.fill(300, 120, 1, 1, [1.0, 0.7, 0.2]);
    s.fill(200, 200, 1, 1, [1.0; 3]);
    s
}

/// The planes of `plan` from a scene that is the whole frame (the frames here are under 1024 px, so the lens
/// plane is the frame itself).
fn planes_of(plan: &LensPlan, scene: &Scene) -> LensPlanes {
    LensPlanes::build(plan, &whole(scene.w, scene.h, &scene.data), None, &|| false).expect("planes")
}

fn run(plan: &LensPlan, planes: &LensPlanes, scene: &Scene) -> Vec<f32> {
    apply(
        plan,
        planes,
        1,
        &whole(scene.w, scene.h, &scene.data),
        scene.rect(),
    )
    .expect("apply")
}

// ---------------------------------------------------------------------------------------------------------
// The light the scene holds

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The extra headroom in EV of every pixel of `scene` (the lens plane: the frames of these tests are under
/// 1024 px), made the way `l3::lights` does, independently: the 8-connected patches of pixels above 0.9 of the
/// clip in every channel, each boosted by `light_boost_ev` of the diameter of the disc of its area, in
/// half-diagonals.
fn boost_map(scene: &Scene, clip: f64) -> Vec<f64> {
    let (w, h) = (scene.w, scene.h);
    let n = w * h;
    let hd = 0.5 * ((w * w + h * h) as f64).sqrt();
    let clipped: Vec<bool> = (0..n)
        .map(|i| {
            (0..3)
                .map(|c| f64::from(scene.data[c * n + i]) / clip)
                .fold(f64::INFINITY, f64::min)
                > 0.9
        })
        .collect();
    let mut ev = vec![0.0; n];
    let mut seen = vec![false; n];
    for start in 0..n {
        if !clipped[start] || seen[start] {
            continue;
        }
        let mut members = vec![start];
        seen[start] = true;
        let mut k = 0;
        while k < members.len() {
            let (x, y) = ((members[k] % w) as isize, (members[k] / w) as isize);
            k += 1;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let q = ny as usize * w + nx as usize;
                    if clipped[q] && !seen[q] {
                        seen[q] = true;
                        members.push(q);
                    }
                }
            }
        }
        let d = 2.0 * (members.len() as f64 / hd / hd / std::f64::consts::PI).sqrt();
        let b = light_boost_ev(d);
        for q in members {
            ev[q] = b;
        }
    }
    ev
}

/// The restored light of a pixel of a scene whose pixels are either well inside or well beyond the clip ramps,
/// of a light assumed `ev` EV above the plain 3 EV of headroom.
fn restored(x: [f64; 3], clip: f64, ev: f64) -> [f64; 3] {
    let m = x.iter().map(|v| v / clip).fold(f64::INFINITY, f64::min);
    let f = if m >= 0.98 { 1.0 } else { 0.0 };
    let g = 1.0 + (2f64.powf(3.0 + ev) - 1.0) * f;
    [x[0] * g, x[1] * g, x[2] * g]
}

/// The highlights of a pixel (the light's own extra headroom `ev`).
fn highlight(x: [f64; 3], clip: f64, ev: f64) -> [f64; 3] {
    let m = x.iter().map(|v| v / clip).fold(f64::NEG_INFINITY, f64::max);
    let k = smoothstep((m - 0.5) / 0.5);
    let r = restored(x, clip, ev);
    [r[0] * k, r[1] * k, r[2] * k]
}

/// Sums over a scene: the restored light the glow and the veil read (the lights' extra headroom in EV, times
/// the glow's share of it), the highlights the ghosts, the starburst and the streak read (the whole extra
/// headroom), and the highlights of the one-pixel lights.
struct Sums {
    r: [f64; 3],
    h: [f64; 3],
    h_compact: [f64; 3],
}

fn sums(scene: &Scene, clip: f64) -> Sums {
    let n = scene.w * scene.h;
    let ev = boost_map(scene, clip);
    let mut s = Sums {
        r: [0.0; 3],
        h: [0.0; 3],
        h_compact: [0.0; 3],
    };
    for i in 0..n {
        let x = [
            f64::from(scene.data[i]),
            f64::from(scene.data[n + i]),
            f64::from(scene.data[2 * n + i]),
        ];
        let r = restored(x, clip, GLOW_BOOST_SHARE * ev[i]);
        let h = highlight(x, clip, ev[i]);
        for c in 0..3 {
            s.r[c] += r[c];
            s.h[c] += h[c];
        }
    }
    s
}

/// The tint of a spectrum on the camera, scaled to a mean of 1 over the channels (`docs/21`, L3).
fn tint(coating: &[f64]) -> [f64; 3] {
    let mut s = [0.0; 31];
    s.copy_from_slice(coating);
    let t = camera().tint(&s);
    let sum = t[0] + t[1] + t[2];
    [3.0 * t[0] / sum, 3.0 * t[1] / sum, 3.0 * t[2] / sum]
}

/// What L3 should return per channel, in units of `F`, from the lens data and the scene's sums.
fn expected_energy(plan: &LensPlan, s: &Sums, strength: f64) -> [f64; 3] {
    let lens = plan.lens();
    let att = plan.attachment();
    let st = plan.flare_strengths();
    let anam = (plan.q() - 1.0) / (att.q_max - 1.0);
    let t_streak = tint(&att.streak.tint);
    let mut out = [0.0; 3];
    for c in 0..3 {
        let mut e = (st.glow + st.veil) * s.r[c];
        for g in &lens.flare.ghosts {
            e += st.ghost_gain * g.fraction * tint(&g.coating)[c] * s.h[c];
        }
        e += st.burst * strength * s.h_compact[c];
        e += anam * st.streak * t_streak[c] * s.h[c];
        for g in &att.ghosts {
            e += anam * st.attachment_ghost_gain * g.fraction * tint(&g.coating)[c] * s.h[c];
        }
        out[c] = e;
    }
    out
}

/// `sum (out - (1 - a) Z) / F` per channel.
fn measured_energy(plan: &LensPlan, out: &[f32], z: &[f32]) -> [f64; 3] {
    let n = out.len() / 3;
    let a = plan.light_fraction();
    let f = plan.flares();
    let mut e = [0.0; 3];
    for c in 0..3 {
        for i in 0..n {
            e[c] += f64::from(out[c * n + i]) - (1.0 - a) * f64::from(z[c * n + i]);
        }
        e[c] /= f;
    }
    e
}

/// Checks the measured energy of a run against the expected one.
fn check_energy(label: &str, got: [f64; 3], want: [f64; 3], tol: f64) {
    for c in 0..3 {
        let rel = (got[c] - want[c]).abs() / want[c].abs().max(1e-30);
        assert!(
            rel <= tol,
            "{label}, channel {c}: measured {} expected {} (relative {rel:e}, allowed {tol:e})",
            got[c],
            want[c]
        );
    }
}

/// One part of the flare on the energy scene: the light it spreads is the fraction times what it reads.
fn part_energy(p: Parts, axis: Axis, anamorphic: f64, tol: f64) -> f64 {
    let lib = parts_library(p);
    let mut scene = energy_scene();
    if axis == Axis::Y {
        scene = scene.transposed();
    }
    let size = (scene.w, scene.h);
    let plan = plan_of(&lib, 100.0, 0.0, anamorphic, size, axis, Some(16.0), 1.0);
    assert!(plan.light_fraction() > 0.0);
    let planes = planes_of(&plan, &scene);
    let out = run(&plan, &planes, &scene);
    // Z is the picture itself when the field optics have nothing to do, else L2's output.
    let z = if anamorphic > 0.0 {
        let zplan = plan_of(&lib, 0.0, 0.0, anamorphic, size, axis, Some(16.0), 1.0);
        run(&zplan, &planes_of(&zplan, &scene), &scene)
    } else {
        scene.data.clone()
    };
    let got = measured_energy(&plan, &out, &z);
    let want = expected_energy(&plan, &sums_for(&scene, size), 1.0);
    check_energy(&format!("{axis:?} A{anamorphic}"), got, want, tol);
    // Energy is also spread, not lost: the output is finite and non-negative.
    assert!(out.iter().all(|v| v.is_finite() && *v >= -1e-6));
    got.iter().sum::<f64>()
}

/// The sums of a scene, with the one-pixel lights known by position of the standard scene.
fn sums_for(scene: &Scene, size: (usize, usize)) -> Sums {
    let mut s = sums(scene, 1.0);
    // The lights B and C of the standard scene are the compact ones.
    let at = |x: usize, y: usize| -> [f64; 3] {
        let (x, y) = if size == (scene.w, scene.h) && scene.w < scene.h {
            (y, x)
        } else {
            (x, y)
        };
        let n = scene.w * scene.h;
        let i = y * scene.w + x;
        [
            f64::from(scene.data[i]),
            f64::from(scene.data[n + i]),
            f64::from(scene.data[2 * n + i]),
        ]
    };
    let ev = boost_map(scene, 1.0);
    for (x, y) in [(300, 120), (200, 200)] {
        let (px, py) = if size == (scene.w, scene.h) && scene.w < scene.h {
            (y, x)
        } else {
            (x, y)
        };
        let h = highlight(at(x, y), 1.0, ev[py * scene.w + px]);
        for (acc, v) in s.h_compact.iter_mut().zip(&h) {
            *acc += v;
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------------------
// Energy

#[test]
fn each_part_spreads_exactly_the_light_it_takes() {
    let tight = 1e-6;
    for (label, p) in [
        (
            "glow",
            Parts {
                glow: true,
                ..Parts::NONE
            },
        ),
        (
            "veil",
            Parts {
                veil: true,
                ..Parts::NONE
            },
        ),
        (
            "ghosts",
            Parts {
                ghosts: true,
                ..Parts::NONE
            },
        ),
        (
            "starburst",
            Parts {
                burst: true,
                ..Parts::NONE
            },
        ),
    ] {
        let e = part_energy(p, Axis::X, 0.0, tight);
        assert!(e > 0.0, "{label} spread no light");
    }
    // The streak and the attachment's ghosts need Anamorphic; the axis may be either.
    for (label, p) in [
        (
            "streak",
            Parts {
                streak: true,
                ..Parts::NONE
            },
        ),
        (
            "attachment ghosts",
            Parts {
                att_ghosts: true,
                ..Parts::NONE
            },
        ),
    ] {
        for axis in [Axis::X, Axis::Y] {
            let e = part_energy(p, axis, 100.0, tight);
            assert!(e > 0.0, "{label} {axis:?} spread no light");
        }
    }
}

#[test]
fn all_parts_together_keep_the_budget() {
    // Every part at once, at Anamorphic 100 and at 50, on both axes.
    for (axis, anamorphic) in [
        (Axis::X, 100.0),
        (Axis::Y, 100.0),
        (Axis::X, 50.0),
        (Axis::X, 0.0),
    ] {
        part_energy(Parts::ALL, axis, anamorphic, 1e-6);
    }
}

#[test]
fn polygon_ghosts_keep_their_light_closely() {
    // The corners of a polygon aperture are creases that a pixel grid samples with a small error.
    let p = Parts {
        ghosts: true,
        att_ghosts: true,
        blades: 6,
        ..Parts::NONE
    };
    part_energy(p, Axis::X, 100.0, 1e-3);
}

#[test]
fn a_weak_starburst_takes_less_light_and_leaves_the_rest() {
    // The starburst grows with the f-number, and the light it does not take stays in the picture.
    let lib = parts_library(Parts {
        burst: true,
        ..Parts::NONE
    });
    let scene = energy_scene();
    let size = (scene.w, scene.h);
    let sm = sums_for(&scene, size);
    let full = lib.get(LensKind::Classic).unwrap().flare.starburst;
    let mut energies = Vec::new();
    for n in [2.0, 3.0, 4.0, 8.0, 11.0, 22.0] {
        let plan = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, Some(n), 1.0);
        let planes = planes_of(&plan, &scene);
        let out = run(&plan, &planes, &scene);
        let strength = ((n - 2.0) / (full.f_number_full - 2.0)).clamp(0.0, 1.0);
        // The plan's own fraction is the full one; the light actually taken is the strength's share.
        let a = full.fraction * strength;
        let nn = out.len() / 3;
        let mut e = [0.0; 3];
        for c in 0..3 {
            for i in 0..nn {
                e[c] += f64::from(out[c * nn + i]) - (1.0 - a) * f64::from(scene.data[c * nn + i]);
            }
        }
        check_energy(
            &format!("f/{n}"),
            e,
            [
                a * sm.h_compact[0],
                a * sm.h_compact[1],
                a * sm.h_compact[2],
            ],
            // The output is f32 and the lights' own pixels round alike (a few parts in a million of a
            // starburst that carries a tenth of nine pixels).
            2e-5,
        );
        energies.push(e.iter().sum::<f64>());
    }
    assert!(energies[0].abs() < 1e-9, "no strength at f/2: {energies:?}");
    assert!(
        energies.windows(2).all(|p| p[1] >= p[0]),
        "the starburst must not shrink as the f-number grows: {energies:?}"
    );
    // Beyond the full f-number it stops growing.
    assert!((energies[4] - energies[5]).abs() < 1e-9 * energies[4]);
    // No f-number means f/4.
    let plan = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, None, 1.0);
    let planes = planes_of(&plan, &scene);
    let out = run(&plan, &planes, &scene);
    let a = full.fraction * (2.0 / (full.f_number_full - 2.0));
    let nn = out.len() / 3;
    let e: f64 = out[..nn]
        .iter()
        .zip(&scene.data[..nn])
        .map(|(o, z)| f64::from(*o) - (1.0 - a) * f64::from(*z))
        .sum();
    assert!((e - a * sm.h_compact[0]).abs() < 2e-5 * e.abs(), "{e}");
}

#[test]
fn a_clipped_light_gets_its_headroom_in_the_flare_and_not_in_its_own_pixels() {
    let lib = parts_library(Parts {
        glow: true,
        ..Parts::NONE
    });
    let mut scene = Scene::new(480, 320);
    scene.fill(250, 170, 3, 3, [1.0; 3]);
    let size = (scene.w, scene.h);
    // Clipped: the pixels are at their clip level. Not clipped: the same pixels are an eighth of it.
    let clipped = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, None, 1.0);
    let open = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, None, 8.0);
    let (out_c, out_o) = (
        run(&clipped, &planes_of(&clipped, &scene), &scene),
        run(&open, &planes_of(&open, &scene), &scene),
    );
    let e_c = measured_energy(&clipped, &out_c, &scene.data);
    let e_o = measured_energy(&open, &out_o, &scene.data);
    // The light is small, so it holds more than the plain 3 EV: its extra headroom, the glow's share of it.
    let hd = 0.5 * (480.0f64 * 480.0 + 320.0 * 320.0).sqrt();
    let diameter = 2.0 * (9.0 / std::f64::consts::PI).sqrt() / hd;
    let headroom = 2f64.powf(3.0 + GLOW_BOOST_SHARE * light_boost_ev(diameter));
    assert!(headroom > 8.0, "a 3 x 3 light must be boosted: {headroom}");
    for c in 0..3 {
        // The output is f32, so nine identical pixels round alike: a few parts in a million at most.
        assert!(
            (e_c[c] / e_o[c] / headroom - 1.0).abs() < 5e-6,
            "channel {c}: {} against {}",
            e_c[c],
            e_o[c]
        );
        // And the absolute amount: the glow's fraction of nine pixels at the headroom.
        assert!((e_c[c] - 0.15 * 9.0 * headroom).abs() < 1e-6 * e_c[c]);
    }
    // The light's own pixels are not raised: they keep (1 - a) of themselves and what the glow gives back.
    let at = 171 * scene.w + 251;
    for c in 0..3 {
        let v = out_c[c * scene.w * scene.h + at];
        // (1 - a) of itself, plus what the glow's core and tail give back: below the light itself.
        let keep = 1.0 - clipped.light_fraction();
        assert!(v < 1.0 && f64::from(v) > keep, "channel {c}: {v}");
    }
}

#[test]
fn a_large_clipped_area_is_a_surface_and_keeps_the_plain_headroom() {
    // A clipped block 0.43 half-diagonals across is no light: its flare is its own light at 3 EV, while a small
    // light of the same brightness holds far more. The block's flares stay inside the frame.
    let lib = parts_library(Parts {
        glow: true,
        ghosts: true,
        ..Parts::NONE
    });
    let mut big = Scene::new(480, 320);
    big.fill(185, 105, 110, 110, [1.0; 3]);
    let size = (big.w, big.h);
    let plan = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, None, 1.0);
    let ev = boost_map(&big, 1.0);
    assert!(ev.iter().all(|e| *e == 0.0), "the block must get no boost");
    let out = run(&plan, &planes_of(&plan, &big), &big);
    let got = measured_energy(&plan, &out, &big.data);
    let want = expected_energy(&plan, &sums_for(&big, size), 1.0);
    check_energy("a big clipped block", got, want, 1e-6);
    // The glow's share of that is its fraction of the block's light at the plain 3 EV.
    let glow_only = 0.15 * 110.0 * 110.0 * 8.0;
    let sm = sums_for(&big, size);
    assert!((sm.r[1] / (110.0 * 110.0 * 8.0) - 1.0).abs() < 1e-9);
    assert!(got[1] > glow_only, "the ghosts add to the glow's {glow_only}: {}", got[1]);
}

#[test]
fn the_extra_headroom_falls_as_the_light_grows() {
    // Lights of 1, 3, 9 and 27 px at the same brightness: the light a pixel sends into the glow falls with
    // the size, from the whole extra headroom to the plain one, and the energy test holds for each.
    let lib = parts_library(Parts {
        glow: true,
        ..Parts::NONE
    });
    let mut per_pixel = Vec::new();
    for side in [1usize, 3, 9, 27] {
        let mut scene = Scene::new(480, 320);
        scene.fill(230, 150, side, side, [1.0; 3]);
        let size = (scene.w, scene.h);
        let plan = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, None, 1.0);
        let out = run(&plan, &planes_of(&plan, &scene), &scene);
        let got = measured_energy(&plan, &out, &scene.data);
        let want = expected_energy(&plan, &sums_for(&scene, size), 1.0);
        check_energy(&format!("{side} px"), got, want, 1e-6);
        per_pixel.push(got[1] / (side * side) as f64);
    }
    assert!(
        per_pixel.windows(2).all(|p| p[0] > p[1]),
        "the light per clipped pixel must fall with the light's size: {per_pixel:?}"
    );
}

/// The scene reduced 2 times by 2 x 2 means.
fn reduced(scene: &Scene) -> Scene {
    let (w, h) = (scene.w / 2, scene.h / 2);
    let mut r = Scene::new(w, h);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let at = |dx: usize, dy: usize| {
                    scene.data[c * scene.w * scene.h + (2 * y + dy) * scene.w + 2 * x + dx]
                };
                r.data[c * w * h + y * w + x] = 0.25 * (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1));
            }
        }
    }
    r
}

#[test]
fn a_reduced_view_spreads_the_same_light() {
    // The planes are made once from the full frame; a view reduced 2 times reads them at its own pixels and
    // draws the same lights, so its pixels (4 times fewer) hold a quarter of the flare light. The parts the
    // tail plane and the lists make are checked to 2 %; the core, which reads the reduced view itself, is a
    // small part of the glow.
    let p = Parts {
        burst: false,
        ..Parts::ALL
    };
    let lib = parts_library(p);
    let mut scene = Scene::new(480, 320);
    scene.fill(250, 170, 8, 8, [1.0; 3]);
    scene.fill(300, 120, 6, 6, [1.0, 1.0, 0.9]);
    for c in 0..3 {
        for y in 140..180 {
            for x in 150..200 {
                scene.data[c * 480 * 320 + y * 480 + x] = 0.4 * pattern(x, y, c);
            }
        }
    }
    let half = reduced(&scene);
    let full_plan = plan_of(
        &lib,
        100.0,
        0.0,
        100.0,
        (480, 320),
        Axis::X,
        Some(16.0),
        1.0,
    );
    let planes = planes_of(&full_plan, &scene);
    let energy = |k: usize, s: &Scene| -> f64 {
        let out = apply(&full_plan, &planes, k, &whole(s.w, s.h, &s.data), s.rect()).unwrap();
        let zp = plan_of(&lib, 0.0, 0.0, 100.0, (480, 320), Axis::X, Some(16.0), 1.0);
        let z = apply(&zp, &planes, k, &whole(s.w, s.h, &s.data), s.rect()).unwrap();
        measured_energy(&full_plan, &out, &z).iter().sum::<f64>() * (k * k) as f64
    };
    let (e1, e2) = (energy(1, &scene), energy(2, &half));
    assert!(
        (e2 / e1 - 1.0).abs() < 0.02,
        "the flare light is {e1} at full size and {e2} reduced 2 times"
    );
}

// ---------------------------------------------------------------------------------------------------------
// Where things are

/// The weighted centroid of `d` (summed over channels) in the window of half-size `half` around `(cx, cy)`
/// (continuous pixel coordinates), and the weight.
fn centroid(d: &[f64], w: usize, h: usize, cx: f64, cy: f64, half: f64) -> ((f64, f64), f64) {
    let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            if (px - cx).abs() <= half && (py - cy).abs() <= half {
                let v: f64 = (0..3).map(|c| d[c * w * h + y * w + x]).sum();
                sx += v * px;
                sy += v * py;
                sw += v;
            }
        }
    }
    ((sx / sw, sy / sw), sw)
}

/// `out - (1 - a) Z` of a plan, planar `f64`.
fn flare_light(plan: &LensPlan, out: &[f32], z: &[f32]) -> Vec<f64> {
    let a = plan.light_fraction();
    out.iter()
        .zip(z)
        .map(|(o, z)| f64::from(*o) - (1.0 - a) * f64::from(*z))
        .collect()
}

#[test]
fn ghost_centres_lie_on_the_line_from_the_source_through_the_centre() {
    let lib = library_with(
        &|lens| {
            lens["flare"]["glow"]["fraction"] = json!(0.0);
            lens["flare"]["veil"]["fraction"] = json!(0.0);
            lens["flare"]["starburst"]["fraction"] = json!(0.0);
            lens["flare"]["aperture"] =
                json!({ "blades": 6, "rotation_deg": 10.0, "roundness": 0.5 });
            lens["flare"]["ghosts"] = json!([
                ghost_json(-0.5, 0.0, 0.03, 3.0, 0.02, 450.0),
                ghost_json(0.7, 0.0, 0.03, 3.0, 0.02, 600.0),
                ghost_json(1.4, 0.0, 0.03, 3.0, 0.02, 520.0),
            ]);
        },
        &|att| {
            att["streak"]["fraction"] = json!(0.0);
            att["ghosts"] = json!([]);
        },
    );
    let ms = [-0.5, 0.7, 1.4];
    for (axis, anamorphic) in [
        (Axis::X, 0.0),
        (Axis::Y, 0.0),
        (Axis::X, 100.0),
        (Axis::Y, 60.0),
    ] {
        let mut scene = Scene::new(480, 320);
        scene.fill(300, 200, 3, 3, [1.0; 3]);
        if axis == Axis::Y {
            scene = scene.transposed();
        }
        let size = (scene.w, scene.h);
        let plan = plan_of(&lib, 100.0, 0.0, anamorphic, size, axis, None, 1.0);
        let planes = planes_of(&plan, &scene);
        let out = run(&plan, &planes, &scene);
        let z = if anamorphic > 0.0 {
            let zp = plan_of(&lib, 0.0, 0.0, anamorphic, size, axis, None, 1.0);
            run(&zp, &planes_of(&zp, &scene), &scene)
        } else {
            scene.data.clone()
        };
        let d = flare_light(&plan, &out, &z);
        // The light's centroid: pixel 301.5, 201.5 of the 480 x 320 frame (or the transposed one).
        let (centre, source) = if axis == Axis::X {
            ((240.0, 160.0), (301.5, 201.5))
        } else {
            ((160.0, 240.0), (201.5, 301.5))
        };
        for m in ms {
            let want = (
                centre.0 + m * (source.0 - centre.0),
                centre.1 + m * (source.1 - centre.1),
            );
            let (got, weight) = centroid(&d, scene.w, scene.h, want.0, want.1, 16.0);
            assert!(weight > 0.0);
            let off = ((got.0 - want.0).powi(2) + (got.1 - want.1).powi(2)).sqrt();
            assert!(
                off < 0.5,
                "{axis:?} A{anamorphic} m = {m}: the ghost is at {got:?}, expected {want:?}"
            );
            // On the line through the source and the centre, and on the far side for negative m.
            let (sx, sy) = (source.0 - centre.0, source.1 - centre.1);
            let (gx, gy) = (got.0 - centre.0, got.1 - centre.1);
            let cross = (sx * gy - sy * gx) / (sx.hypot(sy));
            assert!(
                cross.abs() < 0.5,
                "{axis:?} m = {m}: {cross} px off the line"
            );
            let along = sx * gx + sy * gy;
            assert_eq!(along < 0.0, m < 0.0, "{axis:?} m = {m}: wrong side");
        }
    }
}

/// The light of `d` in the pixels of a band: `|across - c| <= half_h` and `lo <= |along - a| <= hi`, along and
/// across being x and y for [`Axis::X`] and y and x for [`Axis::Y`].
fn band_light(
    d: &[f64],
    w: usize,
    h: usize,
    axis: Axis,
    source: (usize, usize),
    along_band: bool,
) -> f64 {
    let mut s = 0.0;
    for y in 0..h {
        for x in 0..w {
            let (a, c) = match axis {
                Axis::X => (
                    x as isize - source.0 as isize,
                    y as isize - source.1 as isize,
                ),
                Axis::Y => (
                    y as isize - source.1 as isize,
                    x as isize - source.0 as isize,
                ),
            };
            let (a, c) = (a.abs(), c.abs());
            // Along the axis through the source, or across it: the same pixels turned by a quarter.
            let (p, q) = if along_band { (a, c) } else { (c, a) };
            if q <= 1 && (12..=80).contains(&p) {
                s += (0..3).map(|k| d[k * w * h + y * w + x]).sum::<f64>();
            }
        }
    }
    s
}

#[test]
fn streaks_lie_along_the_axis_and_only_with_anamorphic() {
    let streak_only = Parts {
        streak: true,
        ..Parts::NONE
    };
    let lib = parts_library(streak_only);
    for axis in [Axis::X, Axis::Y] {
        let mut scene = Scene::new(480, 320);
        scene.fill(250, 170, 3, 3, [1.0; 3]);
        if axis == Axis::Y {
            scene = scene.transposed();
        }
        let size = (scene.w, scene.h);
        let source = if axis == Axis::X {
            (251, 171)
        } else {
            (171, 251)
        };

        // Anamorphic 0: no streak at all, so no light and the picture is untouched.
        let flat = plan_of(&lib, 100.0, 0.0, 0.0, size, axis, None, 1.0);
        assert_eq!(flat.light_fraction(), 0.0);
        let out = run(&flat, &planes_of(&flat, &scene), &scene);
        assert_eq!(out, scene.data, "{axis:?}: a streak without Anamorphic");

        // Anamorphic 100: the streak lies along the axis.
        let plan = plan_of(&lib, 100.0, 0.0, 100.0, size, axis, None, 1.0);
        let planes = planes_of(&plan, &scene);
        let out = run(&plan, &planes, &scene);
        let zp = plan_of(&lib, 0.0, 0.0, 100.0, size, axis, None, 1.0);
        let z = run(&zp, &planes_of(&zp, &scene), &scene);
        let d = flare_light(&plan, &out, &z);
        let along = band_light(&d, scene.w, scene.h, axis, source, true);
        let across = band_light(&d, scene.w, scene.h, axis, source, false);
        assert!(
            along > 100.0 * across.max(1e-12) && along > 0.1,
            "{axis:?}: {along} along the axis against {across} across it"
        );
    }
}

#[test]
fn the_attachment_adds_nothing_at_anamorphic_zero() {
    // The same lens with and without the attachment's streak and ghosts: bit for bit equal at Anamorphic 0,
    // different at Anamorphic 100.
    let full = parts_library(Parts::ALL);
    let bare = parts_library(Parts {
        streak: false,
        att_ghosts: false,
        ..Parts::ALL
    });
    let scene = energy_scene();
    let size = (scene.w, scene.h);
    let run_with = |lib: &LensLibrary, anamorphic: f64| {
        let plan = plan_of(lib, 100.0, 0.0, anamorphic, size, Axis::X, None, 1.0);
        run(&plan, &planes_of(&plan, &scene), &scene)
    };
    assert_eq!(run_with(&full, 0.0), run_with(&bare, 0.0));
    assert_ne!(run_with(&full, 100.0), run_with(&bare, 100.0));
}

#[test]
fn a_lens_without_flare_light_leaves_the_picture_to_the_field_optics() {
    // No fractions of light: L3 has nothing to take or give, so with Edges and Anamorphic at 0 the picture comes
    // back bit for bit, whatever Flares says; with the field optics on, Flares changes nothing either.
    let lib = parts_library(Parts::NONE);
    let scene = energy_scene();
    let size = (scene.w, scene.h);
    let plan = plan_of(&lib, 70.0, 0.0, 0.0, size, Axis::X, None, 1.0);
    assert_eq!(plan.light_fraction(), 0.0);
    assert_eq!(run(&plan, &planes_of(&plan, &scene), &scene), scene.data);
    let with_field = plan_of(&lib, 70.0, 60.0, 30.0, size, Axis::X, None, 1.0);
    let without = plan_of(&lib, 0.0, 60.0, 30.0, size, Axis::X, None, 1.0);
    assert_eq!(
        run(&with_field, &planes_of(&with_field, &scene), &scene),
        run(&without, &planes_of(&without, &scene), &scene)
    );
}

// ---------------------------------------------------------------------------------------------------------
// Tiles, determinism, cancel

/// A textured scene with lights of different sizes at fractions of the frame, and one large clipped block.
fn varied_scene(w: usize, h: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; 3 * w * h];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data[c * w * h + y * w + x] = 0.35 * pattern(x, y, c);
            }
        }
    }
    let diag = ((w * w + h * h) as f64).sqrt();
    let mut light = |fx: f64, fy: f64, size_hd: f64, rgb: [f32; 3]| {
        let s = ((size_hd * 0.5 * diag).round() as usize).max(1);
        let (x0, y0) = ((fx * w as f64) as usize, (fy * h as f64) as usize);
        for (c, v) in rgb.iter().enumerate() {
            for y in y0..(y0 + s).min(h) {
                for x in x0..(x0 + s).min(w) {
                    data[c * w * h + y * w + x] = *v;
                }
            }
        }
    };
    light(0.31, 0.42, 0.0005, [1.0; 3]);
    light(0.62, 0.30, 0.0035, [1.0; 3]);
    light(0.50, 0.77, 0.008, [1.0, 1.0, 1.0]);
    light(0.80, 0.55, 0.0005, [1.0, 0.6, 0.3]);
    light(0.12, 0.20, 0.0005, [1.0; 3]);
    light(0.47, 0.49, 0.004, [1.0; 3]);
    light(0.70, 0.70, 0.06, [1.0; 3]);
    data
}

#[test]
fn tiles_equal_the_whole_frame_bit_for_bit() {
    let lib = common::library();
    for (axis, k, (fw, fh), full) in [
        (
            Axis::X,
            1usize,
            (1200usize, 800usize),
            (1200usize, 800usize),
        ),
        (Axis::Y, 1, (800, 1200), (800, 1200)),
        (Axis::X, 4, (600, 400), (2400, 1600)),
        (Axis::Y, 4, (400, 600), (1600, 2400)),
    ] {
        for (kind, (flares, edges, anamorphic)) in [
            (LensKind::Petzval, (100.0, 0.0, 100.0)),
            (LensKind::Classic, (60.0, 0.0, 0.0)),
        ] {
            let p = common::plan(kind, flares, edges, anamorphic, full.0, full.1, axis);
            // The lens plane is the frame reduced `lens_plane_scale` times.
            let ks = p.lens_plane_scale();
            let (pw, ph) = (full.0.div_ceil(ks), full.1.div_ceil(ks));
            let plane = varied_scene(pw, ph);
            let planes = LensPlanes::build(&p, &whole(pw, ph, &plane), None, &|| false).unwrap();
            let data = varied_scene(fw, fh);
            let reference = apply_tile(&p, &planes, k, &data, fw, fh, PixelRect::new(0, 0, fw, fh));
            assert_ne!(
                reference, data,
                "{kind} {axis:?} k={k}: the lens changed nothing"
            );
            for (nx, ny) in [(2, 2), (3, 3)] {
                for t in tiles(fw, fh, nx, ny) {
                    let got = apply_tile(&p, &planes, k, &data, fw, fh, t);
                    for c in 0..3 {
                        for y in 0..t.h {
                            let want = &reference[c * fw * fh + (t.y + y) * fw + t.x..][..t.w];
                            assert_eq!(
                                &got[c * t.w * t.h + y * t.w..][..t.w],
                                want,
                                "{kind} {axis:?} k={k} {nx}x{ny} tile {t:?} channel {c} row {y}"
                            );
                        }
                    }
                }
            }
        }
        let _ = lib;
    }
}

/// The planes of `plan` from `varied_scene` at the lens plane's scale.
fn varied_planes(plan: &LensPlan) -> LensPlanes {
    let ks = plan.lens_plane_scale();
    let f = plan.frame();
    let (pw, ph) = (f.width.div_ceil(ks), f.height.div_ceil(ks));
    let plane = varied_scene(pw, ph);
    LensPlanes::build(plan, &whole(pw, ph, &plane), None, &|| false).expect("planes")
}

#[test]
fn the_core_widens_the_halo_and_flares_off_leaves_it_alone() {
    // At 1200 x 800 the glow's core is 1.08 px: one pixel of context.
    let (w, h) = (1200, 800);
    let on = common::plan(LensKind::Classic, 100.0, 0.0, 0.0, w, h, Axis::X);
    let plane = varied_scene(w, h);
    let planes = varied_planes(&on);
    assert_eq!(ym_lens::halo_px(&on, &planes, 1), 1);
    // A reduced view needs none; a frame that is too small for the core to reach a pixel needs none.
    assert_eq!(ym_lens::halo_px(&on, &planes, 4), 0);
    let off = common::plan(LensKind::Classic, 0.0, 50.0, 0.0, w, h, Axis::X);
    let planes_off = varied_planes(&off);
    assert!(ym_lens::halo_px(&off, &planes_off, 1) > 0);
    // Flares 0: L3 is the identity, the field optics alone decide.
    let both = common::plan(LensKind::Classic, 0.0, 50.0, 0.0, w, h, Axis::X);
    let a = apply_tile(
        &both,
        &planes_off,
        1,
        &plane,
        w,
        h,
        PixelRect::new(100, 100, 200, 150),
    );
    let none = common::plan(LensKind::Classic, 0.0, 50.0, 0.0, w, h, Axis::X);
    let b = apply_tile(
        &none,
        &planes_off,
        1,
        &plane,
        w,
        h,
        PixelRect::new(100, 100, 200, 150),
    );
    assert_eq!(a, b);
}

#[test]
fn two_runs_and_any_thread_count_give_the_same_bits() {
    let (w, h) = (1200usize, 800usize);
    let p = common::plan(LensKind::Vintage, 100.0, 30.0, 100.0, w, h, Axis::X);
    let plane = varied_scene(w, h);
    let go = || {
        let planes = varied_planes(&p);
        apply(
            &p,
            &planes,
            1,
            &whole(w, h, &plane),
            PixelRect::new(0, 0, w, h),
        )
        .unwrap()
    };
    let reference = go();
    assert_eq!(reference, go());
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(reference, pool.install(go), "{threads} threads");
    }
}

#[test]
fn a_cancel_during_build_returns_cancelled() {
    let (w, h) = (600usize, 400usize);
    let p = common::plan(LensKind::Classic, 100.0, 0.0, 100.0, w, h, Axis::X);
    let plane = varied_scene(w, h);
    let polls = AtomicUsize::new(0);
    let counted = || {
        polls.fetch_add(1, Ordering::SeqCst);
        false
    };
    LensPlanes::build(&p, &whole(w, h, &plane), None, &counted).unwrap();
    let total = polls.load(Ordering::SeqCst);
    assert!(total >= 8, "the build polls the cancel only {total} times");
    for stop in [1, 2, 3, total / 2, total - 1, total] {
        let n = AtomicUsize::new(0);
        let cancel = || n.fetch_add(1, Ordering::SeqCst) + 1 >= stop;
        let r = LensPlanes::build(&p, &whole(w, h, &plane), None, &cancel);
        assert!(
            matches!(r, Err(LensError::Cancelled)),
            "cancelled at poll {stop} of {total}"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------
// Many lights: the cost of the ghosts

fn soft_ghost(m: f64, n: f64, s: f64, clip: f64, fraction: f64, soft: f64, nm: f64) -> Value {
    json!({ "m": m, "n": n, "s": s, "clip_radius": clip, "fraction": fraction, "soft": soft,
            "coating": coating(nm) })
}

/// A lens that only has ghosts, big and soft ones (the apothem is up to a whole half-diagonal, as in the
/// shipped lenses), some cut by their clipping circle, with the fractions `fraction` times 1, 2 and 3.
fn big_ghost_library(fraction: f64) -> LensLibrary {
    library_with(
        &|lens| {
            lens["flare"]["glow"] = json!({ "fraction": 0.0, "core_radius": 0.0015, "r0": 0.01,
                "power": 2.0, "reach": 0.3, "chroma": [1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8] });
            lens["flare"]["veil"] = json!({ "fraction": 0.0, "radius": 0.3 });
            lens["flare"]["aperture"] =
                json!({ "blades": 7, "rotation_deg": 10.0, "roundness": 0.5 });
            lens["flare"]["ghosts"] = json!([
                soft_ghost(-0.9, 0.0, 0.8, 4.0, fraction, 0.30, 450.0),
                soft_ghost(0.7, 0.6, 0.45, 0.9, 2.0 * fraction, 0.12, 600.0),
                soft_ghost(1.6, 0.0, 0.9, 4.0, 3.0 * fraction, 0.05, 520.0),
                soft_ghost(0.3, 0.2, 0.15, 4.0, fraction, 0.03, 500.0),
            ]);
            lens["flare"]["starburst"] =
                json!({ "fraction": 0.0, "radius": 0.06, "f_number_full": 11.0 });
        },
        &|att| {
            att["streak"] = json!({ "fraction": 0.0, "reach": 0.3, "core": 0.002, "power": 1.5,
                "tint": coating(470.0) });
            att["ghosts"] = json!([soft_ghost(-0.5, 0.0, 0.7, 4.0, fraction, 0.2, 470.0)]);
        },
    )
}

/// A 600 x 400 scene with `n` clipped lights of 2 to 4 px at seeded places.
fn lights_scene(n: usize) -> Scene {
    let mut s = Scene::new(600, 400);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f64 / f64::from(1u32 << 31)
    };
    for _ in 0..n {
        let size = 2 + (next() * 3.0) as usize;
        let x = 10 + (next() * 570.0) as usize;
        let y = 10 + (next() * 370.0) as usize;
        s.fill(x, y, size, size, [1.0, 0.9 + 0.1 * next() as f32, 0.8]);
    }
    s
}

/// The ghosts that vary slowly are drawn on grids of nodes and read back by interpolation: the light they
/// add is the exact sum within a small share of its peak, and its total is the same.
#[test]
fn the_grids_of_the_ghosts_are_the_exact_sum_within_a_small_share() {
    let lib = big_ghost_library(0.02);
    let plan = plan_of(&lib, 100.0, 0.0, 100.0, (600, 400), Axis::X, Some(8.0), 1.0);
    let scene = lights_scene(40);
    let planes = planes_of(&plan, &scene);
    let input = whole(scene.w, scene.h, &scene.data);
    let rect = scene.rect();
    let got = ym_lens::flare_terms(&plan, &planes, 1, &input, rect)
        .expect("terms")
        .expect("some light");
    let want = ym_lens::flare_terms_exact(&plan, &planes, 1, &input, rect)
        .expect("exact terms")
        .expect("some light");
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, v| m.max(*v));
    let worst = got
        .iter()
        .zip(&want)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let (sum_got, sum_want): (f64, f64) = (got.iter().sum(), want.iter().sum());
    let _ = writeln!(
        std::io::stderr(),
        "[flare] grids against every pixel: worst {worst:.3e} of a peak {peak:.3e} ({:.3} %), total light {:.6} of {:.6}",
        100.0 * worst / peak,
        sum_got,
        sum_want
    );
    assert!(worst > 0.0, "the grids were not used");
    assert!(
        worst <= 3e-3 * peak,
        "the grids differ by {worst} of a peak {peak}"
    );
    assert!(
        (sum_got - sum_want).abs() <= 1e-3 * sum_want,
        "{sum_got} against {sum_want}"
    );
}

/// Tiles of any size read the same nodes: the terms of every tile equal the whole frame's bit for bit, with
/// the nodes beyond the tile's edge included.
#[test]
fn tiles_of_the_grids_equal_the_whole_frame_bit_for_bit() {
    let lib = big_ghost_library(0.02);
    let plan = plan_of(&lib, 100.0, 0.0, 100.0, (600, 400), Axis::X, Some(8.0), 1.0);
    let scene = lights_scene(40);
    let planes = planes_of(&plan, &scene);
    let input = whole(scene.w, scene.h, &scene.data);
    let reference = ym_lens::flare_terms(&plan, &planes, 1, &input, scene.rect())
        .expect("terms")
        .expect("some light");
    for (nx, ny) in [(2, 2), (3, 3), (5, 2)] {
        for t in tiles(scene.w, scene.h, nx, ny) {
            let got = ym_lens::flare_terms(&plan, &planes, 1, &input, t)
                .expect("tile")
                .expect("some light");
            for c in 0..3 {
                for y in 0..t.h {
                    let want =
                        &reference[c * scene.w * scene.h + (t.y + y) * scene.w + t.x..][..t.w];
                    assert_eq!(
                        &got[c * t.w * t.h + y * t.w..][..t.w],
                        want,
                        "{nx}x{ny} tile {t:?} channel {c} row {y}"
                    );
                }
            }
        }
    }
}

/// Ghosts whose peaks add up to under the budget (a quarter of the step of a 16-bit code at the clip level) are
/// not drawn: the terms of a lens whose ghosts are that faint are those of a lens without ghosts.
#[test]
fn ghosts_too_faint_to_see_are_not_drawn() {
    let scene = lights_scene(40);
    let faint = big_ghost_library(1e-12);
    let none = library_with(
        &|lens| {
            lens["flare"]["ghosts"] = json!([]);
            lens["flare"]["glow"]["fraction"] = json!(0.0);
            lens["flare"]["veil"]["fraction"] = json!(0.0);
            lens["flare"]["starburst"]["fraction"] = json!(0.0);
        },
        &|att| {
            att["ghosts"] = json!([]);
            att["streak"]["fraction"] = json!(0.0);
        },
    );
    let terms = |lib: &LensLibrary| {
        let plan = plan_of(lib, 100.0, 0.0, 100.0, (600, 400), Axis::X, Some(8.0), 1.0);
        let planes = planes_of(&plan, &scene);
        ym_lens::flare_terms(
            &plan,
            &planes,
            1,
            &whole(scene.w, scene.h, &scene.data),
            scene.rect(),
        )
        .expect("terms")
    };
    let faint_terms = terms(&faint);
    let none_terms = terms(&none);
    // Without any fraction a plan adds no light at all; with faint ghosts the terms are drawn and all zero.
    assert!(
        none_terms
            .as_ref()
            .is_none_or(|t| t.iter().all(|v| *v == 0.0))
    );
    let faint_terms = faint_terms.expect("terms");
    assert!(
        faint_terms.iter().all(|v| *v == 0.0),
        "a faint ghost was drawn"
    );
    // Visible ghosts are drawn.
    let visible = terms(&big_ghost_library(0.02)).expect("terms");
    assert!(visible.iter().any(|v| *v > 0.0));
}

// ---------------------------------------------------------------------------------------------------------
// The light budget of a frame with no lights

/// A uniform frame loses only the light the glow's tail and the veil send beyond its edges (`docs/21` section 4,
/// L3: light that leaves the frame is lost), on the shipped lenses. That is what a lens does, too: the veiling
/// glare a pixel receives is the sum of the scene's light within the frame, so the corners get less than the
/// centre and the frame's mean goes down a little. The bounds are about twice the measured losses at Flares 100
/// (Default 0.2 %, Vintage 2.8 %; the centre of a Vintage frame loses 1.6 % and its corner 6 %).
#[test]
fn a_uniform_frame_loses_only_the_light_that_leaves_it() {
    let lib = LensLibrary::load(&common::repo_root().join("data")).expect("the shipped lens data");
    let (w, h) = (600usize, 400usize);
    for (kind, max_loss) in [(LensKind::Default, 0.01), (LensKind::Vintage, 0.06)] {
        let amounts = LensAmounts {
            kind,
            flares: 100.0,
            edges: 0.0,
            anamorphic: 0.0,
        };
        let p = LensPlan::new(
            &lib,
            &amounts,
            common::frame(w, h, Axis::X),
            common::camera(),
        )
        .expect("plan");
        let ks = p.lens_plane_scale();
        let (pw, ph) = (w.div_ceil(ks), h.div_ceil(ks));
        let plane = vec![0.2f32; 3 * pw * ph];
        let planes = LensPlanes::build(&p, &whole(pw, ph, &plane), None, &|| false).unwrap();
        let data = vec![0.2f32; 3 * w * h];
        let out = apply_tile(&p, &planes, 1, &data, w, h, PixelRect::new(0, 0, w, h));
        let mean = out[..w * h].iter().map(|v| f64::from(*v)).sum::<f64>() / (w * h) as f64;
        let loss = 1.0 - mean / 0.2;
        let (centre, corner) = (out[(h / 2) * w + w / 2], out[0]);
        let _ = writeln!(
            std::io::stderr(),
            "[flare] {kind}: a uniform frame keeps {:.2} % of its light; centre {centre:.5}, corner {corner:.5}",
            100.0 * (1.0 - loss)
        );
        assert!(
            (0.0..=max_loss).contains(&loss),
            "{kind}: the frame lost {loss} of its light"
        );
        assert!(
            centre > corner,
            "{kind}: the corner is not darker than the centre"
        );
    }
}
