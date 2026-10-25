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
use ym_lens::consts::{
    GHOST_COATING_SHIFT_NM, GHOST_FULL_RADIUS, GHOST_PEAK_MAX, GHOST_SHADE, GHOST_SMALL_SHARE,
    GHOST_TINT_MAX_RATIO, GLOW_BOOST_SHARE, HAZE_NEUTRAL,
};
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
    /// Every part whose light is the fraction times the energy (the streak is drawn by its own law).
    const ALL: Parts = Parts {
        glow: true,
        veil: true,
        ghosts: true,
        burst: true,
        streak: false,
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
                // The fractions sum to the floor (`GHOST_FRACTION_FLOOR`, 0.0012: no gain) and the ghosts
                // are 0.35 half-diagonals across: a clipped light of three pixels is assumed thousands of times
                // brighter than the clip level, so with more light or smaller ghosts their peaks would pass the
                // bound on a ghost's brightness (`GHOST_PEAK_MAX`, 0.003) and the light they spread would no
                // longer be the fraction times the light's energy (`ghost_peaks_are_bounded...` tests the bound).
                json!([
                    ghost_json(-0.5, 0.3, 0.35, 3.0, 0.00050, 450.0),
                    ghost_json(0.7, -0.2, 0.36, 3.0, 0.00040, 600.0),
                    ghost_json(1.4, 0.0, 0.35, 3.0, 0.00030, 520.0),
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
                    ghost_json(-0.4, 0.2, 0.30, 3.0, 0.0004, 470.0),
                    ghost_json(0.9, 0.1, 0.28, 3.0, 0.0003, 560.0),
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
/// of a light assumed `ev` EV above the plain 3 EV of headroom, of which the share `scale` is kept (the cap on
/// a photo's small lights, 1 for the scenes of most tests).
fn restored(x: [f64; 3], clip: f64, ev: f64, scale: f64) -> [f64; 3] {
    let m = x.iter().map(|v| v / clip).fold(f64::INFINITY, f64::min);
    let f = if m >= 0.98 { 1.0 } else { 0.0 };
    let plain = 8.0;
    let g = 1.0 + (plain + scale * (2f64.powf(3.0 + ev) - plain) - 1.0) * f;
    [x[0] * g, x[1] * g, x[2] * g]
}

/// The highlights of a pixel (the light's own extra headroom `ev`).
fn highlight(x: [f64; 3], clip: f64, ev: f64, scale: f64) -> [f64; 3] {
    let m = x.iter().map(|v| v / clip).fold(f64::NEG_INFINITY, f64::max);
    let k = smoothstep((m - 0.5) / 0.5);
    let r = restored(x, clip, ev, scale);
    [r[0] * k, r[1] * k, r[2] * k]
}

/// Sums over a scene: the restored light the glow and the veil read (the lights' extra headroom in EV, times
/// the glow's share of it, with the cap's scale), the highlights the ghosts read (the whole extra headroom with
/// the cap's scale), the highlights the streak and the starburst read (the whole extra headroom, whatever the
/// cap says), and the highlights of the one-pixel lights.
struct Sums {
    r: [f64; 3],
    h: [f64; 3],
    h_ghost: [f64; 3],
    h_compact: [f64; 3],
}

fn sums(scene: &Scene, clip: f64, scale: f64) -> Sums {
    let n = scene.w * scene.h;
    let ev = boost_map(scene, clip);
    let mut s = Sums {
        r: [0.0; 3],
        h: [0.0; 3],
        h_ghost: [0.0; 3],
        h_compact: [0.0; 3],
    };
    for (i, &ev_i) in ev.iter().enumerate().take(n) {
        let x = [
            f64::from(scene.data[i]),
            f64::from(scene.data[n + i]),
            f64::from(scene.data[2 * n + i]),
        ];
        let r = restored(x, clip, GLOW_BOOST_SHARE * ev_i, scale);
        let h = highlight(x, clip, ev_i, 1.0);
        let hg = highlight(x, clip, ev_i, scale);
        for c in 0..3 {
            s.r[c] += r[c];
            s.h[c] += h[c];
            s.h_ghost[c] += hg[c];
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

/// The shift of the coating of the lens's ghost `index`, in nm: the fractional parts of the multiples of the golden
/// ratio's conjugate over `GHOST_COATING_SHIFT_NM` (`docs/21`, L3, *As built (ghosts as light)*).
fn coating_shift_nm(index: usize) -> f64 {
    let u = (0.618_033_988_749_894_9 * (index as f64 + 1.0)).fract();
    GHOST_COATING_SHIFT_NM.0 + (GHOST_COATING_SHIFT_NM.1 - GHOST_COATING_SHIFT_NM.0) * u
}

/// The coating moved `shift_nm` toward longer wavelengths on the 10 nm grid (linear, the ends held).
fn shifted(coating: &[f64], shift_nm: f64) -> Vec<f64> {
    let n = coating.len();
    (0..n)
        .map(|k| {
            let t = (k as f64 - shift_nm / 10.0).clamp(0.0, (n - 1) as f64);
            let i = (t.floor() as usize).min(n - 1);
            let j = (i + 1).min(n - 1);
            coating[i] + (t - i as f64) * (coating[j] - coating[i])
        })
        .collect()
}

/// The tint of the lens's ghost `index`: its coating shifted, then [`ghost_tint`].
fn lens_ghost_tint(coating: &[f64], index: usize) -> [f64; 3] {
    ghost_tint(&shifted(coating, coating_shift_nm(index)))
}

/// The tint of a ghost's coating: [`tint`] with its spread between the channels limited to
/// `GHOST_TINT_MAX_RATIO` by a power below 1 (`docs/21`, L3, *As built (flares as light)*).
fn ghost_tint(coating: &[f64]) -> [f64; 3] {
    let t = tint(coating);
    let hi = t[0].max(t[1]).max(t[2]);
    let lo = t[0].min(t[1]).min(t[2]);
    if hi / lo <= GHOST_TINT_MAX_RATIO {
        return t;
    }
    let gamma = GHOST_TINT_MAX_RATIO.ln() / (hi / lo).ln();
    let p = t.map(|v| v.powf(gamma));
    let mean = (p[0] + p[1] + p[2]) / 3.0;
    p.map(|v| v / mean)
}

/// What L3 should return per channel, in units of `F`, from the lens data and the scene's sums (the streak is
/// not in it: it has its own law).
fn expected_energy(plan: &LensPlan, s: &Sums, strength: f64) -> [f64; 3] {
    let lens = plan.lens();
    let att = plan.attachment();
    let st = plan.flare_strengths();
    let anam = (plan.q() - 1.0) / (att.q_max - 1.0);
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        // The veil is a haze: `HAZE_NEUTRAL` of it is the mean of the channels.
        let mean_r = (s.r[0] + s.r[1] + s.r[2]) / 3.0;
        let mut e = st.glow * s.r[c] + st.veil * (s.r[c] + HAZE_NEUTRAL * (mean_r - s.r[c]));
        for (i, (g, fraction)) in lens.flare.ghosts.iter().zip(&st.ghosts).enumerate() {
            e += fraction * lens_ghost_tint(&g.coating, i)[c] * s.h_ghost[c];
        }
        e += st.burst * strength * s.h_compact[c];
        for (g, fraction) in att.ghosts.iter().zip(&st.attachment_ghosts) {
            e += anam * fraction * ghost_tint(&g.coating)[c] * s.h_ghost[c];
        }
        *o = e;
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

/// Checks the measured energy of a glow or veil run: the light is the same in total (`tol`), and each channel is
/// within a few per cent of the model's, because the veil and the glow's far tail are a haze, `HAZE_NEUTRAL` of
/// which is the mean of the channels, so the channels differ from the picture's by the share of the glow that is
/// far, which the model does not know.
fn check_haze_energy(label: &str, got: [f64; 3], want: [f64; 3], tol: f64) {
    check_energy(
        &format!("{label}, the three channels together"),
        [got.iter().sum::<f64>(); 3],
        [want.iter().sum::<f64>(); 3],
        tol,
    );
    check_energy(label, got, want, 0.03);
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
    let scale = planes.light_boost_scale();
    assert_eq!(
        scale, 1.0,
        "the standard scene's few lights are under the cap"
    );
    let want = expected_energy(&plan, &sums_for(&scene, size, scale), 1.0);
    if p.glow || p.veil {
        check_haze_energy(&format!("{axis:?} A{anamorphic}"), got, want, tol);
    } else {
        check_energy(&format!("{axis:?} A{anamorphic}"), got, want, tol);
    }
    // Energy is also spread, not lost: the output is finite and non-negative.
    assert!(out.iter().all(|v| v.is_finite() && *v >= -1e-6));
    got.iter().sum::<f64>()
}

/// The sums of a scene, with the one-pixel lights known by position of the standard scene.
fn sums_for(scene: &Scene, size: (usize, usize), scale: f64) -> Sums {
    let mut s = sums(scene, 1.0, scale);
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
        let h = highlight(at(x, y), 1.0, ev[py * scene.w + px], scale);
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
    // 3e-4: the ghosts of the faint coloured pixel B are left out under the prune budget (`GHOST_PRUNE_BUDGET`,
    // 4e-6 of the clip level), which with ghosts that hold 0.12 % of the light is 2e-4 of the light they spread.
    let tight = 3e-4;
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
    // The attachment's ghosts need Anamorphic; the axis may be either.
    for (label, p) in [(
        "attachment ghosts",
        Parts {
            att_ghosts: true,
            ..Parts::NONE
        },
    )] {
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
        part_energy(Parts::ALL, axis, anamorphic, 1e-4);
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
    let sm = sums_for(&scene, size, 1.0);
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
    // The light is the same in total whatever the pixels' clip level: the headroom times the glow's fraction of nine
    // pixels. (The output is f32, so nine identical pixels round alike: a few parts in a million at most.) Each
    // channel is within a few per cent: the glow's far tail is a haze that is neutral in part.
    let (sum_c, sum_o) = (e_c.iter().sum::<f64>(), e_o.iter().sum::<f64>());
    assert!(
        (sum_c / sum_o / headroom - 1.0).abs() < 5e-6,
        "{sum_c} against {sum_o}"
    );
    assert!((sum_c / 3.0 - 0.15 * 9.0 * headroom).abs() < 1e-6 * sum_c);
    for c in 0..3 {
        assert!(
            (e_c[c] / e_o[c] / headroom - 1.0).abs() < 0.03,
            "channel {c}: {} against {}",
            e_c[c],
            e_o[c]
        );
        assert!((e_c[c] / (0.15 * 9.0 * headroom) - 1.0).abs() < 0.03);
    }
    // The light's own pixels are not raised: they keep (1 - a) of themselves and what the glow gives back.
    let at = 171 * scene.w + 251;
    for c in 0..3 {
        let v = out_c[c * scene.w * scene.h + at];
        // (1 - a) of itself plus what the glow gives back (the glare makes a light look brighter than it was),
        // and nowhere near the light's restored light, which is what the flares spread.
        let keep = 1.0 - clipped.light_fraction();
        assert!(
            f64::from(v) > keep && f64::from(v) < 0.5 * headroom,
            "channel {c}: {v}"
        );
    }
}

#[test]
fn a_large_clipped_area_is_a_surface_and_keeps_the_plain_headroom() {
    // A clipped block 0.43 half-diagonals across is no light: its flare is its own light at 3 EV, while a small
    // light of the same brightness holds far more. The block's flares stay inside the frame.
    // Glow and veil only: a block this large is more cells than the source list holds (256), which would
    // leave the ghosts' light short.
    let lib = parts_library(Parts {
        glow: true,
        veil: true,
        ..Parts::NONE
    });
    let mut big = Scene::new(480, 320);
    big.fill(185, 105, 110, 110, [1.0; 3]);
    let size = (big.w, big.h);
    let plan = plan_of(&lib, 100.0, 0.0, 0.0, size, Axis::X, None, 1.0);
    let ev = boost_map(&big, 1.0);
    assert!(ev.iter().all(|e| *e == 0.0), "the block must get no boost");
    let planes = planes_of(&plan, &big);
    let out = run(&plan, &planes, &big);
    let got = measured_energy(&plan, &out, &big.data);
    let want = expected_energy(
        &plan,
        &sums_for(&big, size, planes.light_boost_scale()),
        1.0,
    );
    check_haze_energy("a big clipped block", got, want, 1e-6);
    // The light the glow and the veil spread is their fractions of the block's light at the plain 3 EV.
    let plain = (0.15 + 0.10) * 110.0 * 110.0 * 8.0;
    let mean = got.iter().sum::<f64>() / 3.0;
    assert!((mean / plain - 1.0).abs() < 1e-5, "{mean} against {plain}");
}

#[test]
fn many_lights_share_the_cap_and_the_energy_still_adds_up() {
    // Sixty lights of 3 px would add far more than the cap: every light's extra headroom is reduced alike for the
    // glow and the ghosts, whose haze would add up, so the photo flares like one of a few, and they still spread
    // exactly what the reduced lights hold. A single light is under the cap and is not reduced.
    let glow = parts_library(Parts {
        glow: true,
        veil: true,
        ..Parts::NONE
    });
    let mut one = Scene::new(480, 320);
    one.fill(230, 150, 3, 3, [1.0; 3]);
    let mut many = Scene::new(480, 320);
    for i in 0..60usize {
        many.fill(110 + 28 * (i % 10), 95 + 24 * (i / 10), 3, 3, [1.0; 3]);
    }
    let size = (480, 320);
    let mut scales = Vec::new();
    for scene in [&one, &many] {
        // The glow and the veil.
        let plan = plan_of(&glow, 100.0, 0.0, 0.0, size, Axis::X, None, 1.0);
        let planes = planes_of(&plan, scene);
        scales.push(planes.light_boost_scale());
        let out = run(&plan, &planes, scene);
        let got = measured_energy(&plan, &out, &scene.data);
        let want = expected_energy(
            &plan,
            &sums_for(scene, size, planes.light_boost_scale()),
            1.0,
        );
        check_haze_energy("the cap, glow", got, want, 1e-6);
    }
    assert_eq!(scales[0], 1.0);
    assert!(scales[1] < 0.5 && scales[1] > 0.0, "{scales:?}");
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
        let planes = planes_of(&plan, &scene);
        let out = run(&plan, &planes, &scene);
        let got = measured_energy(&plan, &out, &scene.data);
        let want = expected_energy(
            &plan,
            &sums_for(&scene, size, planes.light_boost_scale()),
            1.0,
        );
        check_haze_energy(&format!("{side} px"), got, want, 1e-6);
        per_pixel.push(got.iter().sum::<f64>() / 3.0 / (side * side) as f64);
    }
    // The 1 and 3 px lights both hold the whole extra headroom; then it falls.
    assert!(
        per_pixel.windows(2).all(|p| p[0] >= p[1] * (1.0 - 1e-6)),
        "the light per clipped pixel must not grow with the light's size: {per_pixel:?}"
    );
    assert!(per_pixel[3] < 0.5 * per_pixel[0], "{per_pixel:?}");
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
            // The shading (brighter toward the optical axis) moves the centroid along the line by a quarter of
            // `GHOST_SHADE` of the ghost's apothem (here 0.03 half-diagonals, 8.6 px) and more where the
            // attachment widens the ghost along the axis.
            assert!(
                off < 0.4 * GHOST_SHADE * 0.03 * 288.0 + 0.5,
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
        // The streak's halo (a fifth of its light, four times as wide as its core) reaches 12 to 80 px off the axis
        // with a few per cent of the light the line carries there.
        assert!(
            along > 20.0 * across.max(1e-12) && along > 0.1,
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

/// The shipped lenses, from the data folder.
fn shipped() -> LensLibrary {
    LensLibrary::load(&common::repo_root().join("data")).expect("the shipped lens data")
}

/// A plan of a shipped lens on a 960 x 640 frame (under 1024 px: the lens plane is the frame).
fn shipped_plan(
    lib: &LensLibrary,
    kind: LensKind,
    flares: f64,
    anamorphic: f64,
    axis: Axis,
) -> LensPlan {
    let amounts = LensAmounts {
        kind,
        flares,
        edges: 0.0,
        anamorphic,
    };
    let (w, h) = if axis == Axis::X {
        (960, 640)
    } else {
        (640, 960)
    };
    LensPlan::new(lib, &amounts, common::frame(w, h, axis), common::camera()).expect("plan")
}

/// A dim grey frame with one clipped light of 4 x 4 px off the centre.
fn light_on_grey(axis: Axis) -> Scene {
    let mut s = Scene::new(960, 640);
    s.fill(0, 0, 960, 640, [0.05; 3]);
    s.fill(660, 200, 4, 4, [1.0; 3]);
    if axis == Axis::Y { s.transposed() } else { s }
}

/// The flare light `terms` adds, per pixel (the mean of the channels), of a shipped lens on `scene`.
fn shipped_terms(plan: &LensPlan, scene: &Scene) -> Vec<f64> {
    let planes = planes_of(plan, scene);
    let t = ym_lens::flare_terms(
        plan,
        &planes,
        1,
        &whole(scene.w, scene.h, &scene.data),
        scene.rect(),
    )
    .expect("terms");
    let n = scene.w * scene.h;
    match t {
        None => vec![0.0; n],
        Some(t) => (0..n)
            .map(|i| (t[i] + t[n + i] + t[2 * n + i]) / 3.0)
            .collect(),
    }
}

#[test]
fn every_shipped_lens_draws_visible_ghosts_at_the_top_of_the_slider() {
    // The Default lens's ghosts hold 1e-5 of a light each (two multi-layer surfaces): drawn as the data say they
    // are 1e-8 of the clip level and the budget leaves them out. The top of Flares raises them (docs/21: twice
    // the data's fractions at most), so a clipped light on a dim frame has ghosts above what 8 bits can show
    // (a thousandth of the clip level is code 8 at the camera's exposure).
    let lib = shipped();
    let scene = light_on_grey(Axis::X);
    for kind in LensKind::ALL {
        let plan = shipped_plan(&lib, kind, 100.0, 0.0, Axis::X);
        let t = shipped_terms(&plan, &scene);
        // Away from the light itself, whose glow and starburst are not the ghosts, in the picture's far half.
        let ghost_peak = (0..scene.h)
            .flat_map(|y| (0..scene.w).map(move |x| (x, y)))
            .filter(|(x, y)| ((*x as f64 - 662.0).hypot(*y as f64 - 202.0)) > 100.0)
            .map(|(x, y)| t[y * scene.w + x])
            .fold(0.0, f64::max);
        assert!(
            ghost_peak > 1e-3,
            "{kind}: the brightest flare light away from the light is {ghost_peak:e} of the clip level"
        );
    }
    // Flares 0 adds nothing at all.
    let off = shipped_plan(&lib, LensKind::Default, 0.0, 100.0, Axis::X);
    assert!(shipped_terms(&off, &scene).iter().all(|v| *v == 0.0));
}

/// The streak's light along the axis of `scene` (960 x 640 frame, light at (662, 202) or its transpose): the
/// terms at Anamorphic 100 minus those at Anamorphic 0, per channel, at `da` pixels from the light's centre
/// along the axis (negative: to the left) and `dc` across, the largest of the three rows around `dc`.
fn streak_at(
    lib: &LensLibrary,
    kind: LensKind,
    axis: Axis,
    scene: &Scene,
    light: (usize, usize),
) -> impl Fn(isize, isize) -> [f64; 3] {
    let n = scene.w * scene.h;
    let (w, h) = (scene.w, scene.h);
    let terms = |anamorphic: f64| -> Vec<f64> {
        let plan = shipped_plan(lib, kind, 100.0, anamorphic, axis);
        let planes = planes_of(&plan, scene);
        ym_lens::flare_terms(&plan, &planes, 1, &whole(w, h, &scene.data), scene.rect())
            .expect("terms")
            .expect("flares")
    };
    let (with, without) = (terms(100.0), terms(0.0));
    move |da: isize, dc: isize| -> [f64; 3] {
        let (x, y) = if axis == Axis::X {
            (light.0 as isize + da, light.1 as isize + dc)
        } else {
            (light.0 as isize + dc, light.1 as isize + da)
        };
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            return [0.0; 3];
        }
        let i = y as usize * w + x as usize;
        std::array::from_fn(|c| with[c * n + i] - without[c * n + i])
    }
}

/// The streak's tint of the shipped attachment on the test camera, and the tail's colour (the tint at
/// `STREAK_TAIL_CHROMA` of its chroma, mean 1): the blue over red the far end of a streak has.
fn tail_blue_over_red(lib: &LensLibrary) -> f64 {
    let t = tint(&lib.attachment().streak.tint);
    let k = ym_lens::consts::STREAK_TAIL_CHROMA;
    (1.0 + k * (t[2] - 1.0)) / (1.0 + k * (t[0] - 1.0))
}

#[test]
fn the_streak_is_a_soft_line_of_light_that_starts_in_the_lights_colour_and_turns_to_a_pale_blue() {
    // Anamorphic 100 on the shipped lenses: a clipped light of 4 x 4 px on a dim frame gets a soft horizontal
    // line through it, on either axis, which is (a) the light's own colour next to it and a pale blue at the far
    // end (the coating's tint at half its chroma, never more), (b) everywhere beyond a few pixels dimmer than
    // `STREAK_PEAK_MAX_CLIP` clip levels, (c) a soft line, not a 1 px thread and not a haze, (d) fading steadily
    // with the distance.
    let lib = shipped();
    let tail = tail_blue_over_red(&lib);
    for axis in [Axis::X, Axis::Y] {
        let scene = light_on_grey(axis);
        let light = if axis == Axis::X {
            (662, 202)
        } else {
            (202, 662)
        };
        for kind in [LensKind::Default, LensKind::Vintage] {
            let at = streak_at(&lib, kind, axis, &scene, light);
            let line = |d: isize| -> [f64; 3] {
                let rows = [at(-d, -1), at(-d, 0), at(-d, 1)];
                std::array::from_fn(|c| rows.iter().map(|r| r[c]).fold(0.0, f64::max))
            };
            // Colour: next to the light the blue is within 25 % of the red (the light itself is neutral) and
            // it gains on the red as the line fades, to the tail's colour, which is a pale blue: the tint at
            // half its chroma, so well under the coating's own blue over red. (The streak of a light this
            // size is about 100 px long on this frame.)
            let near = line(12);
            let far = line(70);
            let (near_ratio, far_ratio) = (near[2] / near[0], far[2] / far[0]);
            assert!(
                (0.8..1.25).contains(&near_ratio),
                "{kind} {axis:?}: blue over red {near_ratio} 12 px from the light: {near:?}"
            );
            assert!(
                (far_ratio / tail - 1.0).abs() < 0.12 && far_ratio > 1.3,
                "{kind} {axis:?}: blue over red {far_ratio} at 70 px, the tail's is {tail}"
            );
            // Brightness: under the clip with a clear margin everywhere beyond 6 px, falling with the distance.
            let mut last = f64::INFINITY;
            for d in [8isize, 16, 32, 50, 70] {
                let v = line(d);
                let peak = v[0].max(v[1]).max(v[2]);
                assert!(
                    peak < STREAK_PEAK_MAX_CLIP,
                    "{kind} {axis:?}: the streak is {peak} of the clip level {d} px from the light"
                );
                assert!(peak < last, "{kind} {axis:?}: not falling at {d} px");
                last = peak;
                // A soft line: 2 px off the axis it is still most of the way up (the core is 1.6 px wide here),
                // 30 px off there is much less.
                let two: f64 = at(-d, 2).iter().cloned().fold(0.0, f64::max);
                assert!(
                    two > 0.35 * peak,
                    "{kind} {axis:?}: only {two} 2 px off the axis against {peak} on it at {d} px: a thread"
                );
                let off = at(-d, 30).iter().cloned().fold(0.0, f64::max);
                assert!(
                    peak > 5.0 * off,
                    "{kind} {axis:?}: {peak} on the axis against {off} 30 px away at {d} px"
                );
            }
            // And it is there: a clearly visible line a little way out.
            let v = line(32);
            assert!(
                v[1] > 2e-3,
                "{kind} {axis:?}: the streak adds only {} 32 px from the light",
                v[1]
            );
        }
    }
}

/// The least a streak's brightest channel may be called "under the clip with a margin": `STREAK_PEAK_PER_FRACTION`
/// times the shipped attachment's fraction (data version 4: 0.004), in clip levels.
const STREAK_PEAK_MAX_CLIP: f64 = 0.4 + 1e-9;

#[test]
fn the_streak_saturates_with_the_light_and_a_large_light_is_a_soft_band() {
    // A light of 400 clipped pixels (20 x 20) holds 25 times the light of one of 16 pixels (4 x 4): its streak
    // is not 25 times stronger, and it is a wider soft band, not a line.
    let lib = shipped();
    let small = light_on_grey(Axis::X);
    let mut big = Scene::new(960, 640);
    big.fill(0, 0, 960, 640, [0.05; 3]);
    big.fill(652, 192, 20, 20, [1.0; 3]);
    let s_small = streak_at(&lib, LensKind::Default, Axis::X, &small, (662, 202));
    let s_big = streak_at(&lib, LensKind::Default, Axis::X, &big, (662, 202));
    for d in [60isize, 120] {
        // Away from the block's own edge (10 px from its centre) along the axis.
        let a = s_small(-d, 0)[1];
        let b = s_big(-d, 0)[1];
        assert!(
            a > 0.0 && b < 3.0 * a,
            "{b} against {a} at {d} px: not saturating"
        );
        // The big light's band is wide: 12 px off the axis it is still a good part of the way up.
        let wide = s_big(-d, 12)[1] / s_big(-d, 0)[1].max(1e-12);
        let thin = s_small(-d, 12)[1] / s_small(-d, 0)[1].max(1e-12);
        assert!(
            wide > 3.0 * thin,
            "band {wide} against line {thin} at {d} px"
        );
    }
}

#[test]
fn a_dim_light_makes_a_short_faint_streak_and_a_bright_one_a_long_one() {
    let lib = shipped();
    // A light of 2 x 2 px, one of 4 x 4 and one of 12 x 12 on the same frame, one at a time.
    let reach_of = |side: usize| -> (f64, f64) {
        let mut scene = Scene::new(960, 640);
        scene.fill(0, 0, 960, 640, [0.02; 3]);
        scene.fill(662 - side / 2, 202 - side / 2, side, side, [1.0; 3]);
        let at = streak_at(&lib, LensKind::Default, Axis::X, &scene, (662, 202));
        // The distance to the left at which the streak falls under 2e-4 of the clip level, and the value at
        // 12 px.
        let mut reach = 0.0;
        for d in (10..640).step_by(4) {
            if at(-(d as isize), 0)[1] > 2e-4 {
                reach = f64::from(d);
            }
        }
        (reach, at(-12, 0)[1])
    };
    let (r2, v2) = reach_of(2);
    let (r4, v4) = reach_of(4);
    let (r12, v12) = reach_of(12);
    assert!(r2 <= r4 && r4 <= r12, "reach {r2} {r4} {r12}");
    assert!(v2 <= v4 && v4 <= v12 * 1.5, "peak {v2} {v4} {v12}");
    assert!(r2 < 400.0, "a light of 2 px makes a line {r2} px long");
}

/// The synthetic lenses with the shipped attachment's streak and nothing else of the attachment (no oval ghosts, no
/// ghost stretch): Anamorphic then adds the streaks and nothing more.
fn streak_lib() -> LensLibrary {
    let shipped: Value = serde_json::from_slice(
        &fs::read(common::repo_root().join("data/lenses/anamorphic.json")).unwrap(),
    )
    .unwrap();
    let streak = shipped["streak"].clone();
    library_with(&|_| {}, &|att| {
        att["ghosts"] = json!([]);
        att["ghost_stretch"] = json!(1.0);
        att["streak"] = streak.clone();
    })
}

/// The streak-only light along the rows of `scene` at `d` px left of the lights' column `x`, for rows `y0..y1`
/// (Anamorphic 100 minus 0, the brightest channel over its clip level).
fn streak_column(lib: &LensLibrary, scene: &Scene, x: usize, y0: usize, y1: usize) -> Vec<f64> {
    let at = streak_at(lib, LensKind::Classic, Axis::X, scene, (x, 0));
    (y0..y1)
        .map(|y| at(0, y as isize).iter().cloned().fold(0.0, f64::max))
        .collect()
}

#[test]
fn lights_in_a_row_make_one_streak_and_not_a_set_of_lines() {
    // Four lights on rows 6 px apart, spread along the axis (a row of lamps seen at an angle): one soft band
    // through them, not four lines. Seen across the axis well away from the lights the band has one maximum.
    let lib = streak_lib();
    let mut scene = Scene::new(960, 640);
    scene.fill(0, 0, 960, 640, [0.05; 3]);
    for (i, x) in [560usize, 600, 640, 680].into_iter().enumerate() {
        scene.fill(x, 200 + 6 * i, 4, 4, [1.0; 3]);
    }
    let col = streak_column(&lib, &scene, 500, 170, 250);
    let peak = col.iter().cloned().fold(0.0, f64::max);
    assert!(peak > 1e-3, "a streak at all: {peak}");
    let maxima = (1..col.len() - 1)
        .filter(|i| col[*i] > col[*i - 1] && col[*i] >= col[*i + 1] && col[*i] > 0.1 * peak)
        .count();
    assert_eq!(maxima, 1, "{maxima} lines across the rows: {col:?}");
}

#[test]
fn a_large_bright_area_makes_no_streak_and_a_glint_in_it_none_of_its_own() {
    // A lit window of 240 x 120 px with a separate clipped glint of 4 x 4 px inside it (a dark gap round it), on
    // a dark frame: a surface does not draw a band across the picture, and the glint inside it belongs to it.
    // Beyond the window along the axis there is no streak light at all.
    let lib = streak_lib();
    let mut scene = Scene::new(960, 640);
    scene.fill(0, 0, 960, 640, [0.03; 3]);
    scene.fill(560, 250, 240, 120, [1.0; 3]);
    scene.fill(566, 294, 10, 10, [0.1; 3]);
    scene.fill(569, 297, 4, 4, [1.0; 3]);
    let at = streak_at(&lib, LensKind::Classic, Axis::X, &scene, (680, 310));
    for d in [-450isize, -380, 260, 280] {
        for dc in [-60isize, -20, 0, 20, 60] {
            let v = at(d, dc);
            assert!(
                v.iter().all(|x| x.abs() < 1e-5),
                "streak light {v:?} at {d} px along and {dc} across a window"
            );
        }
        // The glint's own row.
        let v = at(d, -11);
        assert!(
            v.iter().all(|x| x.abs() < 1e-5),
            "streak light {v:?} on the glint's row at {d} px"
        );
    }
}

#[test]
fn a_crowd_of_equal_lights_makes_a_few_faint_streaks_and_not_a_ladder() {
    // Twelve lights of 6 x 6 px of the same brightness, 40 px apart across the axis, spread along it: at most
    // `STREAK_MAX_LIGHTS` of them draw a streak, the first the strongest, each after it fainter.
    let lib = streak_lib();
    let mut scene = Scene::new(960, 640);
    scene.fill(0, 0, 960, 640, [0.03; 3]);
    let rows: Vec<usize> = (0..12).map(|i| 60 + 45 * i).collect();
    for (i, y) in rows.iter().enumerate() {
        scene.fill(500 + (i * 53) % 300, *y, 6, 6, [1.0; 3]);
    }
    let at = streak_at(&lib, LensKind::Classic, Axis::X, &scene, (0, 0));
    let mut levels: Vec<f64> = Vec::new();
    for (i, y) in rows.iter().enumerate() {
        let x = 500 + (i * 53) % 300;
        // 120 px to the left of the light, on its row.
        let v = (0..3)
            .map(|k| {
                let row = *y as isize + 3;
                at(x as isize - 120, row)[k]
            })
            .fold(0.0, f64::max);
        levels.push(v);
    }
    let drawn = levels.iter().filter(|v| **v > 1e-3).count();
    assert!(
        (1..=ym_lens::consts::STREAK_MAX_LIGHTS).contains(&drawn),
        "{drawn} streaks of {levels:?}"
    );
    // The strongest of a crowd is faint (the frame's streaks share `STREAK_TOTAL_CAP`), and the weakest of the drawn
    // ones is under half of it.
    let top = levels.iter().cloned().fold(0.0, f64::max);
    assert!(top < 0.2, "the first of a crowd is faint: {top}");
    let weakest = levels
        .iter()
        .cloned()
        .filter(|v| *v > 1e-3)
        .fold(f64::INFINITY, f64::min);
    if drawn > 1 {
        assert!(weakest < 0.5 * top, "a ladder: {levels:?}");
    }
}

#[test]
fn the_streak_ends_smoothly_to_nothing() {
    // The last tenth of a streak's length falls to 0 with no step: the largest difference between neighbouring
    // pixels there is a tenth of a code at the clip level at most, the streak falls all the way and is gone
    // at its reach.
    let lib = streak_lib();
    for axis in [Axis::X, Axis::Y] {
        let scene = light_on_grey(axis);
        let light = if axis == Axis::X {
            (662, 202)
        } else {
            (202, 662)
        };
        let at = streak_at(&lib, LensKind::Classic, axis, &scene, light);
        let v: Vec<f64> = (0..=660isize).map(|d| at(-d, 0)[2]).collect();
        let peak = v[8];
        assert!(peak > 0.01, "{axis:?}: a streak: {peak}");
        // Where the streak is gone.
        let end = v.iter().rposition(|x| *x > 1e-9).expect("an end");
        assert!(end < 660, "{axis:?}: the streak still goes on at 660 px");
        assert!(end > 40, "{axis:?}: the streak is only {end} px long");
        // The last tenth: no step between neighbouring pixels.
        let from = (end as f64 * 0.9) as usize;
        let step = (from..=end)
            .map(|d| (v[d + 1] - v[d]).abs())
            .fold(0.0, f64::max);
        assert!(
            step < 1.0 / 255.0 / 10.0,
            "{axis:?}: a step of {step} near the end of the streak at {end} px"
        );
        for d in from..end {
            assert!(
                v[d + 1] <= v[d] * 1.0001 + 1e-12,
                "{axis:?}: not falling at {d} px"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Ghosts as light: soft edges, pale tints, a bound on the brightness

/// A lens with one ghost (`m`, `s`, `soft`, `fraction`, the coating's centre in nm), nothing else, round, with
/// a flat rim, on the synthetic folder.
fn one_ghost_library(m: f64, s: f64, soft: f64, fraction: f64, nm: f64) -> LensLibrary {
    library_with(
        &|lens| {
            lens["flare"]["glow"]["fraction"] = json!(0.0);
            lens["flare"]["veil"]["fraction"] = json!(0.0);
            lens["flare"]["starburst"]["fraction"] = json!(0.0);
            lens["flare"]["aperture"] =
                json!({ "blades": 0, "rotation_deg": 0.0, "roundness": 1.0 });
            lens["flare"]["ghosts"] = json!([soft_ghost(m, 0.0, s, 4.0, fraction, soft, nm)]);
            lens["bokeh"]["rim"] = json!(vec![1.0; 16]);
        },
        &|att| {
            att["streak"]["fraction"] = json!(0.0);
            att["ghosts"] = json!([]);
        },
    )
}

/// The flare terms (per channel) of `scene` on a 480 x 320 frame under `lib` at Flares 100.
fn terms_of(lib: &LensLibrary, scene: &Scene) -> Vec<f64> {
    let plan = plan_of(lib, 100.0, 0.0, 0.0, (scene.w, scene.h), Axis::X, None, 1.0);
    let planes = planes_of(&plan, scene);
    ym_lens::flare_terms(
        &plan,
        &planes,
        1,
        &whole(scene.w, scene.h, &scene.data),
        scene.rect(),
    )
    .expect("terms")
    .expect("flares")
}

#[test]
fn a_ghosts_edge_is_a_ramp_of_a_quarter_of_its_radius_whatever_the_data_say() {
    // The data's `soft` of 0.0 (a hard disc) is drawn with a step of GHOST_SOFT_MIN radii: the 10 to 90 % width
    // of the edge is then 2.56 x 0.10 = 26 % of the radius.
    let (s, centre) = (0.15, (240.0, 160.0));
    let lib = one_ghost_library(-0.5, s, 0.0, 0.004, 520.0);
    let mut scene = Scene::new(480, 320);
    // A bright unclipped light, so that the light's own boost does not enter: 3 x 3 px at 0.85.
    scene.fill(300, 160, 3, 3, [0.85; 3]);
    let t = terms_of(&lib, &scene);
    let n = scene.w * scene.h;
    let row = |x: usize| t[n + 161 * scene.w + x];
    // The ghost's centre is at -0.5 x (the light's offset from the frame's centre).
    let hd = 0.5 * ((480.0f64).powi(2) + (320.0f64).powi(2)).sqrt();
    let cx = centre.0 - 0.5 * (301.5 - centre.0);
    let radius = s * hd;
    let inner = (0..40)
        .map(|k| row((cx - 0.5 * radius) as usize + k))
        .sum::<f64>()
        / 40.0;
    // The edge on the left of the centre: where the row crosses 90 % and 10 % of the inside.
    let (mut at90, mut at10) = (f64::NAN, f64::NAN);
    for x in ((cx - 1.6 * radius) as usize..(cx - 0.5 * radius) as usize).rev() {
        if at90.is_nan() && row(x) < 0.9 * inner {
            at90 = x as f64;
        }
        if at10.is_nan() && row(x) < 0.1 * inner {
            at10 = x as f64;
        }
    }
    let width = at90 - at10;
    assert!(
        width >= 0.22 * radius,
        "the edge is {width} px wide for a ghost of radius {radius} px (inside {inner})"
    );
}

#[test]
fn a_ghost_is_brighter_on_the_side_that_faces_the_axis() {
    // The image of the aperture is vignetted by the barrel: a ghost is 1 + GHOST_SHADE times its mean at the edge
    // that faces the optical axis and 1 - GHOST_SHADE at the other, and carries the same light as without it.
    let (s, centre) = (0.2, (240.0, 160.0));
    let lib = one_ghost_library(-0.5, s, 0.1, 0.0004, 520.0);
    let mut scene = Scene::new(480, 320);
    // A clipped light: a light of 0.85 holds so little that its ghost is under the budget and is not drawn.
    scene.fill(300, 160, 3, 3, [1.0; 3]);
    let t = terms_of(&lib, &scene);
    let n = scene.w * scene.h;
    let hd = 0.5 * ((480.0f64).powi(2) + (320.0f64).powi(2)).sqrt();
    let cx = centre.0 - 0.5 * (301.5 - centre.0);
    let radius = s * hd;
    // The ghost lies left of the optical axis (the light is on the right): its axis side is its right side.
    let mean = |x0: f64, x1: f64| {
        (x0 as usize..x1 as usize)
            .map(|x| t[n + 161 * scene.w + x])
            .sum::<f64>()
            / (x1 - x0)
    };
    let toward = mean(cx + 0.35 * radius, cx + 0.55 * radius);
    let away = mean(cx - 0.55 * radius, cx - 0.35 * radius);
    assert!(
        toward > 1.25 * away,
        "toward the axis {toward:e}, away from it {away:e}"
    );
}

#[test]
fn ghost_tints_are_pale() {
    // A coating that reflects mostly one end of the spectrum has a tint that differs several times between the
    // channels; the ghost it makes differs by at most GHOST_TINT_MAX_RATIO.
    let raw = tint(&coating(430.0));
    let spread = |t: [f64; 3]| t[0].max(t[1]).max(t[2]) / t[0].min(t[1]).min(t[2]);
    assert!(
        spread(raw) > 2.0 * GHOST_TINT_MAX_RATIO,
        "the coating is not saturated: {raw:?}"
    );
    let lib = one_ghost_library(-0.5, 0.15, 0.3, 0.004, 430.0);
    let mut scene = Scene::new(480, 320);
    scene.fill(300, 160, 3, 3, [0.85; 3]);
    let t = terms_of(&lib, &scene);
    let n = scene.w * scene.h;
    // The ghost's middle: its centre is at 240 - 0.5 x 61.5 = 209 on row 160.
    let at = |c: usize| t[c * n + 161 * scene.w + 209];
    let got = [at(0), at(1), at(2)];
    assert!(got.iter().all(|v| *v > 0.0), "no ghost light: {got:?}");
    assert!(
        spread(got) <= GHOST_TINT_MAX_RATIO + 1e-6,
        "the ghost's channels differ {} times, at most {GHOST_TINT_MAX_RATIO} wanted: {got:?}",
        spread(got)
    );
}

#[test]
fn a_ghost_is_never_brighter_than_the_bound_and_a_faint_one_is_untouched() {
    // A clipped light of 3 px is assumed thousands of times brighter than the clip level: a ghost of
    // 10 % of the light's energy would be many times the clip level. It stays under GHOST_PEAK_MAX (a ghost of
    // apothem 0.15 may be as bright as that), and with an unclipped light of 0.85 the same ghost is as the
    // data say (its light is the fraction times the light's energy, to 1e-6).
    let (s, fraction) = (0.15, 0.1);
    let lib = one_ghost_library(-0.5, s, 0.3, fraction, 520.0);
    let mut bright = Scene::new(480, 320);
    bright.fill(300, 160, 3, 3, [1.0; 3]);
    let t = terms_of(&lib, &bright);
    let peak = t.iter().cloned().fold(0.0, f64::max);
    let cap = GHOST_PEAK_MAX * (s / GHOST_FULL_RADIUS).clamp(GHOST_SMALL_SHARE, 1.0);
    assert!(
        peak <= 1.3 * cap && peak >= 0.5 * cap,
        "the ghost's peak is {peak} of the clip level, the bound is {cap}"
    );
    let mut faint = Scene::new(480, 320);
    faint.fill(300, 160, 3, 3, [0.85; 3]);
    let plan = plan_of(&lib, 100.0, 0.0, 0.0, (480, 320), Axis::X, None, 1.0);
    let planes = planes_of(&plan, &faint);
    let out = run(&plan, &planes, &faint);
    let got = measured_energy(&plan, &out, &faint.data);
    let want = expected_energy(&plan, &sums_for(&faint, (480, 320), 1.0), 1.0);
    check_energy("a faint ghost", got, want, 1e-6);
}

// ---------------------------------------------------------------------------------------------------------
// The light budget of a frame with no lights

/// A uniform frame loses only the light the glow's tail and the veil send beyond its edges (`docs/21` section 4,
/// L3: light that leaves the frame is lost), on the shipped lenses. That is what a lens does, too: the veiling
/// glare a pixel receives is the sum of the scene's light within the frame, so the corners get less than the
/// centre and the frame's mean goes down a little. The picture also gives up the share of its light that the
/// ghosts, the starburst and the glow's floor take (`FlareStrengths`: 1.0 % of it at Default, where no light
/// exists to draw them from), so the bounds are about twice the measured losses at Flares 100 (Default 1.1 %,
/// Vintage 3.6 %).
#[test]
fn a_uniform_frame_loses_only_the_light_that_leaves_it() {
    let lib = LensLibrary::load(&common::repo_root().join("data")).expect("the shipped lens data");
    let (w, h) = (600usize, 400usize);
    for (kind, max_loss) in [(LensKind::Default, 0.03), (LensKind::Vintage, 0.08)] {
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
