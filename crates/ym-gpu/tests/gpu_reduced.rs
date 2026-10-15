//! G-07: the reduced fit-view path on the device: the sigma plane (`shaders/sigma.wgsl`), the k x k
//! binning (`shaders/bin.wgsl`) and the sigma it leaves, against the CPU functions of `ym-core`
//! (copied into `tests/g07/mod.rs`; `crates/ym-core/tests/gpu_reduced.rs` runs the originals), and the
//! whole group `GpuReduced` against `reduced_bayer`.
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/gpu-plan.md` X-16):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | sigma plane: 4 CFA phases, even and odd frames, uneven levels, a model with offsets | relative to the CPU's at most [`SIGMA_BOUND`] | see the printout |
//! | binning, k = 2, 4, 8, four phases, odd frame sizes, slabs of any height | at most [`BIN_BOUND`] against the CPU (white 1.0) | see the printout |
//! | binning with the vignetting gain | at most [`BIN_VIG_BOUND`] | see the printout |
//! | binned sigma | relative at most [`SIGMA_BOUND`] | see the printout |
//! | slab heights, tiles: the same bits | bit-identical | 0 differing |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |

mod common;
#[path = "g06/mod.rs"]
mod g06;
#[path = "g07/mod.rs"]
mod g07;

use common::exact::assert_poison_neutral;
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use g07::{
    RawScene, compare_scaled, noise_model, on_device, ref_bin_mosaic, ref_binned_sigma,
    ref_gain_plane, ref_noise_factor, ref_sigma_params, ref_sigma_plane, uneven_levels,
    vignette_profile,
};
use ym_gpu::cpu::normalize::Levels;
use ym_gpu::cpu::{Cfa, Placement, Rect, synth};
use ym_gpu::device::StorageMode;
use ym_gpu::image::GpuImage;
use ym_gpu::kernels::bin::{self, BinParams};
use ym_gpu::kernels::prep::{DeviceRaw, VignetteTable};
use ym_gpu::kernels::sigma;
use ym_gpu::stream::PassList;

/// Largest relative difference `|gpu - cpu| / max(cpu, 1e-6)` of the sigma plane (measured 0: the
/// CPU's bits) and of the binned sigma (measured 2.3e-7, the root and the division of the device).
/// The plan's number is 1e-5; the bound is twice the larger measured value.
const SIGMA_BOUND: f32 = 5e-7;
/// The same on a hardware adapter: the plan's number for the sigma plane (Metal's square root and
/// division are not correctly rounded), to be lowered from the Metal numbers.
const SIGMA_HW_BOUND: f32 = 1e-5;
/// Largest `|gpu - cpu|` of the binning (white 1.0; measured 0: the CPU's bits). The plan's number is
/// 1e-6, kept: it is the margin for a platform whose division and fused multiply-add differ.
const BIN_BOUND: f32 = 1e-6;
/// The same with the vignetting gain (values up to 4 after the gains; measured 0: the gain is formed
/// in double-floats, `front_vig.wgsl`). One ulp is allowed.
const BIN_VIG_BOUND: f32 = 1.2e-7;
/// The same on a hardware adapter: the plan's number for `bin`.
const BIN_VIG_HW_BOUND: f32 = 1e-6;

fn hw(dev: &TestGpu, software: f32, hardware: f32) -> f32 {
    if dev.gpu.info().is_software {
        software
    } else {
        hardware
    }
}

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

const G0: [f32; 3] = [2.0, 1.0, 1.6];

fn rel(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = if x == y {
            0.0
        } else {
            (x - y).abs() / y.abs().max(1e-6)
        };
        if d.is_nan() || d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

#[test]
fn the_sigma_plane_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let model = noise_model();
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for (n, cfa) in Cfa::ALL.into_iter().enumerate() {
        for (w, h) in [(96, 64), (131, 77), (33, 41)] {
            let levels = if (n + w) % 2 == 0 {
                uneven_levels()
            } else {
                g07::full_levels()
            };
            let scene = RawScene::new(w, h, cfa, 3 + n as u64, &levels);
            let want = ref_sigma_plane(&scene, &levels, G0, &model);
            let params = ref_sigma_params(&scene, G0, &model);
            let s2 = scene.clone();
            let got = on_device(dev, move |ctx| {
                let raw = DeviceRaw::upload(ctx, "s", s2.width, s2.height, s2.cfa, &s2.data, None)?;
                sigma::sigma_plane(ctx, &raw, &levels, &params)
            });
            let (e, at) = rel(&got, &want);
            if e > worst {
                worst = e;
                worst_at = format!("{} {w}x{h} cell {at}", cfa.name());
            }
        }
    }
    eprintln!("[G-07] sigma plane: worst relative difference {worst:e} ({worst_at})");
    assert!(
        worst <= hw(dev, SIGMA_BOUND, SIGMA_HW_BOUND),
        "{worst:e} {worst_at}"
    );
}

fn natural_scene_mosaic(w: usize, h: usize, cfa: Cfa, seed: u64) -> ym_gpu::cpu::Mosaic {
    let img = synth::natural(w, h, seed);
    let mut m = synth::mosaic_from_rgb(&img, cfa);
    // Values past white too: the binning sums whatever it is given.
    for (i, v) in m.data.iter_mut().enumerate() {
        *v = *v * 1.5 - 0.02 * ((i % 7) as f32);
    }
    m
}

#[test]
fn the_binning_matches_the_cpu_for_every_phase_and_slab_height() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut worst_vig = 0.0f32;
    let profile = vignette_profile();
    let table = VignetteTable::from_profile(&profile).expect("table");
    for cfa in Cfa::ALL {
        for (w, h) in [(160, 96), (131, 77), (67, 130)] {
            for k in [2usize, 4, 8] {
                let m = natural_scene_mosaic(w, h, cfa, 5);
                let (ow, oh) = (w / k, h / k);
                if ow == 0 || oh == 0 {
                    continue;
                }
                let want = ref_bin_mosaic(&m, k);
                for slab in [0usize, 3, 1000] {
                    let (m2, t2) = (m.clone(), table.clone());
                    let got = on_device(dev, move |ctx| bin::bin_mosaic(ctx, &m2, k, None, slab));
                    worst = worst.max(g06::compare(&got.data, &want.data).max);
                    // With the profile: the CPU multiplies the binned planes by the gain plane.
                    let gains = ref_gain_plane(&profile, w, h, k, ow, oh);
                    let mut want_v = want.clone();
                    let n = ow * oh;
                    for plane in want_v.data.chunks_mut(n) {
                        for (v, g) in plane.iter_mut().zip(&gains) {
                            *v *= *g;
                        }
                    }
                    let (m3, t3) = (m.clone(), t2);
                    let got_v =
                        on_device(dev, move |ctx| bin::bin_mosaic(ctx, &m3, k, Some(t3), slab));
                    worst_vig = worst_vig.max(compare_scaled(&got_v.data, &want_v.data).0);
                }
            }
        }
    }
    eprintln!("[G-07] bin: worst {worst:e}; with the vignetting gain (scaled) {worst_vig:e}");
    assert!(worst <= BIN_BOUND, "{worst:e}");
    assert!(
        worst_vig <= hw(dev, BIN_VIG_BOUND, BIN_VIG_HW_BOUND),
        "{worst_vig:e}"
    );
}

#[test]
fn slabs_of_every_height_give_the_bits_of_one() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (130, 100);
    let m = natural_scene_mosaic(w, h, Cfa::GBRG, 9);
    let table = VignetteTable::from_profile(&vignette_profile()).expect("table");
    for k in [2usize, 4, 8] {
        let (m1, t1) = (m.clone(), table.clone());
        let one = on_device(dev, move |ctx| bin::bin_mosaic(ctx, &m1, k, Some(t1), 0));
        common::exact::assert_all_finite("bin", &one.data);
        for slab in [1usize, 2, 5] {
            let (m2, t2) = (m.clone(), table.clone());
            let many = on_device(dev, move |ctx| bin::bin_mosaic(ctx, &m2, k, Some(t2), slab));
            common::exact::assert_bit_equal(
                &format!("bin k={k} slab {slab}"),
                &many.data,
                &one.data,
            );
        }
    }
}

#[test]
fn the_binned_sigma_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let table = VignetteTable::from_profile(&profile).expect("table");
    let mut worst = 0.0f32;
    for (w, h) in [(160, 96), (131, 77)] {
        let (hw, hh) = (w / 2, h / 2);
        let mut rng = synth::Rng::new(4);
        let sigma_plane: Vec<f32> = (0..hw * hh)
            .map(|_| 0.001 + 0.02 * rng.next_f32())
            .collect();
        for k in [2usize, 4, 8] {
            let (ow, oh) = (w / k, h / k);
            for vig in [false, true] {
                let gains = vig.then(|| ref_gain_plane(&profile, w, h, k, ow, oh));
                let want = ref_binned_sigma(&sigma_plane, hw, hh, gains.as_deref(), k, ow, oh);
                let params = BinParams {
                    k,
                    cfa: Cfa::RGGB,
                    frame_w: w,
                    frame_h: h,
                    vignette: vig.then(|| table.clone()),
                };
                let sp = sigma_plane.clone();
                let got = on_device(dev, move |ctx| {
                    let plane =
                        GpuImage::upload(ctx, "sig", hw, hh, 1, &sp, Placement::whole(hw, hh))?;
                    let mut passes = PassList::new(ctx, "bin sigma");
                    let out =
                        bin::encode_bin_sigma(&mut passes, &plane, &params, ref_noise_factor(k))?;
                    let read = out.encode_read(&mut passes)?;
                    let done = passes.submit()?.finish()?;
                    out.take(&done, read)
                });
                worst = worst.max(rel(&got, &want).0);
            }
        }
    }
    eprintln!("[G-07] binned sigma: worst relative difference {worst:e}");
    assert!(worst <= hw(dev, SIGMA_BOUND, SIGMA_HW_BOUND), "{worst:e}");
}

#[test]
fn the_reduced_kernels_are_poison_clean() {
    if f32_device().is_none() {
        return;
    }
    let levels = uneven_levels();
    let scene = RawScene::new(100, 60, Cfa::RGGB, 7, &levels);
    let params = ref_sigma_params(&scene, G0, &noise_model());
    let m = natural_scene_mosaic(100, 60, Cfa::RGGB, 3);
    let table = VignetteTable::from_profile(&vignette_profile()).expect("table");
    let out = assert_poison_neutral("sigma and bin", |gpu| {
        let s = scene.clone();
        let (m2, t2) = (m.clone(), table.clone());
        gpu.run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |ctx, _| {
                let raw = DeviceRaw::upload(ctx, "s", s.width, s.height, s.cfa, &s.data, None)?;
                let mut v = sigma::sigma_plane(ctx, &raw, &levels, &params)?;
                v.extend(bin::bin_mosaic(ctx, &m2, 4, Some(t2), 3)?.data);
                Ok(v)
            },
        )
        .expect("job")
    });
    assert!(out.is_some());
}

#[test]
fn the_bindings_fit_the_design_binding() {
    // A 50 MP frame: the sigma plane, and a slab of 1024 rows of the full width.
    assert!(ym_gpu::tiles::fits_design_binding(
        sigma::max_binding_bytes(8658, 5772)
    ));
    assert!(ym_gpu::tiles::fits_design_binding(bin::max_binding_bytes(
        9504,
        1024 + 16
    )));
    assert_max_binding_bytes("bin slab", Rect::new(0, 0, 4096, 4096), |r| {
        bin::max_binding_bytes(r.w, r.h / 8)
    });
}

// ---------------------------------------------------------------------------------------------
// The whole group
// ---------------------------------------------------------------------------------------------

/// Largest `|gpu - cpu| / max(1, |cpu|)` of the whole reduced group against the CPU composition of
/// the stages (R1, R4, R5, the binning, the vignetting gain, the sigma, the bilateral denoise and R9
/// at `1/k`), on frames of a few hundred photosites. The plan's number for the group is the sum of the
/// kernels' (about 1e-5 with the denoise); the bound is twice the first measured value (9.8e-7, R5,
/// the vignetting gain, the denoise and R9 together).
const GROUP_BOUND: f32 = 2e-6;
/// The same on a hardware adapter: the plan's sum for the group (about 1e-5).
const GROUP_HW_BOUND: f32 = 1e-5;

/// The pieces a case of the group switches on.
#[derive(Clone, Copy, Debug)]
struct Case {
    k: usize,
    recovery: bool,
    vignette: bool,
    denoise: bool,
    lens: bool,
}

/// The same at 50 MP on a synthetic frame with a hard-edged clipped blob in the far corner: R9's
/// position (an `f32` displacement of tens of binned pixels, see `gpu_lens.rs` `CORNER_BOUND`) limits
/// it. Measured per case: 7.0e-7 without R9, 5.7e-6 for R9 alone at k = 8, 9.5e-6 for the whole group
/// at k = 8 and 2.0e-5 at k = 4; the bound is twice the largest.
const GROUP_BOUND_50MP: f32 = 4e-5;
/// The same on a hardware adapter (see `gpu_lens.rs`, `HW_CORNER_BOUND`).
const GROUP_HW_BOUND_50MP: f32 = 8e-5;

const DENOISE_SCALE: f32 = 0.8;

fn lens_tags() -> ym_gpu::cpu::lens::LensTags {
    ym_gpu::cpu::lens::LensTags {
        vignetting: vec![],
        distortion: vec![0, 120, 400, 900, 1600, 2500],
        ca: vec![0, 5000, 12000, 20000, 0, -5000, -12000, -20000],
    }
}

fn lens_geometry(ow: usize, oh: usize) -> ym_gpu::cpu::lens::LensGeometry {
    ym_gpu::cpu::lens::LensGeometry::new(
        ow,
        oh,
        &lens_tags(),
        ym_gpu::cpu::lens::LensSwitches::default(),
    )
    .expect("lens")
    .with_auto_crop()
}

/// The CPU composition of the reduced path (`reduced_bayer` without the file plumbing).
fn cpu_reduced(
    scene: &RawScene,
    levels: &Levels,
    c: Case,
    profile: &ym_gpu::cpu::vignette::VignetteProfile,
) -> ym_gpu::cpu::RgbImage {
    let (mut m, mask) = g07::cpu_front(scene, levels, G0, None);
    if c.recovery {
        g07::cpu_highlights(&mut m, &mask);
    }
    let mut img = ref_bin_mosaic(&m, c.k);
    let (ow, oh) = (img.width, img.height);
    let gains = c
        .vignette
        .then(|| ref_gain_plane(profile, scene.width, scene.height, c.k, ow, oh));
    if let Some(g) = &gains {
        for plane in img.data.chunks_mut(ow * oh) {
            for (v, g) in plane.iter_mut().zip(g) {
                *v *= *g;
            }
        }
    }
    if c.denoise {
        let sp = ref_sigma_plane(scene, levels, G0, &noise_model());
        let mut s = ref_binned_sigma(
            &sp,
            scene.width / 2,
            scene.height / 2,
            gains.as_deref(),
            c.k,
            ow,
            oh,
        );
        s.iter_mut().for_each(|v| *v *= DENOISE_SCALE);
        img = ym_gpu::cpu::denoise::bilateral_opponent(
            &img,
            &ym_gpu::cpu::NoiseSigma::Plane(&s),
            &ym_gpu::cpu::denoise::BilateralParams::default(),
        );
    }
    if c.lens {
        let geo = lens_geometry(ow, oh);
        let (cw, ch) = geo.output_size();
        img = ym_gpu::cpu::resample::resample_lens(&img, &geo, Rect::whole(cw, ch))
            .expect("resample");
    }
    img
}

fn reduced_params(
    scene: &RawScene,
    levels: Levels,
    c: Case,
    profile: &ym_gpu::cpu::vignette::VignetteProfile,
) -> ym_gpu::chain::ReducedParams {
    let (ow, oh) = (scene.width / c.k, scene.height / c.k);
    ym_gpu::chain::ReducedParams {
        k: c.k,
        levels,
        g0: G0,
        highlight_recovery: c.recovery,
        vignette: c
            .vignette
            .then(|| VignetteTable::from_profile(profile).expect("table")),
        noise: ref_sigma_params(scene, G0, &noise_model()),
        noise_factor: ref_noise_factor(c.k),
        denoise: c.denoise.then(|| ym_gpu::chain::ReducedDenoise {
            scale: DENOISE_SCALE,
            params: ym_gpu::cpu::denoise::BilateralParams::default(),
        }),
        lens: c.lens.then(|| {
            ym_gpu::kernels::lens::LensTable::from_geometry(&lens_geometry(ow, oh)).expect("table")
        }),
    }
}

fn run_group(
    dev: &'static TestGpu,
    scene: &RawScene,
    levels: Levels,
    c: Case,
    profile: &ym_gpu::cpu::vignette::VignetteProfile,
    slab_rows: usize,
) -> ym_gpu::cpu::RgbImage {
    let p = reduced_params(scene, levels, c, profile);
    let s = scene.clone();
    on_device(dev, move |ctx| {
        let raw = DeviceRaw::upload(
            ctx,
            "r",
            s.width,
            s.height,
            s.cfa,
            &s.data,
            s.qstep.as_deref(),
        )?;
        ym_gpu::chain::GpuReduced::run_with_slabs(
            ctx,
            &raw,
            &p,
            &ym_gpu::queue::CancelToken::new(),
            slab_rows,
        )?
        .read_rgb(ctx)
    })
}

#[test]
fn the_reduced_group_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let mut worst = (0.0f32, String::new());
    let mut cases = Vec::new();
    for k in [2usize, 4, 8] {
        cases.push(Case {
            k,
            recovery: false,
            vignette: false,
            denoise: false,
            lens: false,
        });
        cases.push(Case {
            k,
            recovery: true,
            vignette: false,
            denoise: false,
            lens: false,
        });
        cases.push(Case {
            k,
            recovery: true,
            vignette: true,
            denoise: true,
            lens: true,
        });
    }
    cases.push(Case {
        k: 4,
        recovery: false,
        vignette: true,
        denoise: true,
        lens: false,
    });
    cases.push(Case {
        k: 2,
        recovery: false,
        vignette: false,
        denoise: false,
        lens: true,
    });
    for (n, cfa) in Cfa::ALL.into_iter().enumerate() {
        for (w, h) in [(160usize, 96usize), (131, 77)] {
            let levels = if (n + w) % 2 == 0 {
                uneven_levels()
            } else {
                g07::full_levels()
            };
            let scene = RawScene::new(w, h, cfa, 11 + n as u64, &levels);
            for c in &cases {
                if w / c.k < 8 || h / c.k < 8 {
                    continue;
                }
                let want = cpu_reduced(&scene, &levels, *c, &profile);
                let got = run_group(dev, &scene, levels, *c, &profile, 1024);
                assert_eq!((got.width, got.height), (want.width, want.height), "{c:?}");
                assert_eq!(got.place, want.place, "{c:?}");
                let (e, at) = compare_scaled(&got.data, &want.data);
                if e > worst.0 {
                    worst = (e, format!("{} {w}x{h} {c:?} sample {at}", cfa.name()));
                }
            }
        }
    }
    // The cases are not vacuous: R5, the denoise and R9 each change the image by far more than the bound.
    let base = Case {
        k: 2,
        recovery: false,
        vignette: false,
        denoise: false,
        lens: false,
    };
    let levels = uneven_levels();
    let scene = RawScene::new(160, 96, Cfa::RGGB, 11, &levels);
    let plain = cpu_reduced(&scene, &levels, base, &profile);
    for c in [
        Case {
            recovery: true,
            ..base
        },
        Case {
            vignette: true,
            ..base
        },
        Case {
            denoise: true,
            ..base
        },
    ] {
        let other = cpu_reduced(&scene, &levels, c, &profile);
        let (d, _) = compare_scaled(&other.data, &plain.data);
        assert!(
            d > 100.0 * GROUP_BOUND,
            "{c:?} changes the image by only {d:e}"
        );
    }
    let lensed = cpu_reduced(&scene, &levels, Case { lens: true, ..base }, &profile);
    assert!(lensed.width < plain.width || lensed.data != plain.data);
    eprintln!(
        "[G-07] reduced group: worst scaled difference {:e} ({})",
        worst.0, worst.1
    );
    assert!(
        worst.0 <= hw(dev, GROUP_BOUND, GROUP_HW_BOUND),
        "{:e} {}",
        worst.0,
        worst.1
    );
}

#[test]
fn the_reduced_group_gives_the_same_bits_for_every_slab_height() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let levels = uneven_levels();
    let scene = RawScene::new(150, 118, Cfa::GBRG, 21, &levels);
    for c in [
        Case {
            k: 2,
            recovery: true,
            vignette: true,
            denoise: true,
            lens: true,
        },
        Case {
            k: 4,
            recovery: true,
            vignette: false,
            denoise: false,
            lens: false,
        },
        Case {
            k: 8,
            recovery: false,
            vignette: true,
            denoise: true,
            lens: false,
        },
    ] {
        let one = run_group(dev, &scene, levels, c, &profile, 1 << 20);
        common::exact::assert_all_finite("reduced", &one.data);
        for slab in [8usize, 16, 24, 40] {
            let many = run_group(dev, &scene, levels, c, &profile, slab);
            common::exact::assert_bit_equal(&format!("{c:?} slab {slab}"), &many.data, &one.data);
        }
    }
}

#[test]
fn the_reduced_group_is_poison_clean_and_cancels() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let levels = uneven_levels();
    let scene = RawScene::new(128, 96, Cfa::RGGB, 5, &levels);
    let c = Case {
        k: 2,
        recovery: true,
        vignette: true,
        denoise: true,
        lens: true,
    };
    let p = reduced_params(&scene, levels, c, &profile);
    let out = assert_poison_neutral("reduced group", |gpu| {
        let (s, p) = (scene.clone(), p.clone());
        gpu.run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |ctx, _| {
                let raw = DeviceRaw::upload(ctx, "r", s.width, s.height, s.cfa, &s.data, None)?;
                let img = ym_gpu::chain::GpuReduced::run_with_slabs(
                    ctx,
                    &raw,
                    &p,
                    &ym_gpu::queue::CancelToken::new(),
                    16,
                )?;
                Ok(img.read_rgb(ctx)?.data)
            },
        )
        .expect("job")
    });
    assert!(out.is_some());

    // A token that is already cancelled: nothing runs.
    let token = ym_gpu::queue::CancelToken::new();
    token.cancel();
    let s = scene.clone();
    let r = on_device_result(dev, move |ctx| {
        let raw = DeviceRaw::upload(ctx, "r", s.width, s.height, s.cfa, &s.data, None)?;
        ym_gpu::chain::GpuReduced::run(ctx, &raw, &p, &token).map(|_| ())
    });
    assert!(
        matches!(r, Err(ym_gpu::device::GpuError::Cancelled)),
        "{r:?}"
    );
}

/// Runs a job on the device and returns its result as it is, errors included.
fn on_device_result<R: Send + 'static>(
    dev: &'static TestGpu,
    f: impl FnOnce(&ym_gpu::device::GpuContext) -> Result<R, ym_gpu::device::GpuError> + Send + 'static,
) -> Result<R, ym_gpu::device::GpuError> {
    dev.gpu.run(
        ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
        move |ctx, _| f(ctx),
    )
}

#[test]
fn the_reduced_group_refuses_what_the_kernels_refuse() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let levels = uneven_levels();
    let scene = RawScene::new(64, 48, Cfa::RGGB, 5, &levels);
    // An odd factor, and a frame smaller than one block.
    for (k, w, h) in [(3usize, 64usize, 48usize), (8, 6, 48)] {
        let mut s = scene.clone();
        s.width = w;
        s.height = h;
        s.data.truncate(w * h);
        let c = Case {
            k: k.max(2),
            recovery: false,
            vignette: false,
            denoise: false,
            lens: false,
        };
        let mut p = reduced_params(&s, levels, c, &profile);
        p.k = k;
        let r = on_device_result(dev, move |ctx| {
            let raw = DeviceRaw::upload(ctx, "r", s.width, s.height, s.cfa, &s.data, None)?;
            ym_gpu::chain::GpuReduced::run(ctx, &raw, &p, &ym_gpu::queue::CancelToken::new())
                .map(|_| ())
        });
        assert!(
            matches!(r, Err(ym_gpu::device::GpuError::InvalidInput(_))),
            "k {k} {w}x{h}: {r:?}"
        );
    }
}

/// A 50 MP frame (8658 x 5772) through the default slabs of 1024 rows: R5 (a clipped blob in the
/// far corner), the vignetting gain, the denoise and R9, against the CPU composition. The frame is a
/// smooth gradient with hashed noise (the natural-scene generator would need 600 MB for its image).
#[test]
#[ignore = "50 MP: 100 MB of words and the CPU composition; run with --ignored (wave 5 runs it on Metal)"]
fn a_50_mp_frame_matches_the_cpu_in_default_slabs() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (8658usize, 5772usize);
    let levels = uneven_levels();
    let mut rng = synth::Rng::new(50);
    let data: Vec<u16> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let base = 400.0 + 9000.0 * (x as f32 / w as f32) * (0.4 + 0.6 * (y as f32 / h as f32));
            let blob = x > w - 900 && y > h - 700;
            let v = if blob {
                20000.0
            } else {
                base + 120.0 * rng.next_f32()
            };
            v.min(65535.0) as u16
        })
        .collect();
    let scene = RawScene {
        width: w,
        height: h,
        cfa: Cfa::RGGB,
        data,
        qstep: None,
    };
    let profile = vignette_profile();
    let mut worst = 0.0f32;
    for (k, recovery, vignette, denoise, lens) in [
        (8usize, true, true, true, false),
        (8, true, false, false, false),
        (8, false, false, false, true),
        (8, true, true, true, true),
        (4, true, true, true, true),
    ] {
        let c = Case {
            k,
            recovery,
            vignette,
            denoise,
            lens,
        };
        let want = cpu_reduced(&scene, &levels, c, &profile);
        let got = run_group(
            dev,
            &scene,
            levels,
            c,
            &profile,
            ym_gpu::chain::REDUCED_SLAB_ROWS,
        );
        assert_eq!((got.width, got.height), (want.width, want.height));
        let (e, at) = compare_scaled(&got.data, &want.data);
        eprintln!(
            "[G-07] 50 MP reduced group {c:?}: worst scaled difference {e:e} (sample {at} of {})",
            got.data.len()
        );
        worst = worst.max(e);
    }
    assert!(
        worst <= hw(dev, GROUP_BOUND_50MP, GROUP_HW_BOUND_50MP),
        "{worst:e}"
    );
}

/// Review of G-07: the fit view of a 50 MP photo on a retina display is the group at `k = 2`, whose
/// binned image is three planes of 4329 x 2886 `f32` (150 MB), more than the 128 MiB design binding.
/// The group must not be given to a device it cannot fit (or must run within the binding): `Ok` with
/// the CPU's size, or the refusal `InvalidInput`/`BufferTooLarge` that sends the group to the CPU
/// without tainting the device; a validation error that leaves the device unusable is the defect.
#[test]
#[ignore = "50 MP: 100 MB of words; run with --ignored"]
fn a_50_mp_frame_at_k_2_fits_the_design_binding_or_is_refused_cleanly() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (8658usize, 5772usize);
    let levels = uneven_levels();
    let mut rng = synth::Rng::new(51);
    let data: Vec<u16> = (0..w * h)
        .map(|i| (400.0 + 9000.0 * ((i % w) as f32 / w as f32) + 120.0 * rng.next_f32()) as u16)
        .collect();
    let scene = RawScene {
        width: w,
        height: h,
        cfa: Cfa::RGGB,
        data,
        qstep: None,
    };
    let profile = vignette_profile();
    let c = Case {
        k: 2,
        recovery: false,
        vignette: false,
        denoise: false,
        lens: false,
    };
    let p = reduced_params(&scene, levels, c, &profile);
    let s = scene.clone();
    let r = on_device_result(dev, move |ctx| {
        let raw = DeviceRaw::upload(ctx, "r", s.width, s.height, s.cfa, &s.data, None)?;
        ym_gpu::chain::GpuReduced::run(ctx, &raw, &p, &ym_gpu::queue::CancelToken::new())
            .map(|img| (img.width, img.height))
    });
    eprintln!("[review] 50 MP k = 2: {r:?}");
    match r {
        Ok(size) => assert_eq!(size, (w / 2, h / 2)),
        Err(e) => assert!(
            matches!(
                e,
                ym_gpu::device::GpuError::InvalidInput(_)
                    | ym_gpu::device::GpuError::BufferTooLarge { .. }
            ),
            "{e:?}"
        ),
    }
    // The device must still work afterwards.
    let again = on_device_result(dev, move |ctx| Ok(ctx.check_alive().is_ok()));
    assert_eq!(
        again.ok(),
        Some(true),
        "the device is unusable after the group"
    );
}

/// Review of G-07: `LP_NUM_THREADS=1` against the default thread count of the software adapter,
/// bit-identical, for the reduced group: the offsets reduction (a tree per workgroup and a fold, D2),
/// the sigma plane, the binning, the binned sigma, the denoise and R9, in slabs of several heights.
#[test]
fn the_reduced_group_is_thread_count_independent() {
    let Some(dev) = f32_device() else { return };
    g06::assert_thread_count_independent("the_reduced_group_is_thread_count_independent", || {
        let profile = vignette_profile();
        let levels = uneven_levels();
        let mut h = g06::FNV_START;
        for (n, (cfa, w, ht, k, slab)) in [
            (Cfa::RGGB, 150usize, 118usize, 2usize, 24usize),
            (Cfa::GBRG, 131, 77, 4, 16),
            (Cfa::BGGR, 160, 96, 8, 1024),
        ]
        .into_iter()
        .enumerate()
        {
            let scene = RawScene::new(w, ht, cfa, 31 + n as u64, &levels);
            let c = Case {
                k,
                recovery: true,
                vignette: true,
                denoise: true,
                lens: true,
            };
            h = g06::fnv(h, &run_group(dev, &scene, levels, c, &profile, slab).data);
        }
        h
    });
}
