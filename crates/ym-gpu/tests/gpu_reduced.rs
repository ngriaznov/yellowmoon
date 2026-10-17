//! The reduced fit-view path on the device: the sigma plane (`shaders/sigma.wgsl`), the k x k
//! binning (`shaders/bin.wgsl`) and the sigma it leaves, against the CPU functions of `ym-core`
//! (copied into `tests/front_common/mod.rs`; `crates/ym-core/tests/gpu_reduced.rs` runs the
//! originals), and the whole group `GpuReduced` against `reduced_bayer`.
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/17-gpu.md` section 12):
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
#[path = "denoise_common/mod.rs"]
mod denoise_common;
#[path = "front_common/mod.rs"]
mod front_common;

use common::exact::assert_poison_neutral;
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use front_common::{
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
/// The starting threshold is 1e-5; the bound is twice the larger measured value.
const SIGMA_BOUND: f32 = 5e-7;
/// Largest `|gpu - cpu|` of the binning (white 1.0; measured 0: the CPU's bits). The starting
/// threshold is 1e-6, kept: it is the margin for a platform whose division and fused multiply-add
/// differ.
const BIN_BOUND: f32 = 1e-6;
/// The same with the vignetting gain (values up to 4 after the gains; measured 0: the gain is formed
/// in double-floats, `front_vig.wgsl`). One ulp is allowed.
const BIN_VIG_BOUND: f32 = 1.2e-7;
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
                front_common::full_levels()
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
    eprintln!("[fit view] sigma plane: worst relative difference {worst:e} ({worst_at})");
    assert!(worst <= SIGMA_BOUND, "{worst:e} {worst_at}");
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
                    worst = worst.max(denoise_common::compare(&got.data, &want.data).max);
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
    eprintln!("[fit view] bin: worst {worst:e}; with the vignetting gain (scaled) {worst_vig:e}");
    assert!(worst <= BIN_BOUND, "{worst:e}");
    assert!(worst_vig <= BIN_VIG_BOUND, "{worst_vig:e}");
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
    eprintln!("[fit view] binned sigma: worst relative difference {worst:e}");
    assert!(worst <= SIGMA_BOUND, "{worst:e}");
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
/// the stages (R1, R4, R5, the binning, the vignetting gain, the sigma, the bilateral denoise and
/// R9 at `1/k`), on frames of a few hundred photosites. The starting threshold for the group is the
/// sum of the kernels' (about 1e-5 with the denoise); the bound is twice the first measured value
/// (9.8e-7, R5, the vignetting gain, the denoise and R9 together).
const GROUP_BOUND: f32 = 2e-6;
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
/// The same at `k = 2`: R9's `f32` displacement is in binned pixels, so it doubles each time
/// `k` halves (9.5e-6 at k = 8, 2.0e-5 at k = 4); first measured 4.47e-5 for the whole group; the
/// bound is twice that.
const GROUP_BOUND_50MP_K2: f32 = 9e-5;

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
    let (mut m, mask) = front_common::cpu_front(scene, levels, G0, None);
    if c.recovery {
        front_common::cpu_highlights(&mut m, &mask);
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
                front_common::full_levels()
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
        "[fit view] reduced group: worst scaled difference {:e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= GROUP_BOUND, "{:e} {}", worst.0, worst.1);
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

/// Review of the fit view in bands: the smallest frames the reduced path accepts (one block and a
/// few more, odd sizes, every CFA phase), made in bands of one row (`k = 8`, slabs of 8 rows) and
/// in one piece: the same bits, finite, and never a panic; a frame that holds no block is refused
/// with an error.
#[test]
fn the_smallest_frames_give_the_same_bits_in_bands_of_one_row_and_in_one_piece() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let levels = uneven_levels();
    let mut compared = 0usize;
    for k in [2usize, 4, 8] {
        for (i, (w, h)) in [
            (k, k),
            (k + 1, k),
            (k, 2 * k + 1),
            (2 * k + 3, k + 2),
            (3 * k, 2 * k),
        ]
        .into_iter()
        .enumerate()
        {
            let cfa = Cfa::ALL[i % 4];
            let scene = RawScene::new(w, h, cfa, 700 + (k * 10 + i) as u64, &levels);
            let c = Case {
                k,
                recovery: i % 2 == 0,
                vignette: i % 3 != 0,
                denoise: true,
                lens: w / k >= 4 && h / k >= 4,
            };
            let run = |slab: usize| {
                let p = reduced_params(&scene, levels, c, &profile);
                let s = scene.clone();
                on_device_result(dev, move |ctx| {
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
                        slab,
                    )?
                    .read_rgb(ctx)
                })
            };
            let (one, many) = (run(1 << 20).expect("one piece"), run(8).expect("bands"));
            common::exact::assert_all_finite("tiny frame", &one.data);
            common::exact::assert_bit_equal(
                &format!("{} {w}x{h} {c:?}", cfa.name()),
                &many.data,
                &one.data,
            );
            assert_eq!((one.width, one.height), (many.width, many.height));
            compared += 1;
        }
    }
    assert_eq!(compared, 15);
    // A frame that holds no block is refused, not a panic.
    let scene = RawScene::new(5, 9, Cfa::RGGB, 1, &levels);
    let c = Case {
        k: 8,
        recovery: false,
        vignette: false,
        denoise: false,
        lens: false,
    };
    let p = reduced_params(&scene, levels, c, &profile);
    let refused = on_device_result(dev, move |ctx| {
        let raw = DeviceRaw::upload(ctx, "r", 5, 9, Cfa::RGGB, &scene.data, None)?;
        ym_gpu::chain::GpuReduced::run(ctx, &raw, &p, &ym_gpu::queue::CancelToken::new())
            .map(|_| ())
    });
    assert!(
        matches!(refused, Err(ym_gpu::device::GpuError::InvalidInput(_))),
        "{refused:?}"
    );
}

/// Review of the fit view in bands (D7 for the stepwise run). The steps of two groups, each taken
/// as a job of its own the way `GpuRoute::reduced_resident` takes them (the run's state travels
/// from job to job), with another group made whole between every two steps on the same poisoned
/// device, give each group the bits of a run that nothing interrupted: nothing a step leaves on the
/// device (the reducer's partials, the bands so far, the sigma) is read or recycled by anything
/// else.
#[test]
fn steps_interleaved_with_other_jobs_give_the_bits_of_an_uninterrupted_run() {
    use std::sync::Arc;
    use ym_gpu::queue::{JobSpec, Priority};
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let levels = uneven_levels();
    let all = |k| Case {
        k,
        recovery: true,
        vignette: true,
        denoise: true,
        lens: true,
    };
    let cases = [
        (RawScene::new(150, 118, Cfa::GBRG, 31, &levels), all(2)),
        (RawScene::new(131, 97, Cfa::RGGB, 32, &levels), all(4)),
        (
            RawScene::new(120, 90, Cfa::BGGR, 33, &levels),
            Case {
                k: 2,
                recovery: false,
                vignette: false,
                denoise: true,
                lens: false,
            },
        ),
    ];
    let alone: Vec<_> = cases
        .iter()
        .map(|(s, c)| run_group(dev, s, levels, *c, &profile, 1 << 20))
        .collect();
    let spec = JobSpec::new(Priority::Viewport);
    let mut runs: Vec<Option<ym_gpu::chain::ReducedRun<Arc<DeviceRaw>>>> = Vec::new();
    for (s, c) in &cases {
        let p = reduced_params(s, levels, *c, &profile);
        let s = s.clone();
        let raw = dev
            .gpu
            .run(spec, move |ctx, _| {
                DeviceRaw::upload(
                    ctx,
                    "r",
                    s.width,
                    s.height,
                    s.cfa,
                    &s.data,
                    s.qstep.as_deref(),
                )
                .map(Arc::new)
            })
            .expect("upload");
        runs.push(Some(
            ym_gpu::chain::GpuReduced::start_with_slabs(raw, p, 16).expect("start"),
        ));
    }
    let mut finished: Vec<Option<ym_gpu::image::RowBands>> =
        (0..runs.len()).map(|_| None).collect();
    let mut steps = 0usize;
    while runs.iter().any(Option::is_some) {
        for i in 0..runs.len() {
            let Some(run) = runs[i].take() else { continue };
            let (run, more) = dev
                .gpu
                .run(spec, move |ctx, token| {
                    let mut run = run;
                    let more = run.step(ctx, token)?;
                    Ok((run, more))
                })
                .expect("a step");
            steps += 1;
            if more {
                runs[i] = Some(run);
            } else {
                finished[i] = Some(run.finish().expect("finish"));
            }
            // Something else runs between the steps: a whole group of the next case, so that the
            // pool recycles and poisons buffers around the steps' live ones.
            let (s, c) = &cases[(i + 1) % cases.len()];
            let other = run_group(dev, s, levels, *c, &profile, 40);
            common::exact::assert_bit_equal(
                "the group made between the steps",
                &other.data,
                &alone[(i + 1) % cases.len()].data,
            );
        }
    }
    assert!(
        steps > 20,
        "{steps} steps: the groups are made in many jobs"
    );
    for (i, bands) in finished.into_iter().enumerate() {
        let bands = bands.expect("finished");
        let got = dev
            .gpu
            .run(spec, move |ctx, _| bands.read_rgb(ctx))
            .expect("read");
        common::exact::assert_bit_equal(
            &format!("case {i} with other jobs between its steps"),
            &got.data,
            &alone[i].data,
        );
    }
    // A run dropped between two steps (a cancelled render) hands every buffer back.
    let s = cases[0].0.clone();
    let p = reduced_params(&s, levels, cases[0].1, &profile);
    let before = on_device_result(dev, |ctx| Ok(ctx.pool().stats().in_use_bytes)).expect("pool");
    let (raw, run) = dev
        .gpu
        .run(spec, move |ctx, token| {
            let raw = Arc::new(DeviceRaw::upload(
                ctx,
                "r",
                s.width,
                s.height,
                s.cfa,
                &s.data,
                s.qstep.as_deref(),
            )?);
            let mut run = ym_gpu::chain::GpuReduced::start_with_slabs(Arc::clone(&raw), p, 16)?;
            for _ in 0..9 {
                run.step(ctx, token)?;
            }
            Ok((raw, run))
        })
        .expect("nine steps");
    assert!(!run.is_done());
    drop(run);
    drop(raw);
    let after = on_device_result(dev, |ctx| Ok(ctx.pool().stats().in_use_bytes)).expect("pool");
    assert_eq!(after, before, "a dropped run holds nothing");
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
#[ignore = "50 MP: 100 MB of words and the CPU composition; run with --ignored (mac-gpu.yml runs it on Metal)"]
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
            "[fit view] 50 MP reduced group {c:?}: worst scaled difference {e:e} (sample {at} of {})",
            got.data.len()
        );
        worst = worst.max(e);
    }
    assert!(worst <= GROUP_BOUND_50MP, "{worst:e}");
}

/// The fit view of a 50 MP photo on a retina display is the group at `k = 2`, whose picture is
/// three planes of 4329 x 2886 `f32` (150 MB), more than the 128 MiB design binding. The device
/// first refused it cleanly (the CPU made the group, and the result depended on the device's
/// binding limit); now it makes the group in bands of full-width rows (`RowBands`), so no buffer of
/// the run, the binning, the denoise and R9 included, is above the binding: every stage of the
/// group on, against the CPU composition, with the largest buffer of the result checked against the
/// constant (the device here has the 128 MiB of lavapipe, so a binding that is too large would fail
/// the run itself).
#[test]
#[ignore = "50 MP: 100 MB of words and the CPU composition; run with --ignored (mac-gpu.yml runs it on Metal)"]
fn a_50_mp_frame_at_k_2_fits_the_design_binding_or_is_refused_cleanly() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (8658usize, 5772usize);
    let levels = uneven_levels();
    let mut rng = synth::Rng::new(51);
    let data: Vec<u16> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let base =
                400.0 + 9000.0 * ((x as f32) / w as f32) * (0.4 + 0.6 * (y as f32 / h as f32));
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
    let c = Case {
        k: 2,
        recovery: true,
        vignette: true,
        denoise: true,
        lens: true,
    };
    let p = reduced_params(&scene, levels, c, &profile);
    let s = scene.clone();
    let started = std::time::Instant::now();
    let made = on_device_result(dev, move |ctx| {
        let raw = DeviceRaw::upload(ctx, "r", s.width, s.height, s.cfa, &s.data, None)?;
        let mut run = ym_gpu::chain::GpuReduced::start(&raw, p)?;
        let mut steps = 0usize;
        // The longest step of each stage, the job length the scheduler sees (K5).
        let mut longest: Vec<(&'static str, f64, f64, usize)> = Vec::new();
        loop {
            let stage = run.next_stage();
            let t = std::time::Instant::now();
            let more = run.step(ctx, &ym_gpu::queue::CancelToken::new())?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            steps += 1;
            match longest.iter_mut().find(|l| l.0 == stage) {
                Some(l) => {
                    l.1 = l.1.max(ms);
                    l.2 += ms;
                    l.3 += 1;
                }
                None => longest.push((stage, ms, ms, 1)),
            }
            if !more {
                break;
            }
        }
        for (stage, max, total, n) in &longest {
            eprintln!(
                "[fit view] reduced step `{stage}`: {n} step(s), longest {max:.1} ms, together {total:.1} ms"
            );
        }
        let picture = run.finish()?;
        let largest = picture.max_band_bytes();
        let bands = picture.bands().len();
        Ok((picture.read_rgb(ctx)?, largest, bands, steps))
    });
    let (got, largest, bands, steps) = made.expect("the group at k = 2 of a 50 MP frame");
    let want = cpu_reduced(&scene, &levels, c, &profile);
    assert_eq!((got.width, got.height), (want.width, want.height));
    let (e, at) = compare_scaled(&got.data, &want.data);
    eprintln!(
        "[fit view] 50 MP k = 2, every stage: {} x {} in {bands} bands (largest buffer {} MiB), {steps} steps, worst scaled difference {e:e} (sample {at}), {:.1} s",
        got.width,
        got.height,
        largest >> 20,
        started.elapsed().as_secs_f64()
    );
    assert!(
        largest <= ym_gpu::tiles::DESIGN_BINDING_BYTES,
        "a band of {largest} bytes is over the design binding"
    );
    assert!(bands > 1, "the picture is more than one buffer");
    assert!(e <= GROUP_BOUND_50MP_K2, "{e:e}");
    // The device is usable afterwards.
    let again = on_device_result(dev, move |ctx| Ok(ctx.check_alive().is_ok()));
    assert_eq!(
        again.ok(),
        Some(true),
        "the device is unusable after the group"
    );
}

/// Review of the fit view: `LP_NUM_THREADS=1` against the default thread count of the software
/// adapter, bit-identical, for the reduced group: the offsets reduction (a tree per workgroup and a
/// fold, D2), the sigma plane, the binning, the binned sigma, the denoise and R9, in slabs of
/// several heights.
#[test]
fn the_reduced_group_is_thread_count_independent() {
    let Some(dev) = f32_device() else { return };
    denoise_common::assert_thread_count_independent(
        "the_reduced_group_is_thread_count_independent",
        || {
            let profile = vignette_profile();
            let levels = uneven_levels();
            let mut h = denoise_common::FNV_START;
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
                h = denoise_common::fnv(h, &run_group(dev, &scene, levels, c, &profile, slab).data);
            }
            h
        },
    );
}

/// Review of the fit view: a differential fuzz of the reduced group against the CPU composition:
/// random frame sizes (odd and even), phases, factors, switches, levels and slab heights. The tile
/// independence of the slabs is the second oracle (a slab of any height gives the bits of one).
#[test]
fn a_fuzz_of_the_reduced_group_matches_the_cpu_and_every_slab_height() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let mut rng = synth::Rng::new(0xF1_07);
    let mut pick = |n: u64| (rng.next_u64() % n) as usize;
    let total: u64 = std::env::var("YM_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let mut worst = (0.0f32, String::new());
    let mut cases = 0usize;
    for case in 0..total {
        let k = [2usize, 4, 8][pick(3)];
        let (w, h) = (k * 8 + pick(150), k * 8 + pick(110));
        let cfa = Cfa::ALL[pick(4)];
        let levels = if pick(2) == 0 {
            uneven_levels()
        } else {
            front_common::full_levels()
        };
        let scene = RawScene::new(w, h, cfa, 900 + case, &levels);
        let c = Case {
            k,
            recovery: pick(3) != 0,
            vignette: pick(2) == 0,
            denoise: pick(2) == 0,
            lens: pick(2) == 0,
        };
        let label = format!("case {case}: {} {w}x{h} {c:?}", cfa.name());
        let want = cpu_reduced(&scene, &levels, c, &profile);
        let slab = [8usize, 16, 40, 1024][pick(4)];
        let got = run_group(dev, &scene, levels, c, &profile, slab);
        assert_eq!(
            (got.width, got.height),
            (want.width, want.height),
            "{label}"
        );
        let (e, at) = compare_scaled(&got.data, &want.data);
        if e > worst.0 {
            worst = (e, format!("{label} slab {slab} sample {at}"));
        }
        assert!(e <= GROUP_BOUND, "{label} slab {slab}: {e:e} at {at}");
        let one = run_group(dev, &scene, levels, c, &profile, 1 << 20);
        common::exact::assert_bit_equal(&format!("{label} slab {slab}"), &got.data, &one.data);
        cases += 1;
    }
    eprintln!(
        "[review] reduced group fuzz: {cases} cases, worst scaled difference {:e} ({})",
        worst.0, worst.1
    );
}
