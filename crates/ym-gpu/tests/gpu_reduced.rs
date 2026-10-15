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
/// Largest `|gpu - cpu|` of the binning (white 1.0; measured 0: the CPU's bits). The plan's number is
/// 1e-6, kept: it is the margin for a platform whose division and fused multiply-add differ.
const BIN_BOUND: f32 = 1e-6;
/// The same with the vignetting gain (values up to 4 after the gains; measured 2.4e-7 of the value,
/// the f32 profile against the CPU's f64 gain).
const BIN_VIG_BOUND: f32 = 5e-7;

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

const G0: [f32; 3] = [2.0, 1.0, 1.6];

fn rel(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = if x == y { 0.0 } else { (x - y).abs() / y.abs().max(1e-6) };
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
            common::exact::assert_bit_equal(&format!("bin k={k} slab {slab}"), &many.data, &one.data);
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
        let sigma_plane: Vec<f32> = (0..hw * hh).map(|_| 0.001 + 0.02 * rng.next_f32()).collect();
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
                    let plane = GpuImage::upload(ctx, "sig", hw, hh, 1, &sp, Placement::whole(hw, hh))?;
                    let mut passes = PassList::new(ctx, "bin sigma");
                    let out = bin::encode_bin_sigma(&mut passes, &plane, &params, ref_noise_factor(k))?;
                    let read = out.encode_read(&mut passes)?;
                    let done = passes.submit()?.finish()?;
                    out.take(&done, read)
                });
                worst = worst.max(rel(&got, &want).0);
            }
        }
    }
    eprintln!("[G-07] binned sigma: worst relative difference {worst:e}");
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
    assert!(ym_gpu::tiles::fits_design_binding(sigma::max_binding_bytes(8658, 5772)));
    assert!(ym_gpu::tiles::fits_design_binding(bin::max_binding_bytes(9504, 1024 + 16)));
    assert_max_binding_bytes("bin slab", Rect::new(0, 0, 4096, 4096), |r| {
        bin::max_binding_bytes(r.w, r.h / 8)
    });
}
