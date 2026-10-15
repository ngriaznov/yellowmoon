//! G-07: the raw front on the device against the CPU stages. `prep` (R1 normalise, R3 vignetting, R4
//! gains and the clip bits, `shaders/prep.wgsl`) here; the R5 highlight reconstruction and the R5b
//! lateral CA on the mosaic are added by the tests further down.
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/gpu-plan.md` X-16):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | R1 and R4 (no profile): 4 CFA phases, even and odd sizes, rectangles at odd origins, unequal levels, words 0 and 65535 | bit-identical to the CPU | 0 differing |
//! | R1, R3, R4 with a profile | at most [`VIGNETTE_BOUND`] | see the printout of `prep_with_the_profile_matches_the_cpu` |
//! | the clip bits, no step plane, f16 steps, table steps | identical to `ClipMask` | 0 differing |
//! | tiles of the prep pass (three sizes) | bit-identical to the whole frame | 0 differing |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//! | the far corner of an 8658 x 5772 frame | within the vignette bound of the CPU | see the printout |
//! | one slab and two slabs of the same words | bit-identical | 0 differing |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); the Metal runners run the same file (`mac-gpu.yml`).

mod common;
#[path = "g06/mod.rs"]
mod g06;
#[path = "g07/mod.rs"]
mod g07;

use common::exact::{assert_bit_equal, assert_poison_neutral, assert_tiles_equal_whole};
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use g07::{
    RawScene, compare_scaled, cpu_front, crop, full_levels, on_device, uneven_levels,
    vignette_profile,
};
use ym_gpu::cpu::normalize::Levels;
use ym_gpu::cpu::{Cfa, Rect};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::prep::{
    self, DeviceRaw, MAX_SLAB_SAMPLES, PrepParams, QuantMode, VignetteTable,
};

/// Largest `|gpu - cpu| / max(1, |cpu|)` of R1 + R3 + R4 with the vignetting profile: an absolute
/// error up to white, a relative one above it (the scenes hold words beyond the white level, which
/// reach values of 14 after the gains, where one ulp is 1e-6). Twice the first measured value (2.65e-7, about 4 ulp: the profile and its reciprocal rounded in f32 against the CPU's f64 gain rounded once).
const VIGNETTE_BOUND: f32 = 5.3e-7;

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

const GAINS: [f32; 3] = [2.0, 1.0, 1.6];

fn params(levels: &Levels, vignette: bool) -> PrepParams {
    PrepParams {
        levels: *levels,
        gains: GAINS,
        apply_gains: true,
        vignette: vignette.then(|| VignetteTable::from_profile(&vignette_profile()).expect("table")),
    }
}

/// Uploads `scene` and runs the prep pass for `rect` on the device, returning the mosaic and the bits.
fn run_prep(
    dev: &TestGpu,
    scene: &RawScene,
    levels: Levels,
    vignette: bool,
    rect: Rect,
    slab: usize,
) -> (Vec<f32>, Vec<u8>) {
    let scene = scene.clone();
    on_device(dev, move |ctx| {
        let raw = DeviceRaw::upload_in_slabs(
            ctx,
            "test raw",
            scene.width,
            scene.height,
            scene.cfa,
            &scene.data,
            scene.qstep.as_deref(),
            slab,
        )?;
        let (m, mask) = prep::prep(ctx, &raw, rect, &params(&levels, vignette), true)?;
        Ok((m, mask.expect("mask")))
    })
}

#[test]
fn prep_without_a_profile_has_the_cpus_bits() {
    let Some(dev) = f32_device() else { return };
    let mut cases = 0;
    for (cfa_i, cfa) in Cfa::ALL.into_iter().enumerate() {
        for (w, h) in [(64, 48), (67, 53), (96, 33), (5, 7)] {
            let levels = if cases % 2 == 0 {
                uneven_levels()
            } else {
                full_levels()
            };
            let scene = RawScene::new(w, h, cfa, 11 + cases as u64, &levels);
            let (cpu, cpu_mask) = cpu_front(&scene, &levels, GAINS, None);
            let rects = [
                Rect::whole(w, h),
                Rect::new(1, 1, w - 2, h - 2),
                Rect::new(w / 2 | 1, h / 3 | 1, w - (w / 2 | 1), (h - h / 3).max(1) - 1),
            ];
            for rect in rects {
                if rect.is_empty() || rect.x1() > w || rect.y1() > h {
                    continue;
                }
                let (gpu, mask) = run_prep(dev, &scene, levels, false, rect, MAX_SLAB_SAMPLES);
                assert_bit_equal(
                    &format!("prep {} {w}x{h} {rect:?}", cfa.name()),
                    &gpu,
                    &crop(&cpu.data, w, rect),
                );
                let want: Vec<u8> = (rect.y..rect.y1())
                    .flat_map(|y| (rect.x..rect.x1()).map(move |x| (x, y)))
                    .map(|(x, y)| cpu_mask.data[y * w + x])
                    .collect();
                assert_eq!(mask, want, "clip bits {} {w}x{h} {rect:?}", cfa.name());
            }
            cases += 1;
        }
        let _ = cfa_i;
    }
    assert!(cases >= 16);
}

#[test]
fn prep_with_the_profile_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let profile = vignette_profile();
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for cfa in Cfa::ALL {
        for (w, h) in [(160, 96), (131, 77)] {
            let levels = uneven_levels();
            let scene = RawScene::new(w, h, cfa, 5, &levels);
            let (cpu, _) = cpu_front(&scene, &levels, GAINS, Some(&profile));
            let rect = Rect::whole(w, h);
            let (gpu, _) = run_prep(dev, &scene, levels, true, rect, MAX_SLAB_SAMPLES);
            let (e, at) = compare_scaled(&gpu, &cpu.data);
            if e > worst {
                worst = e;
                worst_at = format!("{} {w}x{h} at sample {at}", cfa.name());
            }
        }
    }
    eprintln!("[G-07] prep with the profile: worst {worst:e} ({worst_at})");
    assert!(worst <= VIGNETTE_BOUND, "{worst:e} at {worst_at}");
}

#[test]
fn the_clip_bits_follow_the_step_plane_in_both_representations() {
    let Some(dev) = f32_device() else { return };
    let levels = uneven_levels();
    for (many, mode) in [(false, QuantMode::Table), (true, QuantMode::F16)] {
        let scene = RawScene::new(70, 41, Cfa::GRBG, 3, &levels).with_steps(many, 9);
        let (cpu, cpu_mask) = cpu_front(&scene, &levels, GAINS, None);
        let s = scene.clone();
        let got_mode = on_device(dev, move |ctx| {
            Ok(DeviceRaw::upload(
                ctx,
                "q",
                s.width,
                s.height,
                s.cfa,
                &s.data,
                s.qstep.as_deref(),
            )?
            .quant_mode())
        });
        assert_eq!(got_mode, mode, "many = {many}");
        let rect = Rect::whole(70, 41);
        let (gpu, mask) = run_prep(dev, &scene, levels, false, rect, MAX_SLAB_SAMPLES);
        assert_eq!(mask, cpu_mask.data, "clip bits, many = {many}");
        assert_bit_equal("prep with steps", &gpu, &cpu.data);
        assert!(cpu_mask.count() > 20, "the scene must clip");
    }
    // A step of zero clips only at the white level; no step plane clips at 0.995.
    let mut scene = RawScene::new(32, 8, Cfa::RGGB, 1, &levels);
    // Words between 0.995 of the range and the white level: clipped without a step plane, not with a
    // step of zero.
    for (i, x) in [(40usize, 0usize), (41, 1), (100, 0), (130, 1)].iter().map(|&(i, p)| (i, p)) {
        let pos = (i / 32 % 2) * 2 + i % 2;
        let _ = x;
        scene.data[i] = (levels.black[pos] + 0.997 * (levels.white[pos] - levels.black[pos])) as u16;
    }
    let plain = cpu_front(&scene, &levels, GAINS, None).1;
    scene.qstep = Some(vec![0.0; 32 * 8]);
    let zero = cpu_front(&scene, &levels, GAINS, None).1;
    assert_ne!(plain.data, zero.data, "the two rules differ on this scene");
    let (_, mask) = run_prep(dev, &scene, levels, false, Rect::whole(32, 8), MAX_SLAB_SAMPLES);
    assert_eq!(mask, zero.data);
}

#[test]
fn tiles_of_the_prep_pass_equal_the_whole_frame() {
    let Some(dev) = f32_device() else { return };
    let levels = uneven_levels();
    let profile = vignette_profile();
    let (w, h) = (203, 157);
    let scene = RawScene::new(w, h, Cfa::BGGR, 21, &levels).with_steps(false, 4);
    let whole = assert_tiles_equal_whole("prep tiles", w, h, 1, &[32, 64, 100], |rect| {
        run_prep(dev, &scene, levels, true, rect, MAX_SLAB_SAMPLES).0
    });
    // And the whole frame is the CPU's within the bound.
    let (cpu, _) = cpu_front(&scene, &levels, GAINS, Some(&profile));
    assert!(compare_scaled(&whole, &cpu.data).0 <= VIGNETTE_BOUND);
}

#[test]
fn a_poisoned_device_equals_a_plain_one() {
    if f32_device().is_none() {
        return;
    }
    let levels = uneven_levels();
    let scene = RawScene::new(97, 61, Cfa::RGGB, 8, &levels).with_steps(true, 2);
    let out = assert_poison_neutral("prep", |gpu| {
        let s = scene.clone();
        gpu.run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |ctx, _| {
                let raw = DeviceRaw::upload(
                    ctx,
                    "p",
                    s.width,
                    s.height,
                    s.cfa,
                    &s.data,
                    s.qstep.as_deref(),
                )?;
                prep::prep(
                    ctx,
                    &raw,
                    Rect::whole(s.width, s.height),
                    &params(&levels, true),
                    true,
                )
                .map(|(m, _)| m)
            },
        )
        .expect("job")
    });
    assert!(out.is_some());
}

#[test]
fn two_slabs_give_the_bits_of_one() {
    let Some(dev) = f32_device() else { return };
    let levels = uneven_levels();
    let (w, h) = (67, 90);
    let scene = RawScene::new(w, h, Cfa::GBRG, 33, &levels).with_steps(true, 6);
    let rect = Rect::whole(w, h);
    let (one, one_mask) = run_prep(dev, &scene, levels, true, rect, MAX_SLAB_SAMPLES);
    // Slabs of 47 rows (47 * 67 samples) and the 43 that are left; and of 30 rows, which is three
    // slabs and is refused.
    let (two, two_mask) = run_prep(dev, &scene, levels, true, rect, 47 * w);
    assert_bit_equal("two slabs", &two, &one);
    assert_eq!(two_mask, one_mask);
    let s = scene.clone();
    let words = on_device(dev, move |ctx| {
        let raw = DeviceRaw::upload_in_slabs(
            ctx,
            "r",
            s.width,
            s.height,
            s.cfa,
            &s.data,
            None,
            47 * s.width,
        )?;
        assert_eq!(raw.slab_count(), 2);
        raw.read(ctx)
    });
    assert_eq!(words, scene.data);
    let s = scene.clone();
    let refused = on_device(dev, move |ctx| {
        Ok(DeviceRaw::upload_in_slabs(
            ctx,
            "r",
            s.width,
            s.height,
            s.cfa,
            &s.data,
            None,
            30 * s.width,
        )
        .is_err())
    });
    assert!(refused, "three slabs are not supported");
}

/// A 70 MP header (9792 x 7168 photosites): the words need two slabs of at most 128 MiB, the corner
/// tile still has the CPU's values. Takes a few seconds on lavapipe.
#[test]
#[ignore = "70 MP: 140 MB of words; run with --ignored (wave 5 runs it on Metal)"]
fn a_70_mp_frame_runs_through_two_slabs() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (9792usize, 7168usize);
    let levels = uneven_levels();
    let mut rng = ym_gpu::cpu::synth::Rng::new(77);
    let data: Vec<u16> = (0..w * h)
        .map(|_| (150.0 + rng.next_f32() * 16000.0) as u16)
        .collect();
    let scene = RawScene {
        width: w,
        height: h,
        cfa: Cfa::RGGB,
        data,
        qstep: None,
    };
    let profile = vignette_profile();
    // The corner tile in the second slab, and one that straddles the slab boundary.
    let split = (MAX_SLAB_SAMPLES / w) as usize;
    let tiles = [
        Rect::new(w - 512, h - 512, 512, 512),
        Rect::new(4000, split - 100, 600, 200),
    ];
    let s = scene.clone();
    let got = on_device(dev, move |ctx| {
        let raw = DeviceRaw::upload(ctx, "70mp", s.width, s.height, s.cfa, &s.data, None)?;
        assert_eq!(raw.slab_count(), 2);
        let mut out = Vec::new();
        for r in tiles {
            out.push(prep::prep(ctx, &raw, r, &params(&uneven_levels(), true), false)?.0);
        }
        Ok(out)
    });
    for (r, g) in tiles.iter().zip(&got) {
        // The CPU reference of the tile alone (R1 + R3 + R4 are pointwise).
        let tile_raw: Vec<u16> = (r.y..r.y1())
            .flat_map(|y| scene.data[y * w + r.x..y * w + r.x1()].to_vec())
            .collect();
        let place = ym_gpu::cpu::Placement::whole(w, h).offset(r.x, r.y);
        let (mut m, _) = ym_gpu::cpu::normalize::normalize_tile(
            &tile_raw,
            r.w,
            r.h,
            place,
            scene.cfa.shifted(r.x, r.y),
            &levels,
            None,
        )
        .expect("tile");
        ym_gpu::cpu::vignette::apply_vignette(&mut m, &profile);
        ym_gpu::cpu::gains::apply_gains(&mut m, GAINS);
        let e = compare_scaled(g, &m.data).0;
        eprintln!("[G-07] 70 MP tile {r:?}: worst {e:e}");
        assert!(e <= VIGNETTE_BOUND, "{r:?}: {e:e}");
    }
}

#[test]
fn the_far_corner_of_a_50_mp_frame_has_the_cpus_gain() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (8658usize, 5772usize);
    let levels = uneven_levels();
    let mut rng = ym_gpu::cpu::synth::Rng::new(5);
    let data: Vec<u16> = (0..w * h)
        .map(|_| (120.0 + rng.next_f32() * 15000.0) as u16)
        .collect();
    let profile = vignette_profile();
    let tiles = [
        Rect::new(w - 64, h - 64, 64, 64),
        Rect::new(0, 0, 64, 64),
        Rect::new(w / 2 + 1, h / 2 + 1, 64, 64),
    ];
    let d = data.clone();
    let got = on_device(dev, move |ctx| {
        let raw = DeviceRaw::upload(ctx, "50mp", w, h, Cfa::RGGB, &d, None)?;
        let mut out = Vec::new();
        for r in tiles {
            out.push(prep::prep(ctx, &raw, r, &params(&uneven_levels(), true), false)?.0);
        }
        Ok(out)
    });
    let mut worst = 0.0f32;
    for (r, g) in tiles.iter().zip(&got) {
        let tile_raw: Vec<u16> = (r.y..r.y1())
            .flat_map(|y| data[y * w + r.x..y * w + r.x1()].to_vec())
            .collect();
        let place = ym_gpu::cpu::Placement::whole(w, h).offset(r.x, r.y);
        let (mut m, _) = ym_gpu::cpu::normalize::normalize_tile(
            &tile_raw,
            r.w,
            r.h,
            place,
            Cfa::RGGB.shifted(r.x, r.y),
            &levels,
            None,
        )
        .expect("tile");
        ym_gpu::cpu::vignette::apply_vignette(&mut m, &profile);
        ym_gpu::cpu::gains::apply_gains(&mut m, GAINS);
        worst = worst.max(compare_scaled(g, &m.data).0);
    }
    eprintln!("[G-07] far corner of 8658 x 5772: worst {worst:e}");
    assert!(worst <= VIGNETTE_BOUND, "{worst:e}");
}

#[test]
fn the_binding_of_a_prep_pass_fits_the_design_binding() {
    assert_max_binding_bytes(
        "prep",
        Rect::new(0, 0, 1581, 1581),
        prep::max_binding_bytes,
    );
}

// ---------------------------------------------------------------------------------------------
// R5 highlight reconstruction
// ---------------------------------------------------------------------------------------------

use g07::{cpu_highlights, highlight_scenes};
use ym_gpu::cpu::highlights;
use ym_gpu::cpu::tiling::Halo;
use ym_gpu::kernels::highlights as hl;

/// Largest `|gpu - cpu|` of the apply pass (values up to about 3 after the gains) with the CPU's
/// offsets, on the ten scenes. Measured 0 (the cube root is correctly rounded and the CPU's is the
/// glibc one, which agrees on these values): the plan's number is kept as the margin for a libm whose
/// `cbrtf` is one ulp off in one case of ten.
const HIGHLIGHT_BOUND: f32 = 1e-6;
/// Largest relative difference of the device offsets from the CPU's. Measured 7.0e-8; the plan's
/// number is 1e-6, kept for the same reason.
const OFFSET_BOUND: f32 = 1e-6;

/// prep (no vignette) then the apply pass for the `core` of the `rect` of the frame, on the device,
/// with the CPU's offsets.
fn run_highlights(
    dev: &TestGpu,
    scene: &RawScene,
    levels: Levels,
    rect: Rect,
    core: Rect,
    offsets: [f32; 3],
) -> Vec<f32> {
    let scene = scene.clone();
    on_device(dev, move |ctx| {
        let raw = DeviceRaw::upload(
            ctx,
            "hl raw",
            scene.width,
            scene.height,
            scene.cfa,
            &scene.data,
            scene.qstep.as_deref(),
        )?;
        let mut passes = ym_gpu::stream::PassList::new(ctx, "hl test");
        let out = prep::encode_prep(&mut passes, &raw, rect, &params(&levels, false), true)?;
        let bits = out.mask.expect("mask");
        let off = hl::Offsets::from_host(&mut passes, offsets)?;
        let cfa = scene.cfa.shifted(rect.x, rect.y);
        let img = hl::encode_highlights(&mut passes, &out.mosaic, &bits, cfa, core, &off)?;
        let read = img.encode_read(&mut passes)?;
        let done = passes.submit()?.finish()?;
        img.take(&done, read)
    })
}

fn offsets_f32(o: &highlights::ChromaOffsets) -> [f32; 3] {
    [o.offset[0] as f32, o.offset[1] as f32, o.offset[2] as f32]
}

#[test]
fn r5_matches_the_cpu_on_the_ten_synthetic_highlight_scenes() {
    let Some(dev) = f32_device() else { return };
    let levels = uneven_levels();
    let size = 160;
    let mut worst = 0.0f32;
    let mut worst_scene = String::new();
    let mut changed = 0;
    for (k, s) in highlight_scenes().iter().enumerate() {
        let cfa = Cfa::ALL[k % 4];
        let scene = s.raw(size, cfa, &levels, GAINS);
        let (mut cpu, mask) = cpu_front(&scene, &levels, GAINS, None);
        let before = cpu.clone();
        let off = highlights::chroma_offsets(&cpu, &mask).expect("offsets");
        highlights::apply_offsets(&mut cpu, &mask, &off).expect("apply");
        changed += usize::from(
            cpu.data
                .iter()
                .zip(&before.data)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
        );
        let gpu = run_highlights(
            dev,
            &scene,
            levels,
            Rect::whole(size, size),
            Rect::whole(size, size),
            offsets_f32(&off),
        );
        let e = compare_scaled(&gpu, &cpu.data);
        if e.0 > worst {
            worst = e.0;
            worst_scene = format!("{} ({}, {})", s.name, cfa.name(), if s.disc { "disc" } else { "glow" });
        }
    }
    // A magenta source clips red and blue far above the green estimate: nothing to raise there.
    assert!(changed >= 8, "only {changed} of ten scenes had something to reconstruct");
    eprintln!("[G-07] R5 apply on ten scenes ({changed} changed): worst {worst:e} ({worst_scene})");
    assert!(worst <= HIGHLIGHT_BOUND, "{worst:e} {worst_scene}");
}

#[test]
fn r5_tiles_equal_the_whole_frame() {
    let Some(dev) = f32_device() else { return };
    let levels = uneven_levels();
    let size = 150;
    let scene = highlight_scenes()[4].raw(size, Cfa::GBRG, &levels, GAINS);
    let (m, mask) = cpu_front(&scene, &levels, GAINS, None);
    let off = highlights::chroma_offsets(&m, &mask).expect("offsets");
    let o = offsets_f32(&off);
    let whole = assert_tiles_equal_whole("r5 tiles", size, size, 1, &[32, 64, 90], |core| {
        // The stage's own halo (4 px), even-aligned, as the chain cuts it.
        let frame = Rect::whole(size, size);
        let rect = core.grow_aligned(Halo::uniform(4), frame, 2);
        run_highlights(dev, &scene, levels, rect, core.relative_to(&rect), o)
    });
    assert!(
        whole
            .iter()
            .zip(&m.data)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the frame must have been reconstructed"
    );
}

#[test]
fn an_odd_sized_frame_leaves_its_last_row_and_column_alone() {
    let Some(dev) = f32_device() else { return };
    let levels = full_levels();
    let (w, h) = (61, 47);
    let scene = RawScene::new(w, h, Cfa::RGGB, 9, &levels);
    let (mut cpu, mask) = cpu_front(&scene, &levels, GAINS, None);
    let off = highlights::chroma_offsets(&cpu, &mask).expect("offsets");
    let o = offsets_f32(&off);
    highlights::apply_offsets(&mut cpu, &mask, &off).expect("apply");
    let gpu = run_highlights(dev, &scene, levels, Rect::whole(w, h), Rect::whole(w, h), o);
    let e = compare_scaled(&gpu, &cpu.data);
    eprintln!("[G-07] R5 on a 61 x 47 frame: worst {:e}", e.0);
    assert!(e.0 <= HIGHLIGHT_BOUND, "{:e}", e.0);
}

#[test]
fn the_device_offsets_match_the_cpus_within_1e_6() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut cases = 0;
    for (k, s) in highlight_scenes().iter().enumerate() {
        let levels = if k % 2 == 0 { uneven_levels() } else { full_levels() };
        let scene = s.raw(160, Cfa::ALL[k % 4], &levels, GAINS);
        let (m, mask) = cpu_front(&scene, &levels, GAINS, None);
        let want = highlights::chroma_offsets(&m, &mask).expect("offsets");
        for slab_rows in [160usize, 48, 22] {
            let (m2, mask2) = (m.clone(), mask.clone());
            let (got, counts) = on_device(dev, move |ctx| {
                hl::reduce_offsets(ctx, &m2, &mask2, slab_rows)
            });
            for c in 0..3 {
                assert_eq!(u64::from(counts[c]), want.count[c], "{} count {c}", s.name);
                if want.count[c] > 0 {
                    let rel = (f64::from(got[c]) - want.offset[c]).abs() / want.offset[c].abs().max(1e-3);
                    worst = worst.max(rel as f32);
                }
            }
            cases += 1;
        }
    }
    // A natural scene blown out in places (many reference superpixels).
    for (gain, seed) in [(1.9f32, 3u64), (1.3, 4)] {
        let levels = uneven_levels();
        let mut scene = RawScene::new(211, 190, Cfa::RGGB, seed, &levels);
        for v in &mut scene.data {
            *v = ((f32::from(*v) - 100.0) * gain / 1.35 + 100.0).clamp(0.0, 65535.0) as u16;
        }
        let (m, mask) = cpu_front(&scene, &levels, GAINS, None);
        let want = highlights::chroma_offsets(&m, &mask).expect("offsets");
        let (m2, mask2) = (m.clone(), mask.clone());
        let (got, counts) = on_device(dev, move |ctx| hl::reduce_offsets(ctx, &m2, &mask2, 64));
        for c in 0..3 {
            assert_eq!(u64::from(counts[c]), want.count[c], "natural count {c}");
            if want.count[c] > 0 {
                let rel = (f64::from(got[c]) - want.offset[c]).abs() / want.offset[c].abs().max(1e-3);
                worst = worst.max(rel as f32);
            }
        }
        cases += 1;
    }
    eprintln!("[G-07] chroma offsets, {cases} runs: worst relative difference {worst:e}");
    assert!(worst <= OFFSET_BOUND, "{worst:e}");
}

#[test]
fn the_highlight_passes_are_poison_clean() {
    if f32_device().is_none() {
        return;
    }
    let levels = uneven_levels();
    let size = 130;
    let scene = highlight_scenes()[2].raw(size, Cfa::RGGB, &levels, GAINS);
    let (m, mask) = cpu_front(&scene, &levels, GAINS, None);
    let off = highlights::chroma_offsets(&m, &mask).expect("offsets");
    let o = offsets_f32(&off);
    let out = assert_poison_neutral("highlights", |gpu| {
        let s = scene.clone();
        let (m2, mask2) = (m.clone(), mask.clone());
        gpu.run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |ctx, _| {
                let mut v = hl::highlights(ctx, &m2, &mask2, o, Rect::whole(size, size))?;
                let (offs, _) = hl::reduce_offsets(ctx, &m2, &mask2, 40)?;
                v.extend_from_slice(&offs);
                let _ = &s;
                Ok(v)
            },
        )
        .expect("job")
    });
    assert!(out.is_some());
}

#[test]
fn the_binding_of_the_highlight_passes_fits_the_design_binding() {
    assert_max_binding_bytes(
        "highlights",
        Rect::new(0, 0, 1581, 1581),
        hl::max_binding_bytes,
    );
}

#[test]
fn the_helper_runs_cpu_highlights() {
    // Keeps the helper used and pins that the CPU reconstruction has something to do on a scene.
    let levels = uneven_levels();
    let scene = highlight_scenes()[0].raw(64, Cfa::RGGB, &levels, GAINS);
    let (mut m, mask) = cpu_front(&scene, &levels, GAINS, None);
    let off = cpu_highlights(&mut m, &mask);
    assert!(off.count.iter().any(|&c| c > 0));
}

// ---------------------------------------------------------------------------------------------
// R5b lateral CA on the mosaic
// ---------------------------------------------------------------------------------------------

use ym_gpu::cpu::lens::{self, LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::synth;
use ym_gpu::kernels::mosaic_ca;

/// Largest `|gpu - cpu|` of R5b on natural content (white 1.0): the plan's number is 1e-5, the bound is
/// twice the first measured value (3.0e-7, about one ulp of the values), see the test.
const MOSAIC_CA_BOUND: f32 = 6e-7;

/// Lens tags with a lateral CA of up to a few pixels at the corner of a 200-pixel frame.
fn ca_tags() -> LensTags {
    LensTags {
        vignetting: vec![],
        distortion: vec![0, 30, 100, 220, 400, 650],
        ca: vec![0, 5000, 12000, 20000, 0, -5000, -12000, -20000],
    }
}

fn mosaic_ca_curves(w: usize, h: usize, tags: &LensTags) -> ym_gpu::cpu::demosaic::CaCurves {
    LensGeometry::new(w, h, tags, LensSwitches::default())
        .expect("lens")
        .with_auto_crop()
        .with_ca_on_mosaic()
        .mosaic_ca_curves()
        .expect("the lens has a CA")
}

fn natural_mosaic(w: usize, h: usize, cfa: Cfa, seed: u64) -> ym_gpu::cpu::Mosaic {
    let img = synth::natural(w, h, seed);
    synth::mosaic_from_rgb(&img, cfa)
}

#[test]
fn r5b_matches_the_cpu_for_every_phase_and_odd_sizes() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for cfa in Cfa::ALL {
        for (w, h) in [(200, 140), (131, 97)] {
            let curves = mosaic_ca_curves(w, h, &ca_tags());
            let m = natural_mosaic(w, h, cfa, 3);
            let mut cpu = m.clone();
            lens::correct_mosaic_ca(&mut cpu, &curves).expect("cpu");
            assert!(
                compare_scaled(&cpu.data, &m.data).0 > 1e-3,
                "the correction must move values on this scene"
            );
            let (m2, c2) = (m.clone(), curves.clone());
            let gpu = on_device(dev, move |ctx| {
                mosaic_ca::mosaic_ca(ctx, &m2, Rect::whole(w, h), &c2)
            });
            let (e, at) = compare_scaled(&gpu, &cpu.data);
            if e > worst {
                worst = e;
                worst_at = format!("{} {w}x{h} sample {at}", cfa.name());
            }
        }
    }
    eprintln!("[G-07] R5b: worst {worst:e} ({worst_at})");
    assert!(worst <= MOSAIC_CA_BOUND, "{worst:e} {worst_at}");
}

#[test]
fn r5b_tiles_with_the_stage_halo_equal_the_whole_frame() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (203, 157);
    let curves = mosaic_ca_curves(w, h, &ca_tags());
    let halo = lens::mosaic_ca_halo_px(&curves);
    let m = natural_mosaic(w, h, Cfa::GBRG, 5);
    let whole = assert_tiles_equal_whole("r5b tiles", w, h, 1, &[40, 64, 100], |core| {
        let frame = Rect::whole(w, h);
        let rect = core.grow_aligned(Halo::uniform(halo), frame, 2);
        let sub = m.extract(rect);
        let c2 = curves.clone();
        let rel = core.relative_to(&rect);
        on_device(dev, move |ctx| mosaic_ca::mosaic_ca(ctx, &sub, rel, &c2))
    });
    let mut cpu = m.clone();
    lens::correct_mosaic_ca(&mut cpu, &curves).expect("cpu");
    assert!(compare_scaled(&whole, &cpu.data).0 <= MOSAIC_CA_BOUND);
}

#[test]
fn r5b_at_the_far_corner_of_a_50_mp_frame() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (8658usize, 5772usize);
    let mut tags = ca_tags();
    // Scale the table to a frame 40 times larger: the same shifts in pixels at the corner need
    // 40 times fewer knots units per pixel; keep a few pixels of shift.
    tags.ca = vec![0, 400, 900, 1500, 0, -400, -900, -1500];
    let curves = mosaic_ca_curves(w, h, &tags);
    let halo = lens::mosaic_ca_halo_px(&curves);
    // A 96 x 96 corner of the frame as a mosaic with its placement, with the halo around a core.
    let core = Rect::new(w - 96, h - 96, 96, 96);
    let frame = Rect::whole(w, h);
    let rect = core.grow_aligned(Halo::uniform(halo), frame, 2);
    let img = synth::natural(rect.w, rect.h, 8);
    let mut m = synth::mosaic_from_rgb(&img, Cfa::RGGB.shifted(rect.x, rect.y));
    m.place = ym_gpu::cpu::Placement::whole(w, h).offset(rect.x, rect.y);
    let mut cpu = m.clone();
    lens::correct_mosaic_ca(&mut cpu, &curves).expect("cpu");
    let rel = core.relative_to(&rect);
    let (m2, c2) = (m.clone(), curves.clone());
    let gpu = on_device(dev, move |ctx| mosaic_ca::mosaic_ca(ctx, &m2, rel, &c2));
    let want = crop(&cpu.data, rect.w, rel);
    let e = compare_scaled(&gpu, &want);
    eprintln!("[G-07] R5b at the far corner of 8658 x 5772: worst {:e}", e.0);
    assert!(e.0 <= MOSAIC_CA_BOUND, "{:e}", e.0);
    assert!(
        compare_scaled(&want, &crop(&m.data, rect.w, rel)).0 > 1e-4,
        "the correction must move values there"
    );
}

#[test]
fn r5b_is_poison_clean() {
    if f32_device().is_none() {
        return;
    }
    let (w, h) = (120, 90);
    let curves = mosaic_ca_curves(w, h, &ca_tags());
    let m = natural_mosaic(w, h, Cfa::RGGB, 2);
    let out = assert_poison_neutral("r5b", |gpu| {
        let (m2, c2) = (m.clone(), curves.clone());
        gpu.run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |ctx, _| mosaic_ca::mosaic_ca(ctx, &m2, Rect::whole(w, h), &c2),
        )
        .expect("job")
    });
    assert!(out.is_some());
}

#[test]
fn the_binding_of_r5b_fits_the_design_binding() {
    assert_max_binding_bytes(
        "mosaic ca",
        Rect::new(0, 0, 1581, 1581),
        mosaic_ca::max_binding_bytes,
    );
}
