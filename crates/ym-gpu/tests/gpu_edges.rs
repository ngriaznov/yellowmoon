//! Review of G-07: the raw-front kernels on degenerate sizes. A kernel wrapper must refuse what it
//! cannot take with an error and must not panic, and a frame of one photosite or one column must
//! give the CPU's values where the CPU stage takes it.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1`.

mod common;
#[path = "g07/mod.rs"]
mod g07;

use common::{TestGpu, gpus};
use g07::{RawScene, cpu_front, full_levels, on_device, uneven_levels, vignette_profile};
use ym_gpu::cpu::{Cfa, Rect};
use ym_gpu::device::{GpuError, StorageMode};
use ym_gpu::kernels::highlights as hl;
use ym_gpu::kernels::prep::{self, DeviceRaw, PrepParams, VignetteTable};
use ym_gpu::kernels::sigma;

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

const GAINS: [f32; 3] = [2.0, 1.0, 1.6];

fn params(levels: &ym_gpu::cpu::normalize::Levels, vignette: bool) -> PrepParams {
    PrepParams {
        levels: *levels,
        gains: GAINS,
        apply_gains: true,
        vignette: vignette
            .then(|| VignetteTable::from_profile(&vignette_profile()).expect("table")),
    }
}

/// Frames of one photosite, one row, one column and two by two: the prep pass (with and without the
/// vignetting profile and the clip bits) equals the CPU's R1, R3 and R4 on each, for a rectangle of
/// one photosite as well as the whole frame.
#[test]
fn prep_takes_frames_of_one_row_one_column_and_one_photosite() {
    let Some(dev) = f32_device() else { return };
    let levels = full_levels();
    let profile = vignette_profile();
    for cfa in Cfa::ALL {
        for (w, h) in [(1usize, 1usize), (1, 7), (9, 1), (2, 2), (3, 1), (1, 3)] {
            let scene = RawScene::new(w, h, cfa, 3, &levels);
            for vignette in [false, true] {
                let want = cpu_front(&scene, &levels, GAINS, vignette.then_some(&profile));
                let s = scene.clone();
                let (got, bits) = on_device(dev, move |ctx| {
                    let raw = DeviceRaw::upload(ctx, "e", s.width, s.height, s.cfa, &s.data, None)?;
                    prep::prep(
                        ctx,
                        &raw,
                        Rect::whole(s.width, s.height),
                        &params(&levels, vignette),
                        true,
                    )
                });
                let bits = bits.expect("bits");
                assert_eq!(
                    bits,
                    want.1.data,
                    "{} {w}x{h} vignette {vignette}: clip bits",
                    cfa.name()
                );
                for (i, (a, b)) in got.iter().zip(&want.0.data).enumerate() {
                    assert!(
                        a.is_finite(),
                        "{} {w}x{h} {vignette}: sample {i}",
                        cfa.name()
                    );
                    let d = (a - b).abs() / b.abs().max(1.0);
                    assert!(
                        d <= 1.2e-7,
                        "{} {w}x{h} vignette {vignette}: sample {i}: {a} against {b}",
                        cfa.name()
                    );
                }
                // A rectangle of one photosite in the middle of the frame.
                let (x, y) = (w / 2, h / 2);
                let s = scene.clone();
                let one = on_device(dev, move |ctx| {
                    let raw = DeviceRaw::upload(ctx, "e", s.width, s.height, s.cfa, &s.data, None)?;
                    prep::prep(
                        ctx,
                        &raw,
                        Rect::new(x, y, 1, 1),
                        &params(&levels, vignette),
                        true,
                    )
                });
                assert_eq!(one.0.len(), 1);
                assert_eq!(
                    one.0[0].to_bits(),
                    got[y * w + x].to_bits(),
                    "{} {w}x{h}",
                    cfa.name()
                );
            }
        }
    }
}

/// What the wrappers refuse is an error, not a panic.
#[test]
fn the_wrappers_refuse_degenerate_input_with_an_error() {
    let Some(dev) = f32_device() else { return };
    let levels = uneven_levels();
    let scene = RawScene::new(8, 6, Cfa::RGGB, 5, &levels);
    let s = scene.clone();
    let results = on_device(dev, move |ctx| {
        let mut errs: Vec<(&'static str, bool)> = Vec::new();
        let mut note = |what: &'static str, r: Result<(), GpuError>| {
            errs.push((what, matches!(r, Err(GpuError::InvalidInput(_)))));
        };
        // Empty readouts and wrong lengths.
        note(
            "zero width",
            DeviceRaw::upload(ctx, "e", 0, 4, s.cfa, &[], None).map(|_| ()),
        );
        note(
            "zero height",
            DeviceRaw::upload(ctx, "e", 4, 0, s.cfa, &[], None).map(|_| ()),
        );
        note(
            "short data",
            DeviceRaw::upload(ctx, "e", 8, 6, s.cfa, &s.data[..47], None).map(|_| ()),
        );
        note(
            "short steps",
            DeviceRaw::upload(ctx, "e", 8, 6, s.cfa, &s.data, Some(&[0.0; 3])).map(|_| ()),
        );
        note(
            "a step that is not an f16 value",
            DeviceRaw::upload(ctx, "e", 8, 6, s.cfa, &s.data, Some(&[0.1; 48])).map(|_| ()),
        );
        note(
            "a row wider than a slab",
            DeviceRaw::upload_in_slabs(ctx, "e", 8, 6, s.cfa, &s.data, None, 4).map(|_| ()),
        );
        // Rectangles outside the frame, and empty ones.
        let raw = DeviceRaw::upload(ctx, "e", 8, 6, s.cfa, &s.data, None)?;
        let p = params(&levels, true);
        for (name, r) in [
            ("outside", Rect::new(5, 0, 4, 4)),
            ("empty", Rect::new(0, 0, 0, 3)),
            ("below the frame", Rect::new(0, 6, 8, 1)),
        ] {
            note(name, prep::prep(ctx, &raw, r, &p, true).map(|_| ()));
        }
        // Levels without a range.
        let mut bad = p.clone();
        bad.levels.white[0] = bad.levels.black[0];
        note(
            "levels without a range",
            prep::prep(ctx, &raw, Rect::whole(8, 6), &bad, false).map(|_| ()),
        );
        // A sigma plane of a frame below 2 x 2.
        let tiny = DeviceRaw::upload(ctx, "e", 1, 5, s.cfa, &s.data[..5], None)?;
        let model = g07::noise_model();
        let sp = g07::ref_sigma_params(&s, GAINS, &model);
        note(
            "sigma of a column",
            sigma::sigma_plane(ctx, &tiny, &levels, &sp).map(|_| ()),
        );
        Ok(errs)
    });
    for (what, refused) in results {
        assert!(refused, "{what}: expected GpuError::InvalidInput");
    }
}

/// R5 on the smallest frames: 2 x 2 (one superpixel), 3 x 3 (odd), 2 x 5, with one clipped photosite
/// and a core of one photosite, against the CPU.
#[test]
fn r5_takes_the_smallest_frames() {
    let Some(dev) = f32_device() else { return };
    let levels = full_levels();
    for (w, h) in [(2usize, 2usize), (3, 3), (2, 5), (5, 2), (4, 3)] {
        let mut scene = RawScene::new(w, h, Cfa::RGGB, 9, &levels);
        scene.data.iter_mut().for_each(|v| *v = 20000);
        scene.data[0] = 65535;
        let (mut cpu, mask) = cpu_front(&scene, &levels, GAINS, None);
        let off = ym_gpu::cpu::highlights::chroma_offsets(&cpu, &mask).expect("offsets");
        ym_gpu::cpu::highlights::apply_offsets(&mut cpu, &mask, &off).expect("apply");
        let o = [
            off.offset[0] as f32,
            off.offset[1] as f32,
            off.offset[2] as f32,
        ];
        let s = scene.clone();
        let got = on_device(dev, move |ctx| {
            let raw = DeviceRaw::upload(ctx, "e", s.width, s.height, s.cfa, &s.data, None)?;
            let mut passes = ym_gpu::stream::PassList::new(ctx, "e");
            let out = prep::encode_prep(
                &mut passes,
                &raw,
                Rect::whole(s.width, s.height),
                &params(&levels, false),
                true,
            )?;
            let bits = out.mask.expect("mask");
            let offs = hl::Offsets::from_host(&mut passes, o)?;
            let img = hl::encode_highlights(
                &mut passes,
                &out.mosaic,
                &bits,
                s.cfa,
                Rect::whole(s.width, s.height),
                &offs,
            )?;
            let single = hl::encode_highlights(
                &mut passes,
                &out.mosaic,
                &bits,
                s.cfa,
                Rect::new(w - 1, h - 1, 1, 1),
                &offs,
            )?;
            let r1 = img.encode_read(&mut passes)?;
            let r2 = single.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            Ok((img.take(&done, r1)?, single.take(&done, r2)?))
        });
        assert_eq!(got.0.len(), w * h);
        for (i, (a, b)) in got.0.iter().zip(&cpu.data).enumerate() {
            assert!(a.is_finite(), "{w}x{h}: sample {i}");
            assert!(
                (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                "{w}x{h}: sample {i}: {a} against {b}"
            );
        }
        assert_eq!(
            got.1[0].to_bits(),
            got.0[(h - 1) * w + (w - 1)].to_bits(),
            "{w}x{h}: a core of one photosite"
        );
    }
}
