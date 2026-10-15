//! G-07: R9, the lens resample (`shaders/lens.wgsl`, `kernels::lens`), against
//! `cpu::resample::resample_lens`: the distortion and CA tables of `LensGeometry`, the output
//! positions kept small (the integer pixel index plus the fraction of the displacement, D4).
//!
//! What is asserted, and what was measured on lavapipe (Mesa 25.2, LLVM 20.1; first measured value,
//! the bound is twice that, `docs/gpu-plan.md` X-16):
//!
//! | Case | Asserted | Measured |
//! |---|---|---|
//! | distortion only, distortion + CA in R9, CA on the mosaic (the render's geometry), strong and mild tables, odd sizes, whole output | at most [`BOUND`] against the CPU (white 1.0) | see the printout of `r9_matches_the_cpu` |
//! | the output in tiles of three sizes, each from its own source rectangle (`source_rect`) | bit-identical to the whole output | 0 differing |
//! | the far corner of an 8658 x 5772 frame, its centre and its origin | at most [`CORNER_BOUND`] against the CPU | 3.0e-5 (hard-edged content), 1.4e-5 (noise) |
//! | a poisoned device and a plain one | bit-identical, finite | 0 differing |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`,
//! `YM_REQUIRE_GPU=1`); the Metal runners run the same file (`mac-gpu.yml`).

mod common;
#[path = "g06/mod.rs"]
mod g06;
#[path = "g07/mod.rs"]
mod g07;

use common::exact::{assert_poison_neutral, assert_tiles_equal_whole};
use common::limits::assert_max_binding_bytes;
use common::{TestGpu, gpus};
use g06::compare;
use g07::on_device;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::resample::resample_lens;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Placement, Rect, RgbImage};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::lens as klens;

/// Largest `|gpu - cpu|` of the resample (white 1.0) on frames of a few hundred pixels. The plan's
/// number is 1e-5; the bound is twice the first measured value (2.9e-6).
const BOUND: f32 = 6e-6;
/// The same on a hardware adapter: the plan's number for R9, to be lowered from the Metal numbers.
const HW_BOUND: f32 = 1e-5;
/// The same at the far corner of a 50 MP frame on synthetic content with hard edges (gradients of
/// about one per pixel), where the displacement is about 160 pixels and an `f32` position holds it to
/// 1e-5 to 3e-5 pixel (the table value, the radius and the product each round at about 1e-5 pixel).
/// Twice the first measured value (3.0e-5). On a photograph (gradients of 0.05) this is a value error
/// of 2e-6.
const CORNER_BOUND: f32 = 6e-5;
/// The same on a hardware adapter: twice the software bound, because Metal's fast square root of the
/// doubled radius (a number of up to 25 000) moves the position by a few 1e-4 pixel.
const HW_CORNER_BOUND: f32 = 1.2e-4;

fn bound(dev: &TestGpu) -> f32 {
    if dev.gpu.info().is_software {
        BOUND
    } else {
        HW_BOUND
    }
}

fn corner_bound(dev: &TestGpu) -> f32 {
    if dev.gpu.info().is_software {
        CORNER_BOUND
    } else {
        HW_CORNER_BOUND
    }
}

fn f32_device() -> Option<&'static TestGpu> {
    gpus().iter().find(|d| d.mode == StorageMode::F32)
}

/// A mild and a strong lens (the strong one needs a real auto crop).
fn tags(strong: bool) -> LensTags {
    if strong {
        LensTags {
            vignetting: vec![],
            distortion: vec![0, 300, 900, 1800, 3000, 4500, 6200],
            ca: vec![0, 9000, 18000, 30000, 0, -9000, -18000, -30000],
        }
    } else {
        LensTags {
            vignetting: vec![],
            distortion: vec![0, 30, 100, 220, 400, 650],
            ca: vec![0, 5000, 12000, 20000, 0, -5000, -12000, -20000],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Distortion,
    CaInR9,
    CaOnMosaic,
}

fn geometry(w: usize, h: usize, strong: bool, kind: Kind) -> LensGeometry {
    let mut t = tags(strong);
    if kind == Kind::Distortion {
        t.ca = vec![];
    }
    let g = LensGeometry::new(w, h, &t, LensSwitches::default())
        .expect("lens")
        .with_auto_crop();
    if kind == Kind::CaOnMosaic {
        g.with_ca_on_mosaic()
    } else {
        g
    }
}

/// The source rectangle of `out` cut out of `img` (whole-frame placement kept).
fn cut(img: &RgbImage, geo: &LensGeometry, out: Rect) -> RgbImage {
    img.extract(geo.source_rect(out))
}

#[test]
fn r9_matches_the_cpu() {
    let Some(dev) = f32_device() else { return };
    let mut worst = 0.0f32;
    let mut worst_case = String::new();
    for (w, h) in [(200, 140), (131, 97)] {
        for strong in [false, true] {
            for kind in [Kind::Distortion, Kind::CaInR9, Kind::CaOnMosaic] {
                let geo = geometry(w, h, strong, kind);
                let (cw, ch) = geo.output_size();
                if strong {
                    assert!(cw < w, "a strong lens must crop ({cw} x {ch} of {w} x {h})");
                }
                let img = synth::natural(w, h, 6);
                let out = Rect::whole(cw, ch);
                let want = resample_lens(&img, &geo, out).expect("cpu");
                let (i2, g2) = (img.clone(), geo.clone());
                let got = on_device(dev, move |ctx| klens::lens(ctx, &i2, &g2, out));
                assert_eq!((got.width, got.height), (cw, ch));
                assert_eq!(got.place, want.place);
                let e = compare(&got.data, &want.data);
                if e.max > worst {
                    worst = e.max;
                    worst_case = format!("{w}x{h} strong {strong} {kind:?} at sample {}", e.at);
                }
            }
        }
    }
    eprintln!("[G-07] R9: worst {worst:e} ({worst_case})");
    assert!(worst <= bound(dev), "{worst:e} {worst_case}");
}

#[test]
fn r9_tiles_equal_the_whole_output() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (230, 170);
    for kind in [Kind::CaInR9, Kind::Distortion] {
        let geo = geometry(w, h, true, kind);
        let (cw, ch) = geo.output_size();
        let img = synth::natural(w, h, 7);
        let whole = assert_tiles_equal_whole("r9 tiles", cw, ch, 3, &[32, 64, 100], |out| {
            let sub = cut(&img, &geo, out);
            let table = klens::LensTable::from_geometry(&geo).expect("table");
            on_device(dev, move |ctx| {
                let input = ym_gpu::image::GpuImage::upload_rgb(ctx, "src", &sub)?;
                let mut passes = ym_gpu::stream::PassList::new(ctx, "lens tile");
                let o = klens::encode_lens(
                    &mut passes,
                    &input,
                    Rect::whole(sub.width, sub.height),
                    &table,
                    out,
                )?;
                let read = o.encode_read(&mut passes)?;
                let done = passes.submit()?.finish()?;
                o.take(&done, read)
            })
        });
        let want = resample_lens(&img, &geo, Rect::whole(cw, ch)).expect("cpu");
        assert!(compare(&whole, &want.data).max <= bound(dev), "{kind:?}");
    }
}

#[test]
fn r9_at_the_far_corner_and_the_centre_of_a_50_mp_frame() {
    let Some(dev) = f32_device() else { return };
    let (w, h) = (8658usize, 5772usize);
    let t = LensTags {
        vignetting: vec![],
        distortion: vec![0, 20, 60, 120, 200, 300, 420, 560, 740],
        ca: vec![0, 700, 1500, 2500, 0, -700, -1500, -2500],
    };
    let geo = LensGeometry::new(w, h, &t, LensSwitches::default())
        .expect("lens")
        .with_auto_crop();
    let (cw, ch) = geo.output_size();
    let mut worst = 0.0f32;
    let mut worst_stress = 0.0f32;
    for out in [
        Rect::new(cw - 64, ch - 64, 64, 64),
        Rect::new(cw / 2 + 1, ch / 2 + 1, 64, 64),
        Rect::new(0, 0, 48, 48),
    ] {
        let src_rect = geo.source_rect(out);
        // A natural-looking source patch at its true position in the frame, and a hostile one:
        // uncorrelated noise of 0.05 between neighbours, where a position error of 1e-4 pixel is
        // a value error of 1e-5.
        let mut sub = synth::natural(src_rect.w, src_rect.h, 3);
        sub.place = Placement::whole(w, h).offset(src_rect.x, src_rect.y);
        let mut stress = sub.clone();
        let mut rng = synth::Rng::new(3);
        let n = stress.plane_len();
        for c in 0..3 {
            for y in 0..src_rect.h {
                for x in 0..src_rect.w {
                    stress.data[c * n + y * src_rect.w + x] = 0.3
                        + 0.2 * (((x + y * 3 + c * 5) % 17) as f32 / 17.0)
                        + 0.05 * rng.next_f32();
                }
            }
        }
        for (patch, worst) in [(sub, &mut worst), (stress, &mut worst_stress)] {
            let want = resample_lens(&patch, &geo, out).expect("cpu");
            let (s2, g2) = (patch.clone(), geo.clone());
            let got = on_device(dev, move |ctx| klens::lens(ctx, &s2, &g2, out));
            *worst = worst.max(compare(&got.data, &want.data).max);
        }
    }
    eprintln!(
        "[G-07] R9 at 8658 x 5772: natural worst {worst:e}, hostile noise patch {worst_stress:e}"
    );
    assert!(worst <= corner_bound(dev), "{worst:e}");
    assert!(worst_stress <= corner_bound(dev), "{worst_stress:e}");
}

#[test]
fn r9_is_poison_clean() {
    if f32_device().is_none() {
        return;
    }
    let (w, h) = (120, 90);
    let geo = geometry(w, h, false, Kind::CaInR9);
    let (cw, ch) = geo.output_size();
    let img = synth::natural(w, h, 1);
    let out = assert_poison_neutral("r9", |gpu| {
        let (i2, g2) = (img.clone(), geo.clone());
        gpu.run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            move |ctx, _| klens::lens(ctx, &i2, &g2, Rect::whole(cw, ch)).map(|i| i.data),
        )
        .expect("job")
    });
    assert!(out.is_some());
}

#[test]
fn the_binding_of_r9_fits_the_design_binding() {
    assert_max_binding_bytes(
        "lens",
        Rect::new(0, 0, 1581, 1581),
        klens::max_binding_bytes,
    );
}

#[test]
fn a_source_of_another_frame_is_refused() {
    let Some(dev) = f32_device() else { return };
    let geo = geometry(100, 80, false, Kind::Distortion);
    let img = synth::natural(90, 80, 1);
    let (cw, ch) = geo.output_size();
    let err = on_device(dev, move |ctx| {
        Ok(klens::lens(ctx, &img, &geo, Rect::whole(cw, ch)).is_err())
    });
    assert!(err);
}
