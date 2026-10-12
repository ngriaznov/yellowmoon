//! Tiled-versus-whole test (docs/05 and docs/07): a frame cut into tiles with halos, cropped (not
//! blended) and reassembled must equal the whole-frame result to within 1e-4. Uses the CPU runtime with
//! the seeded random-weight fixtures.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use common::{random_frame, read};
use ym_ml::cpu_runtime::CpuJdd;
use ym_ml::model::{ExecContext, MlError, TileInputs, TileModel, TileShape};
use ym_ml::tiles::{Frame, core_size, reflect_pad, run_tiled, run_tiled_with_progress, tile_count};

const TOL: f32 = 1e-4;

/// Whole-frame reference: reflect-pad by the halo (and up to a multiple of 4), run once, crop. Planar
/// `3 x 2h x 2w` result.
fn run_whole(model: &CpuJdd, frame: &Frame) -> Vec<f32> {
    let halo = model.halo_half_px();
    let pad_r = (4 - frame.width % 4) % 4;
    let pad_b = (4 - frame.height % 4) % 4;
    let padded = reflect_pad(frame, halo, halo, halo + pad_r, halo + pad_b);
    let shape = TileShape {
        height: padded.height,
        width: padded.width,
    };
    let out = model
        .run_dynamic(&TileInputs::new(shape, padded.as_slice().to_vec()).unwrap())
        .unwrap();
    let (ow, oh) = (2 * frame.width, 2 * frame.height);
    let full_w = shape.out_width();
    let h2 = 2 * halo;
    let mut data = vec![0.0f32; 3 * ow * oh];
    for c in 0..3 {
        for y in 0..oh {
            let src = &out.plane(c)[(h2 + y) * full_w + h2..(h2 + y) * full_w + h2 + ow];
            data[c * ow * oh + y * ow..c * ow * oh + (y + 1) * ow].copy_from_slice(src);
        }
    }
    data
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn model(file: &str, tile_side: usize) -> CpuJdd {
    CpuJdd::from_bytes(&read(file), None)
        .unwrap()
        .with_tile_shape(TileShape::square(tile_side))
}

#[test]
fn tiled_equals_whole_small_model() {
    // Halo 32 (half-res px), tile 96 -> core 32; the frame is not a multiple of the core or of 4.
    let m = model("golden_small.safetensors", 96);
    for &(w, h, seed) in &[(100usize, 70usize, 1u64), (37, 53, 2), (96, 96, 3)] {
        let frame = random_frame(w, h, seed);
        let tiled = run_tiled(&m, &frame, &ExecContext::cpu()).unwrap();
        assert_eq!((tiled.width, tiled.height), (2 * w, 2 * h));
        let d = max_diff(tiled.as_slice(), &run_whole(&m, &frame));
        assert!(d <= TOL, "frame {w}x{h}: tiled vs whole max diff {d:e}");
    }
}

#[test]
fn tiled_equals_whole_with_3x3_up_and_head() {
    let m = CpuJdd::from_bytes(&read("golden_k3.safetensors"), None).unwrap();
    let halo = m.halo_half_px();
    let m = m.with_tile_shape(TileShape::square(2 * halo + 24));
    let frame = random_frame(61, 49, 7);
    let tiled = run_tiled(&m, &frame, &ExecContext::cpu()).unwrap();
    let d = max_diff(tiled.as_slice(), &run_whole(&m, &frame));
    assert!(d <= TOL, "k3 model: tiled vs whole max diff {d:e}");
}

#[test]
fn frame_smaller_than_one_tile() {
    let m = model("golden_small.safetensors", 96);
    let frame = random_frame(12, 8, 9);
    let tiled = run_tiled(&m, &frame, &ExecContext::cpu()).unwrap();
    assert!(max_diff(tiled.as_slice(), &run_whole(&m, &frame)) <= TOL);
}

#[test]
fn a_too_small_halo_would_show_seams() {
    // Sanity check of the test itself: with a halo of 8 instead of 32 the tiles disagree with the whole run.
    struct ShortHalo(CpuJdd);
    impl TileModel for ShortHalo {
        fn input_shape(&self) -> TileShape {
            TileShape::square(8 * 2 + 32)
        }
        fn halo_half_px(&self) -> usize {
            8
        }
        fn run(
            &self,
            i: &TileInputs,
            o: &mut ym_ml::model::TileOutput,
            _c: &ExecContext<'_>,
        ) -> ym_ml::model::Result<()> {
            let out = self.0.run_dynamic(i)?;
            o.as_mut_slice().copy_from_slice(out.as_slice());
            Ok(())
        }
    }
    let inner = model("golden_small.safetensors", 96);
    let frame = random_frame(64, 64, 4);
    let whole = run_whole(&inner, &frame);
    let tiled = run_tiled(&ShortHalo(inner), &frame, &ExecContext::cpu()).unwrap();
    assert!(max_diff(tiled.as_slice(), &whole) > 1e-3);
}

#[test]
fn progress_callback_and_cancellation() {
    let m = model("golden_small.safetensors", 96);
    let frame = random_frame(70, 40, 5);
    let (cw, ch) = core_size(m.input_shape(), m.halo_half_px()).unwrap();
    let expected = tile_count(70, 40, m.input_shape(), m.halo_half_px()).unwrap();
    assert_eq!(expected, 70usize.div_ceil(cw) * 40usize.div_ceil(ch));
    let calls = AtomicUsize::new(0);
    run_tiled_with_progress(&m, &frame, &ExecContext::cpu(), |done, total| {
        assert_eq!(total, expected);
        calls.store(done, Ordering::SeqCst);
        true
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), expected);
    let r = run_tiled_with_progress(&m, &frame, &ExecContext::cpu(), |_, _| false);
    assert!(matches!(r, Err(MlError::Cancelled)));
}

#[test]
fn output_of_a_flat_frame_is_flat_away_from_the_borders() {
    // Constant planes: the network is translation invariant (up to its stride phase), so pixels at the same
    // phase agree wherever they are.
    let m = model("golden_small.safetensors", 96);
    let mut frame = Frame::zeros(80, 80);
    for c in 0..4 {
        frame.plane_mut(c).fill(0.4);
    }
    let out = run_tiled(&m, &frame, &ExecContext::cpu()).unwrap();
    let p = out.plane(1);
    let w = out.width;
    let mid = p[80 * w + 80];
    // strided convolutions make the result periodic with period 4 half-res = 8 full-res pixels
    for &(y, x) in &[(64usize, 64usize), (72, 88), (88, 72), (96, 96)] {
        assert!((p[y * w + x] - mid).abs() < 1e-4, "pixel ({y},{x})");
    }
}
