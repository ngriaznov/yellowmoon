//! Device images, command streams and chunked transfers (`docs/17-gpu.md` section 3): `GpuImage`,
//! `PassList`, `Pending`, the counters, the chunked upload and readback (of the raw words too: the
//! photo's own resident words are `DeviceRaw`, tested in `gpu_front`), and the test helpers of
//! `common` (tile = whole, poison-neutral, the 128 MiB design binding).
//!
//! Every device here runs in poison mode (see `common`).

mod common;

use std::sync::Arc;

use common::exact::{assert_bit_equal, assert_poison_neutral, assert_tiles_equal_whole};
use common::limits::{DESIGN_BINDING_BYTES, assert_fits_design_binding, assert_max_binding_bytes};
use common::{Rng, gpu, start_unpoisoned};
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::{Placement, RgbImage};
use ym_gpu::device::GpuContext;
use ym_gpu::device::{Gpu, GpuConfig, GpuError, StoragePreference};
use ym_gpu::image::{GpuImage, IMAGE_USAGE};
use ym_gpu::kernels::KernelId;
use ym_gpu::pool::PooledBuffer;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::{
    CHUNK_BYTES, MAX_GROUPS_PER_DIMENSION, PassList, read_bytes, read_bytes_in_chunks,
    upload_bytes, upload_bytes_in_chunks,
};

fn job() -> JobSpec {
    JobSpec::new(Priority::Export)
}

/// A kernel that is still a placeholder (it writes one NaN into binding 0 and nothing else), for the
/// tests of the pass list that only need *a* dispatch. The first one in `KernelId::ALL`, so the
/// tests do not name a kernel that a real one may already have replaced.
fn placeholder() -> KernelId {
    KernelId::ALL
        .into_iter()
        .find(|k| k.is_placeholder())
        .expect("a kernel that is still a placeholder")
}

/// Raw sensor words as the decoder produced them, two `u16` per `u32` (little-endian, the first
/// sample in the low half), row after row with no padding: sample `i = y * width + x` is half
/// `i & 1` of word `i >> 1`. This is the layout `DeviceRaw` keeps; the helper exercises the chunked
/// transfers on it without the photo's own parameters.
#[derive(Debug)]
struct Words {
    words: PooledBuffer,
    width: usize,
    height: usize,
}

impl Words {
    /// Bytes of the word buffer for `samples` samples.
    fn bytes(samples: usize) -> u64 {
        (samples as u64).div_ceil(2) * 4
    }

    fn upload(
        ctx: &GpuContext,
        label: &str,
        width: usize,
        height: usize,
        samples: &[u16],
    ) -> Result<Words, GpuError> {
        if width == 0 || height == 0 || samples.len() != width * height {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {} samples for a {width} x {height} readout",
                samples.len()
            )));
        }
        let words = ctx
            .pool()
            .acquire(label, Self::bytes(samples.len()), IMAGE_USAGE)?;
        upload_bytes(ctx, &words, 0, bytemuck::cast_slice(samples))?;
        Ok(Words {
            words,
            width,
            height,
        })
    }

    fn read(&self, ctx: &GpuContext) -> Result<Vec<u16>, GpuError> {
        let n = self.width * self.height;
        let bytes = read_bytes(ctx, &self.words, 0, n as u64 * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Pass lists
// ---------------------------------------------------------------------------------------------

#[test]
fn a_pass_list_dispatches_and_reads_back_in_one_submission_and_counts_it() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            // A placeholder kernel writes one NaN into binding 0 and nothing else: a dispatch that
            // really ran is visible as that NaN in an otherwise zero buffer.
            let buf = ctx.pool().acquire("stream a", 64, IMAGE_USAGE)?;
            upload_bytes(ctx, &buf, 0, &[0u8; 64])?;
            let before = ctx.counters();
            let mut passes = PassList::new(ctx, "counted");
            passes.dispatch(
                placeholder(),
                "placeholder",
                &[(0, buf.binding())],
                (1, 1, 1),
            )?;
            assert_eq!(passes.dispatch_count(), 1);
            let id = passes.read(&buf, 0, 64)?;
            let pending = passes.submit()?;
            assert_eq!(pending.dispatches(), 1);
            let mut done = pending.finish()?;
            let got = f32s(&done.take(id));
            assert!(got[0].is_nan(), "the dispatch ran");
            assert!(
                got[1..16].iter().all(|&v| v == 0.0),
                "and wrote nothing else"
            );
            assert!(done.timings.is_none(), "timestamps are off by default");
            let cost = ctx.counters().since(before);
            assert_eq!(cost.submissions, 1, "one list is one submission");
            assert_eq!(cost.dispatches, 1);
            assert_eq!(cost.readbacks, 1);
            assert_eq!(cost.readback_bytes, 64);
            assert_eq!(cost.uploads, 0);
            Ok(())
        })
        .expect("job");
}

#[test]
fn dispatch_refuses_more_workgroups_than_the_limit_and_records_nothing_for_an_empty_one() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let buf = ctx.pool().acquire("stream b", 64, IMAGE_USAGE)?;
            let mut passes = PassList::new(ctx, "limits");
            for groups in [
                (MAX_GROUPS_PER_DIMENSION + 1, 1, 1),
                (1, MAX_GROUPS_PER_DIMENSION + 1, 1),
                (1, 1, MAX_GROUPS_PER_DIMENSION + 1),
            ] {
                let err = passes
                    .dispatch(placeholder(), "too wide", &[(0, buf.binding())], groups)
                    .expect_err("K4");
                assert!(matches!(err, GpuError::InvalidInput(_)), "{err:?}");
            }
            passes.dispatch(placeholder(), "empty", &[(0, buf.binding())], (0, 4, 1))?;
            assert_eq!(passes.dispatch_count(), 0);
            passes.dispatch(
                placeholder(),
                "largest",
                &[(0, buf.binding())],
                (MAX_GROUPS_PER_DIMENSION, 1, 1),
            )?;
            assert_eq!(passes.dispatch_count(), 1);
            passes.submit()?.wait()
        })
        .expect("job");
}

#[test]
fn the_buffers_of_a_list_stay_out_of_the_pool_until_it_is_dropped() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            ctx.pool().trim();
            let base = ctx.pool().stats().in_use_bytes;
            let mut passes = PassList::new(ctx, "kept");
            let a = passes.uniform("params a", &[1u32, 2, 3, 4])?;
            let b = passes.uniform("params b", &[5u32, 6, 7, 8])?;
            let s = passes.scratch("scratch", 1 << 20)?;
            assert!(!Arc::ptr_eq(&a, &b), "each pass gets its own uniform block");
            let held = ctx.pool().stats().in_use_bytes - base;
            assert_eq!(
                held,
                a.class_size() + b.class_size() + s.class_size(),
                "all three are held by the list"
            );
            // Even after the caller lets go of its handles, the list keeps the buffers...
            drop((a, b, s));
            assert_eq!(ctx.pool().stats().in_use_bytes - base, held);
            // ...until the list itself goes: here without ever being submitted.
            drop(passes);
            assert_eq!(ctx.pool().stats().in_use_bytes, base);

            // A submitted list gives them back when its Pending is gone.
            let mut passes = PassList::new(ctx, "kept 2");
            let _ = passes.scratch("scratch", 1 << 20)?;
            let pending = passes.submit()?;
            assert!(ctx.pool().stats().in_use_bytes > base);
            pending.wait()?;
            assert_eq!(ctx.pool().stats().in_use_bytes, base);
            Ok(())
        })
        .expect("job");
}

#[test]
fn a_uniform_is_one_counted_upload_of_its_own_size() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let before = ctx.counters();
            let mut passes = PassList::new(ctx, "uniforms");
            passes.uniform("u", &[0u32; 8])?;
            passes.uniform("u", &[0u32; 4])?;
            let cost = ctx.counters().since(before);
            assert_eq!((cost.uploads, cost.uploaded_bytes), (2, 48));
            Ok(())
        })
        .expect("job");
}

#[test]
fn pass_timings_are_reported_when_the_adapter_has_timestamp_queries() {
    let Some(dev) = gpu() else { return };
    let g = Gpu::start(GpuConfig {
        timestamps: true,
        ..GpuConfig::default()
    })
    .expect("device");
    let supported = g.info().timestamp_query;
    eprintln!("timestamp queries on this adapter: {supported}");
    let timings = g
        .run(job(), |ctx, _| {
            let buf = ctx.pool().acquire("t", 4096, IMAGE_USAGE)?;
            let mut passes = PassList::new(ctx, "timed");
            for label in ["first", "second"] {
                passes.dispatch(placeholder(), label, &[(0, buf.binding())], (4, 4, 1))?;
            }
            Ok(passes.submit()?.finish()?.timings)
        })
        .expect("job");
    if supported {
        let t = timings.expect("timings");
        let labels: Vec<_> = t.passes.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(labels, ["first", "second"]);
        assert!(
            t.passes.iter().all(|p| p.ms.is_finite() && p.ms >= 0.0),
            "{t:?}"
        );
        if g.info().is_software {
            // A CPU rasteriser's timestamps are real clock readings: a pass of a few workgroups
            // takes milliseconds, not seconds. (A virtualised GPU's counters make no such promise.)
            assert!(t.passes.iter().all(|p| p.ms < 10_000.0), "{t:?}");
        }
        assert!(t.total_ms() >= 0.0);
    } else {
        assert!(timings.is_none());
    }
    // The shared test device never records them.
    assert!(!dev.gpu.info().timestamp_query);
}

// ---------------------------------------------------------------------------------------------
// Chunked transfers
// ---------------------------------------------------------------------------------------------

#[test]
fn a_61_megapixel_raw_plane_goes_up_and_comes_back_bit_exact_in_chunks() {
    let Some(dev) = gpu() else { return };
    // 9504 x 6336: the largest readout kept resident, 115 MiB of u16 (one 128 MiB binding).
    let (w, h) = (9504usize, 6336usize);
    let samples: Vec<u16> = (0..w * h)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 16) as u16)
        .collect();
    assert_eq!(samples.len() * 2, 120_434_688);
    const { assert!(CHUNK_BYTES <= DESIGN_BINDING_BYTES) };
    assert!(
        (samples.len() as u64 * 2) > CHUNK_BYTES,
        "the plane needs more than one chunk"
    );
    dev.gpu
        .run(job(), move |ctx, _| {
            let before = ctx.counters();
            let raw = Words::upload(ctx, "raw 61 MP", w, h, &samples)?;
            assert_fits_design_binding("61 MP raw words", raw.words.size());
            let up = ctx.counters().since(before);
            assert_eq!(up.uploads, 1, "one logical upload");
            assert_eq!(up.uploaded_bytes, 120_434_688);
            assert_eq!(up.submissions, 1, "one flush between its two chunks");
            let back = raw.read(ctx)?;
            let down = ctx.counters().since(before);
            assert_eq!(down.readbacks, 2, "read back in two chunks");
            assert_eq!(down.readback_bytes, 120_434_688);
            assert_eq!(back.len(), samples.len());
            assert!(back == samples, "the plane is not bit exact");
            assert_eq!((raw.width, raw.height), (w, h));
            Ok(())
        })
        .expect("61 MP job");
}

#[test]
fn chunk_sizes_odd_lengths_and_offsets_do_not_change_the_bytes() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let buf = ctx.pool().acquire("chunks", 4096, IMAGE_USAGE)?;
            let data: Vec<u8> = (0..103u32).map(|i| (i * 7 + 3) as u8).collect();
            for chunk in [1u64, 4, 5, 12, 16, 100, 103, 104, 1 << 20] {
                upload_bytes_in_chunks(ctx, &buf, 0, &[0xEEu8; 256], 64)?;
                let before = ctx.counters();
                upload_bytes_in_chunks(ctx, &buf, 8, &data, chunk)?;
                let up = ctx.counters().since(before);
                let size = (chunk / 4 * 4).max(4) as usize;
                assert_eq!(
                    up.submissions,
                    (data.len().div_ceil(size) - 1) as u64,
                    "chunk {chunk}"
                );
                assert_eq!(up.uploads, 1);
                for read_chunk in [4u64, 12, 1000] {
                    let got = read_bytes_in_chunks(ctx, &buf, 8, 103, read_chunk)?;
                    assert_eq!(got, data, "upload chunk {chunk}, read chunk {read_chunk}");
                }
                // The padding of the unaligned tail is zero; what lies before and after is untouched.
                let all = read_bytes(ctx, &buf, 0, 256)?;
                assert_eq!(&all[..8], &[0xEE; 8]);
                assert_eq!(all[8 + 103], 0, "the padding byte is zero");
                assert_eq!(&all[8 + 104..], &[0xEE; 256 - 8 - 104][..]);
            }
            Ok(())
        })
        .expect("job");
}

#[test]
fn transfers_outside_the_buffer_or_off_the_4_byte_grid_are_refused() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let buf = ctx.pool().acquire("range", 1024, IMAGE_USAGE)?;
            let size = buf.size();
            let class = buf.class_size();
            assert!(size <= class);
            for err in [
                upload_bytes(ctx, &buf, 2, &[0; 8]).expect_err("odd offset"),
                upload_bytes(ctx, &buf, class - 4, &[0; 8]).expect_err("past the end"),
                read_bytes(ctx, &buf, 2, 8)
                    .map(|_| ())
                    .expect_err("odd offset"),
                read_bytes(ctx, &buf, 0, class + 4)
                    .map(|_| ())
                    .expect_err("past the end"),
            ] {
                assert!(matches!(err, GpuError::InvalidInput(_)), "{err:?}");
            }
            // Exactly to the end is fine.
            upload_bytes(ctx, &buf, class - 8, &[7; 8])?;
            assert_eq!(read_bytes(ctx, &buf, class - 8, 8)?, [7; 8]);
            Ok(())
        })
        .expect("job");
}

// ---------------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------------

#[test]
fn gpu_images_round_trip_every_plane_count_and_odd_width_with_zero_padding() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let mut rng = Rng(0xA11CE);
            for planes in [1usize, 3, 4] {
                for (w, h) in [(1usize, 1usize), (5, 3), (8, 2), (33, 7), (64, 1)] {
                    let place = Placement {
                        x0: 6,
                        y0: 2,
                        frame_width: 100,
                        frame_height: 80,
                    };
                    let data: Vec<f32> = (0..planes * w * h)
                        .map(|_| rng.range(-4.0, 4.0) as f32)
                        .collect();
                    let img = GpuImage::upload(ctx, "round trip", w, h, planes, &data, place)?;
                    assert_eq!(img.stride % 2, 0);
                    assert!(img.stride >= w && img.stride - w <= 1);
                    assert_eq!(
                        (img.width, img.height, img.planes, img.place),
                        (w, h, planes, place)
                    );
                    assert_eq!(img.byte_len(), GpuImage::buffer_bytes(w, h, planes));
                    assert_bit_equal(&format!("{planes} planes {w}x{h}"), &img.read(ctx)?, &data);
                    // The padding column is zero, not poison.
                    let raw = f32s(&read_bytes(ctx, &img.buf, 0, img.byte_len())?);
                    for c in 0..planes {
                        for y in 0..h {
                            for x in w..img.stride {
                                assert_eq!(raw[c * img.plane_len() + y * img.stride + x], 0.0);
                            }
                        }
                    }
                }
            }
            // A three-plane host image keeps its placement through the device.
            let mut rgb = RgbImage::new(7, 5);
            for (i, v) in rgb.data.iter_mut().enumerate() {
                *v = i as f32 * 0.25;
            }
            rgb.place = Placement::whole(7, 5).offset(2, 4);
            let img = GpuImage::upload_rgb(ctx, "rgb", &rgb)?;
            assert_eq!(img.read_rgb(ctx)?, rgb);
            let one = GpuImage::upload(ctx, "one", 3, 3, 1, &[0.0; 9], Placement::whole(3, 3))?;
            assert!(matches!(one.read_rgb(ctx), Err(GpuError::InvalidInput(_))));
            Ok(())
        })
        .expect("job");
}

#[test]
fn gpu_images_refuse_what_the_kernels_could_not_index_or_bind() {
    let Some(dev) = gpu() else { return };
    let binding = dev.gpu.info().limits.max_storage_buffer_binding_size;
    dev.gpu
        .run(job(), move |ctx, _| {
            let place = Placement::whole(1, 1);
            for (w, h, planes) in [
                (0usize, 4usize, 3usize),
                (4, 0, 3),
                (4, 4, 2),
                (4, 4, 0),
                (4, 4, 5),
            ] {
                let err = GpuImage::alloc(ctx, "bad", w, h, planes, place).expect_err("invalid");
                assert!(
                    matches!(err, GpuError::InvalidInput(_)),
                    "{w}x{h}x{planes}: {err:?}"
                );
            }
            // More samples than a 32-bit index reaches (K4): refused before any allocation.
            let err =
                GpuImage::alloc(ctx, "index", 65_536, 65_536, 1, place).expect_err("u32 index");
            assert!(
                matches!(&err, GpuError::InvalidInput(m) if m.contains("32-bit")),
                "{err:?}"
            );
            // Wrong data length.
            let err = GpuImage::upload(ctx, "len", 4, 4, 3, &[0.0; 47], place).expect_err("length");
            assert!(matches!(err, GpuError::InvalidInput(_)), "{err:?}");
            // Wider than one binding: the pool says so. (Only checked where the binding is small
            // enough to allocate past it cheaply; lavapipe, the CI adapter, has 128 MiB.)
            if binding <= 1 << 30 {
                let w = (binding / 4 / 4096) as usize + 4;
                let err = GpuImage::alloc(ctx, "wide", w, 4096, 1, place).expect_err("too wide");
                assert!(matches!(err, GpuError::BufferTooLarge { .. }), "{err:?}");
            }
            Ok(())
        })
        .expect("job");
}

#[test]
fn a_new_image_is_poison_until_a_kernel_writes_it_and_goes_back_to_the_pool_when_dropped() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            ctx.pool().trim();
            let base = ctx.pool().stats().in_use_bytes;
            {
                let img = GpuImage::alloc(ctx, "unwritten", 9, 4, 3, Placement::whole(9, 4))?;
                assert!(ctx.pool().stats().in_use_bytes > base);
                let data = img.read(ctx)?;
                assert!(
                    data.iter().all(|v| v.is_nan()),
                    "an unwritten image reads as NaN"
                );
            }
            assert_eq!(
                ctx.pool().stats().in_use_bytes,
                base,
                "the buffer went back to the pool"
            );
            Ok(())
        })
        .expect("job");
}

#[test]
fn raw_words_survive_the_smallest_and_oddest_readouts() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            // 1 x 1 (one word, half of it padding), a single row or column of odd length, 2 x 2, and
            // an odd x odd readout whose rows share words: each comes back as it went up.
            for (w, h) in [
                (1usize, 1usize),
                (1, 7),
                (7, 1),
                (2, 2),
                (3, 3),
                (1, 2),
                (5, 5),
            ] {
                let samples: Vec<u16> = (0..w * h).map(|i| 65_535 - (i as u16) * 997).collect();
                let raw = Words::upload(ctx, "tiny raw", w, h, &samples)?;
                assert_eq!(raw.words.size(), Words::bytes(w * h), "{w}x{h}");
                assert_eq!(raw.read(ctx)?, samples, "{w}x{h}");
            }
            // An empty readout is refused, not uploaded.
            for (w, h) in [(0usize, 4usize), (4, 0)] {
                let err = Words::upload(ctx, "empty", w, h, &[]).expect_err("empty");
                assert!(matches!(err, GpuError::InvalidInput(_)), "{err:?}");
            }
            Ok(())
        })
        .expect("job");
}

#[test]
fn an_image_refuses_the_readback_of_another_image() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let place = Placement::whole(1, 1);
            let small = GpuImage::upload(ctx, "small", 3, 2, 1, &[1.0; 6], place)?;
            let big = GpuImage::upload(ctx, "big", 40, 30, 3, &vec![2.0; 3 * 40 * 30], place)?;
            let mut passes = PassList::new(ctx, "mixed up");
            let small_id = small.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            // The right image decodes; the one with more samples cannot, and says so (no slice panic).
            assert_eq!(small.take(&done, small_id)?, vec![1.0; 6]);
            let err = big.take(&done, small_id).expect_err("too short for `big`");
            assert!(matches!(err, GpuError::InvalidInput(_)), "{err:?}");
            Ok(())
        })
        .expect("job");
}

#[test]
fn reading_an_image_through_a_pass_list_equals_the_host_wrapper() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            let mut rng = Rng(7);
            let data: Vec<f32> = (0..3 * 13 * 9)
                .map(|_| rng.range(0.0, 1.0) as f32)
                .collect();
            let img = GpuImage::upload(ctx, "pl", 13, 9, 3, &data, Placement::whole(13, 9))?;
            let mut passes = PassList::new(ctx, "read image");
            let id = img.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            assert_bit_equal(
                "pass list against host wrapper",
                &img.take(&done, id)?,
                &img.read(ctx)?,
            );
            assert_bit_equal("pass list against the input", &img.take(&done, id)?, &data);
            Ok(())
        })
        .expect("job");
}

#[test]
fn raw_words_are_two_samples_per_word_low_half_first_with_no_row_padding() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(job(), |ctx, _| {
            // 5 x 3 = 15 samples: an odd width, and an odd count (the last word's high half is zero).
            let samples: Vec<u16> = (1..=15u16).map(|i| i * 1000).collect();
            let raw = Words::upload(ctx, "raw", 5, 3, &samples)?;
            assert_eq!(raw.words.size(), Words::bytes(15));
            assert_eq!(Words::bytes(15), 32);
            let words = read_bytes(ctx, &raw.words, 0, 32)?;
            let word =
                |i: usize| u32::from_le_bytes(words[4 * i..4 * i + 4].try_into().expect("4"));
            assert_eq!(word(0), 1000 | (2000 << 16));
            assert_eq!(word(2), 5000 | (6000 << 16), "row 1 starts mid-word");
            assert_eq!(word(7), 15_000, "the unused high half is zero");
            assert_eq!(raw.read(ctx)?, samples);
            let err = Words::upload(ctx, "raw", 5, 3, &samples[..14]).expect_err("length");
            assert!(matches!(err, GpuError::InvalidInput(_)));
            Ok(())
        })
        .expect("job");
}

// ---------------------------------------------------------------------------------------------
// The test helpers themselves
// ---------------------------------------------------------------------------------------------

#[test]
fn the_tile_helper_passes_a_pointwise_function_and_catches_one_that_depends_on_the_tile() {
    // A pointwise function of the absolute pixel position is tile-exact.
    let render = |r: Rect| {
        let mut v = Vec::with_capacity(r.w * r.h);
        for y in r.y..r.y + r.h {
            for x in r.x..r.x + r.w {
                v.push((x * 31 + y * 17) as f32 * 0.001);
            }
        }
        v
    };
    let whole = assert_tiles_equal_whole("pointwise", 100, 70, 1, &[32, 64], render);
    assert_eq!(whole.len(), 100 * 70);

    // A function of the position *inside the tile* is not, and the helper must say so.
    let caught = std::panic::catch_unwind(|| {
        assert_tiles_equal_whole("tile-relative", 100, 70, 1, &[32], |r: Rect| {
            (0..r.w * r.h).map(|i| (i % r.w) as f32).collect()
        });
    });
    let msg = caught.expect_err("a tile-dependent function must fail");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(
        msg.contains("not bit-identical") && msg.contains("tiles of 32 px"),
        "{msg}"
    );

    // A frame of NaN equals itself bit for bit, tiled or not: that is what a kernel reading a
    // poisoned buffer gives, so the helper must refuse it instead of calling it tile-exact.
    let caught = std::panic::catch_unwind(|| {
        assert_tiles_equal_whole("all NaN", 100, 70, 1, &[32], |r: Rect| {
            vec![f32::NAN; r.w * r.h]
        });
    });
    let msg = caught.expect_err("a frame of NaN is not a result");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(
        msg.contains("all NaN: whole frame") && msg.contains("nobody wrote"),
        "{msg}"
    );
}

#[test]
fn the_poison_helper_passes_a_kernel_that_writes_what_it_reads_and_catches_a_stale_read() {
    if gpu().is_none() {
        return;
    }
    // Writes everything it reads: identical on the poisoned and the plain device.
    let ok = assert_poison_neutral("uploaded then read", |g| {
        g.run(job(), |ctx, _| {
            let data: Vec<f32> = (0..3 * 6 * 4).map(|i| i as f32).collect();
            GpuImage::upload(ctx, "ok", 6, 4, 3, &data, Placement::whole(6, 4))?.read(ctx)
        })
        .expect("job")
    })
    .expect("a device");
    assert_eq!(ok.len(), 72);

    // Reads an image nobody wrote: NaN on the poisoned device, something else on the plain one.
    let caught = std::panic::catch_unwind(|| {
        assert_poison_neutral("stale", |g| {
            g.run(job(), |ctx, _| {
                GpuImage::alloc(ctx, "stale", 6, 4, 3, Placement::whole(6, 4))?.read(ctx)
            })
            .expect("job")
        });
    });
    assert!(caught.is_err(), "a stale read must be caught");
    // And the plain device really is plain.
    assert!(start_unpoisoned(StoragePreference::Auto).is_some());
}

#[test]
fn the_binding_helper_states_the_plan_numbers() {
    // The adaptive ratio keeps 12 f32 planes (48 bytes) per compute pixel with shadow matching:
    // 2.5 Mpixel fits the 128 MiB design binding, the 4 Mpixel the risk design proposed does not.
    let per_pixel = |r: Rect| (r.w * r.h) as u64 * 48;
    assert_max_binding_bytes(
        "adaptive ratio scratch",
        Rect::new(0, 0, 1581, 1581),
        per_pixel,
    );
    let caught = std::panic::catch_unwind(|| {
        assert_max_binding_bytes(
            "adaptive ratio scratch",
            Rect::new(0, 0, 2000, 2000),
            |r: Rect| (r.w * r.h) as u64 * 48,
        );
    });
    assert!(
        caught.is_err(),
        "4 Mpixel x 48 bytes = 192 MB is above 128 MiB"
    );
    // A function that ignores the rectangle is not a max_binding_bytes.
    let caught = std::panic::catch_unwind(|| {
        assert_max_binding_bytes("constant", Rect::new(0, 0, 100, 100), |_| 4096);
    });
    assert!(caught.is_err());
}
