//! The pool's quarantine (`crates/ym-gpu/src/pool.rs`, *Quarantine*): a buffer released while a `PassList` is recording
//! is not reused, poisoned or trimmed until that list has been submitted. The test reproduces the pattern of two bugs
//! found one at a time (a taps buffer released at record time and reused by the next upload of the list; a guide
//! buffer trimmed before the submit) and shows that it corrupts the first pass with the quarantine switched off and
//! cannot with it on.

mod common;

use common::gpu;
use ym_gpu::stream::PassList;

const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
const WORDS: usize = 4096;

/// Records "upload `1`s to A, copy A to X, release A, upload `2`s to a new buffer B, copy B to Y" in one list and returns
/// (X, Y) as read after the submission.
fn release_then_upload(ctx: &ym_gpu::device::GpuContext) -> (Vec<u32>, Vec<u32>) {
    let bytes = (WORDS * 4) as u64;
    let mut list = PassList::new(ctx, "quarantine");
    let x = list.keep(ctx.pool().acquire("x", bytes, USAGE).expect("x"));
    let y = list.keep(ctx.pool().acquire("y", bytes, USAGE).expect("y"));
    {
        let a = ctx.pool().acquire("a", bytes, USAGE).expect("a");
        ctx.queue.write_buffer(&a, 0, &vec![1u8; WORDS * 4]);
        list.encoder().copy_buffer_to_buffer(&a, 0, &x, 0, bytes);
        // `a` goes back to the pool here, at record time, as the taps buffer did.
    }
    {
        let b = ctx.pool().acquire("b", bytes, USAGE).expect("b");
        ctx.queue.write_buffer(&b, 0, &vec![2u8; WORDS * 4]);
        list.encoder().copy_buffer_to_buffer(&b, 0, &y, 0, bytes);
    }
    let rx = list.read(&x, 0, bytes).expect("read x");
    let ry = list.read(&y, 0, bytes).expect("read y");
    let done = list.submit().expect("submit").finish().expect("finish");
    let words = |b: &[u8]| -> Vec<u32> {
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    (words(done.read(rx)), words(done.read(ry)))
}

#[test]
fn a_buffer_released_while_a_list_records_is_not_reused_in_that_list() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            |ctx, _| {
                let one = u32::from_le_bytes([1; 4]);
                let two = u32::from_le_bytes([2; 4]);

                // Without the quarantine the second acquire recycles the first buffer: the first pass reads the
                // second upload (or the poison). This is the detection: the pattern is corrupt.
                ctx.pool().set_quarantine(false);
                ctx.pool().trim();
                let (x, _) = release_then_upload(ctx);
                ctx.pool().set_quarantine(true);
                assert!(
                    x.iter().any(|&w| w != one),
                    "without the quarantine the first pass must see the later write"
                );

                // With it, nothing is reused inside the list and both passes read their own data.
                let (x, y) = release_then_upload(ctx);
                assert!(x.iter().all(|&w| w == one), "first pass reads its own data");
                assert!(y.iter().all(|&w| w == two));

                // After the submit the buffers are free again and the quarantine is empty.
                let st = ctx.pool().stats();
                assert_eq!(st.quarantined_bytes, 0);
                assert!(st.peak_quarantined_bytes > 0);
                let again = ctx.pool().acquire("again", (WORDS * 4) as u64, USAGE)?;
                assert_eq!(ctx.pool().stats().reuses, st.reuses + 1);
                drop(again);
                Ok(())
            },
        )
        .expect("job");
}

#[test]
fn a_trim_while_a_list_records_destroys_nothing_the_list_uses() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            |ctx, _| {
                let bytes = (WORDS * 4) as u64;
                ctx.pool().trim();
                let mut list = PassList::new(ctx, "trim");
                let out = list.keep(ctx.pool().acquire("out", bytes, USAGE)?);
                {
                    let a = ctx.pool().acquire("guide", bytes, USAGE)?;
                    ctx.queue.write_buffer(&a, 0, &vec![7u8; WORDS * 4]);
                    list.encoder().copy_buffer_to_buffer(&a, 0, &out, 0, bytes);
                }
                // The raster guide's pattern: released at record time, then a trim before the submit.
                assert_eq!(
                    ctx.pool().trim(),
                    0,
                    "the quarantined buffer is not trimmed"
                );
                let r = list.read(&out, 0, bytes)?;
                let done = list.submit()?.finish()?;
                assert!(done.read(r).iter().all(|&b| b == 7));
                // Once submitted it is an ordinary free buffer and a trim may take it.
                assert!(ctx.pool().trim() > 0);
                Ok(())
            },
        )
        .expect("job");
}

#[test]
fn a_dropped_list_releases_its_quarantine() {
    let Some(dev) = gpu() else { return };
    dev.gpu
        .run(
            ym_gpu::queue::JobSpec::new(ym_gpu::queue::Priority::Viewport),
            |ctx, _| {
                ctx.pool().trim();
                let list = PassList::new(ctx, "dropped");
                drop(ctx.pool().acquire("a", 4096, USAGE)?);
                assert!(ctx.pool().stats().quarantined_bytes > 0);
                drop(list);
                assert_eq!(ctx.pool().stats().quarantined_bytes, 0);
                Ok(())
            },
        )
        .expect("job");
}
