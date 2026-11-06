//! The pass budget of a `PassList` (`docs/17-gpu.md` section 8): a list of thousands of compute passes is cut into
//! submissions of at most `PASS_BUDGET` passes, so Metal's 4096 outstanding command buffers per queue (wgpu-core opens about
//! two hal buffers per pass) are never reached, and the cut changes no byte: the queue keeps the order.
//!
//! The work is a chain of single-word copies between two buffers whose source and destination words change with the pass
//! index, so the result depends on every pass and on their order.

mod common;

use common::gpu;
use ym_gpu::device::{Gpu, GpuConfig, GpuContext, GpuError};
use ym_gpu::image::IMAGE_USAGE;
use ym_gpu::kernels::KernelId;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::{PASS_BUDGET, PassList, PassTimings, upload_bytes};

const PASSES: u32 = 3000;
const WORDS: u32 = 64;

/// What one run of the chain gave back.
struct Run {
    a: Vec<u32>,
    b: Vec<u32>,
    timings: Option<PassTimings>,
    submissions: u64,
    max_split_passes: u32,
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn chain(ctx: &GpuContext) -> Result<Run, GpuError> {
    let n = u64::from(WORDS) * 4;
    let a = ctx.pool().acquire("chain a", n, IMAGE_USAGE)?;
    let b = ctx.pool().acquire("chain b", n, IMAGE_USAGE)?;
    let init_a: Vec<u8> = (0..WORDS).flat_map(|j| (j as f32).to_le_bytes()).collect();
    let init_b: Vec<u8> = (0..WORDS)
        .flat_map(|j| (1000.0 + j as f32).to_le_bytes())
        .collect();
    upload_bytes(ctx, &a, 0, &init_a)?;
    upload_bytes(ctx, &b, 0, &init_b)?;
    let before = ctx.counters();
    let mut passes = PassList::new(ctx, "pass budget chain");
    for i in 0..PASSES {
        let (src, dst) = if i % 2 == 0 { (&a, &b) } else { (&b, &a) };
        let cs = (i * 5 + 1) % WORDS;
        let cd = (i * 11 + 3) % WORDS;
        // The copy kernel's uniform: source (stride, plane, column, row), destination likewise, size (w, h, planes, base).
        let u: [u32; 12] = [WORDS, WORDS, cs, 0, WORDS, WORDS, cd, 0, 1, 1, 1, 0];
        let ub = passes.uniform("chain uniform", &u)?;
        passes.dispatch(
            KernelId::Copy,
            "chain copy",
            &[(0, ub.binding()), (1, src.binding()), (2, dst.binding())],
            (1, 1, 1),
        )?;
    }
    let ra = passes.read(&a, 0, n)?;
    let rb = passes.read(&b, 0, n)?;
    let mut done = passes.submit()?.finish()?;
    let cost = ctx.counters().since(before);
    Ok(Run {
        a: words(&done.take(ra)),
        b: words(&done.take(rb)),
        timings: done.timings.take(),
        submissions: cost.submissions,
        max_split_passes: ctx.counters().max_split_passes,
    })
}

fn run_on(g: &Gpu) -> Run {
    g.run(JobSpec::new(Priority::Export), |ctx, _| chain(ctx))
        .expect("job")
}

fn device(timestamps: bool) -> Option<Gpu> {
    gpu()?;
    Some(
        Gpu::start(GpuConfig {
            timestamps,
            poison: true,
            ..GpuConfig::default()
        })
        .expect("device"),
    )
}

#[test]
fn a_list_of_three_thousand_passes_is_cut_by_the_pass_budget_and_gives_the_same_bytes() {
    let Some(g) = device(false) else { return };
    g.set_pass_budget(u32::MAX);
    let whole = run_on(&g);
    assert_eq!(whole.submissions, 1);
    assert_eq!(whole.max_split_passes, PASSES);
    assert_ne!(whole.a, whole.b);

    for (budget, label) in [(0, "default"), (7, "7"), (1, "1")] {
        let g = device(false).expect("device");
        g.set_pass_budget(budget);
        let cut = run_on(&g);
        let limit = if budget == 0 { PASS_BUDGET } else { budget };
        eprintln!(
            "pass budget {label}: {} submissions, at most {} passes in one",
            cut.submissions, cut.max_split_passes
        );
        assert_eq!(cut.a, whole.a, "budget {label}: buffer a");
        assert_eq!(cut.b, whole.b, "budget {label}: buffer b");
        assert!(
            cut.max_split_passes <= limit,
            "budget {label}: {}",
            cut.max_split_passes
        );
        assert!(cut.submissions >= u64::from(PASSES.div_ceil(limit)));
        assert_eq!(cut.submissions, u64::from(PASSES.div_ceil(limit)));
    }
}

#[test]
fn a_timed_list_is_cut_by_the_pass_budget_too_and_still_reports_its_passes() {
    let Some(g) = device(true) else { return };
    if !g.info().timestamp_query {
        eprintln!("no timestamp queries on this adapter: the timed split is not checked");
        return;
    }
    g.set_pass_budget(u32::MAX);
    let whole = run_on(&g);
    for budget in [0, 7] {
        let g = device(true).expect("device");
        g.set_pass_budget(budget);
        let cut = run_on(&g);
        let limit = if budget == 0 { PASS_BUDGET } else { budget };
        assert_eq!(cut.a, whole.a);
        assert_eq!(cut.b, whole.b);
        assert!(cut.max_split_passes <= limit);
        assert_eq!(cut.submissions, u64::from(PASSES.div_ceil(limit)));
        let t = cut.timings.expect("timings of a cut list");
        let w = whole.timings.as_ref().expect("timings of the whole list");
        assert_eq!(t.passes.len(), w.passes.len(), "the same passes are timed");
        assert!(!t.passes.is_empty());
        assert!(
            t.passes.iter().all(|p| p.ms.is_finite() && p.ms >= 0.0),
            "{t:?}"
        );
    }
}
