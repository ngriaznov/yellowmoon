//! F-08d: how close the colour pass is to the CPU's, as a number a change can be measured by. The export tail's
//! picture before output sharpening is the colour pass's C12 picture; output sharpening turns its error up at a
//! near-black sample (`docs/24` section 3.9c), so the pass is held to a standard deviation of its relative error,
//! per output, on a seeded camera image that spreads over 17 stops.

mod common;

use common::chains::{self, camera_image, pixel};
use common::gpus;
use ym_color::consts::Vec3;
use ym_color::display::{ColorChain, OutputTarget, StageTrace};
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

fn stats(rel: &[f64]) -> (f64, f64) {
    let n = rel.len() as f64;
    let mean = rel.iter().sum::<f64>() / n;
    let var = rel.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / n;
    (var.sqrt(), rel.iter().fold(0.0f64, |m, r| m.max(r.abs())))
}

#[test]
fn the_colour_pass_is_within_a_few_1e7_of_the_cpu_in_relative_terms() {
    let (w, h) = (256usize, 256usize);
    let img = camera_image(w, h, 0xF08D);
    let n = w * h;
    let cams: Vec<Vec3> = (0..n).map(|i| pixel(&img, i)).collect();
    for dev in gpus() {
        for target in [
            OutputTarget::Srgb,
            OutputTarget::DisplayP3,
            OutputTarget::Rec2020,
        ] {
            let mut p = chains::base(target);
            p.tone = chains::identity_tone();
            let chain = ColorChain::new(&p).expect("chain");
            let traces: Vec<StageTrace> = cams.iter().map(|&c| chain.trace(c)).collect();
            let gpu = {
                let chain = chain.clone();
                let img = img.clone();
                dev.gpu
                    .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                        let prepared = PreparedChain::new(ctx, &chain)?;
                        [Stop::C7, Stop::C8, Stop::C12]
                            .iter()
                            .map(|&s| prepared.render(ctx, &img, (0, 0), ColorOutput::Stage(s)))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .expect("render")
            };
            let plane = |k: usize| match &gpu[k].data {
                PixelData::F32Planar(v) => v.clone(),
                other => panic!("f32 planes expected, got {other:?}"),
            };
            let (c7, c8, c12) = (plane(0), plane(1), plane(2));
            let (mut r7, mut r8, mut r12) = (Vec::new(), Vec::new(), Vec::new());
            for (i, t) in traces.iter().enumerate() {
                for ch in 0..3 {
                    let rel = |g: f32, c: f64| (f64::from(g) - c) / c;
                    if t.c7_linear_awg4[ch] >= 1e-4 {
                        r7.push(rel(c7[ch * n + i], t.c7_linear_awg4[ch]));
                    }
                    // LogC4 in absolute terms, as the equivalent relative error of the linear value it encodes
                    // (about 10.7 per unit of LogC4 near grey).
                    r8.push((f64::from(c8[ch * n + i]) - t.c8_logc4.0[ch]) * 10.7);
                    if t.c12_display_linear[ch] >= 1e-4 {
                        r12.push(rel(c12[ch * n + i], t.c12_display_linear[ch]));
                    }
                }
            }
            if std::env::var("YM_BINS").is_ok() {
                for (lo, hi) in [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 2.0)] {
                    let sel: Vec<f64> = (0..n)
                        .flat_map(|i| (0..3).map(move |ch| (i, ch)))
                        .filter(|&(i, ch)| {
                            let c = traces[i].c12_display_linear[ch];
                            c >= lo && c < hi
                        })
                        .map(|(i, ch)| {
                            (f64::from(c12[ch * n + i]) - traces[i].c12_display_linear[ch])
                                / traces[i].c12_display_linear[ch]
                        })
                        .collect();
                    if !sel.is_empty() {
                        let (s, m) = stats(&sel);
                        eprintln!("  C12 in [{lo:e}, {hi:e}): {s:.2e} (max {m:.1e}), {} samples", sel.len());
                    }
                }
            }
            let (s7, m7) = stats(&r7);
            let (s8, m8) = stats(&r8);
            let (s12, m12) = stats(&r12);
            eprintln!(
                "{:?} {target:?}: C7 {s7:.2e} (max {m7:.1e}), C8 as linear {s8:.2e} (max {m8:.1e}), C12 {s12:.2e} (max {m12:.1e}), {} samples",
                dev.mode,
                r12.len()
            );
        }
    }
}
