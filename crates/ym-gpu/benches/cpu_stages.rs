//! Per-stage and whole-chain timings of the CPU reference on 24 MP and 50 MP synthetic frames
//! (`docs/05` performance targets: CPU-only fallback export < 90 s per photo on an M1).
//!
//! Run: `cargo bench -p ym-gpu --bench cpu_stages` (all sizes), or pass `24`, `50` or both as
//! arguments, e.g. `cargo bench -p ym-gpu --bench cpu_stages -- 24`. Set `YM_BENCH_DEMOSAIC=all` to
//! time all three demosaic methods (default: the default method only).
//! The frames are synthetic (smooth gradient plus noise with some blown blobs), so absolute
//! numbers say how fast the code runs, not how it scales with image content (the stages are
//! content independent apart from R5, which only works on clipped neighbourhoods).

use std::time::Instant;

use ym_gpu::cpu::chain::{Chain, RawFrame};
use ym_gpu::cpu::demosaic::{self, DEFAULT_METHOD, Method};
use ym_gpu::cpu::denoise::{self, BilateralParams};
use ym_gpu::cpu::detail::{self, NoiseReduction};
use ym_gpu::cpu::gains;
use ym_gpu::cpu::highlights;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::normalize::{self, Levels};
use ym_gpu::cpu::resample;
use ym_gpu::cpu::sharpen;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::vignette::{self, VignetteProfile};
use ym_gpu::cpu::{Cfa, NoiseSigma, RgbImage};

/// CPU seconds (user + system, all threads) used by this process so far; `None` off Linux.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Skip "pid (comm) " then take fields 14 and 15 (utime, stime) in clock ticks of 1/100 s.
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let (u, s): (f64, f64) = (f.get(11)?.parse().ok()?, f.get(12)?.parse().ok()?);
    Some((u + s) / 100.0)
}

/// Times `f`. Prints wall seconds, megapixels per second and, on Linux, the CPU seconds used,
/// whose quarter is what an idle 4-core machine needs (wall time is inflated when other jobs share
/// the cores). Returns the result and the wall seconds.
fn timed<T>(label: &str, megapixels: f64, f: impl FnOnce() -> T) -> (T, f64) {
    let (c0, t) = (cpu_seconds(), Instant::now());
    let out = f();
    let s = t.elapsed().as_secs_f64();
    match (c0, cpu_seconds()) {
        (Some(a), Some(b)) => println!(
            "  {label:<34} wall {s:>7.3} s {:>7.1} MP/s | cpu {:>7.2} s ({:.2} s on 4 idle cores)",
            megapixels / s,
            b - a,
            (b - a) / 4.0
        ),
        _ => println!("  {label:<34} {s:>8.3} s  {:>8.1} MP/s", megapixels / s),
    }
    (out, s)
}

/// Frame dimensions (even, 3:2) for a pixel count in megapixels.
fn dims(mp: f64) -> (usize, usize) {
    let h = ((mp * 1e6 / 1.5).sqrt() as usize) & !1;
    ((h * 3 / 2) & !1, h)
}

fn raw_frame(w: usize, h: usize) -> Vec<u16> {
    let m = synth::noise_mosaic(w, h, Cfa::RGGB, 1);
    let mut raw: Vec<u16> = m
        .data
        .iter()
        .map(|&v| (512.0 + v * 15000.0) as u16)
        .collect();
    // Blown blobs so that R5 has work (about 1% of the frame).
    let mut rng = synth::Rng::new(5);
    for _ in 0..(w * h / 400_000).max(4) {
        let (cx, cy) = (
            (rng.next_f32() * w as f32) as usize,
            (rng.next_f32() * h as f32) as usize,
        );
        let r = 40usize;
        for y in cy.saturating_sub(r)..(cy + r).min(h) {
            for x in cx.saturating_sub(r)..(cx + r).min(w) {
                if (x as isize - cx as isize).pow(2) + (y as isize - cy as isize).pow(2)
                    < (r * r) as isize
                {
                    raw[y * w + x] = 16383;
                }
            }
        }
    }
    raw
}

fn run_size(mp: f64, all_methods: bool) {
    let (w, h) = dims(mp);
    let real_mp = (w * h) as f64 / 1e6;
    println!(
        "== {w} x {h} ({real_mp:.1} MP), {} threads",
        rayon::current_num_threads()
    );
    let raw = raw_frame(w, h);
    let levels = Levels::uniform(512.0, 16383.0);
    let tags = LensTags {
        vignetting: vec![0, -300, -1000, -2200, -3800, -5800, -8200],
        distortion: vec![0, 15, 55, 120, 210, 330, 480, 650, 850],
        ca: vec![0, 20, 60, 120, 200, 0, -20, -60, -120, -200],
    };
    let mut total = 0.0;

    let ((mut mosaic, mask), t) = timed("R1 normalize + clip mask", real_mp, || {
        normalize::normalize(&raw, w, h, Cfa::RGGB, &levels, None).expect("normalize")
    });
    total += t;
    let profile = VignetteProfile::from_tag(&tags.vignetting).expect("vignette");
    total += timed("R3 vignetting", real_mp, || {
        vignette::apply_vignette(&mut mosaic, &profile)
    })
    .1;
    total += timed("R4 gains", real_mp, || {
        gains::apply_gains(&mut mosaic, [2.0, 1.0, 1.6])
    })
    .1;
    total += timed("R5 highlights", real_mp, || {
        highlights::reconstruct(&mut mosaic, &mask).expect("highlights")
    })
    .1;

    let methods: &[Method] = if all_methods {
        &Method::ALL
    } else {
        &[DEFAULT_METHOD]
    };
    let mut rgb: Option<RgbImage> = None;
    for &m in methods {
        let (img, t) = timed(&format!("R6 demosaic {}", m.name()), real_mp, || {
            demosaic::demosaic(&mosaic, m)
        });
        if m == DEFAULT_METHOD {
            total += t;
            rgb = Some(img);
        }
    }
    let Some(mut img) = rgb else {
        let (img, t) = timed("R6 demosaic (default)", real_mp, || {
            demosaic::demosaic(&mosaic, DEFAULT_METHOD)
        });
        total += t;
        return finish(img, total, real_mp, w, h, &raw, &levels, &tags);
    };
    drop(mosaic);
    let (den, t) = timed("R6 bilateral denoise", real_mp, || {
        denoise::bilateral_opponent(&img, &NoiseSigma::Scalar(0.01), &BilateralParams::default())
    });
    total += t;
    let nr = NoiseReduction::auto(800.0);
    let (col, t) = timed("R7 colour guided filter (C=50)", real_mp, || {
        detail::chroma_guided_filter(&den, &NoiseSigma::Scalar(0.01), nr.color)
    });
    total += t;
    let (inj, t) = timed("R7 detail re-injection", real_mp, || {
        detail::inject_detail(&col, &img, nr.beta()).expect("equal sizes")
    });
    total += t;
    drop((col, den));
    img = inj;
    let p = sharpen::auto_sharpen(800.0, 100.0, Some(5.6));
    let (sharp, t) = timed(
        &format!("R8 Richardson-Lucy (n={})", p.iterations()),
        real_mp,
        || sharpen::richardson_lucy(&img, &p, &NoiseSigma::Scalar(0.004)),
    );
    total += t;
    drop(img);
    finish(sharp, total, real_mp, w, h, &raw, &levels, &tags);
}

#[allow(clippy::too_many_arguments)]
fn finish(
    img: RgbImage,
    mut total: f64,
    mp: f64,
    w: usize,
    h: usize,
    raw: &[u16],
    levels: &Levels,
    tags: &LensTags,
) {
    let geo = LensGeometry::new(w, h, tags, LensSwitches::default())
        .expect("lens")
        .with_auto_crop();
    let (cw, ch) = geo.output_size();
    let (out, t) = timed("R9 lens resample (dist + CA)", mp, || {
        resample::resample_lens(&img, &geo, Rect::whole(cw, ch)).expect("resample")
    });
    total += t;
    drop((img, out));
    println!(
        "  {:<34} {total:>8.3} s   (sum of the stages above)",
        "stages total"
    );

    // The whole chain through Chain::run, as the CPU fallback export would.
    let chain = Chain {
        levels: *levels,
        vignette: Some(VignetteProfile::from_tag(&tags.vignetting).expect("vignette")),
        gains: [2.0, 1.0, 1.6],
        highlight_recovery: true,
        method: DEFAULT_METHOD,
        denoise: Some(BilateralParams::default()),
        noise_reduction: NoiseReduction::auto(800.0),
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise: None,
        sharpen: Some(sharpen::auto_sharpen(800.0, 100.0, Some(5.6)).into()),
        lens: Some(geo),
    };
    let frame = RawFrame {
        width: w,
        height: h,
        cfa: Cfa::RGGB,
        data: raw,
        qstep: None,
    };
    let (res, t) = timed("Chain::run (R1 to R9, whole frame)", mp, || {
        chain.run(&frame).expect("chain")
    });
    println!(
        "  CPU fallback total for {mp:.0} MP: {t:.1} s (docs/05 target on an M1: < 90 s for 50 MP); output {} x {}",
        res.width, res.height
    );
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    let sizes: Vec<f64> = if args.is_empty() {
        vec![24.0, 50.0]
    } else {
        args.iter().filter_map(|a| a.parse().ok()).collect()
    };
    let all = std::env::var("YM_BENCH_DEMOSAIC").is_ok_and(|v| v == "all");
    for mp in sizes {
        run_size(mp, all);
    }
}
