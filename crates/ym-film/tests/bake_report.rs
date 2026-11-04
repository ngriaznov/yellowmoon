//! The baking error of the whole chain and the bake times (`docs/24-film.md` sections 3.8a and 8).
//!
//! * `baking_error_on_photos` is the gate: the table chain against the direct chain on the photos of
//!   `testdata/film-bake/` (skipped when the folder is missing), 8 stocks, Display P3 and sRGB, at the default
//!   table sizes: largest difference at most 1 code and 99.9 % within a quarter code (10-bit codes after the cine
//!   rendering and C13). Decision of the owner, 2026-11-04: "Gate photos, report random".
//! * The ignored tests print the numbers of the doc. One command runs them all:
//!   `cargo test -p ym-film --release --test bake_report -- --ignored --nocapture --test-threads 1`.
//!   `YM_BAKE_STOCKS`, `YM_BAKE_SIZES`, `YM_BAKE_OUTPUTS` (`p3`, `srgb`) and `YM_BAKE_STRIDE` narrow a run
//!   (`ReportPlan`); `YM_BAKE_THREADS` sets the thread counts of `bake_times`.
mod bake_common;
use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_film::Output;
use ym_film::bake::*;

/// One stock, one output, one T3b size: the whole-chain difference on `pixels`.
fn photo_stats(
    film: &str,
    out: Output,
    size: usize,
    pixels: &[[f64; 3]],
    direct: &[[f64; 3]],
    baker: &Baker,
) -> CodeStats {
    let cine = CineRendering::new(out.target());
    let c = chain(film, out);
    let f = baker
        .bake(
            &c,
            &BakeSettings {
                t3b_size: size,
                t3b_power: t3b_power(),
                ..BakeSettings::of(&c)
            },
        )
        .expect("bake");
    let baked: Vec<[f64; 3]> = par_map(pixels, 4, |p| f.run(*p));
    let mut d = code_diffs(&cine, output_space(out), direct, &baked);
    CodeStats::of(&mut d)
}

/// The gate of `docs/24` section 8 on the photos: every stock, both outputs, the default sizes. Every
/// `YM_BAKE_STRIDE`-th pixel of the photos in each direction (default 6: 373,078 pixels a stock; the ignored
/// `baking_error_every_pixel` takes all of them).
#[test]
fn baking_error_on_photos() {
    let stride = std::env::var("YM_BAKE_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let pixels = photo_pixels(stride);
    if pixels.is_empty() {
        eprintln!("no photos in testdata/film-bake: skipped (docs/24 section 3.8a)");
        return;
    }
    let mut failed = Vec::new();
    for out in [Output::DisplayP3, Output::Srgb] {
        for film in EIGHT {
            let c = chain(film, out);
            let direct: Vec<[f64; 3]> = par_map(&pixels, 4, |p| c.run(*p));
            let s = photo_stats(film, out, DEFAULT_T3B_SIZE, &pixels, &direct, &baker(4));
            eprintln!("{out:?} {film:22} T3b {DEFAULT_T3B_SIZE}^3: {s}");
            if s.max > 1.0 || s.p999 > 0.25 {
                failed.push(format!(
                    "{out:?} {film}: largest {:.3}, 99.9 % within {:.3}",
                    s.max, s.p999
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "above the bound (largest 1 code, 99.9 % within 0.25):\n{}",
        failed.join("\n")
    );
}

/// The same on every pixel of the 20 photos (13 million a stock; about an hour).
#[test]
#[ignore = "every pixel of the photos; very slow"]
fn baking_error_every_pixel() {
    let pixels = photo_pixels(1);
    if pixels.is_empty() {
        eprintln!("no photos in testdata/film-bake: skipped");
        return;
    }
    let plan = ReportPlan::from_env();
    for out in &plan.outputs {
        for film in &plan.stocks {
            let c = chain(film, *out);
            let direct: Vec<[f64; 3]> = par_map(&pixels, 4, |p| c.run(*p));
            let s = photo_stats(film, *out, DEFAULT_T3B_SIZE, &pixels, &direct, &baker(4));
            eprintln!("{out:?} {film:22} T3b {DEFAULT_T3B_SIZE}^3: {s}");
        }
    }
}

/// The random colours (100,000, LogC4 uniform in `[0, 1]^3`, seed 2024), measured and reported, not gated (owner,
/// 2026-11-04): the whole-chain difference by class of the print colour the colour makes. *Front*: the baked front
/// of the chain (T1, curves, couplers, T3a, paper) with an exact scan and return against the direct chain. The
/// classes of the whole-chain difference: the continuous inverse does not reach the print colour (`unreachable`),
/// a display channel held at the floor (1e-6 or less), and the reachable others.
///
/// `cargo test -p ym-film --release --test bake_report random_colours_by_class -- --ignored --nocapture`
#[test]
#[ignore = "prints the random-colour breakdown"]
fn random_colours_by_class() {
    let plan = ReportPlan::from_env();
    let random = random_colours(100_000, 2024, 0.0, 1.0);
    for out in &plan.outputs {
        let cine = CineRendering::new(out.target());
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            for &size in &plan.sizes {
                let f = baker(4)
                    .bake(
                        &c,
                        &BakeSettings {
                            t3b_size: size,
                            t3b_power: t3b_power(),
                            ..BakeSettings::of(&c)
                        },
                    )
                    .expect("bake");
                // class, whole-chain (largest, rounded), front difference
                let rows: Vec<(usize, f64, f64)> = par_map(&random, 4, |p| {
                    let (d_out, dt) = c.run_tapped(*p);
                    let b_out = f.run(*p);
                    let bt = f.run_tapped(*p).1;
                    let dens = if f.is_slide() {
                        bt.density_film
                    } else {
                        bt.density_print
                    };
                    let front = c.return_stage(c.scan_stage().display(dens));
                    let class = if cine.inverse_continuous(dt.display).chroma_fraction < 1.0 {
                        1
                    } else if dt.display.iter().any(|v| *v <= 1e-6) {
                        2
                    } else {
                        3
                    };
                    let (whole, _) = code_diff(&cine, space, d_out, b_out);
                    let (front_err, _) = code_diff(&cine, space, d_out, front);
                    (class, whole, front_err)
                });
                let names = ["", "unreachable", "channel held", "reachable others"];
                let report = |what: &str, mut v: Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    let n = v.len();
                    if n == 0 {
                        return;
                    }
                    eprintln!(
                        "{out:?} {film:22} {size}^3 {what:18} {:>5.1} %  largest {:>8.3}  99.9 % within {:>8.3}",
                        100.0 * n as f64 / random.len() as f64,
                        v[n - 1],
                        v[n * 999 / 1000]
                    );
                };
                report("all, whole chain", rows.iter().map(|r| r.1).collect());
                for (k, name) in names.iter().enumerate().skip(1) {
                    report(
                        name,
                        rows.iter().filter(|r| r.0 == k).map(|r| r.1).collect(),
                    );
                }
                report("all, front only", rows.iter().map(|r| r.2).collect());
                report(
                    "reachable, front",
                    rows.iter().filter(|r| r.0 == 3).map(|r| r.2).collect(),
                );
            }
        }
    }
}

/// The whole-chain baking error on the photos of `testdata/film-bake/` (skipped when the folder is missing) per
/// stock, T3b size and output: largest / 99.9 % within / share over a quarter code.
///
/// `cargo test -p ym-film --release --test bake_report whole_chain_on_photos -- --ignored --nocapture`
#[test]
#[ignore = "prints the baking error on the photos; slow (minutes)"]
fn whole_chain_on_photos() {
    let plan = ReportPlan::from_env();
    let pixels = photo_pixels(plan.stride);
    if pixels.is_empty() {
        eprintln!("no photos in testdata/film-bake: skipped");
        return;
    }
    eprintln!("{} photo pixels", pixels.len());
    for out in &plan.outputs {
        for film in &plan.stocks {
            let c = chain(film, *out);
            let direct: Vec<[f64; 3]> = par_map(&pixels, 4, |p| c.run(*p));
            let b = baker(4);
            for &size in &plan.sizes {
                let s = photo_stats(film, *out, size, &pixels, &direct, &b);
                let verdict = if s.max <= 1.0 && s.p999 <= 0.25 {
                    "met"
                } else {
                    "NOT met"
                };
                eprintln!("{out:?} {film:22} T3b {size:>3}^3: {s} {verdict}");
            }
        }
    }
}

/// Bake times of Portra 400 on Portra Endura (`YM_BAKE_THREADS`, default `4`; run `1` and `4` as separate
/// processes, each on a quiet machine): T1, the curves, T3a, T3b′ (the scan) and the core table of the per-pixel
/// return, which is built once per process and does not depend on the film, the paper or the output.
#[test]
#[ignore = "prints bake times; run alone on a quiet machine"]
fn bake_times() {
    use std::time::Instant;
    let threads: usize = std::env::var("YM_BAKE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let c = chain("kodak_portra_400", Output::DisplayP3);
    let lights = Lights::of(&c);
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let _ = ym_color::display::cine::CoreTable::build(Default::default(), threads);
    eprintln!(
        "{threads} thread(s): core table (once per process) {:.1} ms",
        ms(t)
    );
    let b = baker(threads);
    let t = Instant::now();
    let _ = b.t1(&c);
    eprintln!("{threads} thread(s): T1 {:.1} ms", ms(t));
    let t = Instant::now();
    let _ = b.curves(&c);
    eprintln!("{threads} thread(s): curves {:.1} ms", ms(t));
    for size in [17usize, 33, 49, 65, 97] {
        let t = Instant::now();
        let _ = b.t3a(&c, lights, size);
        eprintln!("{threads} thread(s): T3a {size}^3 {:.1} ms", ms(t));
    }
    for size in [33usize, 49, 65, 97, 113, 129, 161] {
        let t = Instant::now();
        let _ = b.t3b(&c, size);
        eprintln!("{threads} thread(s): T3b' {size}^3 {:.1} ms", ms(t));
    }
    // A slide stock bakes only its own T3b'.
    let slide = chain("fujifilm_velvia_100", Output::DisplayP3);
    let t = Instant::now();
    let _ = b.t3b(&slide, DEFAULT_T3B_SIZE);
    eprintln!(
        "{threads} thread(s): T3b' {DEFAULT_T3B_SIZE}^3 of a slide stock {:.1} ms",
        ms(t)
    );
}

/// The per-pixel cost of the colour part on the CPU, one thread (the numbers of F-05's 24 MP timing): the whole baked
/// chain, and the return alone, on photo pixels of Portra 400.
#[test]
#[ignore = "prints per-pixel costs; run alone on a quiet machine"]
fn per_pixel_cost() {
    use std::time::Instant;
    let pixels = photo_pixels(24);
    if pixels.is_empty() {
        return;
    }
    let c = chain("kodak_portra_400", Output::DisplayP3);
    let f = baker(4).bake(&c, &BakeSettings::of(&c)).expect("bake");
    let displays: Vec<[f64; 3]> = pixels.iter().map(|p| f.run_tapped(*p).1.display).collect();
    let n = pixels.len() as f64;
    let t = Instant::now();
    let s: f64 = pixels.iter().map(|p| f.run(*p)[0]).sum();
    let whole = t.elapsed().as_secs_f64() * 1e6 / n;
    let t = Instant::now();
    let s2: f64 = displays.iter().map(|d| f.return_to_logc4(*d)[0]).sum();
    let ret = t.elapsed().as_secs_f64() * 1e6 / n;
    let t = Instant::now();
    let s3: f64 = pixels.iter().map(|p| f.exposure_of(*p)[0]).sum();
    let front = t.elapsed().as_secs_f64() * 1e6 / n;
    eprintln!(
        "{} pixels: baked chain {whole:.2} us a pixel; the return alone {ret:.2}; stage 1 and 2 alone {front:.2} ({s:.1} {s2:.1} {s3:.1})",
        pixels.len()
    );
}
