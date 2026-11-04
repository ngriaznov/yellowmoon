//! Prints the numbers of F-04 for `docs/STATUS.md`: `cargo test -p ym-film --release --test bake_report -- --ignored --nocapture`.
mod bake_common;
use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_film::Output;
use ym_film::bake::*;

#[test]
#[ignore = "prints the F-04 numbers; slow"]
fn error_by_size() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let photos = load_photos(4);
    let photo_px: Vec<[f64; 3]> = photos
        .iter()
        .flat_map(|p| p.pixels.iter().copied())
        .collect();
    let random = random_colours(100_000, 2024, 0.0, 1.0);
    eprintln!(
        "{} photo pixels, {} random colours",
        photo_px.len(),
        random.len()
    );
    for film in EIGHT {
        let c = chain(film, out);
        let b = baker(4);
        for (sa, sb) in [(33usize, 33usize), (49, 49), (65, 65), (49, 97)] {
            let f = b
                .bake(
                    &c,
                    &BakeSettings {
                        t3a_size: sa,
                        t3b_size: sb,
                        ..BakeSettings::of(&c)
                    },
                )
                .unwrap();
            for (name, set) in [("random", &random), ("photos", &photo_px)] {
                let mut d = par_map(set, 4, |p| code_diff(&cine, space, c.run(*p), f.run(*p)));
                eprintln!(
                    "{film:22} T3a {sa} T3b {sb} {name:6}: {}",
                    CodeStats::of(&mut d)
                );
            }
        }
    }
}

/// The whole-chain baking error of `docs/24` section 3.8a, on the photos of `testdata/film-bake/` (skips when the
/// folder is missing): the table chain against the direct chain after the cine rendering and the 10-bit transfer,
/// per stock, T3b size and output; largest / 99.9 % within / share over a quarter code. The gate is a largest
/// difference of at most 1 code and 99.9 % within 0.25.
///
/// `cargo test -p ym-film --release --test bake_report whole_chain_on_photos -- --ignored --nocapture`
/// (`YM_BAKE_STOCKS`, `YM_BAKE_SIZES`, `YM_BAKE_OUTPUTS`, `YM_BAKE_STRIDE` narrow it; see `ReportPlan`).
#[test]
#[ignore = "prints the baking error on the photos; slow (minutes)"]
fn whole_chain_on_photos() {
    let plan = ReportPlan::from_env();
    let photos = photo_pixels(plan.stride);
    if photos.is_empty() {
        eprintln!("no photos in testdata/film-bake: skipped");
        return;
    }
    eprintln!("{} photo pixels", photos.len());
    for out in &plan.outputs {
        let cine = CineRendering::new(out.target());
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            let direct: Vec<[f64; 3]> = par_map(&photos, 4, |p| c.run(*p));
            let b = baker(4);
            for &size in &plan.sizes {
                let f = b
                    .bake(
                        &c,
                        &BakeSettings {
                            t3b_size: size,
                            ..BakeSettings::of(&c)
                        },
                    )
                    .unwrap();
                let baked: Vec<[f64; 3]> = par_map(&photos, 4, |p| f.run(*p));
                let mut d = code_diffs(&cine, space, &direct, &baked);
                let s = CodeStats::of(&mut d);
                let verdict = if s.max <= 1.0 && s.p999 <= 0.25 {
                    "met"
                } else {
                    "NOT met"
                };
                eprintln!(
                    "{out:?} {film:22} T3b {size:>3}^3: {s} {verdict}",
                );
            }
        }
    }
}

#[test]
#[ignore = "prints bake times; run alone on a quiet machine"]
fn bake_times() {
    use std::time::Instant;
    let c = chain("kodak_portra_400", Output::DisplayP3);
    let lights = Lights::of(&c);
    for threads in [1usize, 4] {
        let b = baker(threads);
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
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
        for size in [33usize, 49, 65, 97] {
            let t = Instant::now();
            let _ = b.t3b(&c, size);
            eprintln!("{threads} thread(s): T3b {size}^3 {:.1} ms", ms(t));
        }
    }
}

/// The baking-error gate of `docs/24` section 8 as written: the table chain (default sizes) against the direct
/// chain on 100,000 random colours and on the pixels of the test photos, after the cine rendering and the 10-bit
/// encode. It does not pass (see the numbers `error_by_size` prints); it stays here, ignored, so the gate is not
/// loosened and the result can be re-run.
#[test]
#[ignore = "F-04 gate, not met: see docs/24 section 3.8 (As built)"]
fn baking_error_gate() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let photos = load_photos(6);
    let photo_px: Vec<[f64; 3]> = photos
        .iter()
        .flat_map(|p| p.pixels.iter().copied())
        .collect();
    let random = random_colours(100_000, 2024, 0.0, 1.0);
    let mut failed = Vec::new();
    for film in EIGHT {
        let c = chain(film, out);
        let f = baker(4).bake(&c, &BakeSettings::of(&c)).unwrap();
        for (name, set) in [("random", &random), ("photos", &photo_px)] {
            let mut d = par_map(set, 4, |p| code_diff(&cine, space, c.run(*p), f.run(*p)));
            let s = CodeStats::of(&mut d);
            eprintln!("{film:22} {name:6}: {s}");
            if s.max > 1.0 || s.p999 > 0.25 {
                failed.push(format!("{film} {name}"));
            }
        }
    }
    assert!(failed.is_empty(), "above the bound: {failed:?}");
}

/// Where the error of one stock sits, stage by stage (diagnostic): the direct chain against (B) the baked front of
/// the chain with an exact scan and return, and (C) the whole table chain; and the largest differences of the
/// intermediate values.
///
/// `YM_BAKE_STOCKS=fujifilm_velvia_100 cargo test -p ym-film --release --test bake_report stage_by_stage -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic; prints where the baking error of a stock comes from"]
fn stage_by_stage() {
    let plan = ReportPlan::from_env();
    let photos = photo_pixels(plan.stride);
    if photos.is_empty() {
        eprintln!("no photos in testdata/film-bake: skipped");
        return;
    }
    for out in &plan.outputs {
        let cine = CineRendering::new(out.target());
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            let b = baker(4);
            for &size in &plan.sizes {
                let f = b
                    .bake(
                        &c,
                        &BakeSettings {
                            t3b_size: size,
                            ..BakeSettings::of(&c)
                        },
                    )
                    .unwrap();
                let rows: Vec<[[f64; 3]; 5]> = par_map(&photos, 4, |p| {
                    let (d_out, dt) = c.run_tapped(*p);
                    let (b_out, bt) = f.run_tapped(*p);
                    let scan = c.scan_stage();
                    let dens = if f.is_slide() {
                        bt.density_film
                    } else {
                        bt.density_print
                    };
                    let front = c.return_stage(scan.display(dens));
                    let rel = |a: [f64; 3], b: [f64; 3]| {
                        [0, 1, 2].map(|k| (a[k] - b[k]).abs())
                    };
                    [
                        rel(dt.log_e_film, bt.log_e_film),
                        rel(dt.density_film, bt.density_film),
                        rel(dt.density_print, bt.density_print),
                        // codes: front only, then whole
                        {
                            let (x, y) = (codes10(&cine, space, d_out), codes10(&cine, space, front));
                            rel(x, y)
                        },
                        {
                            let (x, y) = (codes10(&cine, space, d_out), codes10(&cine, space, b_out));
                            rel(x, y)
                        },
                    ]
                });
                let names = ["log_e_film", "density_film", "density_print", "code front", "code whole"];
                for (k, name) in names.iter().enumerate() {
                    let mut v: Vec<f64> = rows
                        .iter()
                        .map(|r| r[k].iter().copied().fold(0.0, f64::max))
                        .collect();
                    v.sort_by(f64::total_cmp);
                    let n = v.len();
                    eprintln!(
                        "{out:?} {film:22} {size:>3}^3 {name:14} max {:.3e} p99.9 {:.3e} p99 {:.3e} median {:.3e}",
                        v[n - 1],
                        v[n * 999 / 1000],
                        v[n * 99 / 100],
                        v[n / 2]
                    );
                }
            }
        }
    }
}

/// The pixels with the largest whole-chain difference of one stock (diagnostic): the density T3b reads, the
/// display colour, and the two codes.
///
/// `YM_BAKE_STOCKS=kodak_portra_160 YM_BAKE_SIZES=97 YM_BAKE_OUTPUTS=p3 cargo test -p ym-film --release --test bake_report worst_pixels -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic; prints the worst pixels"]
fn worst_pixels() {
    let plan = ReportPlan::from_env();
    let photos = photo_pixels(plan.stride);
    let out = plan.outputs[0];
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let film = &plan.stocks[0];
    let c = chain(film, out);
    let size = plan.sizes[0];
    let f = baker(4)
        .bake(
            &c,
            &BakeSettings {
                t3b_size: size,
                ..BakeSettings::of(&c)
            },
        )
        .unwrap();
    let mut rows: Vec<(f64, [f64; 3], [f64; 3], [f64; 3], [f64; 3], [f64; 3])> =
        par_map(&photos, 4, |p| {
            let (d_out, _) = c.run_tapped(*p);
            let (b_out, bt) = f.run_tapped(*p);
            let dens = if f.is_slide() {
                bt.density_film
            } else {
                bt.density_print
            };
            let disp = c.scan_stage().display(dens);
            let (a, b) = (codes10(&cine, space, d_out), codes10(&cine, space, b_out));
            let e = (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
            (e, *p, dens, disp, a, b)
        });
    rows.sort_by(|x, y| y.0.total_cmp(&x.0));
    for r in rows.iter().take(12) {
        eprintln!(
            "err {:.3} logc4 {:.4?} dens {:.4?} disp {:.5?} direct {:.2?} baked {:.2?}",
            r.0, r.1, r.2, r.3, r.4, r.5
        );
    }
    // The error by the smallest display channel relative to the display luminance.
    let mut bins = [[0usize; 2]; 8];
    let edges = [0.0, 0.005, 0.02, 0.05, 0.12, 0.3, 0.6, 1.1];
    for r in &rows {
        let y = 0.2126 * r.3[0] + 0.7152 * r.3[1] + 0.0722 * r.3[2];
        let rel = r.3.iter().copied().fold(f64::MAX, f64::min) / y.max(1e-9);
        let k = (0..7).rev().find(|&k| rel >= edges[k]).unwrap_or(0);
        bins[k][0] += 1;
        if r.0 > 0.25 {
            bins[k][1] += 1;
        }
    }
    for k in 0..7 {
        eprintln!(
            "min channel / Y in [{:.3}, {:.3}): {} pixels, {} over 1/4 code",
            edges[k],
            edges[k + 1],
            bins[k][0],
            bins[k][1]
        );
    }
}
