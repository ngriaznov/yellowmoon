//! Diagnostics of the T3b bake (F-04c): where the baking error and the bake time come from. All ignored; the
//! numbers they print are in `docs/24-film.md` section 3.8a.
mod bake_common;
use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_film::Output;
use ym_film::bake::*;

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
                    let rel = |a: [f64; 3], b: [f64; 3]| [0, 1, 2].map(|k| (a[k] - b[k]).abs());
                    [
                        rel(dt.log_e_film, bt.log_e_film),
                        rel(dt.density_film, bt.density_film),
                        rel(dt.density_print, bt.density_print),
                        // codes: front only, then whole
                        {
                            let (x, y) =
                                (codes10(&cine, space, d_out), codes10(&cine, space, front));
                            rel(x, y)
                        },
                        {
                            let (x, y) =
                                (codes10(&cine, space, d_out), codes10(&cine, space, b_out));
                            rel(x, y)
                        },
                    ]
                });
                let names = [
                    "log_e_film",
                    "density_film",
                    "density_print",
                    "code front",
                    "code whole",
                ];
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
    #[allow(clippy::type_complexity)]
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

/// The fast path of `inverse_coordinates` against the continuation alone, on every node of T3b's lattice for a
/// stock and output (`YM_BAKE_STOCKS`, `YM_BAKE_SIZES`, `YM_BAKE_OUTPUTS`): the largest coordinate difference and
/// how many nodes differ by more than 1e-5 and 1e-3.
#[test]
#[ignore = "prints how the fast inverse agrees with the continuation on the lattice; minutes"]
fn fast_inverse_on_the_lattice() {
    let plan = ReportPlan::from_env();
    for out in &plan.outputs {
        let cine = CineRendering::new(out.target());
        for film in &plan.stocks {
            let c = chain(film, *out);
            let size = plan.sizes[0];
            let t = baker(4).t3b(&c, 3);
            let (lo, hi) = (t.axis_min(), t.axis_max());
            let nodes: Vec<usize> = (0..size * size * size).collect();
            let diffs: Vec<(f64, f64)> = par_map(&nodes, 4, |&n| {
                let idx = [n % size, (n / size) % size, n / (size * size)];
                let d = [0, 1, 2].map(|k| {
                    let u = idx[k] as f64 / (size - 1) as f64;
                    lo[k] + (hi[k] - lo[k]) * u * u
                });
                let disp = c.scan_stage().display(d);
                let (a, b) = (
                    cine.inverse_coordinates(disp),
                    cine.inverse_coordinates_reference(disp),
                );
                let e = (0..3)
                    .map(|k| (a.coordinates[k] - b.coordinates[k]).abs())
                    .fold(0.0, f64::max);
                let rho = a.coordinates[1].hypot(a.coordinates[2]);
                (e.max((a.chroma_fraction - b.chroma_fraction).abs()), rho)
            });
            if std::env::var("YM_LIST").is_ok() {
                for (n, _) in diffs
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.0 > 1e-3)
                    .take(40)
                {
                    let idx = [n % size, (n / size) % size, n / (size * size)];
                    let dd = [0, 1, 2].map(|k| {
                        let u = idx[k] as f64 / (size - 1) as f64;
                        lo[k] + (hi[k] - lo[k]) * u * u
                    });
                    let disp = c.scan_stage().display(dd);
                    let (a, b) = (
                        cine.inverse_coordinates(disp),
                        cine.inverse_coordinates_reference(disp),
                    );
                    eprintln!(
                        "node {idx:?} d {dd:.3?} disp {disp:.5?} fast {:.4?} ({:.3}) ref {:.4?} ({:.3})",
                        a.coordinates, a.chroma_fraction, b.coordinates, b.chroma_fraction
                    );
                }
            }
            let worst = diffs.iter().map(|d| d.0).fold(0.0, f64::max);
            let over = |t: f64| diffs.iter().filter(|d| d.0 > t).count();
            let min_rho = diffs
                .iter()
                .filter(|d| d.0 > 1e-6)
                .map(|d| d.1)
                .fold(f64::MAX, f64::min);
            let above = diffs.iter().filter(|d| d.1 >= min_rho).count();
            eprintln!(
                "smallest chroma of a differing node {min_rho:.3}; nodes at or above it: {above}"
            );
            eprintln!(
                "{out:?} {film:22} {size}^3: nodes {} largest {worst:.2e}; over 1e-6: {}, 1e-5: {}, 1e-3: {}",
                diffs.len(),
                over(1e-6),
                over(1e-5),
                over(1e-3)
            );
        }
    }
}

/// Where the T3b bake time goes (diagnostic): the scan, the fast inverse and the continuation, one thread, over
/// the nodes of a 33^3 lattice.
#[test]
#[ignore = "diagnostic; timing"]
fn t3b_cost_split() {
    use std::time::Instant;
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let c = chain("kodak_portra_400", out);
    let _ = cine.inverse_coordinates([0.5; 3]);
    let t = baker(1).t3b(&c, 3);
    let (lo, hi) = (t.axis_min(), t.axis_max());
    let size = 33usize;
    let ds: Vec<[f64; 3]> = (0..size * size * size)
        .map(|n| {
            let idx = [n % size, (n / size) % size, n / (size * size)];
            [0, 1, 2].map(|k| {
                let u = idx[k] as f64 / (size - 1) as f64;
                lo[k] + (hi[k] - lo[k]) * u * u
            })
        })
        .collect();
    let t0 = Instant::now();
    let disp: Vec<[f64; 3]> = ds.iter().map(|d| c.scan_stage().display(*d)).collect();
    let scan = t0.elapsed().as_secs_f64() / ds.len() as f64 * 1e6;
    let t0 = Instant::now();
    let s: f64 = disp
        .iter()
        .map(|d| cine.inverse_coordinates(*d).coordinates[0])
        .sum();
    let fast = t0.elapsed().as_secs_f64() / ds.len() as f64 * 1e6;
    let t0 = Instant::now();
    let s2: f64 = disp
        .iter()
        .map(|d| cine.inverse_coordinates_reference(*d).coordinates[0])
        .sum();
    let reference = t0.elapsed().as_secs_f64() / ds.len() as f64 * 1e6;
    // Per call: the time of each and whether the fast result equals the reference (so it was the fast path).
    let mut fast_t = (0usize, 0.0f64);
    let mut slow_t = (0usize, 0.0f64);
    let mut lam = 0usize;
    for d in &disp {
        let t0 = Instant::now();
        let a = cine.inverse_coordinates(*d);
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        if a.chroma_fraction < 1.0 {
            lam += 1;
        }
        if dt < 6.0 {
            fast_t.0 += 1;
            fast_t.1 += dt
        } else {
            slow_t.0 += 1;
            slow_t.1 += dt
        }
    }
    eprintln!(
        "calls under 6 us: {} (mean {:.2}); over: {} (mean {:.2}); unreachable on the connected sheet {}",
        fast_t.0,
        fast_t.1 / fast_t.0.max(1) as f64,
        slow_t.0,
        slow_t.1 / slow_t.0.max(1) as f64,
        lam
    );
    // From the exact answer as the start: how often is it accepted (fast)?
    let (mut quick, mut total_t) = (0usize, 0.0f64);
    let mut n_reach = 0usize;
    let mut slow_examples = 0;
    for d in &disp {
        let r = cine.inverse_coordinates_reference(*d);
        if r.chroma_fraction < 1.0 {
            continue;
        }
        n_reach += 1;
        let t0 = Instant::now();
        let a = cine.inverse_coordinates_from(*d, &r);
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        total_t += dt;
        if dt < 6.0 {
            quick += 1
        } else if slow_examples < 6 {
            slow_examples += 1;
            eprintln!(
                "slow from exact start: d {d:.5?} q {:.4?} got {:.4?} {:.1} us",
                r.coordinates, a.coordinates, dt
            );
        }
    }
    eprintln!(
        "from the exact start: {quick} of {n_reach} reachable calls under 6 us, mean {:.2}",
        total_t / n_reach as f64
    );
    let t0 = Instant::now();
    let s3: f64 = disp.iter().map(|d| cine.inverse(*d).scene.0[0]).sum();
    let plain = t0.elapsed().as_secs_f64() / ds.len() as f64 * 1e6;
    eprintln!("plain inverse {plain:.2} us ({s3:.1})");
    eprintln!(
        "per node, us: scan {scan:.2}, inverse_coordinates {fast:.2}, reference {reference:.2} ({s:.1} {s2:.1})"
    );
}

/// The baked T3b (rows, fast paths) against a table made by the continuation alone at every node: the number of
/// stored values that differ, and the largest difference.
#[test]
#[ignore = "builds two T3b tables per stock; minutes"]
fn t3b_against_the_reference_table() {
    let plan = ReportPlan::from_env();
    for out in &plan.outputs {
        let cine = CineRendering::new(out.target());
        for film in &plan.stocks {
            let c = chain(film, *out);
            let size = plan.sizes[0];
            let fast = baker(4).t3b(&c, size);
            let scan = c.scan_stage();
            let reference = Table3::bake_cine(
                size,
                fast.axis_min(),
                fast.axis_max(),
                T3B_POWER,
                4,
                |d, _| cine.inverse_coordinates_reference(scan.display(d)),
            );
            let mut worst = 0.0f64;
            let (mut o6, mut o5, mut o3) = (0, 0, 0);
            for (a, b) in fast.data().iter().zip(reference.data()) {
                let e = f64::from((a - b).abs());
                worst = worst.max(e);
                o6 += usize::from(e > 1e-6);
                o5 += usize::from(e > 1e-5);
                o3 += usize::from(e > 1e-3);
            }
            eprintln!(
                "{out:?} {film:22} {size}^3: largest {worst:.2e}; values over 1e-6: {o6}, 1e-5: {o5}, 1e-3: {o3}"
            );
        }
    }
}

/// How the row march of T3b spends its time (diagnostic): calls under 6 us against the rest, with their mean.
#[test]
#[ignore = "diagnostic; timing"]
fn t3b_row_march_split() {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let c = chain("kodak_portra_400", out);
    let _ = cine.inverse_coordinates([0.5; 3]);
    let t = baker(1).t3b(&c, 3);
    let scan = c.scan_stage();
    // counters: [fast n, fast ns, slow n, slow ns, unreachable n, start none n]
    let cnt: [AtomicU64; 6] = Default::default();
    let _ = Table3::bake_cine(65, t.axis_min(), t.axis_max(), T3B_POWER, 1, |d, start| {
        let display = scan.display(d);
        let t0 = Instant::now();
        let r = match start {
            Some(s) => cine.inverse_coordinates_from(display, s),
            None => cine.inverse_coordinates(display),
        };
        let ns = t0.elapsed().as_nanos() as u64;
        let k = if ns < 6000 { 0 } else { 2 };
        cnt[k].fetch_add(1, Relaxed);
        cnt[k + 1].fetch_add(ns, Relaxed);
        if r.chroma_fraction < 1.0 {
            cnt[4].fetch_add(1, Relaxed);
        }
        if start.is_none() {
            cnt[5].fetch_add(1, Relaxed);
        }
        r
    });
    let g = |i: usize| cnt[i].load(Relaxed) as f64;
    eprintln!(
        "under 6 us: {} (mean {:.2} us); over: {} (mean {:.2} us); unreachable {}; no start {}",
        g(0),
        g(1) / g(0).max(1.0) / 1e3,
        g(2),
        g(3) / g(2).max(1.0) / 1e3,
        g(4),
        g(5)
    );
}
