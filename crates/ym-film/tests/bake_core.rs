//! The per-pixel return of the Film node (F-04d, `docs/24` section 3.8c): the core table and the Newton steps against
//! the direct chain's `inverse_continuous`, on the print colours of the test photos.
//!
//! The ignored tests print the numbers of the doc:
//! `cargo test -p ym-film --release --test bake_core -- --ignored --nocapture --test-threads 1`
//! (`YM_CORE_DIMS=ns,nt,nk`, `YM_CORE_STEPS`, `YM_BAKE_STOCKS`, `YM_BAKE_OUTPUTS`, `YM_BAKE_STRIDE` narrow a run).
mod bake_common;
use bake_common::*;
use std::time::Instant;
use ym_color::display::cine::{
    CineRendering, CoreDims, CoreTable, DEFAULT_STEPS, scene_from_coordinates,
};
use ym_color::logc::logc4_encode_rgb;
use ym_film::Output;
use ym_film::bake::Table3;

fn env_list(k: &str) -> Option<Vec<usize>> {
    std::env::var(k)
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
}

fn dims_from_env() -> CoreDims {
    match env_list("YM_CORE_DIMS") {
        Some(v) if v.len() == 3 => CoreDims {
            ns: v[0],
            nt: v[1],
            nk: v[2],
        },
        _ => CoreDims::default(),
    }
}

/// The error of a per-pixel return against the direct chain on the print colours of the photos, for one stock and
/// output: the code difference after the Output's rendering.
fn photo_core_error(
    film: &str,
    out: Output,
    pixels: &[[f64; 3]],
    table: &CoreTable,
) -> (CodeStats, f64) {
    let c = chain(film, out);
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let displays: Vec<[f64; 3]> = par_map(pixels, 4, |p| c.run_tapped(*p).1.display);
    let direct: Vec<[f64; 3]> = par_map(&displays, 4, |d| c.return_stage(*d));
    let t = Instant::now();
    let mine: Vec<[f64; 3]> = par_map(&displays, 1, |d| {
        let q = cine.return_coordinates(table, *d);
        logc4_encode_rgb(scene_from_coordinates(q.coordinates).0)
    });
    let us = t.elapsed().as_secs_f64() * 1e6 / displays.len() as f64;
    let mut diffs = code_diffs(&cine, space, &direct, &mine);
    (CodeStats::of(&mut diffs), us)
}

#[test]
#[ignore = "prints the error and cost of the per-pixel return; minutes"]
fn core_return_on_photos() {
    let plan = ReportPlan::from_env();
    let pixels = photo_pixels(plan.stride.max(12));
    if pixels.is_empty() {
        eprintln!("no photos in testdata/film-bake: skipped");
        return;
    }
    let dims = dims_from_env();
    let steps = env_list("YM_CORE_STEPS").map_or(DEFAULT_STEPS, |v| v[0]);
    let t = Instant::now();
    let table = CoreTable::build(dims, 4).with_steps(steps);
    eprintln!(
        "core table {dims:?}, {steps} steps: built in {:.2} s on 4 threads, {} KiB",
        t.elapsed().as_secs_f64(),
        (table.data().len() * 12 + table.ln_reach().len() * 4) / 1024
    );
    for out in &plan.outputs {
        for film in &plan.stocks {
            let (s, us) = photo_core_error(film, *out, &pixels, &table);
            eprintln!("{out:?} {film:22} {s}  {us:.2} us/pixel");
        }
    }
}

#[test]
#[ignore = "prints the core table's build time on 1 and 4 threads"]
fn core_table_build_times() {
    let dims = dims_from_env();
    for threads in [1, 4] {
        let t = Instant::now();
        let table = CoreTable::build(dims, threads);
        eprintln!(
            "core table {dims:?} on {threads} thread(s): {:.2} s, {} KiB",
            t.elapsed().as_secs_f64(),
            (table.data().len() * 12 + table.ln_reach().len() * 4) / 1024
        );
    }
}

/// The worst pixels of one stock and output (diagnostic): the display colour, where it sits against the table's
/// boundary, and the round-trip error of the direct return and of the per-pixel one.
#[test]
#[ignore = "prints the worst pixels of the per-pixel return"]
fn core_return_worst_pixels() {
    let plan = ReportPlan::from_env();
    let pixels = photo_pixels(plan.stride.max(12));
    if pixels.is_empty() {
        return;
    }
    let dims = dims_from_env();
    let steps = env_list("YM_CORE_STEPS").map_or(DEFAULT_STEPS, |v| v[0]);
    let table = CoreTable::build(dims, 4).with_steps(steps);
    for out in &plan.outputs {
        let cine = CineRendering::new(out.target());
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            let displays: Vec<[f64; 3]> = par_map(&pixels, 4, |p| c.run_tapped(*p).1.display);
            let mut rows: Vec<(f64, [f64; 3])> = displays
                .iter()
                .map(|d| {
                    let direct = c.return_stage(*d);
                    let q = cine.return_coordinates(&table, *d);
                    let mine = logc4_encode_rgb(scene_from_coordinates(q.coordinates).0);
                    (code_diff(&cine, space, direct, mine).0, *d)
                })
                .collect();
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (err, d) in rows.iter().take(6) {
                let dc = cine.display_coordinates(*d);
                let direct = cine.inverse_coordinates(*d);
                let q = cine.return_coordinates(&table, *d);
                let q_start = cine.return_coordinates(&table.clone().with_steps(0), *d);
                let held = d.map(|v| v.clamp(1e-6, 1.0));
                let back = |q: [f64; 3]| {
                    let f = cine.forward(ym_color::gamut::LinearAwg4(scene_from_coordinates(q).0));
                    (0..3).map(|k| (f[k] - held[k]).abs()).fold(0.0, f64::max)
                };
                eprintln!(
                    "{out:?} {film} err {err:.3} display {d:.5?} y2 {:.3e} ca2 {:.3} cb2 {:.3} rho {:.3}\n    direct q {:.4?} frac {:.4} resid {:.2e}   mine q {:.4?} frac {:.4} resid {:.2e}",
                    dc.y2,
                    dc.ca2,
                    dc.cb2,
                    dc.ca2.hypot(dc.cb2),
                    direct.coordinates,
                    direct.chroma_fraction,
                    back(direct.coordinates),
                    q.coordinates,
                    q.chroma_fraction,
                    back(q.coordinates)
                );
                eprintln!(
                    "    table start {:.4?} resid {:.2e}; level axis {:?}; reach row {:.4?}",
                    q_start.coordinates,
                    back(q_start.coordinates),
                    table.level_axis(),
                    {
                        let (s0, ds) = table.level_axis();
                        let s = dc.y2.clamp(1e-300, 1.0);
                        let _ = (s0, ds, s);
                        0
                    }
                );
            }
        }
    }
}

/// The storage transfer functions T3b′ was tried with: `(encode, decode)`, both odd and C1.
type Transfer = (fn(f64) -> f64, fn(f64) -> f64);

fn sgn(v: f64, f: impl Fn(f64) -> f64) -> f64 {
    f(v.abs()).copysign(v)
}

fn transfers(name: &str) -> Transfer {
    match name {
        "srgb" => (
            |v| {
                sgn(v, |a| {
                    if a <= 0.003_130_8 {
                        12.92 * a
                    } else {
                        1.055 * a.powf(1.0 / 2.4) - 0.055
                    }
                })
            },
            |u| {
                sgn(u, |a| {
                    if a <= 12.92 * 0.003_130_8 {
                        a / 12.92
                    } else {
                        ((a + 0.055) / 1.055).powf(2.4)
                    }
                })
            },
        ),
        "cbrt" => (
            |v| {
                sgn(v, |a| {
                    if a > 0.008_856 {
                        a.cbrt()
                    } else {
                        7.787 * a + 16.0 / 116.0 - 16.0 / 116.0 * 0.0
                    }
                })
            },
            |u| u,
        ),
        "asinh3" => (|v| (v / 1e-3).asinh(), |u| 1e-3 * u.sinh()),
        "asinh4" => (|v| (v / 1e-4).asinh(), |u| 1e-4 * u.sinh()),
        _ => (|v| v, |u| u),
    }
}

/// T3b′'s own error (the scan table against the exact scan) on the print densities of the photos, in 10-bit codes
/// of the held display colour after C13: the part of the whole-chain error that the scan table adds before the
/// return. `YM_SCAN_SIZES` (default `33,49,65`), `YM_SCAN_POWERS` (x10, default `10,15,20`), `YM_SCAN_ENC`
/// (`lin`, `srgb`, `asinh3`, `asinh4`).
#[test]
#[ignore = "prints T3b' own error; minutes"]
fn scan_table_error() {
    let plan = ReportPlan::from_env();
    let pixels = photo_pixels(plan.stride.max(12));
    if pixels.is_empty() {
        return;
    }
    let sizes = env_list("YM_SCAN_SIZES").unwrap_or_else(|| vec![33, 49, 65]);
    let powers = env_list("YM_SCAN_POWERS").unwrap_or_else(|| vec![10, 15, 20]);
    let encs: Vec<String> = std::env::var("YM_SCAN_ENC")
        .unwrap_or_else(|_| "lin".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    for out in &plan.outputs {
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            let dens: Vec<[f64; 3]> = par_map(&pixels, 4, |p| {
                let t = c.run_tapped(*p).1;
                if c.paper().is_none() {
                    t.density_film
                } else {
                    t.density_print
                }
            });
            let reference = baker(4).t3b(&c, 3);
            let (lo, hi) = (reference.axis_min(), reference.axis_max());
            let scan = c.scan_stage();
            let code = |d: [f64; 3]| {
                space
                    .encode_clamped(d.map(|v| v.clamp(1e-6, 1.0)))
                    .map(|v| v * 1023.0)
            };
            for enc in &encs {
                let (fe, fd) = transfers(enc);
                for &size in &sizes {
                    for &pw in &powers {
                        let power = pw as f64 / 10.0;
                        let t =
                            Table3::bake_range(size, lo, hi, power, 4, |d| scan.display(d).map(fe));
                        let mut diffs: Vec<(f64, u32)> = par_map(&dens, 4, |d| {
                            let (a, b) = (code(scan.display(*d)), code(t.sample(*d).map(fd)));
                            let m = (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
                            (m, 0)
                        });
                        eprintln!(
                            "{out:?} {film:22} {enc} {size}^3 power {power}: {}",
                            CodeStats::of(&mut diffs)
                        );
                    }
                }
            }
        }
    }
}

/// A warp of a density axis for the experiment: `forward(d)` in `[0, 1]` and its inverse.
struct Warp {
    lo: f64,
    hi: f64,
    kind: String,
}

impl Warp {
    fn new(kind: &str, lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            kind: kind.to_owned(),
        }
    }
    /// `kind`: `pK` (power K/10 of the density), `eK` (uniform in `10^(-K/10 d)`), `lK` (log(d + K/1000)).
    fn param(&self) -> f64 {
        self.kind[1..].parse::<f64>().unwrap_or(10.0)
    }
    fn raw(&self, d: f64) -> f64 {
        let k = self.param();
        match &self.kind[..1] {
            "e" => 1.0 - 10f64.powf(-k / 10.0 * d),
            "l" => (d + k / 1000.0).ln(),
            _ => ((d - self.lo) / (self.hi - self.lo))
                .max(0.0)
                .powf(10.0 / k),
        }
    }
    fn forward(&self, d: f64) -> f64 {
        let (a, b) = (self.raw(self.lo), self.raw(self.hi));
        (self.raw(d) - a) / (b - a)
    }
    fn inverse(&self, w: f64) -> f64 {
        let k = self.param();
        let (a, b) = (self.raw(self.lo), self.raw(self.hi));
        let r = a + w * (b - a);
        match &self.kind[..1] {
            "e" => -(1.0 - r).max(1e-300).log10() * 10.0 / k,
            "l" => r.exp() - k / 1000.0,
            _ => self.lo + (self.hi - self.lo) * r.max(0.0).powf(k / 10.0),
        }
    }
}

/// T3b′ own error for warps of the density axis (`YM_SCAN_WARPS`, e.g. `p10,p20,p30,e10,e20,l10`), linear storage.
#[test]
#[ignore = "prints T3b' own error for axis warps; minutes"]
fn scan_table_warps() {
    let plan = ReportPlan::from_env();
    let pixels = photo_pixels(plan.stride.max(12));
    if pixels.is_empty() {
        return;
    }
    let sizes = env_list("YM_SCAN_SIZES").unwrap_or_else(|| vec![33, 49, 65]);
    let warps: Vec<String> = std::env::var("YM_SCAN_WARPS")
        .unwrap_or_else(|_| "p20,p30,e10,e20".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    for out in &plan.outputs {
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            let dens: Vec<[f64; 3]> = par_map(&pixels, 4, |p| {
                let t = c.run_tapped(*p).1;
                if c.paper().is_none() {
                    t.density_film
                } else {
                    t.density_print
                }
            });
            let reference = baker(4).t3b(&c, 3);
            let (lo, hi) = (reference.axis_min(), reference.axis_max());
            let scan = c.scan_stage();
            let code = |d: [f64; 3]| {
                space
                    .encode_clamped(d.map(|v| v.clamp(1e-6, 1.0)))
                    .map(|v| v * 1023.0)
            };
            for kind in &warps {
                let w: Vec<Warp> = (0..3).map(|k| Warp::new(kind, lo[k], hi[k])).collect();
                for &size in &sizes {
                    let t = Table3::bake_range(size, [0.0; 3], [1.0; 3], 1.0, 4, |u| {
                        scan.display([0, 1, 2].map(|k| w[k].inverse(u[k])))
                    });
                    let mut diffs: Vec<(f64, u32)> = par_map(&dens, 4, |d| {
                        let u = [0, 1, 2].map(|k| w[k].forward(d[k].clamp(lo[k], hi[k])));
                        let (a, b) = (code(scan.display(*d)), code(t.sample(u)));
                        ((0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max), 0)
                    });
                    eprintln!(
                        "{out:?} {film:22} warp {kind} {size}^3: {}",
                        CodeStats::of(&mut diffs)
                    );
                }
            }
        }
    }
}

/// Where T3b′'s error is (diagnostic): the largest error by the brightest display channel and by the smallest
/// density, for one warp and size (`YM_SCAN_WARPS` first entry, `YM_SCAN_SIZES` first entry).
#[test]
#[ignore = "prints where T3b' error is"]
fn scan_table_where() {
    let plan = ReportPlan::from_env();
    let pixels = photo_pixels(plan.stride.max(12));
    if pixels.is_empty() {
        return;
    }
    let size = env_list("YM_SCAN_SIZES").map_or(65, |v| v[0]);
    let kind = std::env::var("YM_SCAN_WARPS").unwrap_or_else(|_| "p20".into());
    let kind = kind.split(',').next().unwrap_or("p20").to_owned();
    for out in &plan.outputs {
        let space = output_space(*out);
        for film in &plan.stocks {
            let c = chain(film, *out);
            let dens: Vec<[f64; 3]> = par_map(&pixels, 4, |p| {
                let t = c.run_tapped(*p).1;
                if c.paper().is_none() {
                    t.density_film
                } else {
                    t.density_print
                }
            });
            let reference = baker(4).t3b(&c, 3);
            let (lo, hi) = (reference.axis_min(), reference.axis_max());
            let scan = c.scan_stage();
            let code = |d: [f64; 3]| {
                space
                    .encode_clamped(d.map(|v| v.clamp(1e-6, 1.0)))
                    .map(|v| v * 1023.0)
            };
            let w: Vec<Warp> = (0..3).map(|k| Warp::new(&kind, lo[k], hi[k])).collect();
            let t = Table3::bake_range(size, [0.0; 3], [1.0; 3], 1.0, 4, |u| {
                scan.display([0, 1, 2].map(|k| w[k].inverse(u[k])))
            });
            // bins of the brightest display channel: <0.01, <0.1, <0.5, <0.9, >=0.9
            let mut bins = [(0.0_f64, 0usize, [0.0_f64; 3], [0.0_f64; 3], 0.0_f64); 5];
            for d in &dens {
                let disp = scan.display(*d);
                let u = [0, 1, 2].map(|k| w[k].forward(d[k].clamp(lo[k], hi[k])));
                let (a, b) = (code(disp), code(t.sample(u)));
                let errs = [0, 1, 2].map(|k| (a[k] - b[k]).abs());
                let m = errs[0].max(errs[1]).max(errs[2]);
                let br = disp[0].max(disp[1]).max(disp[2]);
                let k = if br < 0.01 {
                    0
                } else if br < 0.1 {
                    1
                } else if br < 0.5 {
                    2
                } else if br < 0.9 {
                    3
                } else {
                    4
                };
                bins[k].1 += 1;
                if m > bins[k].0 {
                    bins[k].0 = m;
                    bins[k].2 = *d;
                    bins[k].3 = disp;
                    bins[k].4 = errs.iter().position(|e| *e == m).unwrap_or(0) as f64;
                }
            }
            for (k, b) in bins.iter().enumerate() {
                eprintln!(
                    "{out:?} {film} {kind} {size}^3 bright bin {k} ({} px): worst {:.3} at density {:.3?} display {:.4?} channel {}",
                    b.1, b.0, b.2, b.3, b.4
                );
            }
        }
    }
}
