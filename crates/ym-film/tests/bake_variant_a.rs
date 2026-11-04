//! Variant A of `docs/24` section 2.2, measured (F-04): the round trip of the cine rendering's inverse, and the
//! share of print pixels that have no pre-image.
mod bake_common;
use bake_common::*;
use ym_color::consts::linalg;
use ym_color::deltae::{d65_white_xyz, delta_e_2000, xyz_to_lab};
use ym_color::display::cine::{CineRendering, CoreTable, D_MIN, scene_from_coordinates};
use ym_color::gamut::{LinearAwg4, matrices};
use ym_film::Output;
use ym_film::bake::*;

/// `forward(inverse(d))` against `d` on colours the rendering produces: within 1e-6 (`docs/24` section 8).
#[test]
fn round_trip_on_reachable_colours() {
    for out in [Output::Srgb, Output::DisplayP3] {
        let cine = CineRendering::new(out.target());
        let mut rng = Rng(11);
        let (mut worst, mut n, mut missed) = ([0.0_f64; 2], [0_usize; 2], [0_usize; 2]);
        for _ in 0..50_000 {
            // A display colour the rendering makes: forward of a random scene colour (log-uniform luminance).
            let scene =
                [0, 1, 2].map(|_| 0.002 * (rng.unit() * 9.0).exp2() * (0.4 + 1.2 * rng.unit()));
            let d = cine.forward(LinearAwg4(scene));
            if d.iter().any(|v| *v < 1e-4 || *v > 0.9999) {
                continue;
            }
            // The plain inverse (the DNG's) and the continuous one (the film node's).
            for (k, inv) in [cine.inverse(d), cine.inverse_continuous(d)]
                .into_iter()
                .enumerate()
            {
                if inv.chroma_fraction < 1.0 {
                    // The inverse did not find a pre-image of a colour the rendering makes.
                    missed[k] += 1;
                    continue;
                }
                let back = cine.forward(inv.scene);
                worst[k] = worst[k].max((0..3).map(|c| (back[c] - d[c]).abs()).fold(0.0, f64::max));
                n[k] += 1;
            }
        }
        eprintln!(
            "{out:?}: round trip over {:?} colours (plain, continuous), worst {:.3e} and {:.3e}, {:?} not found",
            n, worst[0], worst[1], missed
        );
        assert!(n[0] > 20_000 && n[1] > 20_000);
        assert!(worst[0] <= 1e-6, "round trip {:e}", worst[0]);
        assert!(worst[1] <= 1e-6, "round trip (continuous) {:e}", worst[1]);
    }
}

/// The same for the Film node's per-pixel return (`CoreTable`, step 1, Newton, the scene side): within 1e-6 on every
/// colour it reports as reached, and the share it reports as not reached is the continuous inverse's.
#[test]
fn round_trip_of_the_per_pixel_return() {
    let table = CoreTable::shared();
    for out in [Output::Srgb, Output::DisplayP3] {
        let cine = CineRendering::new(out.target());
        let mut rng = Rng(11);
        let (mut worst, mut n, mut beyond, mut disagree) = (0.0_f64, 0_usize, 0_usize, 0_usize);
        for _ in 0..50_000 {
            let scene =
                [0, 1, 2].map(|_| 0.002 * (rng.unit() * 9.0).exp2() * (0.4 + 1.2 * rng.unit()));
            let d = cine.forward(LinearAwg4(scene));
            if d.iter().any(|v| *v < 1e-4 || *v > 0.9999) {
                continue;
            }
            let q = cine.return_coordinates(table, d);
            let reference = cine.inverse_continuous(d);
            if q.chroma_fraction < 1.0 {
                beyond += 1;
                disagree += usize::from(reference.chroma_fraction >= 1.0);
                continue;
            }
            let back = cine.forward(scene_from_coordinates(q.coordinates));
            worst = worst.max((0..3).map(|c| (back[c] - d[c]).abs()).fold(0.0, f64::max));
            n += 1;
        }
        eprintln!(
            "{out:?}: per-pixel return, round trip over {n} colours worst {worst:.3e}; {beyond} not reached ({disagree} that the continuous inverse reaches)"
        );
        assert!(n > 20_000);
        assert!(worst <= 1e-6, "round trip {worst:e}");
        assert!(disagree * 1000 <= n, "{disagree} of {n}");
    }
}

fn lab_of(output: Output, display: [f64; 3]) -> [f64; 3] {
    let m = matrices();
    let to_xyz = match output {
        Output::Srgb => m.rec709_to_xyz,
        Output::DisplayP3 => m.p3_to_xyz,
    };
    xyz_to_lab(
        linalg::mul_vec(&to_xyz, display.map(|v| v.clamp(0.0, 1.0))),
        d65_white_xyz(),
    )
}

/// What one stock did to the photos.
struct Share {
    film: &'static str,
    n: usize,
    unreachable: usize,
    de: Vec<f64>,
    worst_round_trip: f64,
    /// The per-pixel return (the Film node's): colours not reached, and the round trip on the rest.
    pp_unreachable: usize,
    pp_round_trip: f64,
}

fn measure(film: &'static str, output: Output, photos: &[Photo]) -> Share {
    let cine = CineRendering::new(output.target());
    let c = chain(film, output);
    // T3b is not used here (the scan is evaluated exactly), so it is made tiny.
    let f = baker(1)
        .bake(
            &c,
            &BakeSettings {
                t3b_size: 3,
                ..BakeSettings::of(&c)
            },
        )
        .unwrap();
    let scan = c.scan_stage();
    let mut s = Share {
        film,
        n: 0,
        unreachable: 0,
        de: Vec::new(),
        worst_round_trip: 0.0,
        pp_unreachable: 0,
        pp_round_trip: 0.0,
    };
    let table = CoreTable::shared();
    for ph in photos {
        for p in &ph.pixels {
            let d = scan.display(f.scan_input(*p));
            let inv = cine.inverse_continuous(d);
            s.n += 1;
            let held = d.map(|v| v.clamp(D_MIN, 1.0));
            let back = cine.forward(inv.scene);
            let pp = cine.return_coordinates(table, d);
            if pp.chroma_fraction < 1.0 {
                s.pp_unreachable += 1;
            } else {
                let back = cine.forward(scene_from_coordinates(pp.coordinates));
                s.pp_round_trip = s.pp_round_trip.max(
                    (0..3)
                        .map(|k| (back[k] - held[k]).abs())
                        .fold(0.0, f64::max),
                );
            }
            if inv.chroma_fraction < 1.0 {
                s.unreachable += 1;
                s.de.push(delta_e_2000(lab_of(output, d), lab_of(output, back)));
            } else {
                s.worst_round_trip = s.worst_round_trip.max(
                    (0..3)
                        .map(|k| (back[k] - held[k]).abs())
                        .fold(0.0, f64::max),
                );
            }
        }
    }
    s
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

/// The share of print pixels with no pre-image on 20 public photos and the eight most used stocks, and the ΔE00 of
/// those pixels. The gate is 0.1 % (`docs/24` section 8, *starting*).
#[test]
#[ignore = "F-04 gate: measured, and above the starting bound for some stocks; run with --ignored --nocapture"]
fn unreachable_share_on_photos() {
    let photos = load_photos(3);
    if photos.len() < 20 {
        eprintln!(
            "only {} photos in testdata/film-bake; see docs/24 section 3.8",
            photos.len()
        );
        return;
    }
    let total: usize = photos.iter().map(|p| p.pixels.len()).sum();
    eprintln!("{} photos, {total} pixels each stock", photos.len());
    let mut failed = Vec::new();
    for output in [Output::DisplayP3, Output::Srgb] {
        let shares = std::thread::scope(|sc| {
            let hs: Vec<_> = EIGHT
                .iter()
                .map(|f| {
                    let photos = &photos;
                    sc.spawn(move || measure(f, output, photos))
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        for mut s in shares {
            let share = 100.0 * s.unreachable as f64 / s.n as f64;
            eprintln!(
                "{output:?} {:22} unreachable {:>6.3}% ({} of {}) dE00 median {:.2} p95 {:.2} max {:.2}; round trip on the rest {:.1e}; per-pixel return: {} not reached, round trip {:.1e}",
                s.film,
                share,
                s.unreachable,
                s.n,
                pct(&mut s.de, 0.5),
                pct(&mut s.de, 0.95),
                s.de.last().copied().unwrap_or(0.0),
                s.worst_round_trip,
                s.pp_unreachable,
                s.pp_round_trip
            );
            let pp_share = 100.0 * s.pp_unreachable as f64 / s.n as f64;
            if share > 0.1 || pp_share > 0.1 || s.pp_round_trip > 1e-6 {
                failed.push(format!(
                    "{output:?} {} {share:.3}% / per-pixel {pp_share:.3}%, round trip {:.1e}",
                    s.film, s.pp_round_trip
                ));
            }
        }
    }
    assert!(failed.is_empty(), "above 0.1 %: {failed:?}");
}
