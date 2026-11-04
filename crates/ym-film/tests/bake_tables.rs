//! The tables of F-04 against the direct f64 chain (`docs/24` section 3.8 and section 8).
mod bake_common;
use std::sync::Arc;

use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_color::lut::Lut3d;
use ym_film::bake::*;
use ym_film::scene::SceneLight;
use ym_film::{FilmParams, Output};

/// Stocks that cover the cases: a still negative, a cine negative, a slide.
const THREE: [&str; 3] = [
    "kodak_portra_400",
    "kodak_vision3_500t",
    "fujifilm_velvia_100",
];

fn smooth(d: [f64; 3]) -> [f64; 3] {
    [
        (d[0] * 1.3 + 0.2).sin() + d[1] * d[2],
        (-d[1]).exp() + 0.1 * d[0],
        d[0] * d[1] * d[2] + 0.5,
    ]
}

/// The CPU twin of the tetrahedral lookup is the look LUTs' own formula (`docs/02` C10): the same numbers as
/// `Lut3d::sample` on the same nodes, to rounding; nodes are returned exactly.
#[test]
fn tetrahedral_twin_matches_the_look_lut_formula() {
    let hi = [2.5, 2.1, 1.9];
    let t = Table3::bake(17, hi, 1.0, 2, smooth);
    let data: Vec<[f64; 3]> = t
        .data()
        .chunks_exact(4)
        .map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])])
        .collect();
    let lut = Lut3d::new(17, [0.0; 3], hi, data).expect("lut");
    let mut rng = Rng(5);
    let mut worst = 0.0_f64;
    for _ in 0..20_000 {
        // A little beyond the axes: both hold the edge.
        let p = [0, 1, 2].map(|c| (rng.unit() * 1.1 - 0.05) * hi[c]);
        let (a, b) = (t.sample(p), lut.sample(p));
        worst = worst.max((0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max));
    }
    eprintln!("tetrahedral twin against Lut3d: {worst:.2e}");
    assert!(worst < 1e-12, "{worst:e}");
}

#[test]
fn warped_axes_return_their_nodes_and_hold_the_edge() {
    let hi = [2.0, 1.5, 3.0];
    let t = Table3::bake(9, hi, 2.0, 3, smooth);
    for (i, j, k) in [(0, 0, 0), (3, 5, 8), (8, 8, 8), (1, 7, 2)] {
        let at = |n: usize, c: usize| hi[c] * (n as f64 / 8.0).powi(2);
        let d = [at(i, 0), at(j, 1), at(k, 2)];
        let (got, want) = (t.sample(d), smooth(d));
        for c in 0..3 {
            assert!((got[c] - want[c]).abs() < 1e-6, "node {i} {j} {k}");
        }
    }
    assert_eq!(t.sample([-1.0, -1.0, f64::NAN]), t.sample([0.0; 3]));
    assert_eq!(t.sample([9.0, 9.0, 9.0]), t.sample(hi));
}

/// T1 against the direct stage 2 on 100,000 random colours (the chromaticities they make): within 1e-3 relative.
#[test]
fn t1_matches_the_direct_exposure_stage() {
    let colours = random_colours(100_000, 99, 0.0, 1.0);
    for film in THREE {
        let c = chain(film, Output::DisplayP3);
        let t1 = baker(2).t1(&c);
        let ex = c.exposure_stage();
        let x_mid = ex.mid_log_exposure();
        let worst = par_map(&colours, 4, |p| {
            let l = c.scene_stage().run(*p);
            let direct = ex.raw(
                &SceneLight {
                    x: l.x,
                    y: l.y,
                    luminance: 0.18,
                },
                -x_mid,
            );
            let table = t1.sample(l.x, l.y);
            (0..3)
                .map(|k| (table[k] - direct[k]).abs() / direct[k].abs().max(1e-30))
                .fold(0.0, f64::max)
        })
        .into_iter()
        .fold(0.0, f64::max);
        eprintln!("{film}: T1 worst relative error {worst:.2e}");
        assert!(worst <= 1e-3, "{film}: {worst:e}");
    }
}

/// The 1D tables against the stages, and the baked coupler step against the direct solve.
#[test]
fn curves_and_couplers_match_the_direct_stages() {
    let mut rng = Rng(8);
    let pts: Vec<[f64; 3]> = (0..20_000)
        .map(|_| [0, 1, 2].map(|_| -5.0 + 10.0 * rng.unit()))
        .collect();
    for film in THREE {
        let c = chain(film, Output::DisplayP3);
        let f = baker(2).bake(&c, &BakeSettings::of(&c).draft()).unwrap();
        let dev = c.develop_stage();
        let (mut e_curve, mut e_paper, mut e_couple) = (0.0_f64, 0.0_f64, 0.0_f64);
        for p in &pts {
            let cv = &f.curves().film;
            let direct = dev.curves(*p);
            for k in 0..3 {
                e_curve = e_curve.max((cv[k].eval(p[k]) - direct[k]).abs());
            }
            let d = dev.couple(*p, c.params().coupler_amount);
            let b = f.film_density(*p);
            e_couple = e_couple.max((0..3).map(|k| (d[k] - b[k]).abs()).fold(0.0, f64::max));
            if let (Some(enl), Some(_)) = (c.enlarger_stage(), f.curves().paper.as_ref()) {
                let want = enl.paper_density(*p);
                let got = f.paper_density(*p);
                e_paper = e_paper.max((0..3).map(|k| (want[k] - got[k]).abs()).fold(0.0, f64::max));
            }
        }
        eprintln!(
            "{film}: curves {e_curve:.2e} paper {e_paper:.2e} couplers {e_couple:.2e} (density)"
        );
        assert!(e_curve < 2e-5 && e_paper < 2e-5, "{film}");
        assert!(e_couple < 1e-4, "{film}: couplers {e_couple:e}");
    }
}

fn all_tables(f: &BakedFilm) -> Vec<Vec<f32>> {
    let c = f.curves();
    let mut v = vec![f.t1().data().to_vec(), f.t3b().data().to_vec()];
    if let Some(t) = f.t3a() {
        v.push(t.data().to_vec());
    }
    for k in 0..3 {
        v.push(c.film[k].data().to_vec());
        v.push(c.film0[k].data().to_vec());
        if let Some(p) = &c.paper {
            v.push(p[k].data().to_vec());
        }
    }
    v
}

/// The same table bytes on 1 and 8 threads and on a repeat run (`docs/17` D5).
#[test]
fn bytes_do_not_depend_on_the_thread_count() {
    for film in ["kodak_portra_400", "fujifilm_velvia_100"] {
        let c = chain(film, Output::DisplayP3);
        let settings = BakeSettings {
            t3a_size: 19,
            t3b_size: 21,
            ..BakeSettings::of(&c)
        };
        let one = baker(1).bake(&c, &settings).unwrap();
        let eight = baker(8).bake(&c, &settings).unwrap();
        let again = baker(8).bake(&c, &settings).unwrap();
        let (a, b, r) = (all_tables(&one), all_tables(&eight), all_tables(&again));
        assert_eq!(a.len(), b.len());
        for (k, ((x, y), z)) in a.iter().zip(&b).zip(&r).enumerate() {
            assert!(x == y, "{film}: table {k} differs between 1 and 8 threads");
            assert!(y == z, "{film}: table {k} differs on a repeat run");
        }
        // The per-pixel return is a function of the pixel alone: the same bits from the three bakes.
        let mut rng = Rng(77);
        for _ in 0..300 {
            let p = [0, 1, 2].map(|_| 0.1 + 0.5 * rng.unit());
            let (x, y, z) = (one.run(p), eight.run(p), again.run(p));
            assert!(x == y && y == z, "{film}: run({p:?}) differs");
        }
        assert!(
            one.t3a().is_some() != (film == "fujifilm_velvia_100"),
            "a slide has no T3a"
        );
    }
}

/// The core table of the per-pixel return is shared by the whole process: the bakers, the outputs and the stocks
/// use one, and it is the same bytes on any thread count (`docs/24` section 3.8c).
#[test]
fn the_core_table_is_shared_and_independent_of_the_stock_and_the_output() {
    let c = chain("kodak_portra_400", Output::DisplayP3);
    let s = BakeSettings {
        t3a_size: 17,
        t3b_size: 9,
        ..BakeSettings::of(&c)
    };
    let a = baker(1).bake(&c, &s).unwrap();
    let b = baker(3)
        .bake(&chain("fujifilm_velvia_100", Output::Srgb), &s)
        .unwrap();
    assert!(std::ptr::eq(a.core_table(), b.core_table()));
    assert!(std::ptr::eq(
        a.core_table(),
        ym_color::display::cine::CoreTable::shared()
    ));
}

/// The cache keeps what it should, and a lights move re-bakes T3a alone.
#[test]
fn cache_hits_and_what_a_change_rebakes() {
    let c = chain("kodak_portra_400", Output::DisplayP3);
    let b = baker(2);
    let s = BakeSettings {
        t3a_size: 21,
        t3b_size: 21,
        ..BakeSettings::of(&c)
    };
    let first = b.bake(&c, &s).unwrap();
    let built = b.built();
    assert_eq!(
        built,
        BakeCounts {
            t1: 1,
            curves: 1,
            t3a: 1,
            t3b: 1
        }
    );
    let second = b.bake(&c, &s).unwrap();
    assert_eq!(b.built(), built, "the second bake builds nothing");
    assert!(Arc::ptr_eq(first.t1(), second.t1()) && Arc::ptr_eq(first.t3b(), second.t3b()));

    // The lights move: T3a only.
    let moved = BakeSettings {
        lights: Lights {
            yellow_cc: 7.0,
            magenta_cc: -4.0,
            cyan_cc: 2.0,
        },
        ..s
    };
    let third = b.bake(&c, &moved).unwrap();
    let after = b.built();
    assert_eq!((after.t1, after.curves, after.t3b), (1, 1, 1));
    assert_eq!(after.t3a, 2);
    assert!(third.t3a().unwrap().data() != first.t3a().unwrap().data());

    // A draft is made every time and not kept; the refined table is the cached one.
    let draft = b.bake(&c, &moved.draft()).unwrap();
    assert_eq!(draft.t3a().unwrap().size(), DRAFT_SIZE);
    let _ = b.bake(&c, &moved.draft()).unwrap();
    assert_eq!(b.built().t3a, 4, "two drafts, two builds");
    let refined = draft.with_t3a(b.t3a(&c, moved.lights, 21).unwrap());
    assert_eq!(b.built().t3a, 4, "the refined table was cached");
    assert_eq!(refined.t3a().unwrap().size(), 21);

    // Another output: T3b only.
    let p3 = chain("kodak_portra_400", Output::Srgb);
    let _ = b.bake(&p3, &s).unwrap();
    let after = b.built();
    assert_eq!((after.t1, after.curves, after.t3b), (1, 1, 2));
}

/// The stock picker is tiles: T1 keeps at least 20 entries (`docs/24` section 4.4).
#[test]
fn t1_keeps_twenty_stocks() {
    const { assert!(T1_CACHE_ENTRIES >= 20) };
    let names = ym_film::Names::load(&data().join("film/params")).unwrap();
    let films: Vec<_> = names
        .stocks
        .iter()
        .filter(|s| {
            !matches!(
                s.kind,
                ym_film::StockKind::Paper | ym_film::StockKind::PrintFilm
            )
        })
        .map(|s| s.id.clone())
        .collect();
    assert!(films.len() >= 20);
    let b = baker(4);
    let chains: Vec<_> = films.iter().map(|f| chain(f, Output::DisplayP3)).collect();
    for c in &chains {
        let _ = b.t1(c);
    }
    let built = b.built().t1;
    assert_eq!(built as usize, films.len());
    for c in &chains {
        let _ = b.t1(c);
    }
    assert_eq!(
        b.built().t1,
        built,
        "all {} T1 tables are still cached",
        films.len()
    );
}

/// The film exposure and the print exposure move without a re-bake, and the lights live in T3a: a baked film
/// with the exposures and lights set agrees with a chain built with them.
#[test]
fn exposures_and_lights_agree_with_a_rebuilt_chain() {
    let cine = CineRendering::new(Output::DisplayP3.target());
    let space = output_space(Output::DisplayP3);
    let colours = mild_colours(3_000, 4);
    let base = chain("kodak_portra_400", Output::DisplayP3);
    let b = baker(4);
    let mut film = b
        .bake(
            &base,
            &BakeSettings {
                t3a_size: 49,
                t3b_size: 65,
                ..BakeSettings::of(&base)
            },
        )
        .unwrap();
    let before = b.built();
    film.set_exposure_ev(0.5);
    film.set_print_stops(-0.3);
    assert_eq!(
        b.built(),
        before,
        "exposure and print exposure re-bake nothing"
    );
    let params = FilmParams {
        exposure_ev: 0.5,
        print_exposure_stops: -0.3,
        light_yellow_cc: 4.0,
        light_magenta_cc: -5.0,
        light_cyan_cc: 2.0,
        ..FilmParams::default()
    };
    let moved = ym_film::FilmChain::new(
        &data(),
        "kodak_portra_400",
        &ym_film::PaperChoice::Matching,
        params,
        spectra(),
    )
    .unwrap();
    let film = {
        let lit = b.t3a(&moved, Lights::of(&moved), 49).unwrap();
        film.with_t3a(lit)
    };
    let mut d = par_map(&colours, 4, |p| {
        code_diff(&cine, space, moved.run(*p), film.run(*p))
    });
    let s = CodeStats::of(&mut d);
    eprintln!("exposure 0.5 EV, print -0.3 stop, lights moved: {s}");
    assert!(s.p999 <= 0.5 && s.max <= 3.0, "{s}");
    // Without the lights the picture differs.
    let mut d = par_map(&colours, 4, |p| {
        code_diff(&cine, space, moved.run(*p), base.run(*p))
    });
    assert!(CodeStats::of(&mut d).max > 5.0);
}

/// A slide has no T3a: its T3b reads the film's dyes.
#[test]
fn a_slide_has_no_enlarger_table() {
    let c = chain("fujifilm_velvia_100", Output::DisplayP3);
    let f = baker(2).bake(&c, &BakeSettings::of(&c).draft()).unwrap();
    assert!(f.is_slide() && f.t3a().is_none() && f.curves().paper.is_none());
    let (_, taps) = f.run_tapped([0.4, 0.5, 0.3]);
    assert_eq!(taps.log_e_print, [0.0; 3]);
    assert_eq!(f.scan_input([0.4, 0.5, 0.3]), taps.density_film);
}

/// The front of the chain (T1, curves, couplers, T3a, paper) against the direct chain on the test photos, both
/// finished by the exact scan and return: the largest difference ≤ one 10-bit code, 99.9 % within a quarter.
#[test]
fn front_of_the_chain_meets_the_baking_bound() {
    let photos = load_photos(6);
    if photos.is_empty() {
        eprintln!("no photos in testdata/film-bake; skipping");
        return;
    }
    let cine = CineRendering::new(Output::DisplayP3.target());
    let space = output_space(Output::DisplayP3);
    let px: Vec<[f64; 3]> = photos
        .iter()
        .flat_map(|p| p.pixels.iter().copied())
        .collect();
    for film in THREE {
        let c = chain(film, Output::DisplayP3);
        let f = baker(4)
            .bake(
                &c,
                &BakeSettings {
                    t3b_size: 3,
                    ..BakeSettings::of(&c)
                },
            )
            .unwrap();
        let scan = c.scan_stage();
        let mut d = par_map(&px, 4, |p| {
            let exact_end = c.return_stage(scan.display(f.scan_input(*p)));
            code_diff(&cine, space, c.run(*p), exact_end)
        });
        let s = CodeStats::of(&mut d);
        eprintln!("{film}: front, T3a 49: {s}");
        assert!(s.max <= 1.0 && s.p999 <= 0.25, "{film}: {s}");
    }
}
