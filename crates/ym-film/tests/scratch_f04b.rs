mod bake_common;
use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_film::Output;
use ym_film::bake::*;

#[test]
#[ignore]
fn sweep() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let photos = load_photos(6);
    let photo_px: Vec<[f64; 3]> = photos
        .iter()
        .flat_map(|p| p.pixels.iter().copied())
        .collect();
    let random = random_colours(100_000, 2024, 0.0, 1.0);
    let sizes: Vec<usize> = std::env::var("SIZES")
        .unwrap_or("33,49,65".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let stocks: Vec<String> = std::env::var("STOCKS")
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or(vec!["kodak_portra_400".into()]);
    for film in &stocks {
        let c = chain(film, out);
        let b = baker(4);
        for &sz in &sizes {
            let t0 = std::time::Instant::now();
            let f = b
                .bake(
                    &c,
                    &BakeSettings {
                        t3a_size: sz.min(49),
                        t3b_size: sz,
                        ..BakeSettings::of(&c)
                    },
                )
                .unwrap();
            let el = t0.elapsed();
            for (name, set) in [("photos", &photo_px), ("random", &random)] {
                let mut d = par_map(set, 4, |p| code_diff(&cine, space, c.run(*p), f.run(*p)));
                println!(
                    "{film:22} T3b {sz:3} ({el:.1?}) {name:6}: {}",
                    CodeStats::of(&mut d)
                );
            }
        }
    }
}

#[test]
#[ignore]
fn breakdown() {
    use ym_color::logc::logc4_decode_rgb;
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let random = random_colours(100_000, 2024, 0.0, 1.0);
    let size: usize = std::env::var("SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65);
    for film in ["kodak_portra_400"] {
        let c = chain(film, out);
        let b = baker(4);
        let f = b
            .bake(
                &c,
                &BakeSettings {
                    t3b_size: size,
                    ..BakeSettings::of(&c)
                },
            )
            .unwrap();
        let scan = c.scan_stage();
        struct Row {
            front: f64,
            t3b: f64,
            whole: f64,
            cat: usize,
            photo: bool,
            edge: bool,
            dens: [f64; 3],
            disp: [f64; 3],
        }
        let rows: Vec<Row> = par_map(&random, 4, |p| {
            let (_, tpb) = f.run_tapped(*p);
            let dpb = tpb.density_print;
            let exact_from_baked_front = c.return_stage(scan.display(dpb));
            let front = code_diff(&cine, space, c.run(*p), exact_from_baked_front).0;
            let t3b = code_diff(&cine, space, exact_from_baked_front, f.t3b().sample(dpb)).0;
            let whole = code_diff(&cine, space, c.run(*p), f.run(*p)).0;
            let disp = scan.display(dpb);
            let inv = cine.inverse_continuous(disp);
            let cat = if inv.chroma_fraction < 1.0 {
                0
            } else if disp.iter().any(|&x| x < 1e-6) {
                1
            } else {
                2
            };
            let lin = logc4_decode_rgb(*p);
            let y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
            let photo = lin.iter().all(|&x| x >= 0.0) && y > 0.18 / 512.0 && y < 0.18 * 32.0;
            let hi = f.t3b().axis_max();
            let edge = (0..3).any(|k| dpb[k] > hi[k] * 0.98);
            Row {
                front,
                t3b,
                whole,
                cat,
                photo,
                edge,
                dens: dpb,
                disp,
            }
        });
        let stat = |name: &str, sel: &dyn Fn(&Row) -> bool| {
            for (lab, g) in [
                ("front", &(|r: &Row| r.front) as &dyn Fn(&Row) -> f64),
                ("T3b", &|r: &Row| r.t3b),
                ("whole", &|r: &Row| r.whole),
            ] {
                let mut v: Vec<f64> = rows.iter().filter(|r| sel(r)).map(|r| g(r)).collect();
                if v.is_empty() {
                    continue;
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let n = v.len();
                println!(
                    "{film} {name:26} {lab:6} n {n:6} max {:8.2} p99.9 {:7.3} >1/4 {:6.2}%",
                    v[n - 1],
                    v[(n * 999 / 1000).min(n - 1)],
                    100.0 * v.iter().filter(|x| **x > 0.25).count() as f64 / n as f64
                );
            }
        };
        stat("all", &|_| true);
        stat("photographic", &|r| r.photo);
        stat("not photographic", &|r| !r.photo);
        stat("unreachable", &|r| r.cat == 0);
        stat("clamped channel", &|r| r.cat == 1);
        stat("reachable other", &|r| r.cat == 2);
        stat("axis edge (>98% of max)", &|r| r.edge);
        stat("photo & reachable other", &|r| r.photo && r.cat == 2);
        stat("photo, reachable, no edge", &|r| {
            r.photo && r.cat == 2 && !r.edge
        });
        let share = |sel: &dyn Fn(&Row) -> bool| {
            100.0 * rows.iter().filter(|r| sel(r)).count() as f64 / rows.len() as f64
        };
        println!(
            "share: photographic {:.2}%, unreachable {:.2}%, clamped {:.2}%, edge {:.2}%",
            share(&|r| r.photo),
            share(&|r| r.cat == 0),
            share(&|r| r.cat == 1),
            share(&|r| r.edge)
        );
        let mut w: Vec<&Row> = rows.iter().filter(|r| r.photo && r.cat == 2).collect();
        w.sort_by(|a, b| b.whole.total_cmp(&a.whole));
        for r in w.iter().take(8) {
            let y = 0.2126 * r.disp[0] + 0.7152 * r.disp[1] + 0.0722 * r.disp[2];
            println!(
                "  err {:.2} (front {:.2} t3b {:.2}) dens {:.3?} disp {:.4?} rel {:.3?}",
                r.whole,
                r.front,
                r.t3b,
                r.dens,
                r.disp,
                r.disp.map(|x| x / y)
            );
        }
    }
}

#[test]
#[ignore]
fn cell() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let c = chain("kodak_portra_400", out);
    let f = baker(4).bake(&c, &BakeSettings::of(&c)).unwrap();
    let scan = c.scan_stage();
    let hi = f.t3b().axis_max();
    let n = 65usize;
    for dp in [[1.025, 0.066, 0.121], [1.372, 0.178, 0.296]] {
        let mut idx = [0usize; 3];
        let mut fr = [0.0; 3];
        for k in 0..3 {
            let u = (dp[k] / hi[k]).sqrt() * (n - 1) as f64;
            idx[k] = u.floor() as usize;
            fr[k] = u - idx[k] as f64;
        }
        println!("dp {:?} idx {:?} fr {:.3?}", dp, idx, fr);
        let at = |i: usize, k: usize| {
            let u = i as f64 / (n - 1) as f64;
            hi[k] * u * u
        };
        for dr in 0..2 {
            for dg in 0..2 {
                for db in 0..2 {
                    let d = [at(idx[0] + dr, 0), at(idx[1] + dg, 1), at(idx[2] + db, 2)];
                    let disp = scan.display(d);
                    let ic = cine.inverse_coordinates(disp);
                    println!(
                        "  corner {dr}{dg}{db} d {:.3?} disp {:.4?} q {:.4?} fr {:.3}",
                        d, disp, ic.coordinates, ic.chroma_fraction
                    );
                }
            }
        }
        let disp = scan.display(dp);
        let ic = cine.inverse_coordinates(disp);
        println!(
            "  true q {:.4?} fr {}  table q {:.4?}",
            ic.coordinates,
            ic.chroma_fraction,
            f.t3b().sample_stored(dp)
        );
    }
}

#[test]
#[ignore]
fn lattice() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let n: usize = 65;
    for film in ["kodak_portra_400", "fujifilm_velvia_100"] {
        let c = chain(film, out);
        let f = baker(4).bake(&c, &BakeSettings::of(&c)).unwrap();
        let hi = f.t3b().axis_max();
        let scan = c.scan_stage();
        let at = |i: usize, k: usize| { let u = i as f64 / (n - 1) as f64; hi[k] * u * u };
        let nodes: Vec<[f64;3]> = (0..n*n*n).map(|i| [at(i%n,0), at((i/n)%n,1), at(i/(n*n),2)]).collect();
        for (label, which) in [("old inverse (table+fallbacks)", 0), ("continuous", 1)] {
            let q: Vec<[f64;4]> = par_map(&nodes, 4, |d| {
                let disp = scan.display(*d);
                let inv = if which == 0 { cine.inverse(disp) } else { cine.inverse_continuous(disp) };
                let l = ym_color::logc::logc4_encode_rgb(inv.scene.0);
                [l[0], l[1], l[2], inv.chroma_fraction]
            });
            let idx = |r: usize,g: usize,b: usize| r + n*(g + n*b);
            let mut hist = [0usize; 7];
            let mut big = 0usize;
            for b_ in 1..n-1 { for g in 1..n-1 { for r in 1..n-1 {
                let c0 = q[idx(r,g,b_)];
                for axis in 0..3 {
                    let (m, p) = match axis { 0 => (q[idx(r-1,g,b_)], q[idx(r+1,g,b_)]), 1 => (q[idx(r,g-1,b_)], q[idx(r,g+1,b_)]), _ => (q[idx(r,g,b_-1)], q[idx(r,g,b_+1)]) };
                    if m[3] < 1.0 || p[3] < 1.0 || c0[3] < 1.0 { continue; }
                    let dd = (0..3).map(|k| (m[k] - 2.0*c0[k] + p[k]).abs()).fold(0.0, f64::max);
                    let bkt = if dd < 1e-3 {0} else if dd < 1e-2 {1} else if dd < 3e-2 {2} else if dd<1e-1 {3} else if dd < 3e-1 {4} else if dd < 1.0 {5} else {6};
                    hist[bkt] += 1; if dd > 0.3 { big += 1; }
                }
            }}}
            let unr = q.iter().filter(|v| v[3] < 1.0).count();
            println!("{film} {label}: second-difference hist (<1e-3,<1e-2,<3e-2,<.1,<.3,<1,>=1) {:?} (>0.3: {big}) unreachable nodes {unr}", hist);
        }
    }
}

