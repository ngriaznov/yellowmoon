//! The F-03 gates of `docs/24` section 8, against the real profiles and a stand-in spectra source.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ym_color::display::cine::CineRendering;
use ym_color::gamut::LinearAwg4;
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_film::cie::Cie;
use ym_film::exposure::reference_spectrum;
use ym_film::profile::verify_checksums;
use ym_film::{
    FilmChain, FilmError, FilmParams, Names, PaperChoice, Profile, ReferenceSpectra, Spectra, StockKind,
};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn names() -> Names {
    Names::load(&data().join("film/params")).expect("names.json")
}

fn spectra_for(film: &str) -> Arc<dyn Spectra> {
    let cie = Cie::load(&data().join("film/cie")).expect("cie");
    let p = Profile::load(&data().join("film/spektrafilm/profiles"), film).expect("profile");
    let ill = reference_spectrum(&p, &cie).expect("reference light");
    Arc::new(ReferenceSpectra::new(&cie, ill))
}

fn chain(film: &str, paper: PaperChoice, params: FilmParams) -> FilmChain {
    FilmChain::new(&data(), film, &paper, params, spectra_for(film)).unwrap_or_else(|e| panic!("{film}: {e}"))
}

fn films(kind: &[StockKind]) -> Vec<String> {
    names()
        .stocks
        .iter()
        .filter(|s| kind.contains(&s.kind))
        .map(|s| s.id.clone())
        .collect()
}

fn cameras() -> Vec<String> {
    films(&[StockKind::ColourNegative, StockKind::CineNegative, StockKind::Slide])
}

fn lerp_curve(p: &Profile, c: usize, x: f64) -> f64 {
    let (xs, ys) = p.curve(c);
    let pos = (x - xs[0]) / (xs[1] - xs[0]);
    let i = (pos.floor() as usize).min(xs.len() - 2);
    let t = (pos - i as f64).clamp(0.0, 1.0);
    ys[i] * (1.0 - t) + ys[i + 1] * t
}

#[test]
fn all_profiles_parse_and_match_their_checksums() {
    let dir = data().join("film/spektrafilm");
    let n = verify_checksums(&dir).expect("checksums");
    assert!(n >= 28, "{n} files checked");
    let mut ok = 0;
    for s in &names().stocks {
        Profile::load(&dir.join("profiles"), &s.id).unwrap_or_else(|e| panic!("{e}"));
        ok += 1;
    }
    assert_eq!(ok, 28);
    println!("GATE profile reader: {ok} of 28 parse and validate; {n} files match SHA256SUMS");
}

#[test]
fn damaged_profiles_give_a_plain_error() {
    let src = data().join("film/spektrafilm/profiles/kodak_portra_400.json");
    let text = std::fs::read_to_string(&src).expect("read");
    let tmp = tempfile::tempdir().expect("tmp");
    let write = |name: &str, body: &str| {
        std::fs::write(tmp.path().join(format!("{name}.json")), body).expect("write");
    };
    write("cut", &text[..text.len() / 2]);
    let e = Profile::load(tmp.path(), "cut").expect_err("truncated file");
    let msg = e.to_string();
    assert!(msg.contains("cut.json") && msg.contains("not valid JSON"), "{msg}");

    // A wrong wavelength grid: shift the first sample.
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("json");
    v["data"]["wavelengths"][0] = serde_json::json!(381.0);
    write("grid", &v.to_string());
    let msg = Profile::load(tmp.path(), "grid").expect_err("grid").to_string();
    assert!(msg.contains("grid.json") && msg.contains("wavelength sample 0"), "{msg}");

    // A wrong shape.
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("json");
    v["data"]["density_curves"].as_array_mut().expect("array").pop();
    write("shape", &v.to_string());
    let msg = Profile::load(tmp.path(), "shape").expect_err("shape").to_string();
    assert!(msg.contains("shape.json") && msg.contains("density_curves has 255 rows"), "{msg}");

    // A value out of range.
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("json");
    v["data"]["density_curves"][10][1] = serde_json::json!(40.0);
    write("range", &v.to_string());
    let msg = Profile::load(tmp.path(), "range").expect_err("range").to_string();
    assert!(msg.contains("range.json") && msg.contains("density"), "{msg}");

    // A missing file.
    let e = Profile::load(tmp.path(), "absent").expect_err("missing");
    assert!(matches!(e, FilmError::Io { .. }));
    assert!(e.to_string().contains("absent.json"));

    // A file that no longer matches its checksum.
    std::fs::create_dir_all(tmp.path().join("profiles")).expect("mkdir");
    std::fs::write(tmp.path().join("profiles/a.json"), "{}").expect("write");
    std::fs::write(
        tmp.path().join("SHA256SUMS"),
        format!("{}  profiles/a.json\n", "0".repeat(64)),
    )
    .expect("sums");
    let e = verify_checksums(tmp.path()).expect_err("checksum");
    assert!(matches!(e, FilmError::Checksum { .. }) && e.to_string().contains("profiles/a.json"));
}

#[test]
fn neutral_ramp_reproduces_each_cameras_published_curves() {
    let mut worst = (0.0_f64, String::new());
    for id in cameras() {
        let c = chain(&id, PaperChoice::Matching, FilmParams::default());
        let x_mid = c.exposure_stage().mid_log_exposure();
        for k in 0..70 {
            let x = -2.95 + 0.1 * f64::from(k) + 0.013;
            let g = 0.18 * 10f64.powf(x - x_mid);
            let light = c.scene_stage().run(logc4_encode_rgb([g; 3]));
            let log_e = c.exposure_stage().log_exposure(&light, 0.0);
            for ch in 0..3 {
                assert!((log_e[ch] - x).abs() < 1e-9, "{id}: a neutral must give equal exposures");
            }
            let d = c.develop_stage().couple(log_e, 1.0);
            for ch in 0..3 {
                let want = lerp_curve(c.film(), ch, x);
                let err = (d[ch] - want).abs();
                if err > worst.0 {
                    worst = (err, format!("{id} channel {ch} at logE {x:.2}"));
                }
            }
        }
    }
    println!("GATE published curves (stages 2 to 5, couplers on): worst error {:.5} density ({})", worst.0, worst.1);
    assert!(worst.0 < 0.01, "{worst:?}");
}

#[test]
fn papers_reproduce_their_published_curves() {
    let mut worst = (0.0_f64, String::new());
    let papers = films(&[StockKind::Paper]);
    assert_eq!(papers.len(), 6);
    for paper in &papers {
        let c = chain("kodak_portra_400", PaperChoice::Id(paper.clone()), FilmParams::default());
        let p = c.paper().expect("paper");
        for k in -7..=7 {
            let g = 0.18 * 2f64.powf(f64::from(k) * 0.8);
            let (_, t) = c.run_tapped(logc4_encode_rgb([g; 3]));
            for ch in 0..3 {
                let want = lerp_curve(p, ch, t.log_e_print[ch]);
                let err = (t.density_print[ch] - want).abs();
                if err > worst.0 {
                    worst = (err, format!("{paper} channel {ch}"));
                }
            }
        }
    }
    println!("GATE published curves (through stage 8): worst error {:.5} density ({})", worst.0, worst.1);
    assert!(worst.0 < 0.01, "{worst:?}");
}

#[test]
fn grey_prints_neutral_and_at_the_right_brightness_for_every_pair() {
    let mut worst_ab = (0.0_f64, String::new());
    let mut worst_slide = (0.0_f64, String::new());
    let mut worst_y = (0.0_f64, String::new());
    let mut lines = Vec::new();
    for id in cameras() {
        let c = chain(&id, PaperChoice::Matching, FilmParams::default());
        let grey = logc4_encode_rgb([0.18; 3]);
        let (out, t) = c.run_tapped(grey);
        let lab = c.scan_stage().lab_d50(t.display);
        let ab = lab[1].abs().max(lab[2].abs());
        let cine = CineRendering::new(FilmParams::default().output.target());
        let back = cine.forward(LinearAwg4(logc4_decode_rgb(out)));
        let target = cine.forward(LinearAwg4([0.18; 3]))[0];
        assert!((target - 0.111).abs() < 5e-4, "the cine rendering's grey is {target}");
        // Negatives print neutral, so every channel is the grey; a slide is matched in luminance (its own grey is
        // not exactly neutral: a* and b* are reported, not gated).
        let dy = if c.paper().is_some() {
            back.iter().map(|v| (v - 0.111).abs()).fold(0.0, f64::max)
        } else {
            (c.scan_stage().luminance(back) - 0.111).abs()
        };
        if c.paper().is_some() && ab > worst_ab.0 {
            worst_ab = (ab, id.clone());
        }
        if c.paper().is_none() && ab > worst_slide.0 {
            worst_slide = (ab, id.clone());
        }
        if dy > worst_y.0 {
            worst_y = (dy, id.clone());
        }
        lines.push(format!(
            "  {id:<26} on {:<32} filters Y {:7.2} M {:7.2}  shift {:6.3}  a* {:8.5} b* {:8.5}  display {:.5}",
            c.paper().map_or("(slide)", |p| p.id.as_str()),
            c.neutral_filters_cc()[0],
            c.neutral_filters_cc()[1],
            c.print_shift(),
            lab[1],
            lab[2],
            back[0],
        ));
        if c.paper().is_some() {
            assert!(ab < 1.0, "{id}: a*/b* {ab}");
        }
        assert!(dy < 0.002, "{id}: display {back:?}");
    }
    for l in &lines {
        println!("{l}");
    }
    println!("GATE neutral: worst |a*|,|b*| {:.6} ({}) over the negatives", worst_ab.0, worst_ab.1);
    println!("(not a gate) slides' own grey: worst |a*|,|b*| {:.3} ({})", worst_slide.0, worst_slide.1);
    println!("GATE brightness: worst |display - 0.111| {:.6} ({})", worst_y.0, worst_y.1);
}

#[test]
fn the_cine_rendering_undoes_the_return_stage() {
    // Stage 10 then the forward cine rendering gives stage 9's display colour back, for reachable colours.
    let mut worst = 0.0_f64;
    let (mut total, mut reachable) = (0, 0);
    for id in ["kodak_portra_400", "kodak_ektar_100", "fujifilm_velvia_100", "kodak_vision3_500t"] {
        let c = chain(id, PaperChoice::Matching, FilmParams::default());
        let cine = CineRendering::new(c.params().output.target());
        for r in 0..5 {
            for g in 0..5 {
                for b in 0..5 {
                    let lin = [0.02 * 3f64.powi(r), 0.02 * 3f64.powi(g), 0.02 * 3f64.powi(b)];
                    let logc = logc4_encode_rgb(lin);
                    let display = c.display_of(logc);
                    let held = display.map(|v| v.clamp(1e-6, 1.0));
                    total += 1;
                    if cine.inverse(held).chroma_fraction < 1.0 - 1e-9 {
                        continue;
                    }
                    reachable += 1;
                    let back = cine.forward(LinearAwg4(logc4_decode_rgb(c.return_stage(display))));
                    for k in 0..3 {
                        worst = worst.max((back[k] - held[k]).abs());
                    }
                }
            }
        }
    }
    println!("GATE return: worst round-trip error {worst:.2e} on {reachable} reachable of {total} colours");
    assert!(reachable > total / 2);
    assert!(worst <= 1e-6, "{worst}");
}

#[test]
fn results_do_not_depend_on_the_thread_count_or_the_run() {
    let c = chain("kodak_portra_400", PaperChoice::Matching, FilmParams::default());
    let pixels: Vec<[f64; 3]> = (0..1200)
        .map(|i| {
            let f = |k: u32| 0.01 * 1.0035f64.powi(((i * 7 + k as i32 * 131) % 1500) as i32);
            logc4_encode_rgb([f(1), f(2), f(3)])
        })
        .collect();
    let run = |threads: usize| -> Vec<u64> {
        let chunk = pixels.len().div_ceil(threads);
        let mut out = vec![[0.0; 3]; pixels.len()];
        std::thread::scope(|s| {
            for (src, dst) in pixels.chunks(chunk).zip(out.chunks_mut(chunk)) {
                let c = &c;
                s.spawn(move || {
                    for (p, o) in src.iter().zip(dst.iter_mut()) {
                        *o = c.run(*p);
                    }
                });
            }
        });
        out.iter().flatten().map(|v| v.to_bits()).collect()
    };
    let one = run(1);
    assert_eq!(one, run(8), "1 and 8 threads differ");
    assert_eq!(one, run(1), "two runs differ");
    println!("GATE determinism: {} values identical on 1 and 8 threads and on a repeat", one.len());
}

#[test]
fn couplers_off_changes_saturation_but_not_the_grey() {
    let off = FilmParams {
        coupler_amount: 0.0,
        ..FilmParams::default()
    };
    let c = chain("kodak_portra_400", PaperChoice::Matching, off);
    let g = c.display_of(logc4_encode_rgb([0.18; 3]));
    let target = c.target_grey();
    assert!((c.scan_stage().luminance(g) - target).abs() < 1e-5);
    let on = chain("kodak_portra_400", PaperChoice::Matching, FilmParams::default());
    let red = logc4_encode_rgb([0.4, 0.05, 0.03]);
    assert_ne!(c.run(red), on.run(red));
}

#[test]
fn wrong_stocks_are_refused() {
    let r = FilmChain::new(
        &data(),
        "kodak_portra_endura",
        &PaperChoice::Matching,
        FilmParams::default(),
        spectra_for("kodak_portra_400"),
    );
    assert!(matches!(r, Err(FilmError::WrongKind { .. })));
    let r = FilmChain::new(
        &data(),
        "no_such_film",
        &PaperChoice::Matching,
        FilmParams::default(),
        spectra_for("kodak_portra_400"),
    );
    assert!(matches!(r, Err(FilmError::UnknownStock(_))));
}
