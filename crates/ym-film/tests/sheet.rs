//! The sanity sheet (not a gate): the 24 ColorChecker patches through four stocks, into a PNG.
//!
//! The patches are the Ohta ColorChecker reflectances in `testdata/spectral/` (git-ignored; the test skips when
//! they are absent), lit by D50, adapted to D65 and put in AWG4. The left chart is the cine rendering alone;
//! then Portra 400, Ektar 100, Velvia 100 and Vision3 500T. Output: sRGB. The path is `$YM_F03B_SHEET`, or
//! `target/f03b/patches.png`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ym_color::consts::linalg;
use ym_color::consts::{D65_XY, xy_to_xyz};
use ym_color::display::OutputTarget;
use ym_color::display::cine::CineRendering;
use ym_color::gamut::{LinearAwg4, matrices};
use ym_color::logc::logc4_encode_rgb;
use ym_film::cie::{Cie, Light};
use ym_film::exposure::reference_spectrum;
use ym_film::scene::bradford;
use ym_film::{FilmChain, FilmParams, Output, PaperChoice, Profile, ReferenceSpectra};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn srgb_oetf(v: f64) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (e * 255.0 + 0.5).floor() as u8
}

fn patches(cie: &Cie) -> Option<Vec<[f64; 3]>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/spectral/reflectances_colorchecker_24.csv");
    let text = std::fs::read_to_string(path).ok()?;
    let d50 = cie.light(Light::D50);
    let norm: f64 = (0..81).map(|i| d50[i] * cie.cmf[i][1]).sum();
    let adapt = bradford(cie.xyz_of_light(&d50, true), xy_to_xyz(D65_XY));
    let m = matrices();
    let mut out = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let r: Vec<f64> = line.split(',').skip(1).map(|v| v.trim().parse().unwrap_or(0.0)).collect();
        if r.len() != 81 {
            return None;
        }
        let mut xyz = [0.0; 3];
        for i in 0..81 {
            for c in 0..3 {
                xyz[c] += r[i] * d50[i] * cie.cmf[i][c] / norm;
            }
        }
        let xyz65 = linalg::mul_vec(&adapt, xyz);
        out.push(linalg::mul_vec(&m.xyz_to_awg4, xyz65));
    }
    (out.len() == 24).then_some(out)
}

#[test]
fn colorchecker_through_four_stocks() {
    let cie = Cie::load(&data().join("film/cie")).expect("cie");
    let Some(chart) = patches(&cie) else {
        eprintln!("skipped: testdata/spectral/reflectances_colorchecker_24.csv is absent");
        return;
    };
    let cine = CineRendering::new(OutputTarget::Srgb);
    let params = FilmParams {
        output: Output::Srgb,
        ..FilmParams::default()
    };
    let stocks = ["kodak_portra_400", "kodak_ektar_100", "fujifilm_velvia_100", "kodak_vision3_500t"];
    let mut charts: Vec<Vec<[f64; 3]>> = vec![chart.iter().map(|p| cine.forward(LinearAwg4(*p))).collect()];
    for id in stocks {
        let prof = Profile::load(&data().join("film/spektrafilm/profiles"), id).expect("profile");
        let ill = reference_spectrum(&prof, &cie).expect("light");
        let c = FilmChain::new(
            &data(),
            id,
            &PaperChoice::Matching,
            params.clone(),
            Arc::new(ReferenceSpectra::new(&cie, ill)),
        )
        .expect("chain");
        let mut rows = Vec::new();
        for p in &chart {
            let out = c.run(logc4_encode_rgb(*p));
            rows.push(cine.forward(LinearAwg4(ym_color::logc::logc4_decode_rgb(out))));
        }
        charts.push(rows);
    }
    let (cell, gap, margin) = (56usize, 4usize, 16usize);
    let chart_w = 6 * cell + 5 * gap;
    let chart_h = 4 * cell + 3 * gap;
    let width = margin + charts.len() * (chart_w + margin);
    let height = chart_h + 2 * margin;
    let mut img = vec![40u8; width * height * 3];
    for (ci, rows) in charts.iter().enumerate() {
        let x0 = margin + ci * (chart_w + margin);
        for (pi, rgb) in rows.iter().enumerate() {
            let (col, row) = (pi % 6, pi / 6);
            let (px, py) = (x0 + col * (cell + gap), margin + row * (cell + gap));
            for y in py..py + cell {
                for x in px..px + cell {
                    let o = (y * width + x) * 3;
                    for k in 0..3 {
                        img[o + k] = srgb_oetf(rgb[k]);
                    }
                }
            }
        }
    }
    let out = std::env::var_os("YM_F03B_SHEET").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/f03b/patches.png"),
        PathBuf::from,
    );
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).expect("mkdir");
    }
    let file = std::fs::File::create(&out).expect("create");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().expect("header");
    w.write_image_data(&img).expect("data");
    drop(w);
    println!("sheet written to {}", out.display());
    for (id, rows) in std::iter::once(&"cine rendering only").chain(stocks.iter()).zip(&charts) {
        let g = &rows[21];
        println!("{id:<22} patch 22 (neutral 5) display {:.4} {:.4} {:.4}", g[0], g[1], g[2]);
    }
}
