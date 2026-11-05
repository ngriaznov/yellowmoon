//! F-08d: how far the Film node's colour passes are from the CPU's in relative terms, stage by stage, as numbers a change
//! can be measured by (`docs/24` section 3.9c). The export tail turns a relative error of 1e-6 in a near-black sample into
//! hundreds of codes after output sharpening, so P1's exposure, P3's density and P5's exact display colour (the shadow) are
//! each held to a standard deviation of their relative error.
//!
//! Measured on Mesa lavapipe, before F-08d and after (Portra 400 on Rec.2020 and Display P3, 8,500 photo pixels, relative
//! standard deviation; "chain" is P1, P3 and P5 on the device against the CPU's chain with its f32 stores between passes):
//! P1 exposure 4.5e-7 and 1.9e-7, P3 density 2.4e-7 and 1.05e-7 (on the exposure the device made), P5 alone on the CPU's
//! own densities 4.5e-7 and 2.6e-8, the chain 6.0e-7 and 1.65e-7.
#![allow(clippy::needless_range_loop)]

mod common;
mod film_common;

use common::{TestGpu, gpus};
use film_common::*;
use ym_color::display::OutputTarget;
use ym_color::display::cine::CineRendering;
use ym_color::logc::logc4_encode_rgb;
use ym_film::Output;
use ym_film::bake::BakedFilm;
use ym_film::math::pow10;
use ym_film::spatial::SpatialSettings;
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::film::*;
use ym_gpu::queue::{JobSpec, Priority};
use ym_gpu::stream::PassList;

fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter().filter(|d| d.mode == StorageMode::F32)
}

/// The pixels the bounds are measured on: the photos of `testdata/film-bake` when they are there (they are git-ignored, so
/// not in CI), else a seeded spread of near-neutral LogC4 values. The fallback must stay inside the gamut compression's knee
/// like the photos: P1's compression of a chromaticity near the locus (`ci_compress`) is plain f32 and is not what these
/// bounds measure (`docs/24` section 3.9c, "What the bounds measure"). `YM_FILM_PRECISION_NO_PHOTOS=1` takes the fallback with the photos present (what CI runs). `YM_FILM_PRECISION_SATURATED=1` takes the old
/// fallback, every channel uniform in 0.2 to 0.8, half of which is compressed: P1 is then 1.17e-6 (max 5.3e-5), as CI showed
/// on 2026-11-05 when it had no photos.
fn pixels() -> Vec<[f64; 3]> {
    let saturated = std::env::var_os("YM_FILM_PRECISION_SATURATED").is_some();
    let no_photos = saturated || std::env::var_os("YM_FILM_PRECISION_NO_PHOTOS").is_some();
    let mut px: Vec<[f64; 3]> = if no_photos {
        Vec::new()
    } else {
        photo_pixels(40).into_iter().flat_map(|(_, p)| p).collect()
    };
    if px.is_empty() {
        let mut rng = Rng(0xF08D);
        let cine = CineRendering::new(OutputTarget::Srgb);
        px = (0..8000)
            .map(|_| {
                if saturated {
                    [
                        0.2 + 0.6 * rng.unit(),
                        0.2 + 0.6 * rng.unit(),
                        0.2 + 0.6 * rng.unit(),
                    ]
                } else {
                    // A display colour of an ordinary picture (sRGB, within 0.1 of a grey), as the photos' pixels are.
                    let grey = 0.08 + 0.84 * rng.unit();
                    let e = [0, 1, 2].map(|_| (grey + 0.2 * (rng.unit() - 0.5)).clamp(0.0, 1.0));
                    let lin = e.map(ym_color::encode::srgb_decode);
                    logc4_encode_rgb(cine.inverse(lin).scene.0)
                }
            })
            .collect();
    }
    px
}

/// Whether the scene stage compresses the chromaticity of the LogC4 pixel `px` (it lies beyond the knee, 0.85 of the way from
/// the white to the locus), computed on the CPU in f64 as `ym_film::scene` does.
fn is_compressed(b: &BakedFilm, px: [f64; 3]) -> bool {
    let parts = b.device_parts();
    let rgb = ym_color::logc::logc4_decode_rgb(px);
    let m = parts.scene_to_xyz;
    let xyz = [0, 1, 2].map(|i| m[i][0] * rgb[0] + m[i][1] * rgb[1] + m[i][2] * rgb[2]);
    let sum = xyz[0] + xyz[1] + xyz[2];
    if !(sum > 0.0 && sum.is_finite()) {
        return false;
    }
    let w = parts.scene_white_xy;
    let d = [xyz[0] / sum - w[0], xyz[1] / sum - w[1]];
    let r = d[0].hypot(d[1]);
    if r < 1e-12 {
        return false;
    }
    let u = [d[0] / r, d[1] / r];
    let loc = parts.scene_locus;
    let mut best = f64::INFINITY;
    for i in 0..loc.len() {
        let (a, c) = (loc[i], loc[(i + 1) % loc.len()]);
        let e = [c[0] - a[0], c[1] - a[1]];
        let den = u[0] * e[1] - u[1] * e[0];
        if den.abs() < 1e-18 {
            continue;
        }
        let aw = [a[0] - w[0], a[1] - w[1]];
        let t = (aw[0] * e[1] - aw[1] * e[0]) / den;
        let s = (aw[0] * u[1] - aw[1] * u[0]) / den;
        if t > 0.0 && (-1e-12..=1.0 + 1e-12).contains(&s) && t < best {
            best = t;
        }
    }
    best.is_finite() && r > ym_film::scene::KNEE * best
}

fn side(w: usize, n: usize) -> (usize, usize) {
    (w, n.div_ceil(w))
}

/// (relative standard deviation, maximum) of `got` against `want`, over the entries whose `want` is at least `floor`.
fn rel(got: &[f64], want: &[f64], floor: f64) -> (f64, f64) {
    let r: Vec<f64> = got
        .iter()
        .zip(want)
        .filter(|(_, w)| **w >= floor)
        .map(|(g, w)| (g - w) / w)
        .collect();
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (var.sqrt(), r.iter().fold(0.0f64, |m, v| m.max(v.abs())))
}

struct Run {
    e: Vec<f32>,
    d: Vec<f32>,
    out: Vec<f32>,
    shadow: Vec<f32>,
}

/// P1, P3 and P5 on the device with the shadow, each read back.
fn run(dev: &TestGpu, data: &FilmColourData, input: &[f32], w: usize, h: usize) -> Run {
    let (data, input) = (data.clone(), input.to_vec());
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let film = FilmDevice::upload(ctx, &data)?;
            let rect = FRect {
                x0: 0,
                y0: 0,
                x1: w as i32,
                y1: h as i32,
            };
            let src = FilmImage::upload(ctx, "film precision input", rect, &input)?;
            let e = FilmImage::alloc(ctx, "film precision E", rect)?;
            let d = FilmImage::alloc(ctx, "film precision D", rect)?;
            let out = FilmImage::alloc_with_shadow(ctx, "film precision out", rect)?;
            let mut passes = PassList::new(ctx, "film precision");
            encode_colour_in(&mut passes, &film, (w, h), &src, &e, rect)?;
            encode_develop(&mut passes, &film, DevelopMode::Density, &e, None, &d, rect)?;
            encode_colour_out(&mut passes, &film, &d, &out, rect)?;
            let shadow = out.shadow.as_ref().expect("a shadow");
            let reads = [
                e.img.encode_read(&mut passes)?,
                d.img.encode_read(&mut passes)?,
                out.img.encode_read(&mut passes)?,
                shadow.encode_read(&mut passes)?,
            ];
            let done = passes.submit()?.finish()?;
            Ok(Run {
                e: e.img.take(&done, reads[0])?,
                d: d.img.take(&done, reads[1])?,
                out: out.img.take(&done, reads[2])?,
                shadow: shadow.take(&done, reads[3])?,
            })
        })
        .expect("film precision")
}

/// P5 alone on the CPU's own (f32) densities: the exact display colour (the shadow).
fn run_p5(dev: &TestGpu, data: &FilmColourData, dens: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (data, dens) = (data.clone(), dens.to_vec());
    dev.gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let film = FilmDevice::upload(ctx, &data)?;
            let rect = FRect {
                x0: 0,
                y0: 0,
                x1: w as i32,
                y1: h as i32,
            };
            let d = FilmImage::upload(ctx, "film precision D", rect, &dens)?;
            let out = FilmImage::alloc_with_shadow(ctx, "film precision out", rect)?;
            let mut passes = PassList::new(ctx, "film precision p5");
            encode_colour_out(&mut passes, &film, &d, &out, rect)?;
            let shadow = out.shadow.as_ref().expect("a shadow");
            let read = shadow.encode_read(&mut passes)?;
            let done = passes.submit()?.finish()?;
            shadow.take(&done, read)
        })
        .expect("film precision p5")
}

/// The relative standard deviations of P1's exposure, P3's density, the chain's display colour and P5 alone.
fn report(id: &str, o: Output, b: &BakedFilm, px: &[[f64; 3]], dev: &TestGpu) -> [f64; 4] {
    let (w, h) = side(64, px.len());
    let input = planar(px, w, h);
    let got = run(dev, &colour_data(b), &input, w, h);
    let n = px.len();
    let (mut e_got, mut e_want) = (Vec::new(), Vec::new());
    let (mut d_got, mut d_want) = (Vec::new(), Vec::new());
    let (mut s_got, mut s_want) = (Vec::new(), Vec::new());
    for i in 0..n {
        let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + i]));
        let le = b.exposure_of(q);
        // The CPU's chain as ym-film runs it: f64 inside a pass, an f32 store between passes (E, then the density).
        let e32 = le.map(|v| pow10(v) as f32);
        let dens = b.film_density(e32.map(|v| f64::from(v).max(1e-30).log10()));
        let d32 = dens.map(|v| v as f32);
        let want_display =
            b.scan_display(b.paper_density(b.print_log_exposure(d32.map(f64::from))));
        for c in 0..3 {
            e_got.push(f64::from(got.e[c * w * h + i]));
            e_want.push(pow10(le[c]));
            d_got.push(f64::from(got.d[c * w * h + i]));
            d_want.push(f64::from(d32[c]));
            s_got.push(f64::from(got.shadow[c * w * h + i]));
            s_want.push(want_display[c].clamp(0.0, 1.0));
        }
    }
    // P5 alone, fed the CPU's own f32 densities.
    let mut d_in = vec![0.0_f32; 3 * w * h];
    let mut p5_want = Vec::new();
    for i in 0..w * h {
        let j = i.min(n - 1);
        let q = [0, 1, 2].map(|c| f64::from(input[c * w * h + j]));
        let le = b.exposure_of(q);
        let e32 = le.map(|v| pow10(v) as f32);
        let dens = b.film_density(e32.map(|v| f64::from(v).max(1e-30).log10()));
        for c in 0..3 {
            d_in[c * w * h + i] = dens[c] as f32;
        }
        if i < n {
            let disp = b.scan_display(
                b.paper_density(b.print_log_exposure(dens.map(|v| f64::from(v as f32)))),
            );
            p5_want.extend(disp.map(|v| v.clamp(0.0, 1.0)));
        }
    }
    let p5 = run_p5(dev, &colour_data(b), &d_in, w, h);
    let p5_got: Vec<f64> = (0..n)
        .flat_map(|i| (0..3).map(move |c| (i, c)))
        .map(|(i, c)| f64::from(p5[c * w * h + i]))
        .collect();
    let (p5s, p5m) = rel(&p5_got, &p5_want, 1e-4);
    println!("    P5 alone on the CPU's densities: display {p5s:.2e} (max {p5m:.1e})");
    let (e, em) = rel(&e_got, &e_want, 1e-6);
    // P1 apart for the pixels the scene stage compresses (plain f32) and for the rest.
    let comp: Vec<bool> = px.iter().map(|p| is_compressed(b, *p)).collect();
    let pick = |want_compressed: bool| -> (Vec<f64>, Vec<f64>) {
        (0..e_got.len())
            .filter(|k| comp[k / 3] == want_compressed)
            .map(|k| (e_got[k], e_want[k]))
            .unzip()
    };
    let (cg, cw) = pick(true);
    let (ug, uw) = pick(false);
    let share = cg.len() as f64 / e_got.len() as f64;
    if !cg.is_empty() {
        let (cs, cm) = rel(&cg, &cw, 1e-6);
        let (us, um) = rel(&ug, &uw, 1e-6);
        println!(
            "    P1 E: {:.1} % of the pixels compressed: those {cs:.2e} (max {cm:.1e}), the rest {us:.2e} (max {um:.1e})",
            100.0 * share
        );
    }
    assert!(
        share <= 0.02 || std::env::var_os("YM_FILM_PRECISION_SATURATED").is_some(),
        "{id} {o:?}: {:.1} % of the pixels are gamut-compressed; the bounds below are for in-gamut photos",
        100.0 * share
    );
    let (d, dm) = rel(&d_got, &d_want, 0.05);
    let (s, sm) = rel(&s_got, &s_want, 1e-4);
    println!(
        "{id} {o:?}: P1 E {e:.2e} (max {em:.1e}), P3 density {d:.2e} (max {dm:.1e}), P5 display {s:.2e} (max {sm:.1e}), {n} pixels"
    );
    let _ = (&got.out,);
    [e, d, s, p5s]
}

#[test]
fn the_film_passes_against_the_cpu_in_relative_terms() {
    let px = pixels();
    for dev in f32_devices() {
        for (id, o, chain_bound) in [
            ("kodak_portra_400", Output::Rec2020, 2.2e-7),
            ("kodak_portra_400", Output::DisplayP3, 2.4e-7),
            ("kodak_portra_400", Output::Srgb, 3.4e-7),
            ("kodak_ektar_100", Output::DisplayP3, 2.6e-7),
            // Ektar's saturated colours leave the sRGB gamut: the return's pre-image is ill-conditioned there.
            ("kodak_ektar_100", Output::Srgb, 6.5e-7),
        ] {
            let b = film(id, o, SpatialSettings::default()).baked().clone();
            let [e, d, s, p5] = report(id, o, &b, &px, dev);
            // Regression bounds: the measured values with a margin of a fifth.
            assert!(e <= 2.4e-7, "{id} {o:?}: P1 {e:.3e}");
            assert!(d <= 1.4e-7, "{id} {o:?}: P3 {d:.3e}");
            assert!(p5 <= 1.5e-7, "{id} {o:?}: P5 alone {p5:.3e}");
            assert!(
                s <= chain_bound,
                "{id} {o:?}: chain {s:.3e} (bound {chain_bound:e})"
            );
        }
    }
}

#[test]
fn p1_on_a_grey_ramp_and_on_saturated_colours() {
    let ramp: Vec<[f64; 3]> = (0..4000)
        .map(|i| {
            let e = 0.12 + 0.8 * f64::from(i) / 4000.0;
            [e, e, e]
        })
        .collect();
    let mut rng = Rng(0x51);
    let colours: Vec<[f64; 3]> = (0..4000)
        .map(|_| {
            [
                0.2 + 0.5 * rng.unit(),
                0.2 + 0.5 * rng.unit(),
                0.2 + 0.5 * rng.unit(),
            ]
        })
        .collect();
    for dev in f32_devices() {
        let b = film(
            "kodak_portra_400",
            Output::Rec2020,
            SpatialSettings::default(),
        )
        .baked()
        .clone();
        for (name, px) in [("grey ramp", &ramp), ("random colours", &colours)] {
            let (w, h) = side(64, px.len());
            let input = planar(px, w, h);
            let got = run(dev, &colour_data(&b), &input, w, h);
            let (mut g, mut c) = (Vec::new(), Vec::new());
            for i in 0..px.len() {
                let q = [0, 1, 2].map(|k| f64::from(input[k * w * h + i]));
                let le = b.exposure_of(q);
                for k in 0..3 {
                    g.push(f64::from(got.e[k * w * h + i]));
                    c.push(pow10(le[k]));
                }
            }
            let (s, m) = rel(&g, &c, 1e-6);
            println!("P1 E on {name}: {s:.2e} (max {m:.1e})");
            if name == "grey ramp" {
                assert!(s <= 1.4e-7, "{name}: {s:.3e}");
            }
        }
    }
}
