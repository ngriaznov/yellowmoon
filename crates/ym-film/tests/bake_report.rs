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
