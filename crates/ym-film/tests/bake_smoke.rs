mod bake_common;
use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_film::Output;
use ym_film::bake::*;

#[test]
fn photos() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let photos = load_photos(3);
    eprintln!("{} photos, {} px", photos.len(), photos.iter().map(|p| p.pixels.len()).sum::<usize>());
    for film in ["kodak_portra_400", "kodak_ektar_100", "fujifilm_velvia_100", "kodak_vision3_500t"] {
        let c = chain(film, out);
        let b = baker(4);
        for size in [33usize, 49, 65] {
            let f = b.bake(&c, &BakeSettings { t3a_size: size, t3b_size: size, ..BakeSettings::of(&c) }).unwrap();
            let mut errs = Vec::new();
            for ph in &photos {
                for p in &ph.pixels {
                    let a = codes10(&cine, space, c.run(*p));
                    let q = codes10(&cine, space, f.run(*p));
                    errs.push((0..3).map(|k| (a[k] - q[k]).abs()).fold(0.0, f64::max));
                }
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = errs.len();
            let over = errs.iter().filter(|e| **e > 0.25).count();
            eprintln!("{film} {size}: n {n} max {:.2} p99.99 {:.3} p99.9 {:.3} med {:.4}; >0.25: {:.4}%", errs[n-1], errs[n*9999/10000], errs[n*999/1000], errs[n/2], 100.0*over as f64/n as f64);
        }
    }
}
