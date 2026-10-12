//! Writes `flat.ARW`, `vlines.ARW`, `hlines.ARW` and `tags.json` for the lens oracle (tools/lens/README.md).
//!
//! 3000 x 2000 files for an ILCE-7RM4 (rawspeed has a generic entry for it) carrying known lens
//! tables: vignetting (v = 8192 t^2, a corner 0.5 EV darker in darktable's reading), barrel
//! distortion up to 7% (d = 1200 t^2 / 2^14), lateral CA of +-0.14% (red outward, blue inward; the
//! optional second argument scales it). The raw data itself is undistorted: a flat field at 40% of
//! the range, and dark Gaussian lines every 100 px (vertical in vlines.ARW, horizontal in hlines.ARW).
use ym_raw::testgen::{ArwBuilder, LensTables};

fn knots(n: usize, f: impl Fn(f64) -> f64) -> Vec<i16> {
    let mut v = vec![n as i16];
    v.extend((0..n).map(|i| f(i as f64 / (n - 1) as f64).round() as i16));
    v
}

fn main() {
    let out = std::env::args().nth(1).expect("usage: lensgen OUT_DIR [CA_SCALE]");
    // Optional multiplier of the CA knots (default 1): a larger value lifts the lateral CA far above
    // the sub-pixel measurement noise of the oracle comparison.
    let ca_scale: f64 = std::env::args().nth(2).map_or(1.0, |a| a.parse().expect("CA_SCALE is a number"));
    std::fs::create_dir_all(&out).expect("create output directory");
    let (w, h) = (3000u32, 2000u32);
    let vign = knots(16, |t| 8192.0 * t * t);
    let dist = knots(16, |t| 1200.0 * t * t);
    // Sony's CA tag leads with the number of values that follow (2n: red then blue), not with n;
    // darktable rejects the data unless `2 * nc == first value` (verified against its source).
    let mut ca = vec![32i16];
    ca.extend(knots(16, |t| 3000.0 * ca_scale * t * t)[1..].iter());
    ca.extend(knots(16, |t| -3000.0 * ca_scale * t * t)[1..].iter());
    let tables = LensTables { vignetting: Some(vign.clone()), chromatic_aberration: Some(ca.clone()), distortion: Some(dist.clone()) };
    let b = ArwBuilder::new("ILCE-7RM4", w, h).lens_tables(tables);
    // Flat field at 40% of the range.
    let flat = vec![(512.0 + 0.4 * (16300.0 - 512.0)) as u16; (w * h) as usize];
    b.write_to(&std::path::Path::new(&out).join("flat.ARW"), &flat).unwrap();
    // Dark Gaussian lines every 100 px (a line centre at x = 50 + 100 k), undistorted in the raw
    // data: vertical lines only, and horizontal lines only. No crossings, so a line can be followed
    // along any row (or column) and the radial mapping measured in both directions.
    for (name, vertical) in [("vlines.ARW", true), ("hlines.ARW", false)] {
        let mut lines = vec![0u16; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let t = if vertical { x } else { y };
                let u = (t as f64 + 0.5 - 50.0).rem_euclid(100.0);
                let d = u.min(100.0 - u);
                let l = (-0.5 * (d / 1.6).powi(2)).exp();
                lines[y * w as usize + x] = (512.0 + 0.5 * (1.0 - 0.9 * l) * (16300.0 - 512.0)) as u16;
            }
        }
        b.write_to(&std::path::Path::new(&out).join(name), &lines).unwrap();
    }
    // The tags as JSON for compare_with_darktable.py (knots without the count).
    let json = format!(
        "{{\"vignetting\": {:?}, \"distortion\": {:?}, \"ca\": {:?}}}",
        &vign[1..], &dist[1..], &ca[1..]
    );
    std::fs::write(std::path::Path::new(&out).join("tags.json"), json).unwrap();
    println!("wrote flat.ARW, vlines.ARW, hlines.ARW and tags.json to {out}");
}
