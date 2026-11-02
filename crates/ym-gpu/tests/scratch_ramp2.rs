mod common;
use ym_color::mask::{PhotoMap};
use ym_gpu::cpu::sampler::{GridPositions, SamplerGrid};

// A hard edge with a ramp shape: `linear` true is the exact coverage of a pixel box.
fn shape(t: f64, linear: bool) -> f64 { let t = t.clamp(0.0, 1.0); if linear { t } else { t*t*(3.0-2.0*t) } }
fn mask(kind: u8, p: [f64; 2], k: f64, bw: f64, bh: f64, width_px: f64, linear: bool, angle: f64, c: [f64;2], r: [f64;2]) -> f64 {
    let l = bw.max(bh);
    let (x, y) = ((p[0]-c[0])*bw/l, (p[1]-c[1])*bh/l);
    let (s, co) = angle.to_radians().sin_cos();
    let ramp = width_px * k / l;
    if kind == 0 {
        let sd = -s*x + co*y;
        1.0 - shape((sd + 0.5*ramp)/ramp, linear)
    } else {
        let u = (co*x + s*y)/r[0]; let w = (-s*x + co*y)/r[1];
        let d = (u*u+w*w).sqrt();
        let q = if std::env::var("LOCAL").is_ok() {
            let g = ((u/r[0]).powi(2) + (w/r[1]).powi(2)).sqrt() / d.max(1e-9);
            ramp * g.max(1.0 / r[0].max(r[1]))
        } else { ramp / r[0].min(r[1]) };
        1.0 - shape((d - (1.0 - 0.5*q))/q, linear)
    }
}
fn plane(kind: u8, k: f64, width_px: f64, linear: bool, angle: f64, c: [f64;2], r: [f64;2]) -> (Vec<f64>, usize, usize) {
    let (bw, bh) = (1200usize, 800usize);
    let (w, h) = ((bw as f64 / k) as usize, (bh as f64 / k) as usize);
    let grid = SamplerGrid::new([[k, 0.0, 0.0], [0.0, k, 0.0]], (bw as f64, bh as f64), (bw as f64, bh as f64), 1, 0, true);
    let map = GridPositions::new(&grid, (0, 0));
    let mut out = Vec::new();
    for y in 0..h { for x in 0..w { out.push(mask(kind, map.position(x, y), k, bw as f64, bh as f64, width_px, linear, angle, c, r)); }}
    (out, w, h)
}
fn diff(kind: u8, width_px: f64, linear: bool, angle: f64, c: [f64;2], r: [f64;2]) -> f64 {
    let (full, fw, _) = plane(kind, 1.0, width_px, linear, angle, c, r);
    let (small, sw, sh) = plane(kind, 4.0, width_px, linear, angle, c, r);
    let mut largest = 0.0f64;
    for y in 0..sh { for x in 0..sw {
        let mut acc = 0.0;
        for j in 0..4 { for i in 0..4 { acc += full[(4*y+j)*fw + 4*x+i]; } }
        largest = largest.max((acc/16.0 - small[y*sw+x]).abs());
    }}
    largest
}
#[test]
fn scan() {
    for linear in [false, true] {
        for width in [0.75, 1.0, 1.25, 1.5] {
            let mut wl = 0.0f64;
            for angle in [0.0, 5.0, 10.0, 20.0, 30.0, 35.0, 45.0, 60.0, 80.0, 90.0] {
                for c in [0.4137, 0.5, 0.5333] { wl = wl.max(diff(0, width, linear, angle, [c, 0.5], [0.0, 0.0])); }
            }
            let mut wr = 0.0f64;
            for (c, r, a) in [([0.45,0.55],[0.25,0.15],-40.0), ([0.5,0.5],[0.3,0.2],0.0), ([0.37,0.41],[0.1,0.07],25.0), ([0.6,0.5],[0.04,0.03],70.0), ([0.5,0.5],[0.35,0.3],12.0)] {
                wr = wr.max(diff(1, width, linear, a, c, r));
            }
            eprintln!("{} ramp {width} px: linear-edge worst {wl:.3}, radial worst {wr:.3}", if linear {"LINEAR-SHAPE"} else {"smoothstep "});
        }
    }
}
