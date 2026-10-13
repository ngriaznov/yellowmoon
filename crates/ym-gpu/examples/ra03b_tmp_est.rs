use ym_gpu::cpu::metrics::*;
use ym_gpu::cpu::RgbImage;
fn pattern(x: f64, y: f64) -> f64 {
    let g = (0.071 * x + 0.043 * y).sin() + (0.031 * x - 0.089 * y + 1.0).sin()
        + (0.113 * x + 0.017 * y + 2.0).sin()
        + 0.7 * (0.052 * (x + y) + 0.5).sin();
    0.5 + 0.35 * (g * 1.3).tanh()
}
fn main() {
    let (w, h) = (360usize, 240usize);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let hd = cx.hypot(cy);
    for amount in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0] {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let (dx, dy) = (px - cx, py - cy);
                let r = dx.hypot(dy);
                let u = if r > 0.0 { (dx / r, dy / r) } else { (0.0, 0.0) };
                // constant displacement
                img.plane_mut(0)[y * w + x] = pattern(px - amount * u.0, py - amount * u.1) as f32;
                img.plane_mut(1)[y * w + x] = pattern(px, py) as f32;
                img.plane_mut(2)[y * w + x] = pattern(px + amount * u.0, py + amount * u.1) as f32;
            }
        }
        let opts = CaResidualOptions { patch_px: 48, stride_px: 24, min_radius: 0.3, band_edges: vec![1.2], ..Default::default() };
        let res = lateral_ca_residual(&img, (cx, cy), hd, &opts);
        println!("amount {amount}: patches {} skipped {} red mean {:.3} rms {:.3}; blue mean {:.3} rms {:.3}", res.patches, res.skipped, res.bands[0].red_mean_px, res.bands[0].red_rms_px, res.bands[0].blue_mean_px, res.bands[0].blue_rms_px);
    }
}
