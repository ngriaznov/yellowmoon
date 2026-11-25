//! The cost of the element soft masks (`docs/34` section 4a.3, RG-04): the guided filter at the sizes it is made at, and the guide.
//!
//! `cargo run --release -p ym-color --example element_time` prints, for a synthetic scene and guide, the time of
//! [`ym_color::mask_element::soft_masks_of_picture`] at the analysis sizes (512 and 1024 px on the long edge, one and seven live
//! elements), and of the tiled render at 24 MP (6000 x 4000, tiles of 1024 px grown by the filter's halo, one and seven live elements),
//! with the guide of every pixel (`ColorChain::scene_guide_pixel`) and the working set of one tile. The filter's parallelism is
//! `std::thread::scope` over rows, as the render's; run it under `taskset -c 0` for one core.

use std::sync::Arc;
use std::time::Instant;

use ym_color::consts::linalg::IDENTITY;
use ym_color::display::{ChainParams, ColorChain};
use ym_color::mask::PhotoFrame;
use ym_color::mask_element::{
    ElementJob, ElementRow, ElementScene, element_halo, element_planes, soft_masks_of_picture,
};
use ym_color::mask_raster::Area;
use ym_color::tone::ToneParams;

fn scene(w: usize, h: usize) -> ElementScene {
    let bits = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64 / w as f64, (i / w) as f64 / h as f64);
            let mut b = if y < 0.33 {
                1
            } else if x < 0.5 {
                2
            } else {
                8
            };
            if (x - 0.5).hypot(y - 0.6) < 0.2 {
                b |= 64 | 32;
            }
            if (x - 0.8).hypot(y - 0.8) < 0.1 {
                b |= 4 | 16;
            }
            b
        })
        .collect();
    ElementScene {
        classes: Vec::new(),
        key: 1,
        width: w,
        height: h,
        bits,
    }
}

fn guide(w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            (0.2 + 0.5 * x + 0.2 * (20.0 * y).sin().abs() * (x > 0.5) as u8 as f32).clamp(0.0, 1.0)
        })
        .collect()
}

fn main() {
    for (w, h) in [(512usize, 341usize), (1024, 683)] {
        let s = scene(w, h);
        let g = guide(w, h);
        for live in [1usize, 7] {
            let wanted: [bool; 7] = std::array::from_fn(|e| e < live);
            let t = Instant::now();
            let planes = soft_masks_of_picture(&s, &wanted, &g, w, h);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let made = planes.iter().filter(|p| p.is_some()).count();
            println!("analysis {w} x {h}: {live} live elements ({made} planes): {ms:.1} ms");
        }
    }
    // 24 MP in tiles.
    let (w, h) = (6000usize, 4000usize);
    let s = Arc::new(scene(1024, 683));
    let chain =
        ColorChain::new(&ChainParams::new(IDENTITY, ToneParams::IDENTITY)).expect("a chain");
    let photo = PhotoFrame::new(w as f64, h as f64, 1.0);
    let frame = Area::whole(w, h);
    let r = s.radius(1.0 / photo.pixel);
    let halo = element_halo(r);
    for live in [1usize, 7] {
        let rows = [ElementRow {
            node: 1,
            live: std::array::from_fn(|e| e < live),
            scene: Arc::clone(&s),
        }];
        let tile = 1024usize;
        let (mut guide_ms, mut planes_ms) = (0.0, 0.0);
        let mut worst_px = 0usize;
        let t0 = Instant::now();
        for ty in (0..h).step_by(tile) {
            for tx in (0..w).step_by(tile) {
                let window = Area {
                    x: tx as i64,
                    y: ty as i64,
                    w: tile.min(w - tx),
                    h: tile.min(h - ty),
                };
                let grown = window.grown(halo, &frame);
                worst_px = worst_px.max(grown.w * grown.h);
                let t = Instant::now();
                // The guide of every pixel, across the cores like the render's (`rayon` there, scoped threads here).
                let n = grown.w * grown.h;
                let threads = std::thread::available_parallelism().map_or(1, usize::from);
                let per = n.div_ceil(threads);
                let mut lightness = vec![0.0f32; n];
                std::thread::scope(|scope| {
                    for (c, chunk) in lightness.chunks_mut(per).enumerate() {
                        let chain = &chain;
                        scope.spawn(move || {
                            for (j, d) in chunk.iter_mut().enumerate() {
                                let x = (c * per + j) % grown.w;
                                let v = ((grown.x as usize + x) % 251) as f64 / 251.0;
                                *d = chain.scene_guide_pixel([v, 0.5 * v, 0.3]);
                            }
                        });
                    }
                });
                guide_ms += t.elapsed().as_secs_f64() * 1000.0;
                let off = (grown.x as usize, grown.y as usize);
                let position = move |x: usize, y: usize| {
                    Some([
                        ((x + off.0) as f64 + 0.5) / w as f64,
                        ((y + off.1) as f64 + 0.5) / h as f64,
                    ])
                };
                let t = Instant::now();
                let _ = element_planes(
                    &rows,
                    &ElementJob {
                        photo,
                        grown,
                        frame,
                        window,
                        position: &position,
                        guide: &lightness,
                    },
                );
                planes_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
        }
        println!(
            "24 MP in tiles of {tile} (radius {r}, halo {halo}): {live} live elements: planes {:.2} s, guide {:.2} s, total {:.2} s; largest grown window {} px, about {:.0} MB of f64 and f32 working planes",
            planes_ms / 1000.0,
            guide_ms / 1000.0,
            t0.elapsed().as_secs_f64(),
            worst_px,
            worst_px as f64 * (8.0 * 6.0 + 4.0 * 6.0) / 1e6,
        );
    }
}
