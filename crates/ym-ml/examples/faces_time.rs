//! Times the BlazeFace face finder (`docs/27` S-08): one 192 x 192 network pass, then the finder on a photo and on a 24 MP
//! group picture made by pasting the given photos at several sizes into a 6000 x 4000 background.
//!
//! ```text
//! cargo run --release -p ym-ml --example faces_time -- <weights dir> <photo.jpg>...
//! ```

use std::time::Instant;

use ym_ml::mediapipe::blazeface::{
    BlazeFace, FaceFinder, FinderParams, Letterbox, PixelSource, Resample, Rgb8, Window,
    prepare_tensor,
};

fn decode(path: &str) -> (usize, usize, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read the photo");
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes));
    let rgb = d.decode().expect("decode the photo");
    let info = d.info().expect("jpeg header");
    (usize::from(info.width), usize::from(info.height), rgb)
}

/// Nearest-neighbour paste of `src` scaled to `size` pixels wide at (`x`, `y`).
fn paste(
    canvas: &mut [u8],
    cw: usize,
    src: &(usize, usize, Vec<u8>),
    x: usize,
    y: usize,
    size: usize,
) {
    let (w, h, data) = src;
    let nh = size * h / w;
    for dy in 0..nh {
        for dx in 0..size {
            let (sx, sy) = (dx * w / size, dy * h / nh);
            let s = &data[(sy * w + sx) * 3..][..3];
            canvas[((y + dy) * cw + x + dx) * 3..][..3].copy_from_slice(s);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let net = BlazeFace::load(dir).expect("load the weights");
    let photos: Vec<_> = args[2..].iter().map(|p| decode(p)).collect();

    // One network pass.
    let (w, h, data) = &photos[0];
    let img = Rgb8::new(*w, *h, data).expect("picture");
    let lb = Letterbox::new(Window::whole(*w, *h)).expect("window");
    let input = prepare_tensor(&img, &lb, Resample::Box);
    for _ in 0..2 {
        net.forward(&input).expect("forward");
    }
    let n = 10;
    let t = Instant::now();
    for _ in 0..n {
        net.forward(&input).expect("forward");
    }
    println!(
        "one 192 x 192 pass: {:.1} ms (mean of {n}, {} threads)",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64,
        rayon::current_num_threads()
    );
    let t = Instant::now();
    let _ = prepare_tensor(&img, &lb, Resample::Box);
    println!(
        "letterbox of {w} x {h} (box): {:.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    let finder = FaceFinder::new(net);
    let params = FinderParams::default();
    let t = Instant::now();
    let faces = finder.find(&img, &params).expect("find");
    println!(
        "finder on {w} x {h}: {:.0} ms, {} face(s)",
        t.elapsed().as_secs_f64() * 1000.0,
        faces.len()
    );

    // A 24 MP group picture.
    let (cw, ch) = (6000usize, 4000usize);
    let mut canvas = vec![0u8; cw * ch * 3];
    for y in 0..ch {
        for x in 0..cw {
            let v = (90 + (x * 7 + y * 3) % 60) as u8;
            canvas[(y * cw + x) * 3..][..3].copy_from_slice(&[v, v - 10, v - 20]);
        }
    }
    let placements = [
        (300, 300, 700),
        (2500, 400, 420),
        (4300, 2200, 300),
        (1200, 2600, 220),
        (3300, 3000, 160),
        (5200, 300, 130),
    ];
    for (i, &(x, y, size)) in placements.iter().enumerate() {
        paste(&mut canvas, cw, &photos[i % photos.len()], x, y, size);
    }
    let big = Rgb8::new(cw, ch, &canvas).expect("canvas");
    let windows = ym_ml::mediapipe::blazeface::plan_windows(cw, ch, 0.015 * cw as f64);
    let t = Instant::now();
    let faces = finder.find(&big, &params).expect("find");
    println!(
        "finder on {cw} x {ch} ({} windows): {:.0} ms, {} face(s)",
        windows.len(),
        t.elapsed().as_secs_f64() * 1000.0,
        faces.len()
    );
    for f in &faces {
        println!("  {:?} score {:.3} scale {:.4}", f.rect, f.score, f.scale);
    }
    let _ = big.width();
}
