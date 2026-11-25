//! Times the EoMT small segmenter (`docs/34` section 4b.1, RG-03): the preprocessing, the network and the post-processing of one
//! 1024 x 683 picture (the analysis size: the Input's picture at 1024 px on the long edge), on 1, 2 and 4 threads and on all of them.
//!
//! ```text
//! cargo run --release -p ym-ml --example eomt_time -- <weights dir> [photo.jpg]
//! ```
//!
//! Without a photo the picture is a seeded synthetic one: the network's cost does not depend on the content, but the post-processing's
//! does (it works on the kept queries, and a synthetic picture keeps none), so give a real photo for the whole figure.

use std::time::Instant;

use ym_ml::eomt::{Eomt, input_tensor};

fn picture(width: usize, height: usize) -> Vec<u8> {
    let mut s = 7u64;
    (0..width * height * 3)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let (x, y) = ((i / 3) % width, (i / 3) / width);
            let v = 128.0
                + 70.0 * (x as f32 * 0.02).sin()
                + 40.0 * (y as f32 * 0.03).cos()
                + ((s >> 56) as f32 - 128.0) * 0.1;
            v.clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(args.get(1).map_or("data/models/eomt", String::as_str));
    let model = Eomt::load(dir).expect("load the weights");
    let (w, h, rgb) = match args.get(2) {
        Some(path) => {
            let bytes = std::fs::read(path).expect("read the photo");
            let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes));
            let rgb = d.decode().expect("decode the photo");
            let info = d.info().expect("jpeg header");
            (usize::from(info.width), usize::from(info.height), rgb)
        }
        None => (1024, 683, picture(1024, 683)),
    };
    println!("picture {w} x {h}");
    println!(
        "{} multiply-adds of the network; the machine has {} logical cores",
        ym_ml::eomt::multiply_adds(),
        std::thread::available_parallelism().map_or(1, usize::from)
    );
    let all = std::thread::available_parallelism().map_or(1, usize::from);
    let mut counts = vec![1, 2, 4, all];
    counts.dedup();
    for threads in counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            // One warm-up, then the median of three.
            let mut runs = vec![];
            for i in 0..4 {
                let t0 = Instant::now();
                let (input, (th, tw)) = input_tensor(&rgb, w, h);
                let t1 = Instant::now();
                let raw = model.infer(&input).expect("runs");
                let t2 = Instant::now();
                let labels = model.label_map(&raw, th, tw);
                let t3 = Instant::now();
                assert_eq!(labels.classes.len(), th * tw);
                if i > 0 {
                    runs.push((
                        (t1 - t0).as_secs_f64(),
                        (t2 - t1).as_secs_f64(),
                        (t3 - t2).as_secs_f64(),
                    ));
                }
            }
            runs.sort_by(|a, b| (a.0 + a.1 + a.2).total_cmp(&(b.0 + b.1 + b.2)));
            let (pre, net, post) = runs[1];
            println!(
                "{threads} thread(s): {:.2} s in all (input {:.0} ms, network {:.2} s, post-processing {:.0} ms)",
                pre + net + post,
                pre * 1e3,
                net,
                post * 1e3
            );
        });
    }
}
