//! Writes the synthetic oracle corpus: every raw variant on several bodies, with different image
//! content, into a folder (default `testdata/synthetic`, git-ignored). The oracle
//! (`tools/oracle`) then compares our decoder with rawler and LibRaw on every file.
//!
//! ```text
//! cargo run -p ym-raw --features testgen --release --example gen_corpus -- testdata/synthetic
//! cargo run -p oracle --release -- --corpus testdata/synthetic --report /tmp/oracle-report.json
//! ```
//!
//! File names are `variant__model__content__details.ARW`. A `manifest.json` next to the files
//! lists the parameters of each.

use std::path::Path;

use serde_json::json;

use ym_raw::layout::ChromaSubsampling;
use ym_raw::testgen::{
    Arw1, ArwBuilder, CrawEncoder, HuffmanChoice, LosslessL, LosslessMs, Packed12, SplitMix64,
    gradient_mosaic, gradient_rgb,
};

/// Image content of a synthetic mosaic.
#[derive(Clone, Copy)]
enum Content {
    Gradient,
    Noise,
    Clipped,
    Dark,
}

impl Content {
    fn name(self) -> &'static str {
        match self {
            Self::Gradient => "gradient",
            Self::Noise => "noise",
            Self::Clipped => "clipped",
            Self::Dark => "dark",
        }
    }
}

fn mosaic(w: u32, h: u32, black: u16, top: u16, content: Content, seed: u64) -> Vec<u16> {
    let mut rng = SplitMix64(seed);
    let span = u64::from(top - black).max(1);
    match content {
        Content::Gradient => gradient_mosaic(w, h, black, top, seed),
        Content::Noise => (0..w * h)
            .map(|_| (u64::from(black) + rng.below(span + 1)) as u16)
            .collect(),
        Content::Clipped => (0..w * h)
            .map(|_| {
                if rng.below(10) < 7 {
                    top
                } else {
                    (u64::from(black) + rng.below(span + 1)) as u16
                }
            })
            .collect(),
        Content::Dark => (0..w * h)
            .map(|_| (u64::from(black) + rng.below(span / 200 + 3)) as u16)
            .collect(),
    }
}

struct Writer<'a> {
    dir: &'a Path,
    /// One `{ file, details }` object per written file, in order.
    manifest: Vec<serde_json::Value>,
    count: usize,
}

impl Writer<'_> {
    fn write(&mut self, name: &str, details: &str, bytes: Vec<u8>) {
        let path = self.dir.join(name);
        std::fs::write(&path, bytes).expect("write a corpus file");
        self.manifest
            .push(json!({ "file": name, "details": details }));
        self.count += 1;
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "testdata/synthetic".to_owned());
    std::fs::create_dir_all(&dir).expect("create the output folder");
    let mut out = Writer {
        dir: Path::new(&dir),
        manifest: Vec::new(),
        count: 0,
    };
    let mut seed = 1u64;
    let mut next_seed = || {
        seed += 1;
        seed
    };

    // 14-bit uncompressed on several bodies.
    for (model, content) in [
        ("ILCE-7RM4", Content::Gradient),
        ("ILCE-7M3", Content::Noise),
        ("ILCE-9", Content::Clipped),
        ("ILCE-7SM3", Content::Dark),
        ("ILCE-1M2", Content::Gradient),
    ] {
        let (w, h) = (640u32, 480u32);
        let m = mosaic(w, h, 512, 16000, content, next_seed());
        let file = ArwBuilder::new(model, w, h).build(&m);
        out.write(
            &format!("uncompressed14__{model}__{}.ARW", content.name()),
            "compression 1, 14 bit",
            file,
        );
    }
    // 12-bit words (compression 1).
    for (model, content) in [
        ("ILCE-7M2", Content::Gradient),
        ("ILCE-6000", Content::Noise),
    ] {
        let (w, h) = (512u32, 384u32);
        let m = mosaic(w, h, 128, 4000, content, next_seed());
        let file = ArwBuilder::new(model, w, h).uncompressed(12).build(&m);
        out.write(
            &format!("uncompressed12__{model}__{}.ARW", content.name()),
            "compression 1, 12 bit words",
            file,
        );
    }
    // Packed 12-bit (ARW 2).
    for (model, content) in [
        ("DSLR-A700", Content::Gradient),
        ("DSLR-A850", Content::Noise),
        ("DSLR-A900", Content::Clipped),
        ("DSLR-A700", Content::Dark),
    ] {
        let (w, h) = (512u32, 384u32);
        let m = mosaic(w, h, 128, 4000, content, next_seed());
        let file = ArwBuilder::new(model, w, h).payload(Packed12).build(&m);
        out.write(
            &format!("packed12__{model}__{}.ARW", content.name()),
            "compression 32767, 12 bit packed",
            file,
        );
    }
    // ARW 1: Huffman-coded 12-bit samples (A200 to A390).
    for (model, content) in [
        ("DSLR-A200", Content::Gradient),
        ("DSLR-A300", Content::Noise),
        ("DSLR-A350", Content::Clipped),
        ("DSLR-A390", Content::Dark),
    ] {
        let (w, h) = (512u32, 384u32);
        let m = mosaic(w, h, 0, 4000, content, next_seed());
        let file = ArwBuilder::new(model, w, h)
            .payload(Arw1)
            .black([0; 4])
            .white([3840; 3])
            .build(&m);
        out.write(
            &format!("arw1__{model}__{}.ARW", content.name()),
            "compression 32767, 12 bit, Huffman-coded columns",
            file,
        );
    }
    // cRAW on several bodies and contents.
    let encoder = CrawEncoder::new();
    let top = encoder.max_value();
    for (model, content) in [
        ("ILCE-6000", Content::Gradient),
        ("ILCE-7M3", Content::Noise),
        ("ILCE-7RM3", Content::Clipped),
        ("ILCE-7M4", Content::Dark),
        ("NEX-7", Content::Gradient),
        ("DSLR-A500", Content::Noise),
        ("ILCE-9M3", Content::Clipped),
        ("ILCE-1", Content::Gradient),
    ] {
        let (w, h) = (640u32, 384u32);
        let m = mosaic(w, h, 512, top, content, next_seed());
        let file = ArwBuilder::new(model, w, h).craw(encoder.clone()).build(&m);
        out.write(
            &format!("craw__{model}__{}.ARW", content.name()),
            "compression 32767, one byte per pixel, real camera curve",
            file,
        );
    }
    // Lossless L: Sony's own shape (predictor 1, 512 tiles) on several bodies and sizes. Sizes
    // that are whole tiles can be compared with rawler too; the two with partial tiles at the
    // edges (1100 x 700 and 700 x 1100) only with LibRaw and libjpeg.
    for (model, content, (w, h), tag) in [
        ("ILCE-1", Content::Gradient, (1536u32, 1024u32), ""),
        ("ILCE-7M4", Content::Noise, (1024, 512), ""),
        ("ILCE-7RM5", Content::Clipped, (512, 512), ""),
        ("ILCE-9M3", Content::Dark, (2048, 1024), ""),
        ("ILCE-7CM2", Content::Gradient, (1536, 1536), ""),
        ("ILCE-9M3", Content::Noise, (1024, 1536), ""),
        ("ILCE-1", Content::Gradient, (1100, 700), "__partial"),
        ("ILCE-7M4", Content::Noise, (700, 1100), "__partial"),
    ] {
        let m = mosaic(w, h, 512, 16000, content, next_seed());
        let file = ArwBuilder::new(model, w, h)
            .payload(LosslessL::new())
            .build(&m);
        out.write(
            &format!("lossless_l__{model}__{}{tag}.ARW", content.name()),
            "compression 7, 512x512 tiles, predictor 1",
            file,
        );
    }
    // Lossless L: other predictors, restart markers, skewed tables, other tile sizes.
    for (predictor, restart, tile, huffman, tag) in [
        (2u8, 0usize, (512u32, 512u32), HuffmanChoice::Optimal, "p2"),
        (3, 0, (512, 512), HuffmanChoice::Optimal, "p3"),
        (4, 0, (512, 512), HuffmanChoice::Optimal, "p4"),
        (5, 0, (512, 512), HuffmanChoice::Optimal, "p5"),
        (6, 0, (512, 512), HuffmanChoice::Optimal, "p6"),
        (7, 0, (512, 512), HuffmanChoice::Optimal, "p7"),
        (1, 1, (512, 512), HuffmanChoice::Optimal, "p1_rst1"),
        (4, 2, (512, 512), HuffmanChoice::Optimal, "p4_rst2"),
        (1, 0, (512, 512), HuffmanChoice::Skewed, "p1_skewed"),
        (6, 0, (512, 512), HuffmanChoice::Skewed, "p6_skewed"),
        (1, 0, (256, 256), HuffmanChoice::Optimal, "p1_tile256"),
        (1, 0, (128, 64), HuffmanChoice::Optimal, "p1_tile128x64"),
    ] {
        let (w, h) = (1100u32, 700u32);
        let m = mosaic(w, h, 512, 16000, Content::Gradient, next_seed());
        let file = ArwBuilder::new("ILCE-1", w, h)
            .payload(
                LosslessL::new()
                    .predictor(predictor)
                    .restart_rows(restart)
                    .tile(tile.0, tile.1)
                    .huffman(huffman),
            )
            .build(&m);
        out.write(
            &format!("lossless_l__ILCE-1__gradient__{tag}.ARW"),
            "compression 7, non-default LJPEG parameters",
            file,
        );
    }
    // Lossless M and S.
    for (model, sub, (w, h), gains) in [
        (
            "ILCE-7RM5",
            ChromaSubsampling::Yuv420,
            (1100u32, 700u32),
            [2.05f32, 1.0, 1.62],
        ),
        (
            "ILCE-9M3",
            ChromaSubsampling::Yuv420,
            (1024, 512),
            [1.84, 1.0, 2.31],
        ),
        (
            "ILCE-1M2",
            ChromaSubsampling::Yuv422,
            (1024, 512),
            [2.3, 1.0, 1.4],
        ),
        (
            "DSC-RX1RM3",
            ChromaSubsampling::Yuv420,
            (700, 1100),
            [1.7, 1.0, 1.9],
        ),
        (
            "ILCE-1",
            ChromaSubsampling::Yuv444,
            (512, 512),
            [2.0, 1.0, 1.5],
        ),
    ] {
        let rgb = gradient_rgb(w, h, 1100, 9000, next_seed());
        let payload = LosslessMs::new(sub, rgb, w, h, gains);
        let file = ArwBuilder::new(model, w, h).lossless_ms(payload).build(&[]);
        out.write(
            &format!("lossless_ms__{model}__{}.ARW", sub.name().replace(':', "")),
            "compression 7, YCbCr, 3 components, white balanced",
            file,
        );
    }

    // Written with serde_json: a hand-built list ends in a comma and is not valid JSON.
    let manifest = serde_json::to_string_pretty(&out.manifest).expect("serialise the manifest");
    std::fs::write(Path::new(&dir).join("manifest.json"), manifest + "\n").expect("write manifest");
    println!("wrote {} files to {dir}", out.count);
}
