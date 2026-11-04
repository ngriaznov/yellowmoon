//! Shared helpers of the baking tests (F-04).
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ym_color::display::cine::CineRendering;
use ym_color::encode::OutputSpace;
use ym_color::gamut::LinearAwg4;
use ym_color::logc::logc4_decode_rgb;
use ym_film::bake::Baker;
use ym_film::{FilmChain, FilmParams, Output, PaperChoice, Spectra, TableSpectra};

/// The eight most used stocks of the picker (`docs/24` section 8, variant A).
pub const EIGHT: [&str; 8] = [
    "kodak_portra_400",
    "kodak_portra_160",
    "kodak_ektar_100",
    "kodak_gold_200",
    "kodak_ultramax_400",
    "fujifilm_pro_400h",
    "kodak_vision3_500t",
    "fujifilm_velvia_100",
];

pub fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

pub fn spectra() -> Arc<dyn Spectra> {
    Arc::new(TableSpectra::load(&data().join("film/upsampling")).expect("upsampling table"))
}

pub fn params(output: Output) -> FilmParams {
    FilmParams {
        output,
        ..FilmParams::default()
    }
}

pub fn chain(film: &str, output: Output) -> FilmChain {
    FilmChain::new(
        &data(),
        film,
        &PaperChoice::Matching,
        params(output),
        spectra(),
    )
    .unwrap_or_else(|e| panic!("{film}: {e}"))
}

pub fn baker(threads: usize) -> Baker {
    Baker::new(&data(), "upsampling-256", threads).expect("baker")
}

/// A seeded generator (splitmix64): the same numbers on every platform.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

pub fn output_space(o: Output) -> OutputSpace {
    match o {
        Output::Srgb => OutputSpace::Srgb,
        Output::DisplayP3 => OutputSpace::DisplayP3,
    }
}

/// The unrounded 10-bit code values the Output makes of a LogC4 / AWG4 colour: the cine rendering, then C13's
/// transfer function, scaled to 0 ..= 1023.
pub fn codes10(cine: &CineRendering, space: OutputSpace, logc4: [f64; 3]) -> [f64; 3] {
    let display = cine.forward(LinearAwg4(logc4_decode_rgb(logc4)));
    space.encode_clamped(display).map(|v| v * 1023.0)
}

/// The folder of the LogC4 test photos: 16-bit sRGB PNGs that `ym render --look neutral --output-sharpening 0`
/// made from the public ARW files (see `docs/24` section 3.8, *As built*).
pub fn photo_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/film-bake")
}

/// A photo as LogC4 / AWG4 pixels, every `stride`-th pixel in each direction.
pub struct Photo {
    pub name: String,
    pub pixels: Vec<[f64; 3]>,
}

/// Reads every PNG of [`photo_dir`] (sorted by name), decodes sRGB to display-linear and takes it back to the scene
/// through the inverse of the cine rendering, which is the LogC4 the Output read (the look was Neutral).
pub fn load_photos(stride: usize) -> Vec<Photo> {
    use ym_color::display::OutputTarget;
    use ym_color::logc::logc4_encode_rgb;
    let cine = CineRendering::new(OutputTarget::Srgb);
    let Ok(rd) = std::fs::read_dir(photo_dir()) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let dec = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(&path).expect("open photo"),
        ));
        let mut reader = dec.read_info().expect("png header");
        let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
        let info = reader.next_frame(&mut buf).expect("png frame");
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen, "16-bit PNG");
        let ch = info.color_type.samples();
        let (w, h) = (info.width as usize, info.height as usize);
        let mut pixels = Vec::new();
        for y in (0..h).step_by(stride) {
            for x in (0..w).step_by(stride) {
                let o = (y * w + x) * ch * 2;
                let mut e = [0.0; 3];
                for c in 0..3 {
                    e[c] = f64::from(u16::from_be_bytes([buf[o + 2 * c], buf[o + 2 * c + 1]]))
                        / 65535.0;
                }
                let lin = e.map(ym_color::encode::srgb_decode);
                pixels.push(logc4_encode_rgb(cine.inverse(lin).scene.0));
            }
        }
        let name = path
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().replace("_neutral", ""));
        out.push(Photo { name, pixels });
    }
    out
}

/// What the 10-bit comparison of two chains found.
#[derive(Debug, Clone, Copy)]
pub struct CodeStats {
    pub n: usize,
    /// The largest difference in unrounded code values (any channel).
    pub max: f64,
    /// The difference that 99.9 % of the pixels stay within.
    pub p999: f64,
    /// The share of pixels that differ by more than a quarter code.
    pub over_quarter: f64,
    /// The largest difference of the rounded 10-bit codes.
    pub max_quantised: u32,
}

impl CodeStats {
    pub fn of(diffs: &mut [(f64, u32)]) -> Self {
        diffs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n = diffs.len();
        let over = diffs.iter().filter(|d| d.0 > 0.25).count();
        Self {
            n,
            max: diffs.last().map_or(0.0, |d| d.0),
            p999: diffs[(n * 999 / 1000).min(n - 1)].0,
            over_quarter: over as f64 / n as f64,
            max_quantised: diffs.iter().map(|d| d.1).max().unwrap_or(0),
        }
    }
}

impl std::fmt::Display for CodeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n {:>7} max {:>7.3} p99.9 {:>6.3} >1/4: {:>7.4}% max rounded {}",
            self.n,
            self.max,
            self.p999,
            100.0 * self.over_quarter,
            self.max_quantised
        )
    }
}

/// Runs `f` over `items` on `threads` threads (contiguous runs) and returns the results in order.
pub fn par_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let per = items.len().div_ceil(threads.max(1)).max(1);
    let mut out: Vec<Vec<R>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(per)
            .map(|chunk| {
                let f = &f;
                s.spawn(move || chunk.iter().map(f).collect::<Vec<R>>())
            })
            .collect();
        for h in handles {
            out.push(h.join().expect("worker"));
        }
    });
    out.into_iter().flatten().collect()
}

/// The 10-bit code difference between two LogC4 / AWG4 colours after the Output's cine rendering and C13.
pub fn code_diff(cine: &CineRendering, space: OutputSpace, a: [f64; 3], b: [f64; 3]) -> (f64, u32) {
    let (ca, cb) = (codes10(cine, space, a), codes10(cine, space, b));
    let cont = (0..3).map(|k| (ca[k] - cb[k]).abs()).fold(0.0, f64::max);
    let q = (0..3)
        .map(|k| (ca[k].round() - cb[k].round()).abs() as u32)
        .max()
        .unwrap_or(0);
    (cont, q)
}

/// 100,000 seeded random LogC4 / AWG4 colours, uniform in the cube `[lo, hi]^3`.
pub fn random_colours(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| [0, 1, 2].map(|_| lo + (hi - lo) * rng.unit()))
        .collect()
}

/// Seeded colours near the neutral axis, as most of a photograph is: a grey from 0.2 to 0.5 in LogC4 and a
/// chroma of up to about 0.06 on each channel.
pub fn mild_colours(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let g = 0.2 + 0.3 * rng.unit();
            [0, 1, 2].map(|_| g + 0.03 * (rng.unit() - 0.5))
        })
        .collect()
}
