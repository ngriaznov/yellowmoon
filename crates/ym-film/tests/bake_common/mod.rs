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
