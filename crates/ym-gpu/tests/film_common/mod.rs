//! Shared helpers of the Film kernel tests (`tests/gpu_film*.rs`): the CPU reference built from `ym-film`, the conversion
//! of its baked tables to the plain data `ym_gpu::kernels::film` takes (the template for F-08b's glue in `ym-core`), a
//! synthetic photo and the 10-bit codes the Output makes of a LogC4 colour.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use ym_color::display::OutputTarget;
use ym_color::display::cine::CineRendering;
use ym_color::encode::OutputSpace;
use ym_color::gamut::{LinearAwg4, matrices};
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_film::bake::{BakeSettings, BakedFilm, Baker};
use ym_film::grain::{HASH_CONSTANTS, LUT_ENTRIES};
use ym_film::scene::{KNEE, LIMIT};
use ym_film::spatial::{FilmFrame, Frame, SpatialModel, SpatialSettings};
use ym_film::{FilmChain, FilmParams, Output, PaperChoice, Spectra, TableSpectra};
use ym_gpu::kernels::film::{
    CoreData, CurveSet, FilmColourData, FilmPlan, FilmScalars, GaussTerm, GrainData, Table3Data,
};

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

pub fn spectra() -> Arc<dyn Spectra> {
    Arc::new(TableSpectra::load(&data_dir().join("film/upsampling")).expect("upsampling table"))
}

pub fn baker() -> &'static Baker {
    static B: OnceLock<Baker> = OnceLock::new();
    B.get_or_init(|| Baker::new("upsampling-256", 4))
}

/// The film node of a stock for `output` with the given spatial settings. The 3D tables are small (33^3) to keep the
/// tests quick: the kernels are tested against the same tables on the CPU.
pub fn film(id: &str, output: Output, settings: SpatialSettings) -> FilmFrame {
    let chain = FilmChain::new(
        &data_dir(),
        id,
        &PaperChoice::Matching,
        FilmParams {
            output,
            ..FilmParams::default()
        },
        spectra(),
    )
    .unwrap_or_else(|e| panic!("{id}: {e}"));
    let mut bs = BakeSettings::of(&chain);
    bs.t3a_size = 33;
    bs.t3b_size = 49;
    let baked = baker().bake(&chain, &bs).expect("bake");
    let model = SpatialModel::load(&data_dir(), id).expect("model");
    FilmFrame::new(baked, model, settings)
}

fn curves(c: &[ym_film::bake::Curve1; 3]) -> CurveSet {
    CurveSet {
        lo: [0, 1, 2].map(|i| c[i].domain().0),
        hi: [0, 1, 2].map(|i| c[i].domain().1),
        data: c.iter().flat_map(|k| k.data().iter().copied()).collect(),
    }
}

fn table3(t: &ym_film::bake::Table3) -> Table3Data {
    Table3Data {
        size: t.size(),
        lo: t.axis_min(),
        hi: t.axis_max(),
        power: t.power(),
        data: t.data().to_vec(),
    }
}

pub fn output_to_xyz(o: Output) -> [[f64; 3]; 3] {
    let m = matrices();
    match o {
        Output::Srgb => m.rec709_to_xyz,
        Output::DisplayP3 => m.p3_to_xyz,
    }
}

/// The plain data of a baked film.
pub fn colour_data(b: &BakedFilm) -> FilmColourData {
    let p = b.device_parts();
    let core = p.core;
    let d = core.dims();
    let (s_lo, s_span) = core.level_axis();
    FilmColourData {
        scalars: FilmScalars {
            scene_to_xyz: p.scene_to_xyz,
            white_xy: p.scene_white_xy,
            knee: KNEE,
            limit: LIMIT,
            log10_gain: p.log10_gain,
            coupler_amount: p.coupler_amount,
            coupler_matrix: p.coupler_matrix,
            print_log10_shift: p.print_log10_shift,
            output_to_xyz: output_to_xyz(p.output),
        },
        locus: p.scene_locus.to_vec(),
        t1_dims: {
            let (nx, ny) = p.t1.dims();
            [nx, ny]
        },
        t1_step: p.t1.step(),
        t1: p.t1.data().to_vec(),
        film0: curves(&p.curves.film0),
        paper: p.curves.paper.as_ref().map(curves),
        t3a: p.t3a.map(table3),
        t3b: table3(p.t3b),
        core: CoreData {
            dims: [d.ns, d.nt, d.nk],
            steps: core.steps() as u32,
            s_lo,
            s_span,
            ln_reach: core.ln_reach().to_vec(),
            nodes: core.data().iter().flatten().copied().collect(),
        },
    }
}

/// The plan of a frame as plain data.
pub fn plan_data(f: &FilmFrame, frame: Frame) -> FilmPlan {
    let p = f.device_plan(frame);
    let term = |t: &ym_film::spatial::Term| GaussTerm {
        sigma_px: t.sigma_px,
        weight: t.weight,
    };
    FilmPlan {
        frame: (frame.width, frame.height),
        halation: p.halation.iter().map(term).collect(),
        diffusion: p.diffusion.iter().map(term).collect(),
        diffuse: p.diffuse,
        cloud_px: p.cloud_px,
        grain: p.grain_on.then(|| {
            let mut lut = Vec::with_capacity(9 * LUT_ENTRIES);
            for c in 0..3 {
                for s in 0..3 {
                    lut.extend(p.grain.table(c, s).iter().map(|&v| v as f32));
                }
            }
            GrainData {
                lut,
                d_top: p.grain.d_top(),
                scale: p.grain_scale,
                seed: p.grain_seed,
                hash: HASH_CONSTANTS,
            }
        }),
    }
}

/// A photo-like test image, continuous in the frame's normalised coordinates: a gradient, a bright soft disc (a
/// highlight for the halation), a dark bar with a hard edge and a coloured patch, as LogC4 / AWG4.
pub fn image_at(x: f64, y: f64, aspect: f64) -> [f64; 3] {
    let (u, v) = (x, y);
    let mut lin = [
        0.12 + 0.25 * u,
        0.15 + 0.2 * v / aspect,
        0.1 + 0.1 * (1.0 - u),
    ];
    let r = ((u - 0.3).powi(2) + (v - 0.3 * aspect).powi(2)).sqrt();
    let disc = 1.0 / (1.0 + ((r - 0.06) / 0.012).exp());
    for c in &mut lin {
        *c += 12.0 * disc;
    }
    let w = 0.004;
    let bar = 1.0 / (1.0 + (-(u - 0.55) / w).exp()) / (1.0 + ((u - 0.7) / w).exp());
    let e = bar / (1.0 + (-(v - 0.15 * aspect) / w).exp());
    for c in &mut lin {
        *c *= 1.0 - 0.9 * e;
    }
    let patch =
        1.0 / (1.0 + (((u - 0.8).abs().max((v - 0.6 * aspect).abs()) - 0.05) / 0.004).exp());
    lin[0] += 0.5 * patch;
    lin[1] *= 1.0 - 0.6 * patch;
    logc4_encode_rgb(lin)
}

/// The image over `rect` of a frame as planar f32 (three planes of the rectangle).
pub fn make_input(frame: (usize, usize), rect: ym_gpu::kernels::film::FRect) -> Vec<f32> {
    let aspect = frame.1 as f64 / frame.0 as f64;
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let mut out = vec![0.0_f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let v = image_at(
                ((rect.x0 as usize + x) as f64 + 0.5) / frame.0 as f64,
                ((rect.y0 as usize + y) as f64 + 0.5) / frame.0 as f64,
                aspect,
            );
            for c in 0..3 {
                out[c * w * h + y * w + x] = v[c] as f32;
            }
        }
    }
    out
}

/// A seeded generator (splitmix64).
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

pub fn output_target(o: Output) -> OutputTarget {
    match o {
        Output::Srgb => OutputTarget::Srgb,
        Output::DisplayP3 => OutputTarget::DisplayP3,
    }
}

/// The unrounded 10-bit code values the Output makes of a LogC4 / AWG4 colour: the cine rendering, then C13's transfer
/// function, scaled to 0 ..= 1023.
pub fn codes10(cine: &CineRendering, space: OutputSpace, logc4: [f64; 3]) -> [f64; 3] {
    let display = cine.forward(LinearAwg4(logc4_decode_rgb(logc4)));
    space.encode_clamped(display).map(|v| v * 1023.0)
}

/// The LogC4 photos of `testdata/film-bake` (16-bit sRGB PNGs the lead made from the public ARW files), every
/// `stride`-th pixel in each direction, as LogC4 / AWG4 (the pre-image of the neutral look's display colour). Empty when
/// the folder is absent.
pub fn photo_pixels(stride: usize) -> Vec<(String, Vec<[f64; 3]>)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/film-bake");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let cine = CineRendering::new(OutputTarget::Srgb);
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
        out.push((name, pixels));
    }
    out
}

/// Pixels as a planar f32 image of `w x h` (padded with the last pixel).
pub fn planar(pixels: &[[f64; 3]], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; 3 * w * h];
    for i in 0..w * h {
        let p = pixels[i.min(pixels.len() - 1)];
        for c in 0..3 {
            out[c * w * h + i] = p[c] as f32;
        }
    }
    out
}
