//! Command-line shape of `ym` (`docs/10-cli.md`): every command and option, and the parsers that
//! turn the text of an option into a value or a usage error.
//!
//! Options that take a number are checked against the range the spec gives, so `--contrast 500` is
//! a usage error (exit code 2) and not a quietly clamped value. A settings *file* is different: its
//! values are clamped like a slider (`ym_core::settings`), and the run says which ones were.

use std::path::PathBuf;
use std::sync::OnceLock;

use clap::{Args, Parser, Subcommand, ValueEnum};
use ym_core::apple_auto::AppleAutoMode;
use ym_core::version::{PIPELINE_VERSION, app_version};
use ym_export::{BitDepth, ColorSpace, FormatKind, TiffCompression};

/// Version text shown by `ym --version`: the package version, the pipeline version and the git
/// commit this binary was built from. clap prefixes it with the program name.
pub fn version_text() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        format!(
            "{} (pipeline {PIPELINE_VERSION}, {})",
            app_version(),
            env!("YM_GIT_SHA")
        )
    })
}

/// Develop and export Sony ARW photos with the same core as the Yellowmoon app.
#[derive(Debug, Parser)]
#[command(
    name = "ym",
    version = version_text(),
    arg_required_else_help = true,
    after_help = "Exit codes: 0 every file done, 1 some files failed, 2 usage error, 3 a look is missing, 4 no usable GPU (--require-gpu), 5 no input files, 130 interrupted."
)]
pub struct Cli {
    /// Print only errors on stderr.
    #[arg(long, short = 'q', global = true, conflicts_with = "verbose")]
    pub quiet: bool,
    /// Print what is happening on stderr: stages, timings and notes.
    #[arg(long, short = 'v', global = true)]
    pub verbose: bool,
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Develop and export photos. Defaults match the app: automatic development and the default
    /// look. Options change them for this run only.
    Render(Box<RenderArgs>),
    /// Print metadata and the automatic estimates without rendering.
    Info(InfoArgs),
    /// Manage the look library.
    #[command(subcommand)]
    Looks(LooksCommand),
    /// Print the diagnostics that go into a bug report.
    Doctor(DoctorArgs),
}

// ---------------------------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------------------------

const H_INPUT: &str = "Photos";
const H_LOOK: &str = "Look and display";
const H_LIGHT: &str = "Light";
const H_COLOR: &str = "Colour";
const H_DETAIL: &str = "Detail";
const H_RAW: &str = "RAW";
const H_OUTPUT: &str = "Output";
const H_RUN: &str = "Run";

/// Arguments of `ym render`.
#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  ym render ~/Pictures/2026-10-04 --look neutral --format jpeg --long-edge 3000 --out ~/Exports\n  fd -e ARW . /Volumes/A1 | ym render - --json --format tiff --bits 16 > results.jsonl\n  ym render DSC04204.ARW --exposure-offset -0.5 --wb 5200,4\n\nPer-photo settings beyond these options go in a JSON file given with --settings (the `ym:Settings` schema of an exported file). Options given alongside it override it. A file with several photos' settings is an object keyed by file name."
)]
pub struct RenderArgs {
    /// Files or folders to render; `-` reads file paths from stdin, one per line. Folders are walked
    /// for `*.ARW` (any case).
    #[arg(required = true, value_name = "FILES_OR_FOLDERS", help_heading = H_INPUT)]
    pub inputs: Vec<PathBuf>,
    /// Walk the subfolders of folders too.
    #[arg(long, help_heading = H_INPUT)]
    pub recursive: bool,

    /// Look: its full id (`builtin:neutral`), the part after the colon (`neutral`) or its name,
    /// in any case.
    #[arg(long, value_name = "LOOK", help_heading = H_LOOK)]
    pub look: Option<String>,
    /// How strongly the look applies, in percent (default 100).
    #[arg(long, value_name = "PERCENT", value_parser = percent, help_heading = H_LOOK)]
    pub intensity: Option<f64>,
    /// Display rendering: `aces2`, or `arri` (an imported ARRI display LUT: the one chosen in the
    /// app, else the only one imported; or give its id).
    #[arg(long, value_name = "aces2|arri", help_heading = H_LOOK)]
    pub display: Option<String>,
    /// A JSON file of settings (the `ym:Settings` schema), or `-` for stdin. It covers what these
    /// options can't: colour balance (`{ "kind": "auto" }` puts a shadows or highlights wheel back
    /// to Auto), curves, crop, orientation, white pick.
    #[arg(long, value_name = "FILE", help_heading = H_LOOK)]
    pub settings: Option<PathBuf>,

    /// Exposure in stops (-4 to 4), or `auto`.
    #[arg(long, value_name = "auto|EV", value_parser = exposure, allow_negative_numbers = true, conflicts_with = "exposure_offset", help_heading = H_LIGHT)]
    pub exposure: Option<ExposureArg>,
    /// Stops added to each photo's automatic exposure.
    #[arg(long, value_name = "EV", value_parser = finite, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub exposure_offset: Option<f64>,
    /// Contrast, -100 to 100, or `auto` (the default: 0, or the value Apple Auto chose). Around mid grey;
    /// black and the brightest tones stay where they are.
    #[arg(long, value_name = "auto|N", value_parser = percent_or_auto, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub contrast: Option<PercentArg>,
    /// Brightness, -100 to 100.
    #[arg(long, value_name = "N", value_parser = signed_percent, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub brightness: Option<f64>,
    /// Highlights, -100 to 100, or `auto` (the default: 0, or the value Apple Auto chose).
    #[arg(long, value_name = "auto|N", value_parser = percent_or_auto, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub highlights: Option<PercentArg>,
    /// Shadows, -100 to 100, or `auto` (the default: 0, or the value Apple Auto chose).
    #[arg(long, value_name = "auto|N", value_parser = percent_or_auto, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub shadows: Option<PercentArg>,
    /// White point, -100 to 100.
    #[arg(long, value_name = "N", value_parser = signed_percent, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub white: Option<f64>,
    /// Black point, -100 to 100, or `auto` (the default: 0, or the value Apple Auto chose).
    #[arg(long, value_name = "auto|N", value_parser = percent_or_auto, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub black: Option<PercentArg>,
    /// Stops added to each photo's automatic white lift, -3 to 3. 0 is the automatic amount; an
    /// offset that takes the lift to 0 or below switches it off (`--white-offset -3` always does).
    /// The lift only acts with the built-in ACES 2 display rendering, and a DNG carries it.
    #[arg(long, value_name = "EV", value_parser = white_offset, allow_negative_numbers = true, help_heading = H_LIGHT)]
    pub white_offset: Option<f64>,

    /// White balance: `auto`, or a temperature in kelvin (2000 to 11000) with an optional tint
    /// (-50 to 50), as `5200` or `5200,4`.
    #[arg(long, value_name = "auto|K[,TINT]", value_parser = white_balance, allow_negative_numbers = true, help_heading = H_COLOR)]
    pub wb: Option<WbArg>,
    /// Light preset instead of the estimated light.
    #[arg(long, value_enum, ignore_case = true, value_name = "LIGHT", help_heading = H_COLOR)]
    pub light: Option<LightArg>,
    /// Saturation, -100 to 100, or `auto` (the default: 0, or the value Apple Auto chose). More makes colours
    /// stronger, less makes them greyer; brightness stays.
    #[arg(long, value_name = "auto|N", value_parser = percent_or_auto, allow_negative_numbers = true, help_heading = H_COLOR)]
    pub saturation: Option<PercentArg>,

    /// Noise reduction: `auto`, or luminance, colour and detail strengths (0 to 100 each).
    #[arg(long, value_name = "auto|L,C,D", value_parser = noise_reduction, help_heading = H_DETAIL)]
    pub nr: Option<NrArg>,
    /// Sharpening: `auto`, or amount (0 to 300) and radius in pixels (0.3 to 2.5). A DNG is never sharpened.
    #[arg(long, value_name = "auto|AMOUNT,RADIUS", value_parser = sharpen, help_heading = H_DETAIL)]
    pub sharpen: Option<SharpenArg>,
    /// Output sharpening, 0 to 200 percent (default 100, 0 is off). It sharpens JPEG, TIFF and PNG
    /// files at the size they are written, after the resize; a DNG is never sharpened.
    #[arg(long, value_name = "PERCENT", value_parser = output_sharpening, allow_negative_numbers = true, help_heading = H_DETAIL)]
    pub output_sharpening: Option<f64>,

    /// Use the classical demosaic instead of the learned one.
    #[arg(long, help_heading = H_RAW)]
    pub no_neural: bool,
    /// Switch highlight reconstruction off.
    #[arg(long, help_heading = H_RAW)]
    pub no_highlight_recovery: bool,
    /// Switch the lens corrections (vignetting, distortion) off.
    #[arg(long, help_heading = H_RAW)]
    pub no_lens: bool,
    /// Switch the chromatic-aberration correction off.
    #[arg(long, help_heading = H_RAW)]
    pub no_ca: bool,
    /// Highlight range and low-light toe, 0 to 1 each, as `1,0.5` (the defaults). Highlights: where the camera's
    /// clipping point lands, from where it was recorded (0) to white, the clip of an ALEXA at EI 800 (1).
    #[arg(long, value_name = "HIGHLIGHTS,LOW-LIGHT", value_parser = rolloff, help_heading = H_RAW)]
    pub rolloff: Option<RolloffArg>,
    /// Force a camera profile, for comparisons (for example `sony-ilce-7rm4`).
    #[arg(long, value_name = "ID", help_heading = H_RAW)]
    pub profile: Option<String>,

    /// File format (default: the format chosen in the app, else jpeg).
    #[arg(long, value_enum, ignore_case = true, value_name = "FORMAT", help_heading = H_OUTPUT)]
    pub format: Option<FormatArg>,
    /// JPEG quality, 50 to 100 (default 92).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(50..=100), help_heading = H_OUTPUT)]
    pub quality: Option<u8>,
    /// Bits per channel for TIFF and PNG, 8 or 16 (default 16).
    #[arg(long, value_name = "BITS", value_parser = bits, help_heading = H_OUTPUT)]
    pub bits: Option<BitDepth>,
    /// TIFF compression (default lzw).
    #[arg(long, value_enum, ignore_case = true, value_name = "METHOD", help_heading = H_OUTPUT)]
    pub compression: Option<CompressionArg>,
    /// Colour space of JPEG, TIFF and PNG files (default srgb).
    #[arg(long = "colour-space", visible_alias = "color-space", value_enum, ignore_case = true, value_name = "SPACE", help_heading = H_OUTPUT)]
    pub color_space: Option<ColorSpaceArg>,
    /// Scale JPEG, TIFF and PNG files so the long edge has this many pixels (never enlarges).
    #[arg(long, value_name = "PX", value_parser = clap::value_parser!(u32).range(16..=65535), help_heading = H_OUTPUT)]
    pub long_edge: Option<u32>,
    /// Folder for the exports (default: `<source folder> export`).
    #[arg(long, value_name = "DIR", help_heading = H_OUTPUT)]
    pub out: Option<PathBuf>,
    /// File name template with {name}, {look}, {date}, {seq} and {camera} (default `{name}_{look}`).
    #[arg(long, value_name = "TEMPLATE", help_heading = H_OUTPUT)]
    pub name: Option<String>,
    /// Leave a file that already exists alone (default: keep both, adding " 2", " 3").
    #[arg(long, conflicts_with = "overwrite", help_heading = H_OUTPUT)]
    pub skip_existing: bool,
    /// Replace a file that already exists.
    #[arg(long, help_heading = H_OUTPUT)]
    pub overwrite: bool,

    /// Photos read and analysed at the same time. The export itself renders one photo at a time.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=64), help_heading = H_RUN)]
    pub jobs: Option<u32>,
    /// Print one JSON object per file on stdout, as each file finishes.
    #[arg(long, help_heading = H_RUN)]
    pub json: bool,
    /// Stop with exit code 4 when there is no usable GPU, instead of using the CPU.
    #[arg(long, conflicts_with = "cpu", help_heading = H_RUN)]
    pub require_gpu: bool,
    /// Render on the CPU only and open no graphics device. The golden images are made this way.
    /// Beats YELLOWMOON_GPU.
    #[arg(long, help_heading = H_RUN)]
    pub cpu: bool,
    /// Apple Auto for this run. On a Mac with Apple Intelligence, Apple's on-device model grades each photo:
    /// it sets the automatic exposure, highlights, shadows, black, contrast, warmth, tint and saturation and
    /// the colours of the shadows and highlights. Default: the app's setting where Apple Auto is available,
    /// off elsewhere.
    #[arg(long, value_enum, ignore_case = true, value_name = "on|off", hide_possible_values = true, help_heading = H_RUN)]
    pub apple_auto: Option<AppleAutoArg>,
    /// A text file with instructions for Apple's model, used for this run instead of those in Preferences
    /// (at most 2000 characters; an empty file is the default instructions).
    #[arg(long, value_name = "FILE", help_heading = H_RUN)]
    pub apple_auto_instructions: Option<PathBuf>,
}

impl RenderArgs {
    /// The flag that decides the graphics policy before `YELLOWMOON_GPU` is looked at, if one was
    /// given (`--cpu` and `--require-gpu` cannot be given together).
    #[must_use]
    pub fn policy_flag(&self) -> Option<ym_core::app::gpu::PolicyFlag> {
        use ym_core::app::gpu::PolicyFlag;
        if self.cpu {
            Some(PolicyFlag::Cpu)
        } else if self.require_gpu {
            Some(PolicyFlag::RequireGpu)
        } else {
            None
        }
    }
}

/// `--format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    /// JPEG.
    #[value(alias = "jpg")]
    Jpeg,
    /// TIFF.
    #[value(alias = "tif")]
    Tiff,
    /// PNG.
    Png,
    /// DNG with the look baked in. Apps other than Adobe's show the stored linear image darker until their own exposure handling is applied. With a look that includes its own display rendering, the DNG is close to, not the same as, the picture you see.
    Dng,
}

impl From<FormatArg> for FormatKind {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Jpeg => Self::Jpeg,
            FormatArg::Tiff => Self::Tiff,
            FormatArg::Png => Self::Png,
            FormatArg::Dng => Self::Dng,
        }
    }
}

/// `--compression`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CompressionArg {
    /// No compression.
    None,
    /// LZW.
    Lzw,
    /// Deflate, shown as ZIP.
    #[value(alias = "deflate")]
    Zip,
}

impl From<CompressionArg> for TiffCompression {
    fn from(c: CompressionArg) -> Self {
        match c {
            CompressionArg::None => Self::None,
            CompressionArg::Lzw => Self::Lzw,
            CompressionArg::Zip => Self::Zip,
        }
    }
}

/// `--colour-space`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ColorSpaceArg {
    /// sRGB.
    Srgb,
    /// Display P3.
    P3,
    /// Rec.2020.
    #[value(name = "rec2020")]
    Rec2020,
}

impl From<ColorSpaceArg> for ColorSpace {
    fn from(c: ColorSpaceArg) -> Self {
        match c {
            ColorSpaceArg::Srgb => Self::Srgb,
            ColorSpaceArg::P3 => Self::DisplayP3,
            ColorSpaceArg::Rec2020 => Self::Rec2020,
        }
    }
}

/// `--apple-auto` (`docs/19` section 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AppleAutoArg {
    /// Apple Auto acts where it is available, whatever the app's setting says.
    On,
    /// Auto is Yellowmoon's own for this run.
    Off,
}

impl AppleAutoArg {
    /// The core's mode for the option, given or not: without it the app's setting decides.
    #[must_use]
    pub fn mode(arg: Option<Self>) -> AppleAutoMode {
        match arg {
            None => AppleAutoMode::FromPreference,
            Some(Self::On) => AppleAutoMode::On,
            Some(Self::Off) => AppleAutoMode::Off,
        }
    }
}

/// `--light`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LightArg {
    /// The estimated light (clears a preset from `--settings`).
    Auto,
    /// Daylight, 5500 K.
    Day,
    /// Shade, 7500 K.
    Shade,
    /// Tungsten, 3200 K.
    Tungsten,
    /// LED light at the estimated colour.
    Led,
    /// Fluorescent light at the estimated colour.
    Fluorescent,
}

/// `--exposure`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExposureArg {
    /// The automatic exposure.
    Auto,
    /// An absolute exposure in stops.
    Ev(f64),
}

/// `--contrast`, `--saturation`, `--highlights`, `--shadows` and `--black`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PercentArg {
    /// The photo's Auto value: 0, or Apple Auto's (`docs/19` section 5).
    Auto,
    /// A manual value, -100 to 100.
    Value(f64),
}

/// `--wb`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WbArg {
    /// The estimated white balance.
    Auto,
    /// A temperature and, when given, a tint (otherwise the automatic tint stays).
    Manual {
        /// Kelvin, 2000 to 11000.
        kelvin: f64,
        /// Tint, -50 to 50.
        tint: Option<f64>,
    },
}

/// `--nr`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NrArg {
    /// The automatic noise reduction.
    Auto,
    /// Strengths, 0 to 100.
    Manual {
        /// Luminance.
        luminance: f64,
        /// Colour.
        color: f64,
        /// Detail.
        detail: f64,
    },
}

/// `--sharpen`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SharpenArg {
    /// The automatic sharpening.
    Auto,
    /// Amount 0 to 300 and radius 0.3 to 2.5 pixels.
    Manual {
        /// Amount.
        amount: f64,
        /// Radius in pixels.
        radius_px: f64,
    },
}

/// `--rolloff`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloffArg {
    /// The highlight range, 0 to 1 (`docs/02` C7).
    pub highlights: f64,
    /// The low-light toe, 0 to 1.
    pub low_light: f64,
}

fn number(text: &str, what: &str) -> Result<f64, String> {
    let t = text.trim();
    let value: f64 = t
        .parse()
        .map_err(|_| format!("`{t}` is not a number ({what})"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("`{t}` is not a number ({what})"))
    }
}

fn in_range(text: &str, what: &str, lo: f64, hi: f64) -> Result<f64, String> {
    let v = number(text, what)?;
    if (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{what} must be between {lo} and {hi}, not {v}"))
    }
}

fn finite(text: &str) -> Result<f64, String> {
    number(text, "a number of stops")
}

fn signed_percent(text: &str) -> Result<f64, String> {
    in_range(text, "the value", -100.0, 100.0)
}

fn percent_or_auto(text: &str) -> Result<PercentArg, String> {
    if text.trim().eq_ignore_ascii_case("auto") {
        return Ok(PercentArg::Auto);
    }
    signed_percent(text).map(PercentArg::Value)
}

fn percent(text: &str) -> Result<f64, String> {
    in_range(text, "the value", 0.0, 100.0)
}

fn exposure(text: &str) -> Result<ExposureArg, String> {
    if text.trim().eq_ignore_ascii_case("auto") {
        return Ok(ExposureArg::Auto);
    }
    in_range(text, "the exposure", -4.0, 4.0).map(ExposureArg::Ev)
}

fn parts<'a>(text: &'a str, expected: usize, form: &str) -> Result<Vec<&'a str>, String> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    if parts.len() == expected {
        Ok(parts)
    } else {
        Err(format!("expected {form}, not `{text}`"))
    }
}

fn white_balance(text: &str) -> Result<WbArg, String> {
    if text.trim().eq_ignore_ascii_case("auto") {
        return Ok(WbArg::Auto);
    }
    let mut it = text.split(',').map(str::trim);
    let kelvin = in_range(
        it.next().unwrap_or_default(),
        "the temperature",
        2000.0,
        11000.0,
    )?;
    let tint = match it.next() {
        None => None,
        Some(t) => Some(in_range(t, "the tint", -50.0, 50.0)?),
    };
    if it.next().is_some() {
        return Err(format!("expected `auto`, `K` or `K,TINT`, not `{text}`"));
    }
    Ok(WbArg::Manual { kelvin, tint })
}

fn noise_reduction(text: &str) -> Result<NrArg, String> {
    if text.trim().eq_ignore_ascii_case("auto") {
        return Ok(NrArg::Auto);
    }
    let p = parts(text, 3, "`auto` or `LUMINANCE,COLOUR,DETAIL`")?;
    Ok(NrArg::Manual {
        luminance: in_range(p[0], "the luminance strength", 0.0, 100.0)?,
        color: in_range(p[1], "the colour strength", 0.0, 100.0)?,
        detail: in_range(p[2], "the detail strength", 0.0, 100.0)?,
    })
}

fn sharpen(text: &str) -> Result<SharpenArg, String> {
    if text.trim().eq_ignore_ascii_case("auto") {
        return Ok(SharpenArg::Auto);
    }
    let p = parts(text, 2, "`auto` or `AMOUNT,RADIUS`")?;
    Ok(SharpenArg::Manual {
        amount: in_range(p[0], "the amount", 0.0, 300.0)?,
        radius_px: in_range(p[1], "the radius", 0.3, 2.5)?,
    })
}

fn output_sharpening(text: &str) -> Result<f64, String> {
    in_range(text, "the output sharpening", 0.0, 200.0)
}

fn white_offset(text: &str) -> Result<f64, String> {
    in_range(text, "the white offset", -3.0, 3.0)
}

fn rolloff(text: &str) -> Result<RolloffArg, String> {
    let p = parts(text, 2, "`HIGHLIGHTS,LOW-LIGHT`")?;
    Ok(RolloffArg {
        highlights: in_range(p[0], "the highlight range", 0.0, 1.0)?,
        low_light: in_range(p[1], "the low-light toe", 0.0, 1.0)?,
    })
}

fn bits(text: &str) -> Result<BitDepth, String> {
    match text.trim() {
        "8" => Ok(BitDepth::Eight),
        "16" => Ok(BitDepth::Sixteen),
        other => Err(format!("the bit depth must be 8 or 16, not `{other}`")),
    }
}

// ---------------------------------------------------------------------------------------------
// info, looks, doctor
// ---------------------------------------------------------------------------------------------

/// Arguments of `ym info`.
#[derive(Debug, Args)]
pub struct InfoArgs {
    /// ARW files to inspect.
    #[arg(required = true, value_name = "FILES")]
    pub files: Vec<PathBuf>,
    /// Print one JSON object per file instead of text (the shape of `render --json` without
    /// `output` and `timings_ms`).
    #[arg(long)]
    pub json: bool,
    /// Force a camera profile, for comparisons (for example `sony-ilce-7rm4`).
    #[arg(long, value_name = "ID")]
    pub profile: Option<String>,
    /// Apple Auto for this run. On a Mac with Apple Intelligence, Apple's on-device model grades each photo:
    /// it sets the automatic exposure, highlights, shadows, black, contrast, warmth, tint and saturation and
    /// the colours of the shadows and highlights. Default: the app's setting where Apple Auto is available,
    /// off elsewhere.
    #[arg(
        long,
        value_enum,
        ignore_case = true,
        value_name = "on|off",
        hide_possible_values = true
    )]
    pub apple_auto: Option<AppleAutoArg>,
    /// A text file with instructions for Apple's model, used for this run instead of those in Preferences
    /// (at most 2000 characters; an empty file is the default instructions).
    #[arg(long, value_name = "FILE")]
    pub apple_auto_instructions: Option<PathBuf>,
}

/// Arguments of `ym doctor`.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
    /// Time the render path on this photo and print that report instead of the diagnostics. Follows YELLOWMOON_GPU
    /// (off for the CPU). The report names no file.
    #[arg(long, value_name = "FILE")]
    pub bench: Option<std::path::PathBuf>,
    /// With --bench: how many times the same JPEG is exported to check that the bytes are equal.
    #[arg(long, default_value_t = crate::doctor::bench::DEFAULT_REPEAT, value_parser = clap::value_parser!(u32).range(1..=1000), requires = "bench")]
    pub repeat: u32,
}

/// Subcommands of `ym looks`.
#[derive(Debug, Subcommand)]
pub enum LooksCommand {
    /// List the built-in and imported looks, with ids, names, groups, libraries and inputs; the
    /// default look is marked.
    List(LooksListArgs),
    /// Import ARRI look packages, folders of looks, `.cube` files or ARRI Look Files (`.alf4`,
    /// `.alf4c`, `.aml`) into the look library.
    Import(LooksImportArgs),
    /// Re-hash every imported look and report missing or changed files.
    Verify,
    /// Set the look new photos get.
    Default(LooksDefaultArgs),
    /// Developer tool: bake a parametric look to a 33-point LUT and render a contact sheet.
    Bake(LooksBakeArgs),
}

/// Arguments of `ym looks list`.
#[derive(Debug, Args)]
pub struct LooksListArgs {
    /// Print the list as JSON.
    #[arg(long)]
    pub json: bool,
}

/// What a `.cube` file expects as input (`ym looks import --input`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum InputArg {
    /// LogC4 / ARRI Wide Gamut 4, log to log.
    #[value(name = "logc4")]
    LogC4,
    /// LogC3 EI 800 / ARRI Wide Gamut 3, log to log.
    #[value(name = "logc3")]
    LogC3,
    /// LogC3 in, Rec.709 out: the look and the display rendering baked together.
    #[value(name = "logc3-rec709")]
    LogC3Rec709,
}

/// Arguments of `ym looks import`.
#[derive(Debug, Args)]
pub struct LooksImportArgs {
    /// Zip files, folders, `.cube` files or ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) to import.
    #[arg(required = true, value_name = "PATHS")]
    pub paths: Vec<PathBuf>,
    /// What a `.cube` file that is not part of an ARRI package expects as input. ARRI Look Files say
    /// it themselves, so it does not apply to them.
    #[arg(long, value_enum, value_name = "ENCODING")]
    pub input: Option<InputArg>,
}

/// Arguments of `ym looks default`.
#[derive(Debug, Args)]
pub struct LooksDefaultArgs {
    /// The look: its full id, the part after the colon, or its name.
    #[arg(value_name = "LOOK")]
    pub look: String,
}

/// Arguments of `ym looks bake`.
#[derive(Debug, Args)]
pub struct LooksBakeArgs {
    /// The parametric look file to bake (for example `signature.json`).
    #[arg(value_name = "LOOK_JSON")]
    pub look: PathBuf,
    /// ARW files to render into a contact sheet: each row shows Neutral on the left and the baked
    /// look on the right.
    #[arg(long, num_args = 1.., value_name = "ARW")]
    pub preview: Vec<PathBuf>,
    /// Folder for the `.cube` file and the contact sheet (default: the look file's folder).
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_dng_format_help_says_other_apps_show_it_darker_and_what_a_display_rendering_look_does() {
        let help = FormatArg::Dng
            .to_possible_value()
            .and_then(|v| v.get_help().map(ToString::to_string))
            .expect("the DNG format has help text");
        // clap drops the final full stop of a help text
        for note in [
            ym_export::dng::OTHER_APPS_NOTE,
            ym_export::dng::OWN_DISPLAY_NOTE,
        ] {
            assert!(
                help.contains(note.trim_end_matches('.')),
                "the CLI help and ym-export's note differ: {help}"
            );
        }
    }

    #[test]
    fn apple_auto_is_on_off_or_the_apps_setting_in_any_case() {
        let mode = |args: &[&str]| -> AppleAutoMode {
            let cli = Cli::try_parse_from(args).expect("parses");
            match cli.command {
                Command::Render(r) => AppleAutoArg::mode(r.apple_auto),
                Command::Info(i) => AppleAutoArg::mode(i.apple_auto),
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(
            mode(&["ym", "render", "a.ARW"]),
            AppleAutoMode::FromPreference
        );
        assert_eq!(
            mode(&["ym", "render", "a.ARW", "--apple-auto", "on"]),
            AppleAutoMode::On
        );
        assert_eq!(
            mode(&["ym", "info", "a.ARW", "--apple-auto", "OFF"]),
            AppleAutoMode::Off
        );
        assert_eq!(
            mode(&["ym", "info", "a.ARW"]),
            AppleAutoMode::FromPreference
        );
        assert!(Cli::try_parse_from(["ym", "render", "a.ARW", "--apple-auto", "maybe"]).is_err());
        assert!(Cli::try_parse_from(["ym", "render", "a.ARW", "--apple-auto"]).is_err());
    }

    #[test]
    fn version_text_names_the_pipeline_and_a_commit() {
        let text = version_text();
        assert!(text.starts_with(app_version()), "{text}");
        assert!(
            text.contains(&format!("pipeline {PIPELINE_VERSION}")),
            "{text}"
        );
        assert!(text.ends_with(')'), "{text}");
    }

    #[test]
    fn white_balance_takes_kelvin_with_or_without_a_tint() {
        assert_eq!(white_balance("auto"), Ok(WbArg::Auto));
        assert_eq!(
            white_balance("5200"),
            Ok(WbArg::Manual {
                kelvin: 5200.0,
                tint: None
            })
        );
        assert_eq!(
            white_balance("5200,-4.5"),
            Ok(WbArg::Manual {
                kelvin: 5200.0,
                tint: Some(-4.5)
            })
        );
        for bad in ["1500", "12000", "5200,60", "5200,1,2", "warm", "", "nan"] {
            assert!(white_balance(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_composite_options_check_their_ranges_and_arity() {
        assert_eq!(exposure("auto"), Ok(ExposureArg::Auto));
        assert_eq!(exposure("-1.5"), Ok(ExposureArg::Ev(-1.5)));
        assert!(exposure("4.5").is_err() && exposure("x").is_err());
        assert_eq!(noise_reduction("auto"), Ok(NrArg::Auto));
        assert!(noise_reduction("10,20").is_err());
        assert!(noise_reduction("10,20,101").is_err());
        assert_eq!(
            noise_reduction("10, 20,30"),
            Ok(NrArg::Manual {
                luminance: 10.0,
                color: 20.0,
                detail: 30.0
            })
        );
        assert!(sharpen("100,0.2").is_err() && sharpen("301,1").is_err());
        assert_eq!(
            sharpen("100,1.2"),
            Ok(SharpenArg::Manual {
                amount: 100.0,
                radius_px: 1.2
            })
        );
        assert_eq!(output_sharpening("0"), Ok(0.0));
        assert_eq!(output_sharpening(" 35 "), Ok(35.0));
        assert_eq!(output_sharpening("200"), Ok(200.0));
        for bad in ["-1", "200.5", "300", "nan", "inf", "much", ""] {
            assert!(output_sharpening(bad).is_err(), "{bad:?}");
        }
        assert!(rolloff("0.5").is_err() && rolloff("0.5,1.5").is_err());
        assert!(bits("12").is_err());
        assert_eq!(bits("16"), Ok(BitDepth::Sixteen));
        assert!(signed_percent("-100").is_ok() && signed_percent("101").is_err());
        assert_eq!(percent_or_auto("auto"), Ok(PercentArg::Auto));
        assert_eq!(percent_or_auto(" AUTO "), Ok(PercentArg::Auto));
        assert_eq!(percent_or_auto("-15"), Ok(PercentArg::Value(-15.0)));
        assert_eq!(percent_or_auto("0"), Ok(PercentArg::Value(0.0)));
        for bad in ["101", "-100.5", "automatic", "", "nan"] {
            assert!(percent_or_auto(bad).is_err(), "{bad:?}");
        }
        assert!(percent("-1").is_err() && percent("inf").is_err());
    }
}
