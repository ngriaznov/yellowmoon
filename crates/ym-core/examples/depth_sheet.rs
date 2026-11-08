//! Depth mask contact sheets (D-06, redrawn for the gradient of 2026-11-08): one JPEG per photo that shows what the Depth row
//! does on it, made with the app's own code (the depth network and map, the rank and gradient of `ym_color::mask_depth`,
//! the raster pass of `ym_color::mask_raster` with Refine edges, Pick depth and, for a raw file, the renderer's `depth_map`,
//! `sample_depth` and `focus_plane`). A developer tool for local review (real photos are never committed); it is not shipped.
//!
//! ```text
//! YELLOWMOON_DATA=$PWD/data cargo run --release -p ym-core --example depth_sheet -- [options]
//!
//!   --out DIR        output folder (default ./depth-sheets); sheets go to DIR/sheets, numbers to DIR/stats.json
//!   --oracle DIR     the oracle fixtures (default testdata/depth/oracle-1); every photo of its manifest is a sheet
//!   --no-oracle      skip the oracle photos
//!   --arw FILE       a Sony raw file (repeatable): its sheet has the Focus plane tiles
//!   --only a,b       only these photos (oracle ids or raw file stems)
//!   --long-edge N    long edge of a tile (default 400)
//! ```
//!
//! One sheet per photo, four tiles to a row:
//!
//! 1. the photo, Show depth (the rank `r`, white is near), and the mask alone at the far limit f = 1 (the default, the whole
//!    depth: the Show depth picture itself) and at f = 0.7;
//! 2. the mask alone at f = 0.5 and f = 0.3 (Refine edges 0.5), and both laid over the photo;
//! 3. the f = 0.5 mask alone with Refine edges 0 and with 0.5, then Pick depth at a point (the middle of the largest near
//!    region): the mask at the limit the pick sets (the depth there plus 0.1) over the photo, and alone;
//! 4. for a raw file with a focus point: the AF box on the photo, the mask at the Focus plane's limit over the photo (what
//!    the renderer answers is written in the note) and alone.
//!
//! The network runs as the app runs it (`DepthModel::map_of`: the picture to `prepare_input`, `DepthAnything::infer` on the
//! cores but one). A raw file goes through the renderer itself.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use ym_color::mask::PhotoFrame;
use ym_color::mask_depth::{
    DEPTH_FAR_MIN, DEPTH_PICK_MARGIN, DepthMap, DepthSource, pick_depth, pick_far,
};
use ym_color::mask_raster::{Area, PlaneJob, RasterRow, RasterSource};
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{
    FocusPlane, FrameOptions, PhotoSource, RenderedFrame, ViewportRect, ViewportState,
};
use ym_core::session::OpenOptions as SessionOpenOptions;
use ym_export::ExportError;
use ym_export::jpeg::{JpegMetadata, write_jpeg_bands};
use ym_export::options::JpegQuality;
use ym_export::pixels::{BandSource, ExportHooks};
use ym_gpu::queue::CancelToken;
use ym_ml::depth::{DepthAnything, prepare_input};

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

// -------------------------------------------------------------------------------------------------
// Arguments
// -------------------------------------------------------------------------------------------------

struct Args {
    out: PathBuf,
    oracle: Option<PathBuf>,
    arws: Vec<PathBuf>,
    only: Option<Vec<String>>,
    long_edge: usize,
}

fn parse_args() -> Fallible<Args> {
    let mut a = Args {
        out: PathBuf::from("depth-sheets"),
        oracle: Some(PathBuf::from("testdata/depth/oracle-1")),
        arws: Vec::new(),
        only: None,
        long_edge: 400,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--out" => a.out = PathBuf::from(value("--out")?),
            "--oracle" => a.oracle = Some(PathBuf::from(value("--oracle")?)),
            "--no-oracle" => a.oracle = None,
            "--arw" => a.arws.push(PathBuf::from(value("--arw")?)),
            "--only" => {
                a.only = Some(value("--only")?.split(',').map(str::to_owned).collect());
            }
            "--long-edge" => a.long_edge = value("--long-edge")?.parse()?,
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    Ok(a)
}

// -------------------------------------------------------------------------------------------------
// Fixtures (an uncompressed .npz of float32 arrays, as tests/depth.rs of ym-ml reads them)
// -------------------------------------------------------------------------------------------------

struct Npy {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn u16_at(b: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

fn read_npz_array(path: &Path, key: &str) -> Fallible<Npy> {
    let bytes = fs::read(path)?;
    let eocd = (0..=bytes.len().saturating_sub(22))
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or("no zip end record")?;
    let count = u16_at(&bytes, eocd + 10);
    let mut at = u32_at(&bytes, eocd + 16);
    for _ in 0..count {
        let method = u16_at(&bytes, at + 10);
        let size = u32_at(&bytes, at + 20);
        let (nlen, elen, clen) = (
            u16_at(&bytes, at + 28),
            u16_at(&bytes, at + 30),
            u16_at(&bytes, at + 32),
        );
        let local = u32_at(&bytes, at + 42);
        let name = String::from_utf8(bytes[at + 46..at + 46 + nlen].to_vec())?;
        if name.trim_end_matches(".npy") == key {
            if method != 0 {
                return Err("the npz is compressed".into());
            }
            let (lnlen, lelen) = (u16_at(&bytes, local + 26), u16_at(&bytes, local + 28));
            return parse_npy(&bytes[local + 30 + lnlen + lelen..][..size]);
        }
        at += 46 + nlen + elen + clen;
    }
    Err(format!("no array {key}").into())
}

fn parse_npy(b: &[u8]) -> Fallible<Npy> {
    let (hlen, start) = if b[6] == 1 {
        (u16_at(b, 8), 10)
    } else {
        (u32_at(b, 8), 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen])?;
    let shape_text = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(['(', ')']).nth(1))
        .ok_or("shape")?;
    let shape = shape_text
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let data = b[start + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Npy { shape, data })
}

// -------------------------------------------------------------------------------------------------
// Pixels: tiles, text, JPEG
// -------------------------------------------------------------------------------------------------

/// 8-bit interleaved RGB rows for the JPEG writer.
struct Rgb8<'a> {
    width: u32,
    height: u32,
    rgb: &'a [u8],
}

impl BandSource for Rgb8<'_> {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), ExportError> {
        out.clear();
        let row = self.width as usize * 3;
        let slice = self
            .rgb
            .get(y0 * row..(y0 + rows) * row)
            .ok_or_else(|| ExportError::invalid("rows outside the picture"))?;
        out.extend_from_slice(slice);
        Ok(())
    }
    fn rows_u16(&mut self, _: usize, _: usize, _: &mut Vec<u16>) -> Result<(), ExportError> {
        Err(ExportError::invalid("8-bit rows only"))
    }
}

fn write_jpeg_file(path: &Path, w: usize, h: usize, rgb: &[u8], quality: u8) -> Fallible<()> {
    let mut src = Rgb8 {
        width: w as u32,
        height: h as u32,
        rgb,
    };
    write_jpeg_bands(
        BufWriter::new(File::create(path)?),
        &mut src,
        JpegQuality::new(quality)?,
        &JpegMetadata {
            exif_app1: None,
            xmp: None,
            icc: None,
        },
        &ExportHooks::none(),
    )?;
    Ok(())
}

/// Area average of `ch` interleaved channels from `sw x sh` to `tw x th`.
fn down(src: &[f32], ch: usize, sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0f32; tw * th * ch];
    for y in 0..th {
        let (y0, y1) = (
            y * sh / th,
            ((y + 1) * sh / th).max(y * sh / th + 1).min(sh),
        );
        for x in 0..tw {
            let (x0, x1) = (
                x * sw / tw,
                ((x + 1) * sw / tw).max(x * sw / tw + 1).min(sw),
            );
            let n = ((y1 - y0) * (x1 - x0)) as f32;
            for c in 0..ch {
                let mut s = 0f32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        s += src[(yy * sw + xx) * ch + c];
                    }
                }
                out[(y * tw + x) * ch + c] = s / n;
            }
        }
    }
    out
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// A tile of `w x h` pixels, interleaved RGB bytes.
struct Tile {
    label: String,
    note: String,
    rgb: Vec<u8>,
}

/// The plain photo.
fn photo_tile(photo: &[f32]) -> Vec<u8> {
    photo.iter().map(|&v| to_u8(v)).collect()
}

/// A mask (0 to 1 per pixel) as grey.
fn grey_tile(mask: &[f32]) -> Vec<u8> {
    mask.iter()
        .flat_map(|&m| {
            let g = to_u8(m);
            [g, g, g]
        })
        .collect()
}

/// The mask over the photo: the unselected part darkened, the selected part kept and tinted red.
fn overlay_tile(photo: &[f32], mask: &[f32]) -> Vec<u8> {
    const TINT: [f32; 3] = [1.0, 0.15, 0.1];
    let mut out = Vec::with_capacity(photo.len());
    for (p, &m) in photo.chunks_exact(3).zip(mask) {
        for c in 0..3 {
            let lit = p[c] * (0.25 + 0.75 * m);
            out.push(to_u8(lit * (1.0 - 0.3 * m) + TINT[c] * 0.3 * m));
        }
    }
    out
}

fn put(tile: &mut [u8], tw: usize, x: i64, y: i64, color: [u8; 3]) {
    let th = (tile.len() / 3 / tw) as i64;
    if x >= 0 && y >= 0 && (x as usize) < tw && y < th {
        let i = (y as usize * tw + x as usize) * 3;
        tile[i..i + 3].copy_from_slice(&color);
    }
}

/// A rectangle outline in yellow (shares of the tile), two pixels thick, with a black shadow.
fn draw_box(tile: &mut [u8], tw: usize, th: usize, cx: f64, cy: f64, bw: f64, bh: f64) {
    let (x0, x1) = (
        ((cx - bw / 2.0) * tw as f64) as i64,
        ((cx + bw / 2.0) * tw as f64) as i64,
    );
    let (y0, y1) = (
        ((cy - bh / 2.0) * th as f64) as i64,
        ((cy + bh / 2.0) * th as f64) as i64,
    );
    for (off, color) in [
        (-1i64, [0, 0, 0]),
        (0, [255, 230, 0]),
        (1, [255, 230, 0]),
        (2, [0, 0, 0]),
    ] {
        for x in x0 - off..=x1 + off {
            put(tile, tw, x, y0 - off, color);
            put(tile, tw, x, y1 + off, color);
        }
        for y in y0 - off..=y1 + off {
            put(tile, tw, x0 - off, y, color);
            put(tile, tw, x1 + off, y, color);
        }
    }
}

/// A cross at a point (shares of the tile).
fn draw_cross(tile: &mut [u8], tw: usize, th: usize, x: f64, y: f64) {
    let (px, py) = ((x * tw as f64) as i64, (y * th as f64) as i64);
    for d in -9i64..=9 {
        for t in -1i64..=1 {
            put(tile, tw, px + d, py + t, [0, 255, 255]);
            put(tile, tw, px + t, py + d, [0, 255, 255]);
        }
    }
}

/// The 5 x 7 bitmap font: seven rows of five bits, high bit on the left.
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '.' => [0, 0, 0, 0, 0, 0x0C, 0x0C],
        ',' => [0, 0, 0, 0, 0x0C, 0x04, 0x08],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '+' => [0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0],
        '=' => [0, 0, 0x1F, 0, 0x1F, 0, 0],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        '_' => [0, 0, 0, 0, 0, 0, 0x1F],
        '\'' => [0x04, 0x04, 0x08, 0, 0, 0, 0],
        _ => [0; 7],
    }
}

/// Text at `scale` on a canvas of `cw` pixels across.
fn draw_text(
    canvas: &mut [u8],
    cw: usize,
    x: usize,
    y: usize,
    text: &str,
    scale: usize,
    color: [u8; 3],
) {
    let ch = canvas.len() / 3 / cw;
    for (i, c) in text.chars().enumerate() {
        let g = glyph(c);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits & (0x10 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let (px, py) = (x + (i * 6 + col) * scale + sx, y + row * scale + sy);
                            if px < cw && py < ch {
                                let at = (py * cw + px) * 3;
                                canvas[at..at + 3].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The sheet: tiles of `tw x th`, four to a row, each under two lines of text.
fn compose(title: &str, tiles: &[Tile], tw: usize, th: usize) -> (usize, usize, Vec<u8>) {
    const COLS: usize = 4;
    const GAP: usize = 8;
    const HEAD: usize = 28;
    const LABEL: usize = 40;
    // A narrow (portrait) tile gets a cell wide enough for its labels.
    let cell = tw.max(340);
    let rows = tiles.len().div_ceil(COLS);
    let (w, h) = (
        COLS * cell + (COLS + 1) * GAP,
        HEAD + rows * (th + LABEL) + GAP,
    );
    let mut sheet = vec![28u8; w * h * 3];
    draw_text(&mut sheet, w, GAP, 7, title, 2, [255, 255, 255]);
    for (i, t) in tiles.iter().enumerate() {
        let (cx, cy) = (
            GAP + (i % COLS) * (cell + GAP),
            HEAD + (i / COLS) * (th + LABEL),
        );
        draw_text(&mut sheet, w, cx, cy + 3, &t.label, 2, [255, 255, 255]);
        draw_text(&mut sheet, w, cx, cy + 22, &t.note, 1, [200, 200, 120]);
        for y in 0..th {
            let dst = ((cy + LABEL + y) * w + cx + (cell - tw) / 2) * 3;
            sheet[dst..dst + tw * 3].copy_from_slice(&t.rgb[y * tw * 3..(y + 1) * tw * 3]);
        }
    }
    (w, h, sheet)
}

// -------------------------------------------------------------------------------------------------
// The app's raster pass on a photo
// -------------------------------------------------------------------------------------------------

/// A photo to evaluate: its 8-bit picture (the long edge of 1024, as float in [0, 1], sRGB) and the depth map.
struct Photo {
    name: String,
    cats: String,
    w: usize,
    h: usize,
    rgb: Vec<f32>,
    map: Arc<DepthMap>,
    /// Wall time of the map: the network (and, for a raw file, the picture) in ms.
    map_ms: f64,
    /// Wall time of the ranks in ms.
    rank_ms: f64,
    threads: usize,
    /// The raw file's data for the Focus plane rows.
    raw: Option<RawData>,
}

struct RawData {
    src: PhotoSource,
    core_cancel: CancelToken,
}

fn luma(rgb: &[f32]) -> Vec<f64> {
    rgb.chunks_exact(3)
        .map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
        .collect()
}

/// A row's plane through the app's raster pass on the photo's own grid (the long edge of 1024 output pixels): resample,
/// Refine edges with the photo's luma as the guide, quantise. Returns [0, 1] values.
fn plane(photo: &Photo, source: RasterSource, refine: f64) -> Vec<f32> {
    let (w, h) = (photo.w, photo.h);
    let row = RasterRow {
        source,
        grow: 0.0,
        feather: 0.0,
        refine,
    };
    let guide = luma(&photo.rgb);
    let position =
        move |x: usize, y: usize| Some([(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64]);
    let job = PlaneJob {
        photo: PhotoFrame::new(w as f64, h as f64, 1.0),
        grown: Area::whole(w, h),
        frame: Area::whole(w, h),
        window: Area::whole(w, h),
        position: &position,
        guide: Some(&guide),
    };
    row.plane(&job)
        .into_iter()
        .map(|c| f32::from(c) / 65535.0)
        .collect()
}

fn depth_source(photo: &Photo, far: f64) -> RasterSource {
    RasterSource::Depth(Arc::new(DepthSource {
        map: Arc::clone(&photo.map),
        far,
    }))
}

/// The middle of the largest connected region of the nearest 30 % of the photo (4-connected cells with `z < 0.3`), as shares
/// of the photo: a sensible place for Pick depth.
fn largest_near_region(map: &DepthMap) -> (f64, f64) {
    let (w, h) = (map.width, map.height);
    let near: Vec<bool> = (0..map.len()).map(|i| map.z(i) < 0.3).collect();
    let mut seen = vec![false; map.len()];
    let mut best: Vec<usize> = Vec::new();
    for s in 0..map.len() {
        if !near[s] || seen[s] {
            continue;
        }
        let (mut stack, mut comp) = (vec![s], Vec::new());
        seen[s] = true;
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (x, y) = (i % w, i / w);
            let mut push = |j: usize| {
                if near[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                push(i - 1);
            }
            if x + 1 < w {
                push(i + 1);
            }
            if y > 0 {
                push(i - w);
            }
            if y + 1 < h {
                push(i + w);
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    if best.is_empty() {
        return (0.5, 0.5);
    }
    let n = best.len() as f64;
    let (mx, my) = (
        best.iter().map(|&i| (i % w) as f64).sum::<f64>() / n,
        best.iter().map(|&i| (i / w) as f64).sum::<f64>() / n,
    );
    // The cell of the region nearest the centroid (the centroid itself can lie outside it).
    let &i = best
        .iter()
        .min_by(|&&a, &&b| {
            let da = ((a % w) as f64 - mx).powi(2) + ((a / w) as f64 - my).powi(2);
            let db = ((b % w) as f64 - mx).powi(2) + ((b / w) as f64 - my).powi(2);
            da.total_cmp(&db)
        })
        .unwrap_or(&best[0]);
    (
        ((i % w) as f64 + 0.5) / w as f64,
        ((i / w) as f64 + 0.5) / h as f64,
    )
}

fn share_selected(mask: &[f32]) -> f64 {
    mask.iter().filter(|&&m| m > 0.5).count() as f64 / mask.len().max(1) as f64
}

// -------------------------------------------------------------------------------------------------
// The Focus plane
// -------------------------------------------------------------------------------------------------

/// A point of the camera's frame (sensor orientation) as a share of the photo as the camera oriented it (EXIF 1 to 8).
fn oriented(orientation: u16, [x, y]: [f64; 2]) -> [f64; 2] {
    match orientation {
        2 => [1.0 - x, y],
        3 => [1.0 - x, 1.0 - y],
        4 => [x, 1.0 - y],
        5 => [y, x],
        6 => [1.0 - y, x],
        7 => [1.0 - y, 1.0 - x],
        8 => [y, 1.0 - x],
        _ => [x, y],
    }
}

/// What the example works out for the Focus plane of a raw file.
struct FocusInfo {
    /// The AF point and box as shares of the photo.
    center: [f64; 2],
    box_size: [f64; 2],
    has_box: bool,
    exact_centre: bool,
    /// The median depth position in the box.
    z_f: f64,
}

fn focus_info(photo: &Photo) -> Option<FocusInfo> {
    let raw = photo.raw.as_ref()?;
    let focus = raw.src.raw.meta.focus.filter(|f| f.point != [0, 0])?;
    let o = raw.src.exif_orientation();
    let (fw, fh) = (f64::from(focus.frame[0]), f64::from(focus.frame[1]));
    let center = oriented(
        o,
        [
            f64::from(focus.point[0]) / fw,
            f64::from(focus.point[1]) / fh,
        ],
    );
    let map = &photo.map;
    let (cw, ch) = (map.width as f64, map.height as f64);
    let (bw, bh) = match focus.box_size {
        Some([w, h]) => {
            let (w, h) = (f64::from(w) / fw, f64::from(h) / fh);
            if (5..=8).contains(&o) { (h, w) } else { (w, h) }
        }
        None => {
            let side = ym_color::mask_depth::FOCUS_BOX_DEFAULT * cw.max(ch);
            (side / cw, side / ch)
        }
    };
    let z_f = map.focus_z(center[0], center[1], bw, bh);
    Some(FocusInfo {
        center,
        box_size: [bw, bh],
        has_box: focus.box_size.is_some(),
        exact_centre: 2 * u32::from(focus.point[0]) == u32::from(focus.frame[0])
            && 2 * u32::from(focus.point[1]) == u32::from(focus.frame[1]),
        z_f,
    })
}

// -------------------------------------------------------------------------------------------------
// Making the photos
// -------------------------------------------------------------------------------------------------

fn weights_dir(root: &Path) -> PathBuf {
    root.join("models").join("depth")
}

fn pool() -> (rayon::ThreadPool, usize) {
    // The app's choice: all the cores but one (`DepthModel::run`).
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let n = cores.saturating_sub(1).max(1);
    (
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("a thread pool"),
        n,
    )
}

fn oracle_photo(
    dir: &Path,
    id: &str,
    cats: String,
    net: &DepthAnything,
    key: u64,
) -> Fallible<Photo> {
    let pic = read_npz_array(&dir.join(format!("{id}.npz")), "picture")?;
    let (h, w) = (pic.shape[0], pic.shape[1]);
    let (pool, threads) = pool();
    let t = Instant::now();
    let m = pool.install(|| -> Result<_, String> {
        let input = prepare_input(&pic.data, w, h).map_err(|e| e.to_string())?;
        net.infer(&input.data, input.width, input.height)
            .map_err(|e| e.to_string())
    })?;
    let map_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let map = DepthMap::new(key, m.width, m.height, m.data)?;
    let rank_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok(Photo {
        name: id.to_owned(),
        cats,
        w,
        h,
        rgb: pic.data,
        map: Arc::new(map),
        map_ms,
        rank_ms,
        threads,
        raw: None,
    })
}

fn srgb_decode(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// A Display P3 viewport frame as sRGB floats (the cell is the photo's aspect, so there is no letterbox to cut).
fn frame_to_srgb(frame: &RenderedFrame) -> Vec<f32> {
    const P3_TO_XYZ: [[f32; 3]; 3] = [
        [0.486_570_95, 0.265_667_7, 0.198_217_29],
        [0.228_974_56, 0.691_738_5, 0.079_286_91],
        [0.0, 0.045_113_38, 1.043_944_4],
    ];
    const XYZ_TO_SRGB: [[f32; 3]; 3] = [
        [3.240_454_2, -1.537_138_5, -0.498_531_4],
        [-0.969_266, 1.876_010_8, 0.041_556],
        [0.055_643_4, -0.204_025_9, 1.057_225_2],
    ];
    let mut m = [[0f32; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| XYZ_TO_SRGB[i][k] * P3_TO_XYZ[k][j]).sum();
        }
    }
    let lut: Vec<f32> = (0..256).map(|v| srgb_decode(v as f32 / 255.0)).collect();
    let mut out = Vec::with_capacity(frame.width * frame.height * 3);
    for px in frame.rgba.chunks_exact(4) {
        let p = [
            lut[usize::from(px[0])],
            lut[usize::from(px[1])],
            lut[usize::from(px[2])],
        ];
        for row in &m {
            let lin = row[0] * p[0] + row[1] * p[1] + row[2] * p[2];
            out.push(f32::from(to_u8(srgb_encode(lin))) / 255.0);
        }
    }
    out
}

fn raw_photo(core: &Core, file: &Path, long_edge: usize, key: u64) -> Fallible<Photo> {
    let session = &core.session;
    let outcome = session.open(&[file.to_path_buf()], SessionOpenOptions::default());
    let id = *outcome
        .added
        .first()
        .ok_or("the session did not take the file")?;
    if !session.wait_idle(Duration::from_secs(600)) {
        return Err("the session did not finish developing the photo".into());
    }
    let view = session.view(id)?;
    let raw = session.raw_image(id)?;
    let src = PhotoSource::from_view(&view, raw);
    let cancel = CancelToken::new();
    let r = &core.renderer;
    let image = r.final_size(&src, &cancel)?;
    let s = (long_edge as f64 / image.0.max(image.1) as f64).min(1.0);
    let (cw, ch) = (
        (image.0 as f64 * s).round().max(1.0),
        (image.1 as f64 * s).round().max(1.0),
    );
    let frame = r.viewport_frame(
        &src,
        &ViewportState::new(ViewportRect::cell(cw, ch, 1.0)),
        &FrameOptions::default(),
        &cancel,
    )?;
    let rgb = frame_to_srgb(&frame);
    let t = Instant::now();
    let map = r.depth_map(&src, &cancel)?;
    let map_ms = t.elapsed().as_secs_f64() * 1000.0;
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut map = (*map).clone();
    map.key = key;
    Ok(Photo {
        name: stem,
        cats: "raw file".to_owned(),
        w: frame.width,
        h: frame.height,
        rgb,
        map: Arc::new(map),
        map_ms,
        rank_ms: 0.0,
        threads: std::thread::available_parallelism()
            .map_or(1, usize::from)
            .saturating_sub(1)
            .max(1),
        raw: Some(RawData {
            src,
            core_cancel: cancel,
        }),
    })
}

// -------------------------------------------------------------------------------------------------
// One sheet
// -------------------------------------------------------------------------------------------------

struct Sheet {
    stats: Value,
}

fn make_sheet(photo: &Photo, args: &Args, core: Option<&Core>, dir: &Path) -> Fallible<Sheet> {
    let long = photo.w.max(photo.h) as f64;
    let (tw, th) = (
        ((photo.w as f64 * args.long_edge as f64 / long).round() as usize).max(1),
        ((photo.h as f64 * args.long_edge as f64 / long).round() as usize).max(1),
    );
    let small = |v: &[f32], ch: usize| down(v, ch, photo.w, photo.h, tw, th);
    let base = small(&photo.rgb, 3);
    let map = &photo.map;
    let n = map.len();

    // Show depth is `r` through the resample only (white is near).
    let show = map.show_plane(map.key ^ 0x5348_4f57);
    let r_view = plane(photo, RasterSource::Plane(Arc::new(show)), 0.0);
    // The mask of a far limit, through the raster pass with Refine edges.
    let mask = |far: f64, refine: f64| plane(photo, depth_source(photo, far), refine);

    let mut tiles: Vec<Tile> = Vec::new();
    let mut stats_limits = Vec::new();
    tiles.push(Tile {
        label: "PHOTO".into(),
        note: format!(
            "{} x {} PX, MAP {} x {} CELLS",
            photo.w, photo.h, map.width, map.height
        ),
        rgb: photo_tile(&base),
    });
    tiles.push(Tile {
        label: "DEPTH (RANK)".into(),
        note: "SHOW DEPTH: R, WHITE IS NEAR".into(),
        rgb: grey_tile(&small(&r_view, 1)),
    });
    // The mask at f = 1 (the default), 0.7, 0.5 and 0.3, as the gradient alone (white is the mask).
    let mut by_limit: Vec<(f64, Vec<f32>)> = Vec::new();
    for far in [1.0, 0.7, 0.5, 0.3] {
        let m = mask(far, 0.5);
        stats_limits.push(json!({
            "far": far, "selected": share_selected(&m),
            "mean": m.iter().map(|&v| f64::from(v)).sum::<f64>() / m.len().max(1) as f64,
        }));
        tiles.push(Tile {
            label: format!("MASK F = {far}"),
            note: if far >= 1.0 {
                "THE DEFAULT: WHOLE DEPTH, 1 - Z".into()
            } else {
                format!("FADES OUT AT {:.0} % OF THE DEPTH", far * 100.0)
            },
            rgb: grey_tile(&small(&m, 1)),
        });
        by_limit.push((far, m));
    }
    // The same masks over the photo, for f = 0.5 and 0.3 (where the mask lies on the picture).
    for (far, m) in by_limit.iter().filter(|(f, _)| *f == 0.5 || *f == 0.3) {
        tiles.push(Tile {
            label: format!("OVER THE PHOTO, F = {far}"),
            note: format!("SELECTS {:.0} % (OVER 0.5)", share_selected(m) * 100.0),
            rgb: overlay_tile(&base, &small(m, 1)),
        });
    }
    // Refine edges 0 and 0.5 as masks.
    let refine0 = mask(0.5, 0.0);
    let refine5 = by_limit
        .iter()
        .find(|(f, _)| *f == 0.5)
        .map(|(_, m)| m.clone())
        .unwrap_or_default();
    tiles.push(Tile {
        label: "F = 0.5, REFINE 0".into(),
        note: "MASK ONLY".into(),
        rgb: grey_tile(&small(&refine0, 1)),
    });
    tiles.push(Tile {
        label: "F = 0.5, REFINE 0.5".into(),
        note: "MASK ONLY".into(),
        rgb: grey_tile(&small(&refine5, 1)),
    });
    // Pick depth at the middle of the largest near region.
    let (px, py) = largest_near_region(map);
    let pf = pick_depth(map, px, py);
    let pick = mask(pf, 0.5);
    let mut pick_tile = overlay_tile(&base, &small(&pick, 1));
    draw_cross(&mut pick_tile, tw, th, px, py);
    tiles.push(Tile {
        label: format!("PICK F = {pf:.2}"),
        note: format!("POINT {px:.2},{py:.2} MARGIN {DEPTH_PICK_MARGIN}"),
        rgb: pick_tile,
    });
    tiles.push(Tile {
        label: format!("PICK F = {pf:.2}, MASK"),
        note: "REFINE 0.5".into(),
        rgb: grey_tile(&small(&pick, 1)),
    });

    // The Focus plane.
    let mut focus_stats = Value::Null;
    if let (Some(fi), Some(raw)) = (focus_info(photo), photo.raw.as_ref()) {
        let mut box_tile = photo_tile(&base);
        draw_box(
            &mut box_tile,
            tw,
            th,
            fi.center[0],
            fi.center[1],
            fi.box_size[0],
            fi.box_size[1],
        );
        tiles.push(Tile {
            label: "FOCUS BOX".into(),
            note: format!(
                "AT {:.2},{:.2} BOX {:.3} x {:.3}{}{}",
                fi.center[0],
                fi.center[1],
                fi.box_size[0],
                fi.box_size[1],
                if fi.has_box { "" } else { " (DEFAULT)" },
                if fi.exact_centre { " CENTRE" } else { "" }
            ),
            rgb: box_tile,
        });
        let ff = pick_far(fi.z_f);
        // What the renderer answers.
        let app = core.and_then(|c| c.renderer.focus_plane(&raw.src, &raw.core_cancel).ok());
        let app_far = match app {
            Some(FocusPlane::Far { far, .. }) => Some(far),
            _ => None,
        };
        let fb = mask(ff, 0.5);
        tiles.push(Tile {
            label: format!("FOCUS PLANE F = {ff:.2}"),
            note: format!(
                "Z IN BOX {:.2}; APP SAYS {}",
                fi.z_f,
                app_far.map_or("NONE".to_owned(), |a| format!("{a:.2}"))
            ),
            rgb: overlay_tile(&base, &small(&fb, 1)),
        });
        tiles.push(Tile {
            label: "FOCUS PLANE, MASK".into(),
            note: "REFINE 0.5".into(),
            rgb: grey_tile(&small(&fb, 1)),
        });
        let agrees = app_far.is_some_and(|a| (a - ff).abs() < 1e-9);
        focus_stats = json!({
            "center": fi.center, "box": fi.box_size, "has_box": fi.has_box, "exact_centre": fi.exact_centre,
            "z_in_box": fi.z_f, "far": ff, "app_far": app_far, "example_matches_app": agrees,
        });
    }

    let (w, h, sheet) = compose(&format!("{}  ({})", photo.name, photo.cats), &tiles, tw, th);
    let path = dir.join(format!("{}.jpg", photo.name));
    write_jpeg_file(&path, w, h, &sheet, 88)?;

    // The numbers.
    let mut s = map.d.clone();
    s.sort_unstable_by(f32::total_cmp);
    let q = |p: f64| f64::from(s[((n - 1) as f64 * p).round() as usize]);
    let zeros = map.d.iter().filter(|&&v| v + 0.0 == 0.0).count();
    let (mut best_tie, mut run) = (0usize, 0usize);
    for i in 0..n {
        run = if i > 0 && s[i].to_bits() == s[i - 1].to_bits() {
            run + 1
        } else {
            1
        };
        best_tie = best_tie.max(run);
    }
    let stats = json!({
        "photo": photo.name, "categories": photo.cats, "sheet": path.file_name().map(|f| f.to_string_lossy().into_owned()),
        "size": [photo.w, photo.h], "map": [map.width, map.height],
        "map_ms": photo.map_ms, "rank_ms": photo.rank_ms, "threads": photo.threads,
        "d": {"min": q(0.0), "p2": q(0.02), "p50": q(0.5), "p98": q(0.98), "max": q(1.0),
              "zero_share": zeros as f64 / n as f64, "largest_tie_share": best_tie as f64 / n as f64},
        "refine_changes": {"mean_abs": refine5.iter().zip(&refine0).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>() / refine5.len().max(1) as f64},
        "limits": stats_limits,
        "pick": {"point": [px, py], "far": pf},
        "focus": focus_stats,
        "far_min": DEPTH_FAR_MIN,
    });
    Ok(Sheet { stats })
}

fn main() -> Fallible<()> {
    let args = parse_args()?;
    let sheets = args.out.join("sheets");
    fs::create_dir_all(&sheets)?;
    let root =
        std::env::var_os("YELLOWMOON_DATA").map_or_else(|| PathBuf::from("data"), PathBuf::from);
    let wanted = |name: &str| {
        args.only
            .as_ref()
            .is_none_or(|o| o.iter().any(|x| x == name))
    };
    let mut all: Vec<Value> = Vec::new();
    let mut key = 1u64;

    if let Some(dir) = &args.oracle {
        let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
        let net = DepthAnything::load(&weights_dir(&root))?;
        let mut ids: Vec<(String, String)> = Vec::new();
        for k in ["photos", "depth_only"] {
            if let Some(o) = manifest[k].as_object() {
                for (id, v) in o {
                    let cats = v["categories"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    ids.push((id.clone(), cats));
                }
            }
        }
        ids.sort();
        for (id, cats) in ids.into_iter().filter(|(id, _)| wanted(id)) {
            let photo = oracle_photo(dir, &id, cats, &net, key)?;
            key += 1;
            let sheet = make_sheet(&photo, &args, None, &sheets)?;
            println!(
                "{id}: map {:.0} ms on {} threads, ranks {:.0} ms",
                photo.map_ms, photo.threads, photo.rank_ms
            );
            all.push(sheet.stats);
        }
    }

    if !args.arws.is_empty() {
        let mut options = CoreOptions::from_environment()?;
        options.app_root = args.out.join(".app-home");
        options.thumbnails = false;
        fs::create_dir_all(&options.app_root)?;
        let core = Core::open(options)?;
        for file in &args.arws {
            let stem = file
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if !wanted(&stem) {
                continue;
            }
            match raw_photo(&core, file, 1024, key) {
                Ok(photo) => {
                    key += 1;
                    let sheet = make_sheet(&photo, &args, Some(&core), &sheets)?;
                    println!("{stem}: map {:.0} ms (picture and network)", photo.map_ms);
                    all.push(sheet.stats);
                }
                Err(e) => println!("{stem}: skipped: {e}"),
            }
        }
    }

    fs::write(
        args.out.join("stats.json"),
        serde_json::to_string_pretty(&all)?,
    )?;
    // The index.
    let mut html = String::from(
        "<!doctype html><meta charset=utf-8><title>Depth sheets</title><body style=\"background:#222;color:#ddd;font:14px sans-serif\"><h1>Depth mask sheets</h1><ul>",
    );
    for s in &all {
        let name = s["photo"].as_str().unwrap_or("");
        html += &format!(
            "<li><a href=\"sheets/{name}.jpg\">{name}</a> ({}) map {:.0} ms</li>",
            s["categories"].as_str().unwrap_or(""),
            s["map_ms"].as_f64().unwrap_or(0.0)
        );
    }
    html += "</ul>";
    for s in &all {
        let name = s["photo"].as_str().unwrap_or("");
        html += &format!("<h3>{name}</h3><img src=\"sheets/{name}.jpg\" style=\"max-width:100%\">");
    }
    fs::write(args.out.join("index.html"), html)?;
    println!("{} sheets in {}", all.len(), sheets.display());
    Ok(())
}
