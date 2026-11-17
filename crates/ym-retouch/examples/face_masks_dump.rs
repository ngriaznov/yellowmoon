//! Dev tool (docs/27 S-09, S-10): runs the real `FaceParser` and `FaceMesh` on photos and writes the masks as 8-bit PNGs of
//! the photo's size, plus `faces.json` (the faces and the timings), for `tools/mediapipe/contact_sheet.py`.
//!
//! ```text
//! cargo run --release -p ym-retouch --features mediapipe --example face_masks_dump -- <out dir> [--scale N] <photo.jpg>...
//! ```
//!
//! With `--scale N` the photo is enlarged N times (bilinear) first, so that the faces have the size of a real photograph
//! (a face of 1000 to 2500 px) and the timings are those of one. The photos are read, never written.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::mediapipe::{AWG4_TO_REC709, MediaPipeFaces};
use ym_retouch::{Face, FaceMesh, FaceParser, FacePart, LinearAwg4Image, MeshTable, Rect};

fn decode(path: &Path) -> (usize, usize, Vec<u8>) {
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(
        std::fs::read(path).expect("read the photo"),
    ));
    let rgb = d.decode().expect("decode the photo");
    let info = d.info().expect("jpeg header");
    (usize::from(info.width), usize::from(info.height), rgb)
}

/// Bilinear enlargement of an RGB picture.
fn enlarge(w: usize, h: usize, rgb: &[u8], n: usize) -> (usize, usize, Vec<u8>) {
    let (nw, nh) = (w * n, h * n);
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        let sy = ((y as f32 + 0.5) / n as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) / n as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..3 {
                let p = |xx: usize, yy: usize| f32::from(rgb[(yy * w + xx) * 3 + c]);
                let v = (p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx) * (1.0 - ty)
                    + (p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx) * ty;
                out[(y * nw + x) * 3 + c] = v.round() as u8;
            }
        }
    }
    (nw, nh, out)
}

fn invert(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let a = m.map(|r| r.map(f64::from));
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let c = |r: usize, c: usize| {
        let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
        a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]
    };
    let mut out = [[0.0f32; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = (c(k, r) / det) as f32;
        }
    }
    out
}

fn to_linear_awg4(w: usize, h: usize, rgb: &[u8]) -> LinearAwg4Image {
    let inv = invert(AWG4_TO_REC709);
    let decode = |v: u8| {
        let e = f64::from(v) / 255.0;
        (if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }) as f32
    };
    let data = rgb
        .chunks_exact(3)
        .map(|p| {
            let l = [decode(p[0]), decode(p[1]), decode(p[2])];
            [0, 1, 2].map(|r| inv[r][0] * l[0] + inv[r][1] * l[1] + inv[r][2] * l[2])
        })
        .collect();
    LinearAwg4Image::new(w as u32, h as u32, data).expect("image")
}

fn write_png(path: &Path, w: u32, h: u32, data: &[u8]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(data)
        .expect("png data");
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = PathBuf::from(args.remove(0));
    let mut scale = 1usize;
    if let Some(i) = args.iter().position(|a| a == "--scale") {
        scale = args[i + 1].parse().expect("a scale");
        args.drain(i..i + 2);
    }
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&models).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&models).expect("mesh and segmenter weights");
    let (parser, mesh) = (engine.parser(), engine.mesh());
    let mut entries: Vec<String> = Vec::new();
    for photo in &args {
        let path = Path::new(photo);
        let id = path
            .file_stem()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        let (w0, h0, rgb0) = decode(path);
        let (w, h, rgb) = if scale > 1 {
            enlarge(w0, h0, &rgb0, scale)
        } else {
            (w0, h0, rgb0)
        };
        let image = to_linear_awg4(w, h, &rgb);
        let src = Rgb8::new(w, h, &rgb).expect("picture");
        let t = Instant::now();
        let faces = finder.find(&src, &FinderParams::default()).expect("find");
        let find_ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "{id}: {w} x {h}, {} face(s), finder {find_ms:.0} ms",
            faces.len()
        );
        for (k, d) in faces.iter().enumerate() {
            let face = Face {
                rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
                scale: d.scale,
            };
            let size = (image.width_px, image.height_px);
            // The first part call does the analysis (the two networks); the rest come from the cache.
            let t = Instant::now();
            let skin = parser
                .part_mask(&image, &face, FacePart::SkinWithoutEars)
                .expect("skin");
            let skin_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let eyes = parser
                .part_mask(&image, &face, FacePart::Eyes)
                .expect("eyes");
            let mouth = parser
                .part_mask(&image, &face, FacePart::MouthInterior)
                .expect("mouth");
            let parts_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let under = mesh
                .render(&image, &face, MeshTable::UnderEye, size)
                .expect("under eye");
            let flat = mesh
                .render(&image, &face, MeshTable::InvertedSkinUniformity, size)
                .expect("uniformity");
            let tables_ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "  face {k}: {}x{} px; skin (with the analysis) {skin_ms:.0} ms; eyes and mouth {parts_ms:.0} ms; both tables {tables_ms:.0} ms",
                face.rect.width_px, face.rect.height_px
            );
            let mut files = Vec::new();
            for (name, mask) in [
                ("skin", skin.map(|m| m.to_full(size).expect("full"))),
                ("eyes", eyes.map(|m| m.to_full(size).expect("full"))),
                ("mouth", mouth.map(|m| m.to_full(size).expect("full"))),
                ("under_eye", Some(under.to_full(size).expect("full"))),
                ("non_uniform", Some(flat.to_full(size).expect("full"))),
            ] {
                if let Some(m) = mask {
                    let file = format!("{id}_{k}_{name}.png");
                    write_png(&out_dir.join(&file), m.width_px, m.height_px, &m.data);
                    files.push(format!("\"{name}\": \"{file}\""));
                }
            }
            let r = face.rect;
            let mut e = String::new();
            write!(
                e,
                "  {{\"photo\": \"{}\", \"scale\": {scale}, \"face\": {k}, \"rect\": [{}, {}, {}, {}], \"skin_ms\": {skin_ms:.1}, \"masks\": {{{}}}}}",
                path.display(),
                r.x_px,
                r.y_px,
                r.width_px,
                r.height_px,
                files.join(", ")
            )
            .expect("write");
            entries.push(e);
        }
    }
    std::fs::write(
        out_dir.join("faces.json"),
        format!("[\n{}\n]\n", entries.join(",\n")),
    )
    .expect("faces.json");
}
