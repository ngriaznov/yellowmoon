//! Dev tool (docs/27 S-16): retouches the faces of photos with the real models and the classical proposer at some amounts and
//! writes the face crops (and a crop of the mouth) as sRGB PNGs, for a person to look at in a contact sheet.
//!
//! ```text
//! cargo run --release -p ym-retouch --features mediapipe --example face_retouch_dump -- <out dir> [--scale N] [--tag T] <photo.jpg>...
//! ```
//!
//! Files: `<tag><photo>_<face>_<setting>.png` (the face and a margin) and `<tag><photo>_<face>_mouth_<setting>.png`, for the
//! settings `b0` (nothing), `b30`, `b60`, `b100` (Blemish) and `t30`, `t60`, `t100` (Teeth whitening). The photos are read, never
//! written.

use std::path::{Path, PathBuf};

use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::mediapipe::{AWG4_TO_REC709, MediaPipeFaces};
use ym_retouch::{
    ClassicalBeautyMarks, ClassicalProposer, Face, LinearAwg4Image, Models, Rect, RetouchCache,
    SkinSettings, prepare_skin, retouch_skin,
};

fn decode(path: &Path) -> (usize, usize, Vec<u8>) {
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(
        std::fs::read(path).expect("read the photo"),
    ));
    let rgb = d.decode().expect("decode the photo");
    let info = d.info().expect("jpeg header");
    (usize::from(info.width), usize::from(info.height), rgb)
}

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

/// The crop `(x, y, w, h)` of `img` as sRGB bytes.
fn crop_srgb(img: &LinearAwg4Image, (x0, y0, cw, ch): (usize, usize, usize, usize)) -> Vec<u8> {
    let m = AWG4_TO_REC709;
    let w = img.width_px as usize;
    let mut out = Vec::with_capacity(cw * ch * 3);
    for y in y0..y0 + ch {
        for x in x0..x0 + cw {
            let p = img.data[y * w + x];
            for row in &m {
                let l = (row[0] * p[0] + row[1] * p[1] + row[2] * p[2]).clamp(0.0, 1.0);
                let e = if l <= 0.003_130_8 {
                    12.92 * l
                } else {
                    1.055 * l.powf(1.0 / 2.4) - 0.055
                };
                out.push((e * 255.0).round() as u8);
            }
        }
    }
    out
}

fn write_png(path: &Path, w: usize, h: usize, data: &[u8]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
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
    let mut tag = String::new();
    if let Some(i) = args.iter().position(|a| a == "--tag") {
        tag = args[i + 1].clone();
        args.drain(i..i + 2);
    }
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let weights = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&weights).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&weights).expect("mesh and segmenter weights");
    let models = Models {
        parser: Box::new(engine.parser()),
        mesh: Box::new(engine.mesh()),
        beauty_marks: Some(Box::new(ClassicalBeautyMarks)),
        proposer: Box::new(ClassicalProposer::new(450.0)),
    };
    let settings: Vec<(&str, SkinSettings)> = {
        let b = |blemish: f32| SkinSettings {
            blemish,
            ..Default::default()
        };
        let t = |teeth_whitening: f32| SkinSettings {
            teeth_whitening,
            ..Default::default()
        };
        vec![
            ("b0", b(0.0)),
            ("b30", b(30.0)),
            ("b60", b(60.0)),
            ("b100", b(100.0)),
            ("t30", t(30.0)),
            ("t60", t(60.0)),
            ("t100", t(100.0)),
        ]
    };
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
        let faces = finder.find(&src, &FinderParams::default()).expect("find");
        println!("{id}: {w} x {h}, {} face(s)", faces.len());
        for (k, d) in faces.iter().enumerate() {
            let face = Face {
                rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
                scale: d.scale,
            };
            let mut cache = RetouchCache::new(1, 1);
            prepare_skin(
                &image,
                &[(face, SkinSettings::default())],
                &models,
                &mut cache,
            )
            .expect("prepare");
            // The crop: the face and 15 % around it. The mouth: the lower middle of the face rectangle.
            let r = face.rect;
            let (fx, fy, fw, fh) = (
                r.x_px as usize,
                r.y_px as usize,
                r.width_px as usize,
                r.height_px as usize,
            );
            let pad = fw * 15 / 100;
            let x0 = fx.saturating_sub(pad);
            let y0 = fy.saturating_sub(pad);
            let face_crop = (
                x0,
                y0,
                (fx + fw + pad).min(w) - x0,
                (fy + fh + pad).min(h) - y0,
            );
            let mouth_crop = {
                let (mx, my) = (fx + fw * 25 / 100, fy + fh * 60 / 100);
                (mx, my, fw * 50 / 100, fh * 30 / 100)
            };
            for (name, s) in &settings {
                let out = retouch_skin(
                    &image,
                    (w as u32, h as u32),
                    &[(face, *s)],
                    1.0,
                    &models,
                    &mut cache,
                    None,
                )
                .expect("retouch");
                for (what, crop) in [("", face_crop), ("mouth_", mouth_crop)] {
                    write_png(
                        &out_dir.join(format!("{tag}{id}_{k}_{what}{name}.png")),
                        crop.2,
                        crop.3,
                        &crop_srgb(&out, crop),
                    );
                }
            }
            println!("  face {k}: {}x{} px done", fw, fh);
        }
    }
}
