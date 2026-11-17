//! Dev tool (`docs/27` S-17): contact sheets of the teeth on public portraits, the colour gates of S-16 against SAM 2.1.
//!
//! ```text
//! cargo run -p ym-core --example teeth_sheet -- <data root> <out dir> [--screen] [--faces N] <photo.jpg>...
//! ```
//!
//! Every photo is brought to the analysis size of the product (1400 px on the long edge), its faces are found, and for each
//! face with an open mouth one PNG row shows the mouth at 2x, left to right: the original; the old colour-gate mask as an
//! overlay (magenta); the SAM mask as an overlay (green); Whitening 60 with the old mask and with SAM; Whitening 100 with the
//! old mask and with SAM. `--screen` only prints, per face, its size and how open its mouth is (no network runs but the finder
//! and the mesh). The lines on standard output are the numbers of `docs/27` S-17 as built: the areas of both masks and the time
//! SAM took.

#[path = "teeth_support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use support::{decode, old_teeth_mask, shrink, to_linear_awg4};
use ym_core::pipeline::object::ObjectModel;
use ym_core::pipeline::teeth::{SamTeeth, SharedTeeth};
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::face_parts::mouth_interior_mask;
use ym_retouch::mediapipe::{AWG4_TO_REC709, MediaPipeFaces};
use ym_retouch::{
    BoundedMask, ClassicalBeautyMarks, ClassicalProposer, Face, LinearAwg4Image, Models, Rect,
    RetouchCache, SkinSettings, TeethSegmenter, face_extent_px, prepare_skin, retouch_skin,
};

/// The colour-gate segmenter of S-16, for the "old" column: the mask the gates made, before the final feather the pipeline now
/// applies to every segmenter's answer.
struct OldColourTeeth(Arc<MediaPipeFaces>);

impl TeethSegmenter for OldColourTeeth {
    fn teeth(
        &self,
        image: &LinearAwg4Image,
        face: &Face,
    ) -> ym_retouch::Result<Option<BoundedMask>> {
        let size = (image.width_px, image.height_px);
        let Some(a) = self.0.analyse(image, face)? else {
            return Ok(None);
        };
        let Some(mouth) = mouth_interior_mask(&a.geometry, size)? else {
            return Ok(None);
        };
        let extent = face_extent_px(size, face);
        let m = old_teeth_mask(&mouth.to_full(size)?, image, extent, false);
        Ok(BoundedMask::from_full(&m))
    }
}

fn srgb(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let e = if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0).round() as u8
}

/// The tile `(x, y, w, h)` of `img` at 2x (bilinear), as RGB bytes, with `overlay` (a mask and a colour) blended on top.
fn tile(
    img: &LinearAwg4Image,
    (x0, y0, cw, ch): (i64, i64, usize, usize),
    overlay: Option<(&BoundedMask, [f32; 3])>,
) -> Vec<u8> {
    let (w, h) = (img.width_px as i64, img.height_px as i64);
    let m = AWG4_TO_REC709;
    let at = |x: i64, y: i64| -> [f32; 3] {
        let p = img.data[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];
        [0, 1, 2].map(|r| (m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2]).clamp(0.0, 1.0))
    };
    let (tw, th) = (cw * 2, ch * 2);
    let mut out = Vec::with_capacity(tw * th * 3);
    for j in 0..th {
        let sy = y0 as f64 + (j as f64 + 0.5) / 2.0 - 0.5;
        let (iy, fy) = (sy.floor() as i64, (sy - sy.floor()) as f32);
        for i in 0..tw {
            let sx = x0 as f64 + (i as f64 + 0.5) / 2.0 - 0.5;
            let (ix, fx) = (sx.floor() as i64, (sx - sx.floor()) as f32);
            let (a, b, c, d) = (
                at(ix, iy),
                at(ix + 1, iy),
                at(ix, iy + 1),
                at(ix + 1, iy + 1),
            );
            let mut px = [0.0f32; 3];
            for k in 0..3 {
                px[k] = (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy)
                    + (c[k] * (1.0 - fx) + d[k] * fx) * fy;
            }
            let mut e = px.map(srgb).map(|v| f32::from(v) / 255.0);
            if let Some((mask, colour)) = overlay {
                let v = f32::from(mask.value_at(sx.round() as i64, sy.round() as i64)) / 255.0;
                for k in 0..3 {
                    e[k] = e[k] * (1.0 - 0.65 * v) + colour[k] * 0.65 * v;
                }
            }
            out.extend(e.map(|v| (v * 255.0).round() as u8));
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

fn area(m: &Option<BoundedMask>) -> (usize, usize) {
    m.as_ref().map_or((0, 0), |m| {
        (
            m.mask.data.iter().filter(|&&v| v >= 128).count(),
            m.mask.data.iter().map(|&v| usize::from(v)).sum::<usize>() / 255,
        )
    })
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let screen = args
        .iter()
        .position(|a| a == "--screen")
        .map(|i| args.remove(i))
        .is_some();
    let per_photo: usize = args
        .iter()
        .position(|a| a == "--faces")
        .map(|i| {
            let v = args[i + 1].parse().expect("--faces");
            args.drain(i..i + 2);
            v
        })
        .unwrap_or(1);
    let data_root = PathBuf::from(args.remove(0));
    let out_dir = PathBuf::from(args.remove(0));
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let mp = data_root.join("models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&mp).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&mp).expect("mesh and segmenter weights");
    let sam = ObjectModel::probe(&data_root).expect("the SAM weights");
    let seg = Arc::new(SamTeeth::new(Arc::clone(&engine), sam));
    let models = |teeth: Box<dyn TeethSegmenter>| Models {
        parser: Box::new(engine.parser()),
        mesh: Box::new(engine.mesh()),
        beauty_marks: Some(Box::new(ClassicalBeautyMarks)),
        proposer: Box::new(ClassicalProposer::new(450.0)),
        teeth: Some(teeth),
    };
    let old_models = models(Box::new(OldColourTeeth(Arc::clone(&engine))));
    let new_models = models(Box::new(SharedTeeth(Arc::clone(&seg))));
    for photo in &args {
        let path = Path::new(photo);
        let id = path
            .file_stem()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        let (w0, h0, rgb0) = decode(path);
        let (w, h, rgb) = shrink(w0, h0, &rgb0, 1400);
        let image = to_linear_awg4(w, h, &rgb);
        let src = Rgb8::new(w, h, &rgb).expect("picture");
        let mut faces = finder.find(&src, &FinderParams::default()).expect("find");
        faces.sort_by_key(|d| std::cmp::Reverse(d.rect.width_px));
        println!("{id}: {w0}x{h0} -> {w}x{h}, {} face(s)", faces.len());
        for (k, d) in faces.iter().take(per_photo).enumerate() {
            let face = Face {
                rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
                scale: d.scale,
            };
            let size = (w as u32, h as u32);
            let Some(a) = engine.analyse(&image, &face).expect("analyse") else {
                println!("  face {k}: no mesh");
                continue;
            };
            let Some(mouth) = mouth_interior_mask(&a.geometry, size).expect("mouth") else {
                println!("  face {k}: {} px, mouth closed", d.rect.width_px);
                continue;
            };
            let mouth_area = mouth.mask.data.iter().filter(|&&v| v >= 128).count();
            let lips = a.geometry.lip_polygon();
            let (lx0, lx1) = lips.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(p[0]), hi.max(p[0]))
            });
            let (ly0, ly1) = lips.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(p[1]), hi.max(p[1]))
            });
            println!(
                "  face {k}: {} px, mouth open {:.2} of the lips' width, opening {mouth_area} px",
                d.rect.width_px,
                (ly1 - ly0) / (lx1 - lx0)
            );
            if screen {
                continue;
            }
            let t = Instant::now();
            let mut old_cache = RetouchCache::new(1, 1);
            prepare_skin(
                &image,
                &[(face, SkinSettings::default())],
                &old_models,
                &mut old_cache,
            )
            .expect("prepare");
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut new_cache = RetouchCache::new(1, 1);
            prepare_skin(
                &image,
                &[(face, SkinSettings::default())],
                &new_models,
                &mut new_cache,
            )
            .expect("prepare");
            let new_ms = t.elapsed().as_secs_f64() * 1e3;
            let old_mask = old_cache.masks(0).and_then(|m| m.teeth.clone());
            let new_mask = new_cache.masks(0).and_then(|m| m.teeth.clone());
            let (oa, _) = area(&old_mask);
            let (na, _) = area(&new_mask);
            println!(
                "    teeth: old {oa} px ({:.0} % of the opening), SAM {na} px ({:.0} %); prepare {old_ms:.0} ms with the gates, {new_ms:.0} ms with SAM",
                100.0 * oa as f64 / mouth_area as f64,
                100.0 * na as f64 / mouth_area as f64
            );
            let render = |models: &Models, cache: &mut RetouchCache, amount: f32| {
                retouch_skin(
                    &image,
                    size,
                    &[(
                        face,
                        SkinSettings {
                            teeth_whitening: amount,
                            ..SkinSettings::default()
                        },
                    )],
                    1.0,
                    models,
                    cache,
                    None,
                )
                .expect("retouch")
            };
            let (c0, c1) = (((lx0 + lx1) / 2.0), ((ly0 + ly1) / 2.0));
            let (cw, ch) = (
                ((lx1 - lx0) * 1.5).ceil() as usize,
                ((lx1 - lx0) * 0.9).ceil() as usize,
            );
            let win = (
                (c0 - cw as f64 / 2.0).round() as i64,
                (c1 - ch as f64 / 2.0).round() as i64,
                cw,
                ch,
            );
            let none = BoundedMask::new(
                Rect::new(0, 0, 1, 1),
                ym_retouch::Mask::zeros(1, 1).expect("mask"),
            )
            .expect("mask");
            let mut tiles = vec![tile(&image, win, None)];
            tiles.push(tile(
                &image,
                win,
                Some((old_mask.as_ref().unwrap_or(&none), [1.0, 0.1, 0.9])),
            ));
            tiles.push(tile(
                &image,
                win,
                Some((new_mask.as_ref().unwrap_or(&none), [0.1, 1.0, 0.2])),
            ));
            for amount in [60.0, 100.0] {
                let o = render(&old_models, &mut old_cache, amount);
                let n = render(&new_models, &mut new_cache, amount);
                tiles.push(tile(&o, win, None));
                tiles.push(tile(&n, win, None));
            }
            let (tw, th) = (cw * 2, ch * 2);
            let gap = 4;
            let sheet_w = tiles.len() * tw + (tiles.len() - 1) * gap;
            let mut sheet = vec![40u8; sheet_w * th * 3];
            for (n, t) in tiles.iter().enumerate() {
                for y in 0..th {
                    let dst = (y * sheet_w + n * (tw + gap)) * 3;
                    sheet[dst..dst + tw * 3].copy_from_slice(&t[y * tw * 3..(y + 1) * tw * 3]);
                }
            }
            write_png(&out_dir.join(format!("{id}_f{k}.png")), sheet_w, th, &sheet);
        }
    }
}
