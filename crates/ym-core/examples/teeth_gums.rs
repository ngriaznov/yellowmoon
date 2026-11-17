//! Dev tool (`docs/27` S-18): measures how much of the teeth mask lies on the gums, and compares the ways to bring it down.
//!
//! ```text
//! cargo run -p ym-core --example teeth_gums -- dump  <data root> <images dir> <full labels dir> <dump dir> [--from N] [--to N] [--step N]
//! cargo run -p ym-core --example teeth_gums -- score <data root> <images dir> <full labels dir> <dump dir> <out.tsv>
//! ```
//!
//! `<full labels dir>/<id>.png` holds the class map of the public EasyPortrait dataset (8 = teeth, 7 = lips; for local evaluation
//! only, never committed or used to train anything). `dump` encodes the mouth once per photo and writes SAM's four candidates for
//! every prompt variant to `<dump dir>`; `score` reads them back and measures every selection method on the final mask (the
//! one the whitening uses), so a rule is tried without running the encoder again.

// A dev tool for one evaluation: it keeps the variants that were tried, used or not.
#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::collapsible_if
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "teeth_support/mod.rs"]
mod support;

use support::{decode, to_linear_awg4};
use ym_core::pipeline::object::ObjectModel;
use ym_core::pipeline::teeth::SamTeeth;
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::face_parts::mouth_interior_mask;
use ym_retouch::mediapipe::{FaceAnalysis, MediaPipeFaces};
use ym_retouch::teeth::{CAVITY_ASPECT, CAVITY_COVER, MAX_SPILL, MIN_COVER};
use ym_retouch::teeth::{
    CROP_FACTOR, GRID_PX, GumShape, MouthCrop, PromptShape, PromptStyle, Regions, TeethPrompt,
    crop_mask, gum_prompt, lip_gap, select_candidate, teeth_prompt_with,
};
use ym_retouch::{Face, LinearAwg4Image, Mask, Rect, teeth_from_segment};

const TEETH: u8 = 8;
const N: usize = GRID_PX * GRID_PX;

/// The prompt variants dumped: name and how the prompt is made.
fn shape_variants() -> Vec<(&'static str, PromptShape)> {
    let d = PromptShape::DEFAULT;
    vec![
        ("base", d),
        (
            "neg015",
            PromptShape {
                band_negatives: Some(0.015),
                ..d
            },
        ),
        (
            "neg03",
            PromptShape {
                band_negatives: Some(0.03),
                ..d
            },
        ),
        (
            "deep",
            PromptShape {
                upper_row: 0.4,
                row_cap: 0.10,
                ..d
            },
        ),
    ]
}

fn gum_variants() -> Vec<(&'static str, GumShape)> {
    let d = GumShape::DEFAULT;
    vec![("gumq", d), ("gumq2", GumShape { depth: 0.03, ..d })]
}

struct Located {
    image: LinearAwg4Image,
    label: Vec<u8>,
    face: Face,
    analysis: Arc<FaceAnalysis>,
    mouth: Mask,
    window: (usize, usize, usize, usize),
    extent: f64,
}

fn read_label(path: &Path, w: usize, h: usize) -> Vec<u8> {
    let f = std::fs::File::open(path).expect("label");
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut r = dec.read_info().expect("png header");
    let mut buf = vec![0u8; r.output_buffer_size().expect("png size")];
    let info = r.next_frame(&mut buf).expect("png frame");
    assert_eq!((info.width as usize, info.height as usize), (w, h));
    buf
}

fn locate(
    finder: &FaceFinder,
    engine: &MediaPipeFaces,
    images: &Path,
    labels: &Path,
    id: &str,
) -> Result<Located, String> {
    let (w, h, rgb) = decode(&images.join(format!("{id}.jpg")));
    let lp = labels.join(format!("{id}.png"));
    let unlabelled = !lp.exists();
    let label = if unlabelled {
        vec![0u8; w * h]
    } else {
        read_label(&lp, w, h)
    };
    let image = to_linear_awg4(w, h, &rgb);
    let src = Rgb8::new(w, h, &rgb).expect("picture");
    let found = finder.find(&src, &FinderParams::default()).expect("find");
    let mut best: Option<(usize, Face)> = None;
    for d in &found {
        let (x0, y0) = (d.rect.x_px.max(0) as usize, d.rect.y_px.max(0) as usize);
        let x1 = ((d.rect.x_px + d.rect.width_px as i32).max(0) as usize).min(w);
        let y1 = ((d.rect.y_px + d.rect.height_px as i32).max(0) as usize).min(h);
        let n = (y0..y1)
            .map(|y| (x0..x1).filter(|&x| label[y * w + x] == TEETH).count())
            .sum::<usize>();
        if best.as_ref().is_none_or(|b| n > b.0) {
            best = Some((
                n,
                Face {
                    rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
                    scale: d.scale,
                },
            ));
        }
    }
    if unlabelled {
        best = found.first().map(|d| {
            (
                1,
                Face {
                    rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
                    scale: d.scale,
                },
            )
        });
    }
    let Some((_, face)) = best.filter(|b| b.0 > 0) else {
        return Err("no face holds the teeth".into());
    };
    let size = (image.width_px, image.height_px);
    let extent = (f64::from(face.scale) * w.max(h) as f64).trunc();
    let r = face.rect;
    let grow = (0.25 * f64::from(r.width_px.max(r.height_px))) as i64;
    let window = (
        (i64::from(r.x_px) - grow).max(0) as usize,
        (i64::from(r.y_px) - grow).max(0) as usize,
        ((i64::from(r.x_px) + i64::from(r.width_px) + grow).max(0) as usize).min(w),
        ((i64::from(r.y_px) + i64::from(r.height_px) + grow).max(0) as usize).min(h),
    );
    let Some(analysis) = engine.analyse(&image, &face).expect("analyse") else {
        return Err("no mesh".into());
    };
    let Some(mouth) = mouth_interior_mask(&analysis.geometry, size).expect("mouth") else {
        return Err("closed mouth by the mesh".into());
    };
    let mouth = mouth.to_full(size).expect("mouth");
    Ok(Located {
        image,
        label,
        face,
        analysis,
        mouth,
        window,
        extent,
    })
}

fn quantise(logits: &[f32]) -> impl Iterator<Item = u8> + '_ {
    logits
        .iter()
        .map(|&v| ((v * 4.0).round().clamp(-127.0, 127.0) as i8) as u8)
}

fn write_dump(path: &Path, decoded: &ym_ml::sam2::Decoded) {
    let mut out: Vec<u8> = Vec::new();
    let k = decoded.candidates();
    out.push(k as u8);
    for i in 0..k {
        out.extend_from_slice(&decoded.iou[i].to_le_bytes());
    }
    for i in 0..k {
        out.extend(quantise(decoded.candidate_logits(i)));
    }
    std::fs::write(path, out).expect("write dump");
}

/// A dump read back: the logits are the quantised ones (a quarter of a logit step).
struct Cands {
    iou: Vec<f32>,
    logits: Vec<Vec<f32>>,
}

fn read_dump(path: &Path) -> Option<Cands> {
    let b = std::fs::read(path).ok()?;
    let k = usize::from(b[0]);
    let iou = (0..k)
        .map(|i| f32::from_le_bytes(b[1 + 4 * i..5 + 4 * i].try_into().unwrap()))
        .collect();
    let base = 1 + 4 * k;
    let logits = (0..k)
        .map(|i| {
            b[base + i * N..base + (i + 1) * N]
                .iter()
                .map(|&v| f32::from(v as i8) / 4.0)
                .collect()
        })
        .collect();
    Some(Cands { iou, logits })
}

fn dump(args: &[String], from: usize, to: usize, step: usize) {
    let (data_root, images, labels, out) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
    );
    std::fs::create_dir_all(&out).expect("dump dir");
    let mp = data_root.join("models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&mp).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&mp).expect("mesh weights");
    let sam = ObjectModel::probe(&data_root).expect("the SAM weights");
    let seg = SamTeeth::new(Arc::clone(&engine), Arc::clone(&sam));
    for id in ids(&images)
        .iter()
        .step_by(step)
        .skip(from)
        .take(to.saturating_sub(from))
    {
        if out.join(format!("{id}.base")).exists() {
            continue;
        }
        let loc = match locate(&finder, &engine, &images, &labels, id) {
            Ok(l) => l,
            Err(why) => {
                std::fs::write(out.join(format!("{id}.skip")), why).expect("skip");
                continue;
            }
        };
        let t = std::time::Instant::now();
        let Some(enc) = seg
            .encode(&loc.image, &loc.face, CROP_FACTOR)
            .expect("encode")
        else {
            std::fs::write(out.join(format!("{id}.skip")), "no mouth").expect("skip");
            continue;
        };
        let enc_ms = t.elapsed().as_millis();
        let mut chosen_logits: Option<Vec<f32>> = None;
        for (name, shape) in shape_variants() {
            let Some(prompt) = teeth_prompt_with(
                &loc.analysis.geometry,
                &enc.crop,
                PromptStyle::Points,
                &shape,
            ) else {
                continue;
            };
            let dec = seg.decode_prompt(&enc, &prompt).expect("decode");
            write_dump(&out.join(format!("{id}.{name}")), &dec);
            if name == "base" {
                let cands: Vec<(&[f32], f32)> = (0..dec.candidates())
                    .map(|i| (dec.candidate_logits(i), dec.iou[i]))
                    .collect();
                if let Ok(i) = select_candidate(&cands, &enc.regions).0 {
                    chosen_logits = Some(dec.candidate_logits(i).to_vec());
                }
            }
        }
        if let Some(m1) = chosen_logits {
            for (name, shape) in gum_variants() {
                if let Some(prompt) = gum_prompt(&loc.analysis.geometry, &enc.crop, &m1, &shape) {
                    let dec = seg.decode_prompt(&enc, &prompt).expect("decode");
                    write_dump(&out.join(format!("{id}.{name}")), &dec);
                }
            }
        }
        eprintln!("{id} dumped, encode {enc_ms} ms");
    }
}

/// Screening of unlabelled photos: the mouth of the face the finder ranks first, as a PNG, for the photos whose mesh opens.
fn crops(args: &[String]) {
    let (data_root, images, out) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
    );
    std::fs::create_dir_all(&out).expect("out");
    let mp = data_root.join("models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&mp).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&mp).expect("mesh weights");
    for id in ids(&images) {
        let (w, h, rgb) = decode(&images.join(format!("{id}.jpg")));
        let image = to_linear_awg4(w, h, &rgb);
        let src = Rgb8::new(w, h, &rgb).expect("picture");
        let found = finder.find(&src, &FinderParams::default()).expect("find");
        let Some(d) = found.first() else { continue };
        let face = Face {
            rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
            scale: d.scale,
        };
        let Ok(Some(a)) = engine.analyse(&image, &face) else {
            continue;
        };
        let Some(crop) = MouthCrop::from_geometry(&a.geometry, 1.75) else {
            continue;
        };
        let Some(reg) = Regions::from_geometry(
            &a.geometry,
            &MouthCrop::from_geometry(&a.geometry, CROP_FACTOR).unwrap(),
        ) else {
            continue;
        };
        let side = 200usize;
        let mut o = vec![0u8; side * side * 3];
        for j in 0..side {
            for i in 0..side {
                let x = (crop.x_px + (i as f64 + 0.5) / side as f64 * crop.side_px)
                    .clamp(0.0, w as f64 - 1.0) as usize;
                let y = (crop.y_px + (j as f64 + 0.5) / side as f64 * crop.side_px)
                    .clamp(0.0, h as f64 - 1.0) as usize;
                o[(j * side + i) * 3..(j * side + i) * 3 + 3]
                    .copy_from_slice(&rgb[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        let f =
            std::fs::File::create(out.join(format!("{id}_a{:.2}.png", reg.aspect))).expect("png");
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), side as u32, side as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header()
            .expect("header")
            .write_image_data(&o)
            .expect("data");
    }
}

fn ids(images: &Path) -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(images)
        .expect("images dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();
    ids.sort();
    ids
}

/// What a method gives for a photo: SAM's logits for the teeth on the crop's grid, or nothing.
type Method = Box<dyn Fn(&Cands, &Located, &EncodedGeo, &Dumps) -> Option<Vec<f32>>>;

struct EncodedGeo {
    crop: MouthCrop,
    regions: Regions,
}

struct Dumps {
    by_name: BTreeMap<String, Cands>,
}

fn choose(c: &Cands, regions: &Regions) -> Option<usize> {
    let cands: Vec<(&[f32], f32)> = c
        .logits
        .iter()
        .zip(&c.iou)
        .map(|(l, &i)| (&l[..], i))
        .collect();
    select_candidate(&cands, regions).0.ok()
}

fn score(args: &[String]) {
    let (data_root, images, labels, dumps_dir, out) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
        PathBuf::from(&args[4]),
    );
    let mp = data_root.join("models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&mp).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&mp).expect("mesh weights");
    let methods = methods();
    let mut text = String::new();
    for id in ids(&images) {
        if !dumps_dir.join(format!("{id}.base")).exists() {
            continue;
        }
        let loc = locate(&finder, &engine, &images, &labels, &id).expect("locate");
        let size = (loc.image.width_px, loc.image.height_px);
        let enc = {
            let crop = MouthCrop::from_geometry(&loc.analysis.geometry, CROP_FACTOR).unwrap();
            let regions = Regions::from_geometry(&loc.analysis.geometry, &crop).unwrap();
            EncodedGeo { crop, regions }
        };
        let mut by_name = BTreeMap::new();
        for (name, _) in shape_variants()
            .iter()
            .map(|(n, _)| (*n, ()))
            .chain(gum_variants().iter().map(|(n, _)| (*n, ())))
        {
            if let Some(c) = read_dump(&dumps_dir.join(format!("{id}.{name}"))) {
                by_name.insert(name.to_owned(), c);
            }
        }
        let dumps = Dumps { by_name };
        let base = &dumps.by_name["base"];
        let band = gum_band(&loc);
        writeln!(
            text,
            "{id}\t-\tinfo\tbandh_rel {:.4}\tband_px {}\tteeth_px {}\taspect {:.3}",
            band.rel, band.n_band, band.n_teeth, enc.regions.aspect
        )
        .unwrap();
        if let Some(dir) = std::env::var_os("YM_GRIDS") {
            grid_png(&PathBuf::from(dir), &id, &images, &enc);
        }
        let mut shown: Vec<(String, Option<Mask>)> = Vec::new();
        for (name, method) in &methods {
            let seg = method(base, &loc, &enc, &dumps);
            let line = match seg {
                None => "refused".to_owned(),
                Some(logits) => {
                    let mask = crop_mask(&logits, &enc.crop, size)
                        .expect("crop mask")
                        .map_or_else(
                            || Mask::zeros(size.0, size.1).unwrap(),
                            |m| m.to_full(size).unwrap(),
                        );
                    let fin = teeth_from_segment(&loc.mouth, &mask, loc.extent).expect("teeth");
                    let line = format!(
                        "{}\tlipgap {:.4}",
                        metrics(&fin, &loc, &band),
                        lip_gap(&logits, &enc.regions)
                    );
                    if name == "current" {
                        if let Some(dir) = std::env::var_os("YM_SHEETS") {
                            whiten_sheet(&PathBuf::from(dir), &id, &fin, &loc, &enc);
                        }
                    }
                    if name == "current" && std::env::var_os("YM_WHITEN").is_some() {
                        whiten_stats(&id, &fin, &loc, &band);
                    }
                    if name == "current" && std::env::var_os("YM_PINK").is_some() {
                        pink_stats(&id, &fin, &loc, &band);
                    }
                    shown.push((name.clone(), Some(fin)));
                    line
                }
            };
            writeln!(text, "{id}\t{name}\t{line}").unwrap();
        }
        if let Some(dir) = std::env::var_os("YM_OVERLAYS") {
            if band.rel >= 0.02 || band.n_teeth == 0 {
                overlay(&PathBuf::from(dir), &id, &images, &loc, &enc, &shown);
            }
        }
        eprintln!("{id} scored");
    }
    std::fs::write(out, text).expect("tsv");
}

struct Band {
    /// The gum band: opening pixels above the teeth label in a column that has teeth below.
    mask: Vec<bool>,
    n_band: usize,
    n_teeth: usize,
    /// The band's mean height over the opening's width.
    rel: f64,
}

fn gum_band(loc: &Located) -> Band {
    let (w, h) = (loc.image.width_px as usize, loc.image.height_px as usize);
    let (x0, y0, x1, y1) = loc.window;
    let mut mask = vec![false; w * h];
    let (mut n_band, mut n_teeth, mut cols) = (0usize, 0usize, 0usize);
    let (mut ox0, mut ox1) = (usize::MAX, 0usize);
    for x in x0..x1 {
        let mut top = None;
        for y in y0..y1 {
            if loc.label[y * w + x] == TEETH {
                n_teeth += 1;
                top.get_or_insert(y);
            }
            if loc.mouth.data[y * w + x] >= 128 {
                ox0 = ox0.min(x);
                ox1 = ox1.max(x);
            }
        }
        if let Some(t) = top {
            let mut any = false;
            for y in y0..t {
                if loc.mouth.data[y * w + x] >= 128 && loc.label[y * w + x] != TEETH {
                    mask[y * w + x] = true;
                    n_band += 1;
                    any = true;
                }
            }
            cols += usize::from(any);
        }
    }
    let width = ox1.saturating_sub(ox0).max(1) as f64;
    let _ = cols;
    // The mean over the columns that have teeth (not only those with a band), in the opening's widths.
    let teeth_cols = (x0..x1)
        .filter(|&x| (y0..y1).any(|y| loc.label[y * w + x] == TEETH))
        .count()
        .max(1);
    Band {
        mask,
        n_band,
        n_teeth,
        rel: n_band as f64 / teeth_cols as f64 / width,
    }
}

/// Panels of the mouth: the photo, the label's teeth (green) with the opening's outline (blue), and each method's mask (red).
fn overlay(
    dir: &Path,
    id: &str,
    images: &Path,
    loc: &Located,
    enc: &EncodedGeo,
    shown: &[(String, Option<Mask>)],
) {
    let (w, h, rgb) = decode(&images.join(format!("{id}.jpg")));
    let c = &enc.crop;
    let side = (c.side_px / 2.0).round() as i64;
    let (cx, cy) = (c.x_px + c.side_px / 2.0, c.y_px + c.side_px / 2.0);
    let (x0, y0) = (cx.round() as i64 - side / 2, cy.round() as i64 - side / 2);
    let scale = (512 / side.max(1)).max(1) as usize;
    let pick: Vec<&(String, Option<Mask>)> = shown
        .iter()
        .filter(|(n, _)| {
            ["current", "select_mu5", "gumq_i0.7_s0.25", "gumq_i0.5_s0.4"].contains(&n.as_str())
        })
        .collect();
    let panels = 2 + pick.len();
    let (pw, ph) = (side as usize * scale, side as usize * scale);
    let mut out = vec![0u8; pw * panels * ph * 3];
    for k in 0..panels {
        for j in 0..ph {
            for i in 0..pw {
                let x = (x0 + (i / scale) as i64).clamp(0, w as i64 - 1) as usize;
                let y = (y0 + (j / scale) as i64).clamp(0, h as i64 - 1) as usize;
                let idx = y * w + x;
                let mut px = [rgb[idx * 3], rgb[idx * 3 + 1], rgb[idx * 3 + 2]];
                let blend = |px: &mut [u8; 3], col: [u8; 3]| {
                    for q in 0..3 {
                        px[q] = ((u16::from(px[q]) + u16::from(col[q])) / 2) as u8;
                    }
                };
                if k == 1 {
                    if loc.label[idx] == TEETH {
                        blend(&mut px, [0, 255, 0]);
                    }
                    if loc.mouth.data[idx] >= 128 && loc.label[idx] != TEETH {
                        blend(&mut px, [0, 0, 255]);
                    }
                } else if k >= 2 {
                    if let Some(m) = &pick[k - 2].1 {
                        if m.data[idx] >= 128 {
                            blend(&mut px, [255, 0, 0]);
                        }
                    }
                }
                let o = (j * pw * panels + k * pw + i) * 3;
                out[o..o + 3].copy_from_slice(&px);
            }
        }
    }
    std::fs::create_dir_all(dir).expect("overlay dir");
    let f = std::fs::File::create(dir.join(format!("{id}.png"))).expect("png");
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), (pw * panels) as u32, ph as u32);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()
        .expect("header")
        .write_image_data(&out)
        .expect("data");
}

/// The crop on SAM's grid as a PNG (red channel: the photo's luma is not needed, so RGB), and the opening in a second file.
fn grid_png(dir: &Path, id: &str, images: &Path, enc: &EncodedGeo) {
    let (w, h, rgb) = decode(&images.join(format!("{id}.jpg")));
    std::fs::create_dir_all(dir).expect("dir");
    let mut o = vec![0u8; N * 3];
    let mut op = vec![0u8; N];
    for j in 0..GRID_PX {
        for i in 0..GRID_PX {
            let x = (enc.crop.x_px + (i as f64 + 0.5) / GRID_PX as f64 * enc.crop.side_px)
                .clamp(0.0, w as f64 - 1.0) as usize;
            let y = (enc.crop.y_px + (j as f64 + 0.5) / GRID_PX as f64 * enc.crop.side_px)
                .clamp(0.0, h as f64 - 1.0) as usize;
            o[(j * GRID_PX + i) * 3..(j * GRID_PX + i) * 3 + 3]
                .copy_from_slice(&rgb[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            op[j * GRID_PX + i] = if enc.regions.opening_at(j * GRID_PX + i) {
                255
            } else {
                0
            };
        }
    }
    let save = |name: String, ct: png::ColorType, data: &[u8]| {
        let f = std::fs::File::create(dir.join(name)).expect("png");
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), GRID_PX as u32, GRID_PX as u32);
        e.set_color(ct);
        e.set_depth(png::BitDepth::Eight);
        e.write_header()
            .expect("header")
            .write_image_data(data)
            .expect("data");
    };
    save(format!("{id}.crop.png"), png::ColorType::Rgb, &o);
    save(format!("{id}.open.png"), png::ColorType::Grayscale, &op);
}

/// Pinkness of the pixels in the mask relative to the mask's own core: `(R/G) / (R0/G0)`; percentiles over teeth-label pixels and gum-band pixels.
fn pink_stats(id: &str, fin: &Mask, loc: &Located, band: &Band) {
    let w = loc.image.width_px as usize;
    let rg = |i: usize| {
        let p = loc.image.data[i];
        f64::from(p[0].max(1e-4)) / f64::from(p[1].max(1e-4))
    };
    let mut core: Vec<f64> = (0..fin.data.len())
        .filter(|&i| fin.data[i] >= 230)
        .map(rg)
        .collect();
    if core.len() < 20 {
        return;
    }
    core.sort_unstable_by(f64::total_cmp);
    let r0 = core[core.len() / 2];
    let pct = |mut v: Vec<f64>| -> String {
        if v.len() < 5 {
            return format!("n={}", v.len());
        }
        v.sort_unstable_by(f64::total_cmp);
        let q = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
        format!(
            "n={} p10 {:.2} p50 {:.2} p90 {:.2} p99 {:.2}",
            v.len(),
            q(0.1),
            q(0.5),
            q(0.9),
            q(0.99)
        )
    };
    let _ = w;
    let teeth: Vec<f64> = (0..fin.data.len())
        .filter(|&i| fin.data[i] >= 128 && loc.label[i] == TEETH)
        .map(|i| rg(i) / r0)
        .collect();
    let gum: Vec<f64> = (0..fin.data.len())
        .filter(|&i| fin.data[i] >= 128 && band.mask[i])
        .map(|i| rg(i) / r0)
        .collect();
    println!(
        "PINK {id} r0 {r0:.2} teeth[{}] gum[{}]",
        pct(teeth),
        pct(gum)
    );
}

/// The mouth at 2x: the original, the mask, Whitening 60 and 100 as the pipeline does it, and with the pink gate.
fn whiten_sheet(dir: &Path, id: &str, fin: &Mask, loc: &Located, enc: &EncodedGeo) {
    let (w, h) = (loc.image.width_px as usize, loc.image.height_px as usize);
    let c = &enc.crop;
    let side = (c.side_px / 2.0).round() as i64;
    let (x0, y0) = (
        (c.x_px + c.side_px / 2.0).round() as i64 - side / 2,
        (c.y_px + c.side_px / 2.0).round() as i64 - side / 2,
    );
    let scale = (480 / side.max(1)).max(2) as usize;
    let px = |i: usize| {
        let p = loc.image.data[i];
        [
            f64::from(p[0]).max(0.0),
            f64::from(p[1]).max(0.0),
            f64::from(p[2]).max(0.0),
        ]
    };
    let ls: Vec<f64> = {
        let mut v: Vec<f64> = (0..fin.data.len())
            .filter(|&i| fin.data[i] != 0)
            .map(|i| {
                let p = px(i);
                (p[0] + p[1] + p[2]) / 3.0
            })
            .collect();
        v.sort_unstable_by(f64::total_cmp);
        v
    };
    if ls.is_empty() {
        return;
    }
    let l_max = ls[((0.99 * ls.len() as f64).ceil() as usize).clamp(1, ls.len()) - 1];
    let mut rgs: Vec<f64> = (0..fin.data.len())
        .filter(|&i| fin.data[i] >= 230)
        .map(|i| {
            let p = px(i);
            p[0] / p[1].max(1e-4)
        })
        .collect();
    rgs.sort_unstable_by(f64::total_cmp);
    let r0 = if rgs.is_empty() {
        1.0
    } else {
        rgs[rgs.len() / 2].max(1e-4)
    };
    let gate = |p: [f64; 3]| {
        let t = (((p[0] / p[1].max(1e-4)) / r0 - 1.15) / 0.30).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    // AWG4 linear -> Rec.709 linear -> sRGB.
    let m = ym_retouch::mediapipe::AWG4_TO_REC709;
    let enc8 = |p: [f64; 3]| -> [u8; 3] {
        let mut o = [0u8; 3];
        for r in 0..3 {
            let v =
                (f64::from(m[r][0]) * p[0] + f64::from(m[r][1]) * p[1] + f64::from(m[r][2]) * p[2])
                    .clamp(0.0, 1.0);
            let e = if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            o[r] = (e * 255.0).round() as u8;
        }
        o
    };
    let variants: [(Option<f64>, bool); 6] = [
        (None, false),
        (None, true),
        (Some(0.6), false),
        (Some(1.0), false),
        (Some(0.6), true),
        (Some(1.0), true),
    ];
    let (pw, ph) = (side as usize * scale, side as usize * scale);
    let mut out = vec![0u8; pw * 6 * ph * 3];
    for (k, (tw, gated)) in variants.iter().enumerate() {
        for j in 0..ph {
            for i in 0..pw {
                let x = (x0 + (i / scale) as i64).clamp(0, w as i64 - 1) as usize;
                let y = (y0 + (j / scale) as i64).clamp(0, h as i64 - 1) as usize;
                let idx = y * w + x;
                let src = px(idx);
                let mv = f64::from(fin.data[idx]) / 255.0;
                let mut o = match tw {
                    None => src,
                    Some(tw) => {
                        let g = if *gated { gate(src) } else { 1.0 };
                        if fin.data[idx] == 0 {
                            src
                        } else {
                            ym_retouch::teeth_pixel(src, *tw, mv * g, l_max)
                        }
                    }
                };
                let mut e = enc8(o);
                if tw.is_none() && *gated {
                    // The mask as an overlay.
                    if fin.data[idx] >= 128 {
                        e = [((u16::from(e[0]) + 255) / 2) as u8, e[1] / 2, e[2] / 2];
                    }
                }
                o = [0.0; 3];
                let _ = o;
                let b = (j * pw * 6 + k * pw + i) * 3;
                out[b..b + 3].copy_from_slice(&e);
            }
        }
    }
    std::fs::create_dir_all(dir).expect("dir");
    let f = std::fs::File::create(dir.join(format!("{id}.png"))).expect("png");
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), (pw * 6) as u32, ph as u32);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()
        .expect("header")
        .write_image_data(&out)
        .expect("data");
}

/// Whitening that removes chroma along the teeth's own chroma direction only (the face's mask core), leaving the rest.
fn whiten_proj(src: [f64; 3], u: [f64; 3], tw: f64, m: f64, l_max: f64) -> [f64; 3] {
    let l = (src[0] + src[1] + src[2]) / 3.0;
    let removed = ym_retouch::TEETH_CHROMA_LIN * tw - ym_retouch::TEETH_CHROMA_QUAD * tw * tw;
    let light = ym_retouch::TEETH_LIGHT_LIN * tw + ym_retouch::TEETH_LIGHT_QUAD * tw * tw;
    let c = [src[0] - l, src[1] - l, src[2] - l];
    let d = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
    let cap = (1.0 + ym_retouch::TEETH_CAP_RISE * tw * tw) * l_max / l.max(1e-4);
    let gain = (1.0 + light * m).min(cap.max(1.0));
    [0, 1, 2].map(|k| (l + c[k] - removed * m * d * u[k]) * gain)
}

/// Opponent components of a pixel: red-green-ish `a = 2R - G - B` and yellow-blue-ish `b = R + G - 2B`, over the mean.
fn opp(p: [f64; 3]) -> (f64, f64, f64) {
    let l = (p[0] + p[1] + p[2]) / 3.0;
    (
        (2.0 * p[0] - p[1] - p[2]) / 3.0,
        (p[0] + p[1] - 2.0 * p[2]) / 3.0,
        l,
    )
}

/// What whitening (old and with the chroma taken along the teeth's own direction) does to the gum band and to the teeth.
fn whiten_stats(id: &str, fin: &Mask, loc: &Located, band: &Band) {
    let px = |i: usize| {
        let p = loc.image.data[i];
        [
            f64::from(p[0]).max(0.0),
            f64::from(p[1]).max(0.0),
            f64::from(p[2]).max(0.0),
        ]
    };
    let core: Vec<usize> = (0..fin.data.len())
        .filter(|&i| fin.data[i] >= 230)
        .collect();
    if core.len() < 20 {
        return;
    }
    let mut c0 = [0.0f64; 3];
    for &i in &core {
        let p = px(i);
        let l = (p[0] + p[1] + p[2]) / 3.0;
        for k in 0..3 {
            c0[k] += p[k] - l;
        }
    }
    let n0 = (c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2])
        .sqrt()
        .max(1e-12);
    let u = [c0[0] / n0, c0[1] / n0, c0[2] / n0];
    let mut ls: Vec<f64> = (0..fin.data.len())
        .filter(|&i| fin.data[i] != 0)
        .map(|i| {
            let p = px(i);
            (p[0] + p[1] + p[2]) / 3.0
        })
        .collect();
    ls.sort_unstable_by(f64::total_cmp);
    let l_max = ls[((0.99 * ls.len() as f64).ceil() as usize).clamp(1, ls.len()) - 1];
    // The core's own R/G, the reference of the pink gate.
    let mut rgs: Vec<f64> = core
        .iter()
        .map(|&i| {
            let p = px(i);
            p[0] / p[1].max(1e-4)
        })
        .collect();
    rgs.sort_unstable_by(f64::total_cmp);
    let r0 = rgs[rgs.len() / 2].max(1e-4);
    let gate = |p: [f64; 3]| {
        let t = ((p[0] / p[1].max(1e-4)) / r0 - 1.15) / (1.45 - 1.15);
        let t = t.clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    for (set, pick) in [("gum", 0u8), ("teeth", 1u8)] {
        let idx: Vec<usize> = (0..fin.data.len())
            .filter(|&i| {
                fin.data[i] >= 128
                    && if pick == 0 {
                        band.mask[i]
                    } else {
                        loc.label[i] == TEETH
                    }
            })
            .collect();
        if idx.len() < 20 {
            continue;
        }
        for tw in [0.6, 1.0] {
            let (mut ai, mut bi, mut li) = (0.0, 0.0, 0.0);
            let (mut ao, mut bo, mut lo) = (0.0, 0.0, 0.0);
            let (mut an, mut bn, mut ln) = (0.0, 0.0, 0.0);
            for &i in &idx {
                let src = px(i);
                let m = f64::from(fin.data[i]) / 255.0;
                let (a, b, l) = opp(src);
                ai += a;
                bi += b.abs();
                li += l;
                let (a, b, l) = opp(ym_retouch::teeth_pixel(src, tw, m, l_max));
                ao += a;
                bo += b.abs();
                lo += l;
                let (a, b, l) = opp(ym_retouch::teeth_pixel(src, tw, m * gate(src), l_max));
                an += a;
                bn += b.abs();
                ln += l;
            }
            println!(
                "WHITEN {id} {set} tw {tw} n {} a_old {:.3} a_new {:.3} b_old {:.3} b_new {:.3} l_old {:.3} l_new {:.3}",
                idx.len(),
                ao / ai,
                an / ai,
                bo / bi,
                bn / bi,
                lo / li,
                ln / li
            );
        }
    }
}

fn metrics(fin: &Mask, loc: &Located, band: &Band) -> String {
    let (w, _h) = (loc.image.width_px as usize, loc.image.height_px as usize);
    let (x0, y0, x1, y1) = loc.window;
    let (mut inter, mut union, mut m, mut t) = (0usize, 0usize, 0usize, 0usize);
    let (mut in_open_non_teeth, mut in_band, mut off_opening) = (0usize, 0usize, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * w + x;
            let a = fin.data[i] >= 128;
            let b = loc.label[i] == TEETH;
            m += usize::from(a);
            t += usize::from(b);
            inter += usize::from(a && b);
            union += usize::from(a || b);
            if a {
                if loc.mouth.data[i] >= 128 {
                    if !b {
                        in_open_non_teeth += 1;
                    }
                } else {
                    off_opening += 1;
                }
                if band.mask[i] {
                    in_band += 1;
                }
            }
        }
    }
    let f = |n: usize, d: usize| {
        if d == 0 {
            f64::NAN
        } else {
            n as f64 / d as f64
        }
    };
    format!(
        "iou {:.4}\tcover {:.4}\topen_spill {:.4}\tband_spill {:.4}\toff {:.4}\tarea {m}",
        f(inter, union),
        f(inter, t),
        f(in_open_non_teeth, m),
        f(in_band, m),
        f(off_opening, m)
    )
}

/// Chooses with the gap to the lip rewarded: `score + mu * lip_gap`.
fn choose_gap(c: &Cands, regions: &Regions, mu: f64) -> Option<usize> {
    let cands: Vec<(&[f32], f32)> = c
        .logits
        .iter()
        .zip(&c.iou)
        .map(|(l, &i)| (&l[..], i))
        .collect();
    let (_, stats) = select_candidate(&cands, regions);
    let mut best: Option<(usize, f64)> = None;
    for (i, s) in stats.iter().enumerate() {
        let valid = s.area > 0
            && s.cover >= MIN_COVER
            && s.score.is_finite()
            && s.spill <= MAX_SPILL
            && !(s.cover > CAVITY_COVER && regions.aspect > CAVITY_ASPECT);
        if !valid {
            continue;
        }
        let v = s.score + mu * lip_gap(&c.logits[i], regions);
        if best.is_none_or(|b| v > b.1) {
            best = Some((i, v));
        }
    }
    best.map(|b| b.0)
}

/// The teeth minus the gum: the first answer's logits, cut by the best gum candidate of `name` that is a distinct thin region.
fn minus_gum(
    base: &Cands,
    enc: &EncodedGeo,
    d: &Dumps,
    name: &str,
    min_iou: f32,
    max_share: f64,
    mu: f64,
) -> Option<Vec<f32>> {
    let i = choose_gap(base, &enc.regions, mu)?;
    let m1 = &base.logits[i];
    let Some(g) = d.by_name.get(name) else {
        return Some(m1.clone());
    };
    let area: usize = m1.iter().filter(|&&v| v > 0.0).count();
    let mut best: Option<(usize, f32)> = None;
    for (k, l) in g.logits.iter().enumerate() {
        let both = m1
            .iter()
            .zip(l)
            .filter(|(a, b)| **a > 0.0 && **b > 0.0)
            .count();
        let share = both as f64 / area.max(1) as f64;
        if g.iou[k] >= min_iou
            && share > 0.01
            && share <= max_share
            && best.is_none_or(|b| g.iou[k] > b.1)
        {
            best = Some((k, g.iou[k]));
        }
    }
    Some(match best {
        None => m1.clone(),
        Some((k, _)) => m1
            .iter()
            .zip(&g.logits[k])
            .map(|(a, b)| a.min(-b))
            .collect(),
    })
}

/// The teeth minus the part of the gum answer that lies above the line through the teeth points of its prompt.
fn cut_gum_above(
    base: &Cands,
    loc: &Located,
    enc: &EncodedGeo,
    d: &Dumps,
    name: &str,
    min_iou: f32,
    min_cut: f64,
    max_cut: f64,
    max_thick: f64,
    log: Option<&mut String>,
) -> Option<Vec<f32>> {
    let i = choose(base, &enc.regions)?;
    let m1 = &base.logits[i];
    let g = d.by_name.get(name)?;
    let shape = if name == "gumq2" {
        GumShape {
            depth: 0.03,
            ..GumShape::DEFAULT
        }
    } else {
        GumShape::DEFAULT
    };
    let prompt = gum_prompt(&loc.analysis.geometry, &enc.crop, m1, &shape)?;
    // The teeth points: the negatives that are not the last four (the lips).
    let n = prompt.points.len();
    let teeth: Vec<(f32, f32)> = prompt.points[..n - 4]
        .iter()
        .filter(|p| !p.positive)
        .map(|p| (p.x * GRID_PX as f32, p.y * GRID_PX as f32))
        .collect();
    if teeth.is_empty() {
        return None;
    }
    let line = |x: f32| -> f32 {
        let (mut best, mut bd) = (teeth[0].1, f32::MAX);
        for t in &teeth {
            let dd = (t.0 - x).abs();
            if dd < bd {
                bd = dd;
                best = t.1;
            }
        }
        best
    };
    let area = m1.iter().filter(|&&v| v > 0.0).count().max(1) as f64;
    let mut best: Option<(usize, f32, f64)> = None;
    for (k, l) in g.logits.iter().enumerate() {
        let mut cut = 0usize;
        for y in 0..GRID_PX {
            for x in 0..GRID_PX {
                let j = y * GRID_PX + x;
                if m1[j] > 0.0 && l[j] > 0.0 && (y as f32) < line(x as f32) {
                    cut += 1;
                }
            }
        }
        let share = cut as f64 / area;
        let (mut ox0, mut ox1) = (GRID_PX, 0usize);
        for j in 0..N {
            if enc.regions.opening_at(j) {
                ox0 = ox0.min(j % GRID_PX);
                ox1 = ox1.max(j % GRID_PX);
            }
        }
        let thick = cut as f64
            / (ox1.saturating_sub(ox0) + 1) as f64
            / (ox1.saturating_sub(ox0) + 1) as f64;
        if thick > max_thick {
            continue;
        }
        if let Some(log) = log.as_deref() {
            let _ = log;
        }
        if g.iou[k] >= min_iou
            && share >= min_cut
            && share <= max_cut
            && best.is_none_or(|b| g.iou[k] > b.1)
        {
            best = Some((k, g.iou[k], share));
        }
    }
    let Some((k, _, _)) = best else {
        return Some(m1.clone());
    };
    let l = &g.logits[k];
    let mut out = m1.clone();
    for y in 0..GRID_PX {
        for x in 0..GRID_PX {
            let j = y * GRID_PX + x;
            if (y as f32) < line(x as f32) {
                out[j] = out[j].min(-l[j]);
            }
        }
    }
    Some(out)
}

fn methods() -> Vec<(String, Method)> {
    let mut v: Vec<(String, Method)> = Vec::new();
    v.push((
        "current".into(),
        Box::new(|base, _, enc, _| choose(base, &enc.regions).map(|i| base.logits[i].clone())),
    ));
    for name in ["neg015", "deep"] {
        v.push((
            name.into(),
            Box::new(move |_, _, enc, d| {
                let c = d.by_name.get(name)?;
                choose(c, &enc.regions).map(|i| c.logits[i].clone())
            }),
        ));
    }
    for mu in [2.0f64, 5.0, 10.0] {
        v.push((
            format!("select_mu{mu}"),
            Box::new(move |base, _, enc, _| {
                choose_gap(base, &enc.regions, mu).map(|i| base.logits[i].clone())
            }),
        ));
    }
    for (name, min_iou, share) in [
        ("gumq", 0.5f32, 0.4f64),
        ("gumq", 0.7, 0.4),
        ("gumq", 0.7, 0.25),
        ("gumq2", 0.5, 0.4),
        ("gumq2", 0.7, 0.4),
        ("gumq2", 0.7, 0.25),
    ] {
        v.push((
            format!("{name}_i{min_iou}_s{share}"),
            Box::new(move |base, _, enc, d| minus_gum(base, enc, d, name, min_iou, share, 0.0)),
        ));
    }
    for (name, min_iou, lo, hi, thick) in [
        ("gumq", 0.0f32, 0.02f64, 0.5f64, 0.03f64),
        ("gumq", 0.0, 0.02, 0.5, 0.05),
        ("gumq", 0.0, 0.02, 0.5, 0.08),
        ("gumq2", 0.0, 0.02, 0.5, 0.03),
        ("gumq2", 0.0, 0.02, 0.5, 0.05),
    ] {
        v.push((
            format!("cut_{name}_t{thick}"),
            Box::new(move |base, loc, enc, d| {
                cut_gum_above(base, loc, enc, d, name, min_iou, lo, hi, thick, None)
            }),
        ));
    }
    v
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let take = |args: &mut Vec<String>, flag: &str| -> Option<usize> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args[i + 1].parse().expect("number");
        args.drain(i..i + 2);
        Some(v)
    };
    let from = take(&mut args, "--from").unwrap_or(0);
    let to = take(&mut args, "--to").unwrap_or(usize::MAX);
    let step = take(&mut args, "--step").unwrap_or(1);
    let mode = args.remove(0);
    match mode.as_str() {
        "dump" => dump(&args, from, to, step),
        "score" => score(&args),
        "crops" => crops(&args),
        other => panic!("unknown mode {other}"),
    }
    let _ = TeethPrompt::default();
}
