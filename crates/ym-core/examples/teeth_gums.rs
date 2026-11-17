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
use ym_retouch::teeth::{
    CROP_FACTOR, GRID_PX, GumShape, MouthCrop, PromptShape, PromptStyle, Regions, TeethPrompt,
    crop_mask, gum_prompt, lip_gap, select_candidate, teeth_prompt_with,
};
use ym_retouch::teeth::{CAVITY_ASPECT, CAVITY_COVER, MAX_SPILL, MIN_COVER};
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
    vec![
        ("gumq", d),
        (
            "gumq2",
            GumShape {
                depth: 0.03,
                ..d
            },
        ),
    ]
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
                    rect: Rect::new(
                        d.rect.x_px,
                        d.rect.y_px,
                        d.rect.width_px,
                        d.rect.height_px,
                    ),
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
        let Ok(Some(a)) = engine.analyse(&image, &face) else { continue };
        let Some(crop) = MouthCrop::from_geometry(&a.geometry, 1.75) else { continue };
        let Some(reg) = Regions::from_geometry(&a.geometry, &MouthCrop::from_geometry(&a.geometry, CROP_FACTOR).unwrap()) else { continue };
        let side = 200usize;
        let mut o = vec![0u8; side * side * 3];
        for j in 0..side {
            for i in 0..side {
                let x = (crop.x_px + (i as f64 + 0.5) / side as f64 * crop.side_px).clamp(0.0, w as f64 - 1.0) as usize;
                let y = (crop.y_px + (j as f64 + 0.5) / side as f64 * crop.side_px).clamp(0.0, h as f64 - 1.0) as usize;
                o[(j * side + i) * 3..(j * side + i) * 3 + 3].copy_from_slice(&rgb[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        let f = std::fs::File::create(out.join(format!("{id}_a{:.2}.png", reg.aspect))).expect("png");
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), side as u32, side as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().expect("header").write_image_data(&o).expect("data");
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
                    let line = format!("{}\tlipgap {:.4}", metrics(&fin, &loc, &band), lip_gap(&logits, &enc.regions));
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
fn overlay(dir: &Path, id: &str, images: &Path, loc: &Located, enc: &EncodedGeo, shown: &[(String, Option<Mask>)]) {
    let (w, h, rgb) = decode(&images.join(format!("{id}.jpg")));
    let c = &enc.crop;
    let side = (c.side_px / 2.0).round() as i64;
    let (cx, cy) = (c.x_px + c.side_px / 2.0, c.y_px + c.side_px / 2.0);
    let (x0, y0) = (cx.round() as i64 - side / 2, cy.round() as i64 - side / 2);
    let scale = (512 / side.max(1)).max(1) as usize;
    let pick: Vec<&(String, Option<Mask>)> = shown
        .iter()
        .filter(|(n, _)| ["current", "select_mu5", "gumq_i0.7_s0.25", "gumq_i0.5_s0.4"].contains(&n.as_str()))
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
    e.write_header().expect("header").write_image_data(&out).expect("data");
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
    let f = |n: usize, d: usize| if d == 0 { f64::NAN } else { n as f64 / d as f64 };
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
        let both = m1.iter().zip(l).filter(|(a, b)| **a > 0.0 && **b > 0.0).count();
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
        Some((k, _)) => m1.iter().zip(&g.logits[k]).map(|(a, b)| a.min(-b)).collect(),
    })
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
