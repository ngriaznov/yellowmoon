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
use ym_core::pipeline::teeth::{EncodedMouth, SamTeeth};
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::face_parts::mouth_interior_mask;
use ym_retouch::mediapipe::{FaceAnalysis, MediaPipeFaces};
use ym_retouch::teeth::{
    CROP_FACTOR, GRID_PX, GumShape, MouthCrop, PromptShape, PromptStyle, Regions, TeethPrompt,
    crop_mask, gum_prompt, select_candidate, teeth_prompt_with,
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
    let label = read_label(&labels.join(format!("{id}.png")), w, h);
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
            "{id}\t-\tinfo\tbandh_rel {:.4}\tband_px {}\tteeth_px {}",
            band.rel, band.n_band, band.n_teeth
        )
        .unwrap();
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
                    metrics(&fin, &loc, &band)
                }
            };
            writeln!(text, "{id}\t{name}\t{line}").unwrap();
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

fn methods() -> Vec<(String, Method)> {
    let mut v: Vec<(String, Method)> = Vec::new();
    v.push((
        "current".into(),
        Box::new(|base, _, enc, _| choose(base, &enc.regions).map(|i| base.logits[i].clone())),
    ));
    for name in ["neg015", "neg03", "deep"] {
        v.push((
            name.into(),
            Box::new(move |_, _, enc, d| {
                let c = d.by_name.get(name)?;
                choose(c, &enc.regions).map(|i| c.logits[i].clone())
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
        other => panic!("unknown mode {other}"),
    }
    let _ = TeethPrompt::default();
}
