//! Dev tool (`docs/27` S-17): scores the teeth masks on photos that come with a teeth label, for local evaluation only.
//!
//! ```text
//! cargo run -p ym-core --example teeth_eval -- <data root> <images dir> <labels dir> <out.tsv> \
//!     [--from N] [--to N] [--step N] [--factors 2.0,2.5,3.0] [--styles box,points,both]
//! ```
//!
//! For every `<images dir>/<id>.jpg` that has a `<labels dir>/<id>.png` (255 on teeth, 0 elsewhere; the labels of the public
//! EasyPortrait dataset, fetched for this run and never committed or used to train anything) it finds the face whose mouth holds
//! the label, builds the teeth mask the colour gates of S-16 built (`old`), and the mask SAM gives for every crop size and
//! prompt style (`sam <factor> <style>`), and appends one line per mask to the TSV: the id, the config, how it ended, the IoU
//! of the final teeth mask (threshold 128) with the label inside the face's window, and the areas. `oracle` is the best of SAM's
//! four masks by IoU (what a perfect choice could reach); `chosen` is the mask the rules of `docs/27` 5.6 chose.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[path = "teeth_support/mod.rs"]
mod support;

use support::{decode, old_teeth_mask, to_linear_awg4};
use ym_core::pipeline::object::ObjectModel;
use ym_core::pipeline::teeth::SamTeeth;
use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::face_parts::mouth_interior_mask;
use ym_retouch::mediapipe::MediaPipeFaces;
use ym_retouch::teeth::{PromptShape, PromptStyle, crop_mask};
use ym_retouch::{Face, Mask, Rect, teeth_from_segment};

fn iou(pred: &Mask, label: &[u8], window: (usize, usize, usize, usize)) -> (f64, usize, usize) {
    let (x0, y0, x1, y1) = window;
    let w = pred.width_px as usize;
    let (mut inter, mut union, mut p, mut l) = (0usize, 0usize, 0usize, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let a = pred.data[y * w + x] >= 128;
            let b = label[y * w + x] >= 128;
            p += usize::from(a);
            l += usize::from(b);
            inter += usize::from(a && b);
            union += usize::from(a || b);
        }
    }
    (
        if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        },
        p,
        l,
    )
}

fn shape_of(s: &str) -> PromptShape {
    let d = PromptShape::DEFAULT;
    match s {
        // The upper row only (the spec's shape).
        "default" | "upper" => d,
        // The upper row and a lower row 0.18 of the column's height above the lower lip.
        "lower" => PromptShape {
            lower_row: Some(0.18),
            ..d
        },
        // Both rows and a negative point in the cavity of a tall opening.
        "cavity" => PromptShape {
            lower_row: Some(0.18),
            cavity: Some(0.6),
            ..d
        },
        "upper-cavity" => PromptShape {
            cavity: Some(0.6),
            ..d
        },
        // Seven columns instead of five, out to 0.1 and 0.9 of the opening's width.
        "upper-wide" => PromptShape {
            columns: &[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9],
            ..d
        },
        // Rows at 0.22 of the column's height, with no cap in the width.
        "uncapped" => PromptShape { row_cap: 10.0, ..d },
        other => panic!("unknown shape {other}"),
    }
}

fn style_of(s: &str) -> PromptStyle {
    match s {
        "box" => PromptStyle::Box,
        "points" => PromptStyle::Points,
        "both" => PromptStyle::BoxAndPoints,
        other => panic!("unknown style {other}"),
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let take = |args: &mut Vec<String>, flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args[i + 1].clone();
        args.drain(i..i + 2);
        Some(v)
    };
    let from: usize = take(&mut args, "--from").map_or(0, |v| v.parse().expect("--from"));
    let step: usize = take(&mut args, "--step").map_or(1, |v| v.parse().expect("--step"));
    let to: usize = take(&mut args, "--to").map_or(usize::MAX, |v| v.parse().expect("--to"));
    let factors: Vec<f64> = take(&mut args, "--factors")
        .unwrap_or_else(|| "2.5".to_owned())
        .split(',')
        .map(|v| v.parse().expect("factor"))
        .collect();
    let styles: Vec<String> = take(&mut args, "--styles")
        .unwrap_or_else(|| "box,points,both".to_owned())
        .split(',')
        .map(str::to_owned)
        .collect();
    let shapes: Vec<String> = take(&mut args, "--shapes")
        .unwrap_or_else(|| "default".to_owned())
        .split(',')
        .map(str::to_owned)
        .collect();
    let (data_root, images, labels, out) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
    );
    let mp = data_root.join("models/mediapipe");
    let finder = FaceFinder::new(BlazeFace::load(&mp).expect("blazeface weights"));
    let engine = MediaPipeFaces::load(&mp).expect("mesh and segmenter weights");
    let sam = ObjectModel::probe(&data_root).expect("the SAM weights");
    let mut ids: Vec<String> = std::fs::read_dir(&images)
        .expect("images dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.path()
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();
    ids.sort();
    let mut sink = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .expect("tsv");
    let seg = SamTeeth::new(Arc::clone(&engine), Arc::clone(&sam));
    for id in ids
        .iter()
        .step_by(step)
        .skip(from)
        .take(to.saturating_sub(from))
    {
        let (w, h, rgb) = decode(&images.join(format!("{id}.jpg")));
        let label = {
            let f = std::fs::File::open(labels.join(format!("{id}.png"))).expect("label");
            let dec = png::Decoder::new(std::io::BufReader::new(f));
            let mut r = dec.read_info().expect("png header");
            let mut buf = vec![0u8; r.output_buffer_size().expect("png size")];
            let info = r.next_frame(&mut buf).expect("png frame");
            assert_eq!(
                (info.width as usize, info.height as usize),
                (w, h),
                "{id}: label size"
            );
            buf
        };
        let image = to_linear_awg4(w, h, &rgb);
        let src = Rgb8::new(w, h, &rgb).expect("picture");
        let found = finder.find(&src, &FinderParams::default()).expect("find");
        // The face whose box holds the most of the label's teeth.
        let mut best: Option<(usize, Face)> = None;
        for d in &found {
            let (x0, y0) = (d.rect.x_px.max(0) as usize, d.rect.y_px.max(0) as usize);
            let (x1, y1) = (
                ((d.rect.x_px + d.rect.width_px as i32).max(0) as usize).min(w),
                ((d.rect.y_px + d.rect.height_px as i32).max(0) as usize).min(h),
            );
            let n = (y0..y1)
                .map(|y| (x0..x1).filter(|&x| label[y * w + x] >= 128).count())
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
        let mut lines = String::new();
        let Some((n, face)) = best.filter(|b| b.0 > 0) else {
            writeln!(lines, "{id}\t-\tno face holds the teeth\t\t\t").unwrap();
            sink.write_all(lines.as_bytes()).unwrap();
            continue;
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
        let analysis = engine.analyse(&image, &face).expect("analyse");
        let Some(analysis) = analysis else {
            writeln!(lines, "{id}\t-\tno mesh\t\t\t{n}").unwrap();
            sink.write_all(lines.as_bytes()).unwrap();
            continue;
        };
        let Some(mouth) = mouth_interior_mask(&analysis.geometry, size).expect("mouth") else {
            writeln!(lines, "{id}\t-\tclosed mouth by the mesh\t\t\t{n}").unwrap();
            sink.write_all(lines.as_bytes()).unwrap();
            continue;
        };
        let mouth_full = mouth.to_full(size).expect("mouth");
        let old = old_teeth_mask(&mouth_full, &image, extent, true);
        let (v, p, l) = iou(&old, &label, window);
        writeln!(lines, "{id}\told\tok\t{v:.4}\t{p}\t{l}").unwrap();
        for &factor in &factors {
            let t = Instant::now();
            let Some(enc) = seg.encode(&image, &face, factor).expect("encode") else {
                writeln!(lines, "{id}\tsam {factor}\tno mouth to encode\t\t\t{l}").unwrap();
                continue;
            };
            let enc_ms = t.elapsed().as_secs_f64() * 1e3;
            writeln!(lines, "{id}\taspect\t{:.3}\t\t\t", enc.regions.aspect).unwrap();
            for (style_name, shape_name) in styles
                .iter()
                .flat_map(|st| shapes.iter().map(move |sh| (st, sh)))
                .filter(|(st, sh)| *st != "box" || *sh == &shapes[0])
            {
                let style = style_of(style_name);
                let s = &if style_name == "box" {
                    "box".to_owned()
                } else {
                    format!("{style_name}/{shape_name}")
                };
                let Some(dec) = seg
                    .decide_with(&enc, style, &shape_of(shape_name))
                    .expect("decide")
                else {
                    writeln!(lines, "{id}\tsam {factor} {s}\tno prompt\t\t\t{l}").unwrap();
                    continue;
                };
                let finish = |logits: &[f32]| -> Mask {
                    let seg_mask = crop_mask(logits, &enc.crop, size)
                        .expect("crop mask")
                        .map_or_else(
                            || Mask::zeros(size.0, size.1).unwrap(),
                            |m| m.to_full(size).unwrap(),
                        );
                    teeth_from_segment(&mouth_full, &seg_mask, extent).expect("teeth")
                };
                // The chosen one.
                match dec.chosen {
                    Ok(i) => {
                        let m = finish(dec.decoded.candidate_logits(i));
                        let (v, p, _) = iou(&m, &label, window);
                        let st = dec.stats[i];
                        writeln!(
                            lines,
                            "{id}\tsam {factor} {s}\tchosen {i}\t{v:.4}\t{p}\t{l}\tiou {:.3} cover {:.3} spill {:.3}\t{enc_ms:.0} ms",
                            st.iou, st.cover, st.spill
                        )
                        .unwrap();
                    }
                    Err(why) => {
                        writeln!(lines, "{id}\tsam {factor} {s}\trefused: {why}\t0\t0\t{l}")
                            .unwrap();
                    }
                }
                // The best of the four, by IoU with the label.
                let mut best = (0.0f64, 0usize);
                for i in 0..dec.decoded.candidates() {
                    let m = finish(dec.decoded.candidate_logits(i));
                    let (v, _, _) = iou(&m, &label, window);
                    if v > best.0 {
                        best = (v, i);
                    }
                }
                writeln!(
                    lines,
                    "{id}\tsam {factor} {s} oracle\ttoken {}\t{:.4}\t\t{l}",
                    best.1, best.0
                )
                .unwrap();
                // Every candidate with its stats, for the choice of the rule.
                for i in 0..dec.decoded.candidates() {
                    let m = finish(dec.decoded.candidate_logits(i));
                    let (v, p, _) = iou(&m, &label, window);
                    let st = dec.stats[i];
                    writeln!(
                        lines,
                        "{id}\tsam {factor} {s} cand{i}\tiou {:.3} cover {:.3} spill {:.3}\t{v:.4}\t{p}\t{l}",
                        st.iou, st.cover, st.spill
                    )
                    .unwrap();
                }
            }
        }
        sink.write_all(lines.as_bytes()).unwrap();
        sink.flush().unwrap();
        eprintln!("{id} done");
    }
}
