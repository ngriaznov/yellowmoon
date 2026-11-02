//! The SAM 2.1 small runtime against Meta's own code (`docs/algorithms/06-masks.md` section 6.5, N-17).
//!
//! The fixtures are made by `training/sam2/oracle.py` (Meta's `SAM2ImagePredictor` on CPU in f32, with the published f16
//! weights widened to f32) for 20 public photos and 3 prompts each, and fetched by `tools/fetch/sam2.sh --fixtures` into
//! `testdata/sam2/oracle-1/` (git-ignored, about 700 MB). The weights are `data/models/sam2/` (`tools/fetch/sam2.sh`). Without
//! them the gates print a plain "SKIPPED" line and pass; `YM_REQUIRE_SAM2=1` makes a missing file a failure (CI sets it).
//! `YM_SAM2_PHOTOS=cat,dog` runs only those photos (a quick local run).
//!
//! Every gate compares one stage of ours with Meta's from the same input, so that an error is found where it is made:
//!
//! 1. **resize**: our picture-to-tensor against Meta's tensor made from the same 8-bit picture;
//! 2. **encoder**: our three maps from Meta's tensor against Meta's maps;
//! 3. **decoder**: our logits from Meta's maps against Meta's logits;
//! 4. **whole path**: our picture to logits against Meta's;
//! 5. **masks**: mask IoU at logit > 0 at 256 x 256, and the mask chosen, on all 60 prompts.
//!
//! These tests are slow without optimisation (the encoder is about 170 GMAC): run them with `cargo test -p ym-ml --release
//! --test sam2`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ym_ml::sam2::{
    Decoded, Decoding, Features, MlError, Picture, Point, Prompt, Sam2, WEIGHTS_FILE,
};

// -------------------------------------------------------------------------------------------------
// The gates (docs/algorithms/06-masks.md section 6.5; the logit bounds are the measured ones, proposed in the N-17 notes)
// -------------------------------------------------------------------------------------------------

/// Our resize against Meta's tensor, in normalised values.
const RESIZE_MAX_ABS: f64 = 1e-2;
const RESIZE_MEAN_ABS: f64 = 1e-3;
/// The encoder's maps from the same tensor: RMS of the difference over the RMS of Meta's map.
const ENCODER_REL_RMS: f64 = 2e-3;
/// The logits from Meta's maps, and the whole path: largest and mean absolute difference.
const LOGIT_MAX_ABS: f64 = 0.05;
const LOGIT_MEAN_ABS: f64 = 0.005;
/// Mask IoU at logit > 0, 256 x 256.
const MASK_IOU_MIN: f64 = 0.99;

// -------------------------------------------------------------------------------------------------
// Fixture files
// -------------------------------------------------------------------------------------------------

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn weights_dir() -> PathBuf {
    repo().join("data/models/sam2")
}

fn fixtures_dir() -> PathBuf {
    repo().join("testdata/sam2/oracle-1")
}

/// One array of an `.npz`.
struct Npy {
    shape: Vec<usize>,
    descr: String,
    data: Vec<u8>,
}

impl Npy {
    fn f32s(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4", "float32 array");
        self.data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn i32s(&self) -> Vec<i32> {
        assert_eq!(self.descr, "<i4", "int32 array");
        self.data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
}

fn u16_at(b: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// Reads an uncompressed `.npz` (a zip of `.npy` files, `oracle.py` writes it with `ZIP_STORED`).
fn read_npz(path: &Path) -> HashMap<String, Npy> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let eocd = (0..=bytes.len() - 22)
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .expect("zip end record");
    let count = u16_at(&bytes, eocd + 10);
    let mut at = u32_at(&bytes, eocd + 16);
    let mut out = HashMap::new();
    for _ in 0..count {
        assert_eq!(
            bytes[at..at + 4],
            [0x50, 0x4b, 0x01, 0x02],
            "zip central header"
        );
        let method = u16_at(&bytes, at + 10);
        let size = u32_at(&bytes, at + 20);
        let (nlen, elen, clen) = (
            u16_at(&bytes, at + 28),
            u16_at(&bytes, at + 30),
            u16_at(&bytes, at + 32),
        );
        let local = u32_at(&bytes, at + 42);
        let name = String::from_utf8(bytes[at + 46..at + 46 + nlen].to_vec()).expect("utf-8 name");
        assert_eq!(method, 0, "{name} is stored, not compressed");
        let (lnlen, lelen) = (u16_at(&bytes, local + 26), u16_at(&bytes, local + 28));
        let body = &bytes[local + 30 + lnlen + lelen..][..size];
        out.insert(name.trim_end_matches(".npy").to_owned(), parse_npy(body));
        at += 46 + nlen + elen + clen;
    }
    out
}

fn parse_npy(b: &[u8]) -> Npy {
    assert_eq!(&b[..6], b"\x93NUMPY", "npy magic");
    let (hlen, start) = if b[6] == 1 {
        (u16_at(b, 8), 10)
    } else {
        (u32_at(b, 8), 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).expect("npy header");
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .expect("descr")
        .to_owned();
    assert!(header.contains("'fortran_order': False"), "C order");
    let shape_text = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(['(', ')']).nth(1))
        .expect("shape");
    let shape = shape_text
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    Npy {
        shape,
        descr,
        data: b[start + hlen..].to_vec(),
    }
}

fn require() -> bool {
    std::env::var_os("YM_REQUIRE_SAM2").is_some_and(|v| v != "0" && !v.is_empty())
}

/// Whether the slow tests (a whole encoder run is 14 s optimised and about 90 s in the dev profile) run in this build: always in an
/// optimised build, and in a debug build only when asked for (`YM_SAM2_FULL=1`, `YM_SAM2_PHOTOS=...` or `YM_REQUIRE_SAM2=1`).
fn heavy_ok() -> bool {
    let asked = require()
        || std::env::var_os("YM_SAM2_FULL").is_some_and(|v| v != "0" && !v.is_empty())
        || std::env::var_os("YM_SAM2_PHOTOS").is_some();
    if cfg!(debug_assertions) && !asked {
        println!(
            "SKIPPED: the SAM 2 runtime needs an optimised build for this test (cargo test -p ym-ml --release --test sam2); a debug build takes 90 s a photo (YM_SAM2_FULL=1 runs it anyway)"
        );
        return false;
    }
    true
}

/// The model, or `None` after printing why not (or a panic when `YM_REQUIRE_SAM2` is set).
fn model() -> Option<&'static Sam2> {
    static MODEL: OnceLock<Option<Sam2>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            if !weights_dir().join(WEIGHTS_FILE).exists() {
                let msg = format!(
                    "SKIPPED: no SAM 2 weights in {} (run tools/fetch/sam2.sh)",
                    weights_dir().display()
                );
                assert!(!require(), "{msg}");
                println!("{msg}");
                return None;
            }
            // `load` checks the SHA-256 against data/models/sam2/SHA256SUMS.
            Some(Sam2::load(&weights_dir()).expect("the weights load and match SHA256SUMS"))
        })
        .as_ref()
}

fn photo_ids() -> Option<Vec<String>> {
    if !fixtures_dir().join("manifest.json").exists() {
        let msg = format!(
            "SKIPPED: no SAM 2 oracle fixtures in {} (run tools/fetch/sam2.sh --fixtures)",
            fixtures_dir().display()
        );
        assert!(!require(), "{msg}");
        println!("{msg}");
        return None;
    }
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("manifest.json")).expect("manifest"),
    )
    .expect("json");
    let mut ids: Vec<String> = manifest["photos"]
        .as_object()
        .expect("photos")
        .keys()
        .cloned()
        .collect();
    ids.sort();
    if let Ok(only) = std::env::var("YM_SAM2_PHOTOS") {
        let want: Vec<&str> = only.split(',').collect();
        ids.retain(|id| want.contains(&id.as_str()));
    } else {
        assert_eq!(ids.len(), 20, "the fixtures hold 20 photos");
    }
    Some(ids)
}

// -------------------------------------------------------------------------------------------------
// Measurements
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Diff {
    max_abs: f64,
    mean_abs: f64,
    /// RMS of the difference over the RMS of the reference.
    rel_rms: f64,
}

fn diff(ours: &[f32], reference: &[f32]) -> Diff {
    assert_eq!(ours.len(), reference.len());
    let (mut max, mut sum, mut sq, mut ref_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        let d = (f64::from(a) - f64::from(b)).abs();
        max = max.max(d);
        sum += d;
        sq += d * d;
        ref_sq += f64::from(b) * f64::from(b);
    }
    let n = ours.len() as f64;
    Diff {
        max_abs: max,
        mean_abs: sum / n,
        rel_rms: (sq / n).sqrt() / (ref_sq / n).sqrt().max(1e-30),
    }
}

fn mask_iou(a: &[f32], b: &[f32]) -> f64 {
    let (mut inter, mut union) = (0usize, 0usize);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x > 0.0, y > 0.0);
        inter += usize::from(x && y);
        union += usize::from(x || y);
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// `[c][y][x]` to `[y][x][c]`.
fn chw_to_hwc(src: &[f32], c: usize, plane: usize) -> Vec<f32> {
    let mut out = vec![0.0; src.len()];
    for ch in 0..c {
        for p in 0..plane {
            out[p * c + ch] = src[ch * plane + p];
        }
    }
    out
}

/// A fixture's prompt `k`.
fn prompt_of(photo: &HashMap<String, Npy>, k: usize) -> Prompt {
    let points = photo[&format!("prompt{k}_points")].f32s();
    let labels = photo[&format!("prompt{k}_labels")].i32s();
    let bbox = photo[&format!("prompt{k}_box")].f32s();
    Prompt {
        points: points
            .chunks_exact(2)
            .zip(&labels)
            .map(|(p, &l)| Point {
                x: p[0],
                y: p[1],
                positive: l == 1,
            })
            .collect(),
        bbox: (bbox.len() == 4).then(|| [bbox[0], bbox[1], bbox[2], bbox[3]]),
    }
}

fn picture_of(photo: &HashMap<String, Npy>) -> (usize, usize, Vec<u8>) {
    let p = &photo["picture"];
    assert_eq!(p.descr, "|u1");
    (p.shape[1], p.shape[0], p.data.clone())
}

/// What one prompt gave, against Meta's.
#[derive(Debug, Clone)]
struct PromptResult {
    photo: String,
    prompt: usize,
    multimask: bool,
    /// Logits of all candidates, from Meta's maps (the decoder alone) and from ours (the whole path).
    decoder: Diff,
    whole: Diff,
    iou_decoder: f64,
    iou_whole: f64,
    chosen_decoder: bool,
    chosen_whole: bool,
    token_whole: usize,
    score_diff_decoder: f64,
    iou_pred_diff_decoder: f64,
}

#[derive(Debug, Clone)]
struct PhotoResult {
    id: String,
    resize: Diff,
    encoder: [Diff; 3],
    prompts: Vec<PromptResult>,
    whole_encoder: [Diff; 3],
}

fn compare_decoded(
    ours: &Decoded,
    photo: &HashMap<String, Npy>,
    k: usize,
) -> (Diff, f64, bool, f64, f64) {
    let want_logits = photo[&format!("prompt{k}_low_res_logits")].f32s();
    let want_iou = photo[&format!("prompt{k}_iou")].f32s();
    let want_chosen = photo[&format!("prompt{k}_chosen")].i32s()[0] as usize;
    let want_score = photo[&format!("prompt{k}_object_score")].f32s()[0];
    assert_eq!(
        ours.candidates(),
        want_iou.len(),
        "number of candidate masks"
    );
    let d = diff(&ours.logits, &want_logits);
    let n = 256 * 256;
    let iou = mask_iou(
        ours.low_res_logits(),
        &want_logits[want_chosen * n..(want_chosen + 1) * n],
    );
    let iou_diff = ours
        .iou
        .iter()
        .zip(&want_iou)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    (
        d,
        iou,
        ours.chosen == want_chosen,
        f64::from((ours.object_score - want_score).abs()),
        iou_diff,
    )
}

fn run_photo(model: &Sam2, id: &str) -> PhotoResult {
    let photo = read_npz(&fixtures_dir().join(format!("{id}.npz")));
    let (w, h, rgb) = picture_of(&photo);
    let picture = Picture::new(w, h, &rgb).expect("picture");

    // 1. The resize, in normalised values: ours is [y][x][c], Meta's [c][y][x].
    let ours = ym_ml::sam2::preprocess(&picture);
    let meta_tensor = photo["input_tensor"].f32s();
    let meta_hwc = chw_to_hwc(&meta_tensor, 3, 1024 * 1024);
    let resize = diff(&ours, &meta_hwc);

    let meta_features = Features::from_chw(
        &photo["image_embed"].f32s(),
        &photo["high_res_feat_0"].f32s(),
        &photo["high_res_feat_1"].f32s(),
    )
    .expect("fixture maps");
    let maps_diff = |f: &Features| -> [Diff; 3] {
        [
            diff(f.image_embed(), meta_features.image_embed()),
            diff(f.high_res_0(), meta_features.high_res_0()),
            diff(f.high_res_1(), meta_features.high_res_1()),
        ]
    };

    // 2. The encoder from Meta's tensor.
    let encoded = model
        .encode_input(&meta_hwc)
        .expect("encode from Meta's tensor");
    let encoder = maps_diff(&encoded);
    drop(encoded);

    // 4. The whole path from the 8-bit picture.
    let whole = model.encode(&picture).expect("encode");
    let whole_encoder = maps_diff(&whole);

    let mut prompts = Vec::new();
    for k in 1..=3 {
        let prompt = prompt_of(&photo, k);
        // 3. The decoder from Meta's maps.
        let from_meta = model
            .decode_raw(&meta_features, &prompt)
            .expect("decode from Meta's maps");
        let (d, iou, chosen, score, iou_pred) = compare_decoded(&from_meta, &photo, k);
        let from_ours = model.decode_raw(&whole, &prompt).expect("decode");
        let (wd, wiou, wchosen, _, _) = compare_decoded(&from_ours, &photo, k);
        assert_eq!(
            from_meta.multimask,
            photo[&format!("prompt{k}_multimask")].i32s()[0] == 1,
            "{id} prompt {k}: the multimask rule"
        );
        prompts.push(PromptResult {
            photo: id.to_owned(),
            prompt: k,
            multimask: from_meta.multimask,
            decoder: d,
            whole: wd,
            iou_decoder: iou,
            iou_whole: wiou,
            chosen_decoder: chosen,
            chosen_whole: wchosen,
            token_whole: from_ours.token,
            score_diff_decoder: score,
            iou_pred_diff_decoder: iou_pred,
        });
    }
    PhotoResult {
        id: id.to_owned(),
        resize,
        encoder,
        prompts,
        whole_encoder,
    }
}

/// Every photo, measured once per test binary.
fn results() -> Option<&'static Vec<PhotoResult>> {
    static RESULTS: OnceLock<Option<Vec<PhotoResult>>> = OnceLock::new();
    RESULTS
        .get_or_init(|| {
            if !heavy_ok() {
                return None;
            }
            let model = model()?;
            let ids = photo_ids()?;
            let start = std::time::Instant::now();
            let all: Vec<PhotoResult> = ids
                .iter()
                .map(|id| {
                    let t = std::time::Instant::now();
                    let r = run_photo(model, id);
                    println!("{id:14} measured in {:.1} s", t.elapsed().as_secs_f64());
                    r
                })
                .collect();
            println!(
                "{} photos, {} prompts in {:.0} s",
                all.len(),
                all.len() * 3,
                start.elapsed().as_secs_f64()
            );
            Some(all)
        })
        .as_ref()
}

fn worst_and_mean(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let v: Vec<f64> = values.collect();
    (
        v.iter().copied().fold(0.0, f64::max),
        v.iter().sum::<f64>() / v.len().max(1) as f64,
    )
}

// -------------------------------------------------------------------------------------------------
// The gates
// -------------------------------------------------------------------------------------------------

#[test]
fn the_resize_matches_metas_tensor() {
    let Some(all) = results() else { return };
    let (max, _) = worst_and_mean(all.iter().map(|p| p.resize.max_abs));
    let (mean_worst, mean_avg) = worst_and_mean(all.iter().map(|p| p.resize.mean_abs));
    println!(
        "resize, {} photos: largest difference {max:.2e}, mean difference worst {mean_worst:.2e} (average {mean_avg:.2e})",
        all.len()
    );
    for p in all.iter() {
        assert!(
            p.resize.max_abs <= RESIZE_MAX_ABS,
            "{}: largest difference {:.3e}",
            p.id,
            p.resize.max_abs
        );
        assert!(
            p.resize.mean_abs <= RESIZE_MEAN_ABS,
            "{}: mean difference {:.3e}",
            p.id,
            p.resize.mean_abs
        );
    }
}

#[test]
fn the_encoder_matches_metas_maps_from_the_same_tensor() {
    let Some(all) = results() else { return };
    for (i, name) in [
        "image embedding",
        "high-res 0 (stride 4)",
        "high-res 1 (stride 8)",
    ]
    .iter()
    .enumerate()
    {
        let (worst, mean) = worst_and_mean(all.iter().map(|p| p.encoder[i].rel_rms));
        let (max_abs, _) = worst_and_mean(all.iter().map(|p| p.encoder[i].max_abs));
        println!(
            "encoder {name}: RMS of the difference over the map's RMS, worst {worst:.2e}, mean {mean:.2e}; largest absolute difference {max_abs:.2e}"
        );
    }
    for p in all.iter() {
        for (i, d) in p.encoder.iter().enumerate() {
            assert!(
                d.rel_rms <= ENCODER_REL_RMS,
                "{}: map {i}: {:.3e} of its RMS",
                p.id,
                d.rel_rms
            );
        }
    }
}

#[test]
fn the_decoder_matches_metas_logits_from_metas_maps() {
    let Some(all) = results() else { return };
    let prompts: Vec<&PromptResult> = all.iter().flat_map(|p| &p.prompts).collect();
    let (max, _) = worst_and_mean(prompts.iter().map(|p| p.decoder.max_abs));
    let (mean_worst, mean_avg) = worst_and_mean(prompts.iter().map(|p| p.decoder.mean_abs));
    let (score, _) = worst_and_mean(prompts.iter().map(|p| p.score_diff_decoder));
    let (iou_pred, _) = worst_and_mean(prompts.iter().map(|p| p.iou_pred_diff_decoder));
    println!(
        "decoder from Meta's maps, {} prompts: logits largest difference {max:.3e}, mean difference worst {mean_worst:.3e} (average {mean_avg:.3e}); object score {score:.3e}, predicted IoU {iou_pred:.3e}",
        prompts.len()
    );
    for p in &prompts {
        assert!(
            p.decoder.max_abs <= LOGIT_MAX_ABS,
            "{} prompt {}: largest difference {:.3e}",
            p.photo,
            p.prompt,
            p.decoder.max_abs
        );
        assert!(
            p.decoder.mean_abs <= LOGIT_MEAN_ABS,
            "{} prompt {}: mean difference {:.3e}",
            p.photo,
            p.prompt,
            p.decoder.mean_abs
        );
    }
}

#[test]
fn the_whole_path_matches_metas_logits() {
    let Some(all) = results() else { return };
    let prompts: Vec<&PromptResult> = all.iter().flat_map(|p| &p.prompts).collect();
    let (max, _) = worst_and_mean(prompts.iter().map(|p| p.whole.max_abs));
    let (mean_worst, mean_avg) = worst_and_mean(prompts.iter().map(|p| p.whole.mean_abs));
    println!(
        "whole path, {} prompts: logits largest difference {max:.3e}, mean difference worst {mean_worst:.3e} (average {mean_avg:.3e})",
        prompts.len()
    );
    for (i, name) in ["image embedding", "high-res 0", "high-res 1"]
        .iter()
        .enumerate()
    {
        let (worst, mean) = worst_and_mean(all.iter().map(|p| p.whole_encoder[i].rel_rms));
        println!(
            "whole path, our resize into our encoder: {name} RMS of the difference over the map's RMS, worst {worst:.2e}, mean {mean:.2e}"
        );
    }
    for p in &prompts {
        assert!(
            p.whole.max_abs <= LOGIT_MAX_ABS,
            "{} prompt {}: largest difference {:.3e}",
            p.photo,
            p.prompt,
            p.whole.max_abs
        );
        assert!(
            p.whole.mean_abs <= LOGIT_MEAN_ABS,
            "{} prompt {}: mean difference {:.3e}",
            p.photo,
            p.prompt,
            p.whole.mean_abs
        );
    }
}

#[test]
fn the_masks_agree_with_metas_on_all_prompts_and_the_same_mask_is_chosen() {
    let Some(all) = results() else { return };
    let prompts: Vec<&PromptResult> = all.iter().flat_map(|p| &p.prompts).collect();
    let (_, mean_dec) = worst_and_mean(prompts.iter().map(|p| p.iou_decoder));
    let (_, mean_whole) = worst_and_mean(prompts.iter().map(|p| p.iou_whole));
    let min_dec = prompts.iter().map(|p| p.iou_decoder).fold(1.0, f64::min);
    let min_whole = prompts.iter().map(|p| p.iou_whole).fold(1.0, f64::min);
    let multi = prompts.iter().filter(|p| p.multimask).count();
    let fallbacks = prompts
        .iter()
        .filter(|p| !p.multimask && p.token_whole != 0)
        .count();
    println!(
        "mask IoU at logit > 0, 256 x 256: from Meta's maps min {min_dec:.5} (mean {mean_dec:.5}); whole path min {min_whole:.5} (mean {mean_whole:.5}); {} prompts, {multi} with the multimask output, {fallbacks} single masks that fell back to a multimask token",
        prompts.len()
    );
    for p in &prompts {
        assert!(
            p.iou_decoder >= MASK_IOU_MIN,
            "{} prompt {}: IoU from Meta's maps {:.5}",
            p.photo,
            p.prompt,
            p.iou_decoder
        );
        assert!(
            p.iou_whole >= MASK_IOU_MIN,
            "{} prompt {}: whole-path IoU {:.5}",
            p.photo,
            p.prompt,
            p.iou_whole
        );
        assert!(
            p.chosen_decoder,
            "{} prompt {}: another mask chosen from Meta's maps",
            p.photo, p.prompt
        );
        assert!(
            p.chosen_whole,
            "{} prompt {}: another mask chosen on the whole path",
            p.photo, p.prompt
        );
    }
}

// -------------------------------------------------------------------------------------------------
// Determinism
// -------------------------------------------------------------------------------------------------

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn features_bits(f: &Features) -> Vec<u32> {
    let mut b = bits(f.image_embed());
    b.extend(bits(f.high_res_0()));
    b.extend(bits(f.high_res_1()));
    b
}

fn decoded_bits(d: &Decoded) -> Vec<u32> {
    let mut b = bits(&d.logits);
    b.extend(bits(&d.iou));
    b.push(d.object_score.to_bits());
    b.push(d.chosen as u32);
    b
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(f)
}

#[test]
fn a_second_run_is_bit_identical_and_threads_do_not_change_a_bit() {
    if !heavy_ok() {
        return;
    }
    let Some(model) = model() else { return };
    let Some(ids) = photo_ids() else { return };
    // One photo is enough for the bits (the whole network runs on it); a non-square one tests the picture-to-tensor path.
    let id = ids
        .iter()
        .find(|i| i.as_str() == "bicycle")
        .unwrap_or(&ids[0]);
    let photo = read_npz(&fixtures_dir().join(format!("{id}.npz")));
    let (w, h, rgb) = picture_of(&photo);
    let picture = Picture::new(w, h, &rgb).expect("picture");
    let prompts: Vec<Prompt> = (1..=3).map(|k| prompt_of(&photo, k)).collect();

    let run = |threads: usize| -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
        in_pool(threads, || {
            let input = bits(&ym_ml::sam2::preprocess(&picture));
            let features = model.encode(&picture).expect("encode");
            let masks = prompts
                .iter()
                .map(|p| decoded_bits(&model.decode_raw(&features, p).expect("decode")))
                .collect();
            (input, features_bits(&features), masks)
        })
    };
    let first = run(4);
    let again = run(4);
    assert!(first == again, "{id}: a second run differs");
    for threads in [1, 2, 8] {
        let other = run(threads);
        assert!(
            first.0 == other.0,
            "{id}: the resized tensor differs on {threads} threads"
        );
        assert!(
            first.1 == other.1,
            "{id}: the features differ on {threads} threads"
        );
        assert!(
            first.2 == other.2,
            "{id}: a decoded mask differs on {threads} threads"
        );
    }
    println!(
        "{id}: preprocess, encoder and three decodes bit-identical on 4 threads twice and on 1, 2 and 8 threads"
    );
}

// -------------------------------------------------------------------------------------------------
// Behaviour that needs the weights but not the fixtures
// -------------------------------------------------------------------------------------------------

#[test]
fn a_damaged_or_foreign_weights_file_is_an_error_not_a_panic() {
    if !weights_dir().join(WEIGHTS_FILE).exists() {
        let msg = "SKIPPED: no SAM 2 weights (run tools/fetch/sam2.sh)";
        assert!(!require(), "{msg}");
        println!("{msg}");
        return;
    }
    let bytes = std::fs::read(weights_dir().join(WEIGHTS_FILE)).expect("weights");
    // A checksum that is not the file's.
    let wrong = "0".repeat(64);
    assert!(matches!(
        Sam2::from_bytes(&bytes, Some(&wrong)),
        Err(MlError::Checksum { .. })
    ));
    // A truncated file, nothing at all, and text.
    assert!(Sam2::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    assert!(Sam2::from_bytes(&bytes[..100], None).is_err());
    assert!(Sam2::from_bytes(&[], None).is_err());
    assert!(Sam2::from_bytes(b"not a safetensors file at all", None).is_err());
    // The header with another format or version is refused by name (the tensors are left as they are).
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).expect("json header");
    let swap = |from: &str, to: &str| -> Vec<u8> {
        assert!(header.contains(from), "{from} in the header");
        // The replacement keeps the header's length by trimming/padding with spaces (JSON allows trailing spaces).
        let new = header.replacen(from, to, 1);
        let padded = format!("{new:<width$}", width = header_len);
        assert_eq!(padded.len(), header_len, "the replacement fits");
        let mut out = bytes[..8].to_vec();
        out.extend_from_slice(padded.as_bytes());
        out.extend_from_slice(&bytes[8 + header_len..]);
        out
    };
    match Sam2::from_bytes(
        &swap(
            "\"format\":\"yellowmoon-sam2\"",
            "\"format\":\"yellowmoon-sam3\"",
        ),
        None,
    ) {
        Err(MlError::Weights(m)) => assert!(m.contains("yellowmoon-sam2"), "{m}"),
        other => panic!("another format: {other:?}"),
    }
    match Sam2::from_bytes(&swap("\"version\":\"1\"", "\"version\":\"2\""), None) {
        Err(MlError::Unsupported(m)) => assert!(m.contains("version"), "{m}"),
        other => panic!("another version: {other:?}"),
    }
    // An architecture the runtime does not implement (a different window in the config).
    match Sam2::from_bytes(
        &swap(
            "\\\"window_spec\\\":[8,4,14,7]",
            "\\\"window_spec\\\":[8,4,16,7]",
        ),
        None,
    ) {
        Err(MlError::Unsupported(m)) => assert!(m.contains("window_spec"), "{m}"),
        other => panic!("another architecture: {other:?}"),
    }
}

#[test]
fn a_click_that_finds_nothing_says_why() {
    if !heavy_ok() {
        return;
    }
    let Some(model) = model() else { return };
    // A flat grey picture has no object: whatever the model says, the answer is one of the two refusals or a mask, never an error.
    let rgb = vec![128u8; 64 * 48 * 3];
    let picture = Picture::new(64, 48, &rgb).expect("picture");
    let features = model.encode(&picture).expect("encode");
    match model
        .decode(&features, &Prompt::click(0.5, 0.5))
        .expect("decode")
    {
        Decoding::Found(d) => {
            assert!(d.object_score > 0.0 && d.area() >= ym_ml::sam2::OBJECT_MIN_AREA)
        }
        Decoding::NothingFound {
            reason,
            object_score,
            area,
        } => {
            println!(
                "flat grey picture: nothing found ({reason:?}, object score {object_score:.2}, area {area:.5})"
            );
            assert!(object_score <= 0.0 || area < ym_ml::sam2::OBJECT_MIN_AREA);
        }
    }
    // An empty prompt and a coordinate that is not a number are refused with an error.
    assert!(matches!(
        model.decode(&features, &Prompt::default()),
        Err(MlError::Input(_))
    ));
    assert!(matches!(
        model.decode(&features, &Prompt::click(f32::NAN, 0.5)),
        Err(MlError::Input(_))
    ));
}
