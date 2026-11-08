//! The app's path to the Depth mask (`docs/23-nodes.md` sections 3.8 and 4.8, task D-04): the core asks for the depth map when a
//! Depth row appears on the photo being looked at, a copy of the row pasted on another photo is worked out again from that photo's own
//! pictures, and `mask_kinds` lists Depth where the model is. A fake model stands in for the weights (see `mask_depth.rs` for the
//! renderer's side).
mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::{DepthModel, PhotoSource};
use ym_core::session::OpenOptions;
use ym_core::settings::{MaskKind, SettingsPatch};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

/// A photo of 1200 x 800 with a bright square at `(x, 0.5)` of the sensor.
fn photo_at(x: f64) -> Developed {
    let (w, h) = (1200u32, 800u32);
    let scene = Scene::new(w, h, light(PROFILE, 5500.0), Layout::Uniform, 0.08).noise(0.0, 0.0);
    let mut mosaic = scene.mosaic();
    let code = (scene.black + 0.6 * (scene.white_level - scene.black)) as u16;
    let (cx, cy, half) = (x * f64::from(w), 0.5 * f64::from(h), 0.07);
    for yy in 0..h {
        for xx in 0..w {
            let (dx, dy) = (f64::from(xx) + 0.5 - cx, f64::from(yy) + 0.5 - cy);
            if dx.abs() <= half * f64::from(w) && dy.abs() <= half * f64::from(h) {
                mosaic[(yy * w + xx) as usize] = code;
            }
        }
    }
    let dir = tempfile::tempdir().expect("temp dir");
    scene
        .builder(MODEL)
        .write_to(&dir.path().join("DSC00001.ARW"), &mosaic)
        .expect("write the synthetic ARW");
    Developed::open(dir)
}

fn wait_until(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(secs);
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn the_core_asks_for_the_map_when_the_row_appears_and_a_paste_works_it_out_again() {
    let runs = Arc::new(AtomicUsize::new(0));
    let model = {
        let runs = Arc::clone(&runs);
        // The depth of a cell is the picture's luma there: the bright square is the nearest thing.
        DepthModel::fake(
            "core",
            Arc::new(move |w, h, rgb, _| {
                runs.fetch_add(1, Ordering::SeqCst);
                let (nw, nh) = ym_ml::depth::capped_network_size(w, h);
                let (cw, ch) = (nw / ym_ml::depth::PATCH, nh / ym_ml::depth::PATCH);
                let mut d = Vec::with_capacity(cw * ch);
                for cy in 0..ch {
                    for cx in 0..cw {
                        let x = (((cx as f64 + 0.5) / cw as f64) * w as f64) as usize;
                        let y = (((cy as f64 + 0.5) / ch as f64) * h as f64) as usize;
                        d.push(f32::from(rgb[3 * (y.min(h - 1) * w + x.min(w - 1)) + 1]) / 255.0);
                    }
                }
                Ok((cw, ch, d))
            }),
        )
    };
    let app = tempfile::tempdir().expect("app root");
    let mut o = CoreOptions::new(data_root(), app.path().to_path_buf());
    o.decode_threads = 2;
    o.thumbnails = false;
    o.render.depth_model = Some(model);
    let core = Core::open(o).expect("core");
    // The menu lists Depth right after Background where the model is.
    let kinds = core.mask_kinds();
    let at = kinds
        .iter()
        .position(|k| *k == MaskKind::Background)
        .expect("Background");
    assert_eq!(kinds[at + 1], MaskKind::Depth);
    assert!(
        core.mask_unavailable()
            .iter()
            .all(|(k, _)| *k != MaskKind::Depth)
    );

    // Two photos with their bright squares in different places.
    let (left, right) = (photo_at(0.3), photo_at(0.7));
    let mut ids = Vec::new();
    for d in [&left, &right] {
        let opened = core
            .session
            .open(&[d.dir.path().to_path_buf()], OpenOptions::default());
        assert!(core.session.wait_idle(Duration::from_secs(120)));
        ids.push(opened.added[0]);
    }
    let (a, b) = (ids[0], ids[1]);
    core.session.set_current(a).expect("current");
    core.renderer.navigate(a);
    assert!(!core.renderer.depth_worker_alive(a), "no row, no worker");

    // The patch that makes the row asks for the map in the background.
    let patch: SettingsPatch = serde_json::from_value(
        json!({ "nodes": { "1": { "masks": { "items": [ { "kind": "depth", "far": 0.1 } ] } } } }),
    )
    .expect("a patch");
    core.update_settings(a, &patch).expect("the row is added");
    assert!(
        core.renderer.depth_worker_alive(a),
        "the core asked at once"
    );
    wait_until("the worker", 60, || {
        !core.renderer.depth_worker_alive(a) && core.renderer.depth_networks(a) == 1
    });
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let src_a = PhotoSource::from_session(&core.session, a).expect("source");
    let map_a = core
        .renderer
        .depth_map(&src_a, &CancelToken::new())
        .expect("a map");
    // A slider is no new map.
    let slider: SettingsPatch = serde_json::from_value(
        json!({ "nodes": { "1": { "masks": { "edit": { "0": { "far": 0.3, "feather": 0.2 } } } } } }),
    )
    .expect("a patch");
    core.update_settings(a, &slider).expect("a slider");
    assert!(!core.renderer.depth_worker_alive(a));
    assert_eq!(runs.load(Ordering::SeqCst), 1);

    // Copy, then paste onto the other photo: the row is the same, its map is the other photo's.
    let copied = core.copy_settings(a).expect("copy");
    core.session.set_current(b).expect("current");
    core.renderer.navigate(b);
    assert!(!core.renderer.depth_worker_alive(b));
    let report = core.paste_settings(&[b], &copied).expect("paste");
    assert_eq!(report.pasted.len(), 1);
    wait_until("the worker for the pasted photo", 60, || {
        core.renderer.depth_networks(b) == 1 && !core.renderer.depth_worker_alive(b)
    });
    assert_eq!(runs.load(Ordering::SeqCst), 2, "one map for each photo");
    let src_b = PhotoSource::from_session(&core.session, b).expect("source");
    let row = &src_b.settings.nodes.corrector(1).expect("Main").masks.items;
    assert_eq!(row.len(), 1, "the pasted photo has the row");
    let map_b = core
        .renderer
        .depth_map(&src_b, &CancelToken::new())
        .expect("a map");
    assert_ne!(map_a.key, map_b.key, "another picture, another map");
    // The square is where each photo has it: the nearest cells are left in one and right in the other.
    let centroid_x = |m: &ym_color::mask_depth::DepthMap| -> f64 {
        let (mut sum, mut sx) = (0.0f64, 0.0f64);
        for y in 0..m.height {
            for x in 0..m.width {
                let near = m.r(y * m.width + x);
                if near > 0.9 {
                    sum += 1.0;
                    sx += (x as f64 + 0.5) / m.width as f64;
                }
            }
        }
        sx / sum
    };
    let (ca, cb) = (centroid_x(&map_a), centroid_x(&map_b));
    assert!(
        ca < 0.5 && cb > 0.5 && cb - ca > 0.2,
        "{ca} {cb}: the paste used the new photo's pixels"
    );
}

/// "Try again" (`docs/23` section 4.8): a map that failed is asked for again by `retry_depth` and by nothing else, with no change to the
/// photo's settings; the page hears the worker start and end again. A photo whose map is made, one that is not the photo being looked
/// at and one with no failure are left alone.
#[test]
fn retry_depth_starts_a_failed_map_again_and_nothing_else_does() {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use ym_core::pipeline::DepthPhase;

    let broken = Arc::new(AtomicBool::new(true));
    let runs = Arc::new(AtomicUsize::new(0));
    let model = {
        let (broken, runs) = (Arc::clone(&broken), Arc::clone(&runs));
        DepthModel::fake(
            "retry",
            Arc::new(move |w, h, _, _| {
                runs.fetch_add(1, Ordering::SeqCst);
                if broken.load(Ordering::SeqCst) {
                    return Err("the model stopped before it finished".to_owned());
                }
                let (nw, nh) = ym_ml::depth::capped_network_size(w, h);
                let (cw, ch) = (nw / ym_ml::depth::PATCH, nh / ym_ml::depth::PATCH);
                let d = (0..cw * ch).map(|i| (i % cw) as f32).collect();
                Ok((cw, ch, d))
            }),
        )
    };
    let app = tempfile::tempdir().expect("app root");
    let mut o = CoreOptions::new(data_root(), app.path().to_path_buf());
    o.decode_threads = 2;
    o.thumbnails = false;
    o.render.depth_model = Some(model);
    let core = Core::open(o).expect("core");
    let heard = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&heard);
    core.renderer
        .set_depth_listener(Arc::new(move |_, phase| sink.lock().unwrap().push(phase)));

    let (one, other) = (photo_at(0.3), photo_at(0.7));
    let mut ids = Vec::new();
    for d in [&one, &other] {
        let opened = core
            .session
            .open(&[d.dir.path().to_path_buf()], OpenOptions::default());
        assert!(core.session.wait_idle(Duration::from_secs(120)));
        ids.push(opened.added[0]);
    }
    let (a, b) = (ids[0], ids[1]);
    core.session.set_current(a).expect("current");
    core.renderer.navigate(a);
    // Nothing to retry before a failure: no worker starts.
    core.retry_depth(a).expect("retry with no row");
    assert!(!core.renderer.depth_worker_alive(a));
    assert_eq!(runs.load(Ordering::SeqCst), 0);

    let patch: SettingsPatch = serde_json::from_value(
        json!({ "nodes": { "1": { "masks": { "items": [ { "kind": "depth" } ] } } } }),
    )
    .expect("a patch");
    core.update_settings(a, &patch).expect("the row is added");
    let sentence = "The Depth mask can't be made: the model stopped before it finished.";
    wait_until("the failure", 60, || {
        heard.lock().unwrap().len() == 2 && !core.renderer.depth_worker_alive(a)
    });
    assert_eq!(
        heard.lock().unwrap()[1],
        DepthPhase::Finished {
            failed: Some(sentence.to_owned())
        }
    );
    // Preparing again (any edit of the row) asks for nothing: the failure is kept for the key.
    core.prepare_masks(a);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(runs.load(Ordering::SeqCst), 1, "not asked again by itself");

    // A photo that is not the one being looked at is left alone.
    core.retry_depth(b).expect("retry on another photo");
    assert!(!core.renderer.depth_worker_alive(b));

    // Retry while the model still fails: the worker runs again and tells the same failure again.
    core.retry_depth(a).expect("retry");
    wait_until("the second failure", 60, || {
        heard.lock().unwrap().len() == 4 && !core.renderer.depth_worker_alive(a)
    });
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert_eq!(
        heard.lock().unwrap()[3],
        DepthPhase::Finished {
            failed: Some(sentence.to_owned())
        }
    );

    // The model works now: Try again makes the map and the failure is gone.
    broken.store(false, Ordering::SeqCst);
    core.retry_depth(a).expect("retry");
    wait_until("the map", 60, || {
        heard.lock().unwrap().len() == 6 && !core.renderer.depth_worker_alive(a)
    });
    assert_eq!(
        heard.lock().unwrap()[5],
        DepthPhase::Finished { failed: None }
    );
    assert_eq!(runs.load(Ordering::SeqCst), 3);
    let src = PhotoSource::from_session(&core.session, a).expect("source");
    assert!(core.renderer.depth_ready(&src));
    assert_eq!(core.renderer.depth_failure(&src), None);
    // A map that is made is not made again by a retry.
    core.retry_depth(a).expect("retry when ready");
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(runs.load(Ordering::SeqCst), 3);
    assert_eq!(heard.lock().unwrap().len(), 6);
}
