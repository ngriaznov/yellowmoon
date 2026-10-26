//! Test support: an [`AppApi`] over a temporary app folder, the real data folder and a recorder in
//! place of the webview, and synthetic ARW files to open in it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ym_raw::testgen::{ArwBuilder, gradient_mosaic};

use crate::api::{AppApi, AppConfig};
use crate::cli_link::CliLink;
use crate::cli_link::NoElevation;
use crate::events::Recorder;
use crate::events::{AppEvent, Emitter};
use crate::viewport::{FallbackSink, FrameStore};
use crate::wire::*;
use ym_core::auto_provider::ProviderAdvisors;
use ym_looks::import::watch::WatchConfig;

pub(crate) const WAIT: Duration = Duration::from_secs(90);

/// Waits for a condition; fails the test after `WAIT`.
pub(crate) fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(15));
    }
}

/// A synthetic A7R IV file: a smooth gradient with noise, the as-shot white balance of daylight.
pub(crate) fn write_arw(dir: &Path, name: &str, seed: u64) -> PathBuf {
    let (w, h) = (192_u32, 128_u32);
    let mosaic = gradient_mosaic(w, h, 512, 12_000, seed);
    let path = dir.join(name);
    ArwBuilder::new("ILCE-7RM4", w, h)
        .black([512; 4])
        .white([16_300; 3])
        .wb(Some([2_100, 1_024, 1_024, 1_600]))
        .iso(400)
        .exposure_time(1, 125)
        .f_number(28, 10)
        .capture_time("2026:10:04 17:42:10", "00", None)
        .write_to(&path, &mosaic)
        .expect("write the ARW");
    path
}

pub(crate) struct Harness {
    pub(crate) api: Arc<AppApi>,
    pub(crate) events: Arc<Recorder>,
    pub(crate) frames: Arc<FrameStore>,
    pub(crate) photos: tempfile::TempDir,
    pub(crate) app_root: tempfile::TempDir,
    pub(crate) home: tempfile::TempDir,
}

impl Harness {
    pub(crate) fn new() -> Self {
        Self::with_watch(WatchConfig::default())
    }

    pub(crate) fn with_watch(watch: WatchConfig) -> Self {
        Self::with(watch, ProviderAdvisors::default())
    }

    /// An app whose Apple Intelligence slot asks `advisor` (`docs/19` section 12: `FakeAdvisor`, or a
    /// scripted one); the other harnesses have none, as the golden and parity tests need.
    pub(crate) fn with_advisor(advisor: Arc<dyn ym_core::apple_auto::AutoAdvisor>) -> Self {
        Self::with_advisors(
            ProviderAdvisors::default().with(ym_core::auto_provider::AutoProvider::Apple, advisor),
        )
    }

    /// An app with the given providers (`docs/22` section 12.6: scripted or fake ones for Apple
    /// Intelligence, Claude Code and Antigravity).
    pub(crate) fn with_advisors(advisors: ProviderAdvisors) -> Self {
        Self::with(WatchConfig::default(), advisors)
    }

    /// An app with the given providers over a `preferences.json` written by hand before it starts (a
    /// file with a choice the app itself would not save, such as a program without its consent).
    pub(crate) fn with_preferences_file(advisors: ProviderAdvisors, preferences: &str) -> Self {
        Self::with_files(WatchConfig::default(), advisors, Some(preferences))
    }

    fn with(watch: WatchConfig, advisors: ProviderAdvisors) -> Self {
        Self::with_files(watch, advisors, None)
    }

    fn with_files(
        watch: WatchConfig,
        advisors: ProviderAdvisors,
        preferences: Option<&str>,
    ) -> Self {
        let photos = tempfile::tempdir().expect("photos");
        let app_root = tempfile::tempdir().expect("app root");
        if let Some(text) = preferences {
            std::fs::write(app_root.path().join("preferences.json"), text)
                .expect("preferences.json");
        }
        let home = tempfile::tempdir().expect("home");
        let events = Arc::new(Recorder::new());
        let frames = Arc::new(FrameStore::new());
        let emitter: Arc<dyn Emitter> = events.clone();
        let sink: Arc<dyn crate::viewport::FrameSink> =
            Arc::new(FallbackSink::new(Arc::clone(&frames), Arc::clone(&emitter)));
        let tool = home.path().join("app").join("ym");
        std::fs::create_dir_all(tool.parent().expect("parent")).expect("mkdir");
        std::fs::write(&tool, b"#!/bin/sh\n").expect("write the tool");
        let config = AppConfig {
            data_root: ym_core::data_dir::find_data_dir().expect("the data folder"),
            app_root: app_root.path().to_path_buf(),
            home: Some(home.path().to_path_buf()),
            gpu: false,
            thumbnails: true,
            decode_threads: Some(2),
            watch,
            advisors,
        };
        // Only the folders of the test: `CliLink::new` also names the real `/usr/local/bin` and
        // `/opt/homebrew/bin` of a Mac, where a test would look at (and replace) the link of the
        // person running it.
        let mut cli = CliLink::new(tool, home.path());
        cli.system_link = None;
        cli.other_links.clear();
        let api = AppApi::start(
            config,
            emitter,
            move |_| sink,
            Arc::clone(&frames),
            Box::new(NoElevation),
            Some(cli),
        )
        .expect("the app starts");
        Self {
            api,
            events,
            frames,
            photos,
            app_root,
            home,
        }
    }

    /// Writes `n` photos into the photo folder and opens the folder.
    pub(crate) fn open(&self, n: u64) -> SessionState {
        for i in 1..=n {
            write_arw(self.photos.path(), &format!("DSC{i:05}.ARW"), i);
        }
        self.api
            .open_paths(&[self.photos.path().to_string_lossy().into_owned()])
            .expect("open the folder")
    }

    /// Waits until every photo is developed.
    pub(crate) fn develop(&self) -> SessionState {
        wait_until("the photos to develop", || {
            self.api.session_state().is_ok_and(|s| {
                !s.photos.is_empty() && s.photos.iter().all(|p| p.status == PhotoStatus::Ready)
            })
        });
        self.api.session_state().expect("state")
    }

    pub(crate) fn first(&self) -> PhotoId {
        self.api.session_state().expect("state").photos[0].id
    }

    /// Waits for the viewport to present a frame after `since` frames.
    pub(crate) fn frame_after(&self, since: u64) -> crate::wire::ViewportFrame {
        wait_until("a viewport frame", || {
            self.api.viewport().frames_presented() > since
        });
        self.events
            .events()
            .into_iter()
            .rev()
            .find_map(|e| match e {
                AppEvent::ViewportFrame(f) => Some(f),
                _ => None,
            })
            .expect("a viewport_frame event")
    }

    /// Waits for a frame of this size that is newer than `since` frames.
    pub(crate) fn frame_of_size(&self, width: u32, height: u32) -> crate::wire::ViewportFrame {
        let found = self.events.wait_for(WAIT, |e| match e {
            AppEvent::ViewportFrame(f) if (f.width, f.height) == (width, height) => Some(f.clone()),
            _ => None,
        });
        found.unwrap_or_else(|| panic!("no frame of {width} x {height}"))
    }

    pub(crate) fn rect(&self, width: f64, height: f64) {
        self.api
            .viewport_set_rect(&ViewportRect {
                x: Num(10.0),
                y: Num(20.0),
                width: Num(width),
                height: Num(height),
                scale: Num(1.0),
            })
            .expect("set the rect");
    }
}

/// The `viewport_busy` events `events` has, in order.
pub(crate) fn busy_events(events: &Recorder) -> Vec<ViewportBusy> {
    events
        .events()
        .into_iter()
        .filter_map(|e| match e {
            AppEvent::ViewportBusy(b) => Some(b),
            _ => None,
        })
        .collect()
}

pub(crate) fn png_size(png: &[u8]) -> (u32, u32) {
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
    let w = u32::from_be_bytes(png[16..20].try_into().expect("width"));
    let h = u32::from_be_bytes(png[20..24].try_into().expect("height"));
    (w, h)
}

pub(crate) fn jpeg_job(destination: &Path) -> ExportJob {
    ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            format: ExportFormat::Jpeg,
            jpeg_quality: 90,
            bit_depth: crate::wire::BitDepth(16),
            tiff_compression: TiffCompression::Lzw,
            color_space: crate::wire::ColorSpace::Srgb,
            size: ExportSize::Full,
            destination: Some(destination.to_string_lossy().into_owned()),
            name_template: "{name}_{look}".to_owned(),
            if_exists: IfExists::KeepBoth,
        },
    }
}
