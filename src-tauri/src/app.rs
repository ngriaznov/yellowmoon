//! The application: plugins, the managed state, the menu, the frame protocol and quitting.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::http::Response;
use tauri::menu::MenuEvent;
use tauri::{App, AppHandle, Manager, RunEvent, Runtime, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_specta::Event;

use crate::api::{AppApi, AppConfig};
use crate::cli_link::{CliLink, Elevate};
use crate::commands::{self, AppState};
use crate::events::{Emitter, TauriEmitter};
use crate::menu::{self, Do};
use crate::perf;
use crate::system;
use crate::viewport::{FRAME_SCHEME, FallbackSink, FrameSink, FrameStore};
use crate::wire::{AppError, ErrorScope, MenuActionEvent};

/// Set when the person has agreed to quit with an export running.
static QUIT_CONFIRMED: AtomicBool = AtomicBool::new(false);

/// Set while the question is on screen, so that a second ⌘Q or a second click on the close button
/// does not ask it again.
static ASKING: AtomicBool = AtomicBool::new(false);

/// What the app asks before it quits with an export running.
const QUIT_MESSAGE: &str = "An export is still running. If you quit now, it stops and the file being written is deleted. The files already saved stay.";

/// The question before "Report a Problem" copies the report.
const REPORT_MESSAGE: &str = "Report a Problem opens the bug report form on GitHub. Copy the output of `ym doctor` to the clipboard first, so you can paste it into the form? It lists your graphics card, the camera profiles, the version and the folders where the app keeps its settings. It contains no photos and no file names.";

/// Runs the application until the last window closes.
pub fn run() {
    let frames = Arc::new(FrameStore::new());
    let protocol_frames = Arc::clone(&frames);
    let builder = commands::builder();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        // Asynchronous: the handler of a synchronous scheme runs on the main thread (wry calls it from
        // `webView:startURLSchemeTask:`), and a request copies a PNG under a mutex. The filmstrip
        // asks for hundreds of pictures at once when a folder opens, and the Look tab for as many
        // whenever it is shown, so the copies are made on a blocking thread and only the answer
        // goes back to the main thread.
        .register_asynchronous_uri_scheme_protocol(FRAME_SCHEME, move |_ctx, request, responder| {
            serve_frame(
                Arc::clone(&protocol_frames),
                request.uri().path().to_owned(),
                move |response| responder.respond(response),
            );
        })
        .invoke_handler(builder.invoke_handler())
        .menu(menu::build)
        .on_menu_event(on_menu_event)
        .setup(move |app| {
            builder.mount_events(app);
            let state = start(app, &frames);
            app.manage(state);
            Ok(())
        })
        .build(tauri::generate_context!());
    match app {
        Ok(app) => app.run(on_run_event),
        Err(e) => {
            eprintln!("Yellowmoon could not start: {e}");
            std::process::exit(1);
        }
    }
}

/// Builds the command handlers, or remembers why that was not possible.
fn start(app: &mut App, frames: &Arc<FrameStore>) -> AppState {
    let config = match AppConfig::from_environment() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Yellowmoon could not start: {e}");
            return AppState::failed(e);
        }
    };
    crate::logging::init(&config.app_root);
    tracing::info!(
        version = ym_core::version::app_version(),
        "starting Yellowmoon"
    );
    if perf::enabled() {
        perf::watch_main_thread(app.handle().clone());
    }
    let emitter: Arc<dyn Emitter> = Arc::new(TauriEmitter::new(app.handle().clone()));
    let sink = make_sink(app, frames, &emitter);
    let cli = CliLink::locate()
        .map_err(|e| tracing::warn!(error = %e, "no command-line tool next to the app"))
        .ok();
    match AppApi::start(config, emitter, sink, Arc::clone(frames), elevator(), cli) {
        Ok(api) => {
            #[cfg(debug_assertions)]
            smoke_test(app.handle().clone(), Arc::clone(&api));
            AppState::ready(api)
        }
        Err(e) => {
            tracing::error!(error = %e, "Yellowmoon could not start");
            AppState::failed(e)
        }
    }
}

/// The viewport backend: the native layer on macOS, frames sent to the webview elsewhere (and
/// whenever the native layer cannot be made).
fn make_sink(
    app: &App,
    frames: &Arc<FrameStore>,
    emitter: &Arc<dyn Emitter>,
) -> Arc<dyn FrameSink> {
    // `YELLOWMOON_VIEWPORT=fallback` asks for the frames-to-webview path even on a Mac, to compare
    // the two backends (docs/11).
    let forced_fallback = std::env::var("YELLOWMOON_VIEWPORT").is_ok_and(|v| v == "fallback");
    #[cfg(target_os = "macos")]
    if !forced_fallback {
        match crate::native::NativeSink::create(app.handle(), "main") {
            Ok(sink) => {
                tracing::info!("native viewport");
                return Arc::new(sink);
            }
            Err(e) => {
                tracing::warn!(error = %e, "no native viewport; sending frames to the webview");
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = app;
    tracing::info!(forced = forced_fallback, "fallback viewport");
    Arc::new(FallbackSink::new(Arc::clone(frames), Arc::clone(emitter)))
}

/// Gets an administrator's rights for the command-line tool's link where the system has a prompt.
fn elevator() -> Box<dyn Elevate + Send + Sync> {
    #[cfg(target_os = "macos")]
    {
        Box::new(crate::cli_link::Osascript)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Box::new(crate::cli_link::NoElevation)
    }
}

/// Answers a request of the `ym-frame` scheme without making the thread that got it (the main
/// thread) wait: the response is made on a blocking thread, which may have to wait for the store's
/// locks, and `respond` is called from there.
fn serve_frame(
    store: Arc<FrameStore>,
    path: String,
    respond: impl FnOnce(Response<Cow<'static, [u8]>>) + Send + 'static,
) {
    // Started here, on the thread that was called: the time to get a blocking thread counts.
    let timer = perf::timer(perf::Timed::FrameHandler);
    // The handle is dropped: the task runs to its end and answers by itself.
    drop(tauri::async_runtime::spawn_blocking(move || {
        let response = frame_response(&store, &path);
        drop(timer);
        respond(response);
    }));
}

/// The answer for `path`, `/<launch>/frame-<n>.png` (a frame of the fallback viewport),
/// `/<launch>/thumb-<photo>-<version>.png` (a thumbnail of the filmstrip) or
/// `/<launch>/look-<photo>-<version>.png` (a tile of the Look tab).
///
/// Thumbnails and tiles that exist are cached for good: their address holds the launch id and a
/// version that is never used twice, so what an address names never changes, and the webview need
/// not ask again when the Look tab or the filmstrip is shown again. Frames and every 404 are not
/// cached (see [`crate::viewport::Cache`]).
fn frame_response(store: &FrameStore, path: &str) -> Response<Cow<'static, [u8]>> {
    let served = store.serve(path);
    Response::builder()
        .status(served.status)
        .header("Content-Type", served.mime)
        .header("Cache-Control", served.cache.header())
        .header("Access-Control-Allow-Origin", "*")
        .body(served.body)
        .unwrap_or_else(|_| Response::new(Cow::Borrowed(&[][..])))
}

// --- Menu -------------------------------------------------------------------------------------

fn on_menu_event<R: Runtime>(app: &AppHandle<R>, event: MenuEvent) {
    match menu::action_for(event.id().as_ref()) {
        Some(Do::Webview(action)) => {
            if let Err(e) = (MenuActionEvent { action }).emit(app) {
                tracing::warn!(error = %e, "menu action not delivered");
            }
        }
        Some(Do::OpenGithub) => report(app, system::open_url(system::GITHUB_URL)),
        Some(Do::ShowLooksFolder) => {
            let Some(api) = api_of(app) else { return };
            report(
                app,
                api.looks_folder()
                    .and_then(|folder| system::reveal_in_finder(&folder)),
            );
        }
        Some(Do::ReportProblem) => report_a_problem(app),
        None => {}
    }
}

fn api_of<R: Runtime>(app: &AppHandle<R>) -> Option<Arc<AppApi>> {
    app.try_state::<AppState>().and_then(|s| s.api().ok())
}

/// Shows an error of something the shell did on its own (a menu item) the way a command's error is
/// shown: as the `error` event.
fn report<R: Runtime>(app: &AppHandle<R>, result: Result<(), AppError>) {
    if let Err(e) = result {
        let _ = crate::wire::ErrorEvent::from(e).emit(app);
    }
}

/// Help › Report a Problem: asks, copies `ym doctor` if the person agrees, opens the form.
fn report_a_problem<R: Runtime>(app: &AppHandle<R>) {
    let handle = app.clone();
    app.dialog()
        .message(REPORT_MESSAGE)
        .title("Report a Problem")
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Copy and open the form".to_owned(),
            "Just open the form".to_owned(),
        ))
        .show(move |copy| {
            std::thread::spawn(move || {
                if copy {
                    let result = api_of(&handle)
                        .and_then(|api| api.cli_source().map(std::path::Path::to_path_buf))
                        .ok_or_else(|| {
                            AppError::new(
                                ErrorScope::App,
                                "The report could not be made because the command-line tool isn't in this copy of Yellowmoon.",
                            )
                        })
                        .and_then(|ym| system::doctor_output(&ym))
                        .and_then(|text| system::copy_to_clipboard(&text));
                    report(&handle, result);
                }
                report(&handle, system::open_url(system::ISSUE_URL));
            });
        });
}

// --- Quitting ---------------------------------------------------------------------------------

/// Whether leaving the app has to be held back and put to the person first: an export is running
/// and they have not agreed yet.
fn quit_needs_question(export_busy: bool, confirmed: bool) -> bool {
    export_busy && !confirmed
}

fn export_is_running<R: Runtime>(app: &AppHandle<R>) -> bool {
    quit_needs_question(
        api_of(app).is_some_and(|a| a.export_busy()),
        QUIT_CONFIRMED.load(Ordering::SeqCst),
    )
}

fn on_run_event<R: Runtime>(app: &AppHandle<R>, event: RunEvent) {
    match event {
        RunEvent::ExitRequested { api, code, .. } => {
            // `app.exit(code)` is deliberate (the person agreed, or the app is failing).
            if code.is_none() && export_is_running(app) {
                api.prevent_exit();
                ask_to_quit(app);
            }
        }
        // The close button of the only window quits the app. Asking after the window is gone would
        // leave the export running in an app with no window to show it in, so the window is held
        // back too.
        RunEvent::WindowEvent {
            event: WindowEvent::CloseRequested { api, .. },
            ..
        } => {
            if export_is_running(app) {
                api.prevent_close();
                ask_to_quit(app);
            }
        }
        RunEvent::Exit => {
            if let Some(api) = api_of(app) {
                api.shutdown();
            }
        }
        _ => {}
    }
}

fn ask_to_quit<R: Runtime>(app: &AppHandle<R>) {
    if ASKING.swap(true, Ordering::SeqCst) {
        return;
    }
    let handle = app.clone();
    app.dialog()
        .message(QUIT_MESSAGE)
        .title("Quit Yellowmoon?")
        .kind(MessageDialogKind::Warning)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Stop the export and quit".to_owned(),
            "Keep exporting".to_owned(),
        ))
        .show(move |quit| {
            ASKING.store(false, Ordering::SeqCst);
            if quit {
                QUIT_CONFIRMED.store(true, Ordering::SeqCst);
                handle.exit(0);
            }
        });
}

// --- Development smoke test -------------------------------------------------------------------

/// With `YM_SMOKE=1` a development build checks itself a few seconds after it starts, logs the
/// result and exits: `xvfb-run -a target/debug/yellowmoon-app` is the headless smoke run.
///
/// `YM_SMOKE_OPEN=<path>` opens that folder or file before the page loads (the page then finds the
/// session as after a reload), presses "Open photos" on the Start screen once a photo is ready, as a
/// person would, and the check also wants photos in the session; `YM_SMOKE_HOLD=<s>` waits that
/// many seconds instead of 4. The viewport job of `.github/workflows/build.yml` uses both to capture
/// the screen of a Mac while it shows a photo.
///
/// `YM_PERF_SCRIPT=1` (with `YM_SMOKE_OPEN`) also drives the page like a hand would, once, from
/// `YM_PERF_SCRIPT_AT` seconds after the start (20 if not set): 240 ctrl+wheel and 240 pan wheel
/// events on the photo cell and 60 key presses on the Temperature slider, one every 8 ms or so
/// ([`SMOKE_PERF_SCRIPT`]). The log (`ym::perf` lines, see `perf`), the script's own numbers and the
/// page's `window.__ymPerf.snapshot()` (`src/perf.ts`, when the page has it) are what the viewport
/// job puts in its summary. It never fails the run: the numbers are for reading.
/// Presses the Start screen's "Open photos" button when it is there and enabled.
#[cfg(debug_assertions)]
const SMOKE_OPEN_PHOTOS: &str = r#"(() => {
  const b = [...document.querySelectorAll("button")].find(
    (b) => b.textContent.trim().startsWith("Open photos") && !b.disabled && b.getAttribute("aria-disabled") !== "true",
  );
  if (b) b.click();
})()"#;

/// The page at the middle of the photo cell: the root's `data-viewport`, the backgrounds, and every
/// element under that point with its background.
#[cfg(debug_assertions)]
const SMOKE_DESCRIBE_PAGE: &str = r##"(() => {
  const bg = (e) => (e ? getComputedStyle(e).backgroundColor : null);
  const cell = document.querySelector(".ym-photo-cell");
  const r = cell ? cell.getBoundingClientRect() : null;
  const name = (e) => e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + (e.classList.length ? "." + [...e.classList].join(".") : "");
  const stack = r ? document.elementsFromPoint(r.left + r.width / 2, r.top + r.height / 2).map((e) => name(e) + " " + bg(e)) : [];
  return JSON.stringify({
    viewport: document.documentElement.hasAttribute("data-native-viewport") ? "native" : null,
    native: cell ? cell.dataset.native ?? null : null,
    cell: r ? [r.left, r.top, r.width, r.height] : null,
    html: bg(document.documentElement),
    body: bg(document.body),
    stack,
  });
})()"##;

/// Drives the photo cell and a slider like a hand does (`YM_PERF_SCRIPT`): 240 ctrl+wheel events (zoom
/// in about the middle of the cell, a little out at the end), 240 pan events and 60 key presses on the
/// Temperature slider, 8 ms apart (a timer cannot keep to it: the milliseconds each phase took are
/// in the result). The result is left in `window.__ymPerfResult`; a script that is evaluated does not
/// wait for promises, so [`SMOKE_PERF_RESULT`] reads it later. What it cannot find is listed in
/// `missing`.
#[cfg(debug_assertions)]
const SMOKE_PERF_SCRIPT: &str = r##"(() => {
  if (window.__ymPerfRunning) return "already running";
  window.__ymPerfRunning = true;
  window.__ymPerfResult = null;
  const out = { phases: {}, missing: [] };
  const finish = () => { out.done = true; window.__ymPerfResult = out; };
  const cell = document.querySelector("[data-viewport]") || document.querySelector(".ym-photo-cell");
  if (!cell) { out.missing.push("photo cell"); finish(); return "no photo cell"; }
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const box = cell.getBoundingClientRect();
  const at = { clientX: box.left + box.width / 2, clientY: box.top + box.height / 2 };
  const wheel = (init) => cell.dispatchEvent(new WheelEvent("wheel", { bubbles: true, cancelable: true, ...at, ...init }));
  const phase = async (name, count, step) => {
    const t0 = performance.now();
    for (let i = 0; i < count; i++) { step(i); await sleep(8); }
    const ms = Math.round(performance.now() - t0);
    out.phases[name] = { events: count, ms, interval_ms: Math.round((10 * ms) / count) / 10 };
  };
  (async () => {
    await phase("ctrl_wheel", 240, (i) => wheel({ ctrlKey: true, deltaY: i < 200 ? -2 : 2 }));
    await phase("pan", 240, (i) => wheel({ deltaX: i % 2 ? 6 : -6, deltaY: i % 2 ? 4 : -4 }));
    const label = [...document.querySelectorAll("label")].find((l) => l.textContent.trim() === "Temperature");
    const slider = (label && label.htmlFor && document.getElementById(label.htmlFor)) || document.querySelector("input[type=range]:not(:disabled)");
    if (!slider) {
      out.missing.push("Temperature slider");
    } else {
      slider.focus();
      const key = (i) => (Math.floor(i / 10) % 2 ? "ArrowLeft" : "ArrowRight");
      await phase("slider_keys", 60, (i) => slider.dispatchEvent(new KeyboardEvent("keydown", { key: key(i), shiftKey: true, bubbles: true, cancelable: true })));
    }
    finish();
  })().catch((e) => { out.error = String(e); finish(); });
  return "started";
})()"##;

/// The result of [`SMOKE_PERF_SCRIPT`] as a string of JSON, or `"null"` while it runs.
#[cfg(debug_assertions)]
const SMOKE_PERF_RESULT: &str = r##"JSON.stringify(window.__ymPerfResult ?? null)"##;

/// The page's own counters (`window.__ymPerf.snapshot()`, `src/perf.ts`) as a string of JSON, or
/// `"null"` when the page has none. Read with `eval_with_callback`, which does not wait for promises:
/// `snapshot()` must answer at once.
#[cfg(debug_assertions)]
const SMOKE_PERF_PAGE: &str = r##"(() => {
  try {
    if (!window.__ymPerf || typeof window.__ymPerf.snapshot !== "function") return "null";
    const s = window.__ymPerf.snapshot();
    return typeof s === "string" ? s : JSON.stringify(s);
  } catch (e) {
    return JSON.stringify({ error: String(e) });
  }
})()"##;

/// The string of JSON a script left, out of what `eval_with_callback` hands the callback (a string
/// is handed over as JSON again); `None` for `"null"`.
#[cfg(debug_assertions)]
fn unwrap_script_json(callback_json: &str) -> Option<String> {
    let inner = serde_json::from_str::<String>(callback_json).ok()?;
    (inner != "null").then_some(inner)
}

#[cfg(debug_assertions)]
fn smoke_test<R: Runtime>(app: AppHandle<R>, api: Arc<AppApi>) {
    if std::env::var_os("YM_SMOKE").is_none() {
        return;
    }
    let open = std::env::var("YM_SMOKE_OPEN").ok();
    if let Some(path) = &open {
        match api.open_paths(std::slice::from_ref(path)) {
            Ok(state) => eprintln!("smoke: opened {} photos from {path}", state.photos.len()),
            Err(e) => eprintln!("smoke: could not open {path}: {}", e.message),
        }
    }
    let hold = std::env::var("YM_SMOKE_HOLD")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(4);
    // `YM_PERF_SCRIPT`: when to drive the page, in seconds after the start.
    let perf_at = std::env::var_os("YM_PERF_SCRIPT").map(|_| {
        std::env::var("YM_PERF_SCRIPT_AT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(20)
    });
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let until = std::time::Duration::from_secs(hold);
        let mut described = false;
        let mut perf_started = false;
        let perf_done = Arc::new(AtomicBool::new(false));
        while start.elapsed() < until {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // Once, after the photo had time to show: what the page has at the photo cell.
            if open.is_some()
                && !described
                && start.elapsed() > std::time::Duration::from_secs(14)
                && let Some(window) = app.get_webview_window("main")
            {
                described = true;
                let _ = window.eval_with_callback(SMOKE_DESCRIBE_PAGE, |json| {
                    eprintln!("smoke: page {json}");
                    // The script returns a string of JSON, which the callback gets as JSON again.
                    let page = serde_json::from_str::<String>(&json)
                        .ok()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
                    let clear = page
                        .as_ref()
                        .and_then(|p| p["stack"].as_array())
                        .is_some_and(|stack| {
                            !stack.is_empty()
                                && stack.iter().all(|e| {
                                    e.as_str().is_some_and(|e| e.ends_with("rgba(0, 0, 0, 0)"))
                                })
                        });
                    eprintln!(
                        "smoke: the page lets the viewport through: {}",
                        if clear { "yes" } else { "no" }
                    );
                });
            }
            // Once the photo is up: the page is driven like a hand would (`YM_PERF_SCRIPT`), then
            // polled for its result until it has one.
            if let Some(at) = perf_at
                && start.elapsed() >= std::time::Duration::from_secs(at)
                && let Some(window) = app.get_webview_window("main")
            {
                if !perf_started {
                    perf_started = true;
                    eprintln!("smoke: perf script starts");
                    let _ = window.eval(SMOKE_PERF_SCRIPT);
                } else if !perf_done.load(Ordering::SeqCst) {
                    let done = Arc::clone(&perf_done);
                    let _ = window.eval_with_callback(SMOKE_PERF_RESULT, move |json| {
                        if let Some(result) = unwrap_script_json(&json) {
                            eprintln!("smoke: perf script {result}");
                            done.store(true, Ordering::SeqCst);
                        }
                    });
                }
            }
            // The Start screen lists the photos; its "Open photos" button is enabled once one is
            // ready. On the Photos screen there is no such button and this does nothing.
            if open.is_some()
                && let Some(window) = app.get_webview_window("main")
            {
                let _ = window.eval(SMOKE_OPEN_PHOTOS);
            }
        }
        if perf_started {
            if !perf_done.load(Ordering::SeqCst) {
                eprintln!("smoke: perf script did not finish before the run ended");
            }
            // The page's own counters and the viewport's last window, while the app still runs.
            // The callbacks run on the main thread, which this thread leaves a moment for.
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.eval_with_callback(SMOKE_PERF_PAGE, |json| {
                    let page = unwrap_script_json(&json).unwrap_or_else(|| "null".to_owned());
                    eprintln!("smoke: perf page {page}");
                });
            }
            perf::log_viewport_window();
            std::thread::sleep(std::time::Duration::from_millis(1500));
        }
        let info = api.app_info();
        let prefs = api.preferences_get();
        let backend = api.backend();
        let windows = app.webview_windows().len();
        let photos = api.session_state().map_or(0, |s| s.photos.len());
        let ok = prefs.is_ok() && windows > 0 && (open.is_none() || photos > 0);
        eprintln!(
            "smoke: version {} backend {backend:?} windows {windows} photos {photos} preferences {} -> {}",
            info.version,
            if prefs.is_ok() { "ok" } else { "failed" },
            if ok { "ok" } else { "FAILED" }
        );
        app.exit(i32::from(!ok));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_running_export_that_nobody_agreed_to_stop_holds_the_quit_back() {
        assert!(quit_needs_question(true, false));
        assert!(!quit_needs_question(false, false), "nothing to lose");
        assert!(
            !quit_needs_question(true, true),
            "the person already said yes"
        );
        assert!(!quit_needs_question(false, true));
    }

    /// `/<launch>/<file>` of an address the store made.
    fn path_of(url: &str) -> String {
        let (_, path) = url.split_once("localhost/").expect("a localhost address");
        format!("/{path}")
    }

    #[test]
    fn the_frame_protocol_never_caches_frames_and_misses_but_caches_thumbnails_and_tiles_for_good()
    {
        let store = FrameStore::new();
        store.put(7, vec![1, 2, 3]);
        let frame_path = path_of(&store.frame_url(7));
        let ok = frame_response(&store, &frame_path);
        assert_eq!(ok.status(), 200);
        assert_eq!(ok.headers()["Content-Type"], "image/png");
        assert_eq!(ok.headers()["Cache-Control"], "no-store");
        assert_eq!(ok.body().as_ref(), &[1, 2, 3]);
        let gone = frame_response(&store, &path_of(&store.frame_url(8)));
        assert_eq!(gone.status(), 404);
        assert_eq!(gone.headers()["Cache-Control"], "no-store");

        let thumb = store.put_thumbnail(4, vec![4, 4]);
        let tile = store.put_look_tile(4, "builtin:neutral", vec![5]);
        for (url, body) in [(&thumb, vec![4u8, 4]), (&tile, vec![5u8])] {
            let response = frame_response(&store, &path_of(url));
            assert_eq!(response.status(), 200, "{url}");
            assert_eq!(response.headers()["Content-Type"], "image/png");
            assert_eq!(
                response.headers()["Cache-Control"],
                "private, max-age=31536000, immutable",
                "{url}"
            );
            assert_eq!(response.body().as_ref(), body.as_slice());
        }
        // The same address in an earlier launch is a miss, and a miss is never kept.
        let other = FrameStore::with_launch_id("an-earlier-launch".to_owned());
        let stale = frame_response(&other, &path_of(&thumb));
        assert_eq!(stale.status(), 404);
        assert_eq!(stale.headers()["Cache-Control"], "no-store");
        assert!(
            frame_response(&store, "/thumb-4-1.png").status() == 404,
            "no launch id"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_script_that_has_left_nothing_yet_is_not_a_result() {
        // `eval_with_callback` hands a string over as JSON again.
        assert_eq!(unwrap_script_json("\"null\""), None);
        assert_eq!(unwrap_script_json("null"), None);
        assert_eq!(unwrap_script_json("not json"), None);
        assert_eq!(
            unwrap_script_json(r#""{\"done\":true}""#).as_deref(),
            Some(r#"{"done":true}"#)
        );
    }

    #[test]
    fn a_request_is_answered_from_another_thread_even_when_the_store_is_busy() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};
        let store = Arc::new(FrameStore::new());
        let tile = store.put_look_tile(1, "builtin:neutral", vec![9, 9]);
        let path = path_of(&tile);

        // Another thread holds the tiles (a burst of `put_look_tile`, a 256 MB budget being
        // enforced). The thread the handler is called on, which is the main thread, must not wait.
        let held = store.hold_look_tiles();
        let (tx, rx) = mpsc::channel();
        let called = Instant::now();
        serve_frame(Arc::clone(&store), path, move |response| {
            let _ = tx.send(response);
        });
        assert!(
            called.elapsed() < Duration::from_millis(250),
            "the handler waited for the store: {:?}",
            called.elapsed()
        );
        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "the answer came while the store was held"
        );
        drop(held);
        let response = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the answer arrives once the store is free");
        assert_eq!(response.status(), 200);
        assert_eq!(response.body().as_ref(), &[9, 9]);
        assert_eq!(
            response.headers()["Cache-Control"],
            "private, max-age=31536000, immutable"
        );
    }
}
