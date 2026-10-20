//! The application: plugins, the managed state, the menu, the frame protocol and quitting.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::http::Response;
use tauri::menu::MenuEvent;
use tauri::{App, AppHandle, Manager, RunEvent, Runtime, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_specta::Event;
use ym_gpu::device::Gpu;

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

    let mut app_builder = tauri::Builder::default();
    // The page measures itself (`src/perf.ts`) when asked with `?perf` or `localStorage.ymPerf`; the Mac
    // viewport job can do neither, so `YM_PERF` reaches the page as a flag set before its scripts run.
    if perf::page_requested() {
        app_builder = app_builder.plugin(perf::page_plugin::<tauri::Wry>());
    }
    let app = app_builder
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
    let cli = CliLink::locate()
        .map_err(|e| tracing::warn!(error = %e, "no command-line tool next to the app"))
        .ok();
    // The sink is made once the graphics device is open: the native viewport draws with it.
    let make = {
        let (handle, frames, emitter) = (
            app.handle().clone(),
            Arc::clone(frames),
            Arc::clone(&emitter),
        );
        move |gpu: Option<&Arc<Gpu>>| make_sink(&handle, &frames, &emitter, gpu)
    };
    match AppApi::start(config, emitter, make, Arc::clone(frames), elevator(), cli) {
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
/// whenever the native layer cannot be made). `gpu` is the device the core renders with, if any:
/// the native layer draws with it (`crate::native`).
fn make_sink(
    app: &AppHandle,
    frames: &Arc<FrameStore>,
    emitter: &Arc<dyn Emitter>,
    gpu: Option<&Arc<Gpu>>,
) -> Arc<dyn FrameSink> {
    // `YELLOWMOON_VIEWPORT=fallback` asks for the frames-to-webview path even on a Mac, to compare
    // the two backends (docs/11).
    let forced_fallback = std::env::var("YELLOWMOON_VIEWPORT").is_ok_and(|v| v == "fallback");
    #[cfg(target_os = "macos")]
    if !forced_fallback {
        match crate::native::NativeSink::create(app, "main", gpu) {
            Ok(sink) => return Arc::new(sink),
            Err(e) => {
                tracing::warn!(error = %e, "no native viewport; sending frames to the webview");
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (app, gpu);
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
/// element under that point with its background. `elementsFromPoint` skips what takes no pointer
/// events and ignores masks, so the aurora (a fixed, opaque, `pointer-events: none` layer that a CSS
/// mask cuts the cell out of, `.ym-aurora` in `src/styles/glass-c2.css`) is read directly: its mask
/// and the rectangle the Photos screen reports in `--cell-x/y/w/h`. [`page_paints_over_cell`] reads
/// the result.
#[cfg(debug_assertions)]
const SMOKE_DESCRIBE_PAGE: &str = r##"(() => {
  const bg = (e) => (e ? getComputedStyle(e).backgroundColor : null);
  const cell = document.querySelector(".ym-photo-cell");
  const r = cell ? cell.getBoundingClientRect() : null;
  const name = (e) => e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + (e.classList.length ? "." + [...e.classList].join(".") : "");
  const under = r ? document.elementsFromPoint(r.left + r.width / 2, r.top + r.height / 2) : [];
  const stack = under.map((e) => name(e) + " " + bg(e));
  // What else a box can paint besides a colour: an image (a gradient) and a blur of what is behind it.
  const paints = under.map((e) => {
    const cs = getComputedStyle(e);
    return { element: name(e), image: cs.backgroundImage, backdrop: cs.backdropFilter || cs.webkitBackdropFilter || "none" };
  });
  const au = document.querySelector(".ym-aurora");
  const acs = au ? getComputedStyle(au) : null;
  const root = getComputedStyle(document.documentElement);
  return JSON.stringify({
    viewport: document.documentElement.hasAttribute("data-native-viewport") ? "native" : null,
    native: cell ? cell.dataset.native ?? null : null,
    cell: r ? [r.left, r.top, r.width, r.height] : null,
    html: bg(document.documentElement),
    body: bg(document.body),
    stack,
    paints,
    // The aurora's cut-out, either spelling of each property (older WebKit has the prefixed ones only).
    aurora: acs
      ? {
          composite: acs.maskComposite || acs.webkitMaskComposite || "",
          size: acs.maskSize || acs.webkitMaskSize || "",
          position: acs.maskPosition || acs.webkitMaskPosition || "",
        }
      : null,
    cell_vars: ["--cell-x", "--cell-y", "--cell-w", "--cell-h"].map((v) => root.getPropertyValue(v).trim()),
    // Why a box paints (0.1.4's viewport job: the root's attribute set and the rule loaded, the root still paper): the root's inline
    // style, whether the rule's selector matches, and whether a loaded sheet carries the rule.
    root_style: document.documentElement.getAttribute("style"),
    rule_matches: document.documentElement.matches(":root[data-native-viewport]"),
    reduce_motion: matchMedia("(prefers-reduced-motion: reduce)").matches,
    animations: document.getAnimations ? document.getAnimations().map((a) => a.constructor.name + " " + (a.transitionProperty || a.animationName || "") + " " + a.playState).slice(0, 12) : "none",
    sheets: [...document.styleSheets].map((s) => {
      let rule = false;
      try { rule = [...s.cssRules].some((r) => r.cssText.includes("data-native-viewport")); } catch (e) { rule = "unreadable"; }
      return (s.href || "inline").split("/").pop() + (rule === true ? " (has the rule)" : rule === "unreadable" ? " (unreadable)" : "");
    }),
  });
})()"##;

/// What stands between the photo and the person in a page described by [`SMOKE_DESCRIBE_PAGE`]; an empty
/// list when nothing does (the native viewport is a layer behind the webview, so anything the page paints over the cell hides the photo,
/// the 0.1.0 failure). Four things are held to it:
///
/// 1. every box under the middle of the cell has no background colour;
/// 2. none has a background image or blurs what is behind it;
/// 3. the aurora (opaque, but it takes no pointer events, so the boxes above never list it) is masked with the two-layer exclusion
///    whose second layer is the cell's rectangle;
/// 4. the page has told the aurora where the cell is (`--cell-x/y/w/h`, the rectangle the mask is sized from).
///
/// The cell's rectangle is compared to the mask with a pixel and a half of room: the page writes the shell's rounded rectangle.
#[cfg(debug_assertions)]
fn page_paints_over_cell(page: &serde_json::Value) -> Vec<String> {
    const CLEAR: &str = "rgba(0, 0, 0, 0)";
    const ROOM_PX: f64 = 1.5;
    let mut problems = Vec::new();

    match page["stack"].as_array() {
        Some(stack) if !stack.is_empty() => {
            for entry in stack {
                match entry.as_str() {
                    Some(e) if e.ends_with(CLEAR) => {}
                    Some(e) => problems.push(format!("{e} paints a background colour")),
                    None => problems.push("an entry of the stack is not text".to_owned()),
                }
            }
        }
        _ => problems.push("no element under the middle of the photo cell".to_owned()),
    }

    match page["paints"].as_array() {
        Some(paints) if !paints.is_empty() => {
            for paint in paints {
                let element = paint["element"].as_str().unwrap_or("an element");
                if paint["image"].as_str() != Some("none") {
                    problems.push(format!("{element} paints a background image"));
                }
                if !matches!(paint["backdrop"].as_str(), Some("none" | "")) {
                    problems.push(format!("{element} blurs what is behind it"));
                }
            }
        }
        _ => problems.push("the page did not say what its boxes paint".to_owned()),
    }

    // The cell's rectangle as the page measured it: [left, top, width, height].
    let cell: Option<Vec<f64>> = page["cell"]
        .as_array()
        .map(|v| v.iter().filter_map(serde_json::Value::as_f64).collect())
        .filter(|v: &Vec<f64>| v.len() == 4);
    let vars_set = page["cell_vars"].as_array().is_some_and(|vars| {
        vars.len() == 4
            && vars
                .iter()
                .all(|v| v.as_str().is_some_and(|v| !v.is_empty()))
    });
    if !vars_set {
        problems.push(
            "--cell-x/y/w/h are not set: the Photos screen has not told the aurora where the cell is"
                .to_owned(),
        );
    }

    let aurora = &page["aurora"];
    if aurora.is_null() {
        problems.push("the Photos screen has no .ym-aurora".to_owned());
    } else {
        // The computed value is a list with one entry for each layer ("exclude, exclude"); the first layer is the one that is
        // composited with the layer below it, which is what cuts the cell out.
        let composite = aurora["composite"].as_str().unwrap_or("");
        if !matches!(
            composite.split(',').next().map(str::trim),
            Some("exclude" | "xor")
        ) {
            problems.push(format!(
                "the aurora's mask does not exclude its second layer (mask-composite is {composite:?})"
            ));
        }
        let second_layer = |property: &str| -> Option<Vec<f64>> {
            let layer = aurora[property].as_str()?.split(',').nth(1)?;
            layer
                .split_whitespace()
                .map(|n| n.strip_suffix("px")?.parse::<f64>().ok())
                .collect()
        };
        match (second_layer("size"), second_layer("position"), &cell) {
            (Some(size), Some(position), Some(cell)) if size.len() == 2 && position.len() == 2 => {
                let wanted = [
                    ("left", position[0], cell[0]),
                    ("top", position[1], cell[1]),
                    ("width", size[0], cell[2]),
                    ("height", size[1], cell[3]),
                ];
                for (what, got, want) in wanted {
                    if (got - want).abs() > ROOM_PX {
                        problems.push(format!(
                            "the aurora's cut-out has {what} {got} px where the cell has {want} px"
                        ));
                    }
                }
                if size[0] <= 0.0 || size[1] <= 0.0 {
                    problems.push("the aurora's cut-out is empty".to_owned());
                }
            }
            _ => problems.push(
                "the aurora's mask has no second layer sized and placed in pixels".to_owned(),
            ),
        }
    }
    problems
}

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
    // glasscn's slider is Radix's: the knob has role="slider" and is named by its row's label through aria-labelledby.
    const nameOf = (s) => (s.getAttribute("aria-labelledby") || "").split(" ").map((id) => (document.getElementById(id) || {}).textContent || "").join(" ").trim();
    const slider = [...document.querySelectorAll('[role="slider"]')].find((s) => nameOf(s) === "Temperature") || document.querySelector('[role="slider"]:not([data-disabled])');
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
                    let problems = page.as_ref().map_or_else(
                        || vec!["the page did not answer".to_owned()],
                        page_paints_over_cell,
                    );
                    for problem in &problems {
                        eprintln!("smoke: the page paints over the cell: {problem}");
                    }
                    let clear = problems.is_empty();
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

    /// A page as [`SMOKE_DESCRIBE_PAGE`] describes the Photos screen over the native viewport when all is well: the cell at
    /// (200, 60), 1000 by 600 px, every box under its middle clear, the aurora masked with the cell cut out.
    #[cfg(debug_assertions)]
    fn clear_page() -> serde_json::Value {
        serde_json::json!({
            "cell": [200.0, 60.0, 1000.0, 600.0],
            "stack": [
                "div.ym-photo-cell__overlay rgba(0, 0, 0, 0)",
                "div.ym-photo-cell rgba(0, 0, 0, 0)",
                "html rgba(0, 0, 0, 0)",
            ],
            "paints": [
                { "element": "div.ym-photo-cell__overlay", "image": "none", "backdrop": "none" },
                { "element": "div.ym-photo-cell", "image": "none", "backdrop": "none" },
                { "element": "html", "image": "none", "backdrop": "none" },
            ],
            "aurora": {
                "composite": "exclude, exclude",
                "size": "100% 100%, 1000px 600px",
                "position": "0% 0%, 200px 60px",
            },
            "cell_vars": ["200px", "60px", "1000px", "600px"],
        })
    }

    /// The script's own answer in Chromium (the `viewport` e2e of the frontend runs it there), cut down to what the check reads: the
    /// numbers come as the browser writes them, one entry for each layer of the mask, three decimals.
    #[cfg(debug_assertions)]
    #[test]
    fn the_description_a_browser_gives_is_read_as_it_is_written() {
        let page = serde_json::json!({
            "cell": [0, 75, 1040, 693.328125],
            "stack": [
                "div.ym-photo-cell__view rgba(0, 0, 0, 0)",
                "main.ym-photo-cell rgba(0, 0, 0, 0)",
                "div.ym-screen rgba(0, 0, 0, 0)",
                "div#root rgba(0, 0, 0, 0)",
                "body rgba(0, 0, 0, 0)",
                "html.dark rgba(0, 0, 0, 0)",
            ],
            "paints": [
                { "element": "div.ym-photo-cell__view", "image": "none", "backdrop": "none" },
                { "element": "main.ym-photo-cell", "image": "none", "backdrop": "none" },
                { "element": "div.ym-screen", "image": "none", "backdrop": "none" },
                { "element": "div#root", "image": "none", "backdrop": "none" },
                { "element": "body", "image": "none", "backdrop": "none" },
                { "element": "html.dark", "image": "none", "backdrop": "none" },
            ],
            "aurora": {
                "composite": "exclude, exclude",
                "size": "100% 100%, 1040px 693.328px",
                "position": "0px 0px, 0px 75px",
            },
            "cell_vars": ["0px", "75px", "1040px", "693.328125px"],
        });
        assert_eq!(page_paints_over_cell(&page), Vec::<String>::new());
    }

    /// What the check was before the aurora: the boxes under the cell's middle, by colour alone.
    #[cfg(debug_assertions)]
    fn the_stack_alone_says_clear(page: &serde_json::Value) -> bool {
        page["stack"].as_array().is_some_and(|stack| {
            !stack.is_empty()
                && stack
                    .iter()
                    .all(|e| e.as_str().is_some_and(|e| e.ends_with("rgba(0, 0, 0, 0)")))
        })
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_page_that_cuts_the_cell_out_of_the_aurora_and_paints_nothing_over_it_lets_the_viewport_through()
     {
        assert_eq!(page_paints_over_cell(&clear_page()), Vec::<String>::new());
        // A fraction of a pixel is the page rounding what the shell measured.
        let mut page = clear_page();
        page["aurora"]["size"] = "100% 100%, 1000.4px 599.6px".into();
        page["aurora"]["position"] = "0px 0px, 199.5px 60.5px".into();
        assert_eq!(page_paints_over_cell(&page), Vec::<String>::new());
        // The prefixed spelling of an older WebKit.
        let mut page = clear_page();
        page["aurora"]["composite"] = "xor, xor".into();
        assert_eq!(page_paints_over_cell(&page), Vec::<String>::new());
        page["aurora"]["composite"] = "exclude".into();
        assert_eq!(page_paints_over_cell(&page), Vec::<String>::new());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn an_opaque_aurora_that_the_mask_does_not_cut_is_found_though_every_box_above_it_is_clear() {
        // 0.1.0's symptom with the dark glass: the aurora takes no pointer events, so `elementsFromPoint` never lists it, and the
        // boxes it does list are clear whether or not the mask works.
        type Spoil = fn(&mut serde_json::Value);
        let broken: [(&str, Spoil); 8] = [
            ("a mask that adds its layers", |p| {
                p["aurora"]["composite"] = "add, add".into();
            }),
            ("a mask that keeps the first layer as it is", |p| {
                p["aurora"]["composite"] = "add, exclude".into();
            }),
            ("a mask no property of which the browser kept", |p| {
                p["aurora"]["composite"] = "".into();
            }),
            ("a cut-out of no size (the variables were never set)", |p| {
                p["aurora"]["size"] = "100% 100%, 0px 0px".into();
                p["cell_vars"] = serde_json::json!(["", "", "", ""]);
            }),
            ("a cut-out that is not where the cell is", |p| {
                p["aurora"]["position"] = "0% 0%, 0px 0px".into();
            }),
            ("a cut-out smaller than the cell", |p| {
                p["aurora"]["size"] = "100% 100%, 640px 600px".into();
            }),
            ("a mask of one layer", |p| {
                p["aurora"]["size"] = "100% 100%".into();
            }),
            ("no aurora where the check expects one", |p| {
                p["aurora"] = serde_json::Value::Null;
            }),
        ];
        for (what, spoil) in broken {
            let mut page = clear_page();
            spoil(&mut page);
            assert!(
                the_stack_alone_says_clear(&page),
                "{what}: the old check would have passed"
            );
            assert!(
                !page_paints_over_cell(&page).is_empty(),
                "{what}: the page was let through"
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_box_over_the_cell_that_paints_a_colour_an_image_or_a_blur_is_found() {
        let mut page = clear_page();
        page["stack"][1] = "div.ym-photo-cell rgb(15, 15, 16)".into();
        let problems = page_paints_over_cell(&page);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("div.ym-photo-cell"), "{problems:?}");

        let mut page = clear_page();
        page["paints"][0]["image"] = "linear-gradient(rgb(0, 0, 0), rgb(9, 9, 9))".into();
        assert!(
            the_stack_alone_says_clear(&page),
            "a gradient has no background colour"
        );
        let problems = page_paints_over_cell(&page);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("background image"), "{problems:?}");

        let mut page = clear_page();
        page["paints"][2]["backdrop"] = "blur(24px) saturate(1.4)".into();
        let problems = page_paints_over_cell(&page);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("blurs"), "{problems:?}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_page_that_says_nothing_about_the_cell_is_not_let_through() {
        for page in [
            serde_json::json!({}),
            serde_json::json!({ "stack": [], "paints": [], "aurora": null, "cell": null }),
        ] {
            assert!(!page_paints_over_cell(&page).is_empty(), "{page}");
        }
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
