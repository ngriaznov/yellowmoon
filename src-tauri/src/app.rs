//! The application: plugins, the managed state, the menu, the frame protocol and quitting.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::http::{Request, Response};
use tauri::menu::MenuEvent;
use tauri::{App, AppHandle, Manager, RunEvent, Runtime, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_specta::Event;

use crate::api::{AppApi, AppConfig};
use crate::cli_link::{CliLink, Elevate};
use crate::commands::{self, AppState};
use crate::events::{Emitter, TauriEmitter};
use crate::menu::{self, Do};
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
        .register_uri_scheme_protocol(FRAME_SCHEME, move |_ctx, request| {
            frame_response(&protocol_frames, &request)
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

/// `ym-frame://localhost/frame-<n>.png` (a frame of the fallback viewport) or
/// `ym-frame://localhost/thumb-<photo>-<version>.png` (a thumbnail of the filmstrip).
fn frame_response(store: &FrameStore, request: &Request<Vec<u8>>) -> Response<Cow<'static, [u8]>> {
    let (status, mime, body) = store.respond(request.uri().path());
    Response::builder()
        .status(status)
        .header("Content-Type", mime)
        .header("Cache-Control", "no-store")
        .header("Access-Control-Allow-Origin", "*")
        .body(body)
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
/// session as after a reload) and the check also wants photos in the session; `YM_SMOKE_HOLD=<s>`
/// waits that many seconds instead of 4. The viewport job of `.github/workflows/build.yml` uses
/// both to capture the screen of a Mac while it shows a photo.
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
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(hold));
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

    #[test]
    fn the_frame_protocol_answers_with_the_png_and_never_caches_it() {
        let store = FrameStore::new();
        store.put(7, vec![1, 2, 3]);
        let request = |path: &str| {
            Request::builder()
                .uri(format!("{FRAME_SCHEME}://localhost{path}"))
                .body(Vec::new())
                .expect("request")
        };
        let ok = frame_response(&store, &request("/frame-7.png"));
        assert_eq!(ok.status(), 200);
        assert_eq!(ok.headers()["Content-Type"], "image/png");
        assert_eq!(ok.headers()["Cache-Control"], "no-store");
        assert_eq!(ok.body().as_ref(), &[1, 2, 3]);
        let gone = frame_response(&store, &request("/frame-8.png"));
        assert_eq!(gone.status(), 404);
    }
}
