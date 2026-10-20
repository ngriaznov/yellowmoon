//! The Tauri commands of `docs/01-architecture.md` §IPC between webview and core, and the typed
//! registry `tauri-specta` makes `src/api/bindings.ts` from.
//!
//! Every command is a thin wrapper: it takes the [`AppApi`] out of the managed state and runs one
//! of its methods on a blocking thread, so the webview's IPC never waits on a slow render and a
//! dialog never blocks the main thread. The argument names are the ones `src/api.ts` sends
//! (camelCase on the wire, `photo_id` here). Errors are [`AppError`]: a scope and a plain sentence.

use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, State};
use tauri_plugin_dialog::DialogExt;
use tauri_specta::{Builder, collect_commands, collect_events};

use crate::api::AppApi;
use crate::system;
use crate::wire::{
    self, AppError, AppInfo, AppleAutoStatus, ApplyScope, ChromaSample, ChromaScatter,
    CliLinkResult, CliLinkStatus, DisplayLut, ErrorScope, ExportEstimate, ExportJob, ExportJobId,
    HistogramData, Look, LookId, LookInput, LooksImportResult, MenuState, Num, OpenKind,
    PasteReport, PhotoId, PixelSample, Preferences, PreferencesPatch, ResolvedSettings,
    SessionChange, SessionState, SettingsPatch, SourceRecognition, ThumbnailReady, ViewportInfo,
    ViewportInput, ViewportMode, ViewportRect, WatchStatus,
};

/// The managed state: the command handlers, or why the application could not start.
pub struct AppState {
    api: Result<Arc<AppApi>, AppError>,
}

impl AppState {
    /// A running application.
    #[must_use]
    pub fn ready(api: Arc<AppApi>) -> Self {
        Self { api: Ok(api) }
    }

    /// An application that could not start (no data folder, …): every command answers with `error`.
    #[must_use]
    pub fn failed(error: AppError) -> Self {
        Self { api: Err(error) }
    }

    /// The command handlers.
    ///
    /// # Errors
    ///
    /// The error the application could not start with.
    pub fn api(&self) -> Result<Arc<AppApi>, AppError> {
        self.api.clone()
    }
}

/// Runs a handler on a blocking thread.
async fn run<T, F>(state: &State<'_, AppState>, f: F) -> Result<T, AppError>
where
    T: Send + 'static,
    F: FnOnce(Arc<AppApi>) -> Result<T, AppError> + Send + 'static,
{
    let api = state.api()?;
    tauri::async_runtime::spawn_blocking(move || f(api))
        .await
        .map_err(AppError::internal)?
}

// --- Session ----------------------------------------------------------------------------------

/// `open_paths(paths)`: starts a session or adds photos.
#[tauri::command]
#[specta::specta]
pub async fn open_paths(
    state: State<'_, AppState>,
    paths: Vec<String>,
) -> Result<SessionState, AppError> {
    run(&state, move |api| api.open_paths(&paths)).await
}

/// `open_dialog(kind)`: asks for files or a folder with the system's dialog, then opens them.
/// `null` when the dialog is cancelled.
#[tauri::command]
#[specta::specta]
pub async fn open_dialog(
    app: AppHandle,
    state: State<'_, AppState>,
    kind: OpenKind,
) -> Result<Option<SessionState>, AppError> {
    run(&state, move |api| {
        let start = api
            .preferences_get()?
            .last_used_folders
            .open
            .map(PathBuf::from);
        let mut dialog = app.dialog().file().set_title(match kind {
            OpenKind::Files => "Open ARW files",
            OpenKind::Folder => "Open a folder of ARW files",
        });
        if let Some(dir) = start.filter(|d| d.is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        let paths: Vec<String> = match kind {
            OpenKind::Files => dialog
                .add_filter("Sony ARW", &["arw", "ARW"])
                .blocking_pick_files()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|p| p.into_path().ok())
                .map(|p| p.to_string_lossy().into_owned())
                .collect(),
            OpenKind::Folder => dialog
                .blocking_pick_folder()
                .and_then(|p| p.into_path().ok())
                .map(|p| vec![p.to_string_lossy().into_owned()])
                .unwrap_or_default(),
        };
        if paths.is_empty() {
            return Ok(None);
        }
        api.open_paths(&paths).map(Some)
    })
    .await
}

/// `session_state()`.
#[tauri::command]
#[specta::specta]
pub async fn session_state(state: State<'_, AppState>) -> Result<SessionState, AppError> {
    run(&state, |api| api.session_state()).await
}

/// `close_session()`: forgets the session; running exports keep going.
#[tauri::command]
#[specta::specta]
pub async fn close_session(state: State<'_, AppState>) -> Result<(), AppError> {
    run(&state, |api| {
        api.close_session();
        Ok(())
    })
    .await
}

/// `set_current(photo_id)`.
#[tauri::command]
#[specta::specta]
pub async fn set_current(state: State<'_, AppState>, photo_id: PhotoId) -> Result<(), AppError> {
    run(&state, move |api| api.set_current(photo_id)).await
}

/// `remove_photos(photo_ids)`: takes photos out of the filmstrip. The files on disk are not touched; the photos' edits go with
/// them, in memory only. The last removal can be undone with `restore_photos`.
#[tauri::command]
#[specta::specta]
pub async fn remove_photos(
    state: State<'_, AppState>,
    photo_ids: Vec<PhotoId>,
) -> Result<SessionChange, AppError> {
    run(&state, move |api| api.remove_photos(&photo_ids)).await
}

/// `restore_photos()`: puts back the photos of the last `remove_photos`, where they were, with their settings.
#[tauri::command]
#[specta::specta]
pub async fn restore_photos(state: State<'_, AppState>) -> Result<SessionChange, AppError> {
    run(&state, |api| api.restore_photos()).await
}

/// `set_picked(photo_id, picked)`.
#[tauri::command]
#[specta::specta]
pub async fn set_picked(
    state: State<'_, AppState>,
    photo_id: PhotoId,
    picked: bool,
) -> Result<(), AppError> {
    run(&state, move |api| api.set_picked(photo_id, picked)).await
}

// --- Settings ---------------------------------------------------------------------------------

/// `get_settings(photo_id)`: the resolved settings with the Auto flags.
#[tauri::command]
#[specta::specta]
pub async fn get_settings(
    state: State<'_, AppState>,
    photo_id: PhotoId,
) -> Result<ResolvedSettings, AppError> {
    run(&state, move |api| api.get_settings(photo_id)).await
}

/// `update_settings(photo_id, patch)`: any control change; returns the settings resolved.
#[tauri::command]
#[specta::specta]
pub async fn update_settings(
    state: State<'_, AppState>,
    photo_id: PhotoId,
    patch: SettingsPatch,
) -> Result<ResolvedSettings, AppError> {
    run(&state, move |api| api.update_settings(photo_id, &patch)).await
}

/// `apply_look(scope, look_id, intensity)`: to This photo, the Picked photos or All.
#[tauri::command]
#[specta::specta]
pub async fn apply_look(
    state: State<'_, AppState>,
    scope: ApplyScope,
    look_id: LookId,
    intensity: Num,
) -> Result<(), AppError> {
    run(&state, move |api| {
        api.apply_look(scope, &look_id, intensity.0)
    })
    .await
}

/// `rerun_auto(photo_id)`: recomputes the automatic estimates.
#[tauri::command]
#[specta::specta]
pub async fn rerun_auto(
    state: State<'_, AppState>,
    photo_id: PhotoId,
) -> Result<ResolvedSettings, AppError> {
    run(&state, move |api| api.rerun_auto(photo_id)).await
}

/// `copy_settings(photo_id)`: the photo's settings as a patch to paste onto other photos (not the
/// crop, the orientation or the Pick white point).
#[tauri::command]
#[specta::specta]
pub async fn copy_settings(
    state: State<'_, AppState>,
    photo_id: PhotoId,
) -> Result<SettingsPatch, AppError> {
    run(&state, move |api| api.copy_settings(photo_id)).await
}

/// `paste_settings(photo_ids, patch)`: applies a copied patch to those photos in one call; the
/// answer says which took it (with each one's change as an undo step) and which were skipped.
#[tauri::command]
#[specta::specta]
pub async fn paste_settings(
    state: State<'_, AppState>,
    photo_ids: Vec<PhotoId>,
    patch: SettingsPatch,
) -> Result<PasteReport, AppError> {
    run(&state, move |api| api.paste_settings(&photo_ids, &patch)).await
}

// Not `async` on purpose, for the reason given at `viewport_set_rect` below: the webview sends this
// on every flip of the three flags, and two calls on worker threads of their own could overtake
// each other, or swap items one by one on the main thread, and leave the menu in a mixed state. A
// command that is not `async` runs on the main thread in the order the webview sent them.
/// `set_menu_state(state)`: switches the Edit menu's settings items on or off.
#[tauri::command]
#[specta::specta]
pub fn set_menu_state(app: AppHandle, state: MenuState) -> Result<(), AppError> {
    crate::menu::apply_state(&app, &state).map_err(AppError::internal)
}

/// `histogram(photo_id)`: the RGB histogram of the photo as it is rendered now.
#[tauri::command]
#[specta::specta]
pub async fn histogram(
    state: State<'_, AppState>,
    photo_id: PhotoId,
) -> Result<HistogramData, AppError> {
    run(&state, move |api| api.histogram(photo_id)).await
}

/// `chroma_scatter(photo_id)`: the photo's colours at the colour warp's input, for its plane.
#[tauri::command]
#[specta::specta]
pub async fn chroma_scatter(
    state: State<'_, AppState>,
    photo_id: PhotoId,
) -> Result<ChromaScatter, AppError> {
    run(&state, move |api| api.chroma_scatter(photo_id)).await
}

// --- Viewport ---------------------------------------------------------------------------------

/// `viewport_info()` (addition): which backend draws the photo, `native` or `fallback`.
#[tauri::command]
#[specta::specta]
pub async fn viewport_info(state: State<'_, AppState>) -> Result<ViewportInfo, AppError> {
    run(&state, |api| Ok(api.viewport_info())).await
}

// Not `async` on purpose: Tauri runs a command that is not `async` on the main thread, one after
// the other in the order the webview sent them, while every `async` command is handed to a worker
// thread of its own. A window being resized sends a rectangle per frame, and two of them that
// overtook each other would leave the photo at the size of the second to last. The handler only
// records the rectangle and wakes the viewport worker, so it never makes the main thread wait.
/// `viewport_set_rect(rect)`: where the photo cell is.
#[tauri::command]
#[specta::specta]
pub fn viewport_set_rect(state: State<'_, AppState>, rect: ViewportRect) -> Result<(), AppError> {
    // Timed (with `YM_PERF`, and in development builds): this runs on the main thread.
    let _timer = crate::perf::timer(crate::perf::Timed::SetRect);
    state.api()?.viewport_set_rect(&rect)
}

/// `viewport_input(event)`: pointer input over the photo.
#[tauri::command]
#[specta::specta]
pub async fn viewport_input(
    state: State<'_, AppState>,
    event: ViewportInput,
) -> Result<(), AppError> {
    run(&state, move |api| api.viewport_input(&event)).await
}

/// `viewport_set_mode(mode)`: Fit, or a fixed zoom.
#[tauri::command]
#[specta::specta]
pub async fn viewport_set_mode(
    state: State<'_, AppState>,
    mode: ViewportMode,
) -> Result<(), AppError> {
    run(&state, move |api| api.viewport_set_mode(&mode)).await
}

/// `set_before(before)`: the photo before the look and the display transform.
#[tauri::command]
#[specta::specta]
pub async fn set_before(state: State<'_, AppState>, before: bool) -> Result<(), AppError> {
    run(&state, move |api| {
        api.set_before(before);
        Ok(())
    })
    .await
}

/// `sample_pixel(x, y)`: display values under a point of the photo cell.
#[tauri::command]
#[specta::specta]
pub async fn sample_pixel(
    state: State<'_, AppState>,
    x: Num,
    y: Num,
) -> Result<PixelSample, AppError> {
    run(&state, move |api| api.sample_pixel(x.0, y.0)).await
}

/// `sample_chroma(x, y)`: the chroma at the colour warp's input under a point of the photo cell
/// (Pick colour); `null` on the paper around the photo.
#[tauri::command]
#[specta::specta]
pub async fn sample_chroma(
    state: State<'_, AppState>,
    x: Num,
    y: Num,
) -> Result<Option<ChromaSample>, AppError> {
    run(&state, move |api| api.sample_chroma(x.0, y.0)).await
}

// --- Looks ------------------------------------------------------------------------------------

/// `looks_list()`.
#[tauri::command]
#[specta::specta]
pub async fn looks_list(state: State<'_, AppState>) -> Result<Vec<Look>, AppError> {
    run(&state, |api| api.looks_list()).await
}

/// `look_thumbnail(photo_id, look_id)`: the photo through one look. The answer is the address of
/// a PNG of at most 240 px on the `ym-frame` scheme
/// (`ym-frame://localhost/<launch>/look-<photo>-<n>.png`, `http://ym-frame.localhost/…` on
/// Windows), like `thumbnail_ready`'s.
#[tauri::command]
#[specta::specta]
pub async fn look_thumbnail(
    state: State<'_, AppState>,
    photo_id: PhotoId,
    look_id: LookId,
) -> Result<String, AppError> {
    run(&state, move |api| api.look_thumbnail(photo_id, &look_id)).await
}

/// `looks_import(path, input)`: a zip, a folder or a `.cube` file.
#[tauri::command]
#[specta::specta]
pub async fn looks_import(
    state: State<'_, AppState>,
    path: String,
    input: Option<LookInput>,
) -> Result<LooksImportResult, AppError> {
    run(&state, move |api| api.looks_import(&path, input)).await
}

/// `looks_recognise(path)`: what a zip, a folder or a `.cube` file is. Never fails.
#[tauri::command]
#[specta::specta]
pub async fn looks_recognise(
    state: State<'_, AppState>,
    path: String,
) -> Result<SourceRecognition, AppError> {
    run(&state, move |api| Ok(api.looks_recognise(&path))).await
}

/// `looks_display_luts()`.
#[tauri::command]
#[specta::specta]
pub async fn looks_display_luts(state: State<'_, AppState>) -> Result<Vec<DisplayLut>, AppError> {
    run(&state, |api| api.looks_display_luts()).await
}

/// `looks_watch_downloads(enabled)`: starts or stops the Downloads watcher.
#[tauri::command]
#[specta::specta]
pub async fn looks_watch_downloads(
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<WatchStatus, AppError> {
    run(&state, move |api| Ok(api.looks_watch_downloads(enabled))).await
}

/// `looks_set_default(id)`.
#[tauri::command]
#[specta::specta]
pub async fn looks_set_default(state: State<'_, AppState>, id: LookId) -> Result<(), AppError> {
    run(&state, move |api| api.looks_set_default(&id)).await
}

/// `looks_remove(id)`.
#[tauri::command]
#[specta::specta]
pub async fn looks_remove(state: State<'_, AppState>, id: LookId) -> Result<(), AppError> {
    run(&state, move |api| api.looks_remove(&id)).await
}

// --- Export -----------------------------------------------------------------------------------

/// `export_estimate(job)`: bytes and seconds.
#[tauri::command]
#[specta::specta]
pub async fn export_estimate(
    state: State<'_, AppState>,
    job: ExportJob,
) -> Result<ExportEstimate, AppError> {
    run(&state, move |api| api.export_estimate(&job)).await
}

/// `export_start(job)`: queues a job; progress arrives as `export_progress`.
#[tauri::command]
#[specta::specta]
pub async fn export_start(
    state: State<'_, AppState>,
    job: ExportJob,
) -> Result<ExportJobId, AppError> {
    run(&state, move |api| api.export_start(&job)).await
}

/// `export_cancel(job_id)`: stops after the current tile and deletes the partial file.
#[tauri::command]
#[specta::specta]
pub async fn export_cancel(
    state: State<'_, AppState>,
    job_id: ExportJobId,
) -> Result<(), AppError> {
    run(&state, move |api| {
        api.export_cancel(job_id);
        Ok(())
    })
    .await
}

/// `cli_preview(job)`: the `ym render …` command for a job.
#[tauri::command]
#[specta::specta]
pub async fn cli_preview(state: State<'_, AppState>, job: ExportJob) -> Result<String, AppError> {
    run(&state, move |api| api.cli_preview(&job)).await
}

/// `export_busy()` (addition): an export is queued or running. "Update ready" waits while it is.
#[tauri::command]
#[specta::specta]
pub async fn export_busy(state: State<'_, AppState>) -> Result<bool, AppError> {
    run(&state, |api| Ok(api.export_busy())).await
}

/// `restart_app()` (addition): restarts into the update that was installed ("Update ready"). Refused
/// with "An export is still running. Restart after it finishes." while an export is queued or
/// running; the webview's own check is a courtesy, this is the guard (`AppApi::begin_restart`).
/// Does not return when it works: the process is replaced.
#[tauri::command]
#[specta::specta]
pub async fn restart_app(app: AppHandle, state: State<'_, AppState>) -> Result<(), AppError> {
    run(&state, |api| api.begin_restart()).await?;
    app.restart()
}

/// `thumbnails()` (addition): the address of the thumbnail of every photo that has one, so that a
/// webview that reloaded can show the filmstrip again.
#[tauri::command]
#[specta::specta]
pub async fn thumbnails(state: State<'_, AppState>) -> Result<Vec<ThumbnailReady>, AppError> {
    run(&state, |api| Ok(api.thumbnails_now())).await
}

// --- The system -------------------------------------------------------------------------------

/// `choose_folder(start)`: the system's folder chooser. `null` when cancelled.
#[tauri::command]
#[specta::specta]
pub async fn choose_folder(
    app: AppHandle,
    state: State<'_, AppState>,
    start: Option<String>,
) -> Result<Option<String>, AppError> {
    run(&state, move |_api| {
        let mut dialog = app.dialog().file().set_title("Choose a folder");
        if let Some(dir) = start.map(PathBuf::from).filter(|d| d.is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        Ok(dialog
            .blocking_pick_folder()
            .and_then(|p| p.into_path().ok())
            .map(|p| p.to_string_lossy().into_owned()))
    })
    .await
}

/// `choose_look_files()`: the system's file chooser for zips, `.cube` files and ARRI Look Files
/// (`.alf4`, `.alf4c`, `.aml`). `null` when cancelled.
#[tauri::command]
#[specta::specta]
pub async fn choose_look_files(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<Vec<String>>, AppError> {
    run(&state, move |api| {
        let start = api
            .preferences_get()?
            .last_used_folders
            .looks_import
            .map(PathBuf::from)
            .filter(|d| d.is_dir());
        let mut dialog = app
            .dialog()
            .file()
            .set_title("Choose look files")
            .add_filter(
                "ARRI packages, look files and LUTs",
                &["zip", "cube", "alf4", "alf4c", "aml"],
            );
        if let Some(dir) = start {
            dialog = dialog.set_directory(dir);
        }
        let paths: Vec<String> = dialog
            .blocking_pick_files()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|p| p.into_path().ok())
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        Ok((!paths.is_empty()).then_some(paths))
    })
    .await
}

/// `open_url(url)`: opens a page of arri.com or github.com in the system browser.
#[tauri::command]
#[specta::specta]
pub async fn open_url(url: String) -> Result<(), AppError> {
    tauri::async_runtime::spawn_blocking(move || system::open_url(&url))
        .await
        .map_err(AppError::internal)?
}

/// `reveal_in_finder(path)`: shows a file or folder in Finder.
#[tauri::command]
#[specta::specta]
pub async fn reveal_in_finder(path: String) -> Result<(), AppError> {
    tauri::async_runtime::spawn_blocking(move || system::reveal_in_finder(&wire::path_of(&path)))
        .await
        .map_err(AppError::internal)?
}

/// `trash_path(path)`: moves a downloaded zip to the Trash, recoverable.
#[tauri::command]
#[specta::specta]
pub async fn trash_path(path: String) -> Result<(), AppError> {
    tauri::async_runtime::spawn_blocking(move || system::trash_path(&wire::path_of(&path)))
        .await
        .map_err(AppError::internal)?
}

// --- Preferences, About and the command-line tool ---------------------------------------------

/// `preferences_get()`.
#[tauri::command]
#[specta::specta]
pub async fn preferences_get(state: State<'_, AppState>) -> Result<Preferences, AppError> {
    run(&state, |api| api.preferences_get()).await
}

/// `preferences_set(patch)`: applies a patch and returns the new preferences.
#[tauri::command]
#[specta::specta]
pub async fn preferences_set(
    state: State<'_, AppState>,
    patch: PreferencesPatch,
) -> Result<Preferences, AppError> {
    run(&state, move |api| api.preferences_set(&patch)).await
}

/// `apple_auto_status()`: whether Apple Auto is available on this computer (read again now) and
/// whether it acts; Preferences asks when it opens (`docs/19` sections 1 and 2).
#[tauri::command]
#[specta::specta]
pub async fn apple_auto_status(state: State<'_, AppState>) -> Result<AppleAutoStatus, AppError> {
    run(&state, |api| api.apple_auto_status()).await
}

/// `install_cli_link()`: the "Install Command-Line Tool…" menu item.
#[tauri::command]
#[specta::specta]
pub async fn install_cli_link(state: State<'_, AppState>) -> Result<CliLinkResult, AppError> {
    run(&state, |api| api.install_cli_link()).await
}

/// `cli_link_status()`.
#[tauri::command]
#[specta::specta]
pub async fn cli_link_status(state: State<'_, AppState>) -> Result<CliLinkStatus, AppError> {
    run(&state, |api| Ok(api.cli_link_status())).await
}

/// `app_info()`: version numbers for About.
#[tauri::command]
#[specta::specta]
pub async fn app_info(state: State<'_, AppState>) -> Result<AppInfo, AppError> {
    run(&state, |api| Ok(api.app_info())).await
}

// --- The registry -----------------------------------------------------------------------------

/// Builds the typed command and event registry shared by the app and the TypeScript bindings.
///
/// Add every new command to [`collect_commands!`] and every new event to [`collect_events!`] so its
/// TypeScript signature is generated; `bindings::tests::the_bindings_are_up_to_date` fails until
/// `src/api/bindings.ts` is regenerated.
#[must_use]
pub fn builder() -> Builder<tauri::Wry> {
    Builder::<tauri::Wry>::new()
        .error_handling(tauri_specta::ErrorHandlingMode::Throw)
        .disable_serde_phases()
        .commands(collect_commands![
            open_paths,
            open_dialog,
            session_state,
            close_session,
            set_current,
            set_picked,
            remove_photos,
            restore_photos,
            get_settings,
            update_settings,
            apply_look,
            rerun_auto,
            copy_settings,
            paste_settings,
            set_menu_state,
            histogram,
            chroma_scatter,
            viewport_info,
            viewport_set_rect,
            viewport_input,
            viewport_set_mode,
            set_before,
            sample_pixel,
            sample_chroma,
            looks_list,
            look_thumbnail,
            looks_import,
            looks_recognise,
            looks_display_luts,
            looks_watch_downloads,
            looks_set_default,
            looks_remove,
            export_estimate,
            export_start,
            export_cancel,
            cli_preview,
            export_busy,
            restart_app,
            thumbnails,
            choose_folder,
            choose_look_files,
            open_url,
            reveal_in_finder,
            trash_path,
            preferences_get,
            preferences_set,
            apple_auto_status,
            install_cli_link,
            cli_link_status,
            app_info,
        ])
        .events(collect_events![
            wire::PhotoProgress,
            wire::ThumbnailReady,
            wire::RenderStats,
            wire::ExportProgress,
            wire::LooksChanged,
            wire::UpdateState,
            wire::ErrorEvent,
            wire::LooksImportProgress,
            wire::DownloadFound,
            wire::MenuActionEvent,
            wire::SettingsChanged,
            wire::ExportBusy,
            wire::ViewportFrame,
        ])
        .typ::<ErrorScope>()
}

#[cfg(test)]
mod tests {
    //! The commands through Tauri's own IPC handler (mock runtime, no webview): the argument names
    //! `src/api.ts` sends, the shapes that come back, and the error that every command gives when
    //! the application could not start.

    use serde_json::{Value, json};
    use tauri::WebviewWindow;
    use tauri::ipc::{CallbackFn, InvokeBody};
    use tauri::test::{
        INVOKE_KEY, MockRuntime, get_ipc_response, mock_builder, mock_context, noop_assets,
    };
    use tauri::webview::InvokeRequest;

    use super::*;
    use crate::testing::{Harness, wait_until};

    fn app_with(state: AppState) -> (tauri::App<MockRuntime>, WebviewWindow<MockRuntime>) {
        let app = mock_builder()
            .invoke_handler(tauri::generate_handler![
                open_paths,
                session_state,
                close_session,
                set_current,
                get_settings,
                update_settings,
                copy_settings,
                paste_settings,
                viewport_info,
                viewport_set_rect,
                export_busy,
                looks_recognise,
                preferences_get,
                apple_auto_status,
                app_info,
            ])
            .manage(state)
            .build(mock_context(noop_assets()))
            .expect("the mock app");
        let window = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .expect("a mock window");
        (app, window)
    }

    /// `set_menu_state` must not be `async`: an `async` command runs on a
    /// worker thread of its own, so two quick calls could overtake each other and leave the Edit
    /// menu half switched. An `async fn` returns a future, so this does not compile if the command
    /// becomes one.
    #[test]
    fn the_menu_state_command_runs_on_the_main_thread_in_order() {
        fn assert_sync<F: Fn(AppHandle, MenuState) -> Result<(), AppError>>(_: F) {}
        assert_sync(set_menu_state);
    }

    fn call(window: &WebviewWindow<MockRuntime>, cmd: &str, args: Value) -> Result<Value, Value> {
        get_ipc_response(
            window,
            InvokeRequest {
                cmd: cmd.into(),
                callback: CallbackFn(0),
                error: CallbackFn(1),
                url: "tauri://localhost/".parse().expect("url"),
                body: InvokeBody::Json(args),
                headers: Default::default(),
                invoke_key: INVOKE_KEY.to_string(),
            },
        )
        .map(|body| body.deserialize::<Value>().expect("json"))
    }

    #[test]
    fn commands_take_the_arguments_of_api_ts_and_answer_in_the_shapes_of_types_ts() {
        let h = Harness::new();
        let (_app, window) = app_with(AppState::ready(h.api.clone()));
        let empty = call(&window, "session_state", json!({})).expect("session_state");
        assert_eq!(
            empty,
            json!({ "folder_name": null, "photos": [], "current_index": null, "picked": [], "default_export_folder": null, "opened": null })
        );
        assert_eq!(
            call(&window, "viewport_info", json!({})),
            Ok(json!({ "backend": "fallback" }))
        );
        assert_eq!(call(&window, "export_busy", json!({})), Ok(json!(false)));
        // The cell: a rectangle in CSS pixels and the device scale, as `PhotoCell` reports it.
        assert_eq!(
            call(
                &window,
                "viewport_set_rect",
                json!({ "rect": { "x": 1, "y": 2, "width": 300, "height": 200, "scale": 2 } })
            ),
            Ok(json!(null))
        );
        assert!(
            call(&window, "viewport_set_rect", json!({ "rect": { "x": 1 } })).is_err(),
            "a rectangle without a size is refused before the handler runs"
        );
        assert_eq!(
            call(&window, "app_info", json!({})).expect("app_info")["version"].as_str(),
            Some(ym_core::version::app_version())
        );

        crate::testing::write_arw(h.photos.path(), "DSC00001.ARW", 1);
        let opened = call(
            &window,
            "open_paths",
            json!({ "paths": [h.photos.path().to_string_lossy()] }),
        )
        .expect("open_paths");
        let id = opened["photos"][0]["id"].as_u64().expect("an id");
        assert_eq!(
            opened["photos"][0]["camera"]["match_kind"],
            json!({ "kind": "exact" })
        );
        wait_until("development", || {
            call(&window, "session_state", json!({}))
                .is_ok_and(|s| s["photos"][0]["status"] == json!({ "kind": "ready" }))
        });

        // `photoId` on the wire is `photo_id` in Rust.
        let settings =
            call(&window, "get_settings", json!({ "photoId": id })).expect("get_settings");
        assert_eq!(settings["photo_id"], json!(id));
        assert_eq!(settings["settings"]["exposure"], json!({ "kind": "auto" }));
        assert_eq!(settings["auto"]["exposure"], json!(true));
        let updated = call(
            &window,
            "update_settings",
            json!({ "photoId": id, "patch": { "exposure": { "kind": "manual", "ev": 0.25 }, "crop": null } }),
        )
        .expect("update_settings");
        assert_eq!(
            updated["settings"]["exposure"],
            json!({ "kind": "manual", "ev": 0.25 })
        );
        assert_eq!(updated["values"]["exposure_ev"], json!(0.25));
        assert_eq!(updated["settings"]["orientation"], json!(0));

        // Copy and paste: the patch leaves out the crop, the orientation and the white pick, and a
        // paste answers with one undo step per photo and the photos it left out.
        let copied =
            call(&window, "copy_settings", json!({ "photoId": id })).expect("copy_settings");
        assert_eq!(
            copied["exposure"],
            json!({ "kind": "manual", "ev": 0.25 }),
            "a manual exposure is copied as it is"
        );
        for left_out in ["crop", "orientation", "white_pick"] {
            assert!(copied.get(left_out).is_none(), "{left_out} is not copied");
        }
        assert_eq!(copied["white_balance"], json!({ "kind": "auto" }));
        // Contrast is Auto or a manual value (`docs/19` section 5); the step says which.
        let pasted = call(
            &window,
            "paste_settings",
            json!({ "photoIds": [id, 9999], "patch": { "contrast": { "kind": "manual", "value": 33 } } }),
        )
        .expect("paste_settings");
        assert_eq!(pasted["skipped"], json!([9999]));
        assert_eq!(pasted["pasted"][0]["id"], json!(id));
        assert_eq!(
            pasted["pasted"][0]["before"],
            json!({ "contrast": { "kind": "auto" } })
        );
        assert_eq!(
            pasted["pasted"][0]["after"],
            json!({ "contrast": { "kind": "manual", "value": 33.0 } })
        );
        // The older plain number still reads: 0 is Auto, any other number Manual.
        let plain = call(
            &window,
            "update_settings",
            json!({ "photoId": id, "patch": { "contrast": 0, "saturation": -12 } }),
        )
        .expect("a plain number");
        assert_eq!(plain["settings"]["contrast"], json!({ "kind": "auto" }));
        assert_eq!(
            plain["settings"]["saturation"],
            json!({ "kind": "manual", "value": -12.0 })
        );
        assert_eq!(plain["auto"]["contrast"], json!(true));
        assert_eq!(plain["auto"]["saturation"], json!(false));
        assert_eq!(plain["values"]["saturation"], json!(-12.0));
        // Without an advisor Apple Auto is off: the rule's 0, and the photo says so.
        assert_eq!(plain["estimates"]["contrast"], json!(0.0));
        assert_eq!(plain["estimates"]["source"], json!("rule"));
        assert_eq!(plain["estimates"]["apple"], json!({ "state": "off" }));
        assert_eq!(
            call(&window, "apple_auto_status", json!({})).expect("apple_auto_status"),
            json!({
                "available": false, "reason": null, "info": null, "preference": true,
                "mode": "from_preference", "active": false, "prompt": 3, "table": 2,
                "instructions": "default",
                "instructions_sha256": ym_core::apple_auto::InstructionsId::default().instructions_sha256,
                "own_instructions": "",
                "default_instructions": ym_core::apple_auto::DEFAULT_INSTRUCTIONS,
                "instructions_max_chars": 2000
            })
        );
        let err = call(&window, "copy_settings", json!({ "photoId": 9999 })).expect_err("unknown");
        assert_eq!(err["scope"], json!("open"));
        assert!(
            call(
                &window,
                "paste_settings",
                json!({ "photo_ids": [id], "patch": {} })
            )
            .is_err(),
            "the wire name is camelCase"
        );

        // An error is `{ scope, message }`.
        let err =
            call(&window, "set_current", json!({ "photoId": 9999 })).expect_err("unknown photo");
        assert_eq!(err["scope"], json!("open"));
        let err =
            call(&window, "get_settings", json!({ "photoId": 9999 })).expect_err("unknown photo");
        assert_eq!(err["scope"], json!("open"));
        assert!(
            err["message"]
                .as_str()
                .expect("message")
                .contains("isn't in the session")
        );
        // A rejected argument is rejected before the handler runs.
        assert!(
            call(&window, "get_settings", json!({ "photo_id": id })).is_err(),
            "the wire name is camelCase"
        );

        // looks_recognise never rejects.
        let seen = call(
            &window,
            "looks_recognise",
            json!({ "path": "/nowhere/Gone.zip" }),
        )
        .expect("recognise");
        assert_eq!(seen["recognition"]["kind"], json!("missing"));
        assert_eq!(seen["source"], json!("zip"));
        assert!(call(&window, "preferences_get", json!({})).expect("prefs")["export_defaults"]["bit_depth"].is_number());
        call(&window, "close_session", json!({})).expect("close");
        assert_eq!(
            call(&window, "session_state", json!({})).expect("state")["photos"],
            json!([])
        );
    }

    #[test]
    fn an_application_that_could_not_start_answers_every_command_with_the_reason() {
        let reason = AppError::new(
            ErrorScope::App,
            "Yellowmoon's data folder is missing. Reinstall the app.",
        );
        let (_app, window) = app_with(AppState::failed(reason.clone()));
        for cmd in [
            "session_state",
            "preferences_get",
            "apple_auto_status",
            "app_info",
            "export_busy",
        ] {
            let err = call(&window, cmd, json!({})).expect_err(cmd);
            assert_eq!(
                err,
                json!({ "scope": "app", "message": reason.message }),
                "{cmd}"
            );
        }
    }
}
