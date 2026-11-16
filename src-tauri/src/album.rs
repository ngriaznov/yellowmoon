//! The shell's side of albums (`docs/31`) and of the panels of presets (`docs/30`): the save and open panels, the question "Save
//! changes to the album Lisbon?", and the window's title and edited dot.
//!
//! The core keeps the album's state ([`ym_core::album::AlbumState`]: file, name, whether it was edited since the last save) and tells
//! the shell when it changes (`album_changed`); the shell shows it ([`sync_window`]) and asks before anything throws it away
//! ([`confirm_close`], and `crate::app` for the window's close and Quit). Nothing here reads or writes a file: the panels give a path
//! and the core writes it (`AppApi::album_save`, `AppApi::preset_export`).
//!
//! **Threads.** [`confirm_close`], [`save`], [`export_preset`] and [`import_presets_dialog`] block on a system panel, so they run on a
//! blocking thread (every command does) and never on the main thread, which the panel is shown by. The close and quit handlers of
//! `crate::app` run on the main thread and use [`ask`] with a callback instead.

use std::path::PathBuf;

use tauri::{AppHandle, Manager, Runtime};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind, MessageDialogResult};

use crate::api::AppApi;
use crate::commands::AppState;
use crate::wire::{AlbumSaved, AppError, ImportResult, PresetExported};

/// The one filter of File › Open Files…: "Photos and albums" (`docs/31` section 4). rfd on macOS takes the union of every filter's
/// extensions into one list, so a second filter would show the same files; one filter says so.
pub const OPEN_FILTER_NAME: &str = "Photos and albums";

/// The extensions of [`OPEN_FILTER_NAME`].
pub const OPEN_FILTER_EXTENSIONS: [&str; 3] = ["arw", "ARW", "ymalbum"];

/// The extension of an album file.
const ALBUM_EXTENSION: &str = "ymalbum";

/// The extension of a preset file.
const PRESET_EXTENSION: &str = "ympreset";

/// The label of the button that saves.
pub const SAVE: &str = "Save";

/// The label of the button that goes on without saving.
pub const DONT_SAVE: &str = "Don't Save";

/// The label of the button that stays.
pub const CANCEL: &str = "Cancel";

/// What the person answered to "Save changes to the album …?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// Write the album, then go on.
    Save,
    /// Go on without writing.
    DontSave,
    /// Stay.
    Cancel,
}

/// "Save changes to the album Lisbon?" (`docs/31` section 6).
#[must_use]
pub fn question(name: &str) -> String {
    format!("Save changes to the album {name}?")
}

/// The line under the question.
const DETAIL: &str = "If you don't save, the changes you made to the album are lost.";

/// What the panel's answer means. Closing the panel is Cancel, as it is for macOS.
#[must_use]
pub fn answer_of(result: &MessageDialogResult) -> Answer {
    match result {
        MessageDialogResult::Yes | MessageDialogResult::Ok => Answer::Save,
        MessageDialogResult::No => Answer::DontSave,
        MessageDialogResult::Custom(label) if label == SAVE => Answer::Save,
        MessageDialogResult::Custom(label) if label == DONT_SAVE => Answer::DontSave,
        _ => Answer::Cancel,
    }
}

/// Shows the question and calls `then` with the answer, from the dialog's thread. Does not wait: this is for the main thread.
pub fn ask<R: Runtime>(
    app: &AppHandle<R>,
    name: &str,
    then: impl FnOnce(Answer) + Send + 'static,
) {
    app.dialog()
        .message(DETAIL)
        .title(question(name))
        .kind(MessageDialogKind::Warning)
        // The first button is the default one: Save.
        .buttons(MessageDialogButtons::YesNoCancelCustom(
            SAVE.to_owned(),
            DONT_SAVE.to_owned(),
            CANCEL.to_owned(),
        ))
        .show_with_result(move |result| then(answer_of(&result)));
}

/// The open album's name when it has changes that are not saved.
#[must_use]
pub fn edited_album(api: &AppApi) -> Option<String> {
    api.album_state()
        .ok()
        .flatten()
        .filter(|a| a.edited)
        .map(|a| a.name)
}

/// Asks the question when the open album has unsaved changes, and says whether to go on (`open_dialog`, with an album chosen): `true` when it was
/// saved, when the person chose Don't Save, or when there was nothing to ask; `false` on Cancel, and on a Save panel that was
/// cancelled. Blocks on the panel: not for the main thread.
///
/// # Errors
///
/// The error of a Save that failed: the session stays, and the page shows the sentence.
pub fn confirm_close<R: Runtime>(app: &AppHandle<R>, api: &AppApi) -> Result<bool, AppError> {
    let Some(name) = edited_album(api) else {
        return Ok(true);
    };
    let (tx, rx) = std::sync::mpsc::channel();
    ask(app, &name, move |answer| {
        let _ = tx.send(answer);
    });
    match rx.recv().unwrap_or(Answer::Cancel) {
        Answer::Cancel => Ok(false),
        Answer::DontSave => Ok(true),
        Answer::Save => Ok(save(app, api, false)?.is_some()),
    }
}

/// The `.ymalbum` extension on `path`, if it has none.
fn with_album_extension(path: PathBuf) -> PathBuf {
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(ALBUM_EXTENSION))
    {
        path
    } else {
        let mut name = path.into_os_string();
        name.push(".");
        name.push(ALBUM_EXTENSION);
        PathBuf::from(name)
    }
}

/// File › Save and Save As… (`docs/31` section 5): with `as_new`, or when there is no album file yet, the system's save panel (the
/// name `Lisbon.ymalbum`, starting in the first photo's folder; for Save As the album's own folder and name); otherwise the album is
/// written in place with no question. `None` when the panel is cancelled. Blocks on the panel: not for the main thread.
///
/// # Errors
///
/// [`AppError`] with scope `album` when the write fails: nothing was changed.
pub fn save<R: Runtime>(
    app: &AppHandle<R>,
    api: &AppApi,
    as_new: bool,
) -> Result<Option<AlbumSaved>, AppError> {
    let path = match api.album_state()? {
        Some(album) if !as_new => PathBuf::from(album.path),
        _ => {
            let (folder, name) = api.album_save_defaults();
            let mut panel = app
                .dialog()
                .file()
                .set_title("Save album")
                .set_file_name(name)
                .add_filter("Yellowmoon album", &[ALBUM_EXTENSION]);
            if let Some(dir) = folder.filter(|d| d.is_dir()) {
                panel = panel.set_directory(dir);
            }
            let Some(chosen) = panel.blocking_save_file().and_then(|p| p.into_path().ok()) else {
                return Ok(None);
            };
            with_album_extension(chosen)
        }
    };
    api.album_save(&path).map(Some)
}

/// The suggested file name of a preset: `Warm evening.ympreset`.
#[must_use]
pub fn preset_file_name(name: &str) -> String {
    format!(
        "{}.{PRESET_EXTENSION}",
        ym_core::presets::PresetStore::safe_file_name(name)
    )
}

/// `preset_export`: the save panel, then the file. `None` when the person cancels. Blocks on the panel.
///
/// # Errors
///
/// [`AppError`] with scope `presets`.
pub fn export_preset<R: Runtime>(
    app: &AppHandle<R>,
    api: &AppApi,
    name: &str,
) -> Result<Option<PresetExported>, AppError> {
    let Some(chosen) = app
        .dialog()
        .file()
        .set_title("Export preset")
        .set_file_name(preset_file_name(name))
        .add_filter("Yellowmoon preset", &[PRESET_EXTENSION])
        .blocking_save_file()
        .and_then(|p| p.into_path().ok())
    else {
        return Ok(None);
    };
    api.preset_export(name, &chosen)?;
    Ok(Some(PresetExported {
        path: chosen.to_string_lossy().into_owned(),
    }))
}

/// `presets_import_dialog`: the open panel for `.ympreset` files (several), then the import. `None` when cancelled.
///
/// # Errors
///
/// [`AppError`] only if the shapes of the core and of the bridge disagree (a bug).
pub fn import_presets_dialog<R: Runtime>(
    app: &AppHandle<R>,
    api: &AppApi,
) -> Result<Option<ImportResult>, AppError> {
    let paths: Vec<String> = app
        .dialog()
        .file()
        .set_title("Import presets")
        .add_filter("Yellowmoon presets", &[PRESET_EXTENSION])
        .blocking_pick_files()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    if paths.is_empty() {
        return Ok(None);
    }
    api.presets_import(&paths, false).map(Some)
}

/// `preset_choose_export`: the open panel for a photo Yellowmoon exported (its `ym:` settings are in the file, `docs/30` section 12).
/// Blocks on the panel. `None` when cancelled.
#[must_use]
pub fn choose_exported_photo<R: Runtime>(app: &AppHandle<R>) -> Option<String> {
    app.dialog()
        .file()
        .set_title("Choose an exported photo")
        .add_filter("Exported photos", &["jpg", "jpeg", "tif", "tiff", "png", "dng", "xmp"])
        .blocking_pick_file()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Whether `paths` (the answer of the open panel) hold an album, which replaces the session. An album with other paths, or with
/// another album, is refused in words.
///
/// # Errors
///
/// [`AppError`] with scope `album`: "Open an album on its own, or open photos. Yellowmoon can't open both in one go." or "Open one
/// album at a time."
pub fn is_album_choice(paths: &[String]) -> Result<bool, AppError> {
    crate::api::album_in(paths).map(|a| a.is_some())
}

/// The window's title: "Yellowmoon" with no album, the album's name while it is open, "Lisbon — Edited" while it has unsaved
/// changes (the Window menu, Mission Control and VoiceOver read the title; the hidden title bar does not draw it).
#[must_use]
pub fn window_title(album: Option<(&str, bool)>) -> String {
    match album {
        None => "Yellowmoon".to_owned(),
        Some((name, false)) => name.to_owned(),
        Some((name, true)) => format!("{name} — Edited"),
    }
}

/// Sets the main window's title and, on macOS, the edited dot of its close button (`NSWindow.isDocumentEdited`) to match the open
/// album. Called on `album_changed`, from any thread: the dot is set on the main thread without waiting for it.
pub fn set_album_window<R: Runtime>(app: &AppHandle<R>, album: Option<(&str, bool)>) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    if let Err(e) = window.set_title(&window_title(album)) {
        tracing::debug!(error = %e, "the window's title was not set");
    }
    #[cfg(target_os = "macos")]
    crate::native::set_document_edited(&window, album.is_some_and(|(_, edited)| edited));
}

/// Reads the album's state and shows it ([`set_album_window`]).
pub fn sync_window<R: Runtime>(app: &AppHandle<R>) {
    let state = app
        .try_state::<AppState>()
        .and_then(|s| s.api().ok())
        .and_then(|api| api.album_state().ok().flatten());
    set_album_window(app, state.as_ref().map(|a| (a.name.as_str(), a.edited)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_question_is_the_one_of_the_spec() {
        assert_eq!(question("Lisbon"), "Save changes to the album Lisbon?");
    }

    #[test]
    fn the_buttons_answer_as_they_are_labelled() {
        assert_eq!(answer_of(&MessageDialogResult::Custom(SAVE.into())), Answer::Save);
        assert_eq!(
            answer_of(&MessageDialogResult::Custom(DONT_SAVE.into())),
            Answer::DontSave
        );
        assert_eq!(
            answer_of(&MessageDialogResult::Custom(CANCEL.into())),
            Answer::Cancel
        );
        assert_eq!(answer_of(&MessageDialogResult::Cancel), Answer::Cancel);
        assert_eq!(answer_of(&MessageDialogResult::Yes), Answer::Save);
        assert_eq!(answer_of(&MessageDialogResult::No), Answer::DontSave);
    }

    #[test]
    fn the_title_says_edited_only_while_it_is() {
        assert_eq!(window_title(None), "Yellowmoon");
        assert_eq!(window_title(Some(("Lisbon", false))), "Lisbon");
        assert_eq!(window_title(Some(("Lisbon", true))), "Lisbon — Edited");
    }

    #[test]
    fn a_name_the_panel_gave_without_an_extension_gets_one() {
        assert_eq!(
            with_album_extension(PathBuf::from("/a/Lisbon")),
            PathBuf::from("/a/Lisbon.ymalbum")
        );
        assert_eq!(
            with_album_extension(PathBuf::from("/a/Lisbon.ymalbum")),
            PathBuf::from("/a/Lisbon.ymalbum")
        );
        assert_eq!(
            with_album_extension(PathBuf::from("/a/Lisbon.YMALBUM")),
            PathBuf::from("/a/Lisbon.YMALBUM")
        );
        // A name with a dot of its own is a name, not an extension.
        assert_eq!(
            with_album_extension(PathBuf::from("/a/Trip 2026.10")),
            PathBuf::from("/a/Trip 2026.10.ymalbum")
        );
    }

    #[test]
    fn the_open_filter_is_one_filter_for_photos_and_albums() {
        assert_eq!(OPEN_FILTER_NAME, "Photos and albums");
        assert_eq!(OPEN_FILTER_EXTENSIONS, ["arw", "ARW", "ymalbum"]);
    }

    #[test]
    fn a_preset_is_suggested_under_its_own_name() {
        assert_eq!(preset_file_name("Warm evening"), "Warm evening.ympreset");
        assert_eq!(preset_file_name("a/b"), "a-b.ympreset");
    }

    #[test]
    fn an_album_is_opened_on_its_own() {
        let one = |p: &str| p.to_owned();
        assert_eq!(is_album_choice(&[one("/a/b.ARW")]), Ok(false));
        assert_eq!(is_album_choice(&[one("/a/Lisbon.ymalbum")]), Ok(true));
        assert_eq!(is_album_choice(&[one("/a/Lisbon.YMALBUM")]), Ok(true));
        let mixed = is_album_choice(&[one("/a/Lisbon.ymalbum"), one("/a/b.ARW")]).expect_err("mixed");
        assert_eq!(
            mixed.message,
            "Open an album on its own, or open photos. Yellowmoon can't open both in one go."
        );
        assert_eq!(mixed.scope, crate::wire::ErrorScope::Album);
        let two = is_album_choice(&[one("/a/A.ymalbum"), one("/a/B.ymalbum")]).expect_err("two");
        assert_eq!(two.message, "Open one album at a time.");
    }
}
