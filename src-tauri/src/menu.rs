//! The menu bar of `docs/11-ui.md` §Menus.
//!
//! [`spec`] is the menu as plain data (titles, labels, shortcuts, what each item does), so the
//! structure is tested without a window; [`build`] turns it into the system's menu. An item either
//! asks the webview to act (`menu_action { action }`, handled by `src/state/menu.ts`) or is the
//! shell's own business: Show Looks Folder, the Help links, Report a Problem, About, Quit and the
//! standard window items.
//!
//! Shortcuts are given to the items whose keys do not also type: ⌘, ⌘O ⇧⌘O ⌘S ⇧⌘S ⌘E ⌘W ⌥⌘S ⌘= ⌘- ⌘Q (ten of ours; ⌘Q is the
//! system's). ⌘S and ⇧⌘S are File › Save and Save As… (`docs/31`), ⌥⌘S is Looks › Save Preset… (`docs/30` section 9.8). ⌘Z,
//! ⇧⌘Z and ⌘A stay with the webview, which sends them to a text box when one has focus and to the
//! photos otherwise; a menu item bound to them would take the keys away from the text boxes. So do
//! the three settings items, whose keys ⇧⌘C, ⇧⌘V and ⌥⇧⌘V are the webview's too: ⇧⌘V and ⌥⇧⌘V mean
//! "Paste and Match Style" in other apps, and a key that the page leaves alone in a text box falls
//! through to the menu on macOS and would paste the settings onto photos from there. The menu
//! shows all of these items without a key. The single-key shortcuts (P, Z, C, W, N, \ and the arrows)
//! are not menu shortcuts for the same reason: View › Nodes shows no key.
//!
//! Seven items change with what is open: Copy Settings needs a developed current photo, Paste
//! Settings something copied as well, and Paste Settings to Picked something copied and a picked
//! photo; Save Preset and the Apply Preset submenu need a developed current photo and no dialog; Save and Save As… need a photo and no
//! dialog. The webview says which are available (`set_menu_state`, [`apply_state`]); the menu starts
//! with all of them off, as Start has nothing to copy or save. The names in Apply Preset are the shell's own: it lists the presets and
//! rebuilds the submenu when they change ([`refresh_presets`], `presets_changed`).

use tauri::menu::{
    AboutMetadata, CheckMenuItem, Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu,
};
use tauri::{AppHandle, Manager, Runtime};

use crate::about;
use crate::commands::AppState;
use crate::wire::{MenuAction, MenuState, PresetsListing};

/// What an item of the menu does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Do {
    /// Ask the webview (`menu_action`).
    Webview(MenuAction),
    /// Open the folder of imported looks in Finder.
    ShowLooksFolder,
    /// Open the folder of saved presets in Finder (made if it is missing).
    ShowPresetsFolder,
    /// Open the project's GitHub page.
    OpenGithub,
    /// Copy `ym doctor` (after asking) and open the bug report form.
    ReportProblem,
}

/// A system item with its own behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standard {
    /// The system's About panel, with the text of [`crate::about`].
    About,
    /// Services (macOS).
    Services,
    /// Hide Yellowmoon.
    Hide,
    /// Hide Others.
    HideOthers,
    /// Show All.
    ShowAll,
    /// Quit (⌘Q).
    Quit,
    /// Cut, Copy and Paste reach the text boxes of the webview through the responder chain.
    Cut,
    /// Copy.
    Copy,
    /// Paste.
    Paste,
    /// Minimize.
    Minimize,
    /// Zoom (maximize).
    Maximize,
    /// Full screen.
    Fullscreen,
}

/// One entry of a menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    /// An item of ours.
    Item {
        /// The id the menu event carries.
        id: &'static str,
        /// The label.
        label: &'static str,
        /// The shortcut, in the system menu's notation.
        shortcut: Option<&'static str>,
        /// What it does.
        action: Do,
    },
    /// A submenu whose items are the saved presets, which the shell makes and remakes ([`refresh_presets`]); it starts with the
    /// greyed "No Presets".
    PresetSubmenu {
        /// The id of the submenu, which is also the id of its state item ([`STATE_ITEMS`]).
        id: &'static str,
        /// The label.
        label: &'static str,
    },
    /// A system item.
    Standard(Standard),
    /// A separator.
    Separator,
}

/// A menu of the bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Menus {
    /// The title.
    pub title: &'static str,
    /// Its entries.
    pub entries: Vec<Entry>,
}

const fn item(
    id: &'static str,
    label: &'static str,
    shortcut: Option<&'static str>,
    action: Do,
) -> Entry {
    Entry::Item {
        id,
        label,
        shortcut,
        action,
    }
}

const fn web(
    id: &'static str,
    label: &'static str,
    shortcut: Option<&'static str>,
    action: MenuAction,
) -> Entry {
    item(id, label, shortcut, Do::Webview(action))
}

/// The menu bar as data (`docs/11` §Menus).
#[must_use]
pub fn spec() -> Vec<Menus> {
    use Entry::{Separator, Standard as Std};
    vec![
        Menus {
            title: "Yellowmoon",
            entries: vec![
                Std(Standard::About),
                Separator,
                web(
                    "preferences",
                    "Preferences…",
                    Some("CmdOrCtrl+,"),
                    MenuAction::Preferences,
                ),
                web(
                    "check_for_updates",
                    "Check for Updates…",
                    None,
                    MenuAction::CheckForUpdates,
                ),
                web(
                    "install_cli",
                    "Install Command-Line Tool…",
                    None,
                    MenuAction::InstallCli,
                ),
                Separator,
                Std(Standard::Services),
                Std(Standard::Hide),
                Std(Standard::HideOthers),
                Std(Standard::ShowAll),
                Separator,
                Std(Standard::Quit),
            ],
        },
        Menus {
            title: "File",
            entries: vec![
                web(
                    "open_files",
                    "Open Files…",
                    Some("CmdOrCtrl+O"),
                    MenuAction::OpenFiles,
                ),
                web(
                    "open_folder",
                    "Open Folder…",
                    Some("CmdOrCtrl+Shift+O"),
                    MenuAction::OpenFolder,
                ),
                Separator,
                web("save", "Save", Some("CmdOrCtrl+S"), MenuAction::Save),
                web(
                    "save_as",
                    "Save As…",
                    Some("CmdOrCtrl+Shift+S"),
                    MenuAction::SaveAs,
                ),
                Separator,
                web("export", "Export…", Some("CmdOrCtrl+E"), MenuAction::Export),
                Separator,
                web(
                    "close_session",
                    "Close Session",
                    Some("CmdOrCtrl+W"),
                    MenuAction::CloseSession,
                ),
            ],
        },
        Menus {
            title: "Edit",
            entries: vec![
                web("undo", "Undo", None, MenuAction::Undo),
                web("redo", "Redo", None, MenuAction::Redo),
                Separator,
                web(
                    "copy_settings",
                    "Copy Settings",
                    None,
                    MenuAction::CopySettings,
                ),
                web(
                    "paste_settings",
                    "Paste Settings",
                    None,
                    MenuAction::PasteSettings,
                ),
                web(
                    "paste_settings_to_picked",
                    "Paste Settings to Picked",
                    None,
                    MenuAction::PasteSettingsToPicked,
                ),
                Separator,
                web("pick", "Pick", None, MenuAction::Pick),
                web("pick_all", "Pick All", None, MenuAction::PickAll),
                web("clear_picks", "Clear Picks", None, MenuAction::ClearPicks),
                Separator,
                Std(Standard::Cut),
                Std(Standard::Copy),
                Std(Standard::Paste),
            ],
        },
        Menus {
            title: "View",
            entries: vec![
                web(
                    "before_after",
                    "Before/After",
                    None,
                    MenuAction::BeforeAfter,
                ),
                web("nodes", "Nodes", None, MenuAction::Nodes),
                web("fit", "Fit", None, MenuAction::Fit),
                web("zoom_100", "100%", None, MenuAction::Zoom100),
                web(
                    "zoom_in",
                    "Zoom In",
                    Some("CmdOrCtrl+="),
                    MenuAction::ZoomIn,
                ),
                web(
                    "zoom_out",
                    "Zoom Out",
                    Some("CmdOrCtrl+-"),
                    MenuAction::ZoomOut,
                ),
                Separator,
                web("next_photo", "Next Photo", None, MenuAction::NextPhoto),
                web(
                    "previous_photo",
                    "Previous Photo",
                    None,
                    MenuAction::PreviousPhoto,
                ),
                Separator,
                Std(Standard::Fullscreen),
            ],
        },
        Menus {
            title: "Photo",
            entries: vec![
                web("crop", "Crop", None, MenuAction::Crop),
                web("auto_crop", "Auto Crop", None, MenuAction::AutoCrop),
                web("pick_white", "Pick White", None, MenuAction::PickWhite),
                Separator,
                web(
                    "remove_photo",
                    "Remove Photo",
                    None,
                    MenuAction::RemovePhoto,
                ),
                Separator,
                web("bypass_node", "Bypass Node", None, MenuAction::BypassNode),
                web(
                    "mask_view",
                    "Show or Hide Node Mask",
                    None,
                    MenuAction::MaskView,
                ),
            ],
        },
        Menus {
            title: "Looks",
            entries: vec![
                web(
                    "import_arri_looks",
                    "Import ARRI Looks…",
                    None,
                    MenuAction::ImportArriLooks,
                ),
                web(
                    "import_cube",
                    "Import LUTs and look files…",
                    None,
                    MenuAction::ImportCube,
                ),
                web(
                    "make_default_look",
                    "Make Current Look Default",
                    None,
                    MenuAction::MakeDefaultLook,
                ),
                Separator,
                web("look_1", "Apply Look 1", None, MenuAction::Look1),
                web("look_2", "Apply Look 2", None, MenuAction::Look2),
                web("look_3", "Apply Look 3", None, MenuAction::Look3),
                web("look_4", "Apply Look 4", None, MenuAction::Look4),
                web("look_5", "Apply Look 5", None, MenuAction::Look5),
                web("look_6", "Apply Look 6", None, MenuAction::Look6),
                web("look_7", "Apply Look 7", None, MenuAction::Look7),
                web("look_8", "Apply Look 8", None, MenuAction::Look8),
                web("look_9", "Apply Look 9", None, MenuAction::Look9),
                Separator,
                web(
                    "save_preset",
                    "Save Preset…",
                    Some("CmdOrCtrl+Alt+S"),
                    MenuAction::SavePreset,
                ),
                web(
                    "import_presets",
                    "Import Presets…",
                    None,
                    MenuAction::ImportPresets,
                ),
                Entry::PresetSubmenu {
                    id: "apply_preset",
                    label: "Apply Preset",
                },
                web(
                    "new_preset_from_export",
                    "New Preset from Exported Photo…",
                    None,
                    MenuAction::NewPresetFromExport,
                ),
                Separator,
                item(
                    "show_presets_folder",
                    "Show Presets Folder",
                    None,
                    Do::ShowPresetsFolder,
                ),
                Separator,
                item(
                    "show_looks_folder",
                    "Show Looks Folder",
                    None,
                    Do::ShowLooksFolder,
                ),
            ],
        },
        Menus {
            title: "Window",
            entries: vec![Std(Standard::Minimize), Std(Standard::Maximize)],
        },
        Menus {
            title: "Help",
            entries: vec![
                item("open_github", "Yellowmoon on GitHub", None, Do::OpenGithub),
                item(
                    "report_problem",
                    "Report a Problem…",
                    None,
                    Do::ReportProblem,
                ),
            ],
        },
    ]
}

/// The ids of the items that change with what is open, in the order of [`MenuState`]'s fields.
pub const STATE_ITEMS: [&str; 7] = [
    "copy_settings",
    "paste_settings",
    "paste_settings_to_picked",
    "save_preset",
    "apply_preset",
    "save",
    "save_as",
];

/// `(id, available)` for each of [`STATE_ITEMS`] under `state`.
#[must_use]
pub fn availability(state: &MenuState) -> [(&'static str, bool); 7] {
    [
        (STATE_ITEMS[0], state.copy_settings),
        (STATE_ITEMS[1], state.paste_settings),
        (STATE_ITEMS[2], state.paste_settings_to_picked),
        (STATE_ITEMS[3], state.save_preset),
        (STATE_ITEMS[4], state.apply_preset),
        (STATE_ITEMS[5], state.save),
        (STATE_ITEMS[6], state.save_as),
    ]
}

/// Switches the items of [`STATE_ITEMS`] on or off to match `state`. Does nothing when the app has
/// no menu (a test, a platform without one). On macOS the menu is changed on the main thread, which
/// Tauri arranges.
///
/// # Errors
///
/// The error of changing an item.
pub fn apply_state<R: Runtime>(app: &AppHandle<R>, state: &MenuState) -> tauri::Result<()> {
    let Some(menu) = app.menu() else {
        return Ok(());
    };
    for kind in menu.items()? {
        let MenuItemKind::Submenu(sub) = kind else {
            continue;
        };
        for (id, on) in availability(state) {
            match sub.get(id) {
                Some(MenuItemKind::MenuItem(item)) => item.set_enabled(on)?,
                // Apply Preset is a submenu: greyed as a whole.
                Some(MenuItemKind::Submenu(item)) => item.set_enabled(on)?,
                _ => {}
            }
        }
    }
    Ok(())
}

/// The prefix of the id of a name in Apply Preset: `apply_preset:Warm evening`.
const PRESET_ITEM_PREFIX: &str = "apply_preset:";

/// The most names Apply Preset lists; the rest are on the Presets card (`docs/30` section 8).
pub const MAX_PRESET_NAMES: usize = 30;

/// The id of the item that applies the preset `name`.
#[must_use]
pub fn preset_item_id(name: &str) -> String {
    format!("{PRESET_ITEM_PREFIX}{name}")
}

/// The preset a menu item id names, if it is one of the Apply Preset items.
#[must_use]
pub fn preset_of_item(id: &str) -> Option<&str> {
    id.strip_prefix(PRESET_ITEM_PREFIX)
}

/// One name of Apply Preset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetItem {
    /// The preset's name, which is the label.
    pub name: String,
    /// The preset would change nothing on the current photo: it gets a checkmark.
    pub checked: bool,
}

/// The names of Apply Preset for `list`: in the list's order (by name), at most [`MAX_PRESET_NAMES`], a checkmark on the ones that
/// match the photo the list was asked for. Empty for no presets, which the menu shows as the greyed "No Presets".
#[must_use]
pub fn preset_items(list: &PresetsListing) -> Vec<PresetItem> {
    list.presets
        .iter()
        .take(MAX_PRESET_NAMES)
        .map(|p| PresetItem {
            name: p.name.clone(),
            checked: p.matches,
        })
        .collect()
}

/// Puts `items` in the Apply Preset submenu, in place of what it holds: the names, or "No Presets" greyed.
///
/// # Errors
///
/// The error of changing the menu.
pub fn set_preset_items<R: Runtime>(
    app: &AppHandle<R>,
    submenu: &Submenu<R>,
    items: &[PresetItem],
) -> tauri::Result<()> {
    for old in submenu.items()? {
        submenu.remove(&old)?;
    }
    if items.is_empty() {
        submenu.append(&MenuItem::with_id(
            app,
            "apply_preset_none",
            "No Presets",
            false,
            None::<&str>,
        )?)?;
    }
    for item in items {
        submenu.append(&CheckMenuItem::with_id(
            app,
            preset_item_id(&item.name),
            &item.name,
            true,
            item.checked,
            None::<&str>,
        )?)?;
    }
    Ok(())
}

/// Finds the Apply Preset submenu of the app's menu.
fn preset_submenu<R: Runtime>(app: &AppHandle<R>) -> Option<Submenu<R>> {
    for kind in app.menu()?.items().ok()? {
        let MenuItemKind::Submenu(sub) = kind else {
            continue;
        };
        if let Some(MenuItemKind::Submenu(apply)) = sub.get("apply_preset") {
            return Some(apply);
        }
    }
    None
}

/// Rebuilds Looks › Apply Preset from the presets folder (`presets_changed`). It lists on a thread of its own: the event comes from
/// the thread that changed the folder, which holds locks the listing needs, and the listing may itself announce a change. The
/// checkmarks are the presets that match the current photo as it is now.
pub fn refresh_presets<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    // The handle is dropped: the thread ends by itself.
    drop(std::thread::spawn(move || {
        let Some(api) = app.try_state::<AppState>().and_then(|s| s.api().ok()) else {
            return;
        };
        let current = api.current_photo();
        let Ok(list) = api.presets_list(current) else {
            return;
        };
        let items = preset_items(&list);
        let Some(submenu) = preset_submenu(&app) else {
            return;
        };
        if let Err(e) = set_preset_items(&app, &submenu, &items) {
            tracing::warn!(error = %e, "Apply Preset could not be rebuilt");
        }
    }));
}

/// What the item with this id does, if it is one of ours.
#[must_use]
pub fn action_for(id: &str) -> Option<Do> {
    spec()
        .into_iter()
        .flat_map(|m| m.entries)
        .find_map(|e| match e {
            Entry::Item {
                id: item_id,
                action,
                ..
            } if item_id == id => Some(action),
            _ => None,
        })
}

/// The About panel's metadata: version, copyright and the notices.
fn about_metadata() -> AboutMetadata<'static> {
    let text = about::about_text();
    AboutMetadata {
        name: Some("Yellowmoon".to_owned()),
        version: Some(ym_core::version::app_version().to_owned()),
        copyright: Some("Copyright 2026 Mykyta Hriaznov and Yellowmoon contributors".to_owned()),
        // macOS shows `credits`; GTK shows `comments`.
        credits: Some(text.clone()),
        comments: Some(text),
        license: Some(about::LICENCE.to_owned()),
        website: Some(crate::system::GITHUB_URL.to_owned()),
        website_label: Some("Yellowmoon on GitHub".to_owned()),
        ..AboutMetadata::default()
    }
}

fn standard<R: Runtime>(
    app: &AppHandle<R>,
    which: Standard,
) -> tauri::Result<PredefinedMenuItem<R>> {
    match which {
        Standard::About => {
            PredefinedMenuItem::about(app, Some("About Yellowmoon"), Some(about_metadata()))
        }
        Standard::Services => PredefinedMenuItem::services(app, None),
        Standard::Hide => PredefinedMenuItem::hide(app, None),
        Standard::HideOthers => PredefinedMenuItem::hide_others(app, None),
        Standard::ShowAll => PredefinedMenuItem::show_all(app, None),
        Standard::Quit => PredefinedMenuItem::quit(app, None),
        Standard::Cut => PredefinedMenuItem::cut(app, None),
        Standard::Copy => PredefinedMenuItem::copy(app, None),
        Standard::Paste => PredefinedMenuItem::paste(app, None),
        Standard::Minimize => PredefinedMenuItem::minimize(app, None),
        Standard::Maximize => PredefinedMenuItem::maximize(app, None),
        Standard::Fullscreen => PredefinedMenuItem::fullscreen(app, None),
    }
}

/// Whether a system item exists on this platform (Services and Hide are macOS's).
fn on_this_platform(which: Standard) -> bool {
    cfg!(target_os = "macos")
        || !matches!(
            which,
            Standard::Services | Standard::Hide | Standard::HideOthers | Standard::ShowAll
        )
}

/// Builds the system's menu bar from [`spec`].
///
/// # Errors
///
/// The error of creating an item.
pub fn build<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let menu = Menu::new(app)?;
    for m in spec() {
        let sub = Submenu::new(app, m.title, true)?;
        let mut last_was_separator = true;
        for entry in &m.entries {
            match *entry {
                Entry::Item {
                    id,
                    label,
                    shortcut,
                    ..
                } => {
                    // The items that follow what is open start off: Start has nothing to copy.
                    let on = !STATE_ITEMS.contains(&id);
                    let item = MenuItem::with_id(app, id, label, on, shortcut)?;
                    sub.append(&item)?;
                    last_was_separator = false;
                }
                Entry::PresetSubmenu { id, label } => {
                    // Off, with "No Presets" in it, until the first listing (`refresh_presets`) and a developed photo.
                    let apply = Submenu::with_id(app, id, label, false)?;
                    set_preset_items(app, &apply, &[])?;
                    sub.append(&apply)?;
                    last_was_separator = false;
                }
                Entry::Standard(which) => {
                    if on_this_platform(which) {
                        sub.append(&standard(app, which)?)?;
                        last_was_separator = false;
                    }
                }
                Entry::Separator => {
                    if !last_was_separator {
                        sub.append(&PredefinedMenuItem::separator(app)?)?;
                        last_was_separator = true;
                    }
                }
            }
        }
        menu.append(&sub)?;
    }
    Ok(menu)
}

/// The labels of an item kind, for tests.
#[cfg(test)]
fn label_of<R: Runtime>(item: &MenuItemKind<R>) -> String {
    match item {
        MenuItemKind::MenuItem(i) => i.text().unwrap_or_default(),
        MenuItemKind::Submenu(i) => i.text().unwrap_or_default(),
        MenuItemKind::Predefined(i) => i.text().unwrap_or_default(),
        MenuItemKind::Check(i) => i.text().unwrap_or_default(),
        MenuItemKind::Icon(i) => i.text().unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::str::FromStr;

    fn items() -> Vec<(&'static str, &'static str, Option<&'static str>, Do)> {
        spec()
            .into_iter()
            .flat_map(|m| m.entries)
            .filter_map(|e| match e {
                Entry::Item {
                    id,
                    label,
                    shortcut,
                    action,
                } => Some((id, label, shortcut, action)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_bar_has_the_menus_of_the_spec_in_order() {
        let titles: Vec<_> = spec().iter().map(|m| m.title).collect();
        assert_eq!(
            titles,
            [
                "Yellowmoon",
                "File",
                "Edit",
                "View",
                "Photo",
                "Looks",
                "Window",
                "Help"
            ]
        );
    }

    #[test]
    fn every_item_of_docs_11_is_there() {
        let labels: HashSet<_> = items().into_iter().map(|(_, l, _, _)| l).collect();
        for wanted in [
            // Yellowmoon (About and Quit are system items)
            "Preferences…",
            "Check for Updates…",
            "Install Command-Line Tool…",
            // File
            "Open Files…",
            "Open Folder…",
            "Save",
            "Save As…",
            "Export…",
            "Close Session",
            // Edit
            "Undo",
            "Redo",
            "Copy Settings",
            "Paste Settings",
            "Paste Settings to Picked",
            "Pick",
            "Pick All",
            "Clear Picks",
            // View
            "Before/After",
            "Nodes",
            "Fit",
            "100%",
            "Zoom In",
            "Zoom Out",
            "Next Photo",
            "Previous Photo",
            // Looks
            "Import ARRI Looks…",
            "Import LUTs and look files…",
            "Make Current Look Default",
            "Save Preset…",
            "Import Presets…",
            "New Preset from Exported Photo…",
            "Show Presets Folder",
            "Show Looks Folder",
            // Help
            "Yellowmoon on GitHub",
            "Report a Problem…",
        ] {
            assert!(labels.contains(wanted), "missing {wanted}");
        }
        let entries: Vec<_> = spec().into_iter().flat_map(|m| m.entries).collect();
        assert!(entries.contains(&Entry::Standard(Standard::About)));
        assert!(entries.contains(&Entry::Standard(Standard::Quit)));
    }

    #[test]
    fn ids_are_unique_and_each_webview_action_has_one_item() {
        let all = items();
        let ids: HashSet<_> = all.iter().map(|i| i.0).collect();
        assert_eq!(ids.len(), all.len(), "duplicate ids");
        let actions: Vec<MenuAction> = all
            .iter()
            .filter_map(|(_, _, _, a)| match a {
                Do::Webview(m) => Some(*m),
                _ => None,
            })
            .collect();
        let distinct: HashSet<_> = actions.iter().map(|a| format!("{a:?}")).collect();
        assert_eq!(distinct.len(), actions.len(), "an action has two items");
        // `About` is the in-window dialog: the menu's About item is the system's panel.
        for expected in [
            "Preferences",
            "CheckForUpdates",
            "InstallCli",
            "Export",
            "MakeDefaultLook",
        ] {
            assert!(distinct.contains(expected), "{expected}");
        }
        assert!(!distinct.contains("About"));
    }

    #[test]
    fn menu_action_names_are_the_ones_the_frontend_switches_on() {
        // `src/state/menu.ts` handles every name of `MenuAction` in `src/api/types.ts`.
        let ts = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/api/types.ts"),
        )
        .expect("types.ts");
        let start = ts.find("export type MenuAction").expect("MenuAction");
        let block = &ts[start..ts[start..].find(';').map_or(ts.len(), |e| start + e)];
        let in_ts: HashSet<&str> = block.split('"').skip(1).step_by(2).collect();
        let all = [
            MenuAction::About,
            MenuAction::Preferences,
            MenuAction::CheckForUpdates,
            MenuAction::InstallCli,
            MenuAction::OpenFiles,
            MenuAction::OpenFolder,
            MenuAction::Export,
            MenuAction::CloseSession,
            MenuAction::Undo,
            MenuAction::Redo,
            MenuAction::Pick,
            MenuAction::PickAll,
            MenuAction::ClearPicks,
            MenuAction::BeforeAfter,
            MenuAction::Nodes,
            MenuAction::Fit,
            MenuAction::Zoom100,
            MenuAction::ZoomIn,
            MenuAction::ZoomOut,
            MenuAction::NextPhoto,
            MenuAction::PreviousPhoto,
            MenuAction::ImportArriLooks,
            MenuAction::ImportCube,
            MenuAction::MakeDefaultLook,
            MenuAction::CopySettings,
            MenuAction::PasteSettings,
            MenuAction::PasteSettingsToPicked,
            MenuAction::Crop,
            MenuAction::AutoCrop,
            MenuAction::PickWhite,
            MenuAction::RemovePhoto,
            MenuAction::BypassNode,
            MenuAction::MaskView,
            MenuAction::Look1,
            MenuAction::Look2,
            MenuAction::Look3,
            MenuAction::Look4,
            MenuAction::Look5,
            MenuAction::Look6,
            MenuAction::Look7,
            MenuAction::Look8,
            MenuAction::Look9,
            MenuAction::Save,
            MenuAction::SaveAs,
            MenuAction::SavePreset,
            MenuAction::ImportPresets,
            MenuAction::ApplyPreset,
            MenuAction::NewPresetFromExport,
        ];
        let ours: HashSet<String> = all
            .iter()
            .map(|a| {
                serde_json::to_value(a)
                    .expect("json")
                    .as_str()
                    .expect("string")
                    .to_owned()
            })
            .collect();
        let theirs: HashSet<String> = in_ts.iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(ours, theirs);
    }

    #[test]
    fn shortcuts_parse_and_do_not_take_the_text_keys() {
        let mut seen = HashSet::new();
        for (id, _, shortcut, _) in items() {
            let Some(s) = shortcut else { continue };
            let accel = muda::accelerator::Accelerator::from_str(s)
                .unwrap_or_else(|e| panic!("{id}: {s}: {e}"));
            assert!(seen.insert(accel), "{s} is used twice");
            assert!(
                ![
                    "CmdOrCtrl+Z",
                    "CmdOrCtrl+Shift+Z",
                    "CmdOrCtrl+A",
                    "CmdOrCtrl+X",
                    "CmdOrCtrl+C",
                    "CmdOrCtrl+V",
                    // The webview's table drops these in a text box (`shortcutFor`); a menu item
                    // bound to them would fire there, because macOS passes the key the page left
                    // alone on to the menu.
                    "CmdOrCtrl+Shift+C",
                    "CmdOrCtrl+Shift+V",
                    "CmdOrCtrl+Alt+Shift+V"
                ]
                .contains(&s),
                "{s} belongs to the text boxes"
            );
        }
        // ⌘, ⌘O ⇧⌘O ⌘S ⇧⌘S ⌘E ⌘W ⌥⌘S ⌘= ⌘-
        assert_eq!(seen.len(), 10);
    }

    /// "Single-key shortcuts" off (`docs/28` A-P11) must really turn those keys off, so the native
    /// menu may not hold a key equivalent that is one key: every shortcut has ⌘, and no item of the
    /// menu takes P, Z, C, W, N, F, B, M, A, a digit, a backslash, Delete or Backspace.
    #[test]
    fn no_item_has_a_single_key_equivalent() {
        for (id, _, shortcut, _) in items() {
            let Some(s) = shortcut else { continue };
            assert!(s.contains("CmdOrCtrl+"), "{id}: {s} is not a ⌘ key");
        }
    }

    /// The actions whose keys the switch turns off each have a menu item (`docs/28` A-P11), so
    /// nothing is stranded: the Photo menu and the nine look items.
    #[test]
    fn every_action_of_a_single_key_has_an_item() {
        let actions: Vec<MenuAction> = items()
            .into_iter()
            .filter_map(|(_, _, _, a)| match a {
                Do::Webview(m) => Some(m),
                _ => None,
            })
            .collect();
        for needed in [
            MenuAction::Pick,
            MenuAction::BeforeAfter,
            MenuAction::Nodes,
            MenuAction::Fit,
            MenuAction::Zoom100,
            MenuAction::Crop,
            MenuAction::AutoCrop,
            MenuAction::PickWhite,
            MenuAction::RemovePhoto,
            MenuAction::BypassNode,
            MenuAction::MaskView,
            MenuAction::Look1,
            MenuAction::Look5,
            MenuAction::Look9,
        ] {
            assert!(actions.contains(&needed), "{needed:?}");
        }
        assert_eq!(
            actions
                .iter()
                .filter(|a| format!("{a:?}").starts_with("Look"))
                .count(),
            9
        );
    }

    #[test]
    fn the_settings_items_have_no_native_key_and_follow_the_state() {
        let by_id: std::collections::HashMap<_, _> = items()
            .into_iter()
            .map(|(id, label, shortcut, action)| (id, (label, shortcut, action)))
            .collect();
        // No key: ⇧⌘C, ⇧⌘V and ⌥⇧⌘V are the webview's, which leaves them to a text box.
        for (id, label, action) in [
            ("copy_settings", "Copy Settings", MenuAction::CopySettings),
            (
                "paste_settings",
                "Paste Settings",
                MenuAction::PasteSettings,
            ),
            (
                "paste_settings_to_picked",
                "Paste Settings to Picked",
                MenuAction::PasteSettingsToPicked,
            ),
        ] {
            assert_eq!(by_id[id], (label, None, Do::Webview(action)), "{id}");
            assert_eq!(action_for(id), Some(Do::Webview(action)));
            assert!(STATE_ITEMS.contains(&id));
        }
        // Plain ⌘C and ⌘V stay the system's Copy and Paste for the text boxes.
        let entries: Vec<_> = spec().into_iter().flat_map(|m| m.entries).collect();
        assert!(entries.contains(&Entry::Standard(Standard::Copy)));
        assert!(entries.contains(&Entry::Standard(Standard::Paste)));

        let none = MenuState {
            copy_settings: false,
            paste_settings: false,
            paste_settings_to_picked: false,
            save_preset: false,
            apply_preset: false,
            save: false,
            save_as: false,
        };
        assert!(availability(&none).iter().all(|(_, on)| !on));
        let state = MenuState {
            copy_settings: true,
            paste_settings: false,
            paste_settings_to_picked: true,
            save_preset: true,
            apply_preset: false,
            save: true,
            save_as: false,
        };
        assert_eq!(
            availability(&state),
            [
                ("copy_settings", true),
                ("paste_settings", false),
                ("paste_settings_to_picked", true),
                ("save_preset", true),
                ("apply_preset", false),
                ("save", true),
                ("save_as", false)
            ]
        );
    }

    #[test]
    fn save_and_save_as_are_in_the_file_menu_with_the_keys_of_docs_31() {
        let file = spec()
            .into_iter()
            .find(|m| m.title == "File")
            .expect("File");
        let labels: Vec<&str> = file
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::Item { label, .. } => Some(*label),
                _ => None,
            })
            .collect();
        assert_eq!(
            labels,
            [
                "Open Files…",
                "Open Folder…",
                "Save",
                "Save As…",
                "Export…",
                "Close Session"
            ]
        );
        let by_id: std::collections::HashMap<_, _> = items()
            .into_iter()
            .map(|(id, _, key, act)| (id, (key, act)))
            .collect();
        assert_eq!(
            by_id["save"],
            (Some("CmdOrCtrl+S"), Do::Webview(MenuAction::Save))
        );
        assert_eq!(
            by_id["save_as"],
            (Some("CmdOrCtrl+Shift+S"), Do::Webview(MenuAction::SaveAs))
        );
        assert!(STATE_ITEMS.contains(&"save") && STATE_ITEMS.contains(&"save_as"));
    }

    #[test]
    fn the_looks_menu_has_the_preset_items_after_the_nine_looks() {
        let looks = spec()
            .into_iter()
            .find(|m| m.title == "Looks")
            .expect("Looks");
        let after: Vec<&str> = looks
            .entries
            .iter()
            .skip_while(|e| !matches!(e, Entry::Item { id: "look_9", .. }))
            .filter_map(|e| match e {
                Entry::Item { label, .. } => Some(*label),
                Entry::PresetSubmenu { label, .. } => Some(*label),
                _ => None,
            })
            .collect();
        assert_eq!(
            after,
            [
                "Apply Look 9",
                "Save Preset…",
                "Import Presets…",
                "Apply Preset",
                "New Preset from Exported Photo…",
                "Show Presets Folder",
                "Show Looks Folder"
            ]
        );
        let by_id: std::collections::HashMap<_, _> = items()
            .into_iter()
            .map(|(id, _, key, act)| (id, (key, act)))
            .collect();
        // ⌥⌘S, not ⇧⌘S (which is Save As…).
        assert_eq!(
            by_id["save_preset"],
            (Some("CmdOrCtrl+Alt+S"), Do::Webview(MenuAction::SavePreset))
        );
        assert_eq!(by_id["show_presets_folder"], (None, Do::ShowPresetsFolder));
        assert!(STATE_ITEMS.contains(&"save_preset") && STATE_ITEMS.contains(&"apply_preset"));
    }

    #[test]
    fn apply_preset_lists_at_most_thirty_names_and_ticks_the_matching_ones() {
        use crate::wire::PresetEntry;
        let entry = |i: usize| PresetEntry {
            name: format!("Preset {i:02}"),
            groups: vec![],
            look: None,
            look_missing: false,
            matches: i == 2,
            revision: String::new(),
        };
        let list = PresetsListing {
            presets: (0..45).map(entry).collect(),
            damaged: vec![],
            folder: String::new(),
        };
        let items = preset_items(&list);
        assert_eq!(items.len(), MAX_PRESET_NAMES);
        assert_eq!(items[0].name, "Preset 00");
        assert!(items[2].checked && !items[1].checked);
        let empty = PresetsListing {
            presets: vec![],
            damaged: vec![],
            folder: String::new(),
        };
        assert!(preset_items(&empty).is_empty());
        // The item of a name carries the name; no item of ours has an id that could be mistaken for one.
        assert_eq!(
            preset_of_item(&preset_item_id("Warm evening")),
            Some("Warm evening")
        );
        assert_eq!(preset_of_item("apply_preset"), None);
        assert!(items_ids_never_start_with_the_prefix());
    }

    fn items_ids_never_start_with_the_prefix() -> bool {
        items().iter().all(|(id, ..)| preset_of_item(id).is_none())
    }

    #[test]
    fn the_nodes_item_is_in_the_view_menu_with_no_key() {
        // N is a plain letter that types, so it is the webview's, as P, Z, C and W are (`docs/11` §Menus).
        let view = spec()
            .into_iter()
            .find(|m| m.title == "View")
            .expect("View");
        let labels: Vec<&str> = view
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::Item { label, .. } => Some(*label),
                _ => None,
            })
            .collect();
        assert_eq!(labels[..2], ["Before/After", "Nodes"]);
        let nodes = items().into_iter().find(|i| i.0 == "nodes").expect("nodes");
        assert_eq!(
            nodes,
            ("nodes", "Nodes", None, Do::Webview(MenuAction::Nodes))
        );
        assert_eq!(action_for("nodes"), Some(Do::Webview(MenuAction::Nodes)));
        assert_eq!(
            serde_json::to_value(MenuAction::Nodes).expect("json"),
            "nodes"
        );
    }

    #[test]
    fn items_resolve_by_id() {
        assert_eq!(action_for("export"), Some(Do::Webview(MenuAction::Export)));
        assert_eq!(action_for("show_looks_folder"), Some(Do::ShowLooksFolder));
        assert_eq!(action_for("report_problem"), Some(Do::ReportProblem));
        assert_eq!(action_for("nonsense"), None);
    }

    #[test]
    fn labels_are_plain_sentences() {
        for (_, label, _, _) in items() {
            assert!(!label.contains('!'));
            assert!(!label.trim().is_empty());
        }
    }

    /// Builds the real menu with the mock runtime. GTK needs a display for menus, so this runs only
    /// where there is one (`xvfb-run cargo test` does). AppKit makes menus on the main thread only,
    /// and a test runs on a thread of its own, so on a Mac the bar is part of the manual checklist.
    #[test]
    fn the_system_menu_builds() {
        if cfg!(target_os = "macos") {
            eprintln!("skipped: AppKit menus are made on the main thread");
            return;
        }
        if cfg!(target_os = "linux")
            && std::env::var_os("DISPLAY").is_none()
            && std::env::var_os("WAYLAND_DISPLAY").is_none()
        {
            eprintln!("skipped: no display (run under xvfb-run)");
            return;
        }
        let app = tauri::test::mock_app();
        let menu = build(app.handle()).expect("menu");
        let titles: Vec<String> = menu.items().expect("items").iter().map(label_of).collect();
        assert_eq!(
            titles,
            [
                "Yellowmoon",
                "File",
                "Edit",
                "View",
                "Photo",
                "Looks",
                "Window",
                "Help"
            ]
        );
        let file = match &menu.items().expect("items")[1] {
            MenuItemKind::Submenu(s) => s.items().expect("file items"),
            _ => panic!("File is a submenu"),
        };
        let labels: Vec<String> = file.iter().map(label_of).collect();
        assert!(labels.contains(&"Open Files…".to_owned()), "{labels:?}");
        assert!(labels.contains(&"Close Session".to_owned()), "{labels:?}");

        // The three settings items start off and follow `apply_state`.
        let enabled = |menu: &Menu<tauri::test::MockRuntime>, id: &str| -> bool {
            for kind in menu.items().expect("items") {
                if let MenuItemKind::Submenu(sub) = kind {
                    match sub.get(id) {
                        Some(MenuItemKind::MenuItem(item)) => {
                            return item.is_enabled().expect("enabled");
                        }
                        Some(MenuItemKind::Submenu(item)) => {
                            return item.is_enabled().expect("enabled");
                        }
                        _ => {}
                    }
                }
            }
            panic!("no item {id}");
        };
        assert!(
            STATE_ITEMS.iter().all(|id| !enabled(&menu, id)),
            "they start off"
        );
        assert!(enabled(&menu, "export"), "the others start on");
        app.set_menu(menu.clone()).expect("set the menu");
        apply_state(
            app.handle(),
            &MenuState {
                copy_settings: true,
                paste_settings: false,
                paste_settings_to_picked: true,
                save_preset: true,
                apply_preset: true,
                save: false,
                save_as: false,
            },
        )
        .expect("apply the state");
        assert!(enabled(&menu, "copy_settings"));
        assert!(!enabled(&menu, "paste_settings"));
        assert!(enabled(&menu, "paste_settings_to_picked"));
        assert!(enabled(&menu, "save_preset"));
        assert!(!enabled(&menu, "save"));
        // Apply Preset is a submenu, greyed and switched on as a whole; empty it holds "No Presets", greyed.
        let apply = preset_submenu(app.handle()).expect("Apply Preset");
        assert!(enabled(&menu, "apply_preset"));
        let names: Vec<String> = apply.items().expect("items").iter().map(label_of).collect();
        assert_eq!(names, ["No Presets"]);
        let items = [
            PresetItem {
                name: "Cool".into(),
                checked: false,
            },
            PresetItem {
                name: "Warm evening".into(),
                checked: true,
            },
        ];
        set_preset_items(app.handle(), &apply, &items).expect("rebuild");
        let names: Vec<String> = apply.items().expect("items").iter().map(label_of).collect();
        assert_eq!(names, ["Cool", "Warm evening"]);
        match apply.get("apply_preset:Warm evening") {
            Some(MenuItemKind::Check(c)) => assert!(c.is_checked().expect("checked")),
            _ => panic!("a check item"),
        }
        assert!(
            enabled(&menu, "export"),
            "the state leaves other items alone"
        );
    }
}
