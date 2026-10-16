//! The menu bar of `docs/11-ui.md` §Menus.
//!
//! [`spec`] is the menu as plain data (titles, labels, shortcuts, what each item does), so the
//! structure is tested without a window; [`build`] turns it into the system's menu. An item either
//! asks the webview to act (`menu_action { action }`, handled by `src/state/menu.ts`) or is the
//! shell's own business: Show Looks Folder, the Help links, Report a Problem, About, Quit and the
//! standard window items.
//!
//! Shortcuts are given to the items whose keys do not also type: ⌘, ⌘O ⇧⌘O ⌘E ⌘W ⌘= ⌘- ⌘Q and the
//! three settings items ⇧⌘C ⇧⌘V ⌥⇧⌘V (no text box uses them: the system's Copy and Paste are plain
//! ⌘C and ⌘V). ⌘Z, ⇧⌘Z and ⌘A stay with the webview, which sends them to a text box when one has
//! focus and to the photos otherwise; a menu item bound to them would take the keys away from the
//! text boxes. The menu shows those items without a key. The single-key shortcuts (P, Z, C, W, \ and
//! the arrows) are not menu shortcuts for the same reason.
//!
//! Three items change with what is open: Copy Settings needs a developed current photo, Paste
//! Settings something copied as well, and Paste Settings to Picked something copied and a picked
//! photo. The webview says which are available (`set_menu_state`, [`apply_state`]); the menu starts
//! with all three off, as Start has nothing to copy.

use tauri::menu::{AboutMetadata, Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Runtime};

use crate::about;
use crate::wire::{MenuAction, MenuState};

/// What an item of the menu does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Do {
    /// Ask the webview (`menu_action`).
    Webview(MenuAction),
    /// Open the folder of imported looks in Finder.
    ShowLooksFolder,
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
                    Some("CmdOrCtrl+Shift+C"),
                    MenuAction::CopySettings,
                ),
                web(
                    "paste_settings",
                    "Paste Settings",
                    Some("CmdOrCtrl+Shift+V"),
                    MenuAction::PasteSettings,
                ),
                web(
                    "paste_settings_to_picked",
                    "Paste Settings to Picked",
                    Some("CmdOrCtrl+Alt+Shift+V"),
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
            title: "Looks",
            entries: vec![
                web(
                    "import_arri_looks",
                    "Import ARRI Looks…",
                    None,
                    MenuAction::ImportArriLooks,
                ),
                web("import_cube", "Import .cube…", None, MenuAction::ImportCube),
                web(
                    "make_default_look",
                    "Make Current Look Default",
                    None,
                    MenuAction::MakeDefaultLook,
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
pub const STATE_ITEMS: [&str; 3] = [
    "copy_settings",
    "paste_settings",
    "paste_settings_to_picked",
];

/// `(id, available)` for each of [`STATE_ITEMS`] under `state`.
#[must_use]
pub fn availability(state: &MenuState) -> [(&'static str, bool); 3] {
    [
        (STATE_ITEMS[0], state.copy_settings),
        (STATE_ITEMS[1], state.paste_settings),
        (STATE_ITEMS[2], state.paste_settings_to_picked),
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
            if let Some(MenuItemKind::MenuItem(item)) = sub.get(id) {
                item.set_enabled(on)?;
            }
        }
    }
    Ok(())
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
            "Fit",
            "100%",
            "Zoom In",
            "Zoom Out",
            "Next Photo",
            "Previous Photo",
            // Looks
            "Import ARRI Looks…",
            "Import .cube…",
            "Make Current Look Default",
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
                    "CmdOrCtrl+V"
                ]
                .contains(&s),
                "{s} belongs to the text boxes"
            );
        }
        // ⌘, ⌘O ⇧⌘O ⌘E ⌘W ⌘= ⌘- and the settings items ⇧⌘C ⇧⌘V ⌥⇧⌘V
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn the_settings_items_have_their_keys_and_follow_the_state() {
        let by_id: std::collections::HashMap<_, _> = items()
            .into_iter()
            .map(|(id, label, shortcut, action)| (id, (label, shortcut, action)))
            .collect();
        for (id, label, shortcut, action) in [
            (
                "copy_settings",
                "Copy Settings",
                "CmdOrCtrl+Shift+C",
                MenuAction::CopySettings,
            ),
            (
                "paste_settings",
                "Paste Settings",
                "CmdOrCtrl+Shift+V",
                MenuAction::PasteSettings,
            ),
            (
                "paste_settings_to_picked",
                "Paste Settings to Picked",
                "CmdOrCtrl+Alt+Shift+V",
                MenuAction::PasteSettingsToPicked,
            ),
        ] {
            assert_eq!(
                by_id[id],
                (label, Some(shortcut), Do::Webview(action)),
                "{id}"
            );
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
        };
        assert!(availability(&none).iter().all(|(_, on)| !on));
        let state = MenuState {
            copy_settings: true,
            paste_settings: false,
            paste_settings_to_picked: true,
        };
        assert_eq!(
            availability(&state),
            [
                ("copy_settings", true),
                ("paste_settings", false),
                ("paste_settings_to_picked", true)
            ]
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
                if let MenuItemKind::Submenu(sub) = kind
                    && let Some(MenuItemKind::MenuItem(item)) = sub.get(id)
                {
                    return item.is_enabled().expect("enabled");
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
            },
        )
        .expect("apply the state");
        assert!(enabled(&menu, "copy_settings"));
        assert!(!enabled(&menu, "paste_settings"));
        assert!(enabled(&menu, "paste_settings_to_picked"));
        assert!(
            enabled(&menu, "export"),
            "the state leaves other items alone"
        );
    }
}
