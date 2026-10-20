//! `src/api/bindings.ts`, generated from the commands and types of this crate by `tauri-specta`
//! (`docs/01` §IPC: "so the two sides can't drift"), and the check that the committed file is the
//! one the code makes.
//!
//! The frontend is written against `src/api/types.ts` and `src/api.ts`; the generated file is the
//! Rust side's statement of the same contract. When a command or a type changes, regenerate it:
//!
//! ```text
//! YM_UPDATE_BINDINGS=1 cargo test -p yellowmoon-app --bin yellowmoon-app bindings
//! ```
//!
//! CI runs the same test without the variable and fails when the committed file differs.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use specta_typescript::Typescript;

    use crate::commands::builder;

    /// Where the bindings are committed.
    fn committed_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/api/bindings.ts")
    }

    /// The bindings as the code makes them, in memory.
    fn generated() -> String {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bindings.ts");
        builder()
            .export(Typescript::default(), &path)
            .expect("export the TypeScript bindings");
        std::fs::read_to_string(&path).expect("read the generated bindings")
    }

    #[test]
    fn the_bindings_are_up_to_date() {
        let fresh = generated();
        let path = committed_path();
        if std::env::var_os("YM_UPDATE_BINDINGS").is_some() {
            std::fs::write(&path, &fresh).expect("write src/api/bindings.ts");
            return;
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            committed == fresh,
            "src/api/bindings.ts is out of date. Regenerate it with:\n  YM_UPDATE_BINDINGS=1 cargo test -p yellowmoon-app --bin yellowmoon-app bindings\nand commit the result."
        );
    }

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(generated(), generated());
    }

    #[test]
    fn every_event_of_the_frontend_is_in_the_bindings() {
        // Commands are checked at compile time: `tauriApi` in `src/api.ts` calls the generated
        // `commands`, and `src/api/contract.ts` makes `tsc -b` fail when a payload, an event or a
        // command argument drifts. Events are subscribed by name, so check those here.
        let api =
            std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/api.ts"))
                .expect("api.ts");
        assert!(
            api.contains("commands."),
            "src/api.ts no longer calls the generated `commands`; restore the command check"
        );
        let bindings = generated();
        for event in [
            "photo_progress",
            "thumbnail_ready",
            "render_stats",
            "export_progress",
            "looks_changed",
            "update_state",
            "error",
            "looks_import_progress",
            "download_found",
            "menu_action",
            "viewport_frame",
            "settings_changed",
            "export_busy",
        ] {
            assert!(
                bindings.contains(&format!("\"{event}\"")),
                "the event `{event}` of `ApiEvents` is not registered"
            );
        }
    }

    // --- The contract: `src/api/types.ts` ------------------------------------------------------

    use std::collections::{BTreeMap, BTreeSet};

    use regex::Regex;

    /// The fields of every `export interface` of `types.ts`.
    fn ts_interfaces(types_ts: &str) -> BTreeMap<String, BTreeSet<String>> {
        let block = Regex::new(r"(?ms)^export interface (\w+)[^{]*\{\n(.*?)^\}").expect("regex");
        let field = Regex::new(r"(?m)^  (\w+)\??:").expect("regex");
        block
            .captures_iter(types_ts)
            .map(|c| {
                let fields = field
                    .captures_iter(&c[2])
                    .map(|f| f[1].to_owned())
                    .collect();
                (c[1].to_owned(), fields)
            })
            .collect()
    }

    /// The fields of every object type of the generated bindings.
    fn generated_structs(bindings: &str) -> BTreeMap<String, BTreeSet<String>> {
        let block = Regex::new(r"(?ms)^export type (\w+) = \{\n(.*?)^\};").expect("regex");
        let field = Regex::new(r"(?m)^\t(\w+)\??:").expect("regex");
        block
            .captures_iter(bindings)
            .map(|c| {
                let fields = field
                    .captures_iter(&c[2])
                    .map(|f| f[1].to_owned())
                    .collect();
                (c[1].to_owned(), fields)
            })
            .collect()
    }

    /// The tags (`kind: "x"`, `phase: "x"`, `state: "x"`) of a union, or its string literals, by
    /// type name.
    fn tags(text: &str) -> BTreeMap<String, BTreeSet<String>> {
        let comments = Regex::new(r"(?s)/\*.*?\*/").expect("regex");
        let text = comments.replace_all(text, "");
        let alias = Regex::new(r"(?ms)^export type (\w+) =(.*?);\n(?:\n|\z)").expect("regex");
        let tag = Regex::new(r#"(?:kind|phase|state): "(\w+)""#).expect("regex");
        let literal = Regex::new(r#""(\w+)""#).expect("regex");
        let only_literals = Regex::new(r#"^\s*(?:"\w+"\s*\|?\s*)+$"#).expect("regex");
        let mut out = BTreeMap::new();
        for c in alias.captures_iter(&text) {
            let body = &c[2];
            let set: BTreeSet<String> = if ["kind:", "phase:", "state:"]
                .iter()
                .any(|t| body.contains(t))
            {
                tag.captures_iter(body).map(|t| t[1].to_owned()).collect()
            } else if only_literals.is_match(body) {
                literal
                    .captures_iter(body)
                    .map(|t| t[1].to_owned())
                    .collect()
            } else {
                continue;
            };
            out.insert(c[1].to_owned(), set);
        }
        out
    }

    fn read(relative: &str) -> String {
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative))
            .unwrap_or_else(|e| panic!("{relative}: {e}"))
    }

    /// Interfaces of `types.ts` that are not structs on the bridge, and why.
    const NOT_STRUCTS: [&str; 1] = [
        // The map from event names to payloads: the bindings have one `events` object instead.
        "ApiEvents",
    ];

    /// Unions whose members differ on purpose (see the module documentation of `wire.rs`).
    const UNIONS_THAT_DIFFER: [&str; 0] = [];

    #[test]
    fn field_names_match_the_typescript_contract() {
        let contract = ts_interfaces(&read("../src/api/types.ts"));
        let generated = generated_structs(&generated());
        let mut compared = 0;
        for (name, fields) in &contract {
            if NOT_STRUCTS.contains(&name.as_str()) {
                continue;
            }
            let ours = generated
                .get(name)
                .unwrap_or_else(|| panic!("`{name}` of types.ts is not a type of the bridge"));
            assert_eq!(ours, fields, "the fields of `{name}` differ from types.ts");
            compared += 1;
        }
        assert!(compared >= 30, "only {compared} interfaces compared");
    }

    #[test]
    fn variants_match_the_typescript_contract() {
        let contract = tags(&read("../src/api/types.ts"));
        let generated = tags(&generated());
        let mut compared = 0;
        for (name, variants) in &contract {
            if UNIONS_THAT_DIFFER.contains(&name.as_str()) {
                continue;
            }
            // `Orientation` and `BitDepth` are number unions; `UpdatePhase` is derived.
            let Some(ours) = generated.get(name) else {
                continue;
            };
            assert_eq!(
                ours, variants,
                "the variants of `{name}` differ from types.ts"
            );
            compared += 1;
        }
        assert!(
            compared >= 20,
            "only {compared} unions compared: {:?}",
            contract.keys()
        );
        // The ones that matter most are among them.
        for name in [
            "PhotoStatus",
            "Recognition",
            "ViewportInput",
            "MenuAction",
            "UpdateState",
            "ErrorScope",
            "AppleAuto",
            "AppleAutoReason",
            "PercentSetting",
        ] {
            assert!(generated.contains_key(name), "{name} is not on the bridge");
        }
    }

    /// The sentences the Export panel shows that the Rust side owns are the same text on both
    /// sides: the webview keeps a copy (it has no way to ask for them before the panel is drawn,
    /// and the Preferences form shows them without a job), and this fails when a copy drifts.
    #[test]
    fn the_export_panel_says_what_the_writer_and_the_cli_say() {
        let form = read("../src/screens/export/ExportOptionsForm.tsx");
        let constants = [
            // The one sentence about other apps: `ym render --help` has it word for word.
            ("OTHER_APPS_NOTE", ym_export::dng::OTHER_APPS_NOTE),
            // Adobe's licence notice.
            ("ADOBE_NOTICE", ym_export::dng::ADOBE_NOTICE),
            // What a DNG is, with a look that has its own display rendering (shown when the job has one).
            ("OWN_DISPLAY_NOTE", ym_export::dng::OWN_DISPLAY_NOTE),
        ];
        for (name, text) in constants {
            let declaration = Regex::new(&format!(
                r#"export const {name}\s*=\s*"{}";"#,
                regex::escape(text)
            ))
            .expect("regex");
            assert!(
                declaration.is_match(&form),
                "`{name}` in src/screens/export/ExportOptionsForm.tsx is not the Rust text:\n{text}"
            );
        }
    }

    #[test]
    fn numeric_unions_are_the_literal_types_of_the_contract() {
        let bindings = generated();
        assert!(
            bindings.contains("orientation: 0 | 90 | 180 | 270"),
            "Orientation"
        );
        assert!(bindings.contains("bit_depth: 8 | 16"), "BitDepth");
        // Floats are numbers, not `number | null`.
        assert!(!bindings.contains("number | null,\n\t/**  The Auto tint"));
        assert!(bindings.contains("exposure_ev: number,"));
    }
}
