//! `ym presets …`: the saved presets without the UI (`docs/30-presets.md` section 11, `docs/10-cli.md`).
//!
//! The commands use the same folder, file format, names, lock and Trash as the app's Presets card
//! (`ym_core::presets`), so a preset saved here shows up in the app and the other way round. `render --preset`
//! is in [`crate::render`]; what it shares with these commands is [`preset_failure`].

use std::io::{self, Write};
use std::sync::Arc;

use serde_json::json;
use ym_core::data_dir::find_data_dir;
use ym_core::looks::LookService;
use ym_core::presets::{
    PresetError, PresetGroup, PresetService, PresetSource, PresetStore,
};
use ym_core::session::{Session, SessionEnv};

use crate::cli::{
    PresetsCommand, PresetsExportArgs, PresetsImportArgs, PresetsListArgs, PresetsRemoveArgs,
    PresetsSaveArgs,
};
use crate::console::{Console, data_line};
use crate::exit::{self, Failure};
use crate::looks_cmd::{open_service, table};
use crate::text::{json_line, line_text, path_text};

/// Runs a `ym presets` command and returns the exit code.
pub fn run(command: &PresetsCommand, console: &mut Console) -> u8 {
    let result = match command {
        PresetsCommand::List(args) => list(args),
        PresetsCommand::Save(args) => save(args, console),
        PresetsCommand::Import(args) => import(args, console),
        PresetsCommand::Export(args) => export(args, console),
        PresetsCommand::Remove(args) => remove(args, console),
    };
    match result {
        Ok(code) => code,
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// The failure a preset error stands for: the core's exit code (`PresetError::exit_code`: 3 for a preset or a path that is
/// missing, 2 for a name that exists, several matches and a file that is refused, 1 otherwise) and its sentence.
pub fn preset_failure(error: &PresetError) -> Failure {
    let code = u8::try_from(error.exit_code()).unwrap_or(exit::SOME_FAILED);
    Failure::new(code, error.to_string())
}

/// The presets folder of the application folder (`YELLOWMOON_HOME` replaces it), and the look library beside it.
fn open_store() -> Result<PresetStore, Failure> {
    let app_root = ym_looks::library::default_root().map_err(|e| Failure::usage(e.to_string()))?;
    Ok(PresetStore::open(&app_root))
}

/// The service the commands that make a preset need: the store, the look library and a session that holds no photo (a
/// preset is made from a file here).
fn open_presets() -> Result<PresetService, Failure> {
    let app_root = ym_looks::library::default_root().map_err(|e| Failure::usage(e.to_string()))?;
    let looks: Arc<LookService> = Arc::new(open_service()?);
    let data_root = find_data_dir().map_err(|e| Failure::usage(e.to_string()))?;
    let env = SessionEnv::load(&data_root)
        .map_err(|e| Failure::usage(e.to_string()))?
        .with_default_look(looks.effective_default());
    let session = Arc::new(Session::new(env));
    Ok(PresetService::open(&app_root, looks, session))
}

fn groups_text(groups: &[PresetGroup]) -> String {
    groups
        .iter()
        .map(|g| g.id())
        .collect::<Vec<_>>()
        .join(",")
}

/// Writes `text` to stdout; a reader that went away is not an error.
fn write_out(text: &str) -> Result<u8, Failure> {
    let mut out = io::stdout().lock();
    match out.write_all(text.as_bytes()).and_then(|()| out.flush()) {
        Ok(()) => Ok(exit::OK),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(exit::OK),
        Err(e) => Err(Failure::failed(format!("Couldn't write the list ({e})."))),
    }
}

fn list(args: &PresetsListArgs) -> Result<u8, Failure> {
    let store = open_store()?;
    let looks = open_service()?;
    let listing = store.list();
    let usable = |p: &ym_core::presets::Preset| {
        p.look
            .as_ref()
            .is_none_or(|l| looks.check_usable(&l.id).is_ok())
    };
    if args.json {
        let presets: Vec<_> = listing
            .presets
            .iter()
            .map(|s| {
                json!({
                    "name": s.preset.name,
                    "file": s.file,
                    "groups": s.preset.groups.iter().map(|g| g.id()).collect::<Vec<_>>(),
                    "look": s.preset.look.as_ref().map(|l| json!({ "id": l.id.to_string(), "name": l.name })),
                    "look_missing": !usable(&s.preset),
                })
            })
            .collect();
        let damaged: Vec<_> = listing
            .damaged
            .iter()
            .map(|d| json!({ "file": d.file, "reason": d.reason }))
            .collect();
        let value = json!({
            "folder": path_text(store.folder()),
            "presets": presets,
            "damaged": damaged,
        });
        return write_out(&format!("{}\n", json_line(&value)));
    }
    let mut rows = vec![vec![
        "NAME".to_owned(),
        "GROUPS".to_owned(),
        "LOOK".to_owned(),
    ]];
    for s in &listing.presets {
        let look = match &s.preset.look {
            None => "-".to_owned(),
            Some(l) if usable(&s.preset) => line_text(&l.name),
            Some(l) => format!("{} (not imported)", line_text(&l.name)),
        };
        rows.push(vec![
            line_text(&s.preset.name),
            groups_text(&s.preset.groups),
            look,
        ]);
    }
    let mut text = table(&rows);
    for d in &listing.damaged {
        text.push_str(&format!("{}: {}\n", line_text(&d.file), line_text(&d.reason)));
    }
    write_out(&text)
}

/// "A preset called Warm evening already exists. Use --replace to write over it."
fn exists_failure(error: &PresetError) -> Failure {
    let mut failure = preset_failure(error);
    if matches!(error, PresetError::Exists(_)) {
        failure.message.push_str(" Use --replace to write over it.");
    }
    failure
}

fn save(args: &PresetsSaveArgs, console: &mut Console) -> Result<u8, Failure> {
    if !args.from.is_file() {
        return Err(Failure::usage(format!(
            "{}: There is no file with this name. Check the path and try again.",
            path_text(&args.from)
        )));
    }
    let service = open_presets()?;
    let source = PresetSource::File {
        path: args.from.clone(),
    };
    let groups = if args.groups.is_empty() {
        let inspect = service.inspect(&source).map_err(|e| preset_failure(&e))?;
        if inspect.changed.is_empty() {
            return Err(Failure::usage(
                "Nothing in this file differs from the default settings. Choose the groups with --groups.",
            ));
        }
        if inspect.older_export {
            console.warn(
                "This photo was exported by an older Yellowmoon, so its automatic values are saved as fixed numbers. Choose the groups with --groups to leave Light, Colour and Detail out.",
            );
        }
        inspect.changed
    } else {
        args.groups.clone()
    };
    let entry = service
        .save(&args.name, &groups, &source, args.replace)
        .map_err(|e| exists_failure(&e))?;
    console.info(&format!(
        "Saved the preset {} ({}).",
        line_text(&entry.name),
        groups_text(&entry.groups)
    ));
    data_line(&line_text(&entry.name));
    Ok(exit::OK)
}

fn import(args: &PresetsImportArgs, console: &mut Console) -> Result<u8, Failure> {
    let service = open_presets()?;
    let mut code = exit::OK;
    for file in &args.files {
        let outcome = service.import(std::slice::from_ref(file), args.replace);
        for name in outcome.added.iter().chain(&outcome.replaced) {
            console.info(&format!(
                "{} the preset {}.",
                if outcome.replaced.contains(name) {
                    "Replaced"
                } else {
                    "Added"
                },
                line_text(name)
            ));
            data_line(&line_text(name));
        }
        for exists in &outcome.exists {
            console.error(&format!(
                "{}: A preset called {} already exists. Use --replace to write over it.",
                path_text(file),
                line_text(&exists.name)
            ));
            code = code.max(exit::USAGE);
        }
        for refused in &outcome.refused {
            console.error(&format!(
                "{}: {}",
                path_text(file),
                line_text(&refused.reason)
            ));
            // A file the core refuses is a usage error; one that could not be read or written is a failure.
            code = code.max(if refused.reason.starts_with("Yellowmoon couldn't") {
                exit::SOME_FAILED
            } else {
                exit::USAGE
            });
        }
    }
    Ok(code)
}

fn export(args: &PresetsExportArgs, console: &mut Console) -> Result<u8, Failure> {
    let store = open_store()?;
    store
        .export(&args.name, &args.file)
        .map_err(|e| preset_failure(&e))?;
    console.info(&format!(
        "Exported the preset {} to {}.",
        line_text(&args.name),
        path_text(&args.file)
    ));
    Ok(exit::OK)
}

fn remove(args: &PresetsRemoveArgs, console: &mut Console) -> Result<u8, Failure> {
    let store = open_store()?;
    let name = store
        .get(&args.name)
        .map(|p| p.name)
        .map_err(|e| preset_failure(&e))?;
    store.remove(&name).map_err(|e| preset_failure(&e))?;
    console.info(&format!(
        "Moved the preset {} to the Trash.",
        line_text(&name)
    ));
    Ok(exit::OK)
}
