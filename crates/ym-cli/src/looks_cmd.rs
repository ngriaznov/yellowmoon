//! `ym looks …`: the look library without the UI (`docs/10-cli.md`, `docs/08-looks.md`).
//!
//! `list`, `import`, `verify` and `default` use the same library folder, recognition rules,
//! checksums and lock as the app's wizard (`ym_core::looks::LookService`), so a look imported here
//! shows up in the app and the other way round. `bake` is the developer tool for the Signature look.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use ym_color::look::{LookInput, ParametricLook};
use ym_core::data_dir::find_data_dir;
use ym_core::looks::{ImportRequest, LookEntry, LookGroupInfo, LookService, LooksError};
use ym_core::pipeline::color::{LookProblem, LookProvider, NeutralOnly, ResolvedLook};
use ym_core::pipeline::thumbs::encode_png_rgb8;
use ym_core::pipeline::{RenderOptions, Renderer};
use ym_core::preferences::PreferencesStore;
use ym_core::session::{OpenOptions, PhotoId, PhotoStatus, Session, SessionEnv};
use ym_core::settings::DevelopSettings;
use ym_gpu::queue::CancelToken as RenderCancel;
use ym_looks::import::{CancelToken, ImportError, Progress, Stage};
use ym_looks::library::{LookId, VerifyOutcome};

use crate::cli::{
    InputArg, LooksBakeArgs, LooksCommand, LooksDefaultArgs, LooksImportArgs, LooksListArgs,
};
use crate::console::{Console, bar, data_line, data_path_line, photos};
use crate::exit::{self, Failure};
use crate::render::{look_failure, resolve_look};
use crate::runtime::{handle_interrupts, interrupted};
use crate::text::{json_line, path_text};

/// Runs a `ym looks` command and returns the exit code.
pub fn run(command: &LooksCommand, console: &mut Console) -> u8 {
    let result = match command {
        LooksCommand::List(args) => list(args),
        LooksCommand::Import(args) => import(args, console),
        LooksCommand::Verify => verify(console),
        LooksCommand::Default(args) => set_default(args, console),
        LooksCommand::Bake(args) => bake(args, console),
    };
    match result {
        Ok(code) => code,
        Err(failure) => {
            console.failure(&failure);
            failure.code
        }
    }
}

/// Opens the look library the way the app does: the shipped built-in looks, the user's imports
/// and the preferences, in the app folder (`YELLOWMOON_HOME` replaces it).
fn open_service() -> Result<LookService, Failure> {
    let data_root = find_data_dir().map_err(|e| Failure::usage(e.to_string()))?;
    let app_root = ym_looks::library::default_root().map_err(|e| Failure::usage(e.to_string()))?;
    let prefs = Arc::new(PreferencesStore::open(&app_root));
    LookService::open(&app_root, &data_root, prefs).map_err(|e| look_failure(&e))
}

fn input_word(input: LookInput) -> &'static str {
    match input {
        LookInput::LogC4 => "logc4",
        LookInput::LogC3Ei800 => "logc3",
        LookInput::LogC3ToRec709 => "logc3-rec709",
    }
}

fn group_text(group: &LookGroupInfo) -> String {
    match group {
        LookGroupInfo::BuiltIn => "built in".to_owned(),
        LookGroupInfo::Arri { no, name } => format!("ARRI {no} {name}"),
        LookGroupInfo::User => "imported".to_owned(),
    }
}

/// Left-aligned columns separated by two spaces; the last column is not padded.
fn table(rows: &[Vec<String>]) -> String {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..columns)
        .map(|c| {
            rows.iter()
                .filter_map(|r| r.get(c))
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut out = String::new();
    for row in rows {
        let mut line = String::new();
        for (c, cell) in row.iter().enumerate() {
            if c + 1 == row.len() {
                line.push_str(cell);
            } else {
                line.push_str(&format!("{cell:<width$}  ", width = widths[c]));
            }
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

/// The text of `ym looks list`.
fn list_text(looks: &[LookEntry], luts: &[ym_core::looks::DisplayLutEntry]) -> String {
    let mut rows = vec![vec![
        "ID".to_owned(),
        "NAME".to_owned(),
        "GROUP".to_owned(),
        "INPUT".to_owned(),
        String::new(),
    ]];
    for l in looks {
        let mut flags = Vec::new();
        if l.is_default {
            flags.push("default");
        }
        if l.missing {
            flags.push("missing: import again");
        }
        rows.push(vec![
            l.id.to_string(),
            l.name.clone(),
            group_text(&l.group),
            input_word(l.input).to_owned(),
            flags.join(", "),
        ]);
    }
    let mut text = table(&rows);
    if !luts.is_empty() {
        text.push_str("\nDisplay LUTs\n");
        let mut rows = vec![vec!["ID".to_owned(), "NAME".to_owned(), "INPUT".to_owned()]];
        for d in luts {
            rows.push(vec![
                d.id.to_string(),
                d.name.clone(),
                input_word(d.input).to_owned(),
            ]);
        }
        text.push_str(&table(&rows));
    }
    text
}

fn list(args: &LooksListArgs) -> Result<u8, Failure> {
    let service = open_service()?;
    let looks = service.list();
    let luts = service.display_luts();
    let mut out = io::stdout().lock();
    let written = if args.json {
        let value = json!({ "looks": looks, "display_luts": luts });
        writeln!(out, "{}", json_line(&value))
    } else {
        write!(out, "{}", list_text(&looks, &luts))
    };
    match written.and_then(|()| out.flush()) {
        Ok(()) => Ok(exit::OK),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(exit::OK),
        Err(e) => Err(Failure::failed(format!("Couldn't write the list ({e})."))),
    }
}

fn input_of(arg: InputArg) -> LookInput {
    match arg {
        InputArg::LogC4 => LookInput::LogC4,
        InputArg::LogC3 => LookInput::LogC3Ei800,
        InputArg::LogC3Rec709 => LookInput::LogC3ToRec709,
    }
}

/// The failure an import error stands for, with the way out for the CLI when the question about
/// the input encoding was not answered.
fn import_failure(error: &LooksError) -> Failure {
    if let LooksError::Import(ImportError::NeedsInput { files }) = error {
        return Failure::usage(format!(
            "These .cube files are not part of an ARRI package, so say what they expect as input: {}. Add --input logc4, --input logc3 or --input logc3-rec709.",
            files.join(", ")
        ));
    }
    look_failure(error)
}

fn import(args: &LooksImportArgs, console: &mut Console) -> Result<u8, Failure> {
    let service = open_service()?;
    // From here Ctrl-C stops the import between files, through the importer's cancel token.
    handle_interrupts();
    let request = ImportRequest {
        input: args.input.map(input_of),
        cancel: Some(CancelToken::new()),
    };
    let mut first_failure: Option<Failure> = None;
    let mut added = 0usize;
    let mut luts = 0usize;
    let mut unchanged = 0usize;
    for path in &args.paths {
        let cancel = request.cancel.clone();
        let result = service.import(path, &request, &mut |p: Progress<'_>| {
            if interrupted()
                && let Some(c) = &cancel
            {
                c.cancel();
            }
            let what = match p.stage {
                Stage::Scan => "Reading",
                Stage::Copy => "Copying",
                Stage::Verify => "Checking",
            };
            console.progress(format!(
                "{what} {} {}/{}",
                bar(p.done as f64 / p.total.max(1) as f64, 20),
                p.done,
                p.total
            ));
        });
        console.finish_progress();
        match result {
            Ok(r) => {
                added += r.added.len();
                luts += r.display_luts_added;
                unchanged += r.unchanged;
                for look in &r.added {
                    data_line(&look.id.to_string());
                }
                console.detail(&format!(
                    "{}: {} looks, {} display LUTs, {} unchanged.",
                    path_text(&path),
                    r.added.len(),
                    r.display_luts_added,
                    r.unchanged
                ));
            }
            Err(e) => {
                console.error(&format!(
                    "{}: {}",
                    path_text(&path),
                    import_failure(&e).message
                ));
                first_failure.get_or_insert_with(|| import_failure(&e));
                if interrupted() {
                    return Ok(exit::INTERRUPTED);
                }
            }
        }
    }
    let mut parts = vec![format!(
        "Imported {} {}",
        added,
        if added == 1 { "look" } else { "looks" }
    )];
    if luts > 0 {
        parts.push(format!(
            "{luts} display {}",
            if luts == 1 { "LUT" } else { "LUTs" }
        ));
    }
    if unchanged > 0 {
        parts.push(format!("{unchanged} already imported"));
    }
    if first_failure.is_none() || added + luts + unchanged > 0 {
        console.info(&format!("{}.", parts.join(", ")));
    }
    Ok(first_failure.map_or(exit::OK, |f| f.code))
}

fn verify(console: &mut Console) -> Result<u8, Failure> {
    let service = open_service()?;
    let report = service.verify().map_err(|e| look_failure(&e))?;
    let mut problems = 0usize;
    for entry in &report.entries {
        let what = match &entry.outcome {
            VerifyOutcome::Ok => continue,
            VerifyOutcome::Missing => "missing".to_owned(),
            VerifyOutcome::Changed { .. } => "changed since it was imported".to_owned(),
            VerifyOutcome::Unreadable { reason } => format!("can't be read ({reason})"),
        };
        problems += 1;
        data_line(&format!("{}: {what} ({})", entry.id, entry.file));
    }
    if problems == 0 {
        console.info(&format!(
            "Checked {}: all fine.",
            plural(report.checked(), "imported file")
        ));
        Ok(exit::OK)
    } else {
        console.error(&format!(
            "{} of {} need attention. Import them again with ym looks import.",
            problems,
            plural(report.checked(), "imported file")
        ));
        Ok(exit::SOME_FAILED)
    }
}

fn plural(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

fn set_default(args: &LooksDefaultArgs, console: &mut Console) -> Result<u8, Failure> {
    let service = open_service()?;
    let id = resolve_look(&service, &args.look)?;
    service.set_default(&id).map_err(|e| look_failure(&e))?;
    console.info(&format!("New photos now start with {id}."));
    data_line(&id.to_string());
    Ok(exit::OK)
}

// ---------------------------------------------------------------------------------------------
// bake
// ---------------------------------------------------------------------------------------------

/// The size of the baked LUT (`docs/08`: the loader bakes the same 33 points).
const BAKE_POINTS: usize = 33;
/// The long edge of a contact sheet cell in pixels.
const CELL_PX: usize = 480;
/// Space between cells in pixels.
const GAP_PX: usize = 12;

/// Offers the baked look under one id and Neutral for everything else.
struct BakeProvider {
    id: LookId,
    baked: ResolvedLook,
}

impl LookProvider for BakeProvider {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        if *id == self.id {
            Ok(self.baked.clone())
        } else {
            NeutralOnly.look(id)
        }
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

fn bake(args: &LooksBakeArgs, console: &mut Console) -> Result<u8, Failure> {
    let text = std::fs::read_to_string(&args.look).map_err(|e| {
        Failure::usage(format!(
            "Couldn't read the look file {}: {e}.",
            path_text(&args.look)
        ))
    })?;
    let look = ParametricLook::from_json(&text)
        .map_err(|e| Failure::usage(format!("{}: {e}.", path_text(&args.look))))?;
    let out_dir = args.out.clone().unwrap_or_else(|| {
        args.look
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    });
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| Failure::failed(format!("Couldn't create {}: {e}.", path_text(&out_dir))))?;
    let stem = args
        .look
        .file_stem()
        .map_or_else(|| "look".to_owned(), |s| s.to_string_lossy().into_owned());

    let cube = look
        .bake(BAKE_POINTS)
        .map_err(|e| Failure::failed(format!("The look can't be baked: {e}.")))?;
    let cube_path = out_dir.join(format!("{stem}.cube"));
    std::fs::write(&cube_path, cube.to_cube_string(look.name()))
        .map_err(|e| Failure::failed(format!("Couldn't write {}: {e}.", path_text(&cube_path))))?;
    console.info(&format!(
        "Baked {} to a {BAKE_POINTS}-point LUT: {}",
        look.name(),
        path_text(&cube_path)
    ));
    data_path_line(&cube_path);

    if args.preview.is_empty() {
        return Ok(exit::OK);
    }
    let sheet_path = out_dir.join(format!("{stem}_contact.png"));
    let used = contact_sheet(&look, &args.preview, &sheet_path, console)?;
    console.info(&format!(
        "Contact sheet of {}: Neutral on the left, {} on the right: {}",
        photos(used),
        look.name(),
        path_text(&sheet_path)
    ));
    data_path_line(&sheet_path);
    Ok(exit::OK)
}

/// Renders every preview photo twice, with Neutral and with the baked look, and lays the pictures
/// out in rows. Returns how many photos made it into the sheet.
fn contact_sheet(
    look: &ParametricLook,
    files: &[PathBuf],
    sheet_path: &Path,
    console: &mut Console,
) -> Result<usize, Failure> {
    handle_interrupts();
    let data_root = find_data_dir().map_err(|e| Failure::usage(e.to_string()))?;
    let id = LookId::user("bake-preview");
    let transform = look
        .to_transform()
        .map_err(|e| Failure::failed(format!("The look can't be baked: {e}.")))?;
    let provider = BakeProvider {
        id: id.clone(),
        baked: ResolvedLook {
            transform: Arc::new(transform),
            fingerprint: format!("bake:{}", look.name()),
        },
    };
    let options = RenderOptions {
        thumbnail_px: CELL_PX,
        ..RenderOptions::default()
    };
    let renderer =
        Renderer::new(Arc::new(provider), options).map_err(|e| Failure::failed(e.to_string()))?;
    let env = SessionEnv::load(&data_root)
        .map_err(|e| Failure::usage(e.to_string()))?
        .with_default_look(LookId::builtin("neutral"));
    let session = Session::new(env);
    let mut ids: Vec<(PathBuf, PhotoId)> = Vec::new();
    for file in files {
        match session
            .open(std::slice::from_ref(file), OpenOptions::default())
            .added
            .first()
        {
            Some(id) => ids.push((file.clone(), *id)),
            None => console.error(&format!(
                "{}: not an ARW file, left out of the contact sheet.",
                path_text(&file)
            )),
        }
    }
    while !session.wait_idle(std::time::Duration::from_millis(100)) {
        if interrupted() {
            return Err(Failure::new(exit::INTERRUPTED, "Interrupted."));
        }
    }

    let cancel = RenderCancel::new();
    let mut rows: Vec<(Picture, Picture)> = Vec::new();
    for (path, photo) in &ids {
        if interrupted() {
            return Err(Failure::new(exit::INTERRUPTED, "Interrupted."));
        }
        if let Some(PhotoStatus::Skipped { reason }) = session.status(*photo) {
            console.error(&format!("{}: {reason}", path_text(&path)));
            continue;
        }
        let rendered = ym_core::pipeline::PhotoSource::from_session(&session, *photo)
            .map_err(|e| e.to_string())
            .and_then(|src| {
                let mut with_look = src.settings.clone();
                with_look.look = id.clone();
                let neutral = DevelopSettings {
                    look: LookId::builtin("neutral"),
                    ..src.settings.clone()
                };
                let left = renderer
                    .thumbnail_png(&src.with_settings(neutral), &cancel)
                    .map_err(|e| e.to_string())?;
                let right = renderer
                    .thumbnail_png(&src.with_settings(with_look), &cancel)
                    .map_err(|e| e.to_string())?;
                Ok((decode_png(&left)?, decode_png(&right)?))
            });
        match rendered {
            Ok(pair) => rows.push(pair),
            Err(message) => console.error(&format!("{}: {message}", path_text(&path))),
        }
    }
    if rows.is_empty() {
        return Err(Failure::failed(
            "None of the preview photos could be rendered, so there is no contact sheet.",
        ));
    }
    let (w, h, rgb) = layout(&rows);
    let png = encode_png_rgb8(w, h, &rgb).map_err(|e| Failure::failed(e.to_string()))?;
    std::fs::write(sheet_path, png)
        .map_err(|e| Failure::failed(format!("Couldn't write {}: {e}.", path_text(&sheet_path))))?;
    Ok(rows.len())
}

/// A decoded picture: width, height and RGB8 samples.
struct Picture {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

fn decode_png(bytes: &[u8]) -> Result<Picture, String> {
    let decoder = png::Decoder::new(io::Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buffer = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("the picture is too large")?
    ];
    let info = reader.next_frame(&mut buffer).map_err(|e| e.to_string())?;
    if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Eight {
        return Err("the preview is not an 8-bit RGB picture".to_owned());
    }
    buffer.truncate(info.buffer_size());
    Ok(Picture {
        width: info.width as usize,
        height: info.height as usize,
        rgb: buffer,
    })
}

/// Lays the pairs out in rows on a white page: two columns, each as wide as the widest picture,
/// each row as tall as its taller picture, pictures centred in their cell.
fn layout(rows: &[(Picture, Picture)]) -> (usize, usize, Vec<u8>) {
    let cell_w = rows
        .iter()
        .flat_map(|(a, b)| [a.width, b.width])
        .max()
        .unwrap_or(1);
    let width = GAP_PX + 2 * (cell_w + GAP_PX);
    let height = GAP_PX
        + rows
            .iter()
            .map(|(a, b)| a.height.max(b.height) + GAP_PX)
            .sum::<usize>();
    let mut page = vec![255u8; width * height * 3];
    let mut top = GAP_PX;
    for (a, b) in rows {
        let row_h = a.height.max(b.height);
        for (column, pic) in [a, b].into_iter().enumerate() {
            let x0 = GAP_PX + column * (cell_w + GAP_PX) + (cell_w - pic.width) / 2;
            let y0 = top + (row_h - pic.height) / 2;
            for y in 0..pic.height {
                let src = &pic.rgb[y * pic.width * 3..(y + 1) * pic.width * 3];
                let at = ((y0 + y) * width + x0) * 3;
                page[at..at + pic.width * 3].copy_from_slice(src);
            }
        }
        top += row_h + GAP_PX;
    }
    (width, height, page)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pic(w: usize, h: usize, v: u8) -> Picture {
        Picture {
            width: w,
            height: h,
            rgb: vec![v; w * h * 3],
        }
    }

    #[test]
    fn the_sheet_has_two_columns_and_a_row_per_photo() {
        let rows = vec![
            (pic(8, 4, 10), pic(8, 4, 20)),
            (pic(4, 8, 30), pic(4, 8, 40)),
        ];
        let (w, h, rgb) = layout(&rows);
        assert_eq!(w, GAP_PX + 2 * (8 + GAP_PX));
        assert_eq!(h, GAP_PX + (4 + GAP_PX) + (8 + GAP_PX));
        assert_eq!(rgb.len(), w * h * 3);
        // The first picture sits at the first cell; the margin around it is the paper colour.
        let px = |x: usize, y: usize| rgb[(y * w + x) * 3];
        assert_eq!(px(0, 0), 255);
        assert_eq!(px(GAP_PX, GAP_PX), 10);
        assert_eq!(px(GAP_PX + 8 + GAP_PX, GAP_PX), 20);
        // The narrow portrait pictures are centred in their cells.
        let second_row = GAP_PX + 4 + GAP_PX;
        assert_eq!(px(GAP_PX + 2, second_row), 30);
        assert_eq!(px(GAP_PX, second_row), 255);
    }

    #[test]
    fn the_look_table_marks_the_default_and_missing_looks() {
        let looks = vec![
            LookEntry {
                id: LookId::builtin("neutral"),
                name: "Neutral".into(),
                group: LookGroupInfo::BuiltIn,
                input: LookInput::LogC4,
                is_default: true,
                missing: false,
            },
            LookEntry {
                id: LookId::arri(3000, "film"),
                name: "Film".into(),
                group: LookGroupInfo::Arri {
                    no: 3000,
                    name: "Film".into(),
                },
                input: LookInput::LogC3Ei800,
                is_default: false,
                missing: true,
            },
        ];
        let text = list_text(&looks, &[]);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("ID"), "{text}");
        assert!(
            lines[1].contains("builtin:neutral") && lines[1].ends_with("default"),
            "{text}"
        );
        assert!(
            lines[2].contains("arri:3000:film")
                && lines[2].contains("logc3")
                && lines[2].ends_with("missing: import again"),
            "{text}"
        );
    }
}
