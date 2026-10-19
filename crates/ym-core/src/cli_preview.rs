//! `cli_preview`: the `ym render …` command that does what an export job does (`docs/09-export.md`
//! section Options, `docs/10-cli.md`).
//!
//! The Export panel shows it under the options with a Copy button. It carries the photos (the
//! folder when the job is every ARW file in it, else each file), the look and its intensity, the
//! format and its options, the colour space, the size, the name template, what to do when the file
//! exists and the destination:
//!
//! ```text
//! ym render /Users/me/Pictures/Lisbon \
//!   --look neutral \
//!   --format jpeg \
//!   --quality 92 \
//!   --colour-space srgb \
//!   --out '/Users/me/Pictures/Lisbon export'
//! ```
//!
//! Options that are the CLI's defaults are left out (`--compression lzw`, the name template
//! `{name}_{look}`, "Keep both"). Per-photo adjustments beyond the look do not fit on a command
//! line (the CLI takes them through `--settings`, `docs/10`): [`CliPreview::adjusted_photos`] counts
//! the photos whose settings the command does not reproduce, for the panel's "N photos have
//! adjustments this command doesn't include."
//!
//! # The look of a job
//!
//! A command has one `--look`. It is the look (with its intensity) most photos of the job have; on a
//! tie, the one that comes first. A photo whose settings are not exactly *Auto with that look at
//! that intensity* is counted as adjusted, so a job in which three photos have another look says
//! so instead of silently exporting them with the wrong one.
//!
//! Output sharpening (`--output-sharpening`, Detail › Sharpening › Output) is carried the same
//! way: the amount most photos of the job have, written only when it is not the default 100, and a
//! photo with another amount counts as adjusted.

use std::path::{Path, PathBuf};

use ym_export::naming;
use ym_export::options::DEFAULT_NAME_TEMPLATE;
use ym_export::{ColorSpace, FormatKind, IfExists, Size, TiffCompression};
use ym_looks::library::LookId;

use crate::export_queue::ExportJob;
use crate::looks::LookService;
use crate::session::Session;
use crate::settings::{DevelopSettings, OUTPUT_SHARPENING_DEFAULT};

/// The command line tool's name.
pub const COMMAND: &str = "ym";

/// The command for a job and what it leaves out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliPreview {
    /// The command, one option per line, ready to paste into a terminal.
    pub command: String,
    /// Photos in the job that have adjustments the command does not include.
    pub adjusted_photos: usize,
}

impl CliPreview {
    /// "N photos have adjustments this command doesn't include.", when there are any.
    #[must_use]
    pub fn note(&self) -> Option<String> {
        adjustments_note(self.adjusted_photos)
    }
}

/// "1 photo has adjustments this command doesn't include." / "N photos have …"; `None` for none.
#[must_use]
pub fn adjustments_note(adjusted_photos: usize) -> Option<String> {
    match adjusted_photos {
        0 => None,
        1 => Some("1 photo has adjustments this command doesn't include.".to_owned()),
        n => Some(format!(
            "{n} photos have adjustments this command doesn't include."
        )),
    }
}

/// The look and intensity a command for these photos carries: the most common pair, the earliest
/// on a tie, Neutral at 100 for no photos.
#[must_use]
pub fn command_look(settings: &[&DevelopSettings]) -> (LookId, f64) {
    let mut counts: Vec<(LookId, f64, usize)> = Vec::new();
    for s in settings {
        match counts
            .iter_mut()
            .find(|(l, i, _)| *l == s.look && *i == s.look_intensity)
        {
            Some(entry) => entry.2 += 1,
            None => counts.push((s.look.clone(), s.look_intensity, 1)),
        }
    }
    let mut best: Option<&(LookId, f64, usize)> = None;
    for entry in &counts {
        if best.is_none_or(|b| entry.2 > b.2) {
            best = Some(entry);
        }
    }
    best.map_or_else(
        || (LookId::builtin("neutral"), 100.0),
        |(l, i, _)| (l.clone(), *i),
    )
}

/// The output sharpening amount (percent) a command for these photos carries: the most common
/// amount, the earliest on a tie, the default 100 for no photos.
#[must_use]
pub fn command_output_sharpening(settings: &[&DevelopSettings]) -> f64 {
    let mut counts: Vec<(f64, usize)> = Vec::new();
    for s in settings {
        match counts.iter_mut().find(|(v, _)| *v == s.output_sharpening) {
            Some(entry) => entry.1 += 1,
            None => counts.push((s.output_sharpening, 1)),
        }
    }
    let mut best: Option<&(f64, usize)> = None;
    for entry in &counts {
        if best.is_none_or(|b| entry.1 > b.1) {
            best = Some(entry);
        }
    }
    best.map_or(OUTPUT_SHARPENING_DEFAULT, |(v, _)| *v)
}

/// How many of `settings` differ from *automatic development with this look at this intensity*
/// (and the default output sharpening).
#[must_use]
pub fn adjusted_photos(settings: &[&DevelopSettings], look: &LookId, intensity: f64) -> usize {
    adjusted_photos_with(settings, look, intensity, OUTPUT_SHARPENING_DEFAULT)
}

/// [`adjusted_photos`] for a command that also carries an output sharpening amount.
#[must_use]
pub fn adjusted_photos_with(
    settings: &[&DevelopSettings],
    look: &LookId,
    intensity: f64,
    output_sharpening: f64,
) -> usize {
    let mut plain = DevelopSettings::with_look(look.clone());
    plain.look_intensity = intensity;
    plain.output_sharpening = output_sharpening;
    settings.iter().filter(|s| ***s != plain).count()
}

/// Quotes a word for a POSIX shell when it needs it.
#[must_use]
pub fn shell_quote(word: &str) -> String {
    let safe = !word.is_empty()
        && word
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&b));
    if safe {
        word.to_owned()
    } else {
        format!("'{}'", word.replace('\'', "'\\''"))
    }
}

fn intensity_text(intensity: f64) -> String {
    if (intensity - intensity.round()).abs() < 1e-9 {
        format!("{}", intensity.round() as i64)
    } else {
        let t = format!("{intensity:.2}");
        t.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

fn color_space_word(c: ColorSpace) -> &'static str {
    match c {
        ColorSpace::Srgb => "srgb",
        ColorSpace::DisplayP3 => "p3",
        ColorSpace::Rec2020 => "rec2020",
    }
}

/// The ARW files directly inside `folder`, the way `Session::open` collects them.
fn arw_files_in(folder: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            !name.starts_with('.')
                && Path::new(&*name)
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("arw"))
                && e.path().is_file()
        })
        .map(|e| e.path())
        .collect()
}

/// The folder when the job is exactly every ARW file in one folder.
fn whole_folder(paths: &[PathBuf]) -> Option<PathBuf> {
    let folder = paths.first()?.parent()?.to_path_buf();
    if paths.iter().any(|p| p.parent() != Some(folder.as_path())) {
        return None;
    }
    let mut in_folder = arw_files_in(&folder);
    in_folder.sort();
    let mut wanted = paths.to_vec();
    wanted.sort();
    wanted.dedup();
    (in_folder == wanted).then_some(folder)
}

/// The words of the command, one `String` per line after the first.
///
/// `looks` makes `--look` a slug where the slug names exactly that look (the full id otherwise).
/// `home` is where `~/Pictures/Yellowmoon export` is, for a source folder that has no parent.
#[must_use]
pub fn cli_preview(
    session: &Session,
    job: &ExportJob,
    looks: Option<&LookService>,
    home: Option<&Path>,
) -> CliPreview {
    let views: Vec<_> = session
        .scope_ids(job.scope)
        .into_iter()
        .filter_map(|id| session.view(id).ok())
        .collect();
    let settings: Vec<&DevelopSettings> = views.iter().map(|v| &v.settings).collect();
    let paths: Vec<PathBuf> = views.iter().map(|v| v.path.clone()).collect();
    build(&paths, &settings, job, looks, home)
}

/// [`cli_preview`] for photos given directly (paths and their settings, in job order).
#[must_use]
pub fn build(
    paths: &[PathBuf],
    settings: &[&DevelopSettings],
    job: &ExportJob,
    looks: Option<&LookService>,
    home: Option<&Path>,
) -> CliPreview {
    let o = &job.options;
    let (look, intensity) = command_look(settings);
    let output_sharpening = command_output_sharpening(settings);
    let look_word = {
        let slug = look.slug();
        let unambiguous = looks.is_some_and(|l| l.resolve(slug).is_ok_and(|id| id == look));
        if unambiguous || looks.is_none() {
            shell_quote(slug)
        } else {
            shell_quote(look.as_str())
        }
    };

    let mut flags = vec![format!("--look {look_word}")];
    if (intensity - 100.0).abs() > 1e-9 {
        flags.push(format!("--intensity {}", intensity_text(intensity)));
    }
    if (output_sharpening - OUTPUT_SHARPENING_DEFAULT).abs() > 1e-9 {
        flags.push(format!(
            "--output-sharpening {}",
            intensity_text(output_sharpening)
        ));
    }
    let format_word = match o.format {
        FormatKind::Jpeg => "jpeg",
        FormatKind::Tiff => "tiff",
        FormatKind::Png => "png",
        FormatKind::Dng => "dng",
    };
    flags.push(format!("--format {format_word}"));
    match o.format {
        FormatKind::Jpeg => flags.push(format!("--quality {}", o.jpeg_quality.get())),
        FormatKind::Tiff | FormatKind::Png => flags.push(format!("--bits {}", o.bit_depth.bits())),
        FormatKind::Dng => {}
    }
    if o.format == FormatKind::Tiff && o.tiff_compression != TiffCompression::Lzw {
        let word = match o.tiff_compression {
            TiffCompression::None => "none",
            TiffCompression::Lzw => "lzw",
            TiffCompression::Zip => "zip",
        };
        flags.push(format!("--compression {word}"));
    }
    if o.format != FormatKind::Dng {
        flags.push(format!(
            "--colour-space {}",
            color_space_word(o.color_space)
        ));
        if let Size::LongEdge { px } = o.size {
            flags.push(format!("--long-edge {px}"));
        }
    }
    if o.name_template != DEFAULT_NAME_TEMPLATE {
        flags.push(format!("--name {}", shell_quote(&o.name_template)));
    }
    match o.if_exists {
        IfExists::Skip => flags.push("--skip-existing".to_owned()),
        IfExists::Replace => flags.push("--overwrite".to_owned()),
        IfExists::KeepBoth => {}
    }
    // The destination: the folder the user chose, else `<source folder> export` when all the
    // photos share one folder (with several folders the CLI's own default, one next to each, is
    // what the app does too).
    let destination = o.destination.clone().or_else(|| {
        let first = paths.first()?;
        let same_folder = paths.iter().all(|p| p.parent() == first.parent());
        same_folder.then(|| {
            naming::default_destination(first)
                .or_else(|| {
                    home.map(Path::to_path_buf)
                        .or_else(naming::home_dir)
                        .map(|h| naming::fallback_destination(&h))
                })
                .unwrap_or_else(|| PathBuf::from("Yellowmoon export"))
        })
    });
    if let Some(dest) = destination {
        flags.push(format!("--out {}", shell_quote(&dest.to_string_lossy())));
    }

    let inputs = match whole_folder(paths) {
        Some(folder) => shell_quote(&folder.to_string_lossy()),
        None => paths
            .iter()
            .map(|p| shell_quote(&p.to_string_lossy()))
            .collect::<Vec<_>>()
            .join(" "),
    };
    let head = if inputs.is_empty() {
        format!("{COMMAND} render")
    } else {
        format!("{COMMAND} render {inputs}")
    };
    CliPreview {
        command: format!("{head} \\\n  {}", flags.join(" \\\n  ")),
        adjusted_photos: adjusted_photos_with(settings, &look, intensity, output_sharpening),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::ApplyScope;
    use ym_export::{BitDepth, ExportOptions, JpegQuality};

    fn job(options: ExportOptions) -> ExportJob {
        ExportJob {
            scope: ApplyScope::All,
            options,
        }
    }

    fn with_look(slug: &str, intensity: f64) -> DevelopSettings {
        let mut s = DevelopSettings::with_look(LookId::builtin(slug));
        s.look_intensity = intensity;
        s
    }

    #[test]
    fn quoting_leaves_plain_paths_alone_and_protects_the_rest() {
        assert_eq!(
            shell_quote("/Users/me/Pictures/Lisbon"),
            "/Users/me/Pictures/Lisbon"
        );
        assert_eq!(shell_quote("neutral"), "neutral");
        assert_eq!(shell_quote("Lisbon export"), "'Lisbon export'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote("{name}_{look}"), "'{name}_{look}'");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("naïve"), "'naïve'");
    }

    #[test]
    fn the_command_look_is_the_most_common_pair_and_the_first_on_a_tie() {
        let a = with_look("neutral", 100.0);
        let b = with_look("signature", 100.0);
        let c = with_look("signature", 60.0);
        assert_eq!(command_look(&[]).0, LookId::builtin("neutral"));
        assert_eq!(command_look(&[&a, &b, &b]).0, LookId::builtin("signature"));
        assert_eq!(command_look(&[&a, &b]).0, LookId::builtin("neutral"));
        let (l, i) = command_look(&[&b, &c, &c]);
        assert_eq!((l, i), (LookId::builtin("signature"), 60.0));
    }

    #[test]
    fn adjusted_photos_are_those_the_command_does_not_reproduce() {
        let plain = with_look("neutral", 100.0);
        let mut warm = plain.clone();
        warm.saturation = crate::settings::PercentSetting::manual_value(20.0);
        let other_look = with_look("signature", 100.0);
        let all = [&plain, &warm, &other_look];
        let look = LookId::builtin("neutral");
        assert_eq!(adjusted_photos(&all, &look, 100.0), 2);
        assert_eq!(adjusted_photos(&[&plain], &look, 100.0), 0);
        // The intensity is part of what the command says.
        assert_eq!(adjusted_photos(&[&plain], &look, 50.0), 1);
        assert_eq!(adjustments_note(0), None);
        assert_eq!(
            adjustments_note(1).as_deref(),
            Some("1 photo has adjustments this command doesn't include.")
        );
        assert_eq!(
            adjustments_note(3).as_deref(),
            Some("3 photos have adjustments this command doesn't include.")
        );
    }

    #[test]
    fn the_command_carries_the_options_and_leaves_out_the_defaults() {
        let dir = tempfile::tempdir().expect("dir");
        let folder = dir.path().join("Lisbon");
        std::fs::create_dir(&folder).expect("folder");
        let a = folder.join("DSC00001.ARW");
        let b = folder.join("DSC00002.ARW");
        std::fs::write(&a, b"x").expect("file");
        std::fs::write(&b, b"x").expect("file");
        std::fs::write(folder.join("notes.txt"), b"x").expect("file");
        let s = with_look("neutral", 100.0);
        let settings = [&s, &s];
        let paths = [a.clone(), b.clone()];

        let jpeg = build(
            &paths,
            &settings,
            &job(ExportOptions::default()),
            None,
            None,
        );
        let folder_text = folder.to_string_lossy();
        assert_eq!(
            jpeg.command,
            format!(
                "ym render {folder_text} \\\n  --look neutral \\\n  --format jpeg \\\n  --quality 92 \\\n  --colour-space srgb \\\n  --out {}",
                shell_quote(&format!("{folder_text} export"))
            )
        );
        assert_eq!(jpeg.adjusted_photos, 0);

        // Only one of the two files: each is listed, and a zip tiff names its compression.
        let tiff = build(
            &paths[..1],
            &settings[..1],
            &job(ExportOptions {
                format: FormatKind::Tiff,
                bit_depth: BitDepth::Eight,
                tiff_compression: TiffCompression::Zip,
                color_space: ColorSpace::DisplayP3,
                size: Size::LongEdge { px: 3000 },
                if_exists: IfExists::Skip,
                name_template: "{name}-{seq}".into(),
                destination: Some(PathBuf::from("/tmp/My Exports")),
                ..ExportOptions::default()
            }),
            None,
            None,
        );
        let c = &tiff.command;
        assert!(
            c.starts_with(&format!("ym render {} \\\n", a.display())),
            "{c}"
        );
        for flag in [
            "--format tiff",
            "--bits 8",
            "--compression zip",
            "--colour-space p3",
            "--long-edge 3000",
            "--name '{name}-{seq}'",
            "--skip-existing",
            "--out '/tmp/My Exports'",
        ] {
            assert!(c.contains(flag), "{flag} in {c}");
        }
        assert!(!c.contains("--quality"), "{c}");

        let dng = build(
            &paths,
            &settings,
            &job(ExportOptions {
                format: FormatKind::Dng,
                if_exists: IfExists::Replace,
                ..ExportOptions::default()
            }),
            None,
            None,
        );
        assert!(!dng.command.contains("--colour-space"));
        assert!(!dng.command.contains("--bits"));
        assert!(dng.command.contains("--overwrite"));
        let png = build(
            &paths,
            &settings,
            &job(ExportOptions {
                format: FormatKind::Png,
                jpeg_quality: JpegQuality::new(70).expect("quality"),
                ..ExportOptions::default()
            }),
            None,
            None,
        );
        assert!(png.command.contains("--bits 16") && !png.command.contains("--quality"));
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_in_the_path_is_shown_as_it_was_given() {
        // How a Mac reaches its temporary folder: `/var` is a link to `/private/var`, and the
        // command must say `/var/…`, in the folder and in `--out`, because that is the name the
        // person knows (and the one `ym render` is given).
        let real = tempfile::tempdir().expect("real");
        let links = tempfile::tempdir().expect("links");
        let via = links.path().join("via");
        std::os::unix::fs::symlink(real.path(), &via).expect("symlink");
        let folder = via.join("Lisbon");
        std::fs::create_dir(&folder).expect("folder");
        let a = folder.join("DSC00001.ARW");
        std::fs::write(&a, b"x").expect("file");
        let s = with_look("neutral", 100.0);

        let preview = build(
            std::slice::from_ref(&a),
            &[&s],
            &job(ExportOptions::default()),
            None,
            None,
        );
        let folder_text = folder.to_string_lossy();
        assert_eq!(
            preview.command,
            format!(
                "ym render {folder_text} \\\n  --look neutral \\\n  --format jpeg \\\n  --quality 92 \\\n  --colour-space srgb \\\n  --out {}",
                shell_quote(&format!("{folder_text} export"))
            )
        );

        // A source folder that is itself a link: the export folder is named after the link and
        // sits next to it, which is where the Export panel says it goes.
        let target = real.path().join("2024-05-trip");
        std::fs::create_dir(&target).expect("target");
        std::fs::write(target.join("DSC00001.ARW"), b"x").expect("file");
        let link = links.path().join("Shoot");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let preview = build(
            &[link.join("DSC00001.ARW")],
            &[&s],
            &job(ExportOptions::default()),
            None,
            None,
        );
        let out = links.path().join("Shoot export");
        assert!(
            preview
                .command
                .ends_with(&format!("--out {}", shell_quote(&out.to_string_lossy()))),
            "{}",
            preview.command
        );
    }

    #[test]
    fn output_sharpening_is_named_only_when_it_is_not_the_default() {
        let paths = [
            PathBuf::from("/a/one/DSC1.ARW"),
            PathBuf::from("/a/one/DSC2.ARW"),
            PathBuf::from("/a/one/DSC3.ARW"),
        ];
        let plain = with_look("neutral", 100.0);
        let mut soft = plain.clone();
        soft.output_sharpening = 35.0;
        let mut off = plain.clone();
        off.output_sharpening = 0.0;
        // The default says nothing, and nothing is adjusted.
        let p = build(
            &paths,
            &[&plain, &plain, &plain],
            &job(ExportOptions::default()),
            None,
            None,
        );
        assert!(!p.command.contains("--output-sharpening"), "{}", p.command);
        assert_eq!(p.adjusted_photos, 0);
        // The amount most photos have is in the command; the others are adjusted.
        let p = build(
            &paths,
            &[&soft, &soft, &plain],
            &job(ExportOptions::default()),
            None,
            None,
        );
        assert!(
            p.command.contains("--output-sharpening 35"),
            "{}",
            p.command
        );
        assert_eq!(p.adjusted_photos, 1);
        // 0 is an amount like any other.
        let p = build(
            &paths,
            &[&off, &off, &off],
            &job(ExportOptions::default()),
            None,
            None,
        );
        assert!(p.command.contains("--output-sharpening 0"), "{}", p.command);
        assert_eq!(p.adjusted_photos, 0);
        // On a tie the first amount wins, as the look does.
        assert_eq!(command_output_sharpening(&[&soft, &plain]), 35.0);
        assert_eq!(command_output_sharpening(&[]), 100.0);
        // Without the command's amount a photo with another amount is adjusted.
        assert_eq!(
            adjusted_photos(&[&soft], &LookId::builtin("neutral"), 100.0),
            1
        );
    }

    #[test]
    fn photos_from_several_folders_get_no_out_and_a_changed_intensity_is_named() {
        let s = with_look("signature", 60.0);
        let paths = [
            PathBuf::from("/a/one/DSC1.ARW"),
            PathBuf::from("/b/two/DSC2.ARW"),
        ];
        let p = build(
            &paths,
            &[&s, &s],
            &job(ExportOptions::default()),
            None,
            None,
        );
        assert!(!p.command.contains("--out"), "{}", p.command);
        assert!(p.command.contains("--look signature"), "{}", p.command);
        assert!(p.command.contains("--intensity 60"), "{}", p.command);
        assert!(
            p.command.contains("/a/one/DSC1.ARW /b/two/DSC2.ARW"),
            "{}",
            p.command
        );
        assert_eq!(p.adjusted_photos, 0);
    }
}
