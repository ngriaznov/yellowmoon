//! Command-line shape of `yellowmoon` (`docs/10-cli.md` section Commands).
//!
//! Only the top-level structure and a few arguments exist so far; each subcommand grows its full
//! set of options when its feature lands.

use std::path::PathBuf;
use std::sync::OnceLock;

use clap::{Args, Parser, Subcommand};
use ym_core::version::{PIPELINE_VERSION, app_version};

/// Version text shown by `yellowmoon --version`: the package version, the pipeline version and the
/// git commit this binary was built from. clap prefixes it with the program name.
pub fn version_text() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        format!(
            "{} (pipeline {PIPELINE_VERSION}, {})",
            app_version(),
            env!("YM_GIT_SHA")
        )
    })
}

/// Develop and export Sony ARW photos with the same core as the Yellowmoon app.
#[derive(Debug, Parser)]
#[command(name = "yellowmoon", version = version_text(), arg_required_else_help = true)]
pub struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Develop and export photos. Defaults match the app: automatic development and the default
    /// look.
    Render(RenderArgs),
    /// Print metadata and automatic estimates without rendering.
    Info(InfoArgs),
    /// Manage the look library.
    #[command(subcommand)]
    Looks(LooksCommand),
    /// Print the diagnostics that go into a bug report.
    Doctor,
}

/// Arguments of `yellowmoon render`.
#[derive(Debug, Args)]
pub struct RenderArgs {
    /// Files or folders to render. Use `-` to read paths from stdin, one per line.
    #[arg(required = true, value_name = "FILES_OR_FOLDERS")]
    pub inputs: Vec<PathBuf>,
    /// Walk subfolders when an input is a folder.
    #[arg(long)]
    pub recursive: bool,
    /// Look to apply: full id, slug or display name.
    #[arg(long)]
    pub look: Option<String>,
    /// Destination folder (default: `<source folder> export`).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Print one JSON object per file on stdout.
    #[arg(long)]
    pub json: bool,
}

/// Arguments of `yellowmoon info`.
#[derive(Debug, Args)]
pub struct InfoArgs {
    /// ARW files to inspect.
    #[arg(required = true, value_name = "FILES")]
    pub files: Vec<PathBuf>,
    /// Print JSON instead of text.
    #[arg(long)]
    pub json: bool,
}

/// Subcommands of `yellowmoon looks`.
#[derive(Debug, Subcommand)]
pub enum LooksCommand {
    /// List built-in and imported looks, with ids, names, groups, inputs and the default.
    List,
    /// Import ARRI look packages, folders or `.cube` files into the look library.
    Import(LooksImportArgs),
    /// Re-hash every imported look and report missing or changed files.
    Verify,
    /// Set the default look.
    Default(LooksDefaultArgs),
    /// Developer tool: bake a look description to a 33-point LUT and render a contact sheet.
    Bake(LooksBakeArgs),
}

/// Arguments of `yellowmoon looks import`.
#[derive(Debug, Args)]
pub struct LooksImportArgs {
    /// Zip files, folders or `.cube` files to import.
    #[arg(required = true, value_name = "PATHS")]
    pub paths: Vec<PathBuf>,
    /// Input encoding for `.cube` files that are not ARRI packages: `logc4`, `logc3` or
    /// `logc3-rec709`.
    #[arg(long, value_name = "ENCODING")]
    pub input: Option<String>,
}

/// Arguments of `yellowmoon looks default`.
#[derive(Debug, Args)]
pub struct LooksDefaultArgs {
    /// Look id to make the default.
    pub id: String,
}

/// Arguments of `yellowmoon looks bake`.
#[derive(Debug, Args)]
pub struct LooksBakeArgs {
    /// Look description to bake (for example `signature.json`).
    pub look: PathBuf,
    /// ARW files to render into the contact sheet.
    #[arg(long, num_args = 1.., value_name = "ARW")]
    pub preview: Vec<PathBuf>,
    /// Output folder.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn version_text_names_the_pipeline_and_a_commit() {
        let text = version_text();
        assert!(text.starts_with(app_version()), "{text}");
        assert!(
            text.contains(&format!("pipeline {PIPELINE_VERSION}")),
            "{text}"
        );
        assert!(text.ends_with(')'), "{text}");
    }
}
