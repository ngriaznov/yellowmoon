//! LibRaw and libjpeg as subprocesses: `rawpy_compare.py` and `libjpeg_dump.py` write the pixels
//! those libraries decode, we read them back.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

/// Which Python oracle a [`Runner`] drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    /// LibRaw through rawpy (`rawpy_compare.py`): every variant.
    LibRaw,
    /// libjpeg through pylibjpeg-libjpeg (`libjpeg_dump.py`): the tiles of Lossless L files only.
    Libjpeg,
}

impl Tool {
    fn script(self) -> &'static str {
        match self {
            Self::LibRaw => "rawpy_compare.py",
            Self::Libjpeg => "libjpeg_dump.py",
        }
    }

    fn version_check(self) -> &'static str {
        match self {
            Self::LibRaw => "import rawpy, numpy; print('.'.join(map(str, rawpy.libraw_version)))",
            Self::Libjpeg => "import libjpeg, numpy; print(libjpeg.__version__)",
        }
    }

    fn install_hint(self) -> &'static str {
        match self {
            Self::LibRaw => "pip install rawpy numpy",
            Self::Libjpeg => "pip install pylibjpeg-libjpeg numpy",
        }
    }
}

/// Runs one of the Python oracle scripts for one file at a time.
pub struct Runner {
    tool: Tool,
    python: String,
    script: PathBuf,
    keep: Option<PathBuf>,
    version: String,
    scratch: tempfile::TempDir,
}

/// What the library made of a file.
pub struct Dump {
    /// The JSON the script printed.
    pub meta: Value,
    /// The pixels as little-endian samples (`raw_image`, or interleaved RGB for three-component files).
    pub pixels: Vec<u16>,
}

impl Runner {
    /// Checks that `python` can import the tool's module and finds the script.
    pub fn new(
        tool: Tool,
        python: &str,
        script: Option<PathBuf>,
        keep: Option<PathBuf>,
    ) -> Result<Self> {
        let script =
            script.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(tool.script()));
        if !script.exists() {
            bail!("the dump script {} does not exist", script.display());
        }
        let check = Command::new(python)
            .args(["-c", tool.version_check()])
            .output()
            .with_context(|| format!("starting {python}"))?;
        if !check.status.success() {
            bail!(
                "{python} cannot import the {tool:?} module ({}): {}",
                tool.install_hint(),
                String::from_utf8_lossy(&check.stderr).trim()
            );
        }
        if let Some(dir) = &keep {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        Ok(Self {
            tool,
            python: python.to_owned(),
            script,
            keep,
            version: String::from_utf8_lossy(&check.stdout).trim().to_owned(),
            scratch: tempfile::tempdir().context("creating a scratch folder")?,
        })
    }

    /// The version of the library behind the script.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Asks the library to decode `file`. `Ok(None)` means the script skipped the file on purpose
    /// (libjpeg only handles Lossless L).
    pub fn dump(&self, file: &Path) -> Result<Option<Dump>> {
        let stem = file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file".to_owned());
        let dir = self.keep.as_deref().unwrap_or(self.scratch.path());
        let tool_name = format!("{:?}", self.tool).to_lowercase();
        let out = dir.join(format!("{stem}.{tool_name}.bin"));
        let output = Command::new(&self.python)
            .arg(&self.script)
            .arg(file)
            .arg("--out")
            .arg(&out)
            .output()
            .with_context(|| format!("running the {:?} dump script", self.tool))?;
        let text = String::from_utf8_lossy(&output.stdout);
        let meta: Value = serde_json::from_str(text.trim())
            .map_err(|e| anyhow!("the {:?} script printed no JSON ({e}): {text}", self.tool))?;
        if meta["kind"] == "skipped" {
            return Ok(None);
        }
        if meta["kind"] == "error" {
            bail!(
                "{:?}: {}",
                self.tool,
                meta["error"].as_str().unwrap_or("unknown error")
            );
        }
        let bytes = std::fs::read(&out).with_context(|| format!("reading {}", out.display()))?;
        if self.keep.is_none() {
            let _ = std::fs::remove_file(&out);
        }
        if bytes.len() % 2 != 0 {
            bail!("the dump has an odd length");
        }
        let pixels = bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok(Some(Dump { meta, pixels }))
    }
}
