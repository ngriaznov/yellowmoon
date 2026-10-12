//! ONNX Runtime backend (`ort` 2.x) behind [`TileModel`] (`docs/07-ml-models.md` section Runtime).
//!
//! The static-shape ONNX file written by `training/jdd/export.py` is run with ONNX Runtime's CPU
//! provider (or CoreML on macOS, cargo feature `coreml`). Needs a copy to and from host memory per tile.
//!
//! The crate is built with ort's `load-dynamic` feature: nothing is downloaded at build time and the
//! ONNX Runtime shared library is found at run time through the `ORT_DYLIB_PATH` environment variable
//! (for example the `libonnxruntime` that ships inside the Python `onnxruntime` wheel), or through
//! [`OrtJdd::with_library`]. A missing library is reported as [`MlError::Backend`], never a panic. The library
//! is loaded once per process: the first attempt decides, and a failed one is reported again by every later
//! call instead of being retried (the `ort` crate cannot recover from a failed load).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::model::{
    ExecContext, IN_CHANNELS, MlError, OUT_CHANNELS, OUT_SCALE, Result, TileInputs, TileModel,
    TileOutput, TileShape,
};

/// File name ONNX Runtime is looked up under when `ORT_DYLIB_PATH` is not set (the loader's search path
/// applies), the same default the `ort` crate uses.
const DEFAULT_LIBRARY: &str = if cfg!(target_os = "windows") {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

/// The library to load: the value of `ORT_DYLIB_PATH` if it is set and not empty, else [`DEFAULT_LIBRARY`].
fn library_from_env(env_value: Option<OsString>) -> PathBuf {
    match env_value {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => PathBuf::from(DEFAULT_LIBRARY),
    }
}

/// Outcome of the first attempt to load ONNX Runtime in this process.
///
/// ONNX Runtime is loaded once per process, and the first attempt decides. `ort` 2.0.0-rc.13 marks its
/// library cell as initialised even when the load failed, so a second attempt (or any later `ort` call)
/// would use an uninitialised library handle and panic. A failed load is therefore remembered and
/// reported again, and `ort::init_from` is never called twice after a failure; the process has to be
/// restarted with a correct `ORT_DYLIB_PATH`.
static LIBRARY: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Load ONNX Runtime (see [`LIBRARY`]): `library` if given, else `ORT_DYLIB_PATH`, else the loader's default.
fn load_library(library: Option<&Path>) -> Result<()> {
    LIBRARY
        .get_or_init(|| {
            let lib = library.map_or_else(
                || library_from_env(std::env::var_os("ORT_DYLIB_PATH")),
                Path::to_path_buf,
            );
            match ort::init_from(&lib) {
                Ok(env) => {
                    env.commit();
                    Ok(())
                }
                Err(e) => Err(e.to_string()),
            }
        })
        .clone()
        .map_err(backend_err)
}

fn backend_err(e: impl std::fmt::Display) -> MlError {
    MlError::Backend {
        backend: "ort",
        detail: e.to_string(),
    }
}

/// The `jdd` network run by ONNX Runtime.
pub struct OrtJdd {
    session: Mutex<Session>,
    shape: TileShape,
    halo_half_px: usize,
    input_name: String,
}

impl OrtJdd {
    /// Load `onnx_path`, a static-shape export for tiles of `shape` with the given halo. The ONNX Runtime
    /// library is taken from `ORT_DYLIB_PATH`.
    pub fn new(onnx_path: impl AsRef<Path>, shape: TileShape, halo_half_px: usize) -> Result<Self> {
        Self::build(onnx_path.as_ref(), shape, halo_half_px, None)
    }

    /// Like [`OrtJdd::new`] with an explicit path to the ONNX Runtime shared library.
    pub fn with_library(
        library: impl AsRef<Path>,
        onnx_path: impl AsRef<Path>,
        shape: TileShape,
        halo_half_px: usize,
    ) -> Result<Self> {
        Self::build(
            onnx_path.as_ref(),
            shape,
            halo_half_px,
            Some(library.as_ref()),
        )
    }

    fn build(
        onnx: &Path,
        shape: TileShape,
        halo_half_px: usize,
        library: Option<&Path>,
    ) -> Result<Self> {
        shape.validate()?;
        // Load the library ourselves: `ort::init()` would load it lazily and panic ("Failed to load ONNX
        // Runtime dylib") when it is missing.
        load_library(library)?;
        #[allow(unused_mut)]
        let mut builder = Session::builder()
            .map_err(backend_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(backend_err)?;
        #[cfg(feature = "coreml")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CoreML::default().build()])
                .map_err(backend_err)?;
        }
        let session = builder.commit_from_file(onnx).map_err(backend_err)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "x".to_string());
        Ok(Self {
            session: Mutex::new(session),
            shape,
            halo_half_px,
            input_name,
        })
    }
}

impl TileModel for OrtJdd {
    fn input_shape(&self) -> TileShape {
        self.shape
    }

    fn halo_half_px(&self) -> usize {
        self.halo_half_px
    }

    fn run(&self, inputs: &TileInputs, out: &mut TileOutput, _ctx: &ExecContext<'_>) -> Result<()> {
        if inputs.shape() != self.shape || out.shape() != self.shape {
            return Err(MlError::Shape(format!(
                "ort model expects tiles of {}, got input {} output {}",
                self.shape,
                inputs.shape(),
                out.shape()
            )));
        }
        let dims = [1usize, IN_CHANNELS, self.shape.height, self.shape.width];
        let tensor = Tensor::from_array((dims, inputs.as_slice().to_vec())).map_err(backend_err)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| backend_err("session lock poisoned"))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(backend_err)?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(backend_err)?;
        let want = [
            1,
            OUT_CHANNELS as i64,
            (OUT_SCALE * self.shape.height) as i64,
            (OUT_SCALE * self.shape.width) as i64,
        ];
        if shape.as_ref() != want.as_slice() {
            return Err(backend_err(format!(
                "unexpected output shape {shape:?}, wanted {want:?}"
            )));
        }
        out.as_mut_slice().copy_from_slice(data);
        Ok(())
    }

    fn name(&self) -> &str {
        "ort"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_path_follows_the_environment_variable() {
        assert_eq!(
            library_from_env(Some("/opt/ort/libonnxruntime.so.1.31.0".into())),
            PathBuf::from("/opt/ort/libonnxruntime.so.1.31.0")
        );
        assert_eq!(library_from_env(None), PathBuf::from(DEFAULT_LIBRARY));
        assert_eq!(
            library_from_env(Some(OsString::new())),
            PathBuf::from(DEFAULT_LIBRARY)
        );
    }
}
