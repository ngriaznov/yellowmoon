//! Looks: the built-in looks, the library folder, the importer for ARRI look packages and other
//! `.cube` files, recognition rules, the manifest and the library lock.
//!
//! ARRI's own look files are never bundled; users import them at runtime.
//!
//! Specs: `docs/08-looks.md`; application maths is in `docs/02-colour-pipeline.md` (stage C10).

#![warn(missing_docs)]

pub mod import;
pub mod library;
pub mod lock;
pub mod manifest;
pub mod recognize;
