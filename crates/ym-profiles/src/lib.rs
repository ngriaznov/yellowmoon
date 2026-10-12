//! Camera profile registry: loads `data/cameras/**` and answers "which profile does this body get?".
//!
//! Lookup order is exact model ID, then same-sensor sibling, then the default profile (A7R IV);
//! a Sony file is never refused for lack of a profile.
//!
//! Specs: `docs/03-camera-profiles.md` (detection, lookup, profile files, tests); the JSON schema
//! lives in `crates/ym-profiles/schema/`.

#![warn(missing_docs)]

pub mod detect;
pub mod profile;
pub mod registry;
