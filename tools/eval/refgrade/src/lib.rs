//! The Reference Grade evaluation tool (`docs/34-reference-grade.md` section 10.1, RG-10).
//!
//! It runs **the product's fit** through the core's own API on a local set of photos and reference stills (never committed), measures
//! the harms H1 to H6 and the look transfer T1 to T3 of every fit on the full fit-size renders, writes the per-fit records in the proof
//! of concept's format, and compares two runs under the owner's three acceptance rules. Test tooling: it links the core and is never
//! part of the app or of `ym`.
//!
//! * [`measure`] the harms and the transfer metrics, on measured pixels;
//! * [`split`] the local `split.json` and the seeded sample for the owner;
//! * [`run`] the fit loop, shards, resume, and the element-mask check against labels;
//! * [`report`] the records, their summaries and the merge of shards;
//! * [`compare`] two runs under the three rules;
//! * [`diagnose`] the objective of a solution made elsewhere against the fit's own, term by term (`docs/34` section 16.10).

pub mod compare;
pub mod diagnose;
pub mod measure;
pub mod report;
pub mod run;
pub mod split;
pub mod stages;
