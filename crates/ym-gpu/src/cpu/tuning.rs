//! The shared pieces of the constant registries (`docs/29-raw-constants.md`, task RC-03).
//!
//! Every raw-algorithm constant that is a number a fit or the owner may change lives in one
//! `Tuning` struct per package ([`adaptive_ratio::Tuning`](super::demosaic::adaptive_ratio::Tuning),
//! [`diffraction::Tuning`](super::sharpen::diffraction::Tuning) and
//! [`output_sharpen::Tuning`](super::output_sharpen::Tuning); the two cubics of the log-guided
//! demosaic keep [`FixedCubics`](super::demosaic::log_guided::FixedCubics)). The CPU code reads the
//! fields, and the kernel wrappers prepend a WGSL `const` block generated from the same struct, so a
//! constant has one home on the CPU and on the device.
//!
//! Each package also has a `pub const REGISTRY`: one [`Constant`] per field (and a few for the
//! structural numbers the inventory of `docs/29` §2.1 names), with the row ID, the value of
//! `Tuning::DEFAULT`, the [`Kind`] and the [`Status`] of its derivation. `tests/constants_registry.rs`
//! holds the registry against the inventory, the generated block and the source text.

use std::fmt::Write as _;

/// What decides a constant (the four classes of `docs/29` §3, G1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A mathematical fact or an error budget gives the value, and a test proves it.
    Maths,
    /// Only image data can choose it; the objective is in `docs/29` §4.
    Fit,
    /// The owner's look (`docs/29` §6); not fitted.
    Design,
    /// A numeric safeguard; it never binds on a photograph.
    Guard,
}

/// Where the derivation of a constant stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The derivation is in the spec's table and a test proves it ([`Kind::Maths`]).
    Proved,
    /// A fit is planned and has not run: the task that runs it ([`Kind::Fit`]).
    Pending(&'static str),
    /// A fit ran: the task and what it found, in words (`"RC-07: within 2 % of the optimum"`).
    Fitted(&'static str),
    /// The owner's look ([`Kind::Design`]).
    Owner,
    /// A safeguard ([`Kind::Guard`]).
    Safeguard,
}

impl Status {
    /// Whether this status may go with `kind`: Maths is proved, Fit is pending or fitted, Design
    /// is the owner's and Guard is a safeguard.
    #[must_use]
    pub const fn fits(self, kind: Kind) -> bool {
        matches!(
            (kind, self),
            (Kind::Maths, Status::Proved)
                | (Kind::Fit, Status::Pending(_) | Status::Fitted(_))
                | (Kind::Design, Status::Owner)
                | (Kind::Guard, Status::Safeguard)
        )
    }
}

/// One registered constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constant {
    /// The row of `docs/29-raw-constants.md` §2.1 (`"A1"`, `"D7"`, `"O3"`).
    pub id: &'static str,
    /// The name of the field of `Tuning` (or of the structural constant) in snake case.
    pub name: &'static str,
    /// The value of `Tuning::DEFAULT`, as `f64` whatever its type in the struct.
    pub value: f64,
    /// What decides it.
    pub kind: Kind,
    /// Where its derivation stands.
    pub status: Status,
    /// The name of the constant in the generated WGSL block, when the shader reads it.
    pub wgsl: Option<&'static str>,
    /// Whether the value is specific enough that the same number written anywhere else in the
    /// package's source (outside the field's own definition and the generated block) is a stray
    /// copy. A generic value (`0.5`, `3.0`) is not: it has too many other uses to be searched for.
    pub distinct: bool,
}

impl Constant {
    /// A field of a `Tuning` with a value of `kind` and `status`, read by the CPU only.
    #[must_use]
    pub const fn cpu(
        id: &'static str,
        name: &'static str,
        value: f64,
        kind: Kind,
        status: Status,
    ) -> Constant {
        Constant {
            id,
            name,
            value,
            kind,
            status,
            wgsl: None,
            distinct: false,
        }
    }

    /// The same constant, also read by the shader under `wgsl`.
    #[must_use]
    pub const fn on_device(mut self, wgsl: &'static str) -> Constant {
        self.wgsl = Some(wgsl);
        self
    }

    /// The same constant with a value specific enough to search the source for.
    #[must_use]
    pub const fn distinct(mut self) -> Constant {
        self.distinct = true;
        self
    }
}

/// An `f32` as a WGSL literal: the shortest decimal that reads back as the same `f32` (Rust's
/// `{:?}`), which always has a `.` or an exponent. The literal is parsed by the shader compiler as an
/// abstract float (a double) and rounded once to `f32`; `tests/constants_registry.rs` proves that this
/// gives the bits of `v` for every constant of the registries.
#[must_use]
pub fn wgsl_f32(v: f32) -> String {
    format!("{v:?}")
}

/// The text of a generated block: one `const NAME: f32 = value;` line per constant, in the order
/// given, with a header line that names the source. Deterministic: the same constants give the same
/// bytes.
#[must_use]
pub fn wgsl_block(header: &str, constants: &[(&str, f32)]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "// {header}");
    for (name, value) in constants {
        let _ = writeln!(s, "const {name}: f32 = {};", wgsl_f32(*value));
    }
    s
}
