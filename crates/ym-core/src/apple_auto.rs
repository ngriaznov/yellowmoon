//! Apple Auto: on Macs where Apple Intelligence's on-device model can look at a photo, that model
//! grades it: it sets Auto's exposure, highlights, shadows, black level, contrast, warmth, tint and
//! saturation and the colours of the shadows and the highlights (`docs/19-apple-auto.md`). This module
//! is the platform-free half: the [`AutoAdvisor`] the app and `ym` hand to the core, the answer's
//! steps and the table that turns them into values, the instructions the model reads, the burst rule,
//! the states a photo goes through and a deterministic [`FakeAdvisor`] for tests. The Swift bridge
//! that talks to Apple's model is `ym-apple`.
//!
//! Nothing here persists (hard rule 4): answers live in the session's memory.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The version of the prompt and the answer's schema the bridge sends (`docs/19` section 4). Part
/// of "version" under hard rule 8: change it whenever the words or the schema change. Version 1
/// asked for four small corrections; version 2 asks the model to grade the photo (the owner's
/// direction of 2026-10-19, `docs/16`); version 3 has the owner's own default instructions of
/// 2026-10-20 (a finished base look with no style, `docs/19` section 4).
pub const PROMPT_VERSION: u32 = 3;

/// The version of [`TABLE`] (`docs/19` section 5). Change it whenever a value of the table changes.
pub const TABLE_VERSION: u32 = 2;

/// The instructions the model reads before each photo unless the person wrote their own in
/// Preferences (`docs/19` sections 2 and 4, prompt version 3; the `apple_auto_instructions`
/// preference). A test (`crates/ym-apple/src/source_tests.rs`) checks it against `docs/19`.
pub const DEFAULT_INSTRUCTIONS: &str = "You are a master colorist grading a raw, neutral camera file inside a RAW processor built on an ARRI color pipeline. Your goal is to transform this flat starting point into the best finished, professional photograph the source, light, subject and scene allow, honoring ARRI's signature color science. This is the base look. Add no style: the shadow colour and highlight colour must be none (the light is already correct), and no mood.\n1. Analyze & Respect the Scene\nPreserve the natural logic of the light. Night scenes stay dark; snow and high-key scenes stay bright; silhouettes stay dark; candlelight and golden hour stay warm. Keep skin tones natural, organic and true to life.\n2. Execute a Comprehensive Grade\nDo not rely on a single adjustment; a finished grade usually changes several controls, each chosen for what this picture needs:\n* Exposure & Tone: set the subject's brightness to guide the eye; brighten a subject that is too dark, hold one that is well lit, never darken a dark photo, and keep night scenes and silhouettes dark. Bring bright skies and whites down for smooth ARRI-style highlight roll-off. Bright sky, clouds, white clothes or lights: highlights darker for roll-off. Shadows brighter only where dark areas hide real detail; leave them when the shadows already show detail. Blacks deeper only if the picture looks hazy or milky; if the blacks are already deep and clean, leave them.\n* Contrast: deliberate dynamic range that fits the light: more for flat or hazy light, less for harsh light, with rich midtone detail.\n* Color Balance & Density: neutral whites, accurate warmth and tint, fixing casts. Saturation true and dense, neither boosted nor washed out.";

/// The longest instructions the model is given, in characters (Unicode scalar values): the
/// on-device model's context holds the picture and the answer's form too (`docs/19` section 2).
pub const INSTRUCTIONS_MAX_CHARS: usize = 2000;

/// The long edge of the picture the model sees, in pixels (`docs/19` section 3). Smaller photos are
/// not enlarged.
pub const PICTURE_LONG_EDGE_PX: usize = 768;

/// How long one request may take before it is abandoned, in milliseconds (`docs/19` section 6).
pub const REQUEST_TIMEOUT_MS: u32 = 30_000;

/// How many times a request that macOS asked to wait (`rate_limited`, `concurrent_requests`) is tried
/// again (`docs/19` section 6).
pub const RETRIES: u32 = 2;

/// The pause before a retry, in milliseconds.
pub const RETRY_PAUSE_MS: u64 = 2_000;

/// The longest scene description kept, in characters; it is shown only in reports.
pub const SCENE_MAX_CHARS: usize = 80;

/// The longest plan kept, in characters; reports and the Automatic card show it.
pub const PLAN_MAX_CHARS: usize = 160;

/// The range of the exposure step: `muchDarker` (−3) to `muchBrighter` (+3).
pub const EXPOSURE_STEPS: (i8, i8) = (-3, 3);

/// The range of the other numeric steps (highlights, shadows, blacks, contrast, warmth, tint,
/// saturation): −2 to +2.
pub const OTHER_STEPS: (i8, i8) = (-2, 2);

/// The range of a colour's strength: `subtle` (1) to `strong` (3).
pub const STRENGTHS: (i8, i8) = (1, 3);

/// The values of the steps (`docs/19` section 5, table version [`TABLE_VERSION`]). The owner tunes
/// them by eye; they are never fitted to Apple's answers.
pub struct Table {
    /// EV added to the rule's exposure, for the steps −3 … +3.
    pub exposure_ev: [f64; 7],
    /// The Auto Highlights (−100 … 100), for the steps −2 … +2.
    pub highlights: [f64; 5],
    /// The Auto Shadows (−100 … 100), for the steps −2 … +2.
    pub shadows: [f64; 5],
    /// The Auto Black (−100 … 100), for the model's blacks −2 … +2.
    pub black: [f64; 5],
    /// The Auto contrast (−100 … 100), for the steps −2 … +2.
    pub contrast: [f64; 5],
    /// How much warmer the picture is made, in mired, for the warmth steps −2 … +2: the user's white
    /// moves by minus this (a warmer picture is a higher Kelvin, a lower mired).
    pub warmer_mired: [f64; 5],
    /// The tint added to the scene's, in the tint's units (positive magenta), for the steps −2 … +2.
    pub tint: [f64; 5],
    /// The Auto saturation (−100 … 100), for the steps −2 … +2.
    pub saturation: [f64; 5],
    /// The radius of a colour in the colour-balance disc for the strengths subtle, medium, strong.
    pub strength_radius: [f64; 3],
    /// The direction `(a, b)` of each shadow colour, by [`ShadowColor::code`] (none is `(0, 0)`).
    pub shadow_directions: [[f64; 2]; 6],
    /// The direction `(a, b)` of each highlight colour, by [`HighlightColor::code`].
    pub highlight_directions: [[f64; 2]; 5],
}

/// `cos 30°` (`sqrt(3) / 2`), the long side of the colour directions.
const COS_30: f64 = 0.866;

/// The table in use (version [`TABLE_VERSION`]).
pub const TABLE: Table = Table {
    exposure_ev: [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0],
    highlights: [-40.0, -20.0, 0.0, 20.0, 40.0],
    shadows: [-30.0, -15.0, 0.0, 15.0, 30.0],
    black: [-30.0, -15.0, 0.0, 15.0, 30.0],
    contrast: [-30.0, -15.0, 0.0, 15.0, 30.0],
    warmer_mired: [-20.0, -10.0, 0.0, 10.0, 20.0],
    tint: [-10.0, -5.0, 0.0, 5.0, 10.0],
    saturation: [-30.0, -15.0, 0.0, 15.0, 30.0],
    strength_radius: [0.3, 0.55, 0.8],
    // none, teal, blue, green, amber, magenta.
    shadow_directions: [
        [0.0, 0.0],
        [-COS_30, -0.5],
        [-0.5, -COS_30],
        [-0.5, COS_30],
        [COS_30, 0.5],
        [0.5, -COS_30],
    ],
    // none, warm, gold, cool, pink.
    highlight_directions: [
        [0.0, 0.0],
        [COS_30, 0.5],
        [0.5, COS_30],
        [-0.5, -COS_30],
        [COS_30, -0.5],
    ],
};

/// The colour the model gives the shadows (`docs/19` section 4): a direction in the shadows'
/// colour-balance wheel, or none for a neutral grade. JSON: `"none"`, `"teal"`, …
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowColor {
    /// A neutral grade: the wheel at its centre.
    None,
    /// Teal, `(−0.866, −0.5)`.
    Teal,
    /// Blue, `(−0.5, −0.866)`.
    Blue,
    /// Green, `(−0.5, 0.866)`.
    Green,
    /// Amber, `(0.866, 0.5)`.
    Amber,
    /// Magenta, `(0.5, −0.866)`.
    Magenta,
}

impl ShadowColor {
    /// Every colour, in the order of their codes.
    pub const ALL: [Self; 6] = [
        Self::None,
        Self::Teal,
        Self::Blue,
        Self::Green,
        Self::Amber,
        Self::Magenta,
    ];

    /// The code the bridge writes (`ym_apple.h`): none 0, teal 1, blue 2, green 3, amber 4,
    /// magenta 5.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Teal => 1,
            Self::Blue => 2,
            Self::Green => 3,
            Self::Amber => 4,
            Self::Magenta => 5,
        }
    }

    /// The colour of a code, `None` for a code that is no colour.
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.code() == code)
    }

    /// The name in reports (`teal`), the JSON name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Teal => "teal",
            Self::Blue => "blue",
            Self::Green => "green",
            Self::Amber => "amber",
            Self::Magenta => "magenta",
        }
    }

    /// The unit direction `(a, b)` in the colour-balance disc ([`Table::shadow_directions`]).
    #[must_use]
    pub fn direction(self) -> [f64; 2] {
        let i = Self::ALL.iter().position(|c| *c == self).unwrap_or(0);
        TABLE.shadow_directions[i]
    }
}

/// The colour the model gives the highlights (`docs/19` section 4). JSON: `"none"`, `"warm"`, …
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HighlightColor {
    /// A neutral grade: the wheel at its centre.
    None,
    /// Warm, `(0.866, 0.5)`.
    Warm,
    /// Gold, `(0.5, 0.866)`.
    Gold,
    /// Cool, `(−0.5, −0.866)`.
    Cool,
    /// Pink, `(0.866, −0.5)`.
    Pink,
}

impl HighlightColor {
    /// Every colour, in the order of their codes.
    pub const ALL: [Self; 5] = [Self::None, Self::Warm, Self::Gold, Self::Cool, Self::Pink];

    /// The code the bridge writes (`ym_apple.h`): none 0, warm 1, gold 2, cool 3, pink 4.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Warm => 1,
            Self::Gold => 2,
            Self::Cool => 3,
            Self::Pink => 4,
        }
    }

    /// The colour of a code, `None` for a code that is no colour.
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.code() == code)
    }

    /// The name in reports (`warm`), the JSON name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Warm => "warm",
            Self::Gold => "gold",
            Self::Cool => "cool",
            Self::Pink => "pink",
        }
    }

    /// The unit direction `(a, b)` in the colour-balance disc ([`Table::highlight_directions`]).
    #[must_use]
    pub fn direction(self) -> [f64; 2] {
        let i = Self::ALL.iter().position(|c| *c == self).unwrap_or(0);
        TABLE.highlight_directions[i]
    }
}

/// The words of a colour's strength, 1 to 3.
#[must_use]
pub fn strength_name(strength: i8) -> &'static str {
    match strength {
        1 => "subtle",
        2 => "medium",
        3 => "strong",
        _ => "unknown",
    }
}

/// The point of a colour at a strength in its wheel: `radius · direction` ([`Table::strength_radius`]).
fn wheel_point(direction: [f64; 2], strength: i8) -> [f64; 2] {
    let i = usize::from((strength - STRENGTHS.0).unsigned_abs());
    let r = TABLE.strength_radius[i];
    [r * direction[0], r * direction[1]]
}

/// The instructions the model is given: the person's own when they wrote some (the
/// `apple_auto_instructions` preference, or `ym --apple-auto-instructions`), else
/// [`DEFAULT_INSTRUCTIONS`]. Text that is empty or only white space is no instructions.
#[must_use]
pub fn effective_instructions(own: &str) -> &str {
    if own.trim().is_empty() {
        DEFAULT_INSTRUCTIONS
    } else {
        own
    }
}

/// Checks the person's instructions (`docs/19` section 2): at most [`INSTRUCTIONS_MAX_CHARS`]
/// characters. The error is the message to show.
///
/// # Errors
///
/// The message, for text that is too long.
pub fn check_instructions(text: &str) -> Result<(), String> {
    let n = text.chars().count();
    if n > INSTRUCTIONS_MAX_CHARS {
        return Err(format!(
            "The instructions for Apple Intelligence are {n} characters long. At most {INSTRUCTIONS_MAX_CHARS} fit, because the model must also see the photo."
        ));
    }
    Ok(())
}

/// Whether the model read the default instructions or the person's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstructionsKind {
    /// [`DEFAULT_INSTRUCTIONS`].
    #[default]
    Default,
    /// The person's own.
    Edited,
}

/// Which instructions the model read (`docs/19` sections 7 to 9): default or edited, and the first
/// 12 hex digits of the SHA-256 of their UTF-8 text, so two answers can be told apart without the
/// text. JSON (flattened into [`AdvisorInfo`]): `"instructions": "default" | "edited",
/// "instructions_sha256": "…"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InstructionsId {
    /// Default or edited.
    pub instructions: InstructionsKind,
    /// The first 12 hex digits of the SHA-256 of the text.
    pub instructions_sha256: String,
}

impl InstructionsId {
    /// The id of the instructions `text` (the text the model reads, [`effective_instructions`]).
    #[must_use]
    pub fn of(text: &str) -> Self {
        let digest = Sha256::digest(text.as_bytes());
        let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
        Self {
            instructions: if text == DEFAULT_INSTRUCTIONS {
                InstructionsKind::Default
            } else {
                InstructionsKind::Edited
            },
            instructions_sha256: hex,
        }
    }
}

impl Default for InstructionsId {
    /// The id of [`DEFAULT_INSTRUCTIONS`].
    fn default() -> Self {
        Self::of(DEFAULT_INSTRUCTIONS)
    }
}

/// Why Apple Auto is not available on this computer (`docs/19` section 1), in the order the
/// conditions are checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
    /// Not a Mac.
    NotAMac,
    /// A Mac with an Intel processor.
    IntelMac,
    /// This binary was built without the macOS 27 SDK, so it has no bridge.
    BuiltWithoutSupport,
    /// macOS older than 27.
    MacosTooOld,
    /// The Mac cannot run Apple Intelligence.
    DeviceNotEligible,
    /// Apple Intelligence is off in System Settings.
    AppleIntelligenceOff,
    /// The model is still downloading, or not ready for another system reason.
    ModelNotReady,
    /// Any other unavailability Apple reports.
    Unavailable,
    /// The model on this Mac cannot take pictures.
    NoImageInput,
}

impl UnavailableReason {
    /// The reason in words, for `ym` and the logs ("Apple Auto is not available on this computer
    /// (`<words>`).").
    #[must_use]
    pub fn words(self) -> &'static str {
        match self {
            Self::NotAMac => "this is not a Mac",
            Self::IntelMac => "this Mac has an Intel processor",
            Self::BuiltWithoutSupport => "this build of Yellowmoon was made without it",
            Self::MacosTooOld => "it needs macOS 27 or later",
            Self::DeviceNotEligible => "this Mac can't run Apple Intelligence",
            Self::AppleIntelligenceOff => "Apple Intelligence is off in System Settings",
            Self::ModelNotReady => "Apple Intelligence is still downloading its model",
            Self::Unavailable => "Apple Intelligence is not available",
            Self::NoImageInput => "Apple's model on this Mac can't look at pictures",
        }
    }
}

impl fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.words())
    }
}

/// Why one request failed (`docs/19` sections 6 and 8). A failed photo uses the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// Apple Auto stopped being available between the check and the request.
    Unavailable,
    /// macOS asked the app to wait (after the retries).
    RateLimited,
    /// No answer within [`REQUEST_TIMEOUT_MS`].
    Timeout,
    /// The model refused.
    Refused,
    /// Apple's safety check stopped the request.
    Guardrail,
    /// The model does not support the language or locale.
    UnsupportedLanguage,
    /// The request did not fit the model's context.
    ContextTooLarge,
    /// The model's assets were not there.
    AssetsUnavailable,
    /// Another request was running (after the retries).
    ConcurrentRequests,
    /// The answer could not be read, or a step was out of its range.
    BadAnswer,
    /// Anything else; the message says what.
    Other,
}

impl FailureKind {
    /// The failure in words, for `ym info` and the logs.
    #[must_use]
    pub fn words(self) -> &'static str {
        match self {
            Self::Unavailable => "Apple Intelligence stopped being available",
            Self::RateLimited => "macOS asked Yellowmoon to wait",
            Self::Timeout => "no answer within 30 seconds",
            Self::Refused => "the model refused",
            Self::Guardrail => "Apple's safety check stopped it",
            Self::UnsupportedLanguage => "the model does not support this language",
            Self::ContextTooLarge => "the request was too large for the model",
            Self::AssetsUnavailable => "the model's files were not available",
            Self::ConcurrentRequests => "another request was running",
            Self::BadAnswer => "the answer could not be read",
            Self::Other => "an error",
        }
    }

    /// Whether the request is tried again ([`RETRIES`] times, [`RETRY_PAUSE_MS`] apart).
    #[must_use]
    pub fn retried(self) -> bool {
        matches!(self, Self::RateLimited | Self::ConcurrentRequests)
    }
}

/// Which model answers, and with which instructions, for reports and `ym:AppleAuto`. JSON:
/// `{ "model", "os", "instructions", "instructions_sha256" }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisorInfo {
    /// The model variant's display name (`AFM 3 Core`).
    pub model: String,
    /// The macOS version and build (`27.0.1 (27A404)`).
    pub os: String,
    /// The instructions it read ([`DEFAULT_INSTRUCTIONS`] until the session says otherwise).
    #[serde(flatten)]
    pub instructions: InstructionsId,
}

impl AdvisorInfo {
    /// A model on a macOS, with the default instructions (the session sets the ones it sends,
    /// [`AdvisorInfo::with_instructions`]).
    #[must_use]
    pub fn new(model: impl Into<String>, os: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            os: os.into(),
            instructions: InstructionsId::default(),
        }
    }

    /// The same model with the instructions `text` (the text the model reads).
    #[must_use]
    pub fn with_instructions(mut self, text: &str) -> Self {
        self.instructions = InstructionsId::of(text);
        self
    }

    /// The version string of `docs/19` section 7:
    /// `Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2`, with
    /// ` · edited instructions (<12 hex digits>)` when the person wrote their own.
    #[must_use]
    pub fn version_string(&self) -> String {
        let mut text = format!(
            "Apple Intelligence · {} · macOS {} · prompt {PROMPT_VERSION} · table {TABLE_VERSION}",
            self.model, self.os
        );
        if self.instructions.instructions == InstructionsKind::Edited {
            text.push_str(&format!(
                " · edited instructions ({})",
                self.instructions.instructions_sha256
            ));
        }
        text
    }
}

/// Whether an advisor can answer now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// It can; which model answers.
    Available(AdvisorInfo),
    /// It cannot, and why.
    Unavailable(UnavailableReason),
}

impl Availability {
    /// Whether it can answer.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }
}

/// The picture the model sees (`docs/19` section 3): 8-bit sRGB, 3 bytes per pixel, rows top to
/// bottom, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisorPicture {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` bytes.
    pub rgb: Vec<u8>,
}

/// The steps of an answer (`docs/19` sections 4 and 5), each `None` when the model left the field
/// empty (nothing is set for that control): exposure −3 … +3, highlights, shadows, blacks,
/// contrast, white (the model's warmth), tint and saturation −2 … +2, 0 for keep; the colours of
/// the shadows and the highlights with their strengths 1 … 3. A strength is kept only with a colour
/// other than none ([`AutoSteps::normalised`]). JSON: every key, `null` when not set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AutoSteps {
    /// `muchDarker` (−3) … `muchBrighter` (+3).
    pub exposure: Option<i8>,
    /// `muchDarker` (−2) … `muchBrighter` (+2): the brightest tones.
    pub highlights: Option<i8>,
    /// `muchDarker` (−2) … `muchBrighter` (+2): the dark tones.
    pub shadows: Option<i8>,
    /// `muchDeeper` (−2) … `muchLifted` (+2): the black level.
    pub blacks: Option<i8>,
    /// `muchLess` (−2) … `muchMore` (+2).
    pub contrast: Option<i8>,
    /// The model's warmth, `muchCooler` (−2) … `muchWarmer` (+2): the nudge of the white.
    pub white: Option<i8>,
    /// `muchGreener` (−2) … `muchMoreMagenta` (+2): the nudge of the tint.
    pub tint: Option<i8>,
    /// `muchLess` (−2) … `muchMore` (+2).
    pub saturation: Option<i8>,
    /// The colour of the shadows.
    pub shadow_color: Option<ShadowColor>,
    /// Its strength, `subtle` (1) … `strong` (3).
    pub shadow_color_strength: Option<i8>,
    /// The colour of the highlights.
    pub highlight_color: Option<HighlightColor>,
    /// Its strength, `subtle` (1) … `strong` (3).
    pub highlight_color_strength: Option<i8>,
}

impl AutoSteps {
    /// Keep on every numeric step and no colour: an answer that changes nothing but sets every
    /// field.
    #[must_use]
    pub fn keep() -> Self {
        Self {
            exposure: Some(0),
            highlights: Some(0),
            shadows: Some(0),
            blacks: Some(0),
            contrast: Some(0),
            white: Some(0),
            tint: Some(0),
            saturation: Some(0),
            shadow_color: Some(ShadowColor::None),
            shadow_color_strength: None,
            highlight_color: Some(HighlightColor::None),
            highlight_color_strength: None,
        }
    }

    /// The numeric steps with their names and ranges, in the order of the table (`docs/19` section
    /// 5): exposure, highlights, shadows, blacks, contrast, white, tint, saturation.
    #[must_use]
    pub fn numeric(&self) -> [(&'static str, Option<i8>, (i8, i8)); 8] {
        [
            ("exposure", self.exposure, EXPOSURE_STEPS),
            ("highlights", self.highlights, OTHER_STEPS),
            ("shadows", self.shadows, OTHER_STEPS),
            ("blacks", self.blacks, OTHER_STEPS),
            ("contrast", self.contrast, OTHER_STEPS),
            ("white", self.white, OTHER_STEPS),
            ("tint", self.tint, OTHER_STEPS),
            ("saturation", self.saturation, OTHER_STEPS),
        ]
    }

    /// Whether every step that is set is inside its range.
    #[must_use]
    pub fn in_range(&self) -> bool {
        let inside = |v: Option<i8>, (lo, hi): (i8, i8)| v.is_none_or(|v| (lo..=hi).contains(&v));
        self.numeric()
            .iter()
            .all(|(_, v, range)| inside(*v, *range))
            && inside(self.shadow_color_strength, STRENGTHS)
            && inside(self.highlight_color_strength, STRENGTHS)
    }

    /// Whether the model set at least one value (a colour of none counts; a strength alone does not,
    /// [`AutoSteps::normalised`]).
    #[must_use]
    pub fn any_set(&self) -> bool {
        let n = self.normalised();
        n.numeric().iter().any(|(_, v, _)| v.is_some())
            || n.shadow_color.is_some()
            || n.highlight_color.is_some()
    }

    /// The same steps with each strength kept only when its colour is set and not none: a strength
    /// is ignored without a colour (`docs/19` section 4).
    #[must_use]
    pub fn normalised(mut self) -> Self {
        if !matches!(self.shadow_color, Some(c) if c != ShadowColor::None) {
            self.shadow_color_strength = None;
        }
        if !matches!(self.highlight_color, Some(c) if c != HighlightColor::None) {
            self.highlight_color_strength = None;
        }
        self
    }

    /// The point the shadows wheel takes (`(0, 0)` for none, or for a colour without a strength:
    /// nothing set), and whether the answer set it.
    fn shadow_point(&self) -> ([f64; 2], bool) {
        match (self.shadow_color, self.shadow_color_strength) {
            (Some(ShadowColor::None), _) => ([0.0, 0.0], true),
            (Some(c), Some(s)) => (wheel_point(c.direction(), s), true),
            _ => ([0.0, 0.0], false),
        }
    }

    /// The point the highlights wheel takes, and whether the answer set it.
    fn highlight_point(&self) -> ([f64; 2], bool) {
        match (self.highlight_color, self.highlight_color_strength) {
            (Some(HighlightColor::None), _) => ([0.0, 0.0], true),
            (Some(c), Some(s)) => (wheel_point(c.direction(), s), true),
            _ => ([0.0, 0.0], false),
        }
    }

    /// Whether the answer set the shadows wheel (a colour, none included, with a strength when it is
    /// not none).
    #[must_use]
    pub fn sets_shadow_color(&self) -> bool {
        self.shadow_point().1
    }

    /// Whether the answer set the highlights wheel.
    #[must_use]
    pub fn sets_highlight_color(&self) -> bool {
        self.highlight_point().1
    }

    /// The values of the steps through [`TABLE`], or `None` when a step is out of its range. A step
    /// that is not set gives the value Auto has without Apple Auto (0, `(0, 0)`, no nudge).
    #[must_use]
    pub fn values(&self) -> Option<AppleValues> {
        if !self.in_range() {
            return None;
        }
        let other = |v: Option<i8>, table: &[f64; 5]| {
            v.map_or(0.0, |v| {
                table[usize::from((v - OTHER_STEPS.0).unsigned_abs())]
            })
        };
        let exposure_ev = self.exposure.map_or(0.0, |v| {
            TABLE.exposure_ev[usize::from((v - EXPOSURE_STEPS.0).unsigned_abs())]
        });
        Some(AppleValues {
            exposure_ev,
            highlights: other(self.highlights, &TABLE.highlights),
            shadows: other(self.shadows, &TABLE.shadows),
            black: other(self.blacks, &TABLE.black),
            contrast: other(self.contrast, &TABLE.contrast),
            // `+ 0.0` turns the negative zero of keep into 0.
            white_shift_mired: -other(self.white, &TABLE.warmer_mired) + 0.0,
            tint_shift: other(self.tint, &TABLE.tint),
            saturation: other(self.saturation, &TABLE.saturation),
            balance_shadows: self.shadow_point().0,
            balance_highlights: self.highlight_point().0,
        })
    }
}

/// What an answer does to Auto (`docs/19` section 5). The default, all zeros, is Auto without
/// Apple Auto.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct AppleValues {
    /// EV added to the rule's exposure.
    pub exposure_ev: f64,
    /// The Auto Highlights.
    pub highlights: f64,
    /// The Auto Shadows.
    pub shadows: f64,
    /// The Auto Black.
    pub black: f64,
    /// The Auto contrast.
    pub contrast: f64,
    /// Mired added to the user's white in Auto (negative: a higher Kelvin, a warmer picture).
    pub white_shift_mired: f64,
    /// Tint added to the scene's in Auto (positive: more magenta).
    pub tint_shift: f64,
    /// The Auto saturation.
    pub saturation: f64,
    /// The Auto point of the shadows colour-balance wheel.
    pub balance_shadows: [f64; 2],
    /// The Auto point of the highlights colour-balance wheel.
    pub balance_highlights: [f64; 2],
}

/// One answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisorAnswer {
    /// The steps; a field the model left empty is `None`.
    pub steps: AutoSteps,
    /// What the photo shows, in a few words (at most [`SCENE_MAX_CHARS`] characters).
    pub scene: String,
    /// The look the model gives the photo and how, in a few words (at most [`PLAN_MAX_CHARS`]
    /// characters).
    #[serde(default)]
    pub plan: String,
}

/// `text` trimmed and cut to `max` characters.
fn cut(text: &str, max: usize) -> String {
    text.trim()
        .chars()
        .take(max)
        .collect::<String>()
        .trim_end()
        .to_owned()
}

impl AdvisorAnswer {
    /// The answer with its scene and plan cut to [`SCENE_MAX_CHARS`] and [`PLAN_MAX_CHARS`]
    /// characters and trimmed, and its strengths kept only with a colour ([`AutoSteps::normalised`]);
    /// a [`FailureKind::BadAnswer`] error when a step is out of its range or no value is set.
    pub fn checked(mut self) -> Result<Self, AdvisorError> {
        if !self.steps.in_range() {
            return Err(AdvisorError::new(
                FailureKind::BadAnswer,
                format!("a step is out of its range: {:?}", self.steps),
            ));
        }
        self.steps = self.steps.normalised();
        if !self.steps.any_set() {
            return Err(AdvisorError::new(
                FailureKind::BadAnswer,
                "the model filled no value",
            ));
        }
        self.scene = cut(&self.scene, SCENE_MAX_CHARS);
        self.plan = cut(&self.plan, PLAN_MAX_CHARS);
        Ok(self)
    }
}

/// A failed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisorError {
    /// What kind of failure.
    pub kind: FailureKind,
    /// Apple's message, when there is one.
    pub message: String,
}

impl fmt::Display for AdvisorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind.words())?;
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for AdvisorError {}

impl AdvisorError {
    /// A failure with a message.
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Something that looks at a photo and grades it: Apple's on-device model through `ym-apple`, or
/// [`FakeAdvisor`] in tests. The app and `ym` hand one to the core; every other caller hands none,
/// which is the same as one that is unavailable.
pub trait AutoAdvisor: Send + Sync {
    /// Whether it can answer now. Cheap; called at launch, when Preferences opens, when a session
    /// opens and before each export.
    fn availability(&self) -> Availability;

    /// Asks about one picture with `instructions` (the text the model reads before the photo,
    /// [`effective_instructions`]: never empty). Blocks until the answer, a failure or
    /// [`REQUEST_TIMEOUT_MS`]; called from one worker thread at a time, never from the main thread.
    /// Retries are the caller's.
    fn ask(
        &self,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError>;
}

/// Where a photo stands with Apple Auto (`docs/19` sections 6 and 12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AppleAuto {
    /// The preference is off (or no advisor was given): Auto is the rule.
    Off,
    /// Apple Auto is not available on this computer: Auto is the rule.
    Unavailable {
        /// Why.
        reason: UnavailableReason,
    },
    /// Asked, or waiting for its turn: Auto is the rule until the answer.
    Pending,
    /// Answered: Auto takes the answer's values.
    Answered {
        /// The answer.
        answer: AdvisorAnswer,
        /// Which model answered.
        info: AdvisorInfo,
    },
    /// The request failed: Auto is the rule.
    Failed {
        /// Why.
        kind: FailureKind,
        /// Apple's message, when there is one.
        message: String,
        /// Which model was asked.
        info: AdvisorInfo,
    },
}

impl AppleAuto {
    /// The steps of an answer, `None` in every other state.
    #[must_use]
    pub fn steps(&self) -> Option<AutoSteps> {
        match self {
            Self::Answered { answer, .. } => Some(answer.steps),
            _ => None,
        }
    }

    /// Whether the state will not change without a new request (answered, failed, off or
    /// unavailable): what an export waits for.
    #[must_use]
    pub fn is_settled(&self) -> bool {
        !matches!(self, Self::Pending)
    }

    /// Whether Apple Auto acts on the photo: pending, answered or failed (not off or unavailable).
    /// Exactly when the Automatic card shows its Contrast and Saturation rows (`docs/11`), and when
    /// Re-run puts contrast and saturation back to Auto (`docs/19` section 5).
    #[must_use]
    pub fn acts(&self) -> bool {
        matches!(
            self,
            Self::Pending | Self::Answered { .. } | Self::Failed { .. }
        )
    }
}

impl AppleAuto {
    /// The `ym:AppleAuto` property of an export (`docs/19` section 9): compact JSON with sorted keys,
    /// written only when Apple Auto was on for the photo, so `None` for [`AppleAuto::Off`],
    /// [`AppleAuto::Unavailable`] and [`AppleAuto::Pending`] (an export waits for the answer; only a
    /// photo whose session closed under its export can still be pending). A step the model left
    /// empty is `null`.
    ///
    /// ```text
    /// {"instructions":"default","instructions_sha256":"…","model":"AFM 3 Core","os":"27.0.1 (27A404)",
    ///  "plan":"…","prompt":3,"result":"answered","scene":"…","steps":{"blacks":null,"contrast":1,…},"table":2}
    /// {"instructions":"default","instructions_sha256":"…","model":"…","os":"…","prompt":3,
    ///  "reason":"rate_limited","result":"failed","table":2}
    /// ```
    #[must_use]
    pub fn xmp_json(&self) -> Option<String> {
        use serde_json::{Value, json};
        use std::collections::BTreeMap;
        let mut map: BTreeMap<&'static str, Value> = BTreeMap::new();
        let info = match self {
            Self::Off | Self::Unavailable { .. } | Self::Pending => return None,
            Self::Answered { answer, info } => {
                map.insert("result", json!("answered"));
                map.insert("scene", json!(answer.scene));
                map.insert("plan", json!(answer.plan));
                map.insert("steps", steps_json(&answer.steps));
                info
            }
            Self::Failed { kind, info, .. } => {
                map.insert("result", json!("failed"));
                map.insert("reason", serde_json::to_value(kind).ok()?);
                info
            }
        };
        map.insert("model", json!(info.model));
        map.insert("os", json!(info.os));
        map.insert(
            "instructions",
            serde_json::to_value(info.instructions.instructions).ok()?,
        );
        map.insert(
            "instructions_sha256",
            json!(info.instructions.instructions_sha256),
        );
        map.insert("prompt", json!(PROMPT_VERSION));
        map.insert("table", json!(TABLE_VERSION));
        serde_json::to_string(&map).ok()
    }
}

/// The steps as a JSON object with sorted keys, `null` for a step the model left empty: what
/// `ym:AppleAuto` and `ym --json` write (`docs/19` sections 8 and 9).
#[must_use]
pub fn steps_json(steps: &AutoSteps) -> serde_json::Value {
    let value = serde_json::to_value(steps).unwrap_or(serde_json::Value::Null);
    // `serde_json::Map` keeps its keys sorted (no `preserve_order` feature in this workspace); the
    // round trip through a `BTreeMap` makes sure of it whatever the features.
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        other => other,
    }
}

/// How a run decides whether Apple Auto is used (`docs/19` sections 2 and 8): the app follows the
/// preference; `ym --apple-auto on|off` overrides it for one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleAutoMode {
    /// The `apple_auto` preference decides (the app, and `ym` without `--apple-auto`).
    #[default]
    FromPreference,
    /// On for this run, whatever the preference says (still only where it is available).
    On,
    /// Off for this run: Auto is the rule.
    Off,
}

/// Whether Apple Auto acts for a session now, and if not, why (`docs/19` sections 1 and 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activity {
    /// It acts: photos are asked, answers change Auto. Which model answers.
    Active(AdvisorInfo),
    /// Off: no advisor was given, `--apple-auto off`, or the preference is off.
    Off,
    /// Not available on this computer.
    Unavailable(UnavailableReason),
}

impl Activity {
    /// Decides the activity. In this order: no advisor is off (tests, the golden tool); a run with
    /// `--apple-auto off` is off; where Apple Auto is not available it is unavailable whatever the
    /// preference says (the preference is kept and ignored there, `docs/19` section 2); the
    /// preference off is off (unless the run says `on`); else it acts. `availability` is the last
    /// one read; none read yet counts as unavailable.
    #[must_use]
    pub fn decide(
        has_advisor: bool,
        mode: AppleAutoMode,
        preference: bool,
        availability: Option<&Availability>,
    ) -> Self {
        if !has_advisor || mode == AppleAutoMode::Off {
            return Self::Off;
        }
        match availability {
            None => Self::Unavailable(UnavailableReason::Unavailable),
            Some(Availability::Unavailable(reason)) => Self::Unavailable(*reason),
            Some(Availability::Available(info)) => {
                if mode == AppleAutoMode::FromPreference && !preference {
                    Self::Off
                } else {
                    Self::Active(info.clone())
                }
            }
        }
    }

    /// Whether Apple Auto acts.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    /// The state of a photo under this activity: its own result while Apple Auto acts (pending until
    /// there is one), else off or unavailable. `result` is [`AppleAuto::Answered`] or
    /// [`AppleAuto::Failed`] when the photo was asked; answers are kept while Apple Auto is off.
    #[must_use]
    pub fn photo_state(&self, result: Option<&AppleAuto>) -> AppleAuto {
        match self {
            Self::Off => AppleAuto::Off,
            Self::Unavailable(reason) => AppleAuto::Unavailable { reason: *reason },
            Self::Active(_) => result.cloned().unwrap_or(AppleAuto::Pending),
        }
    }
}

/// What the app's Preferences and `ym doctor` show about Apple Auto (`docs/19` sections 1, 2 and 8).
/// JSON, snake_case: `{ "available": true, "reason": null, "info": { "model": "…", "os": "…", … },
/// "preference": true, "mode": "from_preference", "active": true, "prompt": 3, "table": 2,
/// "instructions": "default", "instructions_sha256": "…", "own_instructions": "",
/// "default_instructions": "…", "instructions_max_chars": 2000 }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleAutoStatus {
    /// Whether Apple Auto is available on this computer now (read again for this answer). False when
    /// the core was given no advisor.
    pub available: bool,
    /// Why it is not available; `None` when it is, and when the core was given no advisor.
    pub reason: Option<UnavailableReason>,
    /// Which model answers, when it is available.
    pub info: Option<AdvisorInfo>,
    /// The `apple_auto` preference.
    pub preference: bool,
    /// How this run decides ([`AppleAutoMode`]).
    pub mode: AppleAutoMode,
    /// Whether Apple Auto acts for the session now ([`Activity::decide`]).
    pub active: bool,
    /// [`PROMPT_VERSION`].
    pub prompt: u32,
    /// [`TABLE_VERSION`].
    pub table: u32,
    /// Which instructions the model reads: default or edited, and their short digest.
    #[serde(flatten)]
    pub instructions: InstructionsId,
    /// The person's own instructions, `""` when they use the default (the
    /// `apple_auto_instructions` preference, or `ym --apple-auto-instructions`).
    pub own_instructions: String,
    /// [`DEFAULT_INSTRUCTIONS`], for Preferences' text field and its Reset to default.
    pub default_instructions: String,
    /// [`INSTRUCTIONS_MAX_CHARS`].
    pub instructions_max_chars: usize,
}

/// The median of the values that are set, the lower of the two middle ones for an even count;
/// `None` when none is set.
fn median_of(values: impl Iterator<Item = Option<i8>>) -> Option<i8> {
    let mut set: Vec<i8> = values.flatten().collect();
    if set.is_empty() {
        return None;
    }
    set.sort_unstable();
    Some(set[(set.len() - 1) / 2])
}

/// The colour most of the frames chose (ties: the colour of the earliest frame among the tied
/// ones) and the median strength of the frames that chose it; `(None, None)` when no frame set a
/// colour.
fn share_color<C: Copy + PartialEq>(frames: &[(Option<C>, Option<i8>)]) -> (Option<C>, Option<i8>) {
    let mut best: Option<(C, usize)> = None;
    for (color, _) in frames {
        let Some(color) = *color else { continue };
        let count = frames.iter().filter(|(c, _)| *c == Some(color)).count();
        // Frames are in burst order: the first colour to reach the highest count wins a tie.
        if best.is_none_or(|(_, n)| count > n) {
            best = Some((color, count));
        }
    }
    let Some((color, _)) = best else {
        return (None, None);
    };
    let strength = median_of(
        frames
            .iter()
            .filter(|(c, _)| *c == Some(color))
            .map(|(_, s)| *s),
    );
    (Some(color), strength)
}

/// The steps a burst shares (`docs/19` section 5). Numeric steps: field by field, the median of the
/// frames that set that field, the lower of the two middle values for an even count; not set when
/// no frame set it. Colours: the colour most frames chose (ties: the colour of the earliest frame,
/// in burst order, among the tied ones), with the median strength of the frames that chose it.
/// Frames without an answer (`None`) are left out; `None` when no frame answered. `frames` are in
/// burst order.
#[must_use]
pub fn share_burst_steps(frames: &[Option<AutoSteps>]) -> Option<AutoSteps> {
    let answered: Vec<AutoSteps> = frames.iter().flatten().map(|s| s.normalised()).collect();
    if answered.is_empty() {
        return None;
    }
    let median = |field: fn(&AutoSteps) -> Option<i8>| median_of(answered.iter().map(field));
    let (shadow_color, shadow_color_strength) = share_color(
        &answered
            .iter()
            .map(|s| (s.shadow_color, s.shadow_color_strength))
            .collect::<Vec<_>>(),
    );
    let (highlight_color, highlight_color_strength) = share_color(
        &answered
            .iter()
            .map(|s| (s.highlight_color, s.highlight_color_strength))
            .collect::<Vec<_>>(),
    );
    Some(
        AutoSteps {
            exposure: median(|s| s.exposure),
            highlights: median(|s| s.highlights),
            shadows: median(|s| s.shadows),
            blacks: median(|s| s.blacks),
            contrast: median(|s| s.contrast),
            white: median(|s| s.white),
            tint: median(|s| s.tint),
            saturation: median(|s| s.saturation),
            shadow_color,
            shadow_color_strength,
            highlight_color,
            highlight_color_strength,
        }
        .normalised(),
    )
}

/// A deterministic advisor for tests (`docs/19` section 12), always available. From the picture's
/// mean sRGB luma `L` (0…1, Rec.709 weights on the 8-bit values) and its standard deviation `S`:
/// exposure +1 when `L` < 0.30, −1 when `L` > 0.65, else 0; highlights −1 when `L` > 0.5, else not
/// set; shadows +1 when `L` < 0.30, else 0; blacks not set; contrast +1 when `S` < 0.15, else 0;
/// white 0; tint not set; saturation 0; shadow colour teal, subtle, when `S` < 0.15, else none;
/// highlight colour not set; scene "Test picture"; plan "Test plan: " and the first word of the
/// instructions it was given, so tests can see which instructions reached it. The app and `ym` use
/// it when `YELLOWMOON_TEST_APPLE_AUTO=fake` is set.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeAdvisor;

impl FakeAdvisor {
    /// The model name it reports.
    pub const MODEL: &'static str = "Test advisor";
    /// The operating system it reports.
    pub const OS: &'static str = "test";

    /// Its [`AdvisorInfo`].
    #[must_use]
    pub fn info() -> AdvisorInfo {
        AdvisorInfo::new(Self::MODEL, Self::OS)
    }

    /// The plan it answers for `instructions`: "Test plan: " and their first word.
    #[must_use]
    pub fn plan_for(instructions: &str) -> String {
        format!(
            "Test plan: {}",
            instructions.split_whitespace().next().unwrap_or("")
        )
    }
}

impl AutoAdvisor for FakeAdvisor {
    fn availability(&self) -> Availability {
        Availability::Available(Self::info())
    }

    fn ask(
        &self,
        picture: &AdvisorPicture,
        instructions: &str,
    ) -> Result<AdvisorAnswer, AdvisorError> {
        let pixels = picture.width * picture.height;
        if pixels == 0 || picture.rgb.len() != pixels * 3 {
            return Err(AdvisorError::new(
                FailureKind::Other,
                format!(
                    "a {}×{} picture with {} bytes",
                    picture.width,
                    picture.height,
                    picture.rgb.len()
                ),
            ));
        }
        let lumas = picture.rgb.chunks_exact(3).map(|p| {
            (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 255.0
        });
        let n = pixels as f64;
        let (sum, sum_sq) = lumas.fold((0.0, 0.0), |(s, q), l| (s + l, q + l * l));
        let mean = sum / n;
        let sd = (sum_sq / n - mean * mean).max(0.0).sqrt();
        let exposure = if mean < 0.30 {
            1
        } else if mean > 0.65 {
            -1
        } else {
            0
        };
        let flat = sd < 0.15;
        Ok(AdvisorAnswer {
            steps: AutoSteps {
                exposure: Some(exposure),
                highlights: (mean > 0.5).then_some(-1),
                shadows: Some(i8::from(mean < 0.30)),
                blacks: None,
                contrast: Some(i8::from(flat)),
                white: Some(0),
                tint: None,
                saturation: Some(0),
                shadow_color: Some(if flat {
                    ShadowColor::Teal
                } else {
                    ShadowColor::None
                }),
                shadow_color_strength: flat.then_some(1),
                highlight_color: None,
                highlight_color_strength: None,
            },
            scene: "Test picture".to_owned(),
            plan: Self::plan_for(instructions),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steps with every numeric field set, colours none.
    fn steps(exposure: i8, white: i8, contrast: i8, saturation: i8) -> AutoSteps {
        AutoSteps {
            exposure: Some(exposure),
            white: Some(white),
            contrast: Some(contrast),
            saturation: Some(saturation),
            ..AutoSteps::keep()
        }
    }

    #[test]
    fn keep_and_nothing_set_are_both_the_rule() {
        assert_eq!(AutoSteps::keep().values(), Some(AppleValues::default()));
        assert_eq!(AutoSteps::default().values(), Some(AppleValues::default()));
        assert!(AutoSteps::keep().any_set());
        assert!(!AutoSteps::default().any_set());
    }

    #[test]
    fn the_table_maps_every_step() {
        let all = |n: i8, e: i8| AutoSteps {
            exposure: Some(e),
            highlights: Some(n),
            shadows: Some(n),
            blacks: Some(n),
            contrast: Some(n),
            white: Some(n),
            tint: Some(n),
            saturation: Some(n),
            ..AutoSteps::keep()
        };
        let v = all(-2, -3).values().unwrap();
        assert_eq!(
            (
                v.exposure_ev,
                v.highlights,
                v.shadows,
                v.black,
                v.contrast,
                v.white_shift_mired,
                v.tint_shift,
                v.saturation
            ),
            (-1.0, -40.0, -30.0, -30.0, -30.0, 20.0, -10.0, -30.0)
        );
        let v = all(2, 3).values().unwrap();
        assert_eq!(
            (
                v.exposure_ev,
                v.highlights,
                v.shadows,
                v.black,
                v.contrast,
                v.white_shift_mired,
                v.tint_shift,
                v.saturation
            ),
            (1.0, 40.0, 30.0, 30.0, 30.0, -20.0, 10.0, 30.0)
        );
        let v = all(1, -1).values().unwrap();
        assert_eq!(
            (
                v.exposure_ev,
                v.highlights,
                v.shadows,
                v.black,
                v.contrast,
                v.white_shift_mired,
                v.tint_shift,
                v.saturation
            ),
            (-0.3, 20.0, 15.0, 15.0, 15.0, -10.0, 5.0, 15.0)
        );
        let v = all(-1, 2).values().unwrap();
        assert_eq!(
            (
                v.exposure_ev,
                v.highlights,
                v.tint_shift,
                v.white_shift_mired
            ),
            (0.6, -20.0, -5.0, 10.0)
        );
    }

    #[test]
    fn a_step_that_is_not_set_is_the_value_without_apple_auto() {
        let only_contrast = AutoSteps {
            contrast: Some(2),
            ..AutoSteps::default()
        };
        assert_eq!(
            only_contrast.values(),
            Some(AppleValues {
                contrast: 30.0,
                ..AppleValues::default()
            })
        );
        // A colour without a strength sets nothing for its wheel.
        let colour_alone = AutoSteps {
            shadow_color: Some(ShadowColor::Teal),
            highlight_color: Some(HighlightColor::Gold),
            ..AutoSteps::default()
        };
        assert_eq!(colour_alone.values(), Some(AppleValues::default()));
        assert!(!colour_alone.sets_shadow_color() && !colour_alone.sets_highlight_color());
        // A colour of none sets the wheel to its centre, whatever the strength.
        let none = AutoSteps {
            shadow_color: Some(ShadowColor::None),
            shadow_color_strength: Some(3),
            ..AutoSteps::default()
        };
        assert!(none.sets_shadow_color());
        assert_eq!(none.values(), Some(AppleValues::default()));
        assert_eq!(none.normalised().shadow_color_strength, None);
    }

    #[test]
    fn colours_are_their_direction_at_the_strengths_radius() {
        for (strength, r) in [(1, 0.3), (2, 0.55), (3, 0.8)] {
            for c in ShadowColor::ALL {
                let s = AutoSteps {
                    shadow_color: Some(c),
                    shadow_color_strength: Some(strength),
                    ..AutoSteps::default()
                };
                let d = c.direction();
                assert_eq!(s.values().unwrap().balance_shadows, [r * d[0], r * d[1]]);
                assert_eq!(ShadowColor::from_code(c.code()), Some(c));
            }
            for c in HighlightColor::ALL {
                let s = AutoSteps {
                    highlight_color: Some(c),
                    highlight_color_strength: Some(strength),
                    ..AutoSteps::default()
                };
                let d = c.direction();
                assert_eq!(s.values().unwrap().balance_highlights, [r * d[0], r * d[1]]);
                assert_eq!(HighlightColor::from_code(c.code()), Some(c));
            }
        }
        assert_eq!(ShadowColor::Teal.direction(), [-0.866, -0.5]);
        assert_eq!(ShadowColor::Blue.direction(), [-0.5, -0.866]);
        assert_eq!(ShadowColor::Green.direction(), [-0.5, 0.866]);
        assert_eq!(ShadowColor::Amber.direction(), [0.866, 0.5]);
        assert_eq!(ShadowColor::Magenta.direction(), [0.5, -0.866]);
        assert_eq!(HighlightColor::Warm.direction(), [0.866, 0.5]);
        assert_eq!(HighlightColor::Gold.direction(), [0.5, 0.866]);
        assert_eq!(HighlightColor::Cool.direction(), [-0.5, -0.866]);
        assert_eq!(HighlightColor::Pink.direction(), [0.866, -0.5]);
        assert_eq!(ShadowColor::from_code(6), None);
        assert_eq!(HighlightColor::from_code(-1), None);
        // Every point stays inside the unit disc, as a wheel's does.
        for d in TABLE
            .shadow_directions
            .iter()
            .chain(&TABLE.highlight_directions)
        {
            assert!(d[0].hypot(d[1]) <= 1.0);
        }
    }

    #[test]
    fn warmer_is_a_lower_mired_and_magenta_a_positive_tint() {
        // A warmer picture balances the camera for a bluer light: a higher Kelvin, a lower mired.
        assert!(steps(0, 1, 0, 0).values().unwrap().white_shift_mired < 0.0);
        assert!(steps(0, -1, 0, 0).values().unwrap().white_shift_mired > 0.0);
        let magenta = AutoSteps {
            tint: Some(1),
            ..AutoSteps::default()
        };
        assert!(magenta.values().unwrap().tint_shift > 0.0);
    }

    #[test]
    fn steps_out_of_range_have_no_values() {
        let bad = [
            steps(4, 0, 0, 0),
            steps(-4, 0, 0, 0),
            steps(0, 3, 0, 0),
            steps(0, 0, -3, 0),
            steps(0, 0, 0, 3),
            AutoSteps {
                highlights: Some(3),
                ..AutoSteps::keep()
            },
            AutoSteps {
                blacks: Some(-3),
                ..AutoSteps::keep()
            },
            AutoSteps {
                tint: Some(5),
                ..AutoSteps::keep()
            },
            AutoSteps {
                shadow_color: Some(ShadowColor::Teal),
                shadow_color_strength: Some(0),
                ..AutoSteps::keep()
            },
            AutoSteps {
                highlight_color: Some(HighlightColor::Warm),
                highlight_color_strength: Some(4),
                ..AutoSteps::keep()
            },
        ];
        for s in bad {
            assert!(!s.in_range(), "{s:?}");
            assert_eq!(s.values(), None);
        }
    }

    #[test]
    fn a_checked_answer_trims_its_words_and_refuses_bad_or_empty_steps() {
        let long = format!("  {}  ", "x".repeat(400));
        let a = AdvisorAnswer {
            steps: AutoSteps::keep(),
            scene: long.clone(),
            plan: long,
        }
        .checked()
        .unwrap();
        assert_eq!(a.scene.chars().count(), SCENE_MAX_CHARS);
        assert_eq!(a.plan.chars().count(), PLAN_MAX_CHARS);
        let bad = AdvisorAnswer {
            steps: steps(0, 0, 5, 0),
            scene: String::new(),
            plan: String::new(),
        }
        .checked()
        .unwrap_err();
        assert_eq!(bad.kind, FailureKind::BadAnswer);
        // Nothing filled, or only a strength without its colour: a failure.
        for empty in [
            AutoSteps::default(),
            AutoSteps {
                shadow_color_strength: Some(2),
                ..AutoSteps::default()
            },
        ] {
            let e = AdvisorAnswer {
                steps: empty,
                scene: "A scene".to_owned(),
                plan: "A plan".to_owned(),
            }
            .checked()
            .unwrap_err();
            assert_eq!(e.kind, FailureKind::BadAnswer);
        }
        // A strength with none is dropped.
        let a = AdvisorAnswer {
            steps: AutoSteps {
                highlight_color: Some(HighlightColor::None),
                highlight_color_strength: Some(2),
                ..AutoSteps::default()
            },
            scene: String::new(),
            plan: String::new(),
        }
        .checked()
        .unwrap();
        assert_eq!(a.steps.highlight_color_strength, None);
    }

    #[test]
    fn burst_steps_are_field_medians_of_the_frames_that_set_them() {
        assert_eq!(share_burst_steps(&[]), None);
        assert_eq!(share_burst_steps(&[None, None]), None);
        let a = steps(1, 0, 2, -1);
        assert_eq!(share_burst_steps(&[Some(a), None]), Some(a));
        let frames = [
            Some(steps(1, -1, 0, 0)),
            Some(steps(3, 1, 1, 0)),
            None,
            Some(steps(2, 0, -2, 1)),
        ];
        assert_eq!(share_burst_steps(&frames), Some(steps(2, 0, 0, 0)));
        // Even count: the lower of the two middle values.
        let even = [Some(steps(0, 0, 0, 0)), Some(steps(2, 2, 2, 2))];
        assert_eq!(share_burst_steps(&even), Some(steps(0, 0, 0, 0)));
        // A field only some frames set is their median; a field none set stays unset.
        let partial = [
            Some(AutoSteps {
                highlights: Some(-2),
                ..AutoSteps::default()
            }),
            Some(AutoSteps {
                exposure: Some(1),
                ..AutoSteps::default()
            }),
            Some(AutoSteps {
                highlights: Some(1),
                exposure: Some(-1),
                ..AutoSteps::default()
            }),
        ];
        let shared = share_burst_steps(&partial).unwrap();
        assert_eq!(shared.highlights, Some(-2));
        assert_eq!(shared.exposure, Some(-1));
        assert_eq!(shared.tint, None);
        assert_eq!(shared.shadow_color, None);
    }

    fn colored(shadow: Option<(ShadowColor, Option<i8>)>) -> Option<AutoSteps> {
        Some(AutoSteps {
            shadow_color: shadow.map(|(c, _)| c),
            shadow_color_strength: shadow.and_then(|(_, s)| s),
            contrast: Some(0),
            ..AutoSteps::default()
        })
    }

    #[test]
    fn burst_colours_are_the_most_frequent_with_the_median_strength_of_their_frames() {
        use ShadowColor::{Amber, Blue, Teal};
        let frames = [
            colored(Some((Blue, Some(3)))),
            colored(Some((Teal, Some(1)))),
            colored(None),
            colored(Some((Teal, Some(3)))),
            colored(Some((Teal, Some(2)))),
        ];
        let s = share_burst_steps(&frames).unwrap();
        assert_eq!(s.shadow_color, Some(Teal));
        assert_eq!(s.shadow_color_strength, Some(2));
        // A tie goes to the colour of the earliest frame among the tied ones.
        let tie = [
            colored(Some((Amber, Some(1)))),
            colored(Some((Blue, Some(2)))),
            colored(Some((Blue, Some(3)))),
            colored(Some((Amber, Some(3)))),
        ];
        let s = share_burst_steps(&tie).unwrap();
        assert_eq!(
            (s.shadow_color, s.shadow_color_strength),
            (Some(Amber), Some(1))
        );
        // The chosen colour's frames gave no strength: the colour sets nothing.
        let weak = [colored(Some((Teal, None))), colored(Some((Teal, None)))];
        let s = share_burst_steps(&weak).unwrap();
        assert_eq!(
            (s.shadow_color, s.shadow_color_strength),
            (Some(Teal), None)
        );
        assert!(!s.sets_shadow_color());
        // None is a colour like the others.
        let none = [
            colored(Some((ShadowColor::None, None))),
            colored(Some((ShadowColor::None, None))),
            colored(Some((Teal, Some(2)))),
        ];
        let s = share_burst_steps(&none).unwrap();
        assert_eq!(
            (s.shadow_color, s.shadow_color_strength),
            (Some(ShadowColor::None), None)
        );
    }

    fn flat(width: usize, height: usize, value: u8) -> AdvisorPicture {
        AdvisorPicture {
            width,
            height,
            rgb: vec![value; width * height * 3],
        }
    }

    #[test]
    fn the_fake_reads_brightness_and_spread_and_echoes_the_instructions() {
        let fake = FakeAdvisor;
        let ask = |p: &AdvisorPicture| fake.ask(p, DEFAULT_INSTRUCTIONS).unwrap();
        assert!(fake.availability().is_available());
        let dark = ask(&flat(4, 3, 40)).steps;
        assert_eq!(
            (dark.exposure, dark.highlights, dark.shadows, dark.contrast),
            (Some(1), None, Some(1), Some(1))
        );
        assert_eq!(
            (dark.shadow_color, dark.shadow_color_strength),
            (Some(ShadowColor::Teal), Some(1))
        );
        assert_eq!(
            (dark.blacks, dark.tint, dark.highlight_color),
            (None, None, None)
        );
        let bright = ask(&flat(4, 3, 220)).steps;
        assert_eq!(
            (bright.exposure, bright.highlights, bright.shadows),
            (Some(-1), Some(-1), Some(0))
        );
        let mid = ask(&flat(4, 3, 128)).steps;
        assert_eq!((mid.exposure, mid.highlights), (Some(0), Some(-1)));
        // Half black, half white: mean 0.5, standard deviation 0.5.
        let mut rgb = vec![0u8; 12];
        rgb[6..].fill(255);
        let split = AdvisorPicture {
            width: 2,
            height: 2,
            rgb,
        };
        let s = ask(&split).steps;
        assert_eq!(
            (s.exposure, s.highlights, s.contrast),
            (Some(0), None, Some(0))
        );
        assert_eq!(
            (s.shadow_color, s.shadow_color_strength),
            (Some(ShadowColor::None), None)
        );
        assert_eq!(ask(&flat(4, 3, 40)), ask(&flat(4, 3, 40)));
        // The plan names the first word of the instructions it was given.
        assert_eq!(ask(&flat(4, 3, 40)).plan, "Test plan: You");
        assert_eq!(
            fake.ask(&flat(4, 3, 40), "Grade warmly, please")
                .unwrap()
                .plan,
            "Test plan: Grade"
        );
        // Its answers pass the core's checks.
        assert!(ask(&split).checked().is_ok());
    }

    #[test]
    fn the_fake_refuses_a_malformed_picture() {
        let bad = AdvisorPicture {
            width: 4,
            height: 3,
            rgb: vec![0; 5],
        };
        assert_eq!(
            FakeAdvisor
                .ask(&bad, DEFAULT_INSTRUCTIONS)
                .unwrap_err()
                .kind,
            FailureKind::Other
        );
    }

    #[test]
    fn the_version_string_names_model_os_prompt_table_and_edited_instructions() {
        let info = AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)");
        assert_eq!(
            info.version_string(),
            "Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2"
        );
        let edited = info.with_instructions("Make it moody.");
        let digest = edited.instructions.instructions_sha256.clone();
        assert_eq!(
            edited.version_string(),
            format!(
                "Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 3 · table 2 · edited instructions ({digest})"
            )
        );
    }

    #[test]
    fn instructions_resolve_check_and_digest() {
        assert_eq!(effective_instructions(""), DEFAULT_INSTRUCTIONS);
        assert_eq!(effective_instructions("  \n "), DEFAULT_INSTRUCTIONS);
        assert_eq!(effective_instructions("Moody."), "Moody.");
        assert!(check_instructions(DEFAULT_INSTRUCTIONS).is_ok());
        assert!(check_instructions(&"é".repeat(INSTRUCTIONS_MAX_CHARS)).is_ok());
        let long = check_instructions(&"x".repeat(INSTRUCTIONS_MAX_CHARS + 1)).unwrap_err();
        assert!(
            long.contains("2001 characters") && long.contains("At most 2000"),
            "{long}"
        );
        let d = InstructionsId::of(DEFAULT_INSTRUCTIONS);
        assert_eq!(d, InstructionsId::default());
        assert_eq!(d.instructions, InstructionsKind::Default);
        assert_eq!(d.instructions_sha256.len(), 12);
        assert!(d.instructions_sha256.chars().all(|c| c.is_ascii_hexdigit()));
        // The SHA-256 of "abc" starts ba7816bf8f01.
        assert_eq!(
            InstructionsId::of("abc").instructions_sha256,
            "ba7816bf8f01"
        );
        assert_eq!(
            InstructionsId::of("abc").instructions,
            InstructionsKind::Edited
        );
        assert_eq!(
            serde_json::to_value(AdvisorInfo::new("m", "o")).unwrap(),
            serde_json::json!({
                "model": "m",
                "os": "o",
                "instructions": "default",
                "instructions_sha256": d.instructions_sha256,
            })
        );
    }

    #[test]
    fn states_serialise_with_a_state_tag() {
        let v = serde_json::to_value(AppleAuto::Unavailable {
            reason: UnavailableReason::AppleIntelligenceOff,
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({ "state": "unavailable", "reason": "apple_intelligence_off" })
        );
        assert!(!AppleAuto::Pending.is_settled());
        assert!(AppleAuto::Off.is_settled());
        assert_eq!(AppleAuto::Pending.steps(), None);
    }

    fn info() -> AdvisorInfo {
        AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)")
    }

    #[test]
    fn the_xmp_property_is_compact_sorted_json_for_an_answer_or_a_failure_only() {
        let answered = AppleAuto::Answered {
            answer: AdvisorAnswer {
                steps: AutoSteps {
                    exposure: Some(1),
                    white: Some(-1),
                    contrast: Some(1),
                    saturation: Some(0),
                    shadow_color: Some(ShadowColor::Teal),
                    shadow_color_strength: Some(2),
                    ..AutoSteps::default()
                },
                scene: "Backlit portrait \"at\" dusk".to_owned(),
                plan: "Teal shadows, warm skin".to_owned(),
            },
            info: info(),
        };
        let sha = InstructionsId::default().instructions_sha256;
        assert_eq!(
            answered.xmp_json().unwrap(),
            format!(
                r#"{{"instructions":"default","instructions_sha256":"{sha}","model":"AFM 3 Core","os":"27.0.1 (27A404)","plan":"Teal shadows, warm skin","prompt":3,"result":"answered","scene":"Backlit portrait \"at\" dusk","steps":{{"blacks":null,"contrast":1,"exposure":1,"highlight_color":null,"highlight_color_strength":null,"highlights":null,"saturation":0,"shadow_color":"teal","shadow_color_strength":2,"shadows":null,"tint":null,"white":-1}},"table":2}}"#
            )
        );
        let failed = AppleAuto::Failed {
            kind: FailureKind::RateLimited,
            message: "busy".to_owned(),
            info: info().with_instructions("Moody."),
        };
        let edited = InstructionsId::of("Moody.").instructions_sha256;
        assert_eq!(
            failed.xmp_json().unwrap(),
            format!(
                r#"{{"instructions":"edited","instructions_sha256":"{edited}","model":"AFM 3 Core","os":"27.0.1 (27A404)","prompt":3,"reason":"rate_limited","result":"failed","table":2}}"#
            )
        );
        for none in [
            AppleAuto::Off,
            AppleAuto::Pending,
            AppleAuto::Unavailable {
                reason: UnavailableReason::NotAMac,
            },
        ] {
            assert_eq!(none.xmp_json(), None);
        }
    }

    #[test]
    fn the_activity_follows_the_advisor_the_mode_the_availability_and_the_preference() {
        let available = Availability::Available(info());
        let off_here = Availability::Unavailable(UnavailableReason::AppleIntelligenceOff);
        use AppleAutoMode::{FromPreference, Off, On};
        // No advisor, or a run that says off: off.
        assert_eq!(
            Activity::decide(false, On, true, Some(&available)),
            Activity::Off
        );
        assert_eq!(
            Activity::decide(true, Off, true, Some(&available)),
            Activity::Off
        );
        // Not available: unavailable, whatever the preference.
        for (mode, pref) in [(FromPreference, true), (FromPreference, false), (On, false)] {
            assert_eq!(
                Activity::decide(true, mode, pref, Some(&off_here)),
                Activity::Unavailable(UnavailableReason::AppleIntelligenceOff)
            );
        }
        assert_eq!(
            Activity::decide(true, FromPreference, true, None),
            Activity::Unavailable(UnavailableReason::Unavailable)
        );
        // Available: the preference decides, unless the run says on.
        assert!(Activity::decide(true, FromPreference, true, Some(&available)).is_active());
        assert_eq!(
            Activity::decide(true, FromPreference, false, Some(&available)),
            Activity::Off
        );
        assert!(Activity::decide(true, On, false, Some(&available)).is_active());
        // A photo's state: its result while active, pending without one; off or unavailable else.
        let active = Activity::Active(info());
        assert_eq!(active.photo_state(None), AppleAuto::Pending);
        let failed = AppleAuto::Failed {
            kind: FailureKind::Timeout,
            message: String::new(),
            info: info(),
        };
        assert_eq!(active.photo_state(Some(&failed)), failed);
        assert_eq!(Activity::Off.photo_state(Some(&failed)), AppleAuto::Off);
        assert_eq!(
            serde_json::to_value(AppleAutoMode::FromPreference).unwrap(),
            serde_json::json!("from_preference")
        );
    }

    #[test]
    fn only_waiting_failures_are_retried() {
        assert!(FailureKind::RateLimited.retried());
        assert!(FailureKind::ConcurrentRequests.retried());
        assert!(!FailureKind::Timeout.retried());
        assert!(!FailureKind::Refused.retried());
    }
}
