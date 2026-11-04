//! The Film node's settings (`docs/24-film.md` section 4.5): the stock and the print, halation, grain,
//! Strength and the masks, as plain numbers (no Auto kinds).
//!
//! A Film node is a numbered node of the graph, like a corrector: `picture` and `key` ports, masks, Bypass.
//! Its ids share the number space of the correctors. The stock and the paper are profile ids of
//! `data/film/` (checked against `names.json` by [`NodeGraph::sanitise`](super::NodeGraph)).

use serde::{Deserialize, Serialize};

use super::NodePos;
use super::graph::DEFAULT_FILM_NAME;
use super::masks::MaskSettings;

/// Range of the film exposure, in stops.
pub const FILM_EXPOSURE_RANGE_EV: (f64, f64) = (-4.0, 4.0);
/// Range of the print exposure, in stops.
pub const PRINT_EXPOSURE_RANGE_EV: (f64, f64) = (-3.0, 3.0);
/// Range of each enlarger light, in Kodak CC units.
pub const LIGHTS_RANGE: (f64, f64) = (-50.0, 50.0);
/// Range of halation Amount, in percent.
pub const HALATION_AMOUNT_RANGE: (f64, f64) = (0.0, 300.0);
/// Range of halation Size and grain Size, in percent.
pub const SIZE_RANGE: (f64, f64) = (50.0, 200.0);
/// Range of grain Amount, in percent.
pub const GRAIN_AMOUNT_RANGE: (f64, f64) = (0.0, 200.0);
/// Range of Strength, in percent.
pub const STRENGTH_RANGE: (f64, f64) = (0.0, 100.0);
/// The stock a new Film node has.
pub const DEFAULT_STOCK: &str = "kodak_portra_400";

/// `film`: the negative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FilmStockSettings {
    /// The profile id of the stock (a file name of `data/film/spektrafilm/profiles/` without `.json`).
    pub stock: String,
    /// Exposes the negative more or less, in stops; 0 is the Input's exposure.
    pub exposure_ev: f64,
}

impl Default for FilmStockSettings {
    fn default() -> Self {
        Self {
            stock: DEFAULT_STOCK.to_owned(),
            exposure_ev: 0.0,
        }
    }
}

/// The enlarger's filtration, as shifts from neutral in Kodak CC units.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PrintLights {
    /// Cyan.
    pub c: f64,
    /// Magenta.
    pub m: f64,
    /// Yellow.
    pub y: f64,
}

/// `print`: the enlarger and the paper. A slide has none (its `paper` reads as `null`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PrintSettings {
    /// The paper's profile id; `null` is the stock's matching paper.
    pub paper: Option<String>,
    /// The lamp's exposure of the paper, in stops; 0 keeps the app's brightness.
    pub exposure_ev: f64,
    /// The enlarger's filtration.
    pub lights: PrintLights,
}

/// `halation` (percent; 100 is the stock's own).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HalationSettings {
    /// 0 to 300.
    pub amount: f64,
    /// 50 to 200.
    pub size: f64,
}

impl Default for HalationSettings {
    fn default() -> Self {
        Self {
            amount: 100.0,
            size: 100.0,
        }
    }
}

/// `grain` (percent; 100 is the stock's own).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GrainSettings {
    /// 0 to 200.
    pub amount: f64,
    /// 50 to 200.
    pub size: f64,
    /// Not in the interface; it keeps the grain a function of the settings.
    pub seed: u32,
}

impl Default for GrainSettings {
    fn default() -> Self {
        Self {
            amount: 100.0,
            size: 100.0,
            seed: 0,
        }
    }
}

/// A Film node (`docs/24` section 4.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FilmNode {
    /// The card's name, 1 to 32 characters after trimming.
    pub name: String,
    /// A bypassed node passes its picture through; its masks still exist.
    pub bypass: bool,
    /// The card's place.
    pub pos: NodePos,
    /// The negative.
    pub film: FilmStockSettings,
    /// The enlarger and the paper.
    pub print: PrintSettings,
    /// Halation.
    pub halation: HalationSettings,
    /// Grain.
    pub grain: GrainSettings,
    /// How much of the film look is applied, 0 to 100 percent (a node-wide mask value).
    pub strength: f64,
    /// The masks, as a corrector's.
    pub masks: MaskSettings,
}

impl Default for FilmNode {
    /// Portra 400 on its matching paper, every number at its default.
    fn default() -> Self {
        Self {
            name: DEFAULT_FILM_NAME.to_owned(),
            bypass: false,
            pos: [0.0, 0.0],
            film: FilmStockSettings::default(),
            print: PrintSettings::default(),
            halation: HalationSettings::default(),
            grain: GrainSettings::default(),
            strength: 100.0,
            masks: MaskSettings::default(),
        }
    }
}

impl FilmNode {
    /// Whether the node does nothing to the picture: bypassed, or Strength 0.
    #[must_use]
    pub fn is_inert(&self) -> bool {
        self.bypass || self.strength <= 0.0
    }
}
