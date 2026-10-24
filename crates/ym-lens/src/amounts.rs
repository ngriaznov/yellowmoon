//! The lens setting: which lens and how much of it (`docs/21` section 2).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::LensError;

/// Which lens. The ids are the file names of `data/lenses/` and the values of `--lens`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LensKind {
    /// Clean and modern: even sharpness, faint glow.
    #[default]
    Default,
    /// Warm and gentle: soft glow, amber flares, smooth falloff.
    Classic,
    /// Glows wide open, lower contrast, soft edges, a little swirl.
    Vintage,
    /// Very sharp centre, strong swirl, fast falloff.
    Petzval,
    /// Soft corners, busy flares, a crisp centre.
    Triplet,
    /// Smooth and glowing, with gentle edges.
    Sonnar,
    /// Sharp only in the centre, with smeared flares.
    Detuned,
}

impl LensKind {
    /// All seven lenses, in the order of the menu.
    pub const ALL: [LensKind; 7] = [
        LensKind::Default,
        LensKind::Classic,
        LensKind::Vintage,
        LensKind::Petzval,
        LensKind::Triplet,
        LensKind::Sonnar,
        LensKind::Detuned,
    ];

    /// The id: `"default"`, `"classic"`, and so on.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            LensKind::Default => "default",
            LensKind::Classic => "classic",
            LensKind::Vintage => "vintage",
            LensKind::Petzval => "petzval",
            LensKind::Triplet => "triplet",
            LensKind::Sonnar => "sonnar",
            LensKind::Detuned => "detuned",
        }
    }
}

impl fmt::Display for LensKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for LensKind {
    type Err = LensError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        LensKind::ALL
            .into_iter()
            .find(|k| k.as_str() == s)
            .ok_or_else(|| LensError::UnknownLens(s.to_owned()))
    }
}

/// The setting `lens` (`docs/21` section 2): a lens and three amounts, each 0 to 100.
///
/// A document without a field reads it as the default; an unknown field is refused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LensAmounts {
    /// The lens.
    pub kind: LensKind,
    /// Glow, veiling glare, ghosts and starbursts (L3).
    pub flares: f64,
    /// Edge softness, darkening, fringes, the lens's distortion, swirl and cat's-eye (L1, L2, L4).
    pub edges: f64,
    /// The anamorphic attachment (L1 to L4).
    pub anamorphic: f64,
}

impl LensAmounts {
    /// True when all three amounts are 0, whatever the lens: no lens stage runs.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.flares == 0.0 && self.edges == 0.0 && self.anamorphic == 0.0
    }

    /// Checks that every amount is a number from 0 to 100.
    pub(crate) fn validate(&self) -> Result<(), LensError> {
        for (name, value) in [
            ("flares", self.flares),
            ("edges", self.edges),
            ("anamorphic", self.anamorphic),
        ] {
            if !(0.0..=100.0).contains(&value) {
                return Err(LensError::Amount { name, value });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_round_trips_through_text_and_serde() {
        for k in LensKind::ALL {
            assert_eq!(k.as_str().parse::<LensKind>().unwrap(), k);
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(json, format!("\"{}\"", k.as_str()));
            assert_eq!(serde_json::from_str::<LensKind>(&json).unwrap(), k);
        }
        assert!(matches!(
            "Default".parse::<LensKind>(),
            Err(LensError::UnknownLens(_))
        ));
    }

    #[test]
    fn partial_documents_read_with_defaults_and_unknown_fields_are_refused() {
        let a: LensAmounts = serde_json::from_str(r#"{ "flares": 40 }"#).unwrap();
        assert_eq!(
            a,
            LensAmounts {
                kind: LensKind::Default,
                flares: 40.0,
                edges: 0.0,
                anamorphic: 0.0
            }
        );
        assert!(serde_json::from_str::<LensAmounts>(r#"{ "kind": "leica" }"#).is_err());
        assert!(serde_json::from_str::<LensAmounts>(r#"{ "grain": 3 }"#).is_err());
    }

    #[test]
    fn identity_means_all_three_amounts_zero() {
        let mut a = LensAmounts {
            kind: LensKind::Petzval,
            ..LensAmounts::default()
        };
        assert!(a.is_identity());
        a.edges = 1.0;
        assert!(!a.is_identity());
    }

    #[test]
    fn amounts_outside_the_range_are_refused() {
        let a = LensAmounts {
            flares: 100.5,
            ..LensAmounts::default()
        };
        assert!(a.validate().is_err());
        let a = LensAmounts {
            edges: f64::NAN,
            ..LensAmounts::default()
        };
        assert!(a.validate().is_err());
    }
}
