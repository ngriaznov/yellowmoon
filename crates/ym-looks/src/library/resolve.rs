//! Resolving a `--look` argument: full id, slug after the colon, or display name, case-insensitive (`docs/10-cli.md`).

use super::model::Look;

/// CLI exit code for a usage error, including an ambiguous `--look` (`docs/10-cli.md`).
pub const EXIT_USAGE: i32 = 2;
/// CLI exit code for a look that is missing (`docs/10-cli.md`).
pub const EXIT_LOOK_MISSING: i32 = 3;

/// One look an ambiguous query could mean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Its id (always unambiguous).
    pub id: String,
    /// Its display name.
    pub name: String,
}

/// Why a query did not name exactly one look.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// Nothing matches.
    #[error("No look called \"{query}\". Run `yellowmoon looks list` to see the ids and names.")]
    NotFound {
        /// The query as given.
        query: String,
    },
    /// More than one look matches at the first level that matched anything.
    #[error("\"{query}\" matches more than one look: {}. Use the full id.", list(.matches))]
    Ambiguous {
        /// The query as given.
        query: String,
        /// Every look it matches, in library order.
        matches: Vec<Candidate>,
    },
}

fn list(matches: &[Candidate]) -> String {
    matches
        .iter()
        .map(|c| format!("{} ({})", c.id, c.name))
        .collect::<Vec<_>>()
        .join(", ")
}

impl ResolveError {
    /// The CLI exit code: 3 for a missing look, 2 for an ambiguous one.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotFound { .. } => EXIT_LOOK_MISSING,
            Self::Ambiguous { .. } => EXIT_USAGE,
        }
    }
}

/// Lower-cases, trims and collapses runs of white space.
fn fold(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Finds the look a `--look` argument means. Three levels are tried in order and the first that
/// matches anything decides:
///
/// 1. the full id (`builtin:neutral`, `arri:3000:film`),
/// 2. the slug after the last colon (`neutral`, `film`), or everything after the first colon
///    (`3000:film`),
/// 3. the display name (`Neutral`, `Film Warm`).
///
/// All comparisons ignore case. A level that matches several looks is ambiguous; it does not fall
/// through to the next level. Disabled looks (missing or changed file) resolve like any other, so
/// the caller can say they need importing again.
///
/// # Errors
///
/// [`ResolveError::NotFound`] (exit code 3) or [`ResolveError::Ambiguous`] (exit code 2).
pub fn resolve<'a>(looks: &'a [Look], query: &str) -> Result<&'a Look, ResolveError> {
    let wanted = fold(query);
    let levels: [&dyn Fn(&Look) -> bool; 3] = [
        &|l| l.id.as_str().to_lowercase() == wanted,
        &|l| l.id.slug().to_lowercase() == wanted || l.id.remainder().to_lowercase() == wanted,
        &|l| fold(&l.name) == wanted,
    ];
    if !wanted.is_empty() {
        for matches_level in levels {
            let found: Vec<&Look> = looks.iter().filter(|l| matches_level(l)).collect();
            match found.as_slice() {
                [] => {}
                [one] => return Ok(one),
                many => {
                    return Err(ResolveError::Ambiguous {
                        query: query.to_string(),
                        matches: many
                            .iter()
                            .map(|l| Candidate {
                                id: l.id.to_string(),
                                name: l.name.clone(),
                            })
                            .collect(),
                    });
                }
            }
        }
    }
    Err(ResolveError::NotFound {
        query: query.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use ym_color::look::LookInput;

    use super::*;
    use crate::library::{LookGroup, LookId, LookStatus};

    fn look(id: LookId, name: &str, group: LookGroup) -> Look {
        Look {
            id,
            name: name.to_string(),
            group,
            input: LookInput::LogC4,
            cdl: None,
            lut: None,
            source: None,
            status: LookStatus::Ok,
        }
    }

    fn sample() -> Vec<Look> {
        let arri = |no: u32, name: &str| LookGroup::Arri {
            group_no: no,
            group_name: name.to_string(),
        };
        vec![
            look(LookId::builtin("neutral"), "Neutral", LookGroup::BuiltIn),
            look(
                LookId::arri(3000, "film-warm"),
                "Film Warm",
                arri(3000, "Film"),
            ),
            look(
                LookId::arri(3000, "film-warm-logc3"),
                "Film Warm",
                arri(3000, "Film"),
            ),
            look(
                LookId::arri(4000, "film-warm"),
                "Film warm",
                arri(4000, "Other"),
            ),
            look(LookId::user("neutral"), "Neutral", LookGroup::User),
            look(LookId::user("my-lut"), "My LUT", LookGroup::User),
        ]
    }

    #[test]
    fn full_id_always_wins_and_ignores_case() {
        let looks = sample();
        assert_eq!(
            resolve(&looks, "builtin:neutral").unwrap().id.as_str(),
            "builtin:neutral"
        );
        assert_eq!(
            resolve(&looks, "  USER:Neutral ").unwrap().id.as_str(),
            "user:neutral"
        );
    }

    #[test]
    fn slug_and_name_resolve_when_unique() {
        let looks = sample();
        assert_eq!(
            resolve(&looks, "my-lut").unwrap().id.as_str(),
            "user:my-lut"
        );
        assert_eq!(
            resolve(&looks, "My LUT").unwrap().id.as_str(),
            "user:my-lut"
        );
        assert_eq!(
            resolve(&looks, "my   lut").unwrap().id.as_str(),
            "user:my-lut"
        );
        assert_eq!(
            resolve(&looks, "3000:film-warm-logc3").unwrap().id.as_str(),
            "arri:3000:film-warm-logc3"
        );
    }

    #[test]
    fn ambiguity_lists_the_matches_and_exits_2() {
        let looks = sample();
        // Slug level: builtin:neutral and user:neutral both have the slug "neutral".
        let err = resolve(&looks, "neutral").unwrap_err();
        assert_eq!(err.exit_code(), EXIT_USAGE);
        match &err {
            ResolveError::Ambiguous { matches, .. } => {
                let ids: Vec<_> = matches.iter().map(|m| m.id.as_str()).collect();
                assert_eq!(ids, vec!["builtin:neutral", "user:neutral"]);
            }
            other => panic!("{other:?}"),
        }
        let message = err.to_string();
        assert!(message.contains("builtin:neutral"), "{message}");
        assert!(message.contains("user:neutral"), "{message}");
        // Name level: three looks are called "Film Warm" (case-insensitively).
        let err = resolve(&looks, "film warm").unwrap_err();
        assert!(matches!(&err, ResolveError::Ambiguous { matches, .. } if matches.len() == 3));
        // The slug "film-warm" matches two (groups 3000 and 4000), not three.
        let err = resolve(&looks, "film-warm").unwrap_err();
        assert!(matches!(&err, ResolveError::Ambiguous { matches, .. } if matches.len() == 2));
    }

    #[test]
    fn missing_looks_exit_3() {
        let looks = sample();
        for query in ["nope", "", "   ", "arri:3000"] {
            let err = resolve(&looks, query).unwrap_err();
            assert_eq!(err.exit_code(), EXIT_LOOK_MISSING, "{query:?}");
            assert!(matches!(err, ResolveError::NotFound { .. }));
        }
    }

    #[test]
    fn disabled_looks_still_resolve() {
        let mut looks = sample();
        looks[5].status = LookStatus::Missing;
        let found = resolve(&looks, "my-lut").unwrap();
        assert_eq!(
            found.disabled_reason(),
            Some(super::super::DISABLED_MESSAGE)
        );
    }
}
