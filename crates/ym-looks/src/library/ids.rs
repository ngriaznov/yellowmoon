//! Look ids, slugs and file-name clean-up (`docs/08-looks.md` section Look model).
//!
//! Id formats:
//!
//! | Id | Meaning |
//! |---|---|
//! | `builtin:<slug>` | a built-in look (`builtin:neutral`, `builtin:signature`) |
//! | `arri:<group_no>:<slug>` | an imported ARRI look; `<slug>` carries `-logc3` or `-logc3-rec709` for the older packages so the three Look Library packages never collide |
//! | `arri-display:<input>:<target>` | an imported ARRI display LUT, for example `arri-display:logc4:rec709-g24` |
//! | `user:<slug>` | any other imported `.cube` |

use std::fmt;

use serde::{Deserialize, Serialize};

use super::model::DisplayInput;

/// Longest slug, in characters. Longer names are cut.
pub const MAX_SLUG_CHARS: usize = 80;
/// Longest file stem written to the library, in characters.
pub const MAX_FILE_STEM_CHARS: usize = 100;
/// Longest file stem written to the library, in UTF-8 bytes. File systems limit a name to 255
/// bytes, and the stem gets an encoding suffix, a collision number, `.cube` and (while it is being
/// written) a hidden temporary name around it, so 100 three-byte characters would not fit.
pub const MAX_FILE_STEM_BYTES: usize = 120;

/// Lower-case alphanumerics joined by single dashes: `"Film Warm (v2)"` becomes `"film-warm-v2"`.
///
/// Unicode letters and digits are kept (lower-cased); everything else separates words. The result
/// may be empty (for a name with no letters or digits) and is idempotent.
#[must_use]
pub fn slugify(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(MAX_SLUG_CHARS * 4));
    let mut count = 0_usize;
    let mut pending_dash = false;
    'chars: for c in text.chars() {
        if !c.is_alphanumeric() {
            pending_dash = true;
            continue;
        }
        for lower in c.to_lowercase() {
            if !lower.is_alphanumeric() {
                // A lower-casing that produces a combining mark: drop the mark.
                continue;
            }
            let dash = pending_dash && !out.is_empty();
            if count + usize::from(dash) + 1 > MAX_SLUG_CHARS {
                break 'chars;
            }
            if dash {
                out.push('-');
                count += 1;
            }
            pending_dash = false;
            out.push(lower);
            count += 1;
        }
    }
    out
}

/// Whether `slug` is what [`slugify`] could have produced: non-empty, lower-case alphanumerics and
/// single dashes, no dash at either end.
#[must_use]
pub fn is_valid_slug(slug: &str) -> bool {
    !slug.is_empty()
        && !slug.starts_with('-')
        && !slug.ends_with('-')
        && !slug.contains("--")
        && slug.chars().all(|c| {
            c == '-'
                || (c.is_alphanumeric() && {
                    let mut lower = c.to_lowercase();
                    lower.next() == Some(c) && lower.next().is_none()
                })
        })
}

/// `text` cut to at most `max_bytes` bytes, never in the middle of a character.
#[must_use]
pub fn truncate_to_bytes(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// A file stem that is safe to create on macOS, Linux and Windows: letters, digits, spaces and
/// `._-()+&,` are kept, anything else becomes `_`; leading and trailing dots and spaces go. At
/// most [`MAX_FILE_STEM_CHARS`] characters and [`MAX_FILE_STEM_BYTES`] bytes. Never empty (falls
/// back to `"lut"`).
#[must_use]
pub fn sanitize_file_stem(stem: &str) -> String {
    let mapped: String = stem
        .chars()
        .map(|c| {
            if c.is_alphanumeric()
                || matches!(c, ' ' | '.' | '_' | '-' | '(' | ')' | '+' | '&' | ',')
            {
                c
            } else {
                '_'
            }
        })
        .take(MAX_FILE_STEM_CHARS)
        .collect();
    let mapped = truncate_to_bytes(&mapped, MAX_FILE_STEM_BYTES);
    let trimmed = mapped.trim_matches(|c: char| c == '.' || c == ' ');
    if trimmed.is_empty() {
        "lut".to_string()
    } else {
        trimmed.to_string()
    }
}

/// `base` with a numeric suffix for collisions: `"film"` stays `"film"` for `n <= 1`, and becomes
/// `"film-2"`, `"film-3"` after that.
#[must_use]
pub fn numbered_slug(base: &str, n: u32) -> String {
    if n <= 1 {
        base.to_string()
    } else {
        format!("{base}-{n}")
    }
}

/// Why a string is not a look id.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not a look id")]
pub struct IdError(pub String);

/// What kind of look an id names, with the parts that matter to callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdKind {
    /// `builtin:<slug>`.
    BuiltIn,
    /// `arri:<group_no>:<slug>`.
    Arri {
        /// The ARRI group number (for example 3000).
        group_no: u32,
    },
    /// `arri-display:<input>:<target>`.
    ArriDisplay {
        /// The encoding the display LUT expects.
        input: DisplayInput,
    },
    /// `user:<slug>`.
    User,
}

/// A look (or display LUT) id such as `builtin:neutral`. Always well formed: use
/// [`LookId::parse`] for text from outside, the constructors for ids we build.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct LookId(String);

impl LookId {
    /// `builtin:<slug>`. The text is slugified first (and `"look"` stands in for an empty slug).
    #[must_use]
    pub fn builtin(slug: &str) -> Self {
        Self(format!("builtin:{}", slug_or(slug, "look")))
    }

    /// `arri:<group_no>:<slug>`.
    #[must_use]
    pub fn arri(group_no: u32, slug: &str) -> Self {
        Self(format!("arri:{group_no}:{}", slug_or(slug, "look")))
    }

    /// `arri-display:<input>:<target>`.
    #[must_use]
    pub fn arri_display(input: DisplayInput, target: &str) -> Self {
        Self(format!(
            "arri-display:{}:{}",
            input.slug(),
            slug_or(target, "display")
        ))
    }

    /// `user:<slug>`.
    #[must_use]
    pub fn user(slug: &str) -> Self {
        Self(format!("user:{}", slug_or(slug, "lut")))
    }

    /// Parses an id. The text must already be in canonical form (lower-case slugs).
    ///
    /// # Errors
    ///
    /// [`IdError`] when the text is not one of the four id shapes.
    pub fn parse(text: &str) -> Result<Self, IdError> {
        let bad = || IdError(text.to_string());
        let parts: Vec<&str> = text.split(':').collect();
        let ok = match parts.as_slice() {
            ["builtin" | "user", slug] => is_valid_slug(slug),
            ["arri", group, slug] => {
                !group.is_empty()
                    && group.bytes().all(|b| b.is_ascii_digit())
                    && group.parse::<u32>().is_ok()
                    && is_valid_slug(slug)
            }
            ["arri-display", input, target] => {
                DisplayInput::from_slug(input).is_some() && is_valid_slug(target)
            }
            _ => false,
        };
        if ok {
            // Reject a group number written with leading zeros: `arri:03000:x` is not canonical.
            if let ["arri", group, _] = parts.as_slice()
                && group.len() > 1
                && group.starts_with('0')
            {
                return Err(bad());
            }
            Ok(Self(text.to_string()))
        } else {
            Err(bad())
        }
    }

    /// The id as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The part before the first colon: `builtin`, `arri`, `arri-display` or `user`.
    #[must_use]
    pub fn prefix(&self) -> &str {
        self.0.split(':').next().unwrap_or("")
    }

    /// The part after the last colon: the slug (for display LUTs, the target).
    #[must_use]
    pub fn slug(&self) -> &str {
        self.0.rsplit(':').next().unwrap_or("")
    }

    /// Everything after the first colon: `3000:film-warm` for `arri:3000:film-warm`.
    #[must_use]
    pub fn remainder(&self) -> &str {
        self.0.split_once(':').map_or("", |(_, rest)| rest)
    }

    /// What kind of id this is.
    #[must_use]
    pub fn kind(&self) -> IdKind {
        let mut parts = self.0.split(':');
        match (parts.next(), parts.next()) {
            (Some("builtin"), _) => IdKind::BuiltIn,
            (Some("arri"), Some(group)) => IdKind::Arri {
                group_no: group.parse().unwrap_or(0),
            },
            (Some("arri-display"), Some(input)) => IdKind::ArriDisplay {
                input: DisplayInput::from_slug(input).unwrap_or(DisplayInput::LogC4),
            },
            _ => IdKind::User,
        }
    }

    /// The same id with its slug (target for display LUTs) replaced.
    #[must_use]
    pub fn with_slug(&self, slug: &str) -> Self {
        let slug = slug_or(slug, "look");
        match self.0.rsplit_once(':') {
            Some((head, _)) => Self(format!("{head}:{slug}")),
            None => Self(format!("user:{slug}")),
        }
    }

    /// The id with a collision number: [`numbered_slug`] applied to its slug.
    #[must_use]
    pub fn numbered(&self, n: u32) -> Self {
        self.with_slug(&numbered_slug(self.slug(), n))
    }
}

fn slug_or(text: &str, fallback: &str) -> String {
    let slug = slugify(text);
    if slug.is_empty() {
        fallback.to_string()
    } else {
        slug
    }
}

impl fmt::Display for LookId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for LookId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LookId({})", self.0)
    }
}

impl TryFrom<String> for LookId {
    type Error = IdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<LookId> for String {
    fn from(id: LookId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_basics() {
        assert_eq!(slugify("Film Warm (v2)"), "film-warm-v2");
        assert_eq!(slugify("  --Bleach_Bypass--  "), "bleach-bypass");
        assert_eq!(slugify("Café Noir"), "café-noir");
        assert_eq!(slugify("***"), "");
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("A  B"), "a-b");
    }

    #[test]
    fn slugify_is_bounded() {
        let long = "word ".repeat(100);
        let slug = slugify(&long);
        assert!(slug.chars().count() <= MAX_SLUG_CHARS);
        assert!(is_valid_slug(&slug), "{slug:?}");
    }

    #[test]
    fn sanitize_keeps_names_readable() {
        assert_eq!(sanitize_file_stem("Film Warm (v2)"), "Film Warm (v2)");
        assert_eq!(sanitize_file_stem("a/b\\c:d*e?"), "a_b_c_d_e_");
        assert_eq!(sanitize_file_stem("..hidden"), "hidden");
        assert_eq!(sanitize_file_stem("  "), "lut");
        assert_eq!(sanitize_file_stem(""), "lut");
        assert!(sanitize_file_stem(&"x".repeat(500)).chars().count() <= MAX_FILE_STEM_CHARS);
        // Long names in scripts with multi-byte letters are cut by bytes as well.
        let cjk = sanitize_file_stem(&"\u{8272}".repeat(200));
        assert!(cjk.len() <= MAX_FILE_STEM_BYTES, "{}", cjk.len());
        assert_eq!(cjk.chars().count(), MAX_FILE_STEM_BYTES / 3);
        assert_eq!(truncate_to_bytes("a\u{e9}b", 2), "a");
        assert_eq!(truncate_to_bytes("abc", 10), "abc");
        assert_eq!(truncate_to_bytes("\u{1f600}", 3), "");
    }

    #[test]
    fn id_constructors_and_parts() {
        let a = LookId::arri(3000, "Film Warm");
        assert_eq!(a.as_str(), "arri:3000:film-warm");
        assert_eq!(a.prefix(), "arri");
        assert_eq!(a.slug(), "film-warm");
        assert_eq!(a.remainder(), "3000:film-warm");
        assert_eq!(a.kind(), IdKind::Arri { group_no: 3000 });
        let d = LookId::arri_display(DisplayInput::LogC4, "Rec709 G24");
        assert_eq!(d.as_str(), "arri-display:logc4:rec709-g24");
        assert_eq!(
            d.kind(),
            IdKind::ArriDisplay {
                input: DisplayInput::LogC4
            }
        );
        assert_eq!(LookId::user("My LUT").as_str(), "user:my-lut");
        assert_eq!(LookId::builtin("neutral").kind(), IdKind::BuiltIn);
        assert_eq!(LookId::user("").as_str(), "user:lut");
    }

    #[test]
    fn numbering_changes_only_the_slug() {
        let a = LookId::arri(3000, "film");
        assert_eq!(a.numbered(1), a);
        assert_eq!(a.numbered(2).as_str(), "arri:3000:film-2");
        let d = LookId::arri_display(DisplayInput::LogC3, "p3");
        assert_eq!(d.numbered(3).as_str(), "arri-display:logc3:p3-3");
    }

    #[test]
    fn parse_accepts_canonical_ids_only() {
        for good in [
            "builtin:neutral",
            "arri:3000:film-warm",
            "arri-display:logc4:rec709-g24",
            "user:my-lut",
            "arri:0:x",
        ] {
            assert_eq!(LookId::parse(good).unwrap().as_str(), good);
        }
        for bad in [
            "",
            "neutral",
            "builtin:",
            "builtin:Neutral",
            "arri:film",
            "arri:03000:film",
            "arri:x:film",
            "arri-display:logc5:p3",
            "user:a:b",
            "other:a",
            "user:-a",
            "user:a--b",
        ] {
            assert!(LookId::parse(bad).is_err(), "{bad:?}");
        }
    }

    /// Property tests use a fixed seed so every run, on every machine, sees the same cases (the
    /// build rules ask for deterministic tests). `PROPTEST_CASES` still widens the search by hand.
    fn seeded() -> proptest::test_runner::Config {
        proptest::test_runner::Config {
            rng_seed: proptest::test_runner::RngSeed::Fixed(0x594d_4c4f_4f4b_5331),
            ..proptest::test_runner::Config::default()
        }
    }

    proptest::proptest! {
        #![proptest_config(seeded())]

        #[test]
        fn slugify_is_idempotent_and_valid(text in ".{0,200}") {
            let slug = slugify(&text);
            proptest::prop_assert_eq!(slugify(&slug), slug.clone());
            proptest::prop_assert!(slug.chars().count() <= MAX_SLUG_CHARS);
            if !slug.is_empty() {
                proptest::prop_assert!(is_valid_slug(&slug), "{:?}", slug);
            }
        }

        #[test]
        fn ids_built_from_any_text_parse_back(text in ".{0,100}", group in 0_u32..100_000, n in 1_u32..50) {
            let ids = [
                LookId::user(&text),
                LookId::builtin(&text),
                LookId::arri(group, &text),
                LookId::arri_display(DisplayInput::LogC3, &text),
            ];
            for id in ids {
                proptest::prop_assert_eq!(LookId::parse(id.as_str()).unwrap(), id.clone());
                let numbered = id.numbered(n);
                proptest::prop_assert_eq!(LookId::parse(numbered.as_str()).unwrap(), numbered.clone());
                proptest::prop_assert_eq!(numbered.kind(), id.kind());
            }
        }

        #[test]
        fn file_stems_are_always_safe(text in ".{0,300}") {
            let stem = sanitize_file_stem(&text);
            proptest::prop_assert!(!stem.is_empty());
            proptest::prop_assert!(stem.chars().count() <= MAX_FILE_STEM_CHARS);
            proptest::prop_assert!(stem.len() <= MAX_FILE_STEM_BYTES);
            proptest::prop_assert!(!stem.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']));
            proptest::prop_assert!(!stem.starts_with('.') && !stem.ends_with('.') && !stem.ends_with(' '));
        }
    }

    #[test]
    fn serde_round_trip_validates() {
        let id = LookId::user("x");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"user:x\"");
        assert_eq!(serde_json::from_str::<LookId>(&json).unwrap(), id);
        assert!(serde_json::from_str::<LookId>("\"nonsense\"").is_err());
    }
}
