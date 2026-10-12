//! Reading meaning out of file and folder names: group folders, display targets, look names.
//!
//! Everything here works on text only, so the rules are testable without any file.

/// Lower-cases `text` and keeps letters and digits only: `"LogC4_to Rec.709"` becomes
/// `"logc4torec709"`. Package patterns and evidence are compared in this form, so the separators a
/// file name uses (spaces, dashes, underscores, dots) never matter.
#[must_use]
pub fn fold_alnum(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Letter-and-digit runs of `text`, lower-cased.
#[must_use]
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The folder part of a `/`-separated relative path: every component but the last.
#[must_use]
pub fn folder_components(rel_path: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = rel_path.split('/').collect();
    parts.pop();
    parts.retain(|p| !p.is_empty());
    parts
}

/// The file name of a `/`-separated relative path.
#[must_use]
pub fn file_name(rel_path: &str) -> &str {
    rel_path.rsplit('/').next().unwrap_or(rel_path)
}

/// The file name without its last extension.
#[must_use]
pub fn file_stem(rel_path: &str) -> &str {
    let name = file_name(rel_path);
    match name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => stem,
        _ => name,
    }
}

/// A look group parsed from a folder name: `"3000 Film"`, `"3000_Film"`, `"3000-Film"` or
/// `"3000 - Film"` give group 3000 named "Film". The number must be exactly four digits.
/// A folder that is only the number is named "Group 3000". A number that runs on into more
/// digits (`"2026-10-12"`, `"2024_05"`, `"2024.1"`) is a date or a version, not a group.
#[must_use]
pub fn parse_group_folder(component: &str) -> Option<(u32, String)> {
    let bytes = component.as_bytes();
    if bytes.len() < 4 || !bytes[..4].iter().all(u8::is_ascii_digit) {
        return None;
    }
    if bytes.get(4).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    if bytes
        .get(4)
        .is_some_and(|b| matches!(b, b'-' | b'_' | b'.'))
        && bytes.get(5).is_some_and(u8::is_ascii_digit)
    {
        return None;
    }
    let number: u32 = component[..4].parse().ok()?;
    let rest = component[4..].trim_matches(|c: char| c == '_' || c == '-' || c.is_whitespace());
    let name = rest
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let name = if name.is_empty() {
        format!("Group {number}")
    } else {
        name
    };
    Some((number, name))
}

/// The innermost group folder in a relative path (`ARRI Looks/3000 Film/x.cube` gives 3000).
#[must_use]
pub fn group_of(rel_path: &str) -> Option<(u32, String)> {
    folder_components(rel_path)
        .into_iter()
        .rev()
        .find_map(parse_group_folder)
}

/// Splits a name into words at spaces, underscores and dashes.
fn words(text: &str) -> Vec<&str> {
    text.split(|c: char| c.is_whitespace() || c == '_' || c == '-')
        .filter(|w| !w.is_empty())
        .collect()
}

/// Whether the words after a "to" start a display name, for example `Gamma24 Rec709`, `P3 D65`,
/// `HLG`, `PQ`. The words are joined and compared with the folded names, so `Rec 709` and
/// `Rec.709` both count.
fn starts_display(after_to: &[&str], display_names: &[String]) -> bool {
    let joined = fold_alnum(&after_to.join(""));
    !joined.is_empty()
        && display_names
            .iter()
            .any(|d| !d.is_empty() && joined.starts_with(&fold_alnum(d)))
}

/// Where a display rendering is named in a file stem: the index of the "to" word in `words(stem)`
/// that is followed by a display name (`LogC4_to_Gamma24_Rec709` gives the index of "to").
fn display_split(stem_words: &[&str], display_names: &[String]) -> Option<usize> {
    (0..stem_words.len()).find(|&i| {
        stem_words[i].eq_ignore_ascii_case("to")
            && starts_display(&stem_words[i + 1..], display_names)
    })
}

/// Whether a file stem names a display rendering: `-to-` (or `_to_`, or spaces) followed by a
/// display name such as `Gamma24_Rec709`, `P3`, `Rec2020`, `HLG` or `PQ`.
#[must_use]
pub fn has_display_rendering(stem: &str, display_names: &[String]) -> bool {
    let cleaned = stem.replace('.', "");
    display_split(&words(&cleaned), display_names).is_some()
}

/// The display a stem renders to, as an id-friendly slug: `LogC4_to_Gamma24_Rec709` gives
/// `rec709-g24`, `LogC3-to-P3-D65` gives `p3-d65`. The gamma word moves to the end so
/// `Gamma24_Rec709` and `Rec709_Gamma24` agree. `None` when the stem names no display.
#[must_use]
pub fn display_target(stem: &str, display_names: &[String]) -> Option<String> {
    let cleaned = stem.replace('.', "");
    let all = words(&cleaned);
    let at = display_split(&all, display_names)?;
    let mut main: Vec<String> = Vec::new();
    let mut gamma: Vec<String> = Vec::new();
    let mut rest = all[at + 1..].iter().map(|w| w.to_lowercase()).peekable();
    while let Some(word) = rest.next() {
        if word == "gamma"
            && rest
                .peek()
                .is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))
        {
            let digits = rest.next().unwrap_or_default();
            gamma.push(format!("g{digits}"));
        } else if let Some(digits) = word.strip_prefix("gamma")
            && !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
        {
            gamma.push(format!("g{digits}"));
        } else {
            main.push(word);
        }
    }
    main.extend(gamma);
    let slug = crate::library::slugify(&main.join("-"));
    (!slug.is_empty()).then_some(slug)
}

/// Words that are dropped from the start and end of a look's display name: they say which package
/// it is from, not what the look is.
const LEADING_NOISE: [&str; 6] = ["arri", "logc4", "logc3", "awg4", "awg3", "ei800"];

/// The display name of an ARRI look from its file stem: underscores and dashes become spaces; a
/// leading group number, a leading `ARRI`/`LogC4`/`LogC3`/`AWG4`/`AWG3`/`EI800` and a trailing
/// `to <display>` are dropped. Falls back to the plain stem when nothing would be left.
#[must_use]
pub fn look_name(stem: &str, display_names: &[String]) -> String {
    let plain = words(stem).join(" ");
    let all = words(stem);
    let mut start = 0;
    while let Some(word) = all.get(start) {
        let lower = word.to_lowercase();
        let is_group_number = word.len() == 4 && word.bytes().all(|b| b.is_ascii_digit());
        if is_group_number || LEADING_NOISE.contains(&lower.as_str()) {
            start += 1;
        } else {
            break;
        }
    }
    let mut end = all.len();
    let cleaned: Vec<String> = all.iter().map(|w| w.replace('.', "")).collect();
    let cleaned_refs: Vec<&str> = cleaned.iter().map(String::as_str).collect();
    if let Some(at) = display_split(&cleaned_refs, display_names)
        && at > start
    {
        end = at;
    }
    let mut kept: Vec<&str> = all[start..end.max(start)].to_vec();
    while kept.len() > 1
        && kept
            .last()
            .is_some_and(|w| LEADING_NOISE.contains(&w.to_lowercase().as_str()))
    {
        kept.pop();
    }
    let kept = kept.join(" ");
    if !kept.is_empty() {
        kept
    } else if !plain.is_empty() {
        plain
    } else if !stem.trim().is_empty() {
        // Nothing but separators ("---", "_"): show them as they are rather than a blank name.
        stem.trim().to_string()
    } else {
        "Look".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display() -> Vec<String> {
        ["gamma24", "rec709", "p3", "rec2020", "hlg", "pq"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    #[test]
    fn folding_ignores_separators_and_case() {
        assert_eq!(fold_alnum("LogC4_to Rec.709-x"), "logc4torec709x");
        assert_eq!(
            tokens("LogC4_to Rec.709"),
            vec!["logc4", "to", "rec", "709"]
        );
    }

    #[test]
    fn path_parts() {
        assert_eq!(file_name("a/b/c.cube"), "c.cube");
        assert_eq!(file_stem("a/b/c.d.cube"), "c.d");
        assert_eq!(file_stem("noext"), "noext");
        assert_eq!(folder_components("a/b/c.cube"), vec!["a", "b"]);
        assert!(folder_components("c.cube").is_empty());
    }

    #[test]
    fn group_folders() {
        assert_eq!(parse_group_folder("3000 Film"), Some((3000, "Film".into())));
        assert_eq!(
            parse_group_folder("3000_Film_Looks"),
            Some((3000, "Film Looks".into()))
        );
        assert_eq!(
            parse_group_folder("3000 - Film"),
            Some((3000, "Film".into()))
        );
        assert_eq!(parse_group_folder("3000-Film"), Some((3000, "Film".into())));
        assert_eq!(
            parse_group_folder("3000"),
            Some((3000, "Group 3000".into()))
        );
        assert_eq!(parse_group_folder("30000 Film"), None);
        // Dates and versions are not groups.
        assert_eq!(parse_group_folder("2026-10-12"), None);
        assert_eq!(parse_group_folder("2026-10-12 looks"), None);
        assert_eq!(parse_group_folder("2024_05 backup"), None);
        assert_eq!(parse_group_folder("2024.1"), None);
        assert_eq!(parse_group_folder("2024-Film"), Some((2024, "Film".into())));
        assert_eq!(parse_group_folder("300 Film"), None);
        assert_eq!(parse_group_folder("Film 3000"), None);
        assert_eq!(
            group_of("ARRI Looks/3000 Film/x.cube"),
            Some((3000, "Film".into()))
        );
        assert_eq!(
            group_of("1000 A/2000 B/x.cube"),
            Some((2000, "B".into())),
            "innermost wins"
        );
        assert_eq!(group_of("3000 Film.cube"), None, "files are not folders");
        assert_eq!(group_of("x.cube"), None);
    }

    proptest::proptest! {
        // A fixed seed: the same cases on every run (see `library::ids` tests for the reason).
        #![proptest_config(proptest::test_runner::Config {
            rng_seed: proptest::test_runner::RngSeed::Fixed(0x594d_4c4f_4f4b_5332),
            ..proptest::test_runner::Config::default()
        })]

        #[test]
        fn name_helpers_never_panic(text in ".{0,200}") {
            let d = display();
            let _ = parse_group_folder(&text);
            let _ = group_of(&text);
            let _ = has_display_rendering(&text, &d);
            let _ = display_target(&text, &d);
            proptest::prop_assert!(!look_name(&text, &d).trim().is_empty());
        }
    }

    #[test]
    fn display_renderings() {
        let d = display();
        assert!(has_display_rendering("LogC4_to_Gamma24_Rec709", &d));
        assert!(has_display_rendering("LogC3-to-Rec.709", &d));
        assert!(has_display_rendering("LogC4 to P3 D65", &d));
        assert!(has_display_rendering("LogC4_to_Rec2020_PQ", &d));
        assert!(has_display_rendering("LogC4_to_HLG", &d));
        assert!(!has_display_rendering("Soft to Warm", &d));
        assert!(!has_display_rendering("LogC4_Film", &d));
        assert!(!has_display_rendering("to", &d));
    }

    #[test]
    fn display_targets_are_canonical() {
        let d = display();
        assert_eq!(
            display_target("LogC4_to_Gamma24_Rec709", &d).as_deref(),
            Some("rec709-g24")
        );
        assert_eq!(
            display_target("LogC4_to_Rec709_Gamma24", &d).as_deref(),
            Some("rec709-g24")
        );
        assert_eq!(
            display_target("LogC3-to-Gamma-24-Rec.709", &d).as_deref(),
            Some("rec709-g24")
        );
        assert_eq!(
            display_target("LogC4_to_P3_D65", &d).as_deref(),
            Some("p3-d65")
        );
        assert_eq!(
            display_target("LogC4_to_Rec2020_PQ_1000nits", &d).as_deref(),
            Some("rec2020-pq-1000nits")
        );
        assert_eq!(display_target("LogC4_Film", &d), None);
    }

    #[test]
    fn look_names_lose_package_noise() {
        let d = display();
        assert_eq!(look_name("Film_Warm", &d), "Film Warm");
        assert_eq!(look_name("LogC4_Film_Warm", &d), "Film Warm");
        assert_eq!(look_name("ARRI_LogC3_Bleach-Bypass", &d), "Bleach Bypass");
        assert_eq!(look_name("3000_Film_Warm", &d), "Film Warm");
        assert_eq!(look_name("LogC3_Film_Warm_to_Rec709", &d), "Film Warm");
        assert_eq!(look_name("Film_Warm_LogC4", &d), "Film Warm");
        assert_eq!(look_name("Warm_LogC3_to_Rec709", &d), "Warm");
        assert_eq!(look_name("Soft to Warm", &d), "Soft to Warm");
        assert_eq!(look_name("LogC4", &d), "LogC4", "never empty");
    }

    #[test]
    fn look_names_are_never_blank() {
        // Found by the property test above: a stem of only separators or white space used to give
        // an empty display name, which then went into the list of looks.
        let d = display();
        assert_eq!(look_name("\u{b}", &d), "Look");
        assert_eq!(look_name("", &d), "Look");
        assert_eq!(look_name("   ", &d), "Look");
        assert_eq!(look_name("---", &d), "---");
        assert_eq!(look_name("_ _", &d), "_ _");
        assert_eq!(look_name("3000", &d), "3000");
    }
}
