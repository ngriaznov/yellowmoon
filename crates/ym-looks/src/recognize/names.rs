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

/// The look code ARRI starts the file names of its Look Library with: four digits, the first of
/// them the group (`1110 Black and White`, `3250-Cross`, `8120-20s`), then a space, dash,
/// underscore or dot. What follows must not be only digits and separators, which is a date or a
/// version (`2024-05-12`, `2024_05`), not a title.
#[must_use]
pub fn look_code(stem: &str) -> Option<u32> {
    let bytes = stem.as_bytes();
    if bytes.len() < 5 || !bytes[..4].iter().all(u8::is_ascii_digit) || bytes[0] == b'0' {
        return None;
    }
    if !matches!(bytes[4], b' ' | b'-' | b'_' | b'.') {
        return None;
    }
    let rest = &stem[5..];
    if rest
        .chars()
        .all(|c| c.is_ascii_digit() || c.is_whitespace() || matches!(c, '-' | '_' | '.'))
    {
        return None;
    }
    stem[..4].parse().ok()
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
/// `HLG`, `PQ`. Whole words only, so `Pqrs`, `P3ach` and `Hlgx` are not display names:
///
/// * one to three words are joined and compared with the folded names, so `Rec 709`, `Rec.709`
///   and `DCI P3` count;
/// * a single word may also be a display name followed by digits only (`PQ1000`).
///
/// This is the target side of ARRI's naming (`LogC4-to-Gamma24_Rec709-D65_v1_65`): what follows
/// `to`. A log-to-log name (`LogC4-to-LogC4`) names no display.
fn starts_display(after_to: &[&str], display_names: &[String]) -> bool {
    let names: Vec<String> = display_names
        .iter()
        .map(|d| fold_alnum(d))
        .filter(|d| !d.is_empty())
        .collect();
    let mut joined = String::new();
    let mut seen = 0_usize;
    for word in after_to {
        let folded = fold_alnum(word);
        if folded.is_empty() {
            continue;
        }
        joined.push_str(&folded);
        if names.contains(&joined) {
            return true;
        }
        if seen == 0
            && names.iter().any(|n| {
                folded
                    .strip_prefix(n.as_str())
                    .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
            })
        {
            return true;
        }
        seen += 1;
        if seen == 3 {
            break;
        }
    }
    false
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

/// `v1`, `V12`: the version word ARRI puts near the end of its LUT names.
fn is_version_word(word: &str) -> bool {
    word.len() >= 2
        && word.starts_with(['v', 'V'])
        && word[1..].bytes().all(|b| b.is_ascii_digit())
}

/// The mesh size ARRI puts at the very end of its LUT names (`..._v1_65.cube`).
fn is_mesh_word(word: &str) -> bool {
    matches!(word, "17" | "33" | "65" | "129")
}

/// The display name of an ARRI look from its file stem: underscores and dashes become spaces; a
/// leading group number, a leading `ARRI`/`LogC4`/`LogC3`/`AWG4`/`AWG3`/`EI800` (with the `to` and
/// the encoding after it, as in `ARRI_LogC4-to-LogC4_Neutral`), a trailing version and mesh size
/// (`v1_65`) and a trailing `to <display>` are dropped. Falls back to the plain stem when nothing
/// would be left.
#[must_use]
pub fn look_name(stem: &str, display_names: &[String]) -> String {
    let plain = words(stem).join(" ");
    let all = words(stem);
    let is_noise = |w: &str| LEADING_NOISE.contains(&w.to_lowercase().as_str());
    let mut start = 0;
    while let Some(word) = all.get(start) {
        let is_group_number = word.len() == 4 && word.bytes().all(|b| b.is_ascii_digit());
        if is_group_number || is_noise(word) {
            start += 1;
        } else if word.eq_ignore_ascii_case("to") && all.get(start + 1).is_some_and(|n| is_noise(n))
        {
            start += 2;
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
    while kept.len() > 1 {
        let last = kept[kept.len() - 1];
        if is_noise(last) || is_version_word(last) {
            kept.pop();
        } else if is_mesh_word(last) && kept.len() > 2 && is_version_word(kept[kept.len() - 2]) {
            kept.truncate(kept.len() - 2);
        } else {
            break;
        }
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
    fn look_codes() {
        assert_eq!(look_code("1110 Black and White"), Some(1110));
        assert_eq!(look_code("3250-Cross"), Some(3250));
        assert_eq!(look_code("8120-20s"), Some(8120), "a decade after the code is a title");
        assert_eq!(look_code("9211_Day_for_Night"), Some(9211));
        assert_eq!(look_code("1120 Black and White - Night sky"), Some(1120));
        // Dates, versions, short or long numbers and names that merely hold digits are not codes.
        assert_eq!(look_code("2024-05-12"), None);
        assert_eq!(look_code("2024_05"), None);
        assert_eq!(look_code("2024.1"), None);
        assert_eq!(look_code("2024 "), None);
        assert_eq!(look_code("2024"), None);
        assert_eq!(look_code("20245 Film"), None);
        assert_eq!(look_code("300 Film"), None);
        assert_eq!(look_code("0110 Film"), None);
        assert_eq!(look_code("Film 3000"), None);
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
    fn display_renderings_follow_arris_naming() {
        // `<Creator>_<LogC>-to-<Target>_<ColourSpace>-<White>_..._v<N>_<mesh>.cube`
        let d = display();
        for name in [
            "ARRI_LogC4-to-Gamma24_Rec709-D65_v1_33",
            "ARRI_LogC4-to-Gamma24_Rec709-D65_v1_65",
            "ARRI_LogC4-to-HLG_1K_Rec2020-D65_DW203_v1_65",
            "ARRI_LogC3-to-Rec.709_v1_33",
            "ARRI_LogC4-to-Rec 2020_PQ_v1_65",
            "ARRI_LogC4-to-PQ1000_v1_65",
            "ARRI_LogC4-to-DCI P3_v1_65",
        ] {
            let mut names = d.clone();
            names.push("dcip3".to_string());
            assert!(has_display_rendering(name, &names), "{name}");
        }
        // Log to log names no display, whatever else the name holds.
        for name in [
            "ARRI_LogC4-to-LogC4_Neutral_v1_65",
            "ARRI_LogC4-to-LogC4_Kodak_2383_v1_65",
            "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data",
            "ARRI_LogC4-to-LogC4_WVO_v1_33",
        ] {
            assert!(!has_display_rendering(name, &d), "{name}");
        }
    }

    #[test]
    fn display_names_are_whole_words_not_prefixes() {
        let d = display();
        // The short names (p3, pq, hlg) as part of other words, before or after a "to".
        for name in [
            "Pale to Pqrs",
            "Film to P3ach",
            "Ember to Hlgx",
            "Warm P3 Glow",
            "PQ Noir",
            "Spq to Soft",
            "Soft to Pastel",
            "Day to Rec7090x",
            "Toned P3",
            "Sepia to Hlg2x",
        ] {
            assert!(!has_display_rendering(name, &d), "{name}");
        }
        assert!(has_display_rendering("Sepia to PQ", &d));
        assert!(has_display_rendering("Sepia_to_Rec_709", &d));
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
    fn look_names_lose_arris_naming_noise() {
        let d = display();
        assert_eq!(look_name("ARRI_LogC4-to-LogC4_Neutral_v1_65", &d), "Neutral");
        assert_eq!(
            look_name("ARRI_LogC4-to-LogC4_Kodak 2383_v2_33", &d),
            "Kodak 2383"
        );
        assert_eq!(look_name("Warm_Glow_v3", &d), "Warm Glow");
        // A number that is part of the name stays; so does a mesh size with no version before it.
        assert_eq!(look_name("Kodak_2383", &d), "Kodak 2383");
        assert_eq!(look_name("Film_65", &d), "Film 65");
        assert_eq!(look_name("Film_v1_65_v2", &d), "Film");
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
