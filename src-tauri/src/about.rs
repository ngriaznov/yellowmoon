//! The words of the About panel (`docs/11-ui.md` §Menus): the version, the licence, what the
//! project is built on, the Adobe DNG notice that Adobe's patent licence asks for (`docs/09`
//! §Legal), the acknowledgement the IJG licence asks for (`docs/14`, `NOTICE`) and the line that
//! says the project is not affiliated with ARRI or Sony.
//!
//! The menu's "About Yellowmoon" item shows the system's About panel with this text (macOS:
//! `credits`; elsewhere: `comments`). The in-window dialog that Preferences links to is the
//! frontend's own and repeats the same sentences.

/// The licence line.
pub const LICENCE: &str =
    "Apache License 2.0. Copyright 2026 Mykyta Hriaznov and Yellowmoon contributors.";

/// Adobe's notice for DNG files (`docs/09` §Legal).
pub const DNG_NOTICE: &str = "This product includes DNG technology under license by Adobe.";

/// The acknowledgement the IJG licence of the JPEG encoder's forward DCT asks for.
pub const IJG_NOTICE: &str =
    "This software is based in part on the work of the Independent JPEG Group.";

/// The line that says the project belongs to no camera maker (`docs/11` §Menus).
pub const TRADEMARK_NOTICE: &str = "Not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation.";

/// Third-party components and their licences, as in `NOTICE`.
pub const CREDITS: [&str; 4] = [
    "ALEXA colour matrices from the Academy's aces-dev (AMPAS licence).",
    "Sony camera spectral measurements by Weta Digital, from rawtoaces-data (Apache License 2.0).",
    "ACES 2.0 output transform from aces-output (Apache License 2.0).",
    "The Figtree typeface (SIL Open Font License 1.1).",
];

/// What the project is, in one sentence.
pub const BLURB: &str = "A raw processor for Sony ARW files. Every photo is developed automatically, with ALEXA-style colour. Free and open source. Nothing is saved besides the files you export.";

/// The text under the version in the About panel.
#[must_use]
pub fn about_text() -> String {
    let mut text = String::new();
    text.push_str(BLURB);
    text.push_str("\n\n");
    text.push_str(LICENCE);
    text.push_str("\n\nBuilt with:\n");
    for credit in CREDITS {
        text.push_str("- ");
        text.push_str(credit);
        text.push('\n');
    }
    text.push_str("The full licence texts are in the NOTICE and THIRD-PARTY.md files inside the app.\n\n");
    text.push_str(IJG_NOTICE);
    text.push('\n');
    text.push_str(DNG_NOTICE);
    text.push_str("\n\n");
    text.push_str(TRADEMARK_NOTICE);
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_about_text_carries_the_notices_the_licences_ask_for() {
        let text = about_text();
        assert!(text.contains("This product includes DNG technology under license by Adobe."));
        assert!(
            text.contains(
                "This software is based in part on the work of the Independent JPEG Group."
            )
        );
        assert!(text.contains("Not affiliated with ARRI or Sony."));
        assert!(text.contains("Apache License 2.0"));
        assert!(!text.contains('!'), "no exclamation marks in the copy");
    }

    #[test]
    fn the_notice_file_says_the_same_things() {
        // `NOTICE` ships in the bundle; the sentences of the panel must not drift from it.
        let notice = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../NOTICE"),
        )
        .expect("NOTICE");
        assert!(notice.contains(IJG_NOTICE));
    }
}
