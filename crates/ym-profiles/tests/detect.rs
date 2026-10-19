//! M1-04: camera detection and lookup (`docs/03-camera-profiles.md`, "Detection" and "Lookup").
//!
//! The corpus gate ("every Sony sample is detected as the right body") needs real files, which this
//! environment does not have (`docs/13`); this test covers every body of the registry, the aliases, the
//! disagreement rule, unknown bodies, non-Sony makes and the `--profile` override with the metadata the
//! decoder would hand over.

mod common;

use common::{docs_table, repo_root, store};
use ym_profiles::detect::{CameraMatch, MatchKind};
use ym_profiles::profile::ProfileError;
use ym_profiles::registry::BodyMatch;

#[test]
fn every_body_of_the_registry_is_detected_by_its_model_id() {
    let s = store();
    let table = docs_table();
    assert_eq!(table.len(), 28);
    for b in &s.registry().bodies {
        // Whatever Model string the file carries, the ID decides.
        for model in [b.model.as_str(), "", "garbage"] {
            let m = s.detect("SONY", model, Some(b.model_id));
            assert_eq!(m.profile, b.profile, "{} ({model:?})", b.model);
            assert_eq!(m.body, b.body);
            assert_eq!(m.model, b.model, "the registry's name for the body wins");
            assert_eq!(m.model_id, Some(b.model_id));
            assert_eq!(m.make, "SONY");
            assert_eq!(m.profile_version, s.profile(&b.profile).unwrap().version);
            let expected_how = match b.match_kind {
                BodyMatch::Exact => MatchKind::Exact,
                BodyMatch::SensorFamily => MatchKind::SensorFamily {
                    via: b.via.clone().unwrap(),
                },
                BodyMatch::Default => MatchKind::Default,
            };
            assert_eq!(m.how, expected_how, "{}", b.model);
            // And it agrees with docs/03's table.
            let (profile, kind) = &table[&b.model_id];
            assert_eq!(m.profile.as_str(), profile);
            assert_eq!(
                match &m.how {
                    MatchKind::Exact => "exact",
                    MatchKind::SensorFamily { .. } => "sensor_family",
                    MatchKind::Default => "default",
                    MatchKind::Forced { .. } => unreachable!(),
                },
                kind
            );
            assert_eq!(m.is_stand_in(), b.match_kind != BodyMatch::Exact);
        }
    }
}

#[test]
fn model_strings_work_when_the_id_is_missing() {
    let s = store();
    for b in &s.registry().bodies {
        for spelling in [
            b.model.clone(),
            b.model.to_lowercase(),
            format!("  {}\0\0", b.model),
            format!("Sony {}", b.model),
        ] {
            let m = s.detect("SONY", &spelling, None);
            assert_eq!(m.profile, b.profile, "{spelling:?}");
            assert_eq!(m.body, b.body);
            assert_eq!(m.model, b.model);
            assert_eq!(m.model_id, None, "the ID is reported as the file gave it");
        }
    }
    // Same result with or without the ID, apart from `model_id`.
    let with_id = s.detect("SONY", "ILCE-7C", Some(381));
    let without = s.detect("SONY", "ILCE-7C", None);
    assert_eq!(
        CameraMatch {
            model_id: None,
            ..with_id
        },
        without
    );
}

#[test]
fn the_model_id_wins_when_it_disagrees_with_the_model_string() {
    let s = store();
    // A7R IV's ID with an A7 III Model string: an A7R IV.
    let m = s.detect("SONY", "ILCE-7M3", Some(375));
    assert_eq!(m.profile.as_str(), "sony-ilce-7rm4");
    assert_eq!(m.how, MatchKind::Exact);
    assert_eq!(m.body, "Sony A7R IV");
    assert_eq!(m.model, "ILCE-7RM4");
    // A known string with an ID the registry does not know: the ID wins, so it is an unknown body.
    let m = s.detect("SONY", "ILCE-7M3", Some(60000));
    assert_eq!(m.profile.as_str(), "sony-ilce-7rm4");
    assert_eq!(m.how, MatchKind::Default);
    assert_eq!(m.body, "Sony ILCE-7M3");
    assert_eq!(m.model_id, Some(60000));
}

#[test]
fn the_a1_gets_the_default_profile() {
    let s = store();
    let m = s.detect("SONY", "ILCE-1", Some(384));
    assert_eq!(m.body, "Sony A1");
    assert_eq!(m.model, "ILCE-1");
    assert_eq!(m.profile.as_str(), "sony-ilce-7rm4");
    assert_eq!(m.how, MatchKind::Default);
    assert!(m.is_stand_in());
    assert_eq!(m.profile_version, semver::Version::new(1, 0, 0));
}

#[test]
fn unknown_sony_bodies_borrow_the_default_and_are_never_refused() {
    let s = store();
    for (model, id) in [
        ("ILCE-7M6", Some(999)),
        ("ILCE-7M6", None),
        ("", Some(1)),
        ("", None),
        ("ILCE-7M6", Some(0)),
        ("x", Some(u16::MAX)),
    ] {
        let m = s.detect("SONY", model, id);
        assert_eq!(m.profile, s.registry().default_profile, "{model:?} {id:?}");
        assert_eq!(m.how, MatchKind::Default);
        assert!(!m.body.is_empty());
        assert_eq!(m.model_id, id);
    }
    assert_eq!(s.detect("SONY", "ILCE-7M6", None).body, "Sony ILCE-7M6");
    assert_eq!(s.detect("SONY", "", None).body, "Sony");
    // No Make at all: treated as Sony, so the ID still finds the body.
    assert_eq!(s.detect("", "", Some(363)).body, "Sony A7 III");
    assert_eq!(s.detect("", "", Some(363)).make, "SONY");
}

#[test]
fn non_sony_arw_files_get_the_default_profile() {
    // Hasselblad Lunar and Stellar are rebadged Sony bodies; docs/03 sends them to the default profile.
    let s = store();
    for (make, model, id) in [
        ("HASSELBLAD", "Lunar", None),
        ("HASSELBLAD", "Stellar II", None),
        ("Hasselblad", "Lunar", Some(363)), // even with a Sony-looking ID: the Make is not Sony
        ("Canon", "EOS R5", None),
    ] {
        let m = s.detect(make, model, id);
        assert_eq!(m.profile, s.registry().default_profile, "{make} {model}");
        assert_eq!(m.how, MatchKind::Default);
        assert_eq!(m.make, make);
        assert_eq!(m.model, model);
        assert!(m.body.starts_with(&make[..1].to_uppercase()), "{}", m.body);
    }
    assert_eq!(
        s.detect("HASSELBLAD", "Lunar", None).body,
        "Hasselblad Lunar"
    );
}

#[test]
fn forcing_a_profile() {
    let s = store();
    // Forcing the profile detection chose changes nothing.
    let plain = s.detect("SONY", "ILCE-7M3", Some(363));
    assert_eq!(
        s.detect_forced("SONY", "ILCE-7M3", Some(363), "sony-ilce-7m3")
            .unwrap(),
        plain
    );
    // Forcing another: same body, the forced profile and version, and a Forced marker that remembers
    // what detection chose.
    let forced = s
        .detect_forced("SONY", "ILCE-7M3", Some(363), " sony-ilce-6400 ")
        .unwrap();
    assert_eq!(forced.profile.as_str(), "sony-ilce-6400");
    assert_eq!(
        forced.how,
        MatchKind::Forced {
            detected: plain.profile.clone()
        }
    );
    assert_eq!(forced.body, plain.body);
    assert_eq!(forced.model_id, plain.model_id);
    assert!(forced.is_stand_in());
    assert_eq!(
        forced.profile_version,
        s.profile_by_name("sony-ilce-6400").unwrap().version
    );
    // Forcing the default on a body whose default is something else works for unknown bodies too.
    let forced = s
        .detect_forced("HASSELBLAD", "Lunar", None, "sony-ilce-9")
        .unwrap();
    assert_eq!(forced.profile.as_str(), "sony-ilce-9");
    assert!(matches!(forced.how, MatchKind::Forced { .. }));
    // An unknown profile lists the valid ones.
    let err = s
        .detect_forced("SONY", "ILCE-7M3", Some(363), "sony-ilce-12")
        .unwrap_err();
    match &err {
        ProfileError::UnknownProfile {
            requested,
            available,
        } => {
            assert_eq!(requested, "sony-ilce-12");
            for id in s.profile_ids() {
                assert!(available.contains(id.as_str()), "{available}");
            }
        }
        other => panic!("{other}"),
    }
    assert!(err.to_string().contains("sony-dsc-rx100m4"));
    // `resolve` finds the profile a match names.
    assert_eq!(s.resolve(&forced).id.as_str(), "sony-ilce-9");
}

#[test]
fn camera_copy_matches_the_ui_spec() {
    let s = store();
    // docs/11: the default-profile sentence, with the body name substituted.
    let a1 = s.camera_copy(&s.detect("SONY", "ILCE-1", Some(384)));
    assert_eq!(a1.detected, "Sony A1 (ILCE-1)");
    assert_eq!(a1.profile, "A7R IV");
    assert_eq!(a1.automatic_row, "A1 · A7R IV profile");
    assert!(a1.stand_in);
    let sentence = "The A1 has no measured profile yet, so it borrows the A7R IV. Eleven Sony cameras have their own, and cameras that share their sensors use them.";
    assert_eq!(a1.note.as_deref(), Some(sentence));
    // The spec and the mockup quote that sentence; the generated text must be in both.
    let ui = std::fs::read_to_string(repo_root().join("docs/11-ui.md")).unwrap();
    assert!(
        ui.contains(sentence),
        "docs/11 no longer quotes the camera card sentence"
    );
    let mockup = std::fs::read_to_string(repo_root().join("design/mockups/photos.html")).unwrap();
    assert!(
        mockup.contains(sentence),
        "the mockup no longer quotes the camera card sentence"
    );
    assert!(mockup.contains("A1 · A7R IV profile"));

    // docs/11: the sensor-family sentence.
    let a7c = s.camera_copy(&s.detect("SONY", "ILCE-7C", Some(381)));
    assert_eq!(a7c.detected, "Sony A7C (ILCE-7C)");
    assert_eq!(a7c.profile, "A7 III");
    assert_eq!(a7c.automatic_row, "A7C · A7 III profile");
    assert_eq!(
        a7c.note.as_deref(),
        Some("The A7C shares its sensor with the A7 III, so it uses that profile.")
    );
    assert!(a7c.stand_in);

    // A measured body: no stand-in tag, no explanation, the profile name alone on the Automatic card.
    let a7r4 = s.camera_copy(&s.detect("SONY", "ILCE-7RM4", Some(375)));
    assert_eq!(a7r4.detected, "Sony A7R IV (ILCE-7RM4)");
    assert_eq!(a7r4.automatic_row, "A7R IV");
    assert!(!a7r4.stand_in);
    assert_eq!(a7r4.note, None);
    let a7s3 = s.camera_copy(&s.detect("SONY", "ILCE-7SM3", Some(383)));
    assert_eq!(a7s3.automatic_row, "A7S III");

    // Unknown and non-Sony bodies read like the default case.
    let lunar = s.camera_copy(&s.detect("HASSELBLAD", "Lunar", None));
    // The model is shown in brackets only when the body name does not already say it.
    assert_eq!(lunar.detected, "Hasselblad Lunar");
    assert_eq!(
        s.camera_copy(&s.detect("SONY", "ILCE-7M6", None)).detected,
        "Sony ILCE-7M6"
    );
    assert_eq!(
        s.camera_copy(&s.detect("SONY", "", Some(393))).detected,
        "Sony ZV-E1"
    );
    assert_eq!(
        s.camera_copy(&s.detect("SONY", "", Some(385))).detected,
        "Sony FX3 (ILME-FX3)"
    );
    assert!(lunar.note.unwrap().starts_with(
        "The Hasselblad Lunar has no measured profile yet, so it borrows the A7R IV."
    ));

    // A forced profile says so.
    let forced = s
        .detect_forced("SONY", "ILCE-7RM4", Some(375), "sony-ilce-6400")
        .unwrap();
    let copy = s.camera_copy(&forced);
    assert_eq!(copy.profile, "A6400");
    assert_eq!(
        copy.note.as_deref(),
        Some(
            "The profile was chosen on the command line, so the A7R IV uses the A6400 instead of the A7R IV."
        )
    );
    assert!(copy.stand_in);
}

#[test]
fn matches_serialise_for_info_json() {
    let s = store();
    let m = s.detect("SONY", "ILCE-7C", Some(381));
    let json = serde_json::to_value(&m).unwrap();
    assert_eq!(json["profile"], "sony-ilce-7m3");
    assert_eq!(json["profile_version"], "1.0.0");
    assert_eq!(json["how"]["kind"], "sensor_family");
    assert_eq!(json["how"]["via"], "Sony A7 III");
    assert_eq!(json["model_id"], 381);
    let back: CameraMatch = serde_json::from_value(json).unwrap();
    assert_eq!(back, m);
}

#[test]
fn odd_strings_and_every_model_id_never_panic_and_always_name_a_profile() {
    let s = store();
    // Multi-byte characters across the "Sony " prefix boundary (byte 5 falls inside a character), NULs,
    // whitespace, very long and empty strings.
    let long = "x".repeat(10_000);
    let makes = [
        "SONY",
        "sony",
        "",
        "Sony\0",
        "  SONY  ",
        "ＳＯＮＹ",
        "Sönÿ",
        "é",
        long.as_str(),
    ];
    let models = [
        "ILCE-7M3",
        "",
        "Sony",
        "Sony ",
        "Sonyé ILCE-7M3",
        "Sön ILCE-7M3",
        "\0\0",
        " \t\n ",
        "ＩＬＣＥ-７Ｍ３",
        "ILCE-7M3\0garbage",
        long.as_str(),
    ];
    for make in makes {
        for model in models {
            for id in [None, Some(0), Some(363), Some(u16::MAX)] {
                let m = s.detect(make, model, id);
                assert!(s.profile(&m.profile).is_some(), "{make:?} {model:?} {id:?}");
                assert_eq!(m.model_id, id);
                let copy = s.camera_copy(&m);
                assert!(!copy.detected.is_empty() && !copy.automatic_row.is_empty());
            }
        }
    }
    // Every possible model ID resolves to a profile that exists, and only the registry's 28 IDs are
    // anything but the default.
    let known: std::collections::BTreeSet<u16> =
        s.registry().bodies.iter().map(|b| b.model_id).collect();
    for id in 0..=u16::MAX {
        let m = s.detect("SONY", "ILCE-7M3", Some(id));
        assert!(s.profile(&m.profile).is_some());
        if !known.contains(&id) {
            assert_eq!(m.how, MatchKind::Default, "model ID {id}");
            assert_eq!(m.profile, s.registry().default_profile, "model ID {id}");
        }
    }
}

#[test]
fn the_store_can_be_shared_between_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ym_profiles::registry::ProfileStore>();
    assert_send_sync::<CameraMatch>();
    assert_send_sync::<ym_profiles::profile::Profile>();
    // One store, four threads detecting at once.
    let s = store();
    std::thread::scope(|scope| {
        for id in [363_u16, 375, 384, 410] {
            scope.spawn(move || {
                assert_eq!(s.detect("SONY", "", Some(id)).model_id, Some(id));
            });
        }
    });
}

#[test]
fn detection_carries_the_bodys_white_scale() {
    // docs/03, White scale of stand-in bodies: the scale belongs to the registry body.
    let s = store();
    let reg = s.registry();
    for b in &reg.bodies {
        let m = s.detect("SONY", &b.model, Some(b.model_id));
        assert_eq!(m.white_scale, b.white_scale, "{}", b.model);
        // The same through the Model string.
        assert_eq!(
            s.detect("SONY", &b.model, None).white_scale,
            b.white_scale,
            "{}",
            b.model
        );
    }
    // The A1 borrows the A7R IV and needs the scale; the A7R IV itself does not.
    let a1 = s.detect("SONY", "ILCE-1", Some(384));
    assert_eq!(a1.profile.as_str(), "sony-ilce-7rm4");
    assert!(a1.is_stand_in() && a1.white_scale != [1.0; 3]);
    assert_eq!(a1.white_scale[1], 1.0);
    assert_eq!(
        s.detect("SONY", "ILCE-7RM4", Some(375)).white_scale,
        [1.0; 3]
    );
    // A body the registry does not list and a file that is not Sony's: nothing to correct.
    assert_eq!(
        s.detect("SONY", "ILCE-7M6", Some(60000)).white_scale,
        [1.0; 3]
    );
    assert_eq!(
        s.detect("HASSELBLAD", "Lunar", Some(384)).white_scale,
        [1.0; 3]
    );
    // A stand-in without files carries the scale of its published colour matrix (docs/03, kind `matrix`).
    let a9_iii = s.detect("SONY", "ILCE-9M3", Some(392));
    assert!(a9_iii.is_stand_in() && a9_iii.white_scale != [1.0; 3]);
    assert_eq!(a9_iii.white_scale[1], 1.0);
    // Forcing the profile the registry chose keeps it; forcing another one drops it: the scale was fitted to the
    // curves of the profile the registry gave the body.
    let same = s
        .detect_forced("SONY", "ILCE-1", Some(384), "sony-ilce-7rm4")
        .unwrap();
    assert_eq!(same, a1);
    let other = s
        .detect_forced("SONY", "ILCE-1", Some(384), "sony-ilce-7m3")
        .unwrap();
    assert_eq!(other.white_scale, [1.0; 3]);
    assert!(matches!(other.how, MatchKind::Forced { .. }));
    // It is part of the match as it is serialised.
    let json = serde_json::to_value(&a1).unwrap();
    assert_eq!(json["white_scale"][1], 1.0);
    let back: CameraMatch = serde_json::from_value(json).unwrap();
    assert_eq!(back, a1);
}
