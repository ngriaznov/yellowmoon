//! How a failed write is told (`docs/26` P2-1, task H-05): the kind (`disk_full`, `destination` or
//! `write`), the sentence for the app, the sentence for `ym`, and the operating system's text kept in
//! `detail`.

use std::io;
use std::path::Path;

use ym_core::export::ExportError;

fn io_error(code: i32) -> ExportError {
    ExportError::Write(ym_export::ExportError::io(
        "/Volumes/Photos/Lisbon",
        io::Error::from_raw_os_error(code),
    ))
}

fn encoder_error(code: i32) -> ExportError {
    ExportError::Write(
        ym_export::ExportError::write("PNG", io::Error::from_raw_os_error(code))
            .with_folder(Path::new("/Volumes/Photos/Lisbon")),
    )
}

#[test]
fn each_class_has_a_kind_and_two_sentences() {
    // (errno on Linux and macOS, kind, app sentence, `ym` sentence)
    let cases = [
        (
            28,
            "disk_full",
            "The disk is full, so the export stopped. Free some space, then export the photos that are left.",
            "Stopped: the disk holding /Volumes/Photos/Lisbon is full. Free some space, then run the command again with --skip-existing.",
        ),
        (
            30,
            "destination",
            "Lisbon is read-only. Choose another folder.",
            "/Volumes/Photos/Lisbon is read-only. Choose another folder with --out.",
        ),
        (
            2,
            "destination",
            "Yellowmoon can't reach Lisbon. If it is on a drive, check that the drive is connected, then export again.",
            "/Volumes/Photos/Lisbon is not there any more. Check the path or the drive, then run the command again.",
        ),
        (
            13,
            "destination",
            "Yellowmoon isn't allowed to save to Lisbon. Choose another folder, or allow access in System Settings, under Privacy & Security.",
            "Yellowmoon isn't allowed to write to /Volumes/Photos/Lisbon. Choose another folder with --out.",
        ),
        (
            1,
            "destination",
            "Yellowmoon isn't allowed to save to Lisbon. Choose another folder, or allow access in System Settings, under Privacy & Security.",
            "Yellowmoon isn't allowed to write to /Volumes/Photos/Lisbon. Choose another folder with --out.",
        ),
    ];
    for (code, kind, app, cli) in cases {
        for e in [io_error(code), encoder_error(code)] {
            assert_eq!(e.kind(), kind, "errno {code}");
            assert_eq!(e.to_string(), app, "errno {code}");
            assert_eq!(e.cli_message(), cli, "errno {code}");
            assert!(!e.to_string().contains("os error"), "{e}");
            assert!(!e.cli_message().contains("os error"), "{e}");
            assert!(
                e.detail().is_some_and(|d| d.contains("os error")),
                "the detail keeps the system's text: {:?}",
                e.detail()
            );
        }
    }
}

#[test]
fn any_other_write_failure_is_kind_write_and_says_the_same_to_both() {
    let e = ExportError::Write(ym_export::ExportError::write(
        "TIFF",
        io::Error::from(io::ErrorKind::InvalidData),
    ));
    assert_eq!(e.kind(), "write");
    let sentence = "Couldn't write the TIFF. Check that the folder is still there and that you can save to it.";
    assert_eq!(e.to_string(), sentence);
    assert_eq!(e.cli_message(), sentence);
    assert!(e.detail().is_some());
    // An error that is not about writing has no detail.
    assert_eq!(ExportError::Options("x".into()).detail(), None);
}
