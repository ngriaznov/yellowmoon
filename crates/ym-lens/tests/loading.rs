//! Loading and checking the lens folder (`docs/21` section 6): the synthetic folder loads, every kind of damage is
//! refused with a `LensError`, and the real `data/lenses/` loads when it exists.

mod common;

use std::path::Path;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, serialize_to_file};
use serde_json::{Value, json};
use ym_lens::test_support::{
    psf_table, recipe, spikes_table, write_sums, write_synthetic_lenses, write_tables,
};
use ym_lens::{LensError, LensKind, LensLibrary};

/// A fresh synthetic folder.
fn folder() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    write_synthetic_lenses(dir.path()).expect("write");
    dir
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("read")).expect("json")
}

fn write_json(path: &Path, v: &Value) {
    std::fs::write(path, serde_json::to_string_pretty(v).expect("text")).expect("write");
}

/// Changes `<dir>/<file>` (JSON) with `edit`, then re-hashes so that only the content is wrong.
fn damage_json(dir: &Path, file: &str, edit: impl FnOnce(&mut Value)) {
    let path = dir.join(file);
    let mut v = read_json(&path);
    edit(&mut v);
    write_json(&path, &v);
    write_sums(dir).expect("sums");
}

fn load(dir: &Path) -> Result<LensLibrary, LensError> {
    LensLibrary::from_dir(dir)
}

#[test]
fn the_synthetic_folder_loads() {
    let dir = folder();
    let lib = load(dir.path()).expect("loads");
    let kinds: Vec<LensKind> = lib.lenses().iter().map(|l| l.kind).collect();
    assert_eq!(kinds, LensKind::ALL, "index order is the menu order");
    for info in lib.lenses() {
        let rc = recipe(info.kind);
        assert_eq!(info.version, 1);
        assert_eq!((info.start_flares, info.start_edges), rc.start);
        assert!(!info.name.is_empty() && !info.line.is_empty() && !info.design_source.is_empty());
        let data = lib.get(info.kind).expect("lens");
        assert_eq!(data.field.radii.len(), 9);
        assert_eq!(data.field.relative_illumination[0], 1.0);
        for r in 0..9 {
            for j in 0..8 {
                let sum: f64 = data.psf(r, j).iter().map(|v| f64::from(*v)).sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
        for j in 0..8 {
            let sum: f64 = data.spikes(j).iter().map(|v| f64::from(*v)).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
    assert_eq!(lib.attachment().q_max, 2.0);
    assert_eq!(lib.get(LensKind::Default).unwrap().swirl_max(), 0.0);
    assert!(lib.get(LensKind::Petzval).unwrap().swirl_max() > 0.0);
    // `load` reads `<root>/lenses`.
    let root = tempfile::tempdir().expect("temp dir");
    write_synthetic_lenses(&root.path().join("lenses")).expect("write");
    assert_eq!(
        LensLibrary::load(root.path()).expect("load").lenses().len(),
        7
    );
}

#[test]
fn a_changed_file_fails_its_digest() {
    let dir = folder();
    let path = dir.path().join("vintage/lens.json");
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push(' ');
    std::fs::write(&path, text).unwrap();
    assert!(
        matches!(load(dir.path()), Err(LensError::Checksum { file, .. }) if file == "vintage/lens.json")
    );
}

#[test]
fn a_missing_file_is_an_error() {
    let dir = folder();
    std::fs::remove_file(dir.path().join("petzval/tables.safetensors")).unwrap();
    assert!(matches!(load(dir.path()), Err(LensError::Io { .. })));

    let dir = folder();
    std::fs::remove_file(dir.path().join("SHA256SUMS")).unwrap();
    assert!(matches!(load(dir.path()), Err(LensError::Io { .. })));

    let dir = folder();
    std::fs::remove_file(dir.path().join("anamorphic.json")).unwrap();
    assert!(matches!(load(dir.path()), Err(LensError::Io { .. })));

    // Listed in SHA256SUMS but gone (README.md is read only to be checked).
    let dir = folder();
    std::fs::remove_file(dir.path().join("README.md")).unwrap();
    assert!(matches!(load(dir.path()), Err(LensError::Io { .. })));

    assert!(matches!(
        load(Path::new("/nonexistent/lenses")),
        Err(LensError::Io { .. })
    ));
}

#[test]
fn a_file_without_a_line_in_sha256sums_is_refused() {
    let dir = folder();
    let sums = std::fs::read_to_string(dir.path().join("SHA256SUMS")).unwrap();
    let kept: String = sums
        .lines()
        .filter(|l| !l.ends_with(" sonnar/lens.json"))
        .map(|l| format!("{l}\n"))
        .collect();
    std::fs::write(dir.path().join("SHA256SUMS"), kept).unwrap();
    assert!(
        matches!(load(dir.path()), Err(LensError::MissingChecksum { file }) if file == "sonnar/lens.json")
    );
}

#[test]
fn malformed_sha256sums_lines_are_refused() {
    for bad in [
        "nothex  index.json\n",
        "abc  index.json\n",
        &format!("{}  ../outside\n", "0".repeat(64)),
        &format!("{}  /etc/passwd\n", "0".repeat(64)),
    ] {
        let dir = folder();
        std::fs::write(dir.path().join("SHA256SUMS"), bad).unwrap();
        assert!(
            matches!(load(dir.path()), Err(LensError::Invalid { .. })),
            "{bad}"
        );
    }
}

#[test]
fn index_errors() {
    // An unknown lens id.
    let dir = folder();
    damage_json(dir.path(), "index.json", |v| {
        v["lenses"][2] = json!("leica")
    });
    assert!(matches!(load(dir.path()), Err(LensError::UnknownLens(id)) if id == "leica"));

    // A lens listed twice, and a lens left out.
    let dir = folder();
    damage_json(dir.path(), "index.json", |v| {
        v["lenses"][1] = json!("default")
    });
    assert!(matches!(load(dir.path()), Err(LensError::Invalid { .. })));
    let dir = folder();
    damage_json(dir.path(), "index.json", |v| {
        v["lenses"].as_array_mut().unwrap().pop();
    });
    assert!(matches!(load(dir.path()), Err(LensError::Invalid { .. })));

    // Another version, a path that leaves the folder.
    let dir = folder();
    damage_json(dir.path(), "index.json", |v| v["version"] = json!(2));
    assert!(matches!(load(dir.path()), Err(LensError::Invalid { .. })));
    let dir = folder();
    damage_json(dir.path(), "index.json", |v| {
        v["attachment"] = json!("../anamorphic.json")
    });
    assert!(matches!(load(dir.path()), Err(LensError::Invalid { .. })));

    // Not JSON of that shape.
    let dir = folder();
    std::fs::write(dir.path().join("index.json"), "[1, 2]").unwrap();
    write_sums(dir.path()).unwrap();
    assert!(matches!(load(dir.path()), Err(LensError::Json { .. })));
}

#[test]
fn the_provenance_statement_must_be_exact() {
    for file in ["default/lens.json", "anamorphic.json"] {
        let dir = folder();
        damage_json(dir.path(), file, |v| {
            v["provenance"]["statement"] =
                json!("Synthetic, computed from public lens designs, measured.");
        });
        assert!(
            matches!(load(dir.path()), Err(LensError::Invalid { file: f, .. }) if f == file),
            "{file}"
        );
    }
}

#[test]
fn the_lens_id_must_match_its_folder() {
    let dir = folder();
    damage_json(dir.path(), "triplet/lens.json", |v| {
        v["id"] = json!("sonnar")
    });
    assert!(matches!(load(dir.path()), Err(LensError::Invalid { .. })));
}

#[test]
fn a_lens_file_without_astigmatism_reads_as_none() {
    // Data written before the field swirl have no `field.astigmatism`: they load, with zeros.
    let dir = folder();
    damage_json(dir.path(), "petzval/lens.json", |v| {
        v["field"].as_object_mut().unwrap().remove("astigmatism");
    });
    let lib = load(dir.path()).expect("loads");
    let lens = lib.get(LensKind::Petzval).unwrap();
    assert_eq!(lens.field.astigmatism, vec![0.0; 9]);
    assert_eq!(lens.astigmatism_max(), 0.0);
    // And a file with one reads it.
    let dir = folder();
    damage_json(dir.path(), "vintage/lens.json", |v| {
        v["field"]["astigmatism"] = json!([0, 0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012]);
    });
    let lib = load(dir.path()).expect("loads");
    assert_eq!(lib.get(LensKind::Vintage).unwrap().astigmatism_max(), 0.012);
}

#[test]
fn bad_numbers_are_refused() {
    type Edit = fn(&mut Value);
    let cases: Vec<(&str, &str, Edit)> = vec![
        ("wrong count of radii", "default/lens.json", |v| {
            v["field"]["radii"].as_array_mut().unwrap().pop();
        }),
        ("radii not increasing", "default/lens.json", |v| {
            v["field"]["radii"][3] = json!(0.9)
        }),
        ("radii not from 0 to 1", "default/lens.json", |v| {
            v["field"]["radii"][8] = json!(1.2)
        }),
        ("T(0) is not 1", "default/lens.json", |v| {
            v["field"]["relative_illumination"][0] = json!(0.99)
        }),
        ("T rises", "default/lens.json", |v| {
            v["field"]["relative_illumination"][5] = json!(0.99)
        }),
        ("T above 1", "default/lens.json", |v| {
            v["field"]["relative_illumination"][1] = json!(1.1)
        }),
        ("a width of 0", "default/lens.json", |v| {
            v["field"]["psf_half_width"][2] = json!(0.0)
        }),
        ("a null number", "default/lens.json", |v| {
            v["field"]["distortion"]["k1"] = Value::Null
        }),
        ("swirl at the centre", "petzval/lens.json", |v| {
            v["field"]["swirl"][0] = json!(0.1)
        }),
        ("astigmatism at the centre", "petzval/lens.json", |v| {
            v["field"]["astigmatism"][0] = json!(0.001)
        }),
        ("astigmatism that falls", "petzval/lens.json", |v| {
            v["field"]["astigmatism"] = json!([0, 0, 0, 0, 0.01, 0.02, 0.01, 0.02, 0.03])
        }),
        ("a negative astigmatism", "petzval/lens.json", |v| {
            v["field"]["astigmatism"][3] = json!(-0.001)
        }),
        ("astigmatism of 8 values", "petzval/lens.json", |v| {
            v["field"]["astigmatism"].as_array_mut().unwrap().pop();
        }),
        ("a negative cat's-eye", "petzval/lens.json", |v| {
            v["field"]["cats_eye"][4] = json!(-0.1)
        }),
        ("rim of 15 samples", "default/lens.json", |v| {
            v["bokeh"]["rim"].as_array_mut().unwrap().pop();
        }),
        ("rim with another mean", "default/lens.json", |v| {
            v["bokeh"]["rim"][0] = json!(5.0)
        }),
        ("start above 100", "default/lens.json", |v| {
            v["start"]["flares"] = json!(101.0)
        }),
        ("a ghost coating of 30 samples", "classic/lens.json", |v| {
            v["flare"]["ghosts"][0]["coating"]
                .as_array_mut()
                .unwrap()
                .pop();
        }),
        ("a ghost fraction above 1", "classic/lens.json", |v| {
            v["flare"]["ghosts"][0]["fraction"] = json!(1.5)
        }),
        ("fractions adding up past 1", "classic/lens.json", |v| {
            v["flare"]["glow"]["fraction"] = json!(0.9)
        }),
        ("two blades", "classic/lens.json", |v| {
            v["flare"]["aperture"]["blades"] = json!(2)
        }),
        ("starburst past 0.08", "classic/lens.json", |v| {
            v["flare"]["starburst"]["radius"] = json!(0.2)
        }),
        ("chroma of 7", "classic/lens.json", |v| {
            v["flare"]["glow"]["chroma"].as_array_mut().unwrap().pop();
        }),
        ("version 0", "classic/lens.json", |v| {
            v["version"] = json!(0)
        }),
        ("q_max of 1", "anamorphic.json", |v| v["q_max"] = json!(1.0)),
        ("a tint of 30 samples", "anamorphic.json", |v| {
            v["streak"]["tint"].as_array_mut().unwrap().pop();
        }),
        ("a negative kappa", "anamorphic.json", |v| {
            v["field"]["kappa"] = json!(-0.1)
        }),
        ("a bend past the bounds", "anamorphic.json", |v| {
            v["bend"]["mu0"] = json!(3.0)
        }),
    ];
    for (what, file, edit) in cases {
        let dir = folder();
        damage_json(dir.path(), file, edit);
        let err = load(dir.path()).expect_err(what);
        assert!(
            matches!(err, LensError::Invalid { .. } | LensError::Json { .. }),
            "{what}: {err}"
        );
    }
    // A number outside f64 is not valid JSON for the reader.
    let dir = folder();
    let path = dir.path().join("default/lens.json");
    let text =
        std::fs::read_to_string(&path)
            .unwrap()
            .replacen("\"k1\": -0.01", "\"k1\": 1e999", 1);
    std::fs::write(&path, text).unwrap();
    write_sums(dir.path()).unwrap();
    assert!(matches!(load(dir.path()), Err(LensError::Json { .. })));
}

#[test]
fn the_bokeh_table_is_read_under_either_name() {
    // docs/21 names it `bokeh.rim` in the file format and `bokeh.edge` in the stage text.
    let dir = folder();
    damage_json(dir.path(), "default/lens.json", |v| {
        let rim = v["bokeh"]["rim"].take();
        v["bokeh"] = json!({ "edge": rim });
    });
    assert!(load(dir.path()).is_ok());
}

fn write_lens_tables(dir: &Path, kind: LensKind, psf: &[f32], spikes: &[f32]) {
    write_tables(
        &dir.join(kind.as_str()).join("tables.safetensors"),
        psf,
        spikes,
    )
    .expect("tables");
    write_sums(dir).expect("sums");
}

#[test]
fn damaged_tables_are_refused() {
    let rc = recipe(LensKind::Sonnar);
    let (psf, spikes) = (psf_table(&rc), spikes_table(9));

    // A slice that does not sum to 1.
    let dir = folder();
    let mut bad = psf.clone();
    for v in &mut bad[1089 * 5..1089 * 6] {
        *v *= 1.01;
    }
    write_lens_tables(dir.path(), LensKind::Sonnar, &bad, &spikes);
    assert!(
        matches!(load(dir.path()), Err(LensError::Invalid { reason, .. }) if reason.contains("sums to"))
    );

    // NaN and negative samples.
    for poison in [f32::NAN, f32::INFINITY, -0.001] {
        let dir = folder();
        let mut bad = psf.clone();
        bad[100] = poison;
        write_lens_tables(dir.path(), LensKind::Sonnar, &bad, &spikes);
        assert!(
            matches!(load(dir.path()), Err(LensError::Invalid { .. })),
            "{poison}"
        );
    }
    // The same in the spikes.
    let dir = folder();
    let mut bad = spikes.clone();
    bad[7] = f32::NAN;
    write_lens_tables(dir.path(), LensKind::Sonnar, &psf, &bad);
    assert!(matches!(load(dir.path()), Err(LensError::Invalid { .. })));

    // A wrong shape, a wrong type, a missing tensor.
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let path = |d: &Path| d.join("sonnar/tables.safetensors");
    let dir = folder();
    let (pb, sb) = (bytes(&psf[..9 * 8 * 33 * 32]), bytes(&spikes));
    serialize_to_file(
        [
            (
                "psf",
                TensorView::new(Dtype::F32, vec![9, 8, 33, 32], &pb).unwrap(),
            ),
            (
                "spikes",
                TensorView::new(Dtype::F32, vec![8, 129, 129], &sb).unwrap(),
            ),
        ],
        None,
        &path(dir.path()),
    )
    .unwrap();
    write_sums(dir.path()).unwrap();
    assert!(
        matches!(load(dir.path()), Err(LensError::Invalid { reason, .. }) if reason.contains("shape"))
    );

    let dir = folder();
    let (pb, sb) = (bytes(&psf), bytes(&spikes));
    serialize_to_file(
        [(
            "psf",
            TensorView::new(Dtype::F32, vec![9, 8, 33, 33], &pb).unwrap(),
        )],
        None,
        &path(dir.path()),
    )
    .unwrap();
    write_sums(dir.path()).unwrap();
    assert!(matches!(
        load(dir.path()),
        Err(LensError::Safetensors { .. })
    ));
    let _ = sb;

    let dir = folder();
    let halves: Vec<u8> = vec![0; psf.len() * 2];
    let sb = bytes(&spikes);
    serialize_to_file(
        [
            (
                "psf",
                TensorView::new(Dtype::F16, vec![9, 8, 33, 33], &halves).unwrap(),
            ),
            (
                "spikes",
                TensorView::new(Dtype::F32, vec![8, 129, 129], &sb).unwrap(),
            ),
        ],
        None,
        &path(dir.path()),
    )
    .unwrap();
    write_sums(dir.path()).unwrap();
    assert!(
        matches!(load(dir.path()), Err(LensError::Invalid { reason, .. }) if reason.contains("F32"))
    );

    // A truncated file and one that is not safetensors at all.
    let dir = folder();
    let bytes = std::fs::read(path(dir.path())).unwrap();
    std::fs::write(path(dir.path()), &bytes[..bytes.len() / 2]).unwrap();
    write_sums(dir.path()).unwrap();
    assert!(matches!(
        load(dir.path()),
        Err(LensError::Safetensors { .. })
    ));
    let dir = folder();
    std::fs::write(path(dir.path()), b"not a tensor file").unwrap();
    write_sums(dir.path()).unwrap();
    assert!(matches!(
        load(dir.path()),
        Err(LensError::Safetensors { .. })
    ));
}

#[test]
fn the_real_data_loads_when_it_exists() {
    let root = common::repo_root().join("data");
    if !root.join("lenses/index.json").exists() {
        eprintln!(
            "skipped: {} has no lenses yet",
            root.join("lenses").display()
        );
        return;
    }
    let lib = LensLibrary::load(&root).expect("data/lenses loads and passes every check");
    assert_eq!(lib.lenses().len(), 7);
    for kind in LensKind::ALL {
        let lens = lib.get(kind).expect("lens");
        assert!(lens.info.version >= 1);
        assert_eq!(
            lens.provenance.statement,
            ym_lens::consts::PROVENANCE_STATEMENT
        );
    }
}
