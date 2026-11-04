//! `ym film list`, the Film line of `ym doctor` and a Film node in a `--settings` file (`docs/24-film.md`
//! section 4.5, `docs/10-cli.md`).

mod common;

use common::{Sandbox, json_lines, lines};
use serde_json::{Value, json};
use ym_core::film::Catalogue;

fn catalogue() -> Catalogue {
    Catalogue::load(&common::data_dir()).expect("the shipped film data")
}

#[test]
fn the_list_shows_the_stocks_and_papers_with_their_matching_papers() {
    let sb = Sandbox::new();
    let out = sb.ym().args(["film", "list"]).assert().success();
    let l = lines(&out.get_output().stdout);
    for header in ["ID", "NAME", "KIND", "ISO", "MATCHING PAPER"] {
        assert!(l[0].contains(header), "{header}: {:?}", l[0]);
    }
    // 20 films and 6 papers, a blank line and the credit line.
    assert_eq!(l.len(), 1 + 20 + 6 + 2, "{l:?}");
    let row = |id: &str| -> Vec<&str> {
        l.iter()
            .find(|r| r.split_whitespace().next() == Some(id))
            .unwrap_or_else(|| panic!("{id} in {l:?}"))
            .split_whitespace()
            .collect()
    };
    let portra = row("kodak_portra_400");
    assert!(portra.contains(&"400") && portra.contains(&"kodak_portra_endura"), "{portra:?}");
    // A slide has no print; a cine negative's print film is hidden, so a still paper stands in.
    assert_eq!(row("fujifilm_velvia_100").last(), Some(&"-"));
    assert_eq!(row("kodak_vision3_500t").last(), Some(&"kodak_portra_endura"));
    assert!(!l.iter().any(|r| r.contains("kodak_2383") || r.contains("kodak_2393")));
    assert_eq!(
        l.last().map(String::as_str),
        Some("Modelled on published data. Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm.")
    );
}

#[test]
fn the_json_is_the_catalogue() {
    let sb = Sandbox::new();
    let out = sb.ym().args(["film", "list", "--json"]).assert().success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    let c = catalogue();
    assert_eq!(v["profile_version"], json!(c.profile_version()));
    assert!(v["credit"].as_str().unwrap().contains("Andrea Volpato"));
    let stocks = v["stocks"].as_array().unwrap();
    let papers = v["papers"].as_array().unwrap();
    assert_eq!((stocks.len(), papers.len()), (20, 6));
    let mut keys: Vec<&str> = stocks[0].as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["id", "iso", "kind", "maker", "name", "paper"]);
    let mut keys: Vec<&str> = papers[0].as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["id", "maker", "name"]);
    for s in stocks {
        let id = s["id"].as_str().unwrap();
        assert!(c.is_film(id), "{id}");
        assert_eq!(s["paper"], json!(c.matching_paper(id)), "{id}");
        assert_eq!(s["kind"] == json!("slide"), s["paper"].is_null(), "{id}");
    }
    for p in papers {
        assert!(c.is_paper(p["id"].as_str().unwrap()));
    }
    // Every id the list shows is one a Film node accepts (the core's own rule).
    for s in stocks {
        let doc = json!({ "nodes": { "items": [ { "id": 2, "kind": "film", "film": { "stock": s["id"] } } ],
            "links": [
              { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
              { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
              { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
        ym_core::settings::DevelopSettings::from_value(doc, &ym_looks::library::LookId::neutral())
            .unwrap_or_else(|e| panic!("{}: {e}", s["id"]));
    }
}

#[test]
fn a_missing_film_folder_is_a_usage_error_with_a_plain_sentence() {
    let sb = Sandbox::new();
    let empty = sb.root().join("empty-data");
    std::fs::create_dir_all(&empty).unwrap();
    let out = sb
        .ym_data(&empty)
        .args(["film", "list"])
        .assert()
        .code(2);
    let err = String::from_utf8_lossy(&out.get_output().stderr).into_owned();
    assert!(err.contains("Yellowmoon's film data is missing."), "{err}");
}

#[test]
fn doctor_says_how_many_profiles_and_which_data_version() {
    let sb = Sandbox::new();
    let out = sb.ym().arg("doctor").assert().success();
    let text = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
    let row = text
        .lines()
        .find(|l| l.starts_with("Film "))
        .unwrap_or_else(|| panic!("a Film row in {text}"));
    assert!(row.trim_end().ends_with("28 profiles, data version 0.3.2"), "{row}");
    let out = sb.ym().args(["doctor", "--json"]).assert().success();
    let v: Value = serde_json::from_slice(&out.get_output().stdout).expect("JSON");
    assert_eq!(v["film"], json!({ "profiles": 28, "data_version": "0.3.2", "problem": null }));
}

#[test]
fn a_settings_file_with_a_film_node_renders_and_an_unknown_stock_is_a_usage_error() {
    let sb = Sandbox::new();
    let arw = sb.arw("a.ARW");
    let doc = |stock: &str| {
        json!({ "nodes": { "items": [ { "id": 2, "kind": "film", "film": { "stock": stock },
              "grain": { "amount": 0 } } ],
            "links": [
              { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
              { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
              { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } })
    };
    let file = sb.root().join("film.json");
    std::fs::write(&file, doc("fujifilm_velvia_100").to_string()).unwrap();
    let out = sb
        .ym()
        .args(["render", "--json", "--cpu", "--auto-provider", "yellowmoon", "--out"])
        .arg(sb.root().join("out"))
        .arg("--settings")
        .arg(&file)
        .arg(&arw)
        .assert()
        .success();
    let objects = json_lines(&out.get_output().stdout);
    assert_eq!(objects.len(), 1);
    let items = objects[0]["settings"]["nodes"]["items"].as_array().unwrap();
    let film = items.iter().find(|i| i["kind"] == json!("film")).expect("the film node");
    assert_eq!(film["film"]["stock"], json!("fujifilm_velvia_100"));
    // An unknown stock stops before any photo is opened, with the sentence of docs/24.
    std::fs::write(&file, doc("kodak_nope").to_string()).unwrap();
    let out = sb
        .ym()
        .args(["render", "--cpu", "--auto-provider", "yellowmoon", "--out"])
        .arg(sb.root().join("out2"))
        .arg("--settings")
        .arg(&file)
        .arg(&arw)
        .assert()
        .code(2);
    let err = String::from_utf8_lossy(&out.get_output().stderr).into_owned();
    assert!(
        err.contains("Unknown film stock \"kodak_nope\". Run \"ym film list\" to see the stocks."),
        "{err}"
    );
}
