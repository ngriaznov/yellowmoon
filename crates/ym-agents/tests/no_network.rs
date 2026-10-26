//! Hard rule 7 (`docs/22` section 2.1): Yellowmoon opens no network connection, so no HTTP stack may be in
//! the dependency tree of the crate that starts the person's programs, nor in `ym`'s.

use std::process::Command;

/// Crates that open sockets or speak HTTP, TLS or WebSocket.
const FORBIDDEN: [&str; 22] = [
    "reqwest",
    "hyper",
    "hyper-util",
    "hyper-rustls",
    "hyper-tls",
    "h2",
    "ureq",
    "ureq-proto",
    "curl",
    "curl-sys",
    "isahc",
    "attohttpc",
    "surf",
    "awc",
    "native-tls",
    "openssl",
    "openssl-sys",
    "rustls",
    "tokio-rustls",
    "rustls-webpki",
    "tungstenite",
    "tokio-tungstenite",
];

/// The names in the closure of `package`'s normal dependencies.
fn closure(package: &str) -> Vec<String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .args(["tree", "-p", package, "-e", "normal", "--prefix", "none"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree can be run");
    assert!(
        output.status.success(),
        "cargo tree -p {package} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

fn assert_no_http(package: &str) {
    let tree = closure(package);
    assert!(
        tree.iter().any(|name| name == package),
        "the tree of {package} was not read: {tree:?}"
    );
    for crate_name in FORBIDDEN {
        assert!(
            !tree.iter().any(|name| name == crate_name),
            "{crate_name} is in the dependency tree of {package}: no HTTP client may enter Yellowmoon's tree (hard rule 7)"
        );
    }
}

#[test]
fn ym_agents_has_no_http_client() {
    assert_no_http("ym-agents");
}

#[test]
fn ym_cli_has_no_http_client() {
    assert_no_http("ym-cli");
}

#[test]
fn jsonschema_is_used_without_its_default_features() {
    // The default features of `jsonschema` pull an HTTP client for remote references.
    let manifest =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml")).unwrap();
    let line = manifest
        .lines()
        .find(|l| l.trim_start().starts_with("jsonschema"))
        .expect("jsonschema is a workspace dependency");
    assert!(line.contains("default-features = false"), "{line}");
}
