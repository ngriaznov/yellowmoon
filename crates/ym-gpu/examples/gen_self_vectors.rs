//! Writes the self vectors of the raw algorithms (`docs/29-raw-constants.md` RC-04).
//!
//! For each of the four packages it reads the oracle vectors of `tests/cpu_data/<algorithm>/`
//! (never edited), runs our own Rust under the shipped constants (`Tuning::DEFAULT`, the shipped
//! cubics for the log-guided demosaic) on their inputs and writes `tests/cpu_data/<algorithm>_self/`:
//! the same files, fields and tolerances, with only the results taken from us. Run it after a fit
//! has changed a shipped constant and commit the diff; `tests/self_vectors.rs` fails while the
//! committed files differ from what this writes.
//!
//! ```text
//! cargo run -p ym-gpu --example gen_self_vectors            # into tests/cpu_data
//! cargo run -p ym-gpu --example gen_self_vectors -- <dir>   # into <dir>/<algorithm>_self/
//! ```

#[path = "../tests/self_gen/mod.rs"]
mod self_gen;

fn main() {
    let root = std::env::args_os()
        .nth(1)
        .map_or_else(self_gen::data_root, std::path::PathBuf::from);
    for (package, files) in self_gen::generate_all(&root) {
        println!(
            "{}: {} files into {}",
            package.self_folder(),
            files.len(),
            root.join(package.self_folder()).display()
        );
    }
}
