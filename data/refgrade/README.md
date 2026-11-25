# The Reference Grade network

`network.ywn` is the hand-written network that reads the measures of a photo and of a reference picture and writes the grading targets
and gates of the Reference Grade fit (`docs/34-reference-grade.md`, section 4b.5).

- **Ours.** Every weight and bias was written by hand by the project; nothing is trained and nothing is copied from another project.
  It is licensed under Apache-2.0 like the rest of the repository.
- **What it is.** Round 15's file, the proof of concept's v14 file with every neuron unchanged and its comments made neutral (no
  reference is named): 432 neurons after the templates are expanded,
  62 targets (`T.*`) and 21 gates (`G.*`). The header of the file lists what each round added and why.
- **Its outputs.** Targets: the floor, ceiling, contrast and anchor of the lightness, the saturation, the tint of the near-neutrals
  per band, the frame cast and the hue-drift limit, the whites' tint and tolerance, the subject's separation, the sectors' hue and
  chroma, each element's hue, chroma and exposure, and the wall parameters (purple, skin, memory colours, the content turn budget and
  the clip allowance). Gates: key mismatch, unify, the subject, the six sectors and, for each element, `matched` and `hue`.
- **Its grammar** is in `docs/34`, section 4b.5. A file that breaks a rule is refused with its line and the name.
- **Pinned.** `crates/ym-color/src/refgrade/tests.rs` pins the file's SHA-256
  (`6f52dd30aeafdbf1f9e2fc8e9a84bee5c8afc8c5b2fb10e00cc3f457c0395a5a`), its 432 neurons and the SHA-256 of the expanded text. A
  weight changes only by a spec change with the owner's agreement and a pass of the acceptance rules of `docs/34` section 10.1.
- **Shipped.** The library compiles the file in (`ym_color::refgrade::network::NETWORK_TEXT`); the app and `ym` carry it.
