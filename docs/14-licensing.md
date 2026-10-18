# 14 · Licensing and legal

The project is Apache-2.0. Everything shipped must be compatible with it. Everything ARRI-owned stays on the user's machine. Not legal advice; open items are marked [HUMAN].

## Inputs and how we handle them

| Input | Licence | Handling |
|---|---|---|
| ARRI 17-CCT ALEXA matrices (aces-dev v1.3.1) | AMPAS licence; notice retained | **Ship** in `data/arri/` with `AMPAS-LICENSE.md` and the notice in `NOTICE` |
| Our ALEXA curve estimate | ours, Apache-2.0; derived from AMPAS and Apache data | **Ship**, labelled as an estimate, with `NOTICE` |
| Weta Sony curves and 190 training patches (rawtoaces-data) | Apache-2.0 | **Ship** (curves) / use in fitting (patches) |
| LogC3/LogC4, AWG3/AWG4 formulas and constants | published specs, no licence text; functional facts | implement from spec |
| ACES 2 output transform (aces-output) | Apache-2.0 | **port** as the default DRT, with attribution in `NOTICE` |
| ARRI Look Library and display LUTs | all rights reserved, no grant | user imports at runtime; never bundled, hosted, auto-downloaded, committed or used to fit shipped tables |
| ARRI Look File formats (ALF4 / ALF4c / ALF2 `.aml`) | file-format facts; ARRI's tools: EULA | file-format facts from ARRI's public PDFs, our own inspection of the free Look Library downloads and of files ARRI's look-builder (ART CMD 1.0.0, used with the owner's go-ahead on 2026-10-16) wrote from our own LUTs. **The position the work rests on:** ART CMD's EULA (section 4.2) bars reverse engineering, decompiling and disassembling the software, so only the files its look-builder wrote from our own LUTs were analysed, never the software, its libraries or its schemas; the layout of an `.alf4c` and the 17, 33 and 65 point behaviour come from those output files alone. The licence is purpose-restricted to the documented use and is granted to entrepreneurs (its closing clause has the licensee state that it is not a consumer); no ARRI code or data. Users import their own files at runtime. The reader is `ym_color::alf`, written from the layouts; the CRC-32 is our own (a table, no crate). Tests use files from our own writer, never an ARRI file |
| ARRI sample footage, the Image SDK, and any use of the ARRI Reference Tool beyond the documented look-builder use above | no grant / EULA / partner-only | offline developer validation only, if at all |
| HdM ALEXA SXT curve; CIC 2015 figure | no licence / © IS&T | validation only; never shipped |
| CIE 99 CES; CIE illuminant SPDs | CC BY-SA 4.0 | test and fitting only, fetched by script. The sensor-map and white tables we ship are fitted coefficients, not the SPDs. [HUMAN M1-H4] confirm that's acceptable, or restrict fitting to blackbody/daylight formulas plus our own measured SPDs. |
| PixelShift200 | CC BY 4.0 | training, with attribution in the model card and `NOTICE` |
| Cube++ | CC BY 4.0 | illuminant estimator training, attribution |
| DND, SID, ELD data, Gehler–Shi, NUS-8, INTEL-TAU | non-commercial / evaluation-only / unstated | evaluation only |
| LED denoiser, CCMNet, vkdt-nn weights, KARWY, camspec | CC BY-NC / none / unknown / CC BY-NC-SA | avoid |
| Solomatov & Akkaynak 2023 estimated curves | no licence | evaluation only |
| darktable noise profiles | GPL-3.0 | offline reference only; never committed |
| darktable-cli (the program) | GPL-3.0 | test oracle for lens corrections, run as a separate process in CI; never linked, copied or shipped |
| Adobe DNG Converter | Adobe freeware | local one-off check of lens opcodes; never automated against Adobe's servers or shipped |
| Photography Life and Imaging Resource sample raws | copyrighted; personal use / evaluation | local visual review and blind A/B only; never committed, uploaded or cached in CI |
| Foi noise-estimation code | non-commercial | not used; reimplemented from the paper |
| RawTherapee, darktable, librtprocess, RapidRAW, OpenDRT | GPL-3 / AGPL-3 | read for ideas only; clean-room from papers and descriptions |
| rawler, rawloader, rawspeed, LibRaw, `ljpeg` crate | LGPL-2.1 (LibRaw also CDDL) | CI oracles only (`tools/oracle`, rawpy); never linked into shipped binaries |
| lightcraft ARW 6.0 decoder | MIT / Apache-2.0 | reference or dependency, after the Sony question |
| SMPTE RDD 34 (LLVC, behind ARW 6.0) | IP enquiries go to Sony | [HUMAN M2-H3] ask Sony before shipping ARW 6.0 decoding |
| Adobe DNG specification | royalty-free patent licence with a notice | notice in source, docs and About |
| DNG SDK (`dng_validate`) | Adobe DNG SDK licence (permissive) | CI tool only |
| `dng` crate, `jpegxl-rs` | AGPL-3.0, GPL-3.0 | banned in `deny.toml` |
| Tauri, wgpu, burn, ort, moxcms, tiff, clap, specta | MIT / Apache-2.0 / BSD | dependencies |
| `colorimetry` crate | MIT / Apache-2.0 (embeds CIE datasets) | dev/test and fitting dependency only, so no CIE data is compiled into shipped binaries (`docs/02`) |
| colour-science, OpenColorIO | BSD-3 | training and test-vector generation |
| Figtree font | OFL-1.1 | bundle with its licence text: the OFL text is in `THIRD-PARTY.md` in the bundle (generated; `tools/licences/`) |
| glasscn components (`src/components/glass/`, `src/hooks/use-glass-pointer.ts`, `src/hooks/use-sliding-indicator.ts`, `src/lib/glass-variants.ts`) | MIT, Copyright (c) 2026 Tim Mikeladze | the interface's controls (owner's decision of 2026-10-16, `docs/16`). The source was installed from the glasscn shadcn registry (https://glasscn.app/r/) and is changed for Yellowmoon, so it lives in the repository. The MIT text is in `src/components/glass/LICENSE` and in `NOTICE` (item 10), and `tools/licences/check_bundle.py` fails when `NOTICE` loses it |
| Tailwind CSS 4 (`tailwindcss`, `@tailwindcss/vite`) | MIT | build tools (dev dependencies), but their output is part of the app's CSS: Tailwind's preflight and the utility classes we use. The minified stylesheet carries no banner, so `tools/licences/generate.py` lists `tailwindcss` in `THIRD-PARTY.md` with its MIT text (`BUILD_TOOL_OUTPUT`) although `license-checker --production` leaves it out |
| `lightningcss` (a dependency of `@tailwindcss/node`) | MPL-2.0 | **build-time tool only, not shipped.** It is a dev dependency of the build, it transforms our CSS and none of its files is in the app or in the repository, so MPL-2.0 file-level terms do not reach us. It stays out of the npm licence check (`--production`) and out of `THIRD-PARTY.md`. If it ever becomes a runtime dependency, the check fails and this row must be revisited |
| Radix UI (`radix-ui` and the `@radix-ui/*` primitives), `@floating-ui/*`, `react-remove-scroll`, `react-remove-scroll-bar`, `react-style-singleton`, `use-sidecar`, `use-callback-ref`, `aria-hidden`, `get-nonce`, `detect-node-es`, `tslib`, `cn`, `clsx`, `tw-animate-css` | MIT (`tslib` 0BSD) | runtime dependencies of the controls, in `THIRD-PARTY.md`. `react-remove-scroll-bar` ships no LICENSE file in its npm tarball, so its repository's MIT text is kept in `tools/licences/texts/` and used by the generator (`HAND_ADDED_TEXTS`) |
| `class-variance-authority` | Apache-2.0 | runtime dependency of the controls, in `THIRD-PARTY.md` |
| `lucide-react` (icons) | ISC | runtime dependency, in `THIRD-PARTY.md` |
| `jpeg-encoder` crate (JPEG export) | (MIT OR Apache-2.0) AND IJG | dependency; IJG allowed in `deny.toml` (permissive); the IJG acknowledgement goes in `NOTICE`, docs and the About window (decision 2026-10-12) |
| colour-demosaicing (Menon port) | BSD-3-Clause | ported into `ym-gpu::cpu::demosaic` with the full licence text (copyright line, three conditions, disclaimer) in the module docs and in `NOTICE`; a binary distribution must carry it (clause 2), and `tools/licences/check_bundle.py` fails when `NOTICE` lacks the conditions |
| darktable `lens.cc` | GPL-3.0 | read (not copied) to learn Sony's embedded lens-correction conventions; darktable-cli also runs as a separate-process lens oracle |
| Raw algorithms: adaptive ratio demosaic, log-guided demosaic, diffraction deconvolution, output sharpening (`docs/algorithms/`, test vectors in `crates/ym-gpu/tests/cpu_data/`) | algorithm specifications supplied by the project owner (2026-10-13) with C++ references used only as a development oracle outside the repository (never committed); known techniques (ratio-corrected interpolation, log-domain guided filtering, Airy-PSF deconvolution, unsharp masking with halo limiting) plus the owner's own combinations and tuned constants; prior-art record in `research/raw-algorithms-prior-art.md` (74 pieces: 55 known or variants of known work, 16 novel combinations of known parts, 3 with no prior art found) | implemented in Rust from the specs, Apache-2.0; the vectors (generated data) are the fixed oracle in the repository. A token-shingle and structural check (2026-10-13) against RawTherapee, librtprocess, darktable, LibRaw, RCD-Demosaicing and rawspeed found no GPL/LGPL/AGPL code or tables. The Rust ports are written from the specs without consulting GPL demosaic or sharpening code. Patents not checked (part of [HUMAN M4-H4]) |
| US patent 11948279 (joint demosaic+denoise with a lightweight U-Net) | patent | [HUMAN M4-H4] freedom-to-operate check before 1.0 |

## Rules for the codebase

1. `cargo deny check` and the npm licence check pass on every push (`deny.toml`, `ci.yml`).
2. A new dependency's licence goes in the PR description.
3. No code copied from GPL/AGPL/LGPL projects. Reading for format knowledge is fine; write the code yourself and say in the PR which references you read.
4. No ARRI LUT data in the repository, in test fixtures, or in shipped tables. Tests use our own synthetic `.cube` files.
5. Shipped weights only from CC0, CC BY, CC BY-SA, Apache or our own data. A model card per network lists every dataset and licence. ShareAlike data (CC BY-SA, for example Google's HDR+ burst dataset or pixls.us Play Raw files) may train shipped weights when its licence does not directly prohibit that use, with attribution in the model card and `NOTICE` (owner's decision of 2026-10-18, `docs/16`). Creative Commons' own conservative reading (May 2025) asks that models based on ShareAlike content be shared under the same licence, so **weights trained on any CC BY-SA data are themselves licensed CC BY-SA 4.0** (owner's decision of 2026-10-18), in the same repository: they stay separate data files in `data/models/<name>/` that the app loads (never compiled into Rust source, so the Apache-2.0 code and the weights are aggregated, not merged), that folder holds its own `LICENSE` with the CC BY-SA 4.0 text, its model card states the weights' licence and credits every training source, and `NOTICE` and `THIRD-PARTY.md` (both in the app bundle) list them. Weights trained only on CC0, CC BY, Apache or our own data keep the repository's Apache-2.0. Non-commercial, research-only and evaluation-only data (MIT-Adobe FiveK, PPR10K and the sets built on them) stay evaluation only.
6. Product name, bundle id, crate names and UI never use "ARRI" or "ALEXA" as names. Factual compatibility lines only. The About window says "Not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation."
7. A binary release carries the licence texts its dependencies require: `THIRD-PARTY.md` and `NOTICE` are in the app bundle (`bundle.resources`), and both CI (`licences`) and `release.yml` generate and check them.
8. **Numeric facts may come from any source** (Mykyta, 2026-10-14): colour matrices, white-balance coefficients and similar camera constants may be taken as numbers from any source, including projects whose licence we cannot accept. Only the numbers are taken: no code, file, table layout, text or comment from such a project enters the repository, and nothing in the repository or the app refers to such a project as a source. Rule 4 is unchanged: no ARRI LUT data in any form.

## NOTICE file

`NOTICE` at the repository root (draft included) lists: Yellowmoon copyright; the AMPAS notice for the ARRI matrices; rawtoaces-data (Weta curves) attribution; ACES 2 output transform attribution; PixelShift200 and Cube++ attribution (once weights trained on them ship); the Figtree OFL notice; the Adobe DNG notice; the MIT notice of the glasscn components. It ships inside the app bundle and is shown in About. `NOTICE` carries the licence text of anything we ported or copied (the AMPAS terms, the BSD 3-Clause text of the Menon demosaic, the MIT text of glasscn); it names a dataset as training data only for weights that ship (the model cards say what each network was fitted on). The licence texts of the dependencies are in `THIRD-PARTY.md`, generated for every release from `Cargo.lock` and `package-lock.json` (`tools/licences/generate.py`), bundled next to `NOTICE`, and generated by the `licences` CI job on every change so that a dependency without a licence text fails there.

## Emails and checks [HUMAN]

| Item | When | Note |
|---|---|---|
| Email Sony about RDD 34 / ARW 6.0 decoding | M2 | ARW 6.0 stays behind a feature flag until answered |
| Email ARRI (arriraw-dev@arri.de) about in-app download of the Look Library | M3 | optional; import works without it |
| Ask HdM for a licence on the SXT curve | M1 | validation and possible future use |
| Confirm CIE-derived coefficient tables are acceptable | M1 | see table |
| Trademark clearance (EUIPO/TMview, INPI, WIPO, UK IPO; classes 9 and 42) | before announcing | or ask a trademark attorney |
| Freedom-to-operate check on US 11948279 | before 1.0 | |
| Decide on Apple Developer ID signing | before 1.0 | |
