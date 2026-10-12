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
| ARRI sample footage, ARRI Reference Tool, Image SDK | no grant / EULA / partner-only | offline developer validation only, if at all |
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
| Figtree font | OFL-1.1 | bundle with its licence text |
| US patent 11948279 (joint demosaic+denoise with a lightweight U-Net) | patent | [HUMAN M4-H4] freedom-to-operate check before 1.0 |

## Rules for the codebase

1. `cargo deny check` and the npm licence check pass on every push (`deny.toml`, `ci.yml`).
2. A new dependency's licence goes in the PR description.
3. No code copied from GPL/AGPL/LGPL projects. Reading for format knowledge is fine; write the code yourself and say in the PR which references you read.
4. No ARRI LUT data in the repository, in test fixtures, or in shipped tables. Tests use our own synthetic `.cube` files.
5. Shipped weights only from CC0, CC BY, Apache or our own data. A model card per network lists every dataset and licence.
6. Product name, bundle id, crate names and UI never use "ARRI" or "ALEXA" as names. Factual compatibility lines only. The About window says "Not affiliated with ARRI or Sony. ARRI and ALEXA are trademarks of ARRI AG; Sony is a trademark of Sony Group Corporation."

## NOTICE file

`NOTICE` at the repository root (draft included) lists: Yellowmoon copyright; the AMPAS notice for the ARRI matrices; rawtoaces-data (Weta curves) attribution; ACES 2 output transform attribution; PixelShift200 and Cube++ attribution (once weights trained on them ship); the Figtree OFL notice; the Adobe DNG notice. It ships inside the app bundle and is shown in About.

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
