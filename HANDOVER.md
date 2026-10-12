# Handover: how to use this package

For Mykyta. This package is the seed of the Yellowmoon repository: specs, task plan, UI mockups, research, data and CI templates. A coding agent can start from it without this conversation.

## Start a build with a coding agent

1. Create `github.com/ngriaznov/yellowmoon` on your personal account (like plugout) and copy everything in this package into its root. Commit.
2. Point the agent at the repository and give it one message with the domain:
   > Domain: `<domain>`. Read AGENTS.md, then start milestone M0 in docs/15-milestones.md. Keep docs/STATUS.md up to date.
   
   The agent generates the updater signing key and sets the GitHub secrets (it needs `gh` signed in with admin rights on the repository), and builds the test photo set from public sources (`docs/test-photo-sources.md`).
3. `AGENTS.md` (and `CLAUDE.md`, which imports it) tells the agent the rules, the reading order, how to work and what "done" means.
4. Check `docs/STATUS.md` for progress, gate numbers, questions and the list of things waiting on you.

The agent works best on a Mac (the app shell, native viewport and Metal performance need one). On Linux it can still build and test most of the core; it will list the Mac-only checks for you.

## What only you can do

These are marked [HUMAN] in `docs/15-milestones.md`. In order of need:

| When | ID | What |
|---|---|---|
| M0 | M0-H4 | Mac-only checks if the agent isn't on a Mac |
| M0 | — | Move the updater signing key and password from the backup location the agent reports into your password manager |
| M1 | M1-H3, M1-H4 | Email HdM; decide on CIE-derived tables |
| M1 | M1-H5 | Review the first renders |
| M2 | M2-H3 | Email Sony about ARW 6.0 decoding |
| M2 | M2-H4, M2-H5 | GPU time for training; blind A/B of noise |
| M3 | M3-H1 | Design and name the Signature look |
| M3 | M3-H2 … H6 | Fingerprint ARRI packages; optional email to ARRI; app review; a real shoot; DNG checks in other apps |
| M4 | M4-H1 … H5 | Blind A/B; signing decision; trademark clearance; patent check; announce |

V1 needs no shoots from you. Test photos come from public sources, and the gates that would have used grey cards, flat fields, calibration frames or pixel shift use public datasets and independent tools instead. Those captures are listed as V2 work at the end of `docs/15-milestones.md`.

## What's in the package

| Path | Contents |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions for the coding agent |
| `README.md` | the public README, with your motivation section |
| `NOTICE`, `LICENSE` | Apache-2.0 and third-party notices (AMPAS, rawtoaces-data, ACES 2, Figtree, Adobe DNG) |
| `docs/00`–`16` | specs: overview, architecture, colour, cameras, decoding, raw processing, noise, ML, looks, export, CLI, UI, distribution, testing, licensing, milestones, risks |
| `docs/STATUS.md` | progress file the agent keeps current |
| `design/` | four interactive mockups (open `design/mockups/index.html`), 24 screenshots at 2× plus an overview sheet, `tokens.css`/`tokens.json`, Figtree, source boards |
| `data/` | shipped-data candidates: ALEXA curve estimate, ARRI matrices (AMPAS), 11 Weta Sony curves (Apache-2.0) |
| `research/` | both research reports, notes on formats, DNG, noise and naming, the ALEXA curve bundle with scripts and results, and a snapshot of the plan doc |
| `templates/` | CI and release workflows, Homebrew cask, `tauri.conf.json`, `deny.toml`, adapted from plugout |

## The V1 data rule

V1 relies on available measurements for Sony cameras: Weta's published spectral curves for 11 bodies, plus public files and datasets for noise, testing and validation. No measurements or shoots of our own. It's stated at the top of `AGENTS.md` and in `docs/00-overview.md`, so the agent treats it as a rule.

## Changes since the plan doc

The Claude Docs plan ("Yellowmoon development plan") is the earlier summary. This package supersedes it where they differ:

- Eleven measured camera profiles from day one (the plan doc still has two lines saying the A7R IV curves are used for every body).
- Lossless M/S and ARW 6.0 are in V1 scope (M/S through a reduced path; ARW 6.0 gated on Sony).
- The display LUT package count is five (the plan doc says "six" in one place).
- Many details the plan left open now have concrete starting values (tint mapping, auto exposure, tone controls, curve domain, DNG fit procedure). They're marked provisional in `docs/16` where a review should confirm them.
