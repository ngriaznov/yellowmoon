#!/usr/bin/env bash
# The browser-side latency measurement of docs/11 (M3-03) and docs/26 (H-09), against the production build and the mock with 500 photos
# (YM_PERF_PHOTOS=100,500,2000 measures several folder sizes in one run and checks that a key costs about the same at 2000 as at 500).
#
#   e2e/perf/run.sh            builds the app for production, serves it, drives it with Chromium, writes
#                              test-results/perf/latency-<photos>.md and .json per size (YM_PERF_OUT moves them)
#
# Knobs (environment): YM_PERF_PHOTOS (500; a list such as 100,500,2000), YM_PERF_RATIO (1.25, the most p95 at 2000 may be of p95 at 500), YM_PERF_KEYS (80), YM_PERF_SPEED (10, the mock's development speed), YM_PERF_P95_MS (150, the bound),
# YM_PERF_PORT (4173). Needs `npm ci` and the Playwright Chromium (`npx playwright install chromium`; not in the sandbox, where
# it is already installed). Exit code 0 when the bound holds, 1 when it does not or the run failed.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec npx playwright test -c e2e/perf/playwright.config.ts "$@"
