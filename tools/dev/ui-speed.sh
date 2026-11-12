#!/usr/bin/env bash
# How fast the interface answers a slider drag on THIS Mac, with the glass blur on and then off (U-02).
#
#   tools/dev/ui-speed.sh --photos FOLDER [--at SECONDS] [--out DIR] [--skip-build]
#
#   --photos FOLDER  a folder of ARW files to open (YM_SMOKE_OPEN). Required.
#   --at SECONDS     when the scripted hand starts, after the start of the app (default 25: the photo must be up by then).
#   --out DIR        where the two logs go (default: ui-speed-out in the current folder).
#   --skip-build     use the build that is there.
#
# It builds the app (the frontend embedded, development profile: the scripted hand is in debug builds), then starts it twice with
# YM_PERF=1 and YM_PERF_SCRIPT=1: first as it is, then with YM_GLASS=off (no backdrop blur anywhere; nothing else changes). The script
# drags the Temperature slider with 240 pointer moves, 8 ms apart (and does the wheel and key phases before it). The shell writes one
# `tick` line per patch and a `window 5s:` line every five seconds to the log (`ym::perf`), joining the page's stamps with its own:
#
#   tick seq=412 final=false input_to_sent=1.3 ipc_in=0.9 handle=2.1 ipc_back=1.0 request_to_present_called=14.2 request_to_presented=14.6
#        page_painted=22.0 input_to_presented=19.8 sent_to_presented=18.5 (ms)
#   window 5s: ticks 83, input_to_presented p50 25.0 p95 41.2 max 66.0 ms, over 8.3 ms 80, over 16.7 ms 61, pointer events 240, frames shown 71
#
# This prints the busiest window of each run, side by side, and where the full logs are. "presented" is the moment the viewport handed the
# frame to the display; it is on the glass at most one refresh later. Nothing leaves the machine.
set -euo pipefail

photos=""; at=25; out="ui-speed-out"; build=1
while [ $# -gt 0 ]; do
  case "$1" in
    --photos) photos="${2:-}"; shift 2 || exit 2 ;;
    --at) at="${2:-}"; shift 2 || exit 2 ;;
    --out) out="${2:-}"; shift 2 || exit 2 ;;
    --skip-build) build=0; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
{ [ -n "$photos" ] && [ -d "$photos" ]; } || { echo "give --photos a folder of ARW files" >&2; exit 2; }
[ "$(uname -s)" = Darwin ] || echo "note: this is not a Mac; the native viewport and its present stamps need one." >&2

cd "$(dirname "$0")/../.."
if [ "$build" = 1 ]; then
  npm run build
  # tauri-build wants the sidecar and the bundled licence texts to exist; this run never bundles them.
  mkdir -p src-tauri/binaries
  [ -e src-tauri/binaries/ym-aarch64-apple-darwin ] || : > src-tauri/binaries/ym-aarch64-apple-darwin
  [ -e THIRD-PARTY.md ] || : > THIRD-PARTY.md
  cargo build -p yellowmoon-app --features custom-protocol
fi
bin="${CARGO_TARGET_DIR:-target}/debug/yellowmoon-app"
[ -x "$bin" ] || { echo "no $bin: build first (drop --skip-build)" >&2; exit 1; }
mkdir -p "$out"

run() { # run <name> [env...]
  local name="$1"; shift
  echo "== $name: the app runs for about $((at + 40)) s; leave the mouse alone"
  env YM_SMOKE=1 YM_SMOKE_OPEN="$photos" YM_SMOKE_HOLD=$((at + 40)) YM_PERF=1 YM_PERF_SCRIPT=1 YM_PERF_SCRIPT_AT="$at" "$@" \
    "$bin" > "$out/app-$name.log" 2>&1 || true
}
busiest() { # busiest <log>: the window line with the most ticks
  grep -a "window 5s:" "$1" | sed 's/^.*window 5s:/window 5s:/' | awk '{ t = $4; sub(",", "", t); if (t + 0 >= best) { best = t + 0; line = $0 } } END { if (line != "") print line; else print "(no window with ticks: see the log)" }'
}

run glass-on YM_GLASS=on
run glass-off YM_GLASS=off
echo
echo "glass on : $(busiest "$out/app-glass-on.log")"
echo "glass off: $(busiest "$out/app-glass-off.log")"
echo
echo "scripted hand (glass on) : $(grep -a 'smoke: perf script' "$out/app-glass-on.log" | tail -1 | cut -c1-400)"
echo "scripted hand (glass off): $(grep -a 'smoke: perf script' "$out/app-glass-off.log" | tail -1 | cut -c1-400)"
echo "full logs: $out/app-glass-on.log  $out/app-glass-off.log  (grep ym::perf)"
