#!/usr/bin/env bash
# One command for the Mac session of the GPU wave (docs/gpu-plan.md section 9, M0-H4): runs the GPU suites on this
# machine's graphics card, times the render path of one photo, compares the device with the CPU on that photo, and
# writes ONE JSON file for docs/STATUS.md.
#
#   tools/dev/mac-gpu-check.sh [--out FILE.json] [--photo FILE.ARW] [--repeat N] [--with-cpu] [--skip-tests]
#
#   --out FILE.json  where the report goes (default: mac-gpu-check.json in the current folder). The only file this
#                    script writes besides the build output of cargo and a scratch folder it removes again. It refuses
#                    a name that does not end in .json, a folder, and a file that git tracks.
#   --photo FILE.ARW the photo for the timings and for the CPU against device comparison (a 50 MP Sony file makes the
#                    numbers of the plan's targets). Without it only the suites run.
#   --repeat N       exports of the same JPEG that must come out byte for byte equal (default 10).
#   --with-cpu       also time the CPU path (YELLOWMOON_GPU=off), for the comparison; slow on a large photo.
#   --skip-tests     only the photo; the suites take a while.
#
# On a Mac the device is Metal (WGPU_BACKEND=metal is set); anywhere else the platform's default backend, so the same
# command checks a Linux machine with a graphics card or with Mesa lavapipe. YM_REQUIRE_GPU=1 makes a missing
# adapter a failure, not a skip. Exit codes: 0 every suite passed and the exports were byte-identical, 1 something
# failed (the JSON says what), 2 usage, 127 a required tool is missing.
#
# The report (schema 1; every key is always present, `null` where something was not run):
#   schema            1
#   generated         UTC time, ISO 8601
#   machine           { os, arch, model, ram_bytes, graphics }   what the OS says; no user name, no host name
#   git               { commit, dirty }
#   env               { WGPU_BACKEND, YM_REQUIRE_GPU }
#   doctor            `ym doctor --json`: version, adapter, limits, policy, budgets (no file names)
#   suites            [ { name, command, seconds, exit, passed, failed, ignored, binaries: [ { name, passed,
#                         failed, ignored } ], failures: [ test names ] } ]
#                     ym-gpu (kernels against the CPU references, tile = whole, poison, determinism), ym-core
#                     (the renderer on the device, determinism and faults), ym-cli parity_gpu (app = ym)
#   bench             `ym doctor --bench --json` on the device (compile times, views, exports, repeat, memory, GPU
#                     pass times), or null
#   bench_cpu         the same with YELLOWMOON_GPU=off (--with-cpu), or null
#   device_vs_cpu     `golden device-check --report` for the photo: per file the median, p99 and max of the CIEDE2000
#                     difference of the two pictures and the share of samples more than 8 codes off, or null
#   ok                true when every suite passed, the exports were identical and the picture check passed
set -euo pipefail

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
}

out="mac-gpu-check.json"
photo=""
repeat=10
with_cpu=0
skip_tests=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="${2:-}"; shift 2 || { usage; exit 2; } ;;
    --photo) photo="${2:-}"; shift 2 || { usage; exit 2; } ;;
    --repeat) repeat="${2:-}"; shift 2 || { usage; exit 2; } ;;
    --with-cpu) with_cpu=1; shift ;;
    --skip-tests) skip_tests=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$repeat" in
  ''|*[!0-9]*) echo "--repeat takes a whole number" >&2; exit 2 ;;
esac
case "$out" in
  *.json) ;;
  *) echo "--out must name a .json file" >&2; exit 2 ;;
esac
# The script works from the repository root (cargo, the git check below), so a relative --out or --photo is made absolute
# here, against the folder the command was typed in: the report lands where the user said, not in the repository root.
absolute() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}
out="$(absolute "$out")"
if [ -n "$photo" ]; then photo="$(absolute "$photo")"; fi
if [ -d "$out" ]; then
  echo "--out is a folder" >&2
  exit 2
fi
if [ -n "$photo" ] && [ ! -f "$photo" ]; then
  echo "--photo: $photo is not a file" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
for tool in cargo git awk sed; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool is required" >&2; exit 127; }
done
# The report never replaces a file of the repository.
if git ls-files --error-unmatch -- "$out" >/dev/null 2>&1; then
  echo "--out names a file that git tracks; give another name" >&2
  exit 2
fi

scratch="$(mktemp -d "${TMPDIR:-/tmp}/ym-mac-gpu-check.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

# --- helpers ---------------------------------------------------------------------------------------------------

# A JSON string from the text on stdin: backslash and quote escaped. Newline, tab and carriage return become spaces and the
# other control characters are dropped (tr only: BSD sed on a Mac does not read \001 in a pattern like GNU sed does, and nothing
# this script writes through here, a name, a command line, a test name, has more than one line).
json_string() {
  printf '"%s"' "$(tr '\n\t\r' '   ' | tr -d '\000-\037' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
}

# The text of a file as a JSON value: itself when it parses as one JSON line, else null.
json_or_null() {
  if [ -s "$1" ]; then
    # A report is one line of JSON; the last non-empty line of the file.
    line="$(awk 'NF { last = $0 } END { print last }' "$1")"
    case "$line" in
      "{"*"}") printf '%s' "$line" ;;
      *) printf 'null' ;;
    esac
  else
    printf 'null'
  fi
}

bool() { if [ "$1" = 0 ]; then printf 'true'; else printf 'false'; fi; }

# --- the machine -----------------------------------------------------------------------------------------------

os="$(uname -sr)"
arch="$(uname -m)"
model=""
ram=0
graphics=""
case "$(uname -s)" in
  Darwin)
    export WGPU_BACKEND="${WGPU_BACKEND:-metal}"
    model="$(sysctl -n hw.model 2>/dev/null || true)"
    ram="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    graphics="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model|Metal Support|Total Number of Cores/ { gsub(/^ +/, "", $1); printf "%s: %s; ", $1, $2 }' || true)"
    ;;
  Linux)
    model="$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || true)"
    ram="$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo 2>/dev/null || echo 0)"
    if command -v lspci >/dev/null 2>&1; then
      graphics="$(lspci 2>/dev/null | grep -i -E 'vga|3d|display' | head -n 2 | tr '\n' ';' || true)"
    fi
    ;;
esac
export YM_REQUIRE_GPU=1
commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
dirty=false
if [ -n "$(git status --porcelain 2>/dev/null | head -n 1)" ]; then dirty=true; fi

echo "mac-gpu-check: building ym and golden (release) ..." >&2
cargo build --release -p ym-cli >&2
cargo build --release -p golden >&2
target="${CARGO_TARGET_DIR:-$root/target}"
ym="$target/release/ym"
golden="$target/release/golden"
export YELLOWMOON_DATA="$root/data"

ok=0
doctor="$scratch/doctor.json"
"$ym" doctor --json >"$doctor" 2>"$scratch/doctor.err" || true

# --- the suites ------------------------------------------------------------------------------------------------

suites_json=""
run_suite() {
  local name="$1"
  shift
  local cmd_text="$*"
  local log="$scratch/suite.log"
  echo "mac-gpu-check: $name ..." >&2
  local started code=0 seconds
  started="$(date +%s)"
  "$@" >"$log" 2>&1 || code=$?
  seconds=$(( $(date +%s) - started ))
  # One entry per test binary: its "Running <path>" line, then its "test result:" line.
  local binaries totals passed failed ignored failures=""
  binaries="$(awk '
    /^ +Running / { name = $NF; gsub(/[()]/, "", name); sub(/.*\//, "", name); sub(/-[0-9a-f]+$/, "", name); next }
    /^ +Doc-tests / { name = "doc-tests " $2; next }
    /^test result:/ {
      printf "%s{\"name\":\"%s\",\"passed\":%d,\"failed\":%d,\"ignored\":%d}", sep, name, $4, $6, $8
      sep = ","
    }' "$log")"
  totals="$(awk '/^test result:/ { p += $4; f += $6; i += $8 } END { printf "%d %d %d", p, f, i }' "$log")"
  read -r passed failed ignored <<<"$totals"
  local t
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    if [ -n "$failures" ]; then failures="$failures,"; fi
    failures="$failures$(printf '%s' "$t" | json_string)"
  done < <(awk '/^test .* \.\.\. FAILED$/ { print $2 }' "$log")
  local entry
  entry="{\"name\":$(printf '%s' "$name" | json_string),\"command\":$(printf '%s' "$cmd_text" | json_string),\"seconds\":$seconds,\"exit\":$code,\"passed\":$passed,\"failed\":$failed,\"ignored\":$ignored,\"binaries\":[$binaries],\"failures\":[$failures]}"
  if [ -n "$suites_json" ]; then suites_json="$suites_json,$entry"; else suites_json="$entry"; fi
  if [ "$code" != 0 ]; then ok=1; fi
}

if [ "$skip_tests" = 0 ]; then
  run_suite "ym-gpu: kernels against the CPU references, tile = whole, poison" \
    cargo test -p ym-gpu --release --no-fail-fast -- --test-threads=2
  run_suite "ym-core: the renderer on the device, determinism and faults" \
    cargo test -p ym-core --release --features fault-injection --no-fail-fast -- --test-threads=2
  run_suite "ym-cli: the app and ym give the same exports on the device" \
    env YELLOWMOON_GPU=on cargo test -p ym-cli --release --test parity_gpu
fi

# --- the photo -------------------------------------------------------------------------------------------------

bench_file="$scratch/bench.json"
bench_cpu_file="$scratch/bench-cpu.json"
compare_file="$scratch/compare.json"
: >"$bench_file"; : >"$bench_cpu_file"; : >"$compare_file"
if [ -n "$photo" ]; then
  echo "mac-gpu-check: timing the render path on the device ..." >&2
  YELLOWMOON_GPU=require "$ym" doctor --bench "$photo" --json --repeat "$repeat" >"$bench_file" 2>"$scratch/bench.err" || ok=1
  if [ "$with_cpu" = 1 ]; then
    echo "mac-gpu-check: timing the CPU path ..." >&2
    YELLOWMOON_GPU=off "$ym" doctor --bench "$photo" --json --repeat 2 >"$bench_cpu_file" 2>"$scratch/bench-cpu.err" || ok=1
  fi
  echo "mac-gpu-check: the device against the CPU on this photo ..." >&2
  "$golden" device-check --ym "$ym" --data "$root/data" --report "$compare_file" "$photo" >&2 || ok=1
  # Identical bytes in the repeat check are part of the verdict.
  if [ "$(json_or_null "$bench_file")" != null ] && ! grep -q '"identical":true' "$bench_file"; then ok=1; fi
fi

# --- the report ------------------------------------------------------------------------------------------------

report="$scratch/report.json"
{
  printf '{"schema":1,"generated":"%s",' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '"machine":{"os":%s,"arch":%s,"model":%s,"ram_bytes":%s,"graphics":%s},' \
    "$(printf '%s' "$os" | json_string)" "$(printf '%s' "$arch" | json_string)" \
    "$(printf '%s' "$model" | json_string)" "${ram:-0}" "$(printf '%s' "$graphics" | json_string)"
  printf '"git":{"commit":"%s","dirty":%s},' "$commit" "$dirty"
  printf '"env":{"WGPU_BACKEND":%s,"YM_REQUIRE_GPU":"1"},' "$(printf '%s' "${WGPU_BACKEND:-}" | json_string)"
  printf '"doctor":%s,' "$(json_or_null "$doctor")"
  printf '"suites":[%s],' "$suites_json"
  printf '"bench":%s,' "$(json_or_null "$bench_file")"
  printf '"bench_cpu":%s,' "$(json_or_null "$bench_cpu_file")"
  # The golden report is pretty-printed JSON over many lines; embed it whole.
  if [ -s "$compare_file" ]; then
    printf '"device_vs_cpu":%s,' "$(tr -d '\n' <"$compare_file")"
  else
    printf '"device_vs_cpu":null,'
  fi
  printf '"ok":%s}\n' "$(bool "$ok")"
} >"$report"

# One file, all at once: written next to its final name and renamed.
tmp_out="$out.tmp.$$"
cp "$report" "$tmp_out"
mv "$tmp_out" "$out"
echo "mac-gpu-check: wrote $out ($([ "$ok" = 0 ] && echo ok || echo FAILED))" >&2
exit "$ok"
