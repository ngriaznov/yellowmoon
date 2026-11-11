#!/usr/bin/env bash
# The `disk-full` job (docs/26-hardening.md P2-2, task H-06): `ym render` of five synthetic photos to PNG on a 3 MiB tmpfs.
#
# A full disk is a failure of the destination, so the export has to stop at the first photo that hits it. Before H-06 every
# photo was decoded and rendered, then failed to write (87 s of work and no file for ten photos in the audit). The checks:
#   * exit code 1;
#   * stderr: one error line with the "disk is full" sentence (not one per photo), then "Stopped after 0 of 5. 5 not exported.";
#   * stdout empty and the folder empty: no file, no temporary file;
#   * wall time: the run costs one photo's render (the stop comes after the first photo's render fails to write) and the start-up,
#     so it is compared with a run of ONE photo to a folder with room. It must be under two of those; with no stop it is five.
#
# Needs Linux, `sudo` (or root) and a built `ym` (`cargo build -p ym-cli`) and the example that writes the synthetic ARWs
# (`cargo build -p ym-raw --features testgen --example gen_arw`). Usage: tools/ci/disk_full.sh [target-dir]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="${1:-$repo_root/target/debug}"
ym="$target/ym"
gen="$target/examples/gen_arw"
for f in "$ym" "$gen"; do
  [ -x "$f" ] || { echo "missing $f: build ym-cli and the gen_arw example first" >&2; exit 2; }
done

work="$(mktemp -d)"
sudo=""
[ "$(id -u)" -eq 0 ] || sudo="sudo"
cleanup() {
  $sudo umount "$work/full" 2>/dev/null || true
  rm -rf "$work"
}
trap cleanup EXIT

export YELLOWMOON_DATA="$repo_root/data"
export YELLOWMOON_HOME="$work/home"
mkdir -p "$work/photos" "$work/room" "$work/full"

# Big enough that no PNG fits in 3 MiB (16-bit, 2000 x 1334, noise): about 10 MB each.
for n in 1 2 3 4 5; do
  "$gen" ILCE-7RM4 2000 1334 "$work/photos/DSC0000$n.ARW" >/dev/null
done

now() { date +%s.%N; }

echo "One photo to a folder with room (the time of one render)"
start="$(now)"
"$ym" render --format png --bits 16 --out "$work/room" "$work/photos/DSC00001.ARW" >/dev/null 2>"$work/one.err"
one="$(echo "$(now) - $start" | bc -l)"
printf 'one photo: %.1f s\n' "$one"

$sudo mount -t tmpfs -o size=3m tmpfs "$work/full"

echo "Five photos to a 3 MiB tmpfs"
start="$(now)"
set +e
"$ym" render --format png --bits 16 --out "$work/full" "$work"/photos/*.ARW >"$work/out.txt" 2>"$work/err.txt"
code=$?
set -e
five="$(echo "$(now) - $start" | bc -l)"
printf 'five photos on the full disk: %.1f s (exit %s)\n' "$five" "$code"
cat "$work/err.txt"

fail=0
check() { if ! eval "$2"; then echo "FAILED: $1" >&2; fail=1; fi; }
check "exit code is 1" '[ "$code" -eq 1 ]'
check "one 'disk is full' line on stderr" '[ "$(grep -c "is full" "$work/err.txt")" -eq 1 ]'
check "the sentence names the folder and the option" 'grep -q "Stopped: the disk holding $work/full is full. Free some space, then run the command again with --skip-existing." "$work/err.txt"'
check "the stop line says 0 of 5" 'grep -qx "Stopped after 0 of 5. 5 not exported." "$work/err.txt"'
check "stdout is empty" '[ ! -s "$work/out.txt" ]'
check "the folder is empty (no file, no temporary file)" '[ -z "$(ls -A "$work/full")" ]'
check "the run took under two photos' time" '[ "$(echo "$five < 2 * $one" | bc -l)" -eq 1 ]'
exit "$fail"
