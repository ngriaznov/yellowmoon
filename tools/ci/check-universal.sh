#!/usr/bin/env bash
# Assert the architectures of a built Yellowmoon.app and start its x86_64 slice (docs/26, H-16 / P5-1).
#
#   tools/ci/check-universal.sh <path to Yellowmoon.app> [universal|aarch64] [output dir]
#
# universal (the default): the app binary and the bundled `ym` must each hold exactly the architectures x86_64 and
#   arm64 (compared as a set: the order `lipo -archs` prints is the order in the file and is not relied on). Then the
#   x86_64 slice of `ym` is run (`ym --version`, `ym doctor`), under Rosetta on an Apple-silicon runner (installed here
#   if it is missing) or natively on an Intel one. A slice that does not start (a missing dynamic library, a Swift
#   runtime path, an illegal instruction) fails the script.
# aarch64: both binaries must be exactly arm64 and there is no x86_64 slice to run.
#
# The output dir (default: $RUNNER_TEMP or /tmp, then ym-universal-check) gets x86_64-version.txt, x86_64-doctor.txt
# and summary.md; the workflow uploads it as an artifact. summary.md is also appended to $GITHUB_STEP_SUMMARY.
# Needs macOS tools: lipo, arch, softwareupdate (only when Rosetta is missing).
set -euo pipefail

APP="${1:?usage: check-universal.sh <Yellowmoon.app> [universal|aarch64] [output dir]}"
TARGET="${2:-universal}"
OUT="${3:-${RUNNER_TEMP:-/tmp}/ym-universal-check}"

fail() {
  echo "::error title=Architecture check failed::$*" >&2
  exit 1
}

# `lipo -archs FILE` as one sorted, space-separated line.
archs_of() {
  lipo -archs "$1" | tr ' ' '\n' | grep -v '^$' | sort | tr '\n' ' ' | sed 's/ $//'
}

assert_archs() {
  local file="$1" want="$2" have
  [ -f "$file" ] || fail "$file does not exist"
  have="$(archs_of "$file")"
  if [ "$have" != "$want" ]; then
    fail "$file holds \"$have\" and must hold exactly \"$want\" (lipo -archs)"
  fi
  echo "ok: $file is $have"
}

case "$TARGET" in
  universal) WANT="arm64 x86_64" ;;
  aarch64) WANT="arm64" ;;
  *) fail "unknown target \"$TARGET\" (universal or aarch64)" ;;
esac

MAIN="$APP/Contents/MacOS/Yellowmoon"
YM="$APP/Contents/MacOS/ym"
assert_archs "$MAIN" "$WANT"
assert_archs "$YM" "$WANT"

if [ "$TARGET" != universal ]; then
  echo "No x86_64 slice in this build; nothing more to run."
  exit 0
fi

mkdir -p "$OUT"

# Rosetta, when the runner is Apple silicon and has none.
if ! arch -x86_64 /usr/bin/true 2>/dev/null; then
  echo "arch -x86_64 does not run: installing Rosetta"
  softwareupdate --install-rosetta --agree-to-license
  arch -x86_64 /usr/bin/true || fail "arch -x86_64 /usr/bin/true still fails after installing Rosetta"
fi

HOST="$(uname -m)"
if [ "$HOST" = arm64 ]; then HOST_NOTE="Apple silicon, the slice runs under Rosetta"; else HOST_NOTE="Intel, native"; fi

# Informational only: does the machine that runs the slice offer AVX2 to it? ym-ml picks its AVX2+FMA kernels with
# is_x86_feature_detected!, and ym doctor does not print which path was taken. Rosetta before macOS 15 is reported not to
# translate AVX2 (the scalar kernels then run). This line never fails the script.
LEAF7="$(arch -x86_64 sysctl -n machdep.cpu.leaf7_features 2>/dev/null || true)"
if [ -z "$LEAF7" ]; then
  AVX2_NOTE="AVX2: unknown (machdep.cpu.leaf7_features is not readable here)"
  LEAF7_SHOWN="(empty)"
else
  LEAF7_SHOWN="$LEAF7"
  case " $LEAF7 " in
    *" AVX2 "*) AVX2_NOTE="AVX2 is reported to the x86_64 slice (the AVX2 kernels are expected)" ;;
    *) AVX2_NOTE="AVX2 is not reported to the x86_64 slice (the scalar kernels are expected; a valid path)" ;;
  esac
fi
echo "machdep.cpu.leaf7_features: $LEAF7_SHOWN"
echo "Host: $HOST_NOTE; $AVX2_NOTE"

arch -x86_64 "$YM" --version | tee "$OUT/x86_64-version.txt" \
  || fail "the x86_64 slice of ym does not start (--version)"
# `tee` hides a failing ym when pipefail is off; set -o pipefail is on above, so the || fires.

if ! arch -x86_64 "$YM" doctor >"$OUT/x86_64-doctor.txt" 2>&1; then
  cat "$OUT/x86_64-doctor.txt" || true
  fail "the x86_64 slice of ym does not run \`ym doctor\`"
fi
cat "$OUT/x86_64-doctor.txt"

{
  echo "### x86_64 slice of \`ym\`"
  echo
  echo "Host: $HOST_NOTE. $AVX2_NOTE."
  echo
  echo '```text'
  cat "$OUT/x86_64-version.txt"
  echo
  cat "$OUT/x86_64-doctor.txt"
  echo '```'
} >"$OUT/summary.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$OUT/summary.md" >>"$GITHUB_STEP_SUMMARY"; fi
echo "The x86_64 slice starts: ym --version and ym doctor exit 0."
