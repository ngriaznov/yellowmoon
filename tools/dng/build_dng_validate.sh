#!/usr/bin/env bash
# Builds Adobe's `dng_validate` from the DNG SDK, for the DNG gates in docs/09 ("Every exported DNG
# passes dng_validate with no errors in CI") and for tools/dngfit (M3-12).
#
#   tools/dng/build_dng_validate.sh [output folder]      default: testdata/dng_sdk
#
# The SDK source is fetched at a pinned commit of a GitHub mirror of Android's external/dng_sdk
# (android.googlesource.com is not reachable from every network). The pinned commit is AOSP's
# "Update dng_sdk to 1.7.1". The build uses the system zlib, libjpeg and libjxl and turns the XMP toolkit
# off (-DqDNGUseXMP=0: dng_validate then skips XMP, which does not matter for pixels and colour) and
# replaces the SDK's JPEG XL code (it needs libjxl 0.10+ and the XMP toolkit) with the stubs in
# dng_jxl_stub.cpp: Yellowmoon DNGs contain no JPEG XL data.
# Needs: git, a C++17 compiler, libjpeg and zlib (libraries and headers) and the libjxl *headers*
# (Debian/Ubuntu: libjpeg-dev zlib1g-dev libjxl-dev; macOS: brew install jpeg jpeg-xl).
#
# Licence: the DNG SDK is under Adobe's DNG SDK licence (permissive, but not an OSI licence). It is a
# build-time and CI tool only: its source and the binary are never committed or shipped
# (docs/14-licensing.md; testdata/ is git-ignored).
#
# Environment overrides: DNG_SDK_REPO, DNG_SDK_COMMIT, JOBS.
set -euo pipefail

REPO="${DNG_SDK_REPO:-https://github.com/GrapheneOS/platform_external_dng_sdk.git}"
COMMIT="${DNG_SDK_COMMIT:-0f23d021adf627c131e2d02ce3b1c62e905edd4e}"
OUT="${1:-testdata/dng_sdk}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"
CXX="${CXX:-c++}"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
SRC="$OUT/src"
BUILD="$OUT/build"

if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote add origin "$REPO"
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
  GIT_TERMINAL_PROMPT=0 git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" || { echo "unexpected DNG SDK commit" >&2; exit 1; }

mkdir -p "$BUILD"
FLAGS=(-std=c++17 -O2 -fexceptions -frtti -w
  -DqDNGBigEndian=0 -DqDNGThreadSafe=1 -DqDNGUseLibJPEG=1 -DqDNGUseXMP=0 -DqDNGXMPFiles=0 -DqDNGXMPDocOps=0
  -DqDNGValidateTarget=1 -DqDNGValidate=1 -DqDNGUsingSanitizer=0
  -DqLinux=1 -DUNIX_ENV=1 -I"$SRC/source")
case "$(uname -s)" in
  Darwin) FLAGS=("${FLAGS[@]/-DqLinux=1/-DqMacOS=1}" -DMAC_ENV=1) ;;
esac
LDFLAGS=()
# Homebrew keeps libjpeg and libjxl out of the default search path.
if command -v brew >/dev/null 2>&1; then
  for formula in jpeg jpeg-xl; do
    P="$(brew --prefix "$formula" 2>/dev/null || true)"
    if [ -n "$P" ]; then FLAGS+=(-I"$P/include"); LDFLAGS+=(-L"$P/lib"); fi
  done
fi

# Everything except the XMP glue (and dng_update_meta, which needs it) and the JPEG XL code (replaced by the
# stub below). Plain bash 3.2 (macOS) features only.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES=()
for path in "$SRC"/source/*.cpp; do
  name="$(basename "$path")"
  case "$name" in
    dng_xmp.cpp|dng_xmp_sdk.cpp|dng_jxl.cpp|dng_update_meta.cpp) ;;
    *) SOURCES+=("$name") ;;
  esac
done

case "$OUT" in *" "*) echo "the output folder must not contain spaces" >&2; exit 1 ;; esac
export BUILD SRC CXX
export FLAGS_STR="${FLAGS[*]}"
# shellcheck disable=SC2016
printf '%s\0' "${SOURCES[@]}" | xargs -0 -n 1 -P "$JOBS" sh -c '
  f="$1"; o="$BUILD/${f%.cpp}.o"
  if [ ! -f "$o" ] || [ "$SRC/source/$f" -nt "$o" ]; then
    # shellcheck disable=SC2086
    "$CXX" $FLAGS_STR -c "$SRC/source/$f" -o "$o" || { echo "compile failed: $f" >&2; exit 255; }
  fi' sh

"$CXX" "${FLAGS[@]}" -c "$HERE/dng_jxl_stub.cpp" -o "$BUILD/dng_jxl_stub.o"
"$CXX" -o "$OUT/dng_validate" "$BUILD"/*.o ${LDFLAGS[@]+"${LDFLAGS[@]}"} -ljpeg -lz -lpthread
echo "built $OUT/dng_validate"
sha256sum "$OUT/dng_validate" 2>/dev/null || shasum -a 256 "$OUT/dng_validate"
