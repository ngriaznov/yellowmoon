#!/usr/bin/env bash
# The "local-eval" tier: six real Sony ARW files for evaluation on THIS machine only.
#
#   tools/fetch/real-eval.sh              fetch (or verify) the files into testdata/real/ and list them
#   tools/fetch/real-eval.sh --list       list what is in testdata/real/files (model, size); no network
#   tools/fetch/real-eval.sh --force      also replace a file in testdata/real/files whose content differs from the pinned one
#   tools/fetch/real-eval.sh --help
#
# Source: github.com/drewnoakes/metadata-extractor-images, the arw/ folder (DSLR-A500, ILCE-5000, ILCE-7M3, ILCE-7M4, NEX-5N,
# NEX-7; 6 files, about 140 MB), pinned to one commit. The script makes a shallow, blob-less clone and checks out only arw/
# (git sparse checkout), so nothing else of that repository (about 10 000 files) is downloaded. Layout, all git-ignored:
#
#   testdata/real/metadata-extractor-images/arw/   the checkout (the six files, plus small text dumps of their metadata)
#   testdata/real/files/<Model>.ARW                a link to each file, named by the model in the file's own TIFF Model tag;
#                                                  this is the folder the local-only tests read (crates/ym-core/tests/
#                                                  real_files.rs, render_real.rs, export_real.rs; training/illum/smoke_real.py)
#
# LICENCE IS UNCLEAR. The repository's README says "You are free to use these media files however you wish", but it records
# no author and no licence for the individual files, and the files were contributed by third parties. So this tier is for
# local evaluation only: never committed (testdata/ is git-ignored), never used in CI (the script refuses to run there), never
# uploaded, cached or redistributed, and no picture made from them goes into the repository (golden references, screenshots).
# They are not part of the CC0 corpus (tools/fetch/samples.sh) or the golden tiers 1 to 3 (tools/fetch/golden.sh), and no gate
# in docs/13 depends on them. See docs/validation-data.md.
#
# Idempotent: a checkout that is already at the pinned commit needs no network; a link that already points at the pinned file is
# left alone; a file in testdata/real/files that is not the pinned one (your own, or a different version) is never replaced
# without --force. Content is checked against the git blob ids of the pinned commit.
#
# Environment: YM_TESTDATA=/path moves testdata/. YM_REAL_EVAL_REPO and YM_REAL_EVAL_COMMIT replace the pinned source (the
# unit test uses a local repository). Exit codes: 0 ok, 1 a file or the checkout does not match the pin, 2 usage error or
# running in CI, 3 nothing found (--list), 4 the network is unreachable or refused. Needs git 2.30 or newer and python3.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
testdata="${YM_TESTDATA:-$repo_root/testdata}"
real="$testdata/real"
checkout="$real/metadata-extractor-images"
files="$real/files"

# --- pin ----------------------------------------------------------------------------------------------------------------
REPO="${YM_REAL_EVAL_REPO:-https://github.com/drewnoakes/metadata-extractor-images.git}"
COMMIT="${YM_REAL_EVAL_COMMIT:-651ad0e67aa8d43d358ad05f9bc07b52d8b9ac6e}"   # 2026-02-02, "Add JPEG with extended XMP that triggers bug"
# ------------------------------------------------------------------------------------------------------------------------

mode="fetch" force=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list) mode="list" ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "real-eval.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

warn_licence() {
  cat >&2 <<'MSG'

  WARNING: LOCAL EVALUATION ONLY. THE LICENCE OF THESE FILES IS UNCLEAR.
  They come from github.com/drewnoakes/metadata-extractor-images, which records no author and no licence for the
  individual files. Use them on this machine to look at real Sony files, and nowhere else:
    - never commit them or copy them into the repository (testdata/ is git-ignored);
    - never use them in CI, and never upload or cache them there;
    - never redistribute them, or pictures made from them (golden references, screenshots, bug-report attachments).
  Gates and golden tests must not depend on them.

MSG
}

# Never in CI: a fork or a cache would republish files whose licence nobody has checked.
if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "real-eval.sh: refusing to run in CI. These files have an unclear licence and are for local evaluation only." >&2
  exit 2
fi

command -v python3 >/dev/null 2>&1 || { echo "real-eval.sh: python3 is required (standard library only)" >&2; exit 127; }

# links_and_list <arw dir or "-"> : link the files of the checkout into testdata/real/files (unless "-"), then list the folder
links_and_list() {
  python3 -I - "$1" "$files" "$force" "$checkout" <<'PY'
import os, struct, sys, re, hashlib

src_dir, files_dir, force, checkout = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]


def blob_id(path):
    """The git blob id (SHA-1 of "blob <size>\\0" + content) of a file: what the pinned commit lists for it."""
    h = hashlib.sha1()  # noqa: S324 - not security: git's own object id
    h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tiff_model(path):
    """The Model tag (0x0110) of IFD0 of a TIFF-based file (ARW is one), or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return None
            e = "<" if head[:2] == b"II" else ">"
            if struct.unpack(e + "H", head[2:4])[0] != 42:
                return None
            f.seek(struct.unpack(e + "I", head[4:8])[0])
            (n,) = struct.unpack(e + "H", f.read(2))
            for _ in range(min(n, 1000)):
                ent = f.read(12)
                if len(ent) < 12:
                    return None
                tag, typ, cnt = struct.unpack(e + "HHI", ent[:8])
                if tag == 0x0110 and typ == 2 and 0 < cnt < 256:
                    if cnt <= 4:
                        raw = ent[8:8 + cnt]
                    else:
                        here = f.tell()
                        f.seek(struct.unpack(e + "I", ent[8:12])[0])
                        raw = f.read(cnt)
                        f.seek(here)
                    return raw.split(b"\0")[0].decode("ascii", "replace").strip() or None
    except (OSError, struct.error):
        return None
    return None


def is_arw(name):
    return name.lower().endswith(".arw")


problems = 0
pinned = {}  # blob id -> source path, for the "source" column
if src_dir != "-":
    os.makedirs(files_dir, exist_ok=True)
    taken = set()
    for name in sorted(os.listdir(src_dir)):
        path = os.path.join(src_dir, name)
        if not is_arw(name) or not os.path.isfile(path):
            continue
        bid = blob_id(path)
        pinned[bid] = path
        stem = re.sub(r"[^A-Za-z0-9._+-]", "_", tiff_model(path) or re.sub(r"^Sony\s+|\s*\(.*\)$|\.arw$", "", name, flags=re.I))
        target_name, k = stem + ".ARW", 1
        while target_name in taken:
            k += 1
            target_name = f"{stem}_{k}.ARW"
        taken.add(target_name)
        target = os.path.join(files_dir, target_name)
        if os.path.lexists(target):
            if os.path.isfile(target) and blob_id(target) == bid:
                continue  # already the pinned file (a link to it, or an identical copy someone made earlier)
            if os.path.islink(target) and not os.path.exists(target):
                os.remove(target)  # a link whose file is gone (the checkout was deleted): ours, make it again
            elif not force:
                print(f"real-eval.sh: {target} exists and is not the pinned file; left alone (--force replaces it)", file=sys.stderr)
                problems += 1
                continue
            else:
                os.remove(target)
        os.symlink(os.path.relpath(path, files_dir), target)
else:
    # --list: recognise pinned files by their blob ids in the checkout, when there is one
    arw = os.path.join(checkout, "arw")
    if os.path.isdir(arw):
        for name in os.listdir(arw):
            p = os.path.join(arw, name)
            if is_arw(name) and os.path.isfile(p):
                pinned[blob_id(p)] = p

rows = []
if os.path.isdir(files_dir):
    for name in sorted(os.listdir(files_dir)):
        path = os.path.join(files_dir, name)
        if is_arw(name) and os.path.isfile(path):
            where = "metadata-extractor-images (pinned)" if blob_id(path) in pinned else "not from this script"
            rows.append((tiff_model(path) or "?", os.path.getsize(path), name, where))

print(f"{files_dir}  (local evaluation only)")
if not rows:
    print("  no ARW files")
else:
    print(f"  {'model':<14}{'size':>10}  {'file':<22}{'source'}")
    for model, size, name, where in rows:
        print(f"  {model:<14}{size / 1e6:>7.1f} MB  {name:<22}{where}")
    print(f"  {len(rows)} files, {sum(r[1] for r in rows) / 1e6:.1f} MB")
sys.exit(1 if problems else (0 if rows else 3))
PY
}

if [ "$mode" = "list" ]; then
  warn_licence
  status=0
  links_and_list - || status=$?
  if [ "$status" = 3 ]; then echo "real-eval.sh: nothing found; run tools/fetch/real-eval.sh to fetch the files" >&2; fi
  exit "$status"
fi

command -v git >/dev/null 2>&1 || { echo "real-eval.sh: git is required" >&2; exit 127; }
warn_licence
export GIT_TERMINAL_PROMPT=0

ready=0
if [ -d "$checkout/.git" ] \
   && [ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null || true)" = "$COMMIT" ] \
   && [ -d "$checkout/arw" ]; then
  ready=1
  echo "real-eval.sh: checkout already at ${COMMIT:0:12}"
fi

if [ "$ready" = 0 ]; then
  echo "real-eval.sh: fetching arw/ of $REPO @ ${COMMIT:0:12} (shallow, blob-less, sparse)"
  mkdir -p "$real"
  rm -rf "$checkout"
  mkdir -p "$checkout"
  git -C "$checkout" init -q
  git -C "$checkout" remote add origin "$REPO"
  git -C "$checkout" config remote.origin.promisor true
  git -C "$checkout" config remote.origin.partialclonefilter blob:none
  git -C "$checkout" sparse-checkout set --cone arw
  if ! git -C "$checkout" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"; then
    rm -rf "$checkout"
    echo "real-eval.sh: cannot fetch $REPO @ ${COMMIT:0:12}: the network (or a proxy in front of it) refuses it, or the pinned commit is gone" >&2
    exit 4
  fi
  if ! git -C "$checkout" checkout -q --detach FETCH_HEAD; then
    rm -rf "$checkout"
    echo "real-eval.sh: git could not check out arw/ of the fetched commit (the download of the files failed)" >&2
    exit 4
  fi
  got="$(git -C "$checkout" rev-parse HEAD)"
  if [ "$got" != "$COMMIT" ]; then
    echo "real-eval.sh: unexpected commit $got (pinned: $COMMIT)" >&2
    rm -rf "$checkout"
    exit 1
  fi
fi

# the checkout must be what the pinned commit says (a half finished download, or an edited file, is not)
if [ -n "$(git -C "$checkout" status --porcelain 2>/dev/null)" ]; then
  echo "real-eval.sh: $checkout has local changes; delete the folder and run again" >&2
  exit 1
fi

status=0
links_and_list "$checkout/arw" || status=$?
echo "real-eval.sh: reminder: local evaluation only, the licence of these files is unclear; never commit, upload or use them in CI" >&2
exit "$status"
