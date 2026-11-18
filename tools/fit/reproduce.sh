#!/usr/bin/env bash
# Reproduces the measurements of docs/29-raw-constants.md section 2.2 with the fit harness (RC-05),
# at Tuning::ORACLE, and prints them in the audit's table format.
#
#   tools/fit/reproduce.sh check               # the base numbers only, compared with tools/fit/audit_base.txt (about 1 minute)
#   tools/fit/reproduce.sh demosaic|diffraction|sharpen|ca OUTDIR   # one table of section 2.2 (JSON lines and text in OUTDIR)
#   tools/fit/reproduce.sh all OUTDIR          # the four of them (about 10 minutes on 4 cores; each part runs alone in under 5)
#
# Needs testdata/demosaic/{kodak,mcmaster} (tools/fetch/demosaic-sets.sh; YM_TESTDATA moves the root).
# YM_PHOTO=<an ARW> adds the two real-mosaic crops (group C of the demosaic table; the owner's A1 file,
# local only). The data is for evaluation and local fitting and is never committed.
# FIT_BIN does not apply here: the script builds and runs the examples through cargo.
set -euo pipefail
cd "$(dirname "$0")/../.."

run() { # run EXAMPLE ARGS...
  local ex=$1; shift
  cargo run -q -p ym-gpu --example "$ex" --config 'profile.dev.package.ym-gpu.opt-level=3' -- "$@"
}
table() { python3 -I tools/fit/search.py table "$@"; }
photo=()
if [[ -n "${YM_PHOTO:-}" ]]; then photo=(--photo "$YM_PHOTO"); fi

# The factors of the audit: x0.5, x0.9, x1.1, x1.5 for spec 01 and spec 04, x0.9 and x1.1 for spec 03.
F4=0.5,0.9,1.1,1.5
F2=0.9,1.1

demosaic() {
  # offset_snr stands for the audit's `K_50`: the audit moved the 50 of K = 50 (g + sqrt(g^2 + 0.04 r^2))
  # alone; `offset_snr` is the SNR of which both the 50 and the 0.04 follow, so those two rows differ
  # (docs/29 appendix, "What does not reproduce").
  run fit_demosaic ${photo[@]+"${photo[@]}"} \
    --rel second_diff_weight=$F4 --rel ca_relax_slope=$F4 --rel centre_weight_linear=$F4 \
    --rel centre_weight_cubic=$F4 --rel chroma_floor=$F4 --rel sat_dominance=$F4 \
    --rel shadow_amount=$F4 --rel shadow_laplacian_scale=$F4 --rel match_excess_weight=$F4 \
    --rel offset_snr=$F4 --rel snr_per_slider=$F4
}
diffraction() {
  run fit_diffraction \
    --rel weight_ramp_width_nm=$F2 --rel airy_scale=$F2 --rel r80_energy=$F2 \
    --rel support_r80_factor=$F2 --rel support_pitch_factor=$F2 --rel kernel_energy=$F2 \
    --rel kernel_tail=$F2 --rel x80_cutoff=$F2 --rel x80_level=$F2 --rel sigma_factor=$F2 \
    --rel sigma_min=$F2 --rel sigma_max=$F2 --rel gauss_radius_per_sigma=$F2 \
    --rel clip_weight_largest=$F2 --rel clip_weight_middle=$F2 --rel clip_fraction=$F2 \
    --rel pull_gain=$F2 --rel pull_band=$F2 --rel ridge=$F2 --rel gate=$F2 \
    --rel gate_fade_lo=$F2 --rel gate_fade_hi=$F2 --rel weight_min=$F2 --rel ratio_cap=$F2
}
sharpen() {
  run fit_sharpen \
    --rel sigma_slope=$F4 --rel sigma_offset=$F4 --rel taps_per_sigma=$F4 --rel taps_offset=$F4 \
    --rel sigma_knee=$F4 --rel halo_pull_fade=$F4 --rel ratio_max=$F4 --rel screen_radius_px=$F4 \
    --rel screen_amount_pct=$F4 --rel screen_halo_pct=$F4 --rel screen_threshold_levels=$F4
}
ca() {
  run fit_demosaic --ca-only \
    --eval "A1 to 10@second_diff_weight=10" --eval "A2 to 0@ca_relax_slope=0" \
    --rel ca_relax_slope=0.5,1.5,4 \
    --eval "A4 to 0@centre_weight_linear=0,centre_weight_cubic=0" \
    --eval "A4 x0.5@centre_weight_linear=9,centre_weight_cubic=51" \
    --eval "A4 x1.5@centre_weight_linear=27,centre_weight_cubic=153" \
    --eval "A4 x4@centre_weight_linear=72,centre_weight_cubic=408"
}

part() { # part NAME OUTDIR
  mkdir -p "$2"
  local kind=$1
  $1 >"$2/$1.jsonl"
  table "$kind" <"$2/$1.jsonl" | tee "$2/$1.txt"
}

case "${1:-}" in
  check)
    tmp=$(mktemp -d)
    {
      run fit_demosaic | table demosaic | grep '^BASE'
      run fit_diffraction | table diffraction | grep '^BASE'
      run fit_sharpen | table sharpen | grep '^BASE'
      run fit_demosaic --ca-only | table ca | grep '^base'
    } >"$tmp/base.txt"
    diff -u tools/fit/audit_base.txt "$tmp/base.txt" && echo "the base numbers of docs/29 section 2.2 reproduce"
    ;;
  demosaic|diffraction|sharpen|ca) part "$1" "${2:?an output folder}" ;;
  all) for p in diffraction sharpen ca demosaic; do part "$p" "${2:?an output folder}"; done ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
