#!/usr/bin/env bash
# Fetch the two public inputs into ../src (run from this folder).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p src/acesdev src/rtadata

# ARRI ALEXA SUP 3.x native -> ACES AP0 matrices (aces-dev v1.3.1, AMPAS license)
git clone -q --depth 1 --branch v1.3.1 --filter=blob:none --sparse \
  https://github.com/ampas/aces-dev.git src/acesdev/repo
git -C src/acesdev/repo sparse-checkout set transforms/ctl/idt/vendorSupplied/arri

# 52 measured camera curves + 190 training patches (rawtoaces-data, Apache-2.0)
git clone -q --depth 1 https://github.com/AcademySoftwareFoundation/rawtoaces-data.git \
  src/rtadata/repo

echo "Sources ready. Then: pip install colour-science==0.4.7 scipy matplotlib"
