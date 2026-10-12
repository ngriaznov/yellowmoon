# Entry points for the `jdd` training code. Run from anywhere: `make -C training/jdd <target>`.
# PYTHON can point at another interpreter, e.g. `make -C training/jdd PYTHON=/opt/ymml/env/bin/python test`.
PYTHON ?= python3
ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
RUN    ?= training/jdd/runs/jdd-001
CONFIG ?= training/jdd/configs/default.json
DEVICE ?= auto
GOLDEN := crates/ym-ml/tests/data

.PHONY: help train-jdd dry-run overfit test export evaluate model-card sample-sheet golden golden-check bench-weights

help:
	@echo "train-jdd     full training run (needs a GPU and PixelShift200; HUMAN M2-H4): CONFIG=... RUN=... DEVICE=..."
	@echo "dry-run       50 steps on CPU with synthetic data, then ONNX/safetensors checks (CI job training-dry-run)"
	@echo "overfit       short CPU run that overfits 8 samples (M2-05 done-when)"
	@echo "test          pytest for training/jdd"
	@echo "export        safetensors + static ONNX of RUN (core 512 half-res px plus halo)"
	@echo "evaluate      PSNR vs the classical baseline on synthetic samples (gate sets: SET=folder PATH_ARG=testdata/demosaic/kodak, or SET=pixelshift200 PATH_ARG=testdata/pixelshift200)"
	@echo "model-card    write MODEL_CARD.md for RUN"
	@echo "sample-sheet  PNG of the sample generator (training/jdd/out/sample_sheet.png)"
	@echo "golden        regenerate the golden tiles for crates/ym-ml (seeded random weights)"
	@echo "bench-weights seeded random full-size weights + ONNX for the M0-08 benchmark (testdata/ml/random-512)"

train-jdd:
	cd $(ROOT) && $(PYTHON) -m training.jdd.train --config $(CONFIG) --run $(RUN) --device $(DEVICE)

dry-run:
	cd $(ROOT) && $(PYTHON) -m training.jdd.train --dry-run --steps 50 --device cpu --run training/jdd/runs/dry-run

overfit:
	cd $(ROOT) && $(PYTHON) -m training.jdd.train --config training/jdd/configs/overfit.json --run training/jdd/runs/overfit

test:
	cd $(ROOT) && $(PYTHON) -m pytest training/jdd -q

export:
	cd $(ROOT) && $(PYTHON) -m training.jdd.export --weights $(RUN)/weights.safetensors --out $(RUN)/export

SET ?= synthetic
evaluate:
	cd $(ROOT) && $(PYTHON) -m training.jdd.evaluate --weights $(RUN)/weights.safetensors --set $(SET) $(if $(PATH_ARG),--path $(PATH_ARG),) --out $(RUN)/eval-$(SET).json

model-card:
	cd $(ROOT) && $(PYTHON) -m training.jdd.model_card --run $(RUN)

sample-sheet:
	cd $(ROOT) && $(PYTHON) -m training.jdd.sample_sheet

golden:
	cd $(ROOT) && $(PYTHON) -m training.jdd.golden_tiles --out $(GOLDEN) && chmod 644 $(GOLDEN)/*

golden-check:
	cd $(ROOT) && $(PYTHON) -m training.jdd.golden_tiles --check $(GOLDEN)

bench-weights:
	cd $(ROOT) && $(PYTHON) -m training.jdd.export --random 1234 --core 512 --out testdata/ml/random-512
