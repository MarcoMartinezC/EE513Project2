# One-command reproduction:   make reproduce
# Same chain on the simulated camera (no captures needed):   make demo
#
# Override any variable on the command line, e.g.  make train ISOS="100 3200" STEPS=5000

PY        ?= ../.venv/bin/python
CONFIG    ?= config/camera.yaml
DATA      ?= data
ISOS      ?= 100 800 3200          # ISOs of your bursts (folders data/burst/iso<ISO>)
EVAL_ISO  ?= 3200                  # the burst used for the classical-vs-learned table
STEPS     ?= 3000
CROP      ?= 0                     # e.g. CROP=2048 for fast development runs
DETAIL    ?= $(firstword $(wildcard $(DATA)/detail/*.NEF $(DATA)/detail/*.npz))
POOL_RAWS ?= $(wildcard $(DATA)/pool/*.NEF)
RUN       ?= runs/dncnn_all
DERIVED   ?= $(DATA)/derived

.PHONY: help setup test calibrate references demosaic ablation pool train evaluate export benchmark detect results reproduce demo clean-demo

help:
	@grep -E '^#' Makefile | head -6; echo; echo "targets: setup test calibrate references demosaic ablation pool train evaluate export benchmark detect results reproduce demo"

setup:
	uv pip install --python $(PY) -e .

test:
	$(PY) -m pytest

calibrate:
	$(PY) scripts/calibrate.py --config $(CONFIG)

references:
	@for iso in $(ISOS); do $(PY) scripts/build_reference.py --config $(CONFIG) --burst $(DATA)/burst/iso$$iso --crop $(CROP) || exit 1; done

demosaic:
	$(PY) scripts/score_demosaic.py --config $(CONFIG) --reference $(DERIVED)/burst_iso$(firstword $(ISOS)).npz
	@if [ -n "$(DETAIL)" ]; then $(PY) scripts/score_demosaic.py --config $(CONFIG) --raw $(DETAIL); fi

ablation:
	@for iso in $(ISOS); do $(PY) scripts/ablation.py --config $(CONFIG) --reference $(DERIVED)/burst_iso$$iso.npz || exit 1; done

pool:
	$(PY) scripts/build_clean_pool.py --config $(CONFIG) $(POOL_RAWS) --exclude-burst $(DATA)/burst

train:
	$(PY) scripts/train_denoiser.py --config $(CONFIG) --isos $(ISOS) --steps $(STEPS) --out $(RUN)

evaluate:
	@for iso in $(ISOS); do $(PY) scripts/eval_denoisers.py --config $(CONFIG) --reference $(DERIVED)/burst_iso$$iso.npz --model all=$(RUN)/model.pt || exit 1; done

export:
	$(PY) scripts/export_onnx.py $(RUN)/model.pt --sizes 256 512 1024

benchmark:
	$(PY) scripts/benchmark_pipeline.py --config $(CONFIG) $(DETAIL) --runs 5

detect:
	$(PY) scripts/detect_compare.py --config $(CONFIG)

results:
	$(PY) scripts/make_results_json.py --config $(CONFIG)

reproduce: test calibrate references demosaic ablation pool train evaluate export benchmark detect results

# ---------------------------------------------------------------------------
# The same chain on the simulated camera (detector step skipped: no objects).
demo:
	$(PY) scripts/make_synthetic_data.py --out $(DATA)/synthetic
	$(MAKE) calibrate references demosaic ablation pool train evaluate export benchmark \
		CONFIG=$(DATA)/synthetic/camera.yaml DATA=$(DATA)/synthetic DERIVED=$(DATA)/synthetic/derived \
		ISOS="100 1600 6400" STEPS=1500 RUN=$(DATA)/synthetic/runs/all \
		POOL_RAWS="$(DATA)/synthetic/detail/detail.npz" PY="$(PY)"

clean-demo:
	rm -rf $(DATA)/synthetic
