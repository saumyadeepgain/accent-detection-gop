# Common tasks. Override any variable on the command line, e.g.
#   make train ARGS="--lr 3e-4 --batch_size 8"
#   make predict AUDIO=clip.wav TEXT="please call stella"
#   make eval SPLIT=val

PYTHON          ?= python
CONFIG          ?= config.yaml
OUT             ?= runs/default
CKPT            ?= $(OUT)/best.pt
SPLIT           ?= test
AUDIO           ?=
TEXT            ?=
REPORT          ?= report.json
GOP_MANIFEST    ?= data/gop_manifest.csv
ARGS            ?=
DEMO_DIR        ?= runs/demo
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu

.DEFAULT_GOAL := help
.PHONY: help install install-cpu install-dev test cov lint format check train resume \
        eval eval-gop predict demo synthetic clean

help: ## list targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-12s %s\n", $$1, $$2}'

install: ## install pinned runtime dependencies (+ this package)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-cpu: ## same, with CPU-only PyTorch wheels
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url $(TORCH_CPU_INDEX)
	$(PYTHON) -m pip install --no-deps -e .

install-dev: install ## + pinned test / lint tools
	$(PYTHON) -m pip install -r requirements-dev.txt

test: ## run the test suite (CPU, synthetic audio, no downloads)
	$(PYTHON) -m pytest $(ARGS)

cov: ## tests with a coverage report
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing $(ARGS)

lint: ## ruff lint + format check (same as CI)
	ruff check .
	ruff format --check .

format: ## auto-fix lint issues and format
	ruff check --fix .
	ruff format .

check: lint test ## everything CI runs

train: ## train the accent classifier (config.yaml + ARGS overrides)
	$(PYTHON) -m src.train --config $(CONFIG) --out_dir $(OUT) $(ARGS)

resume: ## continue an interrupted run from $(OUT)/last.pt
	$(PYTHON) -m src.train --config $(CONFIG) --out_dir $(OUT) --resume $(OUT)/last.pt $(ARGS)

eval: ## accent metrics on a split (SPLIT=test|val|train|all) + confusion matrix PNG
	$(PYTHON) -m src.evaluate accent --checkpoint $(CKPT) --split $(SPLIT) $(ARGS)

eval-gop: ## GOP metrics (Pearson/Spearman, AUC, EER) for GOP_MANIFEST
	$(PYTHON) -m src.evaluate gop --manifest $(GOP_MANIFEST) --out-dir $(OUT)/gop $(ARGS)

predict: ## accent + pronunciation report: make predict AUDIO=file_or_dir TEXT="..."
	@test -n "$(AUDIO)" || (echo "usage: make predict AUDIO=clip.wav [TEXT='reference text'] [CKPT=...]" && exit 1)
	$(PYTHON) -m src.predict --audio $(AUDIO) --output $(REPORT) \
		$(if $(TEXT),--text "$(TEXT)",) \
		$(if $(wildcard $(CKPT)),--accent-checkpoint $(CKPT),) $(ARGS)

synthetic: ## write a small synthetic (not speech!) dataset to $(DEMO_DIR)/data
	$(PYTHON) -m src.synthetic $(DEMO_DIR)/data --speakers 4 --utterances 3

demo: synthetic ## offline smoke run: synthetic data -> train -> eval -> predict (CPU, < 1 min)
	$(PYTHON) -m src.train --set data.manifest=$(DEMO_DIR)/data/manifest.csv \
		--model.backbone mel_cnn --hidden_dim 64 --epochs 30 --lr 3e-3 --patience 10 \
		--batch_size 8 \
		--num_workers 0 --out_dir $(DEMO_DIR)/run $(ARGS)
	$(PYTHON) -m src.evaluate accent --checkpoint $(DEMO_DIR)/run/best.pt --split test \
		--set data.num_workers=0
	$(PYTHON) -m src.predict --audio $(DEMO_DIR)/data/audio/Tamil \
		--accent-checkpoint $(DEMO_DIR)/run/best.pt --output $(DEMO_DIR)/predictions.json

clean: ## remove caches and the demo run (keeps runs/default)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov pytest-report.xml $(DEMO_DIR)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
