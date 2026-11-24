# Everyday commands. The environment is the one CI uses: see CONTRIBUTING.md.
PYTHON ?= python
export OVL_DEVICE ?= cpu
export OVL_TIMESTAMP ?= 0
export TOKENIZERS_PARALLELISM ?= false

.PHONY: help setup test lint check check-release

help:  ## list the commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## install the hash-locked CPU environment and the dev tools into the current Python
	$(PYTHON) -m pip install --require-hashes -r requirements/preparation.lock
	$(PYTHON) -m pip install pytest pytest-cov ruff gguf safetensors pre-commit

test:  ## run the tests (CPU, about 25 s)
	$(PYTHON) -m pytest -q tests

lint:  ## run ruff, as CI does
	$(PYTHON) -m ruff check simple tests

check: lint test  ## everything CI runs

check-release:  ## check the published release on this computer (about 2 minutes)
	bash simple/check_release.sh
