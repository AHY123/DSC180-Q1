# Graph Learning Research Platform Makefile

.PHONY: help setup-all setup-base setup-submodules clean test lint

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-all          - Setup all environments and submodules"
	@echo "  setup-base         - Setup base project environment"
	@echo "  setup-submodules   - Initialize git submodules"
	@echo "  setup-gps          - Setup GPS model environment"
	@echo "  setup-autograph    - Setup AutoGraph model environment"
	@echo "  setup-sequence     - Setup sequence model environment"
	@echo "  train              - Run experiment (requires CONFIG=path/to/config.yaml)"
	@echo "  clean              - Clean generated files"
	@echo "  test               - Run tests"
	@echo "  lint               - Run code linting"

# Environment setup
setup-all: setup-base setup-submodules setup-gps setup-autograph setup-sequence
	@echo "✅ All environments set up successfully"

setup-base:
	@echo "Setting up base environment..."
	uv sync
	@echo "✅ Base environment ready"

setup-submodules:
	@echo "Initializing git submodules..."
	git submodule update --init --recursive
	@echo "✅ Submodules initialized"

setup-gps: setup-submodules
	@echo "Setting up GPS environment..."
	conda env create -f environments/gps/environment.yml || conda env update -f environments/gps/environment.yml
	@echo "✅ GPS environment ready"

setup-autograph: setup-submodules
	@echo "Setting up AutoGraph environment..."
	conda env create -f environments/autograph/environment.yml || conda env update -f environments/autograph/environment.yml
	@echo "✅ AutoGraph environment ready"

setup-sequence: setup-submodules
	@echo "Setting up sequence model environment..."
	conda env create -f environments/sequence/environment.yml || conda env update -f environments/sequence/environment.yml
	@echo "✅ Sequence environment ready"

# Training and experiments
train:
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Please specify CONFIG=path/to/config.yaml"; \
		exit 1; \
	fi
	@echo "Running experiment with config: $(CONFIG)"
	python scripts/run_experiment.py --config $(CONFIG)

train-gps:
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Please specify CONFIG=path/to/config.yaml"; \
		exit 1; \
	fi
	@echo "Running GPS experiment..."
	conda run -n gps python scripts/run_experiment.py --config $(CONFIG)

train-autograph:
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Please specify CONFIG=path/to/config.yaml"; \
		exit 1; \
	fi
	@echo "Running AutoGraph experiment..."
	conda run -n autograph python scripts/run_experiment.py --config $(CONFIG)

# Development
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running code linting..."
	python -m ruff check src/
	python -m black --check src/

format:
	@echo "Formatting code..."
	python -m black src/
	python -m ruff check --fix src/

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf results/checkpoints/*
	rm -rf results/logs/*
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Data generation
generate-synthetic:
	@echo "Generating synthetic datasets..."
	python scripts/generate_synthetic_data.py

# Submodule management
update-submodules:
	@echo "Updating submodules..."
	git submodule update --remote
	@echo "✅ Submodules updated"

pin-submodule:
	@if [ -z "$(REPO)" ] || [ -z "$(COMMIT)" ]; then \
		echo "❌ Please specify REPO and COMMIT"; \
		echo "Usage: make pin-submodule REPO=GraphGPS COMMIT=abc123"; \
		exit 1; \
	fi
	@echo "Pinning $(REPO) to commit $(COMMIT)..."
	cd external/$(REPO) && git checkout $(COMMIT)
	git add external/$(REPO)
	git commit -m "Pin $(REPO) to $(COMMIT)"
	@echo "✅ $(REPO) pinned to $(COMMIT)"