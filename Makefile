# vLLM Tutorial — Makefile
# Run `make help` to see all available targets.

UV_BINARY ?= $(shell export PATH="$$HOME/.local/bin:$$PATH" && command -v uv 2>/dev/null || echo "$$HOME/.local/bin/uv")
UV_RUN ?= $(UV_BINARY) run
SRC_DIRS := examples scripts
MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
PORT ?= 8000

export UV_PROJECT_ENVIRONMENT := .venv
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: help build serve serve-advanced repl \
		lint format_check format_fix check_all \
		benchmark docker-up docker-down clean

# ──────────────────────────── Help ────────────────────────────

help: ## Show help for available make targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk -F':.*?## ' '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────── Setup ───────────────────────────

build: pyproject.toml ## Install uv + project dependencies
	@command -v uv >/dev/null || ( \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	)
	@export PATH="$$HOME/.local/bin:$$PATH" && uv --version
	@echo "Installing project dependencies..."
	@export PATH="$$HOME/.local/bin:$$PATH" && uv sync --group dev

# ──────────────────────────── Serve ───────────────────────────

serve: build ## Start vLLM server with MODEL (default: Qwen 1.5B)
	@echo "Starting vLLM server with $(MODEL) on port $(PORT)..."
	$(UV_RUN) vllm serve $(MODEL) --host 0.0.0.0 --port $(PORT)

serve-advanced: build ## Start vLLM with production config
	$(UV_RUN) vllm serve --config configs/advanced-serve.yaml

repl: build ## Run the Python REPL quickstart (offline batch inference)
	$(UV_RUN) python examples/batch-inference/batch_generate.py

# ──────────────────────────── Examples ────────────────────────

chat: build ## Run the chat completion example (server must be running)
	$(UV_RUN) python examples/python-clients/chat_completion.py

stream: build ## Run the streaming example (server must be running)
	$(UV_RUN) python examples/streaming/stream_chat.py

lora: build ## Run the LoRA multi-model example (server must be running)
	$(UV_RUN) python examples/multi-model/lora_serving.py

# ──────────────────────────── Quality ─────────────────────────

lint: build ## Run ruff linter on example scripts
	@$(UV_RUN) ruff check $(SRC_DIRS)

format_check: build ## Check code formatting (non-destructive)
	@$(UV_RUN) ruff format --check $(SRC_DIRS)

format_fix: build ## Auto-format and fix lint issues
	@$(UV_RUN) ruff format $(SRC_DIRS)
	@$(UV_RUN) ruff check --fix $(SRC_DIRS)
	@echo "Done — code formatted and lint issues auto-fixed."

check_all: lint format_check ## Run all quality checks

# ──────────────────────────── Benchmark ───────────────────────

benchmark: ## Run benchmark suite (server must be running)
	@bash scripts/benchmark.sh

# ──────────────────────────── Docker ──────────────────────────

docker-up: ## Start full stack (vLLM + Prometheus + Grafana)
	docker compose -f docker/docker-compose.yaml up -d

docker-down: ## Stop full stack
	docker compose -f docker/docker-compose.yaml down

docker-logs: ## Tail vLLM server logs
	docker compose -f docker/docker-compose.yaml logs -f vllm-server

# ──────────────────────────── Clean ───────────────────────────

clean: ## Remove virtualenv, caches, and build artifacts
	@rm -rf .venv .pytest_cache .ruff_cache build/ __pycache__
	@echo "Cleaned."
