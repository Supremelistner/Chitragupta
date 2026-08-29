# Chitragupta — developer task runner
# Run `make` to see available targets, or `make help`.

SHELL := pwsh.exe
SHELLFLAGS := -NoProfile -Command

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help install dev-install test test-unit test-integration lint format typecheck \
        security clean run run-bg run-dev stop status docker-up docker-down docker-monitoring \
        ci setup-hooks

help:  ## Show this help.
	@Write-Host "Chitragupta — make targets:" -ForegroundColor Cyan
	@Select-String -Path $('$(MAKEFILE_LIST)' | Get-Item) -Pattern '^[a-zA-Z_-]+:.*?## .*$' | ForEach-Object { $tokens = $_ -split ':.*?## '; Write-Host ("  {0,-22} {1}" -f $tokens[0], $tokens[1]) }

install:  ## Install runtime dependencies.
	$(PIP) install -e .

dev-install:  ## Install with dev/test/lint extras.
	$(PIP) install -e ".[dev]"

test:  ## Run the full test suite.
	$(PYTHON) -m pytest -q

test-unit:  ## Run unit tests only (skip integration).
	$(PYTHON) -m pytest -q -m "not integration"

test-integration:  ## Run integration tests.
	$(PYTHON) -m pytest -q -m integration

test-cov:  ## Run tests with coverage report.
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:  ## Run ruff linter.
	$(PYTHON) -m ruff check src tests

format:  ## Auto-format with ruff.
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

typecheck:  ## Run mypy.
	$(PYTHON) -m mypy src

security:  ## Run security scanners.
	$(PYTHON) -m bandit -r src
	$(PYTHON) -m safety check

ci: lint typecheck test  ## Run the full CI pipeline locally.

clean:  ## Remove caches, bytecode, and build artifacts.
	Get-ChildItem -Path . -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
	Get-ChildItem -Path . -Recurse -Directory -Filter '.pytest_cache' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
	Get-ChildItem -Path . -Recurse -Directory -Filter '.mypy_cache' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
	Get-ChildItem -Path . -Recurse -Directory -Filter '.ruff_cache' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
	Get-ChildItem -Path . -Recurse -Filter '*.pyc' -ErrorAction SilentlyContinue | Remove-Item -Force
	Write-Host "Cleaned caches." -ForegroundColor Green

run:  ## Start all services in the foreground.
	$(PYTHON) activate.py

run-bg:  ## Start all services in the background.
	$(PYTHON) activate.py --bg

run-dev:  ## Start all services with hot-reload.
	$(PYTHON) activate.py --dev

stop:  ## Stop background services.
	$(PYTHON) activate.py --stop

status:  ## Show service health.
	$(PYTHON) activate.py --status

docker-up:  ## Start PostgreSQL + Qdrant.
	docker compose up -d

docker-down:  ## Stop infrastructure containers.
	docker compose down

docker-monitoring:  ## Start Prometheus + Grafana.
	docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

setup-hooks:  ## Install pre-commit hooks.
	$(PIP) install pre-commit
	pre-commit install
