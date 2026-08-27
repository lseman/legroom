# legroom — build, test, and install
#
# Targets:
#   build      — build wheel/sdist
#   test       — run tests
#   clean      — remove build artifacts
#   install    — install in development mode
#   help       — show targets

SHELL := /bin/bash
PYTHON := python3
PIP    := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

.PHONY: build test clean install help

help:
	@echo "legroom Makefile targets:"
	@echo "  build    — Build wheel/sdist"
	@echo "  test     — Run tests"
	@echo "  clean    — Remove build artifacts"
	@echo "  install  — Install package in dev mode"

build:
	$(PYTHON) -m build

test:
	$(PYTEST) tests -v

clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

install:
	$(PIP) install -e .
