VENV=.venv
SHELL := /bin/bash
PYTHON_VERSION=3.11.10
PYTHON=$(VENV)/bin/python
PIP = $(VENV)/bin/pip


venv: $(VENV)/bin/activate
	@echo "Checking virtual environnement is up-to-date"

$(VENV)/bin/activate:
	@echo "Upgrade pip"
	pip install --upgrade pip
	@echo "Install uv"
	pip install uv
	@echo "Setup and install venv"
	uv sync