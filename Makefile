PYTHON ?= python

.PHONY: run test lint format

run:
	$(PYTHON) main.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m flake8 utils tests core main.py

format:
	$(PYTHON) -m black .
