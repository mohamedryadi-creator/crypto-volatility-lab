.PHONY: install test lint typecheck check demo

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

check: lint typecheck test

demo:
	crypto-vol-lab demo --output reports/generated
