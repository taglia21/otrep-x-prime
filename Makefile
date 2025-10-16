# OTREP-X PRIME Build System

.PHONY: install test run clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run:
	python -m scripts.otrep_cli run

migrate:
	python -m scripts.otrep_cli migrate

seed:
	python -m scripts.otrep_cli seed

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
