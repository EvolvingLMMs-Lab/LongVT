.PHONY: style format


style:
	python -m black --line-length 119 .
	python -m isort .
	ruff check --fix .


quality:
	python -m black --check --line-length 119 .
	python -m isort --check-only .
	ruff check .