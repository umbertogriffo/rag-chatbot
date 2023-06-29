.PHONY: clean

check:
	which pip3
	which python3

install:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install
	echo "Installing sentence-transformers with pip to avoid poetry's issues in installing torch..."
	. .venv/bin/activate && pip3 install sentence-transformers~=2.2.2

tidy:
	poetry run isort --skip=.venv .
	poetry run black --exclude=.venv .
	poetry run flake8 --exclude=.venv .

clean:
	echo "Cleaning Poetry environment..."
	rm -rf .venv
	rm poetry.lock
	echo "Poetry environment cleaned."
	## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
