.PHONY: check install setup update test clean

file=version/llama_cpp
llama_cpp_version=`cat $(file)`

check:
	which pip3
	which python3

install:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install --no-root --no-ansi
	echo "Installing llama-cpp-python with pip to get NVIDIA CUDA acceleration"
	. .venv/bin/activate && CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python==$(llama_cpp_version)

install_pre_commit:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type pre-commit

setup: install install_pre_commit

update:
	poetry lock --no-update
	poetry install

tidy:
	poetry run ruff format --exclude=.venv .
	poetry run ruff check --exclude=.venv . --fix

test:
	poetry run pytest --log-cli-level=DEBUG --capture=tee-sys -v

check-formatting:
	poetry run ruff format . --check

clean:
	echo "Cleaning Poetry environment..."
	rm -rf .venv
	rm poetry.lock
	echo "Poetry environment cleaned."
	## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
