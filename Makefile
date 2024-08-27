.PHONY: check install setup update test clean

llama_cpp_file=version/llama_cpp
llama_cpp_version=`cat $(llama_cpp_file)`

check:
	which pip3
	which python3

install_cuda:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install --extras "cuda-acceleration" --no-root --no-ansi
	echo "Installing llama-cpp-python with pip to get NVIDIA CUDA acceleration"
	. .venv/bin/activate && CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python==$(llama_cpp_version) -v

install_metal:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install --no-root --no-ansi
	echo "Installing llama-cpp-python with pip to get Metal GPU acceleration for macOS systems only (it doesn't install CUDA dependencies)"
	. .venv/bin/activate && CMAKE_ARGS="-DGGML_METAL=on" pip3 install llama-cpp-python==$(llama_cpp_version) -v

install_pre_commit:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type pre-commit

setup_cuda: install_cuda install_pre_commit
setup_metal: install_metal install_pre_commit

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
	echo "Cleaning all compiled Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	echo "Cleaning the cache..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache
