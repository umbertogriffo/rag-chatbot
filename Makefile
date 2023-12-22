.PHONY: clean

check:
	which pip3
	which python3

install:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install
	echo "Installing sentence-transformers with pip to avoid poetry's issues in installing torch... (it doesn't install CUDA dependencies)"
	. .venv/bin/activate && pip3 install sentence-transformers~=2.2.2
	echo "Installing llama-cpp-python with pip to get NVIDIA CUDA acceleration"
	. .venv/bin/activate && CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python~=0.2.23

update:
	poetry lock --no-update
	poetry install

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
