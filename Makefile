.PHONY: check update test clean start migrate_db

check:
	which pip3
	which python3

install_dependencies:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install --no-root --no-ansi

install_pre_commit:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type pre-commit

migrate_db:
	cd backend && poetry run python migration.py

start_llama_server_cuda:
	docker compose up -d

start_llama_server_metal:
	docker compose -f docker-compose.metal.yml up -d

stop_llama_server:
	docker compose down

setup_cuda: install_dependencies install_pre_commit migrate_db start_llama_server_cuda
setup_metal: install_dependencies install_pre_commit migrate_db start_llama_server_metal

start:
	sh start.sh

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
