.PHONY: check setup install_pre_commit migrate_db start update tidy test check-formatting clean start_server start_docker

check:
	which pip3
	which python3

setup:
	echo "Installing..."
	mkdir -p .venv
	poetry config virtualenvs.in-project true
	poetry install --no-root --no-ansi
	$(MAKE) install_pre_commit
	$(MAKE) migrate_db

install_pre_commit:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type pre-commit

migrate_db:
	cd backend && PYTHONPATH=.:../chatbot poetry run python migration.py

start:
	sh start.sh

start_server:
	@echo "Starting llama.cpp server..."
	@echo "Note: This requires llama.cpp server binary and a model file."
	@echo "See notes/llama-server-docker.md for instructions."
	@echo "Example: llama-server -m models/model.gguf --host 0.0.0.0 --port 8080"

start_docker:
	docker-compose up -d

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
