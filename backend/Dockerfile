FROM public.ecr.aws/docker/library/python:3.12-slim AS build

ARG POETRY_VERSION

WORKDIR /app

## Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV LOGLEVEL="INFO"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy requirements and install
COPY pyproject.toml poetry.lock /app/

# Poetry will complain if a `README.md` is not found and as such we create an empty one.
# We could have copied the local one but this would effectively prevent Docker layer caching every time we modify it.
RUN touch /app/README.md

RUN poetry install --no-root --no-ansi --without dev \
    && rm -rf /root/.cache

# Stage 2: Runtime environment
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# Install wget used for the liveness and readiness probe
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy actual source code
COPY chatbot/ /app/chatbot/
COPY backend/ /app/backend/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
