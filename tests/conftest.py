import os
import tempfile
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from api.deps import get_db_session, get_llm_client
from bot.client.lama_cpp_client import LamaCppClient
from bot.model.model_registry import Model, get_model_settings
from main import app
from sqlmodel import Session, create_engine
from starlette.testclient import TestClient


@pytest.fixture(scope="session")
def mock_models_folder(tmp_path_factory):
    models_folder = tmp_path_factory.mktemp("models")
    return models_folder


@pytest.fixture(scope="session")
def cpu_config():
    config = {
        "n_ctx": 512,
        "n_threads": 2,
        "n_gpu_layers": 0,
    }
    return config


@pytest.fixture(scope="session")
def model_settings(cpu_config):
    model_setting = get_model_settings(Model.QWEN_3_5_ZERO_EIGHT.value)
    model_setting.config = cpu_config
    return model_setting


@pytest.fixture(scope="session")
def lamacpp_client(mock_models_folder, model_settings):
    return LamaCppClient(mock_models_folder, model_settings)


@pytest.fixture(name="session")
def session_fixture(request, monkeypatch) -> Session:
    """Create a new database session for a test."""
    # TODO: Use an in-memory SQLite database for faster tests if possible.
    #       https://sqlmodel.tiangolo.com/tutorial/fastapi/tests/#memory-database

    # Create a temporary database file for SQLite
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{path}"

    # Use monkeypatch to set DATABASE_URL environment variable
    monkeypatch.setattr("core.config.settings.DATABASE_URL", db_url)

    # Get path to alembic.ini
    src_dir = Path(__file__).parents[1] / "backend"
    alembic_ini_path = src_dir / "alembic.ini"

    # Create Alembic config and run migrations
    config = Config(str(alembic_ini_path))
    config.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(config, "head")

    engine = create_engine(db_url)

    connection = engine.connect()
    session = Session(bind=connection)

    yield session

    session.close()
    connection.close()

    # Clean up
    engine.dispose()
    if path:
        os.unlink(path)


@pytest.fixture(name="client_with_overridden_deps")
def client_fixture(session: Session, lamacpp_client: LamaCppClient):
    def get_db_session_override():
        return session

    def get_llm_client_override():
        return lamacpp_client

    app.dependency_overrides[get_db_session] = get_db_session_override
    app.dependency_overrides[get_llm_client] = get_llm_client_override

    client = TestClient(app)

    yield client

    app.dependency_overrides.clear()
