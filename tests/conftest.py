from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from api.deps import get_db_session, get_index, get_llm_client
from bot.client.openai_client import OpenAIClient
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import Model, get_model_settings
from main import app
from sqlmodel import Session, create_engine
from starlette.testclient import TestClient


@pytest.fixture(scope="session")
def mock_models_folder(tmp_path_factory):
    models_folder = tmp_path_factory.mktemp("models")
    return models_folder


@pytest.fixture(scope="session")
def llama_server_url():
    """
    URL of the llama.cpp server for integration tests.
    Override this fixture or set LLAMA_SERVER_BASE_URL environment variable.
    """
    import os
    return os.getenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def model_settings():
    """Get model settings for tests."""
    model_setting = get_model_settings(Model.LLAMA_3_2_one.value)
    return model_setting


@pytest.fixture(scope="session")
def openai_client(llama_server_url, model_settings):
    """
    Create OpenAI-compatible client for tests.
    
    Note: This requires a running llama.cpp server at llama_server_url.
    """
    return OpenAIClient(
        base_url=llama_server_url,
        model_name=Model.LLAMA_3_2_one.value,
        model_settings=model_settings,
        timeout=300,
    )


@pytest.fixture
def chroma_instance(tmp_path):
    return Chroma(embedding=Embedder(), persist_directory=str(tmp_path), is_persistent=True)


@pytest.fixture(scope="session")
def db_engine(tmp_path_factory, session_mocker):
    """
    Create a session-scoped database engine.
    Database is created once and migrations run once for all tests.
    """

    # Create a temporary database file for SQLite
    temp_dir = tmp_path_factory.mktemp("db")
    db_path = temp_dir / "test.db"
    db_url = f"sqlite:///{db_path}"

    # Use monkeypatch to set DATABASE_URL environment variable
    session_mocker.patch("core.config.settings.DATABASE_URL", db_url)

    # Get path to alembic.ini
    src_dir = Path(__file__).parents[1] / "backend"
    alembic_ini_path = src_dir / "alembic.ini"

    # Create Alembic config and run migrations
    config = Config(str(alembic_ini_path))
    config.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(config, "head")

    engine = create_engine(db_url, connect_args={"check_same_thread": False})

    yield engine

    # Clean up at the end of the test session
    engine.dispose()


@pytest.fixture(name="session")
def session_fixture(db_engine) -> Session:
    """
    Create a new database session for a test, wrapped in a transaction that is rolled back after the test.
    """

    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    # Rollback the transaction (this undoes all changes made during the test)
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(name="client_with_overridden_deps")
def client_fixture(session: Session, openai_client: OpenAIClient, chroma_instance: Chroma):
    def get_db_session_override():
        return session

    def get_llm_client_override():
        return openai_client

    def get_index_client_override():
        return chroma_instance

    app.dependency_overrides[get_db_session] = get_db_session_override
    app.dependency_overrides[get_llm_client] = get_llm_client_override
    app.dependency_overrides[get_index] = get_index_client_override

    client = TestClient(app)

    yield client

    app.dependency_overrides.clear()
