from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from api.deps import get_db_session, get_index, get_llm_client
from bot.client.lama_cpp_client import LamaCppClient
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
def client_fixture(session: Session, lamacpp_client: LamaCppClient, chroma_instance: Chroma):
    def get_db_session_override():
        return session

    def get_llm_client_override():
        return lamacpp_client

    def get_index_client_override():
        return chroma_instance

    app.dependency_overrides[get_db_session] = get_db_session_override
    app.dependency_overrides[get_llm_client] = get_llm_client_override
    app.dependency_overrides[get_index] = get_index_client_override

    client = TestClient(app)

    yield client

    app.dependency_overrides.clear()
