import os
import tempfile
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlmodel import Session, create_engine


@pytest.fixture
def mock_models_folder(tmp_path):
    models_folder = tmp_path / "models"
    Path(models_folder).mkdir()
    return models_folder


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
    # TODO: Alternatively, you can create tables directly without migrations for simpler setups.
    # create_db_and_tables(engine)

    connection = engine.connect()
    session = Session(bind=connection)

    yield session

    session.close()
    connection.close()

    # Clean up
    engine.dispose()
    if path:
        os.unlink(path)
