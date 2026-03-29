"""Simple CLI for running the database migrations using Alembic."""

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from core.config import settings
from helpers.log import get_logger

logger = get_logger(__name__)


def main():
    """Run alembic upgrade head to apply all migrations."""
    alembic_ini_path = Path(__file__).parent / "alembic.ini"

    if not alembic_ini_path.exists():
        logger.error(f"Error: alembic.ini not found at {alembic_ini_path}")
        sys.exit(1)

    logger.info(f"Running migrations from {alembic_ini_path}")
    try:
        # Create Alembic config and run migrations
        config = Config(str(alembic_ini_path))
        config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
        command.upgrade(config, "head")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
