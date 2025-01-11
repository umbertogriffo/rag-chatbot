from pathlib import Path

import pytest


@pytest.fixture
def mock_models_folder(tmp_path):
    models_folder = tmp_path / "models"
    Path(models_folder).mkdir()
    return models_folder
