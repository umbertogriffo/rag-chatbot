from pathlib import Path

import pytest


@pytest.fixture
def mock_model_folder(tmp_path):
    model_folder = tmp_path / "models"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    return model_folder
