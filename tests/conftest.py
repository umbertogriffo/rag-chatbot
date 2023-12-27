from pathlib import Path

import pytest

root_folder = Path(__file__).resolve().parent.parent


@pytest.fixture
def mock_model_folder(tmp_path):
    model_folder = root_folder / "models"
    return model_folder
