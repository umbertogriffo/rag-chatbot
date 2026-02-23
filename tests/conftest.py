import sys
from pathlib import Path

import pytest

# Ensure both the chatbot and backend packages are importable.
_ROOT = Path(__file__).parent.parent
for _dir in ("backend", "chatbot"):
    _abs = str((_ROOT / _dir).resolve())
    if _abs not in sys.path:
        sys.path.insert(0, _abs)


@pytest.fixture
def mock_models_folder(tmp_path):
    models_folder = tmp_path / "models"
    Path(models_folder).mkdir()
    return models_folder
