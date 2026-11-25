import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DISABLE_SCHEDULER", "1")  # los tests no lanzan la ingesta
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirige la base de datos y el modelo a una carpeta temporal."""
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    return tmp_path
