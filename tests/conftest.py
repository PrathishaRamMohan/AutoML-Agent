import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from automl_agent import config  # noqa: E402

SAMPLE = ROOT / "data" / "sample_churn.csv"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Tests never call the real LLM (fast + free + deterministic)."""
    monkeypatch.setattr(config, "USE_LLM", False)


@pytest.fixture
def sample_path():
    if not SAMPLE.exists():
        import runpy
        runpy.run_path(str(ROOT / "scripts" / "make_sample_data.py"))
    return str(SAMPLE)
