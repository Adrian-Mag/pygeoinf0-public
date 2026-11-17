"""src/API.md is the current public API (plan 026): regenerate it with `python -m pygeoinf0.api_record > API.md`."""
from pathlib import Path

from pygeoinf0.api_record import api_record


def test_api_md_is_current():
    path = Path(__file__).resolve().parents[1] / "API.md"
    assert path.read_text() == api_record() + "\n", "src/API.md is stale: python -m pygeoinf0.api_record > API.md"
