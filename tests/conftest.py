"""Shared test configuration.

Slow tests: a test marked @pytest.mark.slow (optionally @pytest.mark.slow(reason="about 40 s on europa")) is skipped
unless the environment variable PYGEOINF0_SLOW_TESTS is set to 1, e.g. `PYGEOINF0_SLOW_TESTS=1 python -m pytest -q`
from src/. They are the weighted-measure routes of EXP-031 that take tens of seconds to minutes each (the author's
decision, 2026-10-28; src/README.md, Testing).
"""
import os

import pytest

SLOW = os.environ.get("PYGEOINF0_SLOW_TESTS") == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow(reason=...): skipped unless PYGEOINF0_SLOW_TESTS=1 (src/README.md, Testing)")


def pytest_collection_modifyitems(config, items):
    if SLOW:
        return
    for item in items:
        mark = item.get_closest_marker("slow")
        if mark is not None:
            why = mark.kwargs.get("reason") or (mark.args[0] if mark.args else "")
            item.add_marker(pytest.mark.skip(reason="slow test, skipped unless PYGEOINF0_SLOW_TESTS=1"
                                             + (f" ({why})" if why else "")))
