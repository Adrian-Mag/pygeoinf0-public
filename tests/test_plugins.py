"""The plugin rules (plan 019 P3, EXP-034 DESIGN.md section 4): explicit registration and loading, listing from entry
points without importing, the port range, provenance on the objects, claims capped at asserted/checked, core names
protected, and the contract kit."""
import importlib.metadata
import sys

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.claims import ASSERTED, CHECKED


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(pg.port.plugins, "_REGISTRY", {})
    sys.modules.pop("plugin_demo", None)


def fake_entry_points(monkeypatch, *pairs):
    eps = [importlib.metadata.EntryPoint(name, value, pg.port.plugins.GROUP) for name, value in pairs]
    monkeypatch.setattr(pg.port.plugins, "_entry_points", lambda: eps)


def test_available_lists_without_importing(monkeypatch):
    fake_entry_points(monkeypatch, ("demo", "plugin_demo:PLUGIN"))
    assert [a[:2] for a in pg.port.plugins.available()] == [("demo", "plugin_demo:PLUGIN")]
    assert "plugin_demo" not in sys.modules


def test_load_imports_registers_and_stamps(monkeypatch):
    fake_entry_points(monkeypatch, ("demo", "plugin_demo:PLUGIN"))
    p = pg.port.plugins.load("demo")
    assert pg.port.plugins.load("demo") is p and pg.port.plugins.loaded() == {"demo": p}
    sq = p.objects["square"]
    assert sq.claims.status("provenance") == (pg.Truth.TRUE, ASSERTED)
    assert "plugin demo 0.1 (tests/plugin_demo.py)" in sq.provenance
    assert p.versions["numpy"] == np.__version__
    c = p.claims[0]                                            # stated as theorem: capped
    assert c.status == ASSERTED and "stated by plugin demo 0.1 as theorem" in c.reason


def test_explicit_registration_and_its_refusals():
    from plugin_demo import PLUGIN
    assert pg.port.plugins.register(PLUGIN) is PLUGIN
    assert pg.port.plugins.register(PLUGIN) is PLUGIN               # idempotent
    other = pg.port.plugins.Plugin(name="demo", version="0.2")
    with pytest.raises(pg.Refused, match="a different plugin is registered as 'demo'"):
        pg.port.plugins.register(other)
    with pytest.raises(pg.Refused, match="supports port >=2"):
        pg.port.plugins.register(pg.port.plugins.Plugin(name="future", version="1", port=">=2,<3"))
    with pytest.raises(pg.Refused, match="replace core names"):
        pg.port.plugins.register(pg.port.plugins.Plugin(name="bad", version="1", objects={"Map": object()}))


def test_unknown_plugin_and_bad_entry_point(monkeypatch):
    fake_entry_points(monkeypatch, ("odd", "numpy:pi"))
    with pytest.raises(pg.Refused, match="no plugin named 'nothing'"):
        pg.port.plugins.load("nothing")
    with pytest.raises(pg.Refused, match="is not a pg.port.plugins.Plugin"):
        pg.port.plugins.load("odd")


def test_contract_kit():
    from plugin_demo import PLUGIN
    claims = pg.port.plugins.check(PLUGIN)
    assert all(c.status == CHECKED for c in claims)
    assert claims.status("E P = I") == (pg.Truth.TRUE, CHECKED)
    assert claims.status("derivative") == (pg.Truth.TRUE, CHECKED)
    assert claims.status("adjoint") == (pg.Truth.TRUE, CHECKED)
    assert claims.status("claims") == (pg.Truth.FALSE, CHECKED)          # before register caps it, the theorem is too strong
    pg.port.plugins.register(PLUGIN)
    assert pg.port.plugins.check(PLUGIN).status("claims") == (pg.Truth.TRUE, CHECKED)
