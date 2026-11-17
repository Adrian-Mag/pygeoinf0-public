"""The continuum keeps no numerics (plan 069 P2; architecture review 2026-11-13, C2).

- Regions (sets/, domains/) keep chart, Jacobian, measure and the geometric protocol; they define no quadrature. The
  rule that integrates over a region is pygeoinf0.quadrature.rule(S, n), reached by the region's charts (S.charts()),
  and the Gauss counts for the Jacobians live there too (quadrature.jacobian_nodes).
- DensityMeasure holds its weight: constructing one fits nothing; the piecewise Legendre series of an interval density
  is made on first use, by the numerical column (realisations/series/legendre.fit_density).
- core, sets/, measures/, domains/ and spaces/ import no realisation at import time (each module in a fresh
  interpreter, the package namespace pg.* aside, which imports everything by design).
"""
import ast
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pygeoinf0 as pg

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(SRC, "pygeoinf0")
LOWER = ("core", "sets", "measures", "domains", "spaces")


def _region_classes_with(name, packages=("sets", "domains")):
    out = []
    for p in packages:
        for root, _, files in os.walk(os.path.join(PKG, p)):
            for f in files:
                if f.endswith(".py"):
                    tree = ast.parse(open(os.path.join(root, f)).read())
                    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
                        out += [f"{p}/{f}:{cls.name}.{d.name}" for d in cls.body
                                if isinstance(d, ast.FunctionDef) and d.name == name]
    return out


def test_regions_define_no_quadrature():
    assert _region_classes_with("quadrature") == []


def test_the_jacobians_stay_and_their_gauss_counts_leave():
    from pygeoinf0 import quadrature
    from pygeoinf0.domains import JACOBIANS
    t = np.linspace(0.1, 3.0, 7)
    assert np.allclose(JACOBIANS["r2"](t), t * t) and np.allclose(JACOBIANS["4.0*sin"](t), 4 * np.sin(t))
    assert [quadrature.jacobian_nodes(k) for k in ("1", "r", "r2", "sin", "4.0*sin")] == [0, 1, 2, 12, 12]
    with pytest.raises(KeyError):
        quadrature.jacobian_nodes("cos")


REGIONS = {
    "interval": lambda: pg.Interval(-1, 2),
    "box": lambda: pg.Box((0, 1), (0, 2)),
    "disk": lambda: pg.Disk(1.5),
    "ball": lambda: pg.Ball(1.0),
    "shell": lambda: pg.Shell(0.5, 1.0),
    "sphere": lambda: pg.Sphere(2.0),
    "circle": lambda: pg.Circle(1.0),
    "polyline": lambda: pg.Polyline([(0, 0), (1, 0), (1, 1)]),
    "simplex": lambda: pg.Simplex((0, 0), (1, 0), (0, 1)),
}


@pytest.mark.parametrize("name", sorted(REGIONS))
def test_the_rule_of_a_region_carries_its_measure(name):
    from pygeoinf0 import quadrature
    S = REGIONS[name]()
    X, w = quadrature.rule(S, 16)
    assert X.shape == (len(w), S.cartesian_dim if hasattr(S, "cartesian_dim") else S.ambient_dim or S.dim)
    assert abs(w.sum() - S.volume) <= 1e-12 * S.volume


def test_a_region_without_charts_has_no_rule():
    from pygeoinf0 import quadrature
    S = pg.LayeredBall([0.5, 1.0])
    assert S.charts() is None
    with pytest.raises(NotImplementedError, match="states no chart"):
        quadrature.rule(S, 8)


def test_density_measure_holds_its_weight_and_fits_on_demand(monkeypatch):
    from pygeoinf0.realisations.series import legendre as L
    calls = []
    fit = L.fit

    def counted(*a, **k):
        calls.append(1)
        return fit(*a, **k)
    monkeypatch.setattr(L, "fit", counted)
    mu = pg.DensityMeasure(pg.LebesgueMeasure("[0, 1]"), lambda x: 1 + x * x)
    assert calls == [] and mu.weight.kind == "callable"
    assert mu.mass()[0] == pytest.approx(4 / 3, rel=1e-14)
    assert len(calls) == 1
    assert mu.density is mu.density and len(calls) == 1          # fitted once, on first use


def test_density_measure_refuses_an_unresolved_density_when_fitted():
    from pygeoinf0.core.errors import RealisationError
    mu = pg.DensityMeasure(pg.LebesgueMeasure("[0, 1]"), lambda x: 2 + np.sin(1 / (x + 1e-9)))     # held: no fit yet
    with pytest.raises(RealisationError, match="could not resolve the density"):
        mu.density


_PROBE = textwrap.dedent("""
    import importlib, sys, types
    root, name = sys.argv[1], sys.argv[2]
    pkg = types.ModuleType("pygeoinf0"); pkg.__path__ = [root]; sys.modules["pygeoinf0"] = pkg
    importlib.import_module("pygeoinf0." + name)
    print(" ".join(sorted(k for k in sys.modules if k.startswith("pygeoinf0.realisations"))))
""")


def realisations_imported(root, module):
    """The pygeoinf0.realisations modules that importing pygeoinf0.<module> loads, in a fresh interpreter with the
    package's namespace (pygeoinf0/__init__.py, pg.*, which imports everything by design) left out."""
    out = subprocess.run([sys.executable, "-c", _PROBE, root, module], capture_output=True, text=True, check=True)
    return out.stdout.split()


def _lower_modules():
    for p in LOWER:
        for f in sorted(os.listdir(os.path.join(PKG, p))):
            if f.endswith(".py"):
                yield p if f == "__init__.py" else f"{p}.{f[:-3]}"


def test_lower_packages_import_no_realisation_at_import_time():
    bad = {m: r for m in _lower_modules() if (r := realisations_imported(PKG, m))}
    assert not bad, f"imported at import time: {bad}"


def test_the_import_time_check_catches_its_planted_defect(tmp_path):
    root = tmp_path / "pygeoinf0"
    (root / "realisations").mkdir(parents=True)
    (root / "core").mkdir()
    (root / "realisations" / "__init__.py").write_text("")
    (root / "realisations" / "legendre.py").write_text("")
    (root / "core" / "__init__.py").write_text("")
    (root / "core" / "planted.py").write_text("from ..realisations import legendre\n")
    (root / "core" / "lazy.py").write_text("def f():\n    from ..realisations import legendre\n")
    assert realisations_imported(str(root), "core.planted") == ["pygeoinf0.realisations",
                                                                 "pygeoinf0.realisations.legendre"]
    assert realisations_imported(str(root), "core.lazy") == []
