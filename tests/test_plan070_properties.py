"""Plan 070 P1 (review D5, decision Q10): the property space P and the model-property relation have a home, the
top-level package properties/. The property families are defined there and nowhere else; pg.* keeps their names (a
clean break: the old modules operators/families.py and operators/harmonic_targets.py are gone, no forwarders); the
numerics a property family needs but the lower layers also use stay below it (rowquad with the lazy covariance in
probability/, the harmonic coefficients of a symbolic target in realisations/)."""
import ast
import importlib
import pathlib

import pytest

import pygeoinf0 as pg

PKG = pathlib.Path(pg.__file__).parent
PROPERTY_NAMES = ("CapAverages", "RotatedFamily", "FactoredFamily", "harmonic_target", "directions")


def _definitions(name):
    """The modules (relative paths) of the package that define a top-level class or function called name."""
    found = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if any(isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name for n in tree.body):
            found.append(path.relative_to(PKG).as_posix())
    return found


@pytest.mark.parametrize("name", PROPERTY_NAMES)
def test_the_property_families_live_in_properties_and_nowhere_else(name):
    where = _definitions(name)
    assert where and all(w.startswith("properties/") for w in where), f"{name} defined in {where}"


@pytest.mark.parametrize("name", ["CapAverages", "RotatedFamily", "harmonic_target"])
def test_pg_keeps_the_names(name):
    import pygeoinf0.properties as P
    assert name in pg.__all__ and getattr(pg, name) is getattr(P, name)
    assert getattr(pg, name).__module__.startswith("pygeoinf0.properties.")


@pytest.mark.parametrize("module", ["pygeoinf0.operators.families", "pygeoinf0.operators.harmonic_targets"])
def test_the_old_modules_are_gone(module):
    with pytest.raises(ImportError):
        importlib.import_module(module)


def test_the_numerics_below_stay_below():
    """rowquad (diag(Phi V Phi^T)) is the lazy covariance's diagonal; the harmonic coefficients of a symbolic target and
    the box holding a measure's support serve spaces/ and realisations/, so they live in the numerical column."""
    assert _definitions("rowquad") == ["solvers/dataspace/quadratic.py"]          # plan 070 P3: down to the engine
    for name in ("support_box", "realised_moment", "coefficients"):
        assert "realisations/series/symbolic_harmonics.py" in _definitions(name), name


def test_properties_states_the_property_space_and_the_model_property_relation():
    import pygeoinf0.properties as P
    doc = P.__doc__
    assert "property space" in doc and "model-property relation" in doc
    assert "seven elements" in doc


def test_properties_has_its_layer():
    from test_structure import RANK
    assert RANK["operators"] < RANK["relations"] < RANK["properties"] < RANK["inference"]
    assert RANK["states"] < RANK["properties"]
