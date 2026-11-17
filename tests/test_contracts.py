"""Mechanical contracts: every backend implements the Backend protocol (or raises Unsupported), obeys the laws of a
realisation, and every space has claims and a consistent equality."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import Unsupported
from pygeoinf0.realisations.backends import Backend

REQUIRED = ("fit", "lincomb", "combine", "take", "hstack", "gram", "evaluate", "error", "describe", "summary")
OPTIONAL = ("multiply", "integral", "lp_norm", "measure_of_cell", "product_family", "to_tensor", "legendre_decay_generators",
            "riesz")

SPACES = [
    ("interval", lambda: pg.L2("[0, 1]"), lambda x: np.sin(3 * x) + x),
    ("box", lambda: pg.L2(pg.Box("[0, 1]", "[0, 2]")), lambda x, y: np.sin(x) * y),
    ("disk", lambda: pg.L2(pg.Disk(1.0)), lambda x, y: np.exp(x) * np.cos(y)),
    ("disk legendre", lambda: pg.L2(pg.Disk(1.0), realisation="legendre"), lambda x, y: np.exp(x) * np.cos(y)),
    ("ball", lambda: pg.L2(pg.Ball(1.0)), lambda x, y, z: x * y + z),
    ("circle", lambda: pg.L2(pg.Circle(2.0)), lambda x, y: np.exp(x / 2)),
    ("torus", lambda: pg.L2(pg.Torus("[0, 6.283185307179586]", "[0, 1]")), lambda x, y: np.sin(x) * np.cos(2 * np.pi * y)),
    ("sphere", lambda: pg.L2(pg.Sphere(1.0)), lambda x, y, z: x * z + 1),
    ("sphere tensor", lambda: pg.L2(pg.Sphere(1.0), realisation="tensor"), lambda x, y, z: x * z + 1),
]


@pytest.mark.parametrize("name, make, fun", SPACES, ids=[s[0] for s in SPACES])
def test_backend_protocol_and_laws(name, make, fun):
    M = make()
    B = M.backend
    assert isinstance(B, Backend)
    for op in REQUIRED:
        assert type(B).supports(op), f"{type(B).__name__} lacks the required {op}"
    f, g = M(fun), M(lambda *X: 1.0 + 0 * X[0])
    F = M.family([f, g, f + g])
    R = F.realisation
    G = B.gram(R, R)
    assert np.allclose(G, G.T, rtol=1e-12, atol=1e-14 * np.abs(G).max())
    assert np.allclose(B.sqnorms(R), np.diag(G), rtol=1e-12)
    assert np.all(B.error(R) >= 0)
    assert np.allclose(B.gram(B.take(R, [2]), R), G[[2]], rtol=1e-12, atol=1e-14 * np.abs(G).max())
    H = B.hstack([B.take(R, [0]), B.take(R, [1])])
    assert np.allclose(B.gram(H, R), G[:2], rtol=1e-12, atol=1e-14 * np.abs(G).max())
    A = np.array([[1.0], [2.0], [-1.0]])
    assert np.allclose(B.gram(B.combine(R, A), R), A.T @ G, rtol=1e-12, atol=1e-13 * np.abs(G).max())
    args = dict(multiply=(R, R), integral=(R,), lp_norm=(R, 2), measure_of_cell=(0.0, 0.5), product_family=([None], {}),
                to_tensor=(R,), legendre_decay_generators=(0.5, 3), riesz=([],))
    for op in OPTIONAL:
        if not type(B).supports(op):
            with pytest.raises(Unsupported):
                getattr(B, op)(*args[op])


@pytest.mark.parametrize("name, make, fun", SPACES, ids=[s[0] for s in SPACES])
def test_spaces_have_claims_and_consistent_equality(name, make, fun):
    M1, M2 = make(), make()
    assert len(M1.claims) >= 2
    assert M1 == M2 and hash(M1) == hash(M2)


def test_different_realisations_are_one_space():
    """Decision Q4 (plan 069 P1): one space, whatever realises its elements; mixing them is refused by name."""
    from pygeoinf0.core.errors import RealisationMismatch
    X, Y = pg.L2(pg.Disk(1.0)), pg.L2(pg.Disk(1.0), realisation="legendre")
    assert X == Y and X.realisation != Y.realisation
    with pytest.raises(RealisationMismatch):
        X.inner(X.constant(1.0), Y.constant(1.0))
    with pytest.raises(ValueError):
        pg.L2("[0, 1]", realisation="harmonics")


def test_public_names_are_documented():
    for name in pg.__all__:
        obj = getattr(pg, name)
        if callable(obj) and not isinstance(obj, type(np)):
            assert getattr(obj, "__doc__", None) or name in ("R",), f"{name} has no docstring"
