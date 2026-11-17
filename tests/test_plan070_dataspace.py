"""Plan 070 P3 (review D4, decisions Q1 and Q7): the data-space engine is internal and lives in solvers/dataspace/,
below convex/ and probability/; pg.decompose stays the public route to it (DataDecomposition is not a public pg name);
the methods are the public routes; the engine takes its metric as an explicit argument and never infers one from a
set, a measure or nothing (a call without one refuses, naming the choices: D.whitened(C), the noise covariance C, or
D's own inner product)."""
import ast
import importlib
import pathlib

import numpy as np
import pytest

import pygeoinf0 as pg

PKG = pathlib.Path(pg.__file__).parent
ENGINE = ("decomposition", "closed_forms", "whitening", "partial", "factored", "cancellation")
CHOICES = ("D.whitened(C)", "the noise covariance C", "D")


def _definitions(name):
    found = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if any(isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name for n in tree.body):
            found.append(path.relative_to(PKG).as_posix())
    return found


def _imported_packages(rel):
    """The top-level pygeoinf0 packages a module imports (anywhere in it), as test_structure reads them."""
    from test_structure import _imported
    return {p[0] for p in _imported(rel, (PKG / rel).read_text())}


@pytest.mark.parametrize("name", ENGINE)
def test_the_engine_lives_in_solvers_dataspace(name):
    importlib.import_module(f"pygeoinf0.solvers.dataspace.{name}")
    with pytest.raises(ImportError):
        importlib.import_module(f"pygeoinf0.inference.{name}")


def test_data_decomposition_is_not_a_public_name():
    """Only pg.decompose (Q7): the class is the engine's."""
    assert "decompose" in pg.__all__ and callable(pg.decompose)
    for ns in (pg, pg.inference, pg.solvers):
        assert "DataDecomposition" not in getattr(ns, "__all__", ()), ns.__name__
        assert not hasattr(ns, "DataDecomposition"), ns.__name__
    assert _definitions("DataDecomposition") == ["solvers/dataspace/decomposition.py"]
    D = pg.Euclidean(3)
    G = pg.Matrix(np.arange(12.0).reshape(3, 4) + np.eye(3, 4), pg.Euclidean(4), D)
    dec = pg.decompose(G, D.standard_gaussian())
    assert type(dec).__module__ == "pygeoinf0.solvers.dataspace.decomposition"


def test_the_engine_sits_below_the_layers_it_serves():
    """The engine imports nothing of convex/, inference/, properties/ or meta/ (no metric read from a set, no target
    rule of the property families); its one crossing upward is the lazy covariances it returns (probability/
    lazy_covariance.py, recorded in tests/test_structure.py). The callers below inference/ reach it directly."""
    from test_structure import RANK
    assert RANK["optimisers"] < RANK["solvers"] < RANK["convex"] < RANK["probability"]
    for path in sorted((PKG / "solvers" / "dataspace").glob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        assert not _imported_packages(rel) & {"convex", "inference", "properties", "meta", "cells", "relations",
                                              "states"}, rel
    for rel in ("convex/feasible.py", "probability/scale.py", "solvers/choice.py"):
        assert "inference" not in _imported_packages(rel), rel


def _problem():
    rng = np.random.default_rng(7)
    M, D = pg.Euclidean(5), pg.Euclidean(3)
    G = pg.Matrix(rng.standard_normal((3, 5)), M, D, name="G")
    R = rng.standard_normal((3, 3))
    return M, D, G, R @ R.T + 3.0 * np.eye(3)


def _implicit(D, C):
    """Noise statements from which a metric would have to be inferred: nothing, a measure, sets."""
    from pygeoinf0.convex.algebra import Translate
    return {"none": None,
            "measure": pg.GaussianMeasure(D, pg.Matrix(C, D, D)),
            "ball": pg.NormBall(D, 1.0),
            "ellipsoid": pg.Ellipsoid(D, 1.0, C),
            "translated ball": Translate(pg.NormBall(D, 1.0), D(np.ones(3)))}


@pytest.mark.parametrize("kind", ["none", "measure", "ball", "ellipsoid", "translated ball"])
def test_the_engine_refuses_an_implicit_metric_naming_the_choices(kind):
    from pygeoinf0.solvers.dataspace import DataDecomposition, cached, decomposition_key
    M, D, G, C = _problem()
    noise = _implicit(D, C)[kind]
    for call in (lambda: DataDecomposition(G, noise), lambda: cached({}, G, noise),
                 lambda: decomposition_key(G, noise, None)):
        with pytest.raises(pg.Refused) as err:
            call()
        assert all(c in err.value.needs for c in CHOICES), err.value.needs


def test_explicit_metrics_give_the_same_decomposition():
    """D.whitened(C), the covariance C as a matrix of coordinates and as an operator on the plain D: one whitening, the
    generalised eigenvalues of (K, C); D itself: D's inner product (for a plain D the identity)."""
    from pygeoinf0.solvers.dataspace import DataDecomposition
    M, D, G, C = _problem()
    decs = [DataDecomposition(G, D.whitened(C)), DataDecomposition(G, C), DataDecomposition(G, pg.Matrix(C, D, D))]
    for d in decs[1:]:
        assert np.array_equal(d.lam, decs[0].lam)
    K = G.matrix() @ G.matrix().T
    L = np.linalg.cholesky(C)
    Li = np.linalg.inv(L)
    assert np.allclose(np.sort(decs[0].lam), np.linalg.eigvalsh(Li @ K @ Li.T), rtol=1e-12, atol=1e-12)
    plain = DataDecomposition(G, D)
    assert np.allclose(np.sort(plain.lam), np.linalg.eigvalsh(K), rtol=1e-12, atol=1e-12)


def test_pg_decompose_reads_the_noise_law_and_states_it():
    """The public route reads the noise law of a relation or a measure (Q1: the law lives in the relation) and hands
    the engine its covariance, bit for bit the explicit call."""
    from pygeoinf0.solvers.dataspace import DataDecomposition
    M, D, G, C = _problem()
    noise = pg.GaussianMeasure(D, pg.Matrix(C, D, D))
    a, b, c = pg.decompose(G, noise), pg.decompose(pg.Additive(G, noise)), DataDecomposition(G, pg.Matrix(C, D, D))
    assert np.array_equal(a.lam, c.lam) and np.array_equal(b.lam, c.lam)
    assert a.noise.key == c.noise.key


def test_the_engine_refuses_a_symbolic_target_without_a_rule():
    """The rule that turns a target kept symbolic into a factored family (its degree from the data) is the public
    routes' (pg.decompose, a pg.Problem), not the engine's: the engine alone refuses, naming pg.harmonic_target."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import test_harmonic_targets as ht
    from pygeoinf0.solvers.dataspace import DataDecomposition
    M = ht.space()
    G, _ = ht.kernels(M)
    D = G.codomain
    T = ht.rows(M, "a", [ht.bump((0.0, 0.0, 0.5), 0.2).measure()])
    y = D(np.random.default_rng(2).standard_normal(D.dim))
    C = np.full(D.dim, 0.01)
    with pytest.raises(pg.Refused, match="harmonic_target"):
        DataDecomposition(G, C).sola(T, 0.5, y)
    noise = pg.GaussianMeasure(D, pg.Diagonal(C, D))
    ruled = pg.decompose(G, noise).sola(T, 0.5, y)
    explicit = DataDecomposition(G, C).sola(pg.harmonic_target(T, ht.L), 0.5, y)
    assert np.array_equal(ruled.estimate, explicit.estimate)


def test_the_numerics_below_stay_below():
    """rowquad (diag(Phi V Phi^T)) serves the engine, the lazy covariances and the property families: it lives in the
    engine; is_symbolic (whether a target's parts are kept symbolic) is a capability of the realisations."""
    assert _definitions("rowquad") == ["solvers/dataspace/quadratic.py"]
    assert _definitions("is_symbolic") == ["realisations/series/symbolic_harmonics.py"]
    assert _definitions("LowRankCovariance") == ["probability/lazy_covariance.py"]
