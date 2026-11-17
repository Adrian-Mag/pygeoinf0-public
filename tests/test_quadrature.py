"""The one quadrature module (plan 069 P5; architecture review 2026-11-13, B13): pygeoinf0.quadrature holds the rules,
each stating its exactness, and the package calls no Gauss-Legendre generator of its own. Hypothesis checks the same
statements on random polynomials in tests/test_laws.py; these deterministic cases keep them in the suite without it."""
import ast
import math
import os

import numpy as np
import pytest
from numpy.polynomial import legendre as npleg
from scipy.special import eval_jacobi

import pygeoinf0 as pg
from pygeoinf0 import quadrature as Q
from pygeoinf0.core.precision import TOL

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pygeoinf0")
GENERATORS = {"leggauss", "roots_legendre"}


def test_the_package_calls_no_gauss_legendre_generator_but_the_module():
    """numpy's leggauss (and SciPy's roots_legendre) bypass the refined weights: only quadrature.py may call them."""
    found = []
    for root, _, files in os.walk(PKG):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, PKG).replace(os.sep, "/")
            if rel == "quadrature.py":
                continue
            for node in ast.walk(ast.parse(open(path).read())):
                name = getattr(node, "attr", None) if isinstance(node, ast.Attribute) else getattr(node, "id", None)
                if name in GENERATORS:
                    found.append(f"{rel}:{node.lineno}")
    assert not found, f"direct Gauss-Legendre generators (use pygeoinf0.quadrature): {found}"


def test_every_rule_states_its_exactness():
    assert set(Q.RULES) == {"gauss", "gauss01", "panels", "graded_unit", "both_ends", "trapezoid", "simplex"}
    for name in Q.RULES:
        rule = getattr(Q, name)
        assert callable(rule.exact_degree) and "exact" in rule.exactness, name
    assert Q.gauss.exact_degree(5) == 9 and Q.trapezoid.exact_degree(100) == 1 and Q.simplex.exact_degree(3) == 5


@pytest.mark.parametrize("n", [1, 2, 7, 40, 128])
def test_gauss_reaches_its_degree_and_no_more(n):
    t, w = Q.gauss(n)
    c = np.cos(np.arange(2 * n) + 1.0)                                   # a polynomial of degree 2n - 1
    assert abs(w @ npleg.legval(t, c) - 2 * c[0]) <= TOL * 2 * np.abs(c).sum()
    assert abs(w @ npleg.legval(t, np.r_[np.zeros(2 * n), 1.0])) > TOL * 2 / math.sqrt(4 * n + 1)     # P_2n
    t01, w01 = Q.gauss01(n)
    assert np.array_equal(t01, 0.5 * (t + 1)) and np.array_equal(w01, 0.5 * w)


def test_refined_weights_hold_the_gram_of_orthonormal_polynomials():
    """The refinement (Newton on P_n, weights from P_n'): the Gram of orthonormal Legendre polynomials is the identity to
    5e-14 at 128 and 1500 nodes; numpy's leggauss is 8e-13 and 1.2e-11 from it there (measured 2026-11-14)."""
    for n in (128, 1500):
        t, w = Q.gauss(n)
        k = min(n, 201)
        V = npleg.legvander(t, k - 1) * np.sqrt((2 * np.arange(k) + 1) / 2)
        assert np.abs((V * w[:, None]).T @ V - np.eye(k)).max() < 5e-14, n
    t, w = Q.gauss(128)
    assert not t.flags.writeable and not w.flags.writeable              # cached: shared, read-only


def test_composite_and_graded_rules_are_exact_on_their_panels():
    edges = np.r_[0.0, 0.1, 0.5, 1.7]
    x, w = Q.panels(edges, 4)
    for lo, hi in zip(edges[:-1], edges[1:]):                           # x^7 on each panel, 0 elsewhere
        inside = (x > lo) & (x < hi)
        assert w[inside] @ x[inside] ** 7 == pytest.approx((hi ** 8 - lo ** 8) / 8, rel=1e-14)
    x, w, h = Q.graded_unit(6, 3, 1e-10)
    e = Q.graded_edges(3, 1e-10)
    assert e[0] == h and e[-1] == 1.0 and np.all((x > h) & (x < 1))
    assert w.sum() == pytest.approx(1 - h, rel=1e-15) and w @ x ** 11 == pytest.approx((1 - h ** 12) / 12, rel=1e-14)
    x, w = Q.both_ends(6, 4, 1e-8)
    e = Q.both_ends_edges(4, 1e-8)
    assert e[0] == 0.0 and e[-1] == 1.0 and np.allclose(e, 1 - e[::-1], rtol=0, atol=1e-16)
    assert w.sum() == pytest.approx(1.0, rel=1e-15) and w @ x ** 11 == pytest.approx(1 / 12, rel=1e-14)


def test_simplex_rules_reach_their_total_degree_and_no_more():
    for k, n in ((1, 5), (2, 4), (3, 3)):
        X, W = Q.simplex(k, n)
        assert W.sum() == pytest.approx(1 / math.factorial(k), rel=1e-14)
        a = (2 * n - 1,) + (0,) * (k - 1)
        exact = math.factorial(2 * n - 1) / math.factorial(2 * n - 1 + k)
        assert W @ np.prod(X ** np.array(a), axis=1) == pytest.approx(exact, rel=1e-13)
        assert abs(W @ eval_jacobi(2 * n, k - 1, 0, 2 * X[:, 0] - 1)) > 1e-6
    X, W = Q.simplex(0, 4)
    assert X.shape == (1, 0) and W.tolist() == [1.0]
    tri = pg.Simplex((0, 0), (1, 0), (0, 1))
    assert np.array_equal(Q.rule(tri, 4)[1], Q.simplex(2, 4)[1])     # the domain's rule is the module's


def test_trapezoid_is_exact_for_lines_only_and_is_the_realisation_options_rule():
    x, w = Q.trapezoid(11, 0.0, 2.0)
    assert w @ (3 * x - 1) == pytest.approx(4.0, rel=1e-15)
    h = 0.2
    assert w @ x ** 2 - 8 / 3 == pytest.approx(2.0 * h ** 2 * 2 / 12, rel=1e-12)    # (b - a) h^2 f'' / 12
    xo, wo = pg.realisations.Trapezoid(11).nodes(0.0, 2.0)
    assert np.array_equal(x, xo) and np.array_equal(w, wo)
    with pytest.raises(ValueError, match="2 nodes"):
        Q.trapezoid(1, 0.0, 1.0)


def test_gauss_left_the_legendre_realisation():
    """A clean break (Q5): the rule moved, with no forwarder in realisations/series/legendre.py."""
    from pygeoinf0.realisations.series import legendre
    assert not hasattr(legendre, "gauss")
