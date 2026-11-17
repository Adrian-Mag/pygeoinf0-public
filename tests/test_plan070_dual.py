"""Plan 070 P4 (architecture review C5, D6; the author's amendment 1): convex/ holds convex analysis only, the dual
solvers of BGP live in optimisers/dual/, and convex/ calls them through that package's interface.

    convex/ modules            the listed ones only (sets, the set algebra, subspaces, the pointwise box, polytope
                               geometry, the feasible set as a state, the translation of sets into the solvers' forms
                               and the box prior's route); the list may only shrink
    scipy.optimize             imported only by optimisers/minimise.py (the one wrapper, with stated precision) and
                               optimisers/lp.py, and by the listed modules (special-function root finding, which stays
                               where it is, and files outside this phase); the list may only shrink
    one trust-region solve     and one bisection: defined only in optimisers/scalar.py, public
    the interface              convex/ imports from optimisers.dual only names in optimisers.dual.__all__ (the solver
                               interface; its modules reduction, exactness and exactness_reduce are called through
                               the module, so one name is looked up at call time), never a private name; optimisers/
                               imports nothing of convex/ (the layer order of tests/test_structure.py)
    SOLA's slab family         pg.inference.slab_family (moved from pg.convex: a clean break, no forwarder)
"""
import ast
import re

import pytest

from test_structure import _imported, modules

# convex/'s modules and their convex-analysis roles (amendment 1): a new module of convex/ needs a reviewed entry here
CONVEX_MODULES = {
    "convex/__init__.py": "the namespace",
    "convex/sets.py": "convex sets with support functions (balls, ellipsoids, slabs, boxes, polytopes, affine sets)",
    "convex/algebra.py": "the convex set algebra (images, Minkowski sums, translates, scalings, unions, intersections, "
                         "products)",
    "convex/subspaces.py": "linear and closed subspaces, spans, complements, projectors",
    "convex/pointwise.py": "the pointwise box and its support function",
    "convex/polytope.py": "polytope geometry (vertices, hulls, the inner polytope of a feasible image)",
    "convex/feasible.py": "the feasible set and its image as states, with the duality statement phi",
    "convex/forms.py": "the solver interface's translation of noise sets into the dual solvers' forms",
    "convex/feasible_box.py": "the feasible set's route for a pointwise-box prior (the forms of K and the solver calls)",
}

# modules that import scipy.optimize outside optimisers/; may only shrink
SCIPY_OPTIMIZE = {
    "realisations/spectra/bessel.py": "special-function root finding (zeros of Bessel functions): stays (D6)",
    "realisations/spectra/laplace_spectra.py": "special-function root finding (Laplace spectra): stays (D6)",
    "probability/chisquare.py": "special-function root finding and extrema (chi-square laws): stays (D6)",
}
OPTIMISERS_SCIPY = {"optimisers/minimise.py", "optimisers/lp.py"}

TRS = re.compile(r"(^|_)trs($|_)")
BISECT = re.compile(r"(^|_)bisect($|_|ion)")
TRUST_REGION = {"optimisers/scalar.py:trs", "optimisers/scalar.py:trs_columns", "optimisers/scalar.py:trs_floor"}
BISECTION = {"optimisers/scalar.py:bisect"}


def scipy_optimize(source):
    """Whether a module imports scipy.optimize (any form)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "scipy.optimize" or node.module.startswith("scipy.optimize."):
                return True
            if node.module == "scipy" and any(a.name == "optimize" for a in node.names):
                return True
        elif isinstance(node, ast.Import) and any(a.name.startswith("scipy.optimize") for a in node.names):
            return True
    return False


def defined(rel, source, pattern):
    """The functions of a module (any depth) whose names match pattern, as rel:name."""
    return {f"{rel}:{n.name}" for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and pattern.search(n.name)}


def dual_imports(rel, source):
    """The paths below pygeoinf0.optimisers.dual that a module imports (e.g. ['bound', 'Bound'] or ['Bound'])."""
    return [p[2:] for p in _imported(rel, source) if p[:2] == ["optimisers", "dual"]]


def test_convex_holds_convex_analysis_only():
    found = {rel for rel, _ in modules() if rel.startswith("convex/")}
    assert not found - set(CONVEX_MODULES), ("modules of convex/ without a convex-analysis role (dual solver numerics "
                                             f"belong in optimisers/dual/): {sorted(found - set(CONVEX_MODULES))}")
    assert not set(CONVEX_MODULES) - found, f"gone, remove from CONVEX_MODULES: {sorted(set(CONVEX_MODULES) - found)}"


def test_scipy_optimize_is_called_through_the_one_wrapper():
    found = {rel for rel, path in modules() if scipy_optimize(open(path).read())}
    outside = found - OPTIMISERS_SCIPY
    assert not outside - set(SCIPY_OPTIMIZE), ("scipy.optimize outside the wrapper (optimisers/minimise.py): "
                                               f"{sorted(outside - set(SCIPY_OPTIMIZE))}")
    assert not set(SCIPY_OPTIMIZE) - outside, f"fixed, remove from SCIPY_OPTIMIZE: {sorted(set(SCIPY_OPTIMIZE) - outside)}"
    assert not any(rel.startswith(("convex/", "inference/", "optimisers/dual/")) for rel in outside)


def test_one_trust_region_solve_and_one_bisection():
    trs, bis = set(), set()
    for rel, path in modules():
        src = open(path).read()
        trs |= defined(rel, src, TRS)
        bis |= defined(rel, src, BISECT)
    assert trs == TRUST_REGION, f"trust-region solves defined outside optimisers/scalar.py, or renamed: {sorted(trs)}"
    assert bis == BISECTION, f"bisections defined outside optimisers/scalar.py, or renamed: {sorted(bis)}"


def private_members(source, names):
    """Private attributes read through the given module names (n._x for n in names)."""
    return {f"{n.value.id}.{n.attr}" for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id in names and n.attr.startswith("_")
            and not n.attr.startswith("__")}


def test_convex_calls_optimisers_dual_through_its_interface():
    import types
    from pygeoinf0.optimisers import dual
    interface = set(dual.__all__)
    assert all(hasattr(dual, n) for n in interface)
    mods = {n for n in interface if isinstance(getattr(dual, n), types.ModuleType)}
    assert mods == {"reduction", "exactness", "exactness_reduce"}
    bad = set()
    for rel, path in modules():
        if rel.startswith("convex/"):
            src = open(path).read()
            bad |= {(rel, ".".join(p)) for p in dual_imports(rel, src) if len(p) != 1 or p[0] not in interface}
            bad |= {(rel, m) for m in private_members(src, mods)}
    assert not bad, f"convex/ reaches optimisers.dual outside its interface (optimisers.dual.__all__): {sorted(bad)}"


def test_optimisers_import_nothing_of_convex():
    bad = {(rel, p[0]) for rel, path in modules() if rel.startswith("optimisers/")
           for p in _imported(rel, open(path).read()) if p[0] in ("convex", "inference", "probability", "states")
           and not (rel == "optimisers/dual/exactness.py" and p[0] == "probability")}
    assert not bad, f"optimisers/ imports from below its layer: {sorted(bad)}"


def test_the_wrapper_states_its_precision():
    from pygeoinf0.optimisers import minimise as W
    with pytest.raises(TypeError):
        W.minimise(lambda x: float(x @ x), [1.0, 2.0], "BFGS")              # no precision: refused by signature
    res = W.minimise(lambda x: float((x - 1.0) @ (x - 1.0)), [3.0, -2.0], "BFGS", dict(gtol=1e-12))
    assert res.precision == {"gtol": 1e-12} and res.method == "BFGS" and abs(res.x - 1.0).max() < 1e-6
    res = W.minimise_scalar(lambda t: (t - 0.3) ** 2, (0.0, 1.0), xatol=1e-12)
    assert res.precision == {"xatol": 1e-12} and abs(res.x - 0.3) < 1e-9
    r = W.root(lambda t: t ** 3 - 2.0, 0.0, 2.0, xtol=1e-15, rtol=1e-15, maxiter=100)
    assert abs(r - 2.0 ** (1 / 3)) < 1e-14


def test_slab_family_is_sola_s_and_lives_in_inference():
    import numpy as np
    import pygeoinf0 as pg
    assert not hasattr(pg.convex, "slab_family")
    D = pg.Euclidean(3)
    S = pg.inference.slab_family(D, np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]), sigma=[0.5, 1.0], mass=2.0)
    assert [s.half_width for s in S] == [1.0, 2.0] and isinstance(S[0], pg.Slab)


@pytest.mark.parametrize("source, expected", [
    ("from scipy.optimize import minimize\n", True),
    ("def f():\n    import scipy.optimize as so\n", True),
    ("from scipy import optimize\n", True),
    ("from scipy.optimize._minimize import minimize\n", True),
    ("from scipy import linalg\nimport scipy.sparse\n", False),
])
def test_the_scipy_detector_catches_its_planted_defects(source, expected):
    assert scipy_optimize(source) is expected


@pytest.mark.parametrize("source, trs, bis", [
    ("def _trs(a, b, r):\n    pass\n", {"x.py:_trs"}, set()),
    ("class A:\n    def trs_many(self):\n        pass\n", {"x.py:trs_many"}, set()),
    ("def f():\n    def _bisect(g, lo, hi):\n        pass\n", set(), {"x.py:_bisect"}),
    ("def bisection(g):\n    pass\ndef strs(x):\n    pass\ndef mtrs():\n    pass\n", set(), {"x.py:bisection"}),
])
def test_the_definition_detector_catches_its_planted_defects(source, trs, bis):
    assert defined("x.py", source, TRS) == trs and defined("x.py", source, BISECT) == bis


@pytest.mark.parametrize("source, paths", [
    ("from ..optimisers.dual import Bound, reduce\n", [["Bound"], ["reduce"]]),
    ("def f():\n    from ..optimisers.dual.nonsmooth import _maxlin\n", [["nonsmooth", "_maxlin"]]),
    ("from ..optimisers import dual\nfrom ..optimisers.scalar import trs\n", [[]]),
])
def test_the_interface_detector_catches_its_planted_defects(source, paths):
    """Planted controls: a name of the interface is one element; a module path, a private name or the package itself
    (whose attributes the check cannot see) is not, and test_convex_calls_optimisers_dual_through_its_interface
    refuses it."""
    assert dual_imports("convex/planted.py", source) == paths


def test_the_private_member_detector_catches_its_planted_defects():
    src = "def f():\n    reduction._svd(B)\n    reduction.reduce(J, 1)\n    exactness.__name__\n    other._x\n"
    assert private_members(src, {"reduction", "exactness"}) == {"reduction._svd"}
