"""Plan 070 P6 (review E2, E4): pg.* is grouped by mathematical role, every root is a pg.* name, and the periphery
(optimisers, port, plotting, catalogue) is reached only through its subpackage."""
import importlib
import inspect

import pygeoinf0 as pg

# the review's section 2 "Namespace" order (subsets for "convex sets", amendment 1; properties after inference, plan
# 070 P6), then the names of no mathematical group: claims and refusals, the numerics, and the periphery's subpackages
ORDER = ("sets and regions", "measures", "spaces and duals", "operators", "subsets", "probability", "states",
         "relations", "table", "inference and methods", "properties", "meta", "claims and refusals", "numerics", "periphery")
# each root and the group it opens
ROOTS = {"Set": "sets and regions", "Region": "sets and regions", "Domain": "sets and regions",
         "Measure": "measures",
         "VectorSpace": "spaces and duals", "NormedSpace": "spaces and duals", "BanachSpace": "spaces and duals",
         "HilbertSpace": "spaces and duals", "DualSpace": "spaces and duals", "Functional": "spaces and duals",
         "LinearOperator": "operators",
         "Subset": "subsets", "ConvexSet": "subsets",
         "ProbabilityMeasure": "probability", "Covariance": "probability",
         "State": "states",
         "Relation": "relations", "MarkovKernel": "relations",
         "Problem": "inference and methods"}
PERIPHERY = ("optimisers", "port", "plotting", "catalogue")
# the names PLI's code uses (the brief of plan 070 P6): each stays reachable where it is
PLI = ("Additive", "Ball", "CapAverages", "Diagonal", "DirectSum", "Ellipsoid", "Euclidean", "FeasibleSet", "Functionals",
       "GaussianMeasure", "Identity", "Interval", "L2", "LebesgueMeasure", "Matrix", "PointwiseBox", "Sphere", "Truth",
       "decompose", "update", "HarmonicConvention", "PiecewiseSpace", "Functional")


def test_the_groups_are_in_the_stated_order():
    assert tuple(title for title, _ in pg.GROUPS) == ORDER


def test_all_is_the_groups_in_order():
    flat = [n for _, names in pg.GROUPS for n in names]
    assert len(flat) == len(set(flat)), "a name is in two groups"
    assert pg.__all__ == flat


def test_every_name_is_bound():
    assert not [n for n in pg.__all__ if not hasattr(pg, n)]


def test_every_root_is_exported_in_its_group():
    group = {n: title for title, names in pg.GROUPS for n in names}
    assert {r: group.get(r) for r in ROOTS} == ROOTS


def test_each_group_starts_with_its_roots():
    """A newcomer reads a group from its root: the roots come first."""
    for title, names in pg.GROUPS:
        roots = [r for r, g in ROOTS.items() if g == title]
        assert set(names[:len(roots)]) == set(roots), title


def test_the_periphery_is_reached_through_its_subpackage_only():
    """No pg.* name is defined in optimisers, port, plotting or catalogue, and none of their names is top-level."""
    assert dict(pg.GROUPS)["periphery"] == PERIPHERY
    homes = {}
    for n in pg.__all__:
        obj = getattr(pg, n)
        if inspect.ismodule(obj):
            assert n in PERIPHERY or not any(obj.__name__.startswith(f"pygeoinf0.{p}") for p in PERIPHERY), n
            continue
        homes[n] = getattr(obj, "__module__", "") or ""
    bad = sorted(n for n, m in homes.items() if any(m.startswith(f"pygeoinf0.{p}") for p in PERIPHERY))
    assert not bad, f"top-level names from the periphery: {bad}"
    for p in PERIPHERY:
        mod = importlib.import_module(f"pygeoinf0.{p}")
        assert getattr(pg, p) is mod, f"pg.{p} is not reachable"
        public = getattr(mod, "__all__", None) or [n for n in vars(mod) if not n.startswith("_")]
        top = sorted(n for n in set(public) & set(pg.__all__) if getattr(pg, n) is getattr(mod, n))
        assert not top, f"pygeoinf0.{p} names in pg.__all__: {top}"


def test_the_moved_names_are_gone_from_the_top_level_and_live_in_their_subpackage():
    """A clean break (Q5): no top-level alias; the subpackage path works."""
    moved = {"slices": "plotting", "gauss_newton": "optimisers", "lbfgs": "optimisers", "Coordinates": "port",
             "External": "port", "taylor_test": "port", "complex_linearity": "port", "generate": "port",
             "plugins": "port"}
    for n, p in moved.items():
        assert not hasattr(pg, n), f"pg.{n} is still top-level"
        assert hasattr(getattr(pg, p), n), f"pg.{p}.{n} is missing"


def test_aliases_of_one_concept_are_gone():
    """LebesgueSpace = Lp and Rectangle = Box were two names of one object (review E4)."""
    import pygeoinf0.domains as domains
    assert not hasattr(pg, "LebesgueSpace") and not hasattr(pg, "Rectangle") and not hasattr(domains, "Rectangle")


def test_the_names_pli_uses_stay():
    assert not [n for n in PLI if n not in pg.__all__]
    assert callable(pg.plotting.sphere_map) and callable(pg.plotting.slices)
    assert callable(pg.inference.ocsola) and callable(pg.inference.slab_family)
    assert pg.meta.__name__ == "pygeoinf0.meta"


def test_the_methods_are_pg_names():
    """The methods are the public routes (Q7): pg.sola is pg.inference.sola (review E4)."""
    for name in pg.inference.METHODS:
        assert getattr(pg, name) is getattr(pg.inference, name) and name in pg.__all__


def test_api_record_has_the_same_sections():
    from pygeoinf0.api_record import api_record
    text = api_record()
    heads = [line[4:].split(" (")[0] for line in text.splitlines() if line.startswith("### ")]
    assert heads[:len(ORDER)] == [f"{i}. {t}" for i, t in enumerate(ORDER, 1)]
