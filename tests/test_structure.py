"""Structural rules of the package, checked mechanically (see the mission DESIGN.md, architecture rules):
module sizes, layering (core, sets and domains import no realisation at import time), exceptions only from core/errors.py,
the realisation chosen only in realisations/registry.py (the engines of energies included: an engine is the
realisation of the energies, so the exact, H-matrix and spectral engines are constructed there too), public names
documented, and (plan 065 P6) the import layers of the architecture review with no private name crossing a package, each
against a list of today's violations that may only shrink."""
import ast
import os

import pytest

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pygeoinf0")
# module size is a soft rule (the author, 2026-11-13, plan 063): split a module by responsibility, not by count; modules
# over SOFT_LINES are reported as a warning, never a failure
SOFT_LINES = 500
# import-time leaks that predate the rules (documented); no new ones (plan 067 P5: the measures' one leak, the Legendre
# fit of a density on an interval, is now imported where it is used)
LEAKS = set()


def modules():
    for root, _, files in os.walk(PKG):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                yield os.path.relpath(path, PKG).replace(os.sep, "/"), path


def test_module_size():
    sizes = {rel: sum(1 for _ in open(path)) for rel, path in modules()}
    large = sorted((n, rel) for rel, n in sizes.items() if n > SOFT_LINES)
    if large:
        import warnings
        warnings.warn(f"{len(large)} modules over {SOFT_LINES} lines (soft rule: split by responsibility where it helps): "
                      + ", ".join(f"{rel} {n}" for n, rel in reversed(large)), UserWarning)


@pytest.mark.parametrize("rel, path", [m for m in modules() if m[0].startswith(("core/", "sets/", "measures/",
                                                                                  "domains/", "spaces/"))])
def test_layering(rel, path):
    tree = ast.parse(open(path).read())
    for node in tree.body:                                        # top level only: lazy imports inside functions are fine
        if isinstance(node, ast.ImportFrom) and node.module and node.level >= 2:
            top = node.module.split(".")[0]
            if top in ("realisations", "operators", "probability", "plotting"):
                assert (rel, top) in LEAKS, f"{rel} imports {node.module} at import time"


@pytest.mark.parametrize("rel, path", list(modules()))
def test_exceptions_come_from_errors(rel, path):
    if rel == "core/errors.py":
        return
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
            assert not any(b.endswith(("Error", "Exception", "Warning")) for b in bases), \
                f"{rel}: exception class {node.name} outside core/errors.py"


# the one documented exception: product families realise their one-dimensional factors with the interval engine
INTERNAL_ENGINES = {("realisations/series/separated.py", "IntervalBackend")}


def test_realisations_are_chosen_in_one_place():
    names = ("IntervalBackend", "TensorBackend", "SphereBackend", "RKHSBackend", "SphereSobolevBackend",
             "SupIntervalBackend", "SumBackend", "ExactEngine", "HEngine", "SpectralEngine", "Energies")
    for rel, path in modules():
        if rel == "realisations/registry.py":
            continue
        for node in ast.walk(ast.parse(open(path).read())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in names:
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if name == "SphereBackend" and rel == "realisations/series/sphere_sobolev.py":
                    continue                                       # super().__init__ chains do not construct
                assert (rel, name) in INTERNAL_ENGINES, f"{rel} constructs {name}: only realisations/registry.py may"


DOMAIN_TYPES = {"Interval", "Box", "Disk", "Cylinder", "Ball", "Shell", "LayeredBall", "Circle", "Sphere", "Torus", "Flat",
                "Segment", "Patch", "Polyline", "Arc", "Domain"}
# modules that may test the type of a set: the registry (dispatch on the domain of a space) and the parsers of measures
# (the type of a measure's support decides its atoms); everywhere else, behaviour comes from the backend or geometry
TYPE_TESTS = {"realisations/registry.py", "realisations/rkhs/atoms.py", "realisations/series/sphere_sobolev.py",
              "realisations/backends.py"}


@pytest.mark.parametrize("rel, path", [m for m in modules() if m[0].startswith(("realisations/", "operators/",
                                                                                   "probability/", "spaces/sobolev"))])
def test_no_dispatch_on_domain_types_outside_the_registry(rel, path):
    if rel in TYPE_TESTS:
        return
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "isinstance" and len(node.args) == 2:
            t = node.args[1]
            names = {getattr(e, "id", getattr(e, "attr", "")) for e in (t.elts if isinstance(t, ast.Tuple) else [t])}
            assert not names & DOMAIN_TYPES, f"{rel}: isinstance on {names & DOMAIN_TYPES} (use registry.flat_geometry)"


# ---------------------------------------------------------------- import layers (plan 065 P6)
# The layer order of the architecture review 2026-11-13, section 2 (docs/research/findings/2026-11-13-architecture-
# review.md), with the author's amendments at its end applied: no layer is named after convexity (today's convex/
# package sits in the layer "subsets of a space", the set states), and the state x relation table is not an organising
# layer (cells/ sits where it must as a dispatch mechanism: below the states and relations it dispatches on). Each
# package imports only packages of its own rank or above it in this list; imports inside functions count (a lazy
# import is still a dependency: the review's D1 counts them). Today's packages, until Plans B-E move them:
LAYERS = [
    ("settings", "execution"),          # the runtime: switches (imports nothing, read by every layer) and the worker
                                        # pools beside them (plan 069 P4: execution.py moved out of realisations/)
    ("quadrature",),                    # numerical column (plan 069 P5): the quadrature rules, below every mathematical
                                        # layer they serve; imports nothing of the package but settings
    ("core", "sets", "measures", "domains", "spaces"),  # L0-L3: core holds logic, claims, errors, precision, numbers and
                                        # kinds (and until plan 069 the numerics fits, sampling, adaptive, lazy,
                                        # tabulated, estimable; sturm_liouville went to operators/ in plan 068 P3); sets is L1 (the
                                        # Set root, plan 067 P2), measures L2 (plan 067 P5), domains the regions with a
                                        # chart, spaces L3 (plan 068 P1: the spaces out of core). One block still: a rank
                                        # of its own for spaces would add three imports from below (core/fits.py and
                                        # core/lazy.py build Functions, until plan 069 moves them to the numerical
                                        # column; measures/weights.as_weight recognises a Function as a weight)
    ("realisations",),                  # numerical column serving the spaces
    ("operators",),                     # L4 maps between spaces
    ("optimisers",),                    # numerical column: certified scalar searches, LP, dual solvers (they run in a
                                        # data-space decomposition's eigenbasis, which they take as an argument)
    ("solvers",),                       # numerical column: A x = y and the data-space engine (solvers/dataspace/, plan
                                        # 070 P3), below the layers it serves (convex/, probability/, inference/)
    ("convex",),                        # L5a subsets of a space (the set states; convex analysis is what is computable)
    ("probability",),                   # L5b probability measures on spaces
    ("states",),                        # L6a states of knowledge
    ("relations",),                     # L6b relations of three kinds
    ("cells",),                         # the push/update dispatch (a mechanism, not the organising object)
    ("properties",),                    # L8 the property space P and the model-property relation (plan 070 P1, Q10):
                                        # above operators, probability, states and relations, below inference
    ("inference",),                     # L8 the seven elements and the methods
    ("meta",),                          # L9 outside the seven elements
    ("port", "plotting", "catalogue", "functions"),     # periphery, at the edge: nothing above imports it
]
RANK = {p: i for i, ps in enumerate(LAYERS) for p in ps}
NAMESPACE = {"__init__", "api_record"}  # pg.* and its record import everything by design
# today's violations: this list may only shrink (a fixed one must be removed; a new one fails)
LAYER_VIOLATIONS = {
    ('convex/sets.py', 'plotting'),
    ('spaces/complex.py', 'operators'),
    ('spaces/complex.py', 'probability'), ('spaces/complex.py', 'realisations'), ('spaces/constructions.py', 'operators'),
    ('spaces/continuous.py', 'operators'), ('spaces/continuous.py', 'realisations'), ('spaces/direct_sum.py', 'operators'),
    ('spaces/direct_sum.py', 'probability'), ('spaces/direct_sum.py', 'realisations'), ('spaces/elements.py', 'plotting'),
    ('measures/lebesgue.py', 'realisations'), ('spaces/metrics.py', 'operators'),
    ('core/relation.py', 'relations'), ('spaces/representers.py', 'realisations'),
    ('spaces/sobolev.py', 'operators'),
    ('spaces/base.py', 'operators'), ('spaces/base.py', 'probability'), ('spaces/base.py', 'realisations'),
    ('core/tabulated.py', 'realisations'),
    ('domains/curves.py', 'realisations'),
    ('inference/methods.py', 'plotting'),
    ('operators/inverses.py', 'solvers'), ('operators/linear.py', 'cells'),
    ('operators/linear.py', 'relations'), ('operators/unbounded.py', 'relations'), ('operators/plain_gram.py', 'solvers'),
    ('probability/gaussian.py', 'cells'),
    ('probability/gaussian.py', 'relations'),
    ('realisations/spectra/laplacian_covariance.py', 'probability'),
    ('relations/kernels.py', 'cells'), ('relations/linear.py', 'cells'), ('relations/maps.py', 'cells'),
    ('solvers/dataspace/factored.py', 'probability'),   # plan 070 P3: the lazy covariances its posteriors return
}

# a private name (a leading underscore, not a dunder) imported across a package boundary: may only shrink
PRIVATE_CROSSINGS = set()        # empty since plan 070 P7 (the helpers made public in their homes)


def _package(rel):
    return rel.split("/")[0] if "/" in rel else rel[:-3]


def _imported(rel, source):
    """The pygeoinf0 names a module imports, as dotted paths below pygeoinf0 (relative imports resolved), anywhere in
    the module (top level and inside functions)."""
    parts = rel[:-3].split("/")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not node.module or node.module.split(".")[0] != "pygeoinf0":
                    continue
                mod = node.module.split(".")[1:]
            else:
                mod = parts[:len(parts) - node.level] + (node.module.split(".") if node.module else [])
            for a in node.names:
                yield mod + [a.name]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "pygeoinf0" and "." in a.name:
                    yield a.name.split(".")[1:]


def import_violations(rel, source):
    """(layer, private): the packages rel imports from below its layer, and the private names it imports from another
    package."""
    src = _package(rel)
    layer, private = set(), set()
    if src in NAMESPACE:
        return layer, private
    for path in _imported(rel, source):
        tgt = path[0]
        if tgt == src or tgt not in RANK:
            continue
        if RANK[tgt] > RANK[src]:
            layer.add((rel, tgt))
        if any(p.startswith("_") and not p.startswith("__") for p in path[1:]):
            private.add((rel, ".".join(path)))
    return layer, private


def _all_violations():
    layer, private = set(), set()
    for rel, path in modules():
        a, b = import_violations(rel, open(path).read())
        layer |= a
        private |= b
    return layer, private


def test_every_package_has_a_layer():
    found = {_package(rel) for rel, _ in modules()} - NAMESPACE
    assert not found - set(RANK), f"packages without a place in LAYERS (place them): {sorted(found - set(RANK))}"


def test_imports_follow_the_layer_order():
    layer, _ = _all_violations()
    assert not layer - LAYER_VIOLATIONS, f"new imports from below the importer's layer: {sorted(layer - LAYER_VIOLATIONS)}"
    assert not LAYER_VIOLATIONS - layer, f"fixed, remove from LAYER_VIOLATIONS: {sorted(LAYER_VIOLATIONS - layer)}"


def test_no_private_name_crosses_a_package():
    _, private = _all_violations()
    assert not private - PRIVATE_CROSSINGS, f"new private names across packages: {sorted(private - PRIVATE_CROSSINGS)}"
    assert not PRIVATE_CROSSINGS - private, f"fixed, remove from PRIVATE_CROSSINGS: {sorted(PRIVATE_CROSSINGS - private)}"


@pytest.mark.parametrize("rel, source, layer, private", [
    ("core/planted.py", "def f():\n    from ..inference import methods\n", {("core/planted.py", "inference")}, set()),
    ("operators/planted.py", "from pygeoinf0.cells.forward import push\n", {("operators/planted.py", "cells")}, set()),
    ("convex/planted.py", "from ..probability import gaussian\n", {("convex/planted.py", "probability")}, set()),
    ("inference/planted.py", "from ..convex.feasible import _ball_shape\n", set(),
     {("inference/planted.py", "convex.feasible._ball_shape")}),
    ("core/planted.py", "import pygeoinf0.plotting._private\n", {("core/planted.py", "plotting")},
     {("core/planted.py", "plotting._private")}),
    ("inference/planted.py", "from ..convex import feasible\nfrom . import _own\nfrom .methods import _x\n", set(), set()),
    ("quadrature.py", "def f():\n    from .core.claims import Claim\n", {("quadrature.py", "core")}, set()),
])
def test_the_import_checks_catch_their_planted_defects(rel, source, layer, private):
    """Planted-defect controls: a module that imports from below its layer, or a private name of another package, is
    reported (including a lazy import inside a function); importing upwards and private names of one's own package are
    not."""
    assert import_violations(rel, source) == (layer, private)


# ------------------------------------------------------------------------- names (plan 068 P7, review E1 and E2)
# "restrict" in a public name of the package means u|_D (a function restricted to a subdomain, or a realisation of it),
# mu|_D (a measure restricted to a set) or R|_S (a relation restricted to a set of its domain), nothing else. The other
# meanings were renamed: R.select(rows) (the relation of some data; a kernel's marginal), CoordinateProjection,
# RestrictionSpace, C.compressed(P) (P C P), D.subpiece(lo, hi) and X.restrictions(P). The remaining uses, each with its
# meaning; the list may only shrink.
RESTRICTIONS = ("u|_D", "mu|_D", "R|_S")
RESTRICT_NAMES = {
    "measures/restrict.py": "mu|_D",
    "measures/restrict.py:restrict": "mu|_D",
    "measures/base.py:Measure.restricted": "mu|_D",
    "measures/weights.py:RestrictedWeight": "mu|_D",                 # the weight of a density measure restricted to D
    "operators/pieces.py:Restrict": "u|_D",
    "operators/pieces.py:Restrictions": "u|_D",                      # u -> (u|_(Omega_k))_k
    "operators/pieces.py:restrictions": "u|_D",
    "operators/pieces.py:restrict_element": "u|_D",
    "operators/pieces_sobolev.py:SobolevRestrictions": "u|_D",
    "operators/pieces_sobolev.py:restrict_element": "u|_D",
    "spaces/base.py:_FunctionSpace.restrict_to": "u|_D",
    "spaces/base.py:_FunctionSpace.restrictions": "u|_D",
    "spaces/piecewise.py:PiecewiseSpace.restriction": "u|_D",       # R: X_field(Omega) -> its summand on a piece
    "spaces/constructions.py:RestrictionSpace": "u|_D",             # R^s(D) = {U|_D : U in H^s(R^d)}, quotient norm
    "spaces/constructions.py:interval_restriction_form": "u|_D",    # the form of that space on an interval
    "realisations/backends.py:Backend.restricted": "u|_D",          # a realisation of u cut to a subdomain
    "realisations/backends.py:IntervalBackend.restricted": "u|_D",
    "realisations/piecewise_constant.py:CellsBackend.restricted": "u|_D",
    "realisations/series/radial_harmonic.py:RadialHarmonicsBackend.restricted": "u|_D",
    "realisations/series/cuts.py:restricted": "u|_D",
    "realisations/series/legendre.py:restrict": "u|_D",                    # a Legendre series re-expressed on a sub-interval
    "realisations/series/local.py:restrict": "u|_D",                       # a tensor series re-expressed on a sub-box
}


def names_with(word, rel, source):
    """The module path and the public top-level and method names of one module whose name contains word (any case)."""
    out = {rel} if word in rel.rsplit("/", 1)[-1].lower() else set()
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if not node.name.startswith("_") and word in node.name.lower():
            out.add(f"{rel}:{node.name}")
        if isinstance(node, ast.ClassDef):
            out |= {f"{rel}:{node.name}.{m.name}" for m in node.body if isinstance(m, ast.FunctionDef)
                    and not m.name.startswith("_") and word in m.name.lower()}
    return out


def public_names(word):
    return set().union(*(names_with(word, rel, open(path).read()) for rel, path in modules()))


def test_restrict_means_a_restriction():
    assert set(RESTRICT_NAMES.values()) <= set(RESTRICTIONS)
    found = public_names("restrict")
    assert not found - set(RESTRICT_NAMES), ("'restrict' is kept for u|_D, mu|_D and R|_S; rename these or add them "
                                            f"with their meaning: {sorted(found - set(RESTRICT_NAMES))}")
    assert not set(RESTRICT_NAMES) - found, f"gone, remove from RESTRICT_NAMES: {sorted(set(RESTRICT_NAMES) - found)}"
    import pygeoinf0 as pg
    exported = {n for n in dir(pg) if "restrict" in n.lower()}
    assert exported <= {"Restrict", "RestrictionSpace"}, sorted(exported)


def test_a_markov_kernel_is_called_one():
    """E2: "Kernel" also means integral and reproducing kernels and ker; the measure-kind relation is a MarkovKernel."""
    import pygeoinf0 as pg
    from pygeoinf0 import relations
    assert not hasattr(pg, "Kernel") and not hasattr(relations, "Kernel") and not hasattr(relations, "AdditiveKernel")
    assert pg.MarkovKernel is relations.MarkovKernel
    assert issubclass(relations.AdditiveMarkovKernel, pg.MarkovKernel)
    assert {n for n in dir(pg) if n.endswith("Kernel")} <= {"MarkovKernel", "AdditiveMarkovKernel", "PosteriorKernel"}


def test_the_names_check_catches_its_planted_defect():
    planted = "class Restriction:\n    def restrict(self, rows):\n        pass\n    def _restrict(self):\n        pass\n"
    found = names_with("restrict", "relations/restriction.py", planted)
    assert found == {"relations/restriction.py", "relations/restriction.py:Restriction",
                     "relations/restriction.py:Restriction.restrict"}
    assert not found & set(RESTRICT_NAMES)


# ------------------------------------------------- realisation types (plan 069 P3, review C3): the capability protocol
# Outside realisations/ nothing tests the type of a realisation (isinstance or issubclass on a class of realisations/,
# or a comparison with its class name): it asks the realisation for a capability (realisations/registry.py:
# provides, require, capability; CAPABILITIES lists them). Today's remaining type tests, each with its reason; the list
# may only shrink.
REALISATION_TYPE_TESTS = set()       # empty since plan 069 P4 (HarmonicConvention moved to spaces/harmonic.py)


def _realisation_classes():
    out = set()
    for rel, path in modules():
        if rel.startswith("realisations/"):
            out |= {n.name for n in ast.walk(ast.parse(open(path).read())) if isinstance(n, ast.ClassDef)}
    return out


def realisation_type_tests(rel, source, classes):
    """{(rel, class name)} of the type tests on realisation classes in one module outside realisations/: the classes
    named by isinstance/issubclass or type(x) is C (by a name imported from realisations, or an attribute
    module.Class of a realisations module), and string comparisons of a type's __name__ with a realisation class name."""
    if rel.startswith("realisations/"):
        return set()
    tree = ast.parse(source)
    imported, modules_ = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = (n.module or "").split(".")
            if "realisations" in mod and mod[-1] != "realisations":
                imported |= {a.asname or a.name for a in n.names}
            elif "realisations" in mod or (not n.module and n.level >= 1):
                modules_ |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            modules_ |= {a.asname or a.name.split(".")[-1] for a in n.names if "realisations" in a.name}

    def named(t):
        for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
            if isinstance(e, ast.Name) and e.id in imported and e.id in classes:
                yield e.id
            elif isinstance(e, ast.Attribute) and e.attr in classes and \
                    (isinstance(e.value, ast.Attribute) or getattr(e.value, "id", None) in modules_ | imported):
                yield e.attr
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("isinstance", "issubclass") and len(n.args) == 2:
            out |= {(rel, c) for c in named(n.args[1])}
        elif isinstance(n, ast.Compare) and "__name__" in ast.unparse(n):
            out |= {(rel, c.value) for c in ast.walk(n) if isinstance(c, ast.Constant) and c.value in classes}
        elif isinstance(n, ast.Compare) and any(isinstance(s, ast.Call) and getattr(s.func, "id", "") == "type"
                                                for s in [n.left] + n.comparators):
            out |= {(rel, c) for s in [n.left] + n.comparators for c in named(s)}         # type(x) is C
    return out


def test_no_type_tests_on_realisation_classes_outside_realisations():
    classes = _realisation_classes()
    found = set()
    for rel, path in modules():
        found |= realisation_type_tests(rel, open(path).read(), classes)
    new = sorted(found - REALISATION_TYPE_TESTS)
    assert not new, f"type tests on realisation classes (ask a capability, realisations/registry.py): {new}"
    stale = sorted(REALISATION_TYPE_TESTS - found)
    assert not stale, f"REALISATION_TYPE_TESTS lists type tests that are gone (shrink the list): {stale}"


@pytest.mark.parametrize("source, found", [
    ("from ..realisations.series.legendre import PiecewiseLegendre\ndef f(R):\n    return isinstance(R, PiecewiseLegendre)\n",
     {"PiecewiseLegendre"}),
    ("def f(R):\n    from ..realisations.series.radial_harmonic import RadialHarmonicSeries, SeparableRadialHarmonics\n"
     "    return isinstance(R, (int, RadialHarmonicSeries, SeparableRadialHarmonics))\n",
     {"RadialHarmonicSeries", "SeparableRadialHarmonics"}),
    ("from ..realisations.series import legendre as LG\ndef f(R):\n    return isinstance(R, LG.PiecewiseLegendre)\n",
     {"PiecewiseLegendre"}),
    ("def f(R):\n    return type(R).__name__ == 'SphericalHarmonicSeries'\n", {"SphericalHarmonicSeries"}),
    ("from ..realisations.piecewise_constant import CellsBackend\ndef f(B):\n    return type(B) is CellsBackend\n",
     {"CellsBackend"}),
    ("from ..spaces.base import L2\ndef f(H):\n    return isinstance(H, L2)\n", set()),
    ("from ..realisations.registry import provides\ndef f(R):\n    return provides(R, 'box_support')\n", set()),
])
def test_the_realisation_type_check_catches_its_planted_defects(source, found):
    """Planted-defect controls: isinstance on a realisation class (imported by name, lazily, through a module alias)
    and a class-name comparison are reported; a space type test and a capability query are not."""
    classes = {"PiecewiseLegendre", "RadialHarmonicSeries", "SeparableRadialHarmonics", "SphericalHarmonicSeries",
               "CellsBackend"}
    assert realisation_type_tests("convex/planted.py", source, classes) == {("convex/planted.py", c) for c in found}


# ------------------------------------------------------------------------- choice rules (plan 070 P2, review D3)
# The data-dependent choice rules (the discrepancy principle, evidence maximisation, the knee, fit_noise and the
# hyperparameter families they search) live in meta/ (thesis el:59: outside the seven elements) and nowhere else: no
# other package defines one, and only meta/ writes the claim "data-dependent rule" (another package passes on the
# claims of a choice it was given). The cells take a stated alpha.
CHOICE_RULES = {"discrepancy", "maximum_evidence", "knee", "fit_noise", "NoiseFamily", "NoiseFit", "Knee",
                "KneeResult", "Choice", "HyperparameterFamily"}


def choice_rule_sites(rel, source):
    """(rel, what) for each definition named in CHOICE_RULES and each "data-dependent rule" string outside meta/."""
    if _package(rel) == "meta":
        return set()
    out = set()
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in CHOICE_RULES:
            out.add((rel, n.name))
        elif isinstance(n, ast.Constant) and n.value == "data-dependent rule":
            out.add((rel, "data-dependent rule"))
    return out


def test_choice_rules_live_in_meta_and_nowhere_else():
    import pygeoinf0 as pg
    found = set()
    for rel, path in modules():
        found |= choice_rule_sites(rel, open(path).read())
    assert not found, f"choice rules outside meta/ (move them to pygeoinf0.meta): {sorted(found)}"
    rules = getattr(pg.meta, "RULES", {})
    assert {"discrepancy", "maximum_evidence", "knee", "fit_noise"} <= set(rules), sorted(rules)
    assert all(getattr(r, "__module__", "").startswith("pygeoinf0.meta.") for r in rules.values()), rules


@pytest.mark.parametrize("rel, source, found", [
    ("solvers/planted.py", "def discrepancy(A, d):\n    pass\n", {"discrepancy"}),
    ("relations/planted.py", "class NoiseFamily:\n    pass\n", {"NoiseFamily"}),
    ("cells/planted.py", "def f():\n    return Claim('x', 'asserted', 'y', key='data-dependent rule')\n",
     {"data-dependent rule"}),
    ("inference/planted.py", "class C:\n    def knee(self):\n        pass\n", {"knee"}),
    ("meta/planted.py", "def discrepancy(problem):\n    return Claim('x', key='data-dependent rule')\n", set()),
    ("cells/planted.py", "def f(gamma):\n    return gamma.claims.find('used the datum')\n", set()),
])
def test_the_choice_rule_check_catches_its_planted_defects(rel, source, found):
    """Planted-defect controls: a rule defined outside meta/ (a function, a class, a method) and the claim written
    outside meta/ are reported; meta/ itself and reading a choice's claims are not."""
    assert choice_rule_sites(rel, source) == {(rel, f) for f in found}


# plan 070 P5 (review D9): every inference method declares the calibrations it supports, read by Problem.run (not
# inferred from its name or its result): the decorator @method(..., calibrations=(...)) states them.
def undeclared_methods(rel, source):
    """(rel, name) for each @method(...) registration without a calibrations= keyword."""
    out = set()
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            for dec in n.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", getattr(dec.func, "attr", None)) == "method" \
                        and not any(k.arg == "calibrations" for k in dec.keywords):
                    out.add((rel, n.name))
    return out


def test_every_method_declares_its_calibrations():
    from pygeoinf0.inference import calibration, methods
    found = set()
    for rel, path in modules():
        found |= undeclared_methods(rel, open(path).read())
    assert not found, f"methods without calibrations= (declare what they support): {sorted(found)}"
    assert methods.METHODS
    for name, m in methods.METHODS.items():
        cals = getattr(m, "calibrations", None)
        assert isinstance(cals, tuple) and cals, f"{name}: no declared calibrations"
        assert all(isinstance(c, type) and issubclass(c, calibration.Calibration) for c in cals), (name, cals)
        assert calibration.Unstated in cals, f"{name}: refuses a problem with no calibration stated"
    with pytest.raises(TypeError):
        methods.method("planted", [], "nothing")                                  # the declaration is required


@pytest.mark.parametrize("source, found", [
    ("@method('x', [], 'p')\ndef x(**_):\n    pass\n", {"x"}),
    ("@methods.method('y', [], 'p', options=())\ndef y(**_):\n    pass\n", {"y"}),
    ("@method('x', [], 'p', calibrations=(Unstated,))\ndef x(**_):\n    pass\n", set()),
    ("@other('x')\ndef x():\n    pass\n", set()),
])
def test_the_calibration_declaration_check_catches_its_planted_defects(source, found):
    """Planted-defect controls: a registration without calibrations= (by name or through the module) is reported; a
    declared one and another decorator are not."""
    assert undeclared_methods("inference/planted.py", source) == {("inference/planted.py", f) for f in found}
