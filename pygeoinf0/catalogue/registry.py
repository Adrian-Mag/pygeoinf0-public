"""What ``pg.catalogue`` probes: the domains, the spaces, the recipe of each pair, the caveats, the extra checks,
and the forgiving resolution of names (see the package docstring)."""
import difflib
import re
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parent.parent / "spaces_snapshot.json"
FORMAT = 1
LENGTH_SCALE = 0.2
FUNCTIONAL_LIMIT = 20            # seconds for one functional in the smoke test (a slower one is recorded as not usable)
STATUSES = ("ok", "constructed", "refused", "n/a", "error")
SYMBOL = {"ok": "ok", "constructed": "c", "refused": "no", "n/a": "-", "error": "ERR"}
GLYPH = {"ok": "✓", "constructed": "○", "refused": "✗", "n/a": "–", "error": "!"}
COLOUR = {"ok": "#2a9d8f", "constructed": "#f2cc8f", "refused": "#d1495b", "n/a": "#cfd3d8", "error": "#9b5de5"}
MEANING = {"ok": "constructed, and a smoke test gave a finite norm",
           "constructed": "constructed; no tested functional is bounded (nothing tested beyond construction)",
           "refused": "refused by the package (reason: pg.catalogue.how)",
           "n/a": "not applicable (closed manifold has no boundary; mixed needs two boundary parts)",
           "error": "unexpected exception or time-out"}

# ------------------------------------------------------------------------------------------------- the registry
# key, label, intrinsic dimension, constructor, functionals to try in order: (label, code of the measure given to riesz)
_PT = "point evaluation"
DOMAINS = (
    ("Interval", "Interval", 1, 'pg.Interval("[0, 1]")', ((_PT, "0.5"),)),
    ("Segment", "Segment (2D)", 1, "pg.Segment((0, 0), (1, 0.5))", ((_PT, "np.array([0.5, 0.25])"),)),
    ("Polyline", "Polyline (2D)", 1, "pg.Polyline([[0, 0], [1, 0], [1, 1]])", ((_PT, "np.array([0.5, 0.0])"),)),
    ("Arc", "Great-circle arc", 1, "pg.Arc(pg.Sphere(1.0), (1, 0, 0), (0, 1, 0))",
     ((_PT, "np.array([0.7071067811865476, 0.7071067811865476, 0.0])"),)),
    ("Curve", "Curve (2D)", 1, "pg.Curve(lambda t: (t, t ** 2), 0, 1)", ((_PT, "np.array([0.5, 0.25])"),)),
    ("Circle", "Circle", 1, "pg.Circle(1.0)", ((_PT, "np.array([1.0, 0.0])"),)),
    ("Rectangle", "Rectangle", 2, 'pg.Box("[0, 1]", "[0, 1]")',
     ((_PT, "np.array([0.4, 0.5])"), ("ray integral", "pg.Segment((0.1, 0.1), (0.8, 0.7))"))),
    ("Disk", "Disk", 2, "pg.Disk(1.0)",
     ((_PT, "np.array([0.2, 0.1])"), ("ray integral", "pg.Segment((-0.5, -0.3), (0.4, 0.5))"))),
    ("Patch", "Flat patch (3D)", 2, "pg.Patch((0, 0, 0), (1, 0, 0), (0, 1, 0))", ((_PT, "np.array([0.5, 0.5, 0.0])"),)),
    ("Sphere", "Sphere", 2, "pg.Sphere(1.0)",
     ((_PT, "np.array([0.0, 0.0, 1.0])"), ("arc integral", "pg.Arc(D, (1, 0, 0), (0, 1, 0))"))),
    ("Torus2", "Torus 2D", 2, 'pg.Torus("[0, 1]", "[0, 1]")',
     ((_PT, "np.array([0.3, 0.4])"), ("ray integral", "pg.Segment((0.1, 0.1), (0.8, 0.7))"))),
    ("Box", "Box", 3, 'pg.Box("[0, 1]", "[0, 1]", "[0, 1]")',
     ((_PT, "np.array([0.4, 0.5, 0.6])"), ("sphere surface integral", "pg.Sphere(0.2, centre=(0.5, 0.5, 0.5))"))),
    ("Cylinder", "Cylinder", 3, "pg.Cylinder(1.0, z=(0.0, 1.0))",
     ((_PT, "np.array([0.2, 0.1, 0.5])"), ("sphere surface integral", "pg.Sphere(0.2, centre=(0, 0, 0.5))"))),
    ("Ball", "Ball", 3, "pg.Ball(1.0)",
     ((_PT, "np.array([0.1, 0.2, 0.3])"), ("sphere surface integral", "pg.Sphere(0.3, centre=(0, 0, 0))"))),
    ("Shell", "Shell", 3, "pg.Shell(0.5, 1.0)",
     ((_PT, "np.array([0.7, 0.0, 0.0])"), ("sphere surface integral", "pg.Sphere(0.1, centre=(0.75, 0, 0))"))),
    ("Torus3", "Torus 3D", 3, 'pg.Torus("[0, 1]", "[0, 1]", "[0, 1]")',
     ((_PT, "np.array([0.3, 0.4, 0.5])"), ("sphere surface integral", "pg.Sphere(0.2, centre=(0.5, 0.5, 0.5))"))),
    ("Layered", "Layered ball", 3, "pg.LayeredBall([0.5, 1.0])", ((_PT, "np.array([0.7, 0.0, 0.0])"),)),
    # (the spaces of a layered ball are the broken sums pg.PiecewiseSpace.lebesgue / .sobolev on its pieces; the other
    # constructors are tried on the ball itself and their refusals quoted)
)
DOMAIN_ALIASES = {
    "Interval": ("interval", "line", "1d interval"), "Segment": ("segment", "line segment"),
    "Polyline": ("polyline", "path"), "Arc": ("arc integral", "great circle", "great-circle arc", "greatcirclearc"),
    "Curve": ("curve", "parametric curve"), "Circle": ("circle",), "Rectangle": ("rectangle", "rect"), "Disk": ("disk", "disc"),
    "Patch": ("patch", "flat patch", "plane patch"), "Sphere": ("sphere", "s2"),
    "Torus2": ("torus 2d", "2d torus", "torus2", "torus2d", "t2"), "Box": ("box", "cuboid", "brick"),
    "Cylinder": ("cylinder",), "Ball": ("ball",), "Shell": ("shell", "spherical shell"),
    "Torus3": ("torus 3d", "3d torus", "torus3", "torus3d", "t3"),
    "Layered": ("layered", "layered ball", "layeredball", "planet"),
}

# key, label, aliases
SPACES = (
    ("L2", "L2", ("l2", "lebesgue", "l^2")),
    ("Lp", "Lp", ("lp", "l^p", "lebesgue p")),
    ("C", "C", ("c", "continuous", "c(k)", "continuous functions")),
    ("Sobolev", "Sobolev(D)", ("sobolev", "sobolev(d)", "h^s", "restriction", "restriction sobolev", "hs", "sobolev d",
                              "closed spectral", "periodic")),
    ("IntrH1", "Intr H1", ("intr h1", "intrinsic h1", "intrinsich1", "h1", "intrinsic 1")),
    ("IntrH2", "Intr H2", ("intr h2", "intrinsic h2", "intrinsich2", "h2", "intrinsic 2")),
    ("SpecN", "Spec N", ("spec n", "spectral neumann", "neumann", "spec neumann", "specn", "spectral n")),
    ("SpecD", "Spec D", ("spec d", "spectral dirichlet", "dirichlet", "spec dirichlet", "specd", "spectral d")),
    ("SpecMixed", "Spec Mixed", ("spec mixed", "spectral mixed", "mixed", "specmixed")),
    ("SpecRobin", "Spec Robin", ("spec robin", "spectral robin", "robin", "specrobin")),
    ("ComplexL2", "Complex L2", ("complex l2", "complex", "complexified", "complex valued", "l2 complex")),
)
# the smoke test of the spaces that hold constants: a constant's norm
CONSTANT = {"L2": "H(1.0).norm()", "Lp": "H(1.0).norm()", "C": "H(lambda x: 1.0 + 0 * x).norm()",
            "ComplexL2": 'H({"re": 1.0, "im": 0.0}).norm()'}
SLOW = frozenset({("Box", "SpecN"), ("Box", "SpecD"), ("Box", "SpecMixed"), ("Torus3", "Sobolev"), ("Torus3", "IntrH2"),
                  ("Box", "IntrH1"), ("Torus3", "IntrH1"),
                  ("Box", "SpecN", 2.5), ("Box", "SpecD", 2.5), ("Box", "SpecMixed", 2.5)})
BOUNDARY_SPACES = ("SpecN", "SpecD", "SpecMixed", "SpecRobin")

CAVEATS = {
    "L2": "elements are classes of functions: point evaluations and rays are not bounded, only integrals against "
          "square-integrable densities.",
    "Lp": "elements are classes of functions: point evaluations and rays are not bounded on Lp.",
    "C": "the sup norm on C(K) for a closed bounded interval K; its functionals are measures (the dual is not a "
         "Hilbert space).",
    "Sobolev": "on a bounded domain it is the restriction space H^s(D) of the free-space space (the quotient norm); on a "
               "closed manifold (circle, sphere, torus) it is the spectral space of the Laplacian. Point evaluation "
               "needs s > d/2, integration over a k-dimensional set needs s > (d - k)/2.",
    "IntrH1": "the intrinsic norm int u v + l^2 grad u . grad v on H^1 (natural Neumann condition). Point evaluation "
              "needs s > d/2, i.e. it is unbounded on H^1 for d >= 2; integration over a k-dimensional set needs "
              "s > (d - k)/2.",
    "IntrH2": "the intrinsic norm of order 2 (unconstrained: no boundary condition is imposed).",
    "SpecN": "H^s of the spectral scale of A = 1 - l^2 Delta with natural Neumann conditions; point "
             "evaluation needs s > d/2.",
    "SpecD": "H^s of the spectral scale of A = 1 - l^2 Delta with Dirichlet conditions on every boundary part (all "
             "elements vanish there); point evaluation needs s > d/2.",
    "SpecMixed": "Dirichlet on the first boundary part, Neumann on the others; point evaluation needs s > d/2.",
    "SpecRobin": "Robin condition beta = 1 on every boundary part; realised on intervals with integer orders only "
                 "(Robin eigenfunctions are not explicit and fractional powers are realised by images).",
    "ComplexL2": "H (+) H with J(u, v) = (-v, u), the complexification of the real L2 (plan 020); pg.Complexified "
                 "applies to the other real Hilbert function spaces of this table too. Point evaluations and rays are "
                 "not bounded, as on the real L2.",
}


# ------------------------------------------------------------------------------------------------- recipes
def _order(dim):
    """The smallest integer order with s > d/2 (point evaluation bounded)."""
    return dim // 2 + 1


def _space_code(space, domain, dim, parts, order=None):
    """The expression for the space of this column on the domain named D, or None if the pair has no recipe."""
    ls = LENGTH_SCALE
    s = _order(dim) if order is None else order
    s = int(s) if float(s) == int(s) else s
    if domain == "Layered":
        return {"L2": "pg.PiecewiseSpace.lebesgue(D)",
                "Sobolev": f"pg.PiecewiseSpace.sobolev(D, {s}, length_scale={ls})"}.get(space)
    form = f"pg.SobolevForm(D, 1, length_scale={ls}"
    return {
        "L2": "pg.L2(D)", "Lp": "pg.Lp(D, 3)", "C": "pg.Continuous(D)", "ComplexL2": "pg.Complexified(pg.L2(D))",
        "Sobolev": f"pg.Sobolev(D, {s}, length_scale={ls})",
        "IntrH1": f"pg.SobolevForm(D, 1, length_scale={ls}).space()",
        "IntrH2": f"pg.SobolevForm(D, 2, length_scale={ls}).space()",
        "SpecN": f"pg.Sobolev({form}).operator(), {s})",
        "SpecD": f'pg.Sobolev({form}, dirichlet="all").operator(), {s})',
        "SpecMixed": f"pg.Sobolev({form}, dirichlet={tuple(parts[:1])!r}).operator(), {s})",
        "SpecRobin": f"pg.Sobolev({form}, robin=1.0).operator(), {s})",
    }[space]



# ------------------------------------------------------------------------------------------------- extras
# (domain, space, order): the space of that column at a fractional (or other) order
FRACTIONAL = (
    ("Interval", "SpecN", 0.5), ("Interval", "SpecN", 1.5), ("Interval", "SpecN", 2.5),
    ("Interval", "SpecD", 1.5), ("Interval", "SpecD", 2.5), ("Interval", "SpecMixed", 1.5),
    ("Interval", "SpecMixed", 2.5), ("Interval", "SpecRobin", 1), ("Interval", "SpecRobin", 2),
    ("Interval", "SpecRobin", 1.5), ("Interval", "SpecRobin", 2.5),
    ("Box", "SpecN", 2.5), ("Box", "SpecD", 2.5), ("Box", "SpecMixed", 2.5),
)

# damped least squares with three functionals: name, domain, space, order, the three measures, the data
SOLVES = (
    ("Interval, Spec N", "Interval", "SpecN", 2,
     "[pg.Interval(0.1, 0.3), pg.Interval(0.4, 0.6), pg.Interval(0.7, 0.9)]"),
    ("Interval, Sobolev (restriction)", "Interval", "Sobolev", 2,
     "[pg.Interval(0.1, 0.3), pg.Interval(0.4, 0.6), pg.Interval(0.7, 0.9)]"),
    ("Rectangle, Spec N", "Rectangle", "SpecN", 2,
     "[pg.Segment((0.1, 0.1), (0.9, 0.2)), pg.Segment((0.2, 0.9), (0.8, 0.1)), pg.Segment((0.1, 0.5), (0.9, 0.6))]"),
    ("Disk, Sobolev (restriction)", "Disk", "Sobolev", 2,
     "[pg.Segment((-0.8, 0.1), (0.8, 0.2)), pg.Segment((-0.3, -0.8), (0.4, 0.7)), pg.Segment((-0.5, 0.5), (0.6, 0.6))]"),
)
SOLVE_TEMPLATE = ("G = pg.Functionals(H.riesz({rays}), into=pg.Euclidean(3))\n"
                  "d = G.codomain([1.0, 0.5, -0.5])\n"
                  "m = G.adjoint((G @ G.adjoint + 1e-6 * pg.Identity(G.codomain)).inv(d))\n"
                  "residual = (G(m) - d).norm() / d.norm()")
SOLVE_TOLERANCE = 1e-3

# the norm of an element of the Sobolev column given by a formula (domain, formula)
FORMULAS = (("Interval", "lambda x: 1.0 + x ** 2"), ("Rectangle", "lambda x, y: 1.0 + x * y"),
            ("Disk", "lambda x, y: 1.0 + x * y"), ("Box", "lambda x, y, z: 1.0 + x * y * z"))


# ------------------------------------------------------------------------------------------------- names
def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _resolve(name, registry, what, aliases=None):
    """The key for a forgiving name: exact key, label or alias first, then a unique substring or prefix."""
    n = _norm(name)
    names = {}
    for row in registry:
        key, label = row[0], row[1]
        extra = aliases[key] if aliases else row[-1]
        for s in (key, label, *extra):
            names.setdefault(_norm(s), set()).add(key)
    if n in names and len(names[n]) == 1:
        return next(iter(names[n]))
    hits = {k for s, ks in names.items() if n and (s.startswith(n) or n in s) for k in ks}
    if len(hits) == 1:
        return next(iter(hits))
    if n in names:
        hits = names[n]
    labels = {r[0]: r[1] for r in registry}
    if hits:
        raise KeyError(f"{name!r} is ambiguous as a {what}: " + ", ".join(sorted(labels[k] for k in hits)))
    close = difflib.get_close_matches(n, list(names), n=3, cutoff=0.5)
    near = sorted({labels[k] for c in close for k in names[c]})
    raise KeyError(f"unknown {what} {name!r}; " + (f"did you mean {', '.join(near)}? " if near else "")
                   + "choose from " + ", ".join(labels.values()))
