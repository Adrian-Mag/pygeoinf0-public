"""The geometry the bank's families are written against (plan 021; experiments/EXP-037-function-bank/DESIGN.md 3).

This is the one module of pygeoinf0.functions that tests the type of a domain; families use only these functions.

    kind(D)                  "interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus", "curve"
    ambient(D), dimension(D) the number of Cartesian coordinates functions take; the intrinsic dimension
    distance(D, c, *X)       the distance from the point c: Euclidean on flat domains, geodesic on spheres and circles,
                             the periodic minimum image on tori
    room(D, c)               the distance from c to the boundary (inf on closed manifolds)
    injectivity(D)           the largest radius of a geodesic ball (pi R on spheres and circles, half the shortest period
                             on tori, inf on flat domains)
    centre(D), volume(D)     a natural centre; the measure of the domain
    quadrature(D, n)         points (m, ambient) and weights (m,) of the domain's own rule
    check_centre(D, c, inside=True)   c as an array, refused (pg.Refused) unless it is a point of D: on spheres and
                             circles on the manifold (|c - o| = R to 1e-9 R), on flat domains inside or on the boundary
                             (inside=False: any finite point, for functions defined everywhere such as a Gaussian)
    radial_shell_mass(D, profile, lo, hi)   the same integral over the radial band lo <= distance <= hi
    radial_mass(D, profile, rho)   the integral over D of x -> profile(distance(c, x)) for a profile supported in
                             [0, rho] with rho <= room(D, c): Euclidean |S^(d-1)| int_0^rho profile(r) r^(d-1) dr; sphere
                             2 pi R^2 int_0^(rho/R) profile(R t) sin t dt; circle 2 int_0^rho profile(s) ds
    standard(kind)           standard domain instances of a kind (for tests and the catalogue)
"""
import itertools
import math

import numpy as np

from ..quadrature import gauss
from ..sets import Interval
from ..domains import Arc, Ball, Box, Circle, Cylinder, Disk, Polyline, Segment, Shell, Sphere, Torus

SPHERE_AREA = {1: 2.0, 2: 2 * math.pi, 3: 4 * math.pi}          # |S^(d-1)|
_GL = gauss(64)


def kind(D):
    if isinstance(D, Interval):
        return "interval"
    for cls, name in ((Shell, "shell"), (Ball, "ball"), (Disk, "disk"), (Sphere, "sphere"), (Circle, "circle"),
                      (Cylinder, "cylinder"), (Torus, "torus"), (Box, "box")):
        if isinstance(D, cls):
            return name
    if getattr(D, "dim", None) == 1 and getattr(D, "ambient_dim", 1) > 1:
        return "curve"
    raise TypeError(f"the function bank has no geometry for {D!r}")


def ambient(D):
    return int(D.cartesian_dim)                    # the Region protocol (plan 067 P2): an interval's is 1


def dimension(D):
    return int(D.dim)


def centre(D):
    k = kind(D)
    if k == "interval":
        return np.array([0.5 * (D.a + D.b)])
    if k in ("box", "torus"):
        return np.array([0.5 * (s.a + s.b) for s in D.sides])
    if k == "shell":
        c = np.asarray(D.centre, float)
        return c + np.array([0.5 * (D.inner + D.outer), 0.0, 0.0])
    if k == "sphere":
        return np.asarray(D.centre, float) + np.array([0.0, 0.0, D.radius])        # the north pole
    if k == "circle":
        return np.asarray(D.centre, float) + np.array([D.radius, 0.0])
    if k == "cylinder":
        return np.r_[np.asarray(D.centre, float), 0.5 * (D.z.a + D.z.b)]
    if k == "curve":
        return np.asarray(D.to_cartesian(np.array([0.5 * sum(D.reference[0])]))).reshape(-1)
    return np.asarray(D.centre, float)


def _stack(X):
    return np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)


def distance(D, c, *X):
    """The distance from c (an array of length ambient(D)) to the points X (Cartesian coordinate arrays)."""
    k, c = kind(D), np.asarray(c, float).reshape(-1)
    if k not in ("sphere", "circle", "torus"):         # flat: the same sum of squares, axis by axis, without a stack
        s = 0.0
        for x, ck in zip(X, c):
            s = s + (np.asarray(x, float) - ck) ** 2
        return np.sqrt(np.broadcast_to(s, np.broadcast(*[np.asarray(x) for x in X]).shape))
    P = _stack(X)
    if k in ("sphere", "circle"):
        o = np.asarray(D.centre, float)
        u, v = (P - o) / D.radius, (c - o) / D.radius
        return D.radius * np.arccos(np.clip(u @ v, -1.0, 1.0))
    if k == "torus":
        L = np.array([s.b - s.a for s in D.sides])
        d = np.abs(P - c)
        d = np.minimum(d, L - d)
        return np.sqrt(np.sum(d ** 2, axis=-1))
    return np.sqrt(np.sum((P - c) ** 2, axis=-1))


def room(D, c):
    """The distance from c to the boundary of D (inf on closed manifolds; negative if c lies outside)."""
    k, c = kind(D), np.asarray(c, float).reshape(-1)
    if k == "interval":
        return float(min(c[0] - D.a, D.b - c[0]))
    if k == "box":
        return float(min(min(c[i] - s.a, s.b - c[i]) for i, s in enumerate(D.sides)))
    if k in ("disk", "ball"):
        return float(D.radius - np.linalg.norm(c - np.asarray(D.centre, float)))
    if k == "shell":
        r = np.linalg.norm(c - np.asarray(D.centre, float))
        return float(min(r - D.inner, D.outer - r))
    if k == "cylinder":
        r = np.linalg.norm(c[:2] - np.asarray(D.centre, float))
        return float(min(D.radius - r, c[2] - D.z.a, D.z.b - c[2]))
    return math.inf


def injectivity(D):
    k = kind(D)
    if k in ("sphere", "circle"):
        return math.pi * D.radius
    if k == "torus":
        return 0.5 * min(s.b - s.a for s in D.sides)
    return math.inf


def volume(D):
    return float(D.volume)                         # the Hausdorff measure of the Region (an interval's length)


def quadrature(D, n=16):
    from ..quadrature import rule
    X, w = rule(D, n)
    X = np.asarray(X, float)
    return (X[:, None] if X.ndim == 1 else X), np.asarray(w, float)


def check_centre(D, c, inside=True):
    """c as an array if it is a point of D, else refused with the reason (module docstring)."""
    from ..core.errors import Refused
    try:
        c = np.asarray(c, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused(f"a centre is a point (numbers), not {c!r}") from None
    if c.size != ambient(D) or not np.all(np.isfinite(c)):
        raise Refused(f"a centre on {D!r} has {ambient(D)} finite Cartesian coordinates, not {c!r}")
    k = kind(D)
    if k in ("sphere", "circle"):
        gap = abs(np.linalg.norm(c - np.asarray(D.centre, float)) - D.radius)
        if gap > 1e-9 * D.radius:
            raise Refused(f"the centre {tuple(c)} is not on {D!r} (off by {gap:.3g})")
    elif k == "torus" and inside:
        if any(not (s.a <= ci <= s.b) for ci, s in zip(c, D.sides)):
            raise Refused(f"the centre {tuple(c)} is outside {D!r}")
    elif inside and k != "curve" and room(D, c) < 0:
        raise Refused(f"the centre {tuple(c)} is outside {D!r}")
    return c


def _graded(lo, hi, panels):
    """Panel edges on [lo, hi], uniform for the first half of the panels, then halving towards hi (profiles such as
    exp(-k/(1-t^2)) with small k are steep at the end of their support)."""
    n1 = max(1, panels // 2)
    mid = lo + 0.5 * (hi - lo)
    left = np.linspace(lo, mid, n1 + 1)
    right = hi - (hi - mid) * 0.5 ** np.arange(0, 2 * panels)
    return np.unique(np.r_[left, right, hi])


def radial_shell_mass(D, profile, lo, hi, panels=8):
    """The integral over D of profile(distance(c, x)) over the band lo <= distance <= hi (inside D)."""
    return _radial(D, profile, _graded(lo, hi, panels))


def radial_mass(D, profile, rho, panels=8):
    """The integral over D of profile(distance(c, x)), the profile supported in [0, rho] (module docstring); composite
    Gauss-Legendre in r (64 nodes per panel) on panels graded towards rho."""
    return _radial(D, profile, _graded(0.0, rho, panels))


def _radial(D, profile, edges):
    k = kind(D)
    t, w = _GL
    total = 0.0
    for a, b in itertools.pairwise(edges):
        r = 0.5 * (b - a) * (t + 1) + a
        wr = 0.5 * (b - a) * w
        if k == "sphere":
            R = D.radius
            total += 2 * math.pi * R ** 2 * np.sum(wr * profile(r) * np.sin(r / R) / R)
        elif k == "circle":
            total += 2.0 * np.sum(wr * profile(r))
        else:
            d = ambient(D) if k != "curve" else 1
            total += SPHERE_AREA[d] * np.sum(wr * profile(r) * r ** (d - 1))
    return float(total)


def standard(kind_name):
    """Standard instances of a domain kind."""
    return {
        "interval": [Interval("[0, 1]")],
        "box": [Box((0.0, 1.0), (0.0, 1.0)), Box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))],
        "disk": [Disk(1.0)],
        "ball": [Ball(1.0)],
        "shell": [Shell(0.5, 1.0)],
        "sphere": [Sphere(1.0)],
        "circle": [Circle(1.0)],
        "cylinder": [Cylinder(1.0, (0.0, 1.0))],
        "torus": [Torus((0.0, 2 * math.pi), (0.0, 2 * math.pi))],
        "curve": [Segment((0.0, 0.0), (1.0, 0.5)), Polyline([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                  Arc(Sphere(1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
    }[kind_name]
