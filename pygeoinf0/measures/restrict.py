"""Restriction of measures to a set (plan 035 P1, design D2; in measures/ since plan 067 P5): mu|_D, the measure A -> mu(A cap D), f dmu -> f|_D dmu|_D.

    restrict(mu, D, side=None)      (also mu.restricted(D, side)): a measure, or None when mu does not charge D

    Lebesgue, density and bank measures   the exact set S cap D (domains/pieces.meet: a BallPiece keeps both
                                          descriptions); a weight with a known support (a bank function) first narrows
                                          S to it
    paths (Segment, Polyline)             cut where they cross the spheres of a ball or shell D
    great-circle arcs (Arc)               cut where they cross the parallels of a band of their sphere (closed form)
    points, counting measures, spheres    kept or dropped; on the boundary of D side= ("below": nearer the centre,
                                          the lower end, or for a band of a sphere the smaller colatitude; "above") is
                                          required, never split in halves (design D4)
    signed measures                       termwise

The result records where it came from (.restricted_from = (mu, D)) and says so in its claims (key "restriction").
"""
import numpy as np

from ..core.claims import DERIVED, Claim
from .base import SignedMeasure
from .lebesgue import DensityMeasure, LebesgueMeasure, PieceDensity
from .points import CountingMeasure, DiracMeasure
from ..sets import Interval

SIDES = (None, "below", "above")


def _mark(out, mu, D, how):
    out._restriction = Claim(f"the restriction of {mu} to {D}", DERIVED, how, key="restriction")
    out.restricted_from = (mu, D)
    return out


def point_in(D, x, side):
    """Whether the point x lies in the closed set D; on the boundary of a ball, a shell, an interval or a band of a
    sphere side= decides ("below": the side nearer the centre, the lower end of an interval, the smaller colatitude;
    "above": the other), and None is refused."""
    from ..domains import Ball, Box, Shell, Sphere
    from ..core.errors import Unsupported
    if side not in SIDES:
        raise ValueError("side must be 'below', 'above' or None")
    x = np.atleast_1d(np.asarray(x, float))

    def decide(lower):
        if side is None:
            raise ValueError(f"the point ({', '.join(f'{v:g}' for v in x)}) is on the boundary of {D}, where a field on pieces may have two "
                             "values: say side='below' or side='above'")
        return (side == "above") == lower                # on the lower boundary the piece is above it
    if isinstance(D, Interval):
        tol = 1e-12 * max(1.0, abs(D.a), abs(D.b))
        if abs(x[0] - D.a) <= tol:
            return decide(True)
        if abs(x[0] - D.b) <= tol:
            return decide(False)
        return bool(D.a < x[0] < D.b)
    if isinstance(D, (Ball, Shell)):
        from ..domains.layered import ON
        a, b = (0.0, D.radius) if isinstance(D, Ball) else (D.inner, D.outer)
        r, tol = float(np.linalg.norm(x - np.array(D.centre))), ON * b
        if a > 0 and abs(r - a) <= tol:
            return decide(True)
        if abs(r - b) <= tol:
            return decide(False)
        return bool(a < r < b)
    from ..domains.bands import SphereBand, point_side
    if isinstance(D, SphereBand):                        # colatitude: "below" the side nearer the north pole
        where = point_side(D, x)
        return decide(where == "lower") if where in ("lower", "upper") else where == "in"
    if isinstance(D, Box):
        lo, hi = np.array([s.a for s in D.sides]), np.array([s.b for s in D.sides])
        tol = 1e-12 * max(1.0, float(np.max(hi - lo)))
        if np.any(np.abs(x - lo) <= tol) or np.any(np.abs(x - hi) <= tol):
            raise Unsupported(f"the point ({', '.join(f'{v:g}' for v in x)}) is on the boundary of {D}: side= is defined on the spheres of balls "
                              "and shells and the ends of intervals")
        return bool(np.all(x > lo) and np.all(x < hi))
    if isinstance(D, Sphere):
        return bool(D.contains(x))
    raise Unsupported(f"restricting a point mass to {D!r} (balls, shells, intervals, boxes, spheres and their bands are "
                      "decided)")


def _round_of(D):
    """(centre, inner, outer) of a Ball or Shell, else None."""
    from ..domains import Ball, Shell
    if isinstance(D, Shell):
        return np.array(D.centre, float), D.inner, D.outer
    if isinstance(D, Ball):
        return np.array(D.centre, float), 0.0, D.radius
    return None


def _restrict_path(m, D):
    """The pieces of a path measure (arc length or a density on a Segment or Polyline) inside the ball or shell D,
    cut where the path crosses D's spheres: a list of measures (one per run of pieces inside D)."""
    from ..domains import Polyline
    from ..domains.layered import ON, cut_at_spheres
    from ..core.errors import Unsupported
    C, a, b = _round_of(D)
    S = m.set
    pts = np.array(S.endpoints) if not isinstance(S, Polyline) else S._P
    wt = getattr(m, "weight", None)
    pieces = wt is not None and wt.kind == "pieces"
    if wt is not None and not pieces and wt.kind != "constant" and wt.kind != "callable":
        raise Unsupported(f"restricting {m}: a weight of kind {wt.kind} on a path is cut only as one value per piece or "
                          "a callable")
    radii = [x for x in (a, b) if x > 0]
    tol = ON * b
    X, wx, rm = cut_at_spheres(pts, np.asarray(wt.c, float) if pieces else None, C, radii, tol)
    inside = (rm < b - tol) & ((rm > a + tol) | (np.abs(rm - a) <= tol) if a > 0 else np.ones(len(rm), bool))
    out = []
    for j0, j1 in _runs(inside):
        run = Polyline(X[j0:j1 + 1])
        if isinstance(m, PieceDensity):
            out.append(PieceDensity(run, wx[j0:j1], name=m.name))
        elif isinstance(m, DensityMeasure):
            from .weights import RestrictedWeight
            w = DensityMeasure(LebesgueMeasure(run), wx[j0:j1] if pieces else
                               (wt.c if wt.kind == "constant" else RestrictedWeight(run, wt)), name=m.name)
            out.append(w)
        else:
            out.append(LebesgueMeasure(run))
    return out


def _runs(mask):
    """(first, last + 1) of the runs of True in mask."""
    out, j = [], 0
    while j < len(mask):
        if mask[j]:
            k = j
            while k < len(mask) and mask[k]:
                k += 1
            out.append((j, k))
            j = k
        else:
            j += 1
    return out


def _restrict_sphere(m, D, side):
    """A measure on a Sphere restricted to the ball or shell D: the whole measure, nothing, or (on D's boundary) side=."""
    from ..core.errors import Unsupported
    S, R = m.set, _round_of(D)
    C, a, b = R
    d, r0 = float(np.linalg.norm(np.array(S.centre) - C)), S.radius
    from ..domains.layered import ON
    tol = ON * b
    if d <= tol:
        if (a > 0 and abs(r0 - a) <= tol) or abs(r0 - b) <= tol:
            return m if point_in(D, C + np.array([0, 0, r0]), side) else None
        return m if a < r0 < b else None
    if d + r0 < b - tol and d - r0 > a + tol:
        return m
    if d + r0 <= a + tol or d - r0 >= b - tol or r0 - d >= b + tol:
        return None
    raise Unsupported(f"restricting {m} to {D}: the sphere crosses the boundary of {D} (cut spheres are not "
                      "constructed here)")


def restrict(mu, D, side=None):
    """mu|_D (f dmu -> f|_D dmu|_D): the measure A -> mu(A cap D), or None when mu does not charge D (module section
    'restriction'). Measures of full dimension restrict to the exact set S cap D (domains/pieces.py; a weight with a
    known support, a bank function, first narrows S to it); paths are cut at the spheres of a ball or shell; point
    masses and sphere measures on the boundary of D need side= ("below" or "above"), never split in halves; signed
    measures restrict termwise."""
    from ..domains import Arc, Ball, Polyline, Shell, Sphere
    from ..domains.bands import SphereBand, cut_arc
    from ..domains.pieces import meet
    from ..core.errors import Unsupported
    if isinstance(D, str):
        D = Interval(D)
    if isinstance(mu, SignedMeasure):
        terms = [(c, restrict(m, D, side)) for c, m in mu.terms]
        terms = [(c, m) for c, m in terms if m is not None]
        if not terms:
            return None
        if len(terms) == 1 and terms[0][0] == 1.0:
            return terms[0][1]
        return _mark(SignedMeasure(terms), mu, D, "termwise: (sum_t c_t mu_t)|_D = sum_t c_t (mu_t|_D)")
    if isinstance(mu, DiracMeasure):
        return mu if point_in(D, mu.x0, side) else None
    if isinstance(mu, CountingMeasure):
        P = np.asarray(mu.set.points, float)
        keep = np.array([point_in(D, p, side) for p in (P if P.ndim > 1 else P[:, None])], bool)
        if not keep.any():
            return None
        if keep.all():
            return mu
        return _mark(CountingMeasure(P[keep], mu.weights[keep]), mu, D, "the points of the set inside D")
    if not isinstance(mu, (LebesgueMeasure, DensityMeasure)):
        raise Unsupported(f"restricting {mu!r} to a set (Lebesgue, density, path, point and signed measures are "
                          "restricted)")
    S = mu.set
    if S is D or S == D:
        return mu
    dimS, dimD = int(S.dim), int(D.dim)
    if dimS < dimD and isinstance(D, (Sphere, SphereBand)) and isinstance(mu, LebesgueMeasure) and isinstance(S, Arc):
        parts = cut_arc(mu, D, side)
        if not parts:
            return None
        out = parts[0] if len(parts) == 1 else SignedMeasure([(1.0, p) for p in parts])
        return out if out is mu else _mark(out, mu, D, f"the arc cut where it crosses the parallels of {D} (closed form)")
    if dimS < dimD:
        if not isinstance(D, (Ball, Shell)):
            raise Unsupported(f"restricting {mu} to {D!r}: lower-dimensional measures are restricted to balls and "
                              "shells")
        if isinstance(S, Polyline) or (getattr(S, "dim", None) == 1 and hasattr(S, "endpoints")):
            parts = _restrict_path(mu, D)
            if not parts:
                return None
            out = parts[0] if len(parts) == 1 else SignedMeasure([(1.0, p) for p in parts])
            return _mark(out, mu, D, f"the path cut where it crosses the spheres of {D} (exact for straight pieces)")
        if isinstance(S, Sphere):
            return _restrict_sphere(mu, D, side)
        raise Unsupported(f"restricting {mu} to {D!r} (paths and spheres are cut)")
    T = S
    sup = getattr(getattr(mu, "weight", None), "f", None)
    if isinstance(mu, DensityMeasure) and callable(getattr(sup, "support_set", None)):
        try:
            T = meet(S, sup.support_set())          # w vanishes outside its support (the bank function's claim)
        except Unsupported:
            T = S
        if T is None:
            return None
    J = meet(T, D)
    if J is None:
        return None
    if J is S:
        return mu
    how = f"the set {S} cap {D}, constructed exactly (domains/pieces.py)" + \
        ("" if T is S else f"; the weight vanishes outside its support {T} ({sup.support.description})")
    if isinstance(mu, LebesgueMeasure):
        return _mark(LebesgueMeasure(J), mu, D, how)
    from .weights import RestrictedWeight
    wt = mu.weight
    if wt.kind == "pieces":
        raise Unsupported(f"restricting {mu}: one value per piece of {S} does not carry to {J}")
    from ..domains import Ball, Shell
    if wt.kind == "radial" and isinstance(J, (Ball, Shell)) and tuple(J.centre) == tuple(S.centre):
        from .weights import RadialWeight                  # a concentric piece keeps the radial kind (plan 064)
        nw = RadialWeight(J, wt.f, sup=wt.sup)
    else:
        nw = wt.c if wt.kind == "constant" else RestrictedWeight(J, wt)
    out = DensityMeasure(LebesgueMeasure(J), nw, name=mu.name)
    out.w = mu.w                                            # the same weight: equal restrictions compare equal
    return _mark(out, mu, D, how)
