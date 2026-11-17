"""Exact intersections of round sets, the sets of restricted measures (plan 035 P1, design D2).

    meet(S, D)                       S cap D: a Ball, Shell, Interval, Box or band of a sphere (domains/bands.py) when
                                     it is one, a BallPiece, None when it is null (disjoint or touching), Unsupported
                                     otherwise
    BallPiece(ball, around)          B(c, rho) cap {a <= |x - C| <= b} with c != C (a = 0: a ball about C), the part of a
                                     ball inside a ball or shell about another centre, e.g. a bump's support cut by a
                                     layer of a LayeredBall

Both descriptions are kept (.ball, .around, .parts), so membership in either set is decided from the numbers, not
sampled: BallPiece.inside(D) is exact. Its chart uses coordinates about C: the radius r, u = cos of the angle
from the axis C -> c and the azimuth about that axis. For each r the piece is the cap u >= u0(r),
1 - u0(r) = (rho^2 - (r - d)^2) / (2 r d) (d = |c - C|; clipped at 2), so the rule is Gauss in r (split where
u0 = -1), Gauss in 1 - u on [0, 1 - u0(r)] (no cancellation for narrow caps) and the trapezoid rule in the azimuth
(pygeoinf0.quadrature.rule through BallPiece.charts());
the volume is exact (a cubic in r per panel).
"""
import math

import numpy as np

from ..core.logic import Truth
from ..sets.regions import Chart
from .solids import Ball
from .base import Domain, _fmt
from .layered import Shell

TOL = 1e-12          # relative to the larger radius: centres and radii closer than this are equal


def _annulus(S):
    """(centre, inner, outer) of a Ball (inner 0) or Shell; None for other sets."""
    if isinstance(S, Shell):
        return np.array(S.centre, float), S.inner, S.outer
    if isinstance(S, Ball):
        return np.array(S.centre, float), 0.0, S.radius
    return None


def annuli(S):
    """The annuli whose intersection is S (Ball, Shell, BallPiece), or None."""
    if isinstance(S, BallPiece):
        return [(np.array(S.ball[0]), 0.0, S.ball[1]), (np.array(S.around[0]), S.around[1], S.around[2])]
    a = _annulus(S)
    return None if a is None else [a]


def _subset(Y, X, tol):
    """Y within X (annuli (centre, inner, outer)), decided exactly: the outer ball of Y in that of X and the hole of X
    outside Y's outer ball or inside Y's hole."""
    d = float(np.linalg.norm(Y[0] - X[0]))
    if d + Y[2] > X[2] + tol:
        return False
    return X[1] == 0 or d + X[1] <= Y[1] + tol or d >= Y[2] + X[1] - tol


def _apart(Y, X, tol):
    """Y cap X null (touching counts), decided exactly."""
    d = float(np.linalg.norm(Y[0] - X[0]))
    return d >= Y[2] + X[2] - tol or d + Y[2] <= X[1] + tol or d + X[2] <= Y[1] + tol


def disjoint(S, D):
    """Whether the round sets S and D (balls, shells, BallPieces) are apart by more than the tolerance (a gap): TRUE,
    or UNKNOWN when they are not (they meet or touch) or are not round (plan 067 P2: S & D is then empty)."""
    ra, rb = annuli(S), annuli(D)
    if ra is None or rb is None:
        return Truth.UNKNOWN
    tol = TOL * max(r[2] for r in ra + rb)
    return Truth.TRUE if any(_apart(X, Y, -tol) for X in ra for Y in rb) else Truth.UNKNOWN


def _as_set(ring):
    C, a, b = ring
    return Ball(b, tuple(C)) if a == 0 else Shell(a, b, tuple(C))


def meet(S, D):
    """S cap D, exactly (module docstring): a set, None when the intersection is null, Unsupported when it is not one
    of the kinds constructed here."""
    from ..core.errors import Unsupported
    from ..sets import Interval
    from .solids import Box
    if S is D or S == D:
        return S
    if isinstance(S, Interval) and isinstance(D, Interval):
        J = S & D
        return J if isinstance(J, Interval) else None        # a point or nothing: null
    if isinstance(S, Box) and isinstance(D, Box) and S.dim == D.dim:
        lo = [max(a.a, b.a) for a, b in zip(S.sides, D.sides)]
        hi = [min(a.b, b.b) for a, b in zip(S.sides, D.sides)]
        return None if any(h <= l for l, h in zip(lo, hi)) else Box(*zip(lo, hi))
    from .bands import colatitudes, meet_bands
    if colatitudes(S) is not None and colatitudes(D) is not None:    # a sphere and its bands (domains/bands.py)
        return meet_bands(S, D)
    ra, rb = annuli(S), annuli(D)
    if ra is None or rb is None:
        raise Unsupported(f"the intersection of {S!r} and {D!r} is not constructed here (balls, shells, their pieces, "
                          "intervals, boxes, and bands of a sphere are)")
    scale = max(r[2] for r in ra + rb)
    tol = TOL * scale
    rings = []
    for C, a, b in ra + rb:                              # rings about one centre merge into one
        for j, (C2, a2, b2) in enumerate(rings):
            if np.linalg.norm(C - C2) <= tol:
                rings[j] = (C2, max(a, a2), min(b, b2))
                break
        else:
            rings.append((C, a, b))
    if any(b - a <= tol for _, a, b in rings):
        return None
    for i, X in enumerate(rings):
        for Y in rings[i + 1:]:
            if _apart(X, Y, tol):
                return None
    keep = [X for i, X in enumerate(rings)
            if not any(j != i and _subset(Y, X, tol) and not (_subset(X, Y, tol) and j > i) for j, Y in enumerate(rings))]
    for cand in (S, D):                                  # an operand that is the answer keeps its identity
        rc = annuli(cand)
        if rc is not None and len(rc) == len(keep) and all(any(np.linalg.norm(x[0] - y[0]) <= tol and abs(x[1] - y[1])
                                                                <= tol and abs(x[2] - y[2]) <= tol for y in keep)
                                                            for x in rc):
            return cand
    if len(keep) == 1:
        return _as_set(keep[0])
    balls = [X for X in keep if X[1] == 0]
    if len(keep) == 2 and balls:
        ball = balls[0]
        other = keep[1] if keep[0] is ball else keep[0]
        return BallPiece((tuple(ball[0]), ball[2]), (tuple(other[0]), other[1], other[2]), parts=(S, D))
    raise Unsupported(f"the intersection of {S!r} and {D!r} is bounded by {len(keep)} spheres about different centres; "
                      "only a ball cut by one ball or shell is constructed here")


class BallPiece(Domain):
    """B(c, rho) cap {a <= |x - C| <= b}, c != C, a partial overlap (module docstring). ball = (c, rho), around =
    (C, a, b); parts: the sets it was made from (for printing)."""

    dim = 3

    def __init__(self, ball, around, parts=None):
        c, rho = ball
        C, a, b = around
        self.ball = (tuple(float(x) for x in c), float(rho))
        self.around = (tuple(float(x) for x in C), float(a), float(b))
        self.parts = parts
        c, C = np.array(self.ball[0]), np.array(self.around[0])
        self._d = float(np.linalg.norm(c - C))
        if not (self._d > 0 and 0 <= a < b and rho > 0):
            raise ValueError("a ball piece needs a ball about another centre than the ball or shell it is cut by")
        self._axis = (c - C) / self._d
        e = np.eye(3)[int(np.argmin(np.abs(self._axis)))]
        e1 = e - (e @ self._axis) * self._axis
        self._e1 = e1 / np.linalg.norm(e1)
        self._e2 = np.cross(self._axis, self._e1)
        d = self._d
        self._r = (max(a, d - rho, 0.0), min(b, d + rho))
        if self._r[1] <= self._r[0]:
            raise ValueError("the ball and the ball or shell do not overlap")

    def _v0(self, r):
        """1 - u0(r) = (rho^2 - (r - d)^2) / (2 r d), clipped to [0, 2]: the cap's height, without cancellation."""
        d, rho = self._d, self.ball[1]
        return np.clip((rho * rho - (r - d) ** 2) / (2 * np.maximum(r, 1e-300) * d), 0.0, 2.0)

    def _panels(self):
        lo, hi = self._r
        cut = self.ball[1] - self._d                      # r below which the whole sphere |x - C| = r is in the ball
        return [(lo, cut), (cut, hi)] if lo < cut < hi else [(lo, hi)]

    def charts(self):
        """One chart (module docstring): (r, s, alpha) on [r_lo, r_hi] x [-1, 1] x [0, 2 pi), r split where u0 = -1,
        1 - u = v0(r) (1 - s) / 2, with the Jacobian r^2 v0(r) / 2 in r (pygeoinf0.quadrature.rule: Gauss in r and s,
        the trapezoid rule in the azimuth)."""
        panels = self._panels()
        brk = (tuple(lo for lo, _ in panels[1:]), None, None)
        return (Chart(((panels[0][0], panels[-1][1]), (-1.0, 1.0), (0.0, 2 * math.pi)), (False, False, True), brk,
                      self._to_cartesian, (lambda r: r * r * (0.5 * self._v0(r)), None, None), None),)

    def _to_cartesian(self, r, s, al):
        v = 0.5 * self._v0(r) * (1 - s)                                     # v = 1 - u in [0, v0(r)]
        u, q = 1 - v, np.sqrt(v * (2 - v))
        dirs = [u * a + q * (np.cos(al) * e1 + np.sin(al) * e2) for a, e1, e2 in zip(self._axis, self._e1, self._e2)]
        return tuple(C + r * d for C, d in zip(self.around[0], dirs))

    @property
    def volume(self):
        """Exact: 2 pi int r^2 (1 - u0(r)) dr is a cubic in r on each panel (four Gauss points)."""
        from ..quadrature import gauss
        g, gw = gauss(4)
        out = 0.0
        for lo, hi in self._panels():
            r = 0.5 * (lo + hi) + 0.5 * (hi - lo) * g
            out += 0.5 * (hi - lo) * float(gw @ (r * r * self._v0(r)))
        return 2 * math.pi * out

    def _inside(self, x, y, z):
        (c, rho), (C, a, b) = self.ball, self.around
        tol = TOL * max(rho, b) + 1e-9 * b
        q = (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2
        r = np.sqrt((x - C[0]) ** 2 + (y - C[1]) ** 2 + (z - C[2]) ** 2)
        return (q <= (rho + tol) ** 2) & (r >= a - tol) & (r <= b + tol)

    def inside(self, D):
        """Whether the piece lies in the closed set D, a Truth (plan 067 P1; was True or None): TRUE when decided
        exactly from the descriptions (one of the two sets is in D), UNKNOWN when not decided here (never FALSE yet)."""
        if self.parts is not None and any(p is D or p == D for p in self.parts):
            return Truth.TRUE
        rd = annuli(D)
        if rd is None:
            return Truth.UNKNOWN
        mine = [(np.array(self.ball[0]), 0.0, self.ball[1]), (np.array(self.around[0]),) + self.around[1:]]
        tol = TOL * max(self.ball[1], self.around[2], max(r[2] for r in rd))
        if all(any(_subset(Y, X, tol) for Y in mine) for X in rd):
            return Truth.TRUE
        return Truth.UNKNOWN

    @property
    def bounding_box(self):
        (c, rho), (C, _, b) = self.ball, self.around
        return [(max(ci - rho, Ci - b), min(ci + rho, Ci + b)) for ci, Ci in zip(c, C)]

    def key(self):
        return ("BallPiece", self.ball, self.around)

    def __repr__(self):
        (c, rho), (C, a, b) = self.ball, self.around
        ball = f"Ball(radius={_fmt(rho)}, centre=({', '.join(f'{v:.6g}' for v in c)}))"
        other = repr(Ball(b, C)) if a == 0 else repr(Shell(a, b, C))
        return f"{ball} ∩ {other}"
