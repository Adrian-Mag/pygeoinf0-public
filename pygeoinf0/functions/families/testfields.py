"""Test fields, part 1 (plan 021 P1, work package G): checkerboards and layered radial profiles (EXP-037 DESIGN.md 5, 6).

    checkerboard(D, *, cells, amplitude=1.0)
        amplitude * (-1)^(k_1 + ... + k_d), k_i = floor(cells_i (x_i - a_i) / L_i) the cell index along axis i of the box
        prod [a_i, b_i]; a sign pattern (box; bounded with jumps, order -1)
    layered_profile(D, *, variable, breaks, coefficients)
        p_j(v) on the layer b_(j-1) <= v < b_j (p_j a polynomial in v, ascending coefficients), v the radius |x - centre|
        or the depth (top - radius; the distance below the top of an interval); ball, shell, disk, interval

The other test fields (blobs; steps and dykes across a plane) are in testfields_blobs.py and testfields_planes.py.

From testfields_blobs.py (merged here by plan 063):
Test fields, part 2 (plan 021 P1, work package G): blobs, finite sums of bumps or Gaussians (EXP-037 DESIGN.md 5, 6).

    blobs(D, *, shape="bump", centres=None, sizes=None, amplitudes=None, seed=None, count=None,
          size_range=None, amplitude_range=None)
        sum_i amplitudes_i * member_i,  member_i the `bump` (sizes_i the radius) or the `gaussian` (sizes_i the width)
        of functions/families/smooth.py about centres_i: given explicitly, or drawn from numpy's default_rng(seed)

All domain kinds of the bump (a Gaussian blob is refused on a torus: the gaussian family is not defined there).

From testfields_planes.py (merged here by plan 063):
Test fields, part 3 (plan 021 P1, work package G): steps and dykes across a plane (EXP-037 DESIGN.md 5, 6).

    step_across_plane(D, *, normal, point=None, lower=-1.0, upper=1.0, width=None)
        lower + (upper - lower) s(t),  t = n . (x - point) the signed distance to the plane;
        s = 1[t > 0] (value 1/2 on the plane) when width is None, else (1 + tanh(t / width)) / 2
    dyke(D, *, normal, thickness, point=None, amplitude=1.0, background=0.0, width=None)
        background + amplitude g(t),  g = 1[|t| < thickness/2] (1/2 on the faces) when width is None, else
        (tanh((t + h) / width) - tanh((t - h) / width)) / 2 with h = thickness / 2
    Domains: box, disk, ball, cylinder (axis z). Each has a point of symmetry (the centre) and the volume of D cut by a
    plane is elementary for a ball (cap), a disk (segment), a box (inclusion-exclusion, in rational arithmetic) and a
    cylinder cut by a plane parallel or orthogonal to its axis: the closed forms of the sharp versions.
    The kinds are those the work package assignment lists (box, disk, ball, cylinder). A shell (a difference of two caps)
    and an interval (where the "plane" is a point) are not offered; DESIGN.md section 6 names "flat domains and
    balls/shells" for WP-G, so a shell is a candidate for a later extension.
"""


import itertools
import math
import numbers
from fractions import Fraction

import numpy as np
from numpy.polynomial import Polynomial

from ...core.claims import CHECKED, DERIVED
from ...core.errors import Refused
from ...quadrature import gauss
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register
from .smooth import BUMP_KINDS, FLAT, HORMANDER, bump, gaussian

GRISVARD = "Grisvard 1985, Elliptic Problems in Nonsmooth Domains, 1.4 (piecewise smooth functions and their H^s)"
EPS = float(np.finfo(float).eps)


def _count(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
        raise Refused(f"{name} must be a positive integer (got {v!r})")
    return int(v)


def _finite(name, v):
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        raise Refused(f"{name} must be a real number (got {v!r})") from None
    if not math.isfinite(x):
        raise Refused(f"{name} must be finite (got {v!r})")
    return x


# --------------------------------------------------------------------------------------------- checkerboard
def _cells_examples(D):
    d = geo.ambient(D)
    return [{"cells": (2,) * d}, {"cells": (4, 2) + (2,) * (d - 2), "amplitude": 2.5},
            {"cells": (3, 2) + (2,) * (d - 2), "amplitude": -1.0}]


@register("checkerboard", ("box",), "sign pattern amplitude * (-1)^(k_1+...+k_d) of a grid of cells per axis",
          references=(GRISVARD,), examples=_cells_examples)
def checkerboard(D, *, cells, amplitude=1.0):
    """The checkerboard of the box D: axis i is cut into cells_i equal cells, k_i the index of the cell containing x_i
    (0 .. cells_i - 1, the upper cell on a cell boundary, the last cell at the top), and
        f(x) = amplitude * (-1)^(k_1 + ... + k_d).
    `cells` is a tuple of positive integers, one per axis (or one integer for every axis), with at least two cells in
    all; `amplitude` is a nonzero real (default 1).
    Support: the whole domain. Regularity: bounded with jumps across the cell boundaries (order -1, piecewise smooth).
    Closed forms: the integral is amplitude * prod_i (L_i / cells_i if cells_i is odd else 0), because the cells of one
    axis alternate in sign with equal lengths L_i / cells_i; f^2 = amplitude^2 so the squared L2 norm is amplitude^2
    times the volume; the maximum is |amplitude| (a negative cell exists since there are at least two cells)."""
    d, sides = geo.ambient(D), [(s.a, s.b) for s in D.sides]
    n = (_count("cells", cells),) * d if isinstance(cells, numbers.Integral) else None
    if n is None:
        try:
            n = tuple(_count("cells entry", c) for c in cells)
        except TypeError:
            raise Refused(f"cells must be an integer or a tuple of {d} integers (got {cells!r})") from None
        if len(n) != d:
            raise Refused(f"cells needs one entry per axis ({d} on this box), got {len(n)}")
    amplitude = _finite("amplitude", amplitude)
    if amplitude == 0.0:
        raise Refused("checkerboard: amplitude must be nonzero")
    if math.prod(n) < 2:
        raise Refused("checkerboard: one cell is a constant; give at least two cells in all")
    lo = np.array([a for a, _ in sides])
    L = np.array([b - a for a, b in sides])
    n_arr = np.array(n)

    def fn(*X):
        P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
        k = np.clip(np.floor(n_arr * (P - lo) / L), 0, n_arr - 1).astype(np.int64).sum(axis=-1)
        return amplitude * (1 - 2 * (k % 2))

    integral = amplitude * math.prod(Li / ni if ni % 2 == 1 else 0.0 for Li, ni in zip(L, n))
    why = "product over the axes of the signed cell lengths: L_i/cells_i for an odd number of cells, 0 for an even one"
    closed = {"integral": (integral, DERIVED, why),
              "l2_norm_sq": (amplitude ** 2 * geo.volume(D), DERIVED, "f^2 = amplitude^2 everywhere"),
              "max": (abs(amplitude), DERIVED, "the values are +-amplitude and both signs occur")}
    return BankFunction("checkerboard", D, {"cells": n, "amplitude": amplitude}, fn,
                        Support("full", "the whole domain (every cell carries a nonzero value)"),
                        Regularity(-1, piecewise=True), closed,
                        reasons={"regularity": "piecewise constant, jumping by 2 amplitude across the cell boundaries; "
                                               + GRISVARD})


# --------------------------------------------------------------------------------------------- layered profile
class _Radial:
    """The variable v of a layered profile on D: the radius rho = |x - c| (v = rho) or the depth (v = top - rho), the
    density of the measure in v (a polynomial: 4 pi rho^2, 2 pi rho, or 2 and 1 on an interval) and the range of v.
    On an interval the radius is |x - c| about the midpoint and the depth is b - x."""

    def __init__(self, D, variable):
        k = geo.kind(D)
        self.interval, self.depth = k == "interval", variable == "depth"
        if self.interval:
            self.c, self.top = 0.5 * (D.a + D.b), D.b
            half = 0.5 * (D.b - D.a)
            self.lo, self.hi = (0.0, D.b - D.a) if self.depth else (0.0, half)
            self.dens, self.origin = Polynomial([1.0] if self.depth else [2.0]), not self.depth
            return
        self.c = np.asarray(D.centre, float)
        rmin, self.top = (D.inner, D.outer) if k == "shell" else (0.0, D.radius)
        area = Polynomial([0.0, 2 * math.pi]) if k == "disk" else Polynomial([0.0, 0.0, 4 * math.pi])
        self.lo, self.hi = (0.0, self.top - rmin) if self.depth else (rmin, self.top)
        self.dens = area(Polynomial([self.top, -1.0])) if self.depth else area
        self.origin = rmin == 0.0

    def in_rho(self, p):
        """The polynomial p(v) as a polynomial in rho (the interval depth has no rho: unchanged)."""
        return p(Polynomial([self.top, -1.0])) if self.depth and not self.interval else p

    def rho_and_dir(self, X):
        P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
        if self.interval:
            d = P[..., 0] - self.c
            return np.abs(d), np.sign(d)[..., None]
        d = P - self.c
        rho = np.sqrt(np.sum(d * d, axis=-1))
        return rho, np.where(rho[..., None] > 0, d / np.where(rho > 0, rho, 1.0)[..., None], 0.0)

    def v(self, X):
        if self.interval and self.depth:
            return self.top - np.asarray(X[0], float)
        rho = self.rho_and_dir(X)[0]
        return self.top - rho if self.depth else rho

    def dv(self, X):
        """The gradient of v, shape (..., d)."""
        if self.interval and self.depth:
            return -np.ones(np.shape(X[0]) + (1,))
        u = self.rho_and_dir(X)[1]
        return -u if self.depth else u


def _contact_order(pl, pr, b):
    """-1 if the polynomials differ in value at b; else the largest m with equal derivatives of order 0..m (to rounding,
    16 eps of the size of the terms, so relative to the coefficients: there is no absolute floor, and a profile of tiny
    values keeps its jumps); math.inf if all derivatives agree (the polynomials are equal)."""
    for i in range(max(pl.degree(), pr.degree()) + 1):
        dl, dr = pl.deriv(i), pr.deriv(i)
        scale = sum(abs(c) * abs(b) ** j for q in (dl, dr) for j, c in enumerate(q.coef))
        if abs(dl(b) - dr(b)) > 16 * EPS * scale:
            return i - 1
    return math.inf


def _origin_order(q):
    """The order of f = q(rho) at rho = 0: the first odd power rho^k with a nonzero coefficient is C^(k-1) only."""
    for k in range(1, len(q.coef), 2):
        if q.coef[k] != 0.0:
            return k - 1
    return math.inf


def _glued(p, b, order, tail):
    """The coefficients of the polynomial that matches p at b in value and derivatives up to `order`, then continues with
    `tail` times (v - b)^(order + 1)."""
    local = Polynomial([p.deriv(i)(b) / math.factorial(i) for i in range(order + 1)] + [tail])
    return tuple(float(c) for c in local(Polynomial([-b, 1.0])).coef)


def _layered_examples(D):
    """Continuous glueings only: the contract's quadrature is not accurate to 1e-5 across a kink or a jump (the tests of
    testfields do the jumps and kinks with their own panel-wise integration); an interval symmetric jump is exact."""
    rad, dep = _Radial(D, "radius"), _Radial(D, "depth")
    span, dspan = rad.hi - rad.lo, dep.hi - dep.lo
    p0, q0 = Polynomial([1.0, 0.0, -1.0]), Polynomial([1.0, 2.0])
    mid, dmid = rad.lo + 0.5 * span, 0.4 * dspan
    out = [{"variable": "radius", "breaks": (mid,), "coefficients": ((1.0, 0.0, -1.0), _glued(p0, mid, 1, 1.5))},
           {"variable": "depth", "breaks": (dmid,), "coefficients": ((1.0, 2.0), _glued(q0, dmid, 2, 1.0))}]
    if geo.kind(D) == "interval":
        out.append({"variable": "depth", "breaks": (0.5 * dspan,), "coefficients": ((1.0,), (-1.0,))})
    else:
        out.append({"variable": "radius", "breaks": (rad.lo + 0.3 * span, rad.lo + 0.7 * span),
                    "coefficients": ((2.0,), _glued(Polynomial([2.0]), rad.lo + 0.3 * span, 3, -1.0),
                                     _glued(Polynomial(_glued(Polynomial([2.0]), rad.lo + 0.3 * span, 3, -1.0)),
                                            rad.lo + 0.7 * span, 3, 2.0))})
    return out


@register("layered_profile", ("ball", "shell", "disk", "interval"),
          "piecewise polynomial in the radius or the depth with given breakpoints and coefficients",
          references=(GRISVARD,), examples=_layered_examples)
def layered_profile(D, *, variable, breaks, coefficients):
    """f(x) = p_j(v) for b_(j-1) <= v < b_j (b_0 = lowest v, b_n = highest v on D), the layer j = 0..n for the n = len(breaks)
    breakpoints. `variable` is "radius" (v = |x - centre|; on an interval the distance from its midpoint) or "depth"
    (v = outer radius - |x - centre|; on an interval b - x). `breaks`: strictly increasing, strictly inside the range of v
    on D (empty: one layer). `coefficients`: one non-empty tuple per layer, ascending powers of v (absolute, not
    relative to the layer's start): p_j(v) = c_0 + c_1 v + c_2 v^2 + ...
    Support: the whole domain. Regularity (decided from the coefficients, to rounding): across each breakpoint the
    values and derivatives up to order m_j agree (m_j = -1: a jump), and at the centre (ball, disk, interval radius)
    an odd power rho^k of the first nonzero odd coefficient limits f to C^(k-1); the order is the least of these, and
    f is C-infinity when nothing limits it. There is no singularity at rho = 0 beyond that: a polynomial in rho is
    evaluated directly (no division by rho).
    Closed forms (exact polynomial integration of p_j(v) times the density of the measure in v): the integral and the
    squared L2 norm."""
    if variable not in ("radius", "depth"):
        raise Refused(f"layered_profile: variable must be 'radius' or 'depth' (got {variable!r})")
    R = _Radial(D, variable)
    try:
        edges = [float(b) for b in breaks]
        layers = [[float(c) for c in layer] for layer in coefficients]
    except (TypeError, ValueError):
        raise Refused("layered_profile: breaks must be a sequence of numbers and coefficients a sequence of "
                      "non-empty sequences of numbers") from None
    if not all(math.isfinite(x) for x in [*edges, *itertools.chain.from_iterable(layers)]):
        raise Refused("layered_profile: breaks and coefficients must be finite")
    if len(layers) != len(edges) + 1 or any(not layer for layer in layers):
        raise Refused(f"layered_profile: {len(edges)} breaks need {len(edges) + 1} non-empty coefficient tuples "
                      f"(got {len(layers)})")
    polys = [Polynomial(layer) for layer in layers]
    if any(b <= a for a, b in itertools.pairwise(edges)) or any(not R.lo < b < R.hi for b in edges):
        raise Refused(f"layered_profile: breaks must increase strictly and lie inside ({R.lo:g}, {R.hi:g}), the "
                      f"range of the {variable} on {D!r} (got {tuple(edges)})")
    cuts = [R.lo, *edges, R.hi]
    contact = [_contact_order(polys[j], polys[j + 1], b) for j, b in enumerate(edges)]
    q0 = R.in_rho(polys[-1 if R.depth else 0])
    origin = _origin_order(q0) if R.origin else math.inf
    order = min([*contact, origin])
    deriv = [p.deriv() for p in polys]
    breaks_arr = np.array(edges)

    def pick(X, family):
        v = np.asarray(R.v(X), float)
        flat = v.ravel()
        idx = np.searchsorted(breaks_arr, flat, side="right")
        out = np.empty(flat.shape)
        for j, p in enumerate(family):
            m = idx == j
            out[m] = p(flat[m])
        return out.reshape(v.shape)

    def fn(*X):
        return pick(X, polys)

    grad = None
    if order >= 0:
        def grad(*X):
            return pick(X, deriv)[..., None] * R.dv(X)

    def mass(square):
        return sum(float(((p * p if square else p) * R.dens).integ()(b) - ((p * p if square else p) * R.dens).integ()(a))
                   for p, a, b in zip(polys, cuts[:-1], cuts[1:]))

    why = "sum over the layers of the exact integral of p_j(v) (squared) times the density of the measure in v"
    closed = {"integral": (mass(False), DERIVED, why), "l2_norm_sq": (mass(True), DERIVED, why)}
    cite = (f"values and derivatives agree across the breakpoints up to the orders {[int(m) if m != math.inf else 'all' for m in contact]}"
            f" (-1: jump); at the centre the first odd power limits f to order {origin}; " + GRISVARD)
    reg = Regularity(math.inf) if order == math.inf else Regularity(int(order), piecewise=True)
    return BankFunction("layered_profile", D, {"variable": variable, "breaks": tuple(edges),
                                               "coefficients": tuple(tuple(float(c) for c in p.coef) for p in polys)},
                        fn, Support("full", "the whole domain (the layers cover it)"), reg, closed, grad,
                        reasons={"regularity": cite})


# ---------------------------------------------------------------- from testfields_blobs.py (plan 063)
ENTIRE = "the restriction of an entire function of the Cartesian coordinates"
SHAPES = {"bump": bump, "gaussian": gaussian}
DEFAULT_SIZE_RANGE, DEFAULT_AMPLITUDE_RANGE = (0.05, 0.25), (-1.0, 1.0)
TRIES = 2000


def _sample(D, rng):
    """A point of the domain drawn uniformly (rejection from the bounding box for the round flat domains)."""
    k = geo.kind(D)
    if k == "interval":
        return np.array([rng.uniform(D.a, D.b)])
    if k in ("box", "torus"):
        return np.array([rng.uniform(s.a, s.b) for s in D.sides])
    if k == "sphere":
        u = rng.normal(size=3)
        return np.asarray(D.centre, float) + D.radius * u / np.linalg.norm(u)
    if k == "circle":
        a = rng.uniform(0.0, 2 * math.pi)
        return np.asarray(D.centre, float) + D.radius * np.array([math.cos(a), math.sin(a)])
    box = D.bounding_box
    for _ in range(TRIES):
        c = np.array([rng.uniform(lo, hi) for lo, hi in box])
        if geo.room(D, c) > 0:
            return c
    raise Refused(f"blobs: no point of {D!r} found in {TRIES} draws")


def _pair(name, v, default, positive):
    lo, hi = default if v is None else v
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi and (lo > 0 or not positive)):
        raise Refused(f"blobs: {name} must be an increasing pair of finite numbers{' above 0' if positive else ''} "
                      f"(got {v!r})")
    return lo, hi


def _draw(D, shape, seed, count, size_range, amplitude_range):
    """count blobs from default_rng(seed): for each, a centre, a size (bumps: redrawn together with the centre until the
    ball fits inside the domain, with 5% to spare) and an amplitude, in that order."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise Refused(f"blobs: seed must be a non-negative integer (got {seed!r})")
    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 1:
        raise Refused(f"blobs: count must be a positive integer (got {count!r})")
    flo, fhi = _pair("size_range", size_range, DEFAULT_SIZE_RANGE, True)
    alo, ahi = _pair("amplitude_range", amplitude_range, DEFAULT_AMPLITUDE_RANGE, False)
    scale = geo.volume(D) ** (1.0 / geo.dimension(D))
    rng, centres, sizes, amps = np.random.default_rng(int(seed)), [], [], []
    for i in range(int(count)):
        for _ in range(TRIES):
            c, s = _sample(D, rng), rng.uniform(flo, fhi) * scale
            if shape == "gaussian" or s <= 0.95 * min(geo.room(D, c), geo.injectivity(D)):
                break
        else:
            raise Refused(f"blobs: blob {i} does not fit: sizes up to {fhi * scale:g} leave the domain in {TRIES} draws")
        centres.append(c)
        sizes.append(s)
        amps.append(rng.uniform(alo, ahi))
    return centres, sizes, amps


def _explicit(D, centres, sizes, amplitudes):
    try:
        C = [np.asarray(c, float).reshape(-1) for c in centres]
        s, a = [float(v) for v in sizes], [float(v) for v in amplitudes]
    except (TypeError, ValueError):
        raise Refused("blobs: centres, sizes and amplitudes must be sequences of numbers (centres: of points)") from None
    if not C or not (len(C) == len(s) == len(a)):
        raise Refused(f"blobs: centres, sizes and amplitudes must be non-empty and of one length "
                      f"({len(C)}, {len(s)}, {len(a)})")
    if any(c.size != geo.ambient(D) for c in C) or not all(np.all(np.isfinite(c)) for c in C) \
            or not all(math.isfinite(v) for v in a) or not all(math.isfinite(v) for v in s):
        raise Refused(f"blobs: every centre needs {geo.ambient(D)} finite coordinates, and every size and amplitude "
                      f"must be finite")
    return C, s, a


def _examples(D):
    """Seeded bumps sized to fit (three of them, one on a thin shell), two explicit bumps at one centre (equal radii: the contract's quadrature resolves an edge only at the end of its panel), seeded and single
    Gaussians (not on a torus). Positive amplitudes and overlapping sizes keep the contract's quadrature accurate."""
    k, scale = geo.kind(D), geo.volume(D) ** (1.0 / geo.dimension(D))
    rng = np.random.default_rng(0)
    c = _sample(D, rng)
    rmax = max(min(geo.room(D, p), geo.injectivity(D)) for p in (_sample(D, rng) for _ in range(300)))
    fr = min(0.4 * scale, 0.85 * rmax) / scale
    r = min(0.2 * scale, 0.5 * min(geo.room(D, c), geo.injectivity(D)))
    out = [{"seed": 7, "count": 1 if k == "shell" else 3, "size_range": (0.75 * fr, fr), "amplitude_range": (0.5, 1.5)},
           {"centres": (tuple(c), tuple(c)), "sizes": (r, r), "amplitudes": (1.0, 0.5)}]
    if k != "torus":
        out += [{"shape": "gaussian", "seed": 2, "count": 2, "size_range": (0.15, 0.3), "amplitude_range": (0.5, 1.5)},
                {"shape": "gaussian", "centres": (tuple(c),), "sizes": (0.2 * scale,), "amplitudes": (-2.0,)}]
    return out


@register("blobs", BUMP_KINDS, "a finite sum of bumps or Gaussians: given centres, sizes, amplitudes, or seeded random",
          references=(HORMANDER,), examples=_examples)
def blobs(D, *, shape="bump", centres=None, sizes=None, amplitudes=None, seed=None, count=None, size_range=None,
          amplitude_range=None):
    """f = sum_i amplitudes_i m_i, m_i the bump of radius sizes_i (shape "bump") or the Gaussian of width sizes_i (shape
    "gaussian") of functions/families/smooth.py about the point centres_i (refused when a member is: a bump must lie
    inside the domain). Give either centres, sizes and amplitudes (sequences of one length n >= 1; the centres points of
    the domain's ambient space) or seed (an integer >= 0) with count (n >= 1), and optionally
    size_range=(lo, hi) (the sizes are drawn uniformly in [lo, hi] times volume^(1/dimension), default (0.05, 0.25))
    and amplitude_range=(lo, hi) (default (-1, 1)). The random blobs are the stream of numpy.random.default_rng(seed):
    per blob a centre (uniform in the domain), a size (a bump is redrawn with its centre until its ball fits inside, 5% to
    spare) and an amplitude. The resolved centres, sizes and amplitudes are in the parameters.
    Support: bumps: the union of the balls (interior; with the enclosing ball when it lies inside the domain); Gaussians:
    the whole domain. Regularity: C-infinity (a finite sum of C-infinity functions); bumps vanish to all orders at the
    boundary. Closed forms: the integral is the sum of the members' integrals when each has one (linearity); for a single
    blob also the squared L2 norm (amplitude squared times the member's)."""
    if shape not in SHAPES:
        raise Refused(f"blobs: shape must be 'bump' or 'gaussian' (got {shape!r})")
    if seed is not None:
        if centres is not None or sizes is not None or amplitudes is not None:
            raise Refused("blobs: give either seed and count or centres, sizes and amplitudes, not both")
        C, s, a = _draw(D, shape, seed, count, size_range, amplitude_range)
    else:
        if count is not None or size_range is not None or amplitude_range is not None:
            raise Refused("blobs: count, size_range and amplitude_range belong to the seeded form (give a seed)")
        if centres is None or sizes is None or amplitudes is None:
            raise Refused("blobs: give either a seed with count or all of centres, sizes and amplitudes")
        C, s, a = _explicit(D, centres, sizes, amplitudes)
    members = []
    for i, (c, size) in enumerate(zip(C, s)):
        try:
            members.append(SHAPES[shape](D, centre=tuple(c), **({"radius": size} if shape == "bump" else {"width": size})))
        except Refused as e:
            raise Refused(f"blobs: blob {i}: {e}") from None
    n = len(members)

    def fn(*X):
        return sum(ai * m.fn(*X) for ai, m in zip(a, members))

    grad = None
    if all(m.gradient is not None for m in members):
        def grad(*X):
            return sum(ai * m.gradient(*X) for ai, m in zip(a, members))

    closed = {}
    if all("integral" in m.closed for m in members):
        closed["integral"] = (sum(ai * m.closed["integral"][0] for ai, m in zip(a, members)), DERIVED,
                              "sum of amplitude times the member's closed form (linearity)")
    if n == 1 and "l2_norm_sq" in members[0].closed:
        closed["l2_norm_sq"] = (a[0] ** 2 * members[0].closed["l2_norm_sq"][0], DERIVED, "amplitude^2 times the member's")
    if shape == "gaussian":
        support, reg, why = Support("full", "the whole domain"), Regularity(math.inf), ENTIRE
    else:
        radii = np.array(s)
        ball = {}
        if geo.kind(D) in FLAT:
            mean = np.mean(C, axis=0)
            R = float(np.max(np.linalg.norm(np.array(C) - mean, axis=1) + radii))
            if R < geo.room(D, mean):
                ball = {"centre": tuple(mean), "radius": R}
        support = Support("interior", f"the union of {n} closed geodesic balls", contains=lambda *X: np.any(
            [geo.distance(D, c, *X) <= r for c, r in zip(C, radii)], axis=0), **ball)
        reg, why = Regularity(math.inf, vanishes_at_boundary="all orders"), HORMANDER
    name = f"blobs({shape}, n={n}" + (f", seed={int(seed)})" if seed is not None else ")")
    return BankFunction("blobs", D, {"shape": shape, "seed": seed, "count": count if seed is not None else n,
                                     "centres": tuple(tuple(c) for c in C), "sizes": tuple(s),
                                     "amplitudes": tuple(a)}, fn, support, reg, closed, grad, name,
                        reasons={"regularity": why})


# ---------------------------------------------------------------- from testfields_planes.py (plan 063)
KINDS = ("box", "disk", "ball", "cylinder")
SMOOTH_WHY = ("tanh is real-analytic on the real line and d/dt tanh = 1 - tanh^2, so every derivative is a polynomial "
              "in tanh, bounded by induction: the function is C-infinity with bounded derivatives on the closure")


# --------------------------------------------------------------------------------------------- the geometry of a cut
def _unit(D, normal):
    try:
        n = np.asarray(normal, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused(f"normal must be a vector of {geo.ambient(D)} numbers (got {normal!r})") from None
    if n.size != geo.ambient(D) or not np.all(np.isfinite(n)) or np.linalg.norm(n) == 0.0:
        raise Refused(f"normal must be a finite nonzero vector of {geo.ambient(D)} coordinates (got {normal!r})")
    return n / np.linalg.norm(n)


def _half_extent(D, n):
    """E: the largest value of n . (x - centre) over D (the extremes are +-E: the centre is a point of symmetry)."""
    k = geo.kind(D)
    if k == "box":
        return 0.5 * float(sum(abs(ni) * (s.b - s.a) for ni, s in zip(n, D.sides)))
    if k == "cylinder":
        return D.radius * math.hypot(n[0], n[1]) + 0.5 * abs(n[2]) * D.z.length
    return D.radius


def _constant_section(D, n):
    """The cross-section area/length A of D orthogonal to n when it does not depend on n . (x - centre), else None."""
    k, E = geo.kind(D), _half_extent(D, n)
    if (k == "box" and np.count_nonzero(n) == 1) or (k == "cylinder" and n[0] == 0.0 and n[1] == 0.0):
        return geo.volume(D) / (2 * E)
    return None


def _segment(R, d):
    """The area of the part of the disk of radius R beyond the chord at signed distance d from the centre."""
    return R * R * math.acos(d / R) - d * math.sqrt(R * R - d * d)


def _box_below_exact(D, n, d):
    """The measure of the box D intersected with {n . (x - centre) <= d} as an exact Fraction of the float inputs:
    sum_S (-1)^|S| (t - sum_S m_i L_i)_+^k / (k! prod m_i), m = |n|, t = d + sum m_i L_i / 2, over the axes with m_i > 0
    (k of them), times the lengths of the others. The alternating sum cancels badly in floating point for a normal
    nearly along an axis (the terms are of order 1/prod m_i, the result of order 1), so it is done in rational
    arithmetic: no cutoff on the tilt is needed and the result is rounded once."""
    L = [Fraction(s.b) - Fraction(s.a) for s in D.sides]
    m = [abs(Fraction(float(x))) for x in n]
    live = [i for i, mi in enumerate(m) if mi > 0]
    t = Fraction(float(d)) + sum(m[i] * L[i] for i in live) / 2
    total = Fraction(0)
    for mask in range(1 << len(live)):
        sel = [i for j, i in enumerate(live) if (mask >> j) & 1]
        rest = max(t - sum((m[i] * L[i] for i in sel), Fraction(0)), Fraction(0))
        total += (-1) ** len(sel) * rest ** len(live)
    free = math.prod((L[i] for i in range(len(L)) if i not in live), start=Fraction(1))
    return free * total / (math.factorial(len(live)) * math.prod((m[i] for i in live), start=Fraction(1)))


def _below(D, n, d):
    """The measure of D intersected with {n . (x - centre) <= d}: a cap, a segment, an inclusion-exclusion sum for a
    box (_box_below_exact, in rational arithmetic), or None when the cut is not elementary."""
    k, V, E = geo.kind(D), geo.volume(D), _half_extent(D, n)
    if k == "box":
        return float(_box_below_exact(D, n, d))
    if d <= -E:
        return 0.0
    if d >= E:
        return V
    if k == "ball":
        return V - math.pi * (D.radius - d) ** 2 * (2 * D.radius + d) / 3
    if k == "disk":
        return V - _segment(D.radius, d)
    if n[2] == 0.0:
        h = math.hypot(n[0], n[1])
        return D.z.length * (math.pi * D.radius ** 2 - _segment(D.radius, d / h))
    if n[0] == 0.0 and n[1] == 0.0:
        return math.pi * D.radius ** 2 * min(max(d + 0.5 * D.z.length, 0.0), D.z.length)
    return None


def _slab(D, n, lo, hi):
    """The measure of D intersected with {lo < n . (x - centre) < hi}, or None when the cut is not elementary. For a box
    the difference is taken exactly (a thin slab would otherwise lose digits to the subtraction)."""
    if geo.kind(D) == "box":
        return float(_box_below_exact(D, n, hi) - _box_below_exact(D, n, lo))
    top, bottom = _below(D, n, hi), _below(D, n, lo)
    return None if top is None or bottom is None else top - bottom


def _through_plane(D, n, value, slope, knots, E):
    """int_D g(u) with u = n . (x - centre): by parts, g(E) V - int_(-E)^(E) g'(u) V(u) du with V(u) the exact measure of the side
    below u. Composite Gauss-Legendre between the knots, panels doubled until two results agree to 1e-12. None when V is not
    elementary or it does not converge."""
    if _below(D, n, 0.0) is None:
        return None
    breaks = np.unique(np.clip(np.r_[-E, knots, E], -E, E))
    t, w = gauss(32)
    below = np.vectorize(lambda u: float(_below(D, n, u)))
    previous = None
    for panels in (2, 4, 8, 16, 32, 64):
        total = 0.0
        for a, b in zip(breaks[:-1], breaks[1:]):
            edges = np.linspace(a, b, panels + 1)
            for lo, hi in zip(edges[:-1], edges[1:]):
                u = 0.5 * (hi - lo) * (t + 1) + lo
                total += 0.5 * (hi - lo) * float(np.sum(w * slope(u) * below(u)))
        current = value(E) * geo.volume(D) - total
        if previous is not None and abs(current - previous) <= 1e-12 * max(abs(current), abs(value(E)) * geo.volume(D) * 1e-3):
            return current
        previous = current
    return None


THROUGH_WHY = ("by parts over the plane's offset, with the exact measure of the side below it; composite Gauss-Legendre with doubled "
               "panels agreeing to 1e-12")


def _logcosh(x):
    ax = abs(x)
    return ax + math.log1p(math.exp(-2 * ax)) - math.log(2.0)


def _sech2(x):
    e = np.exp(-2 * np.abs(x))
    return 4 * e / (1 + e) ** 2


def _plane(D, normal, point, what):
    """(n, p, d0, E): the unit normal, the point of the plane, its signed offset from the centre along n, the half-extent."""
    n, c0 = _unit(D, normal), geo.centre(D)
    try:
        p = c0 if point is None else np.asarray(point, float).reshape(-1)
    except (TypeError, ValueError):
        raise Refused(f"{what}: point needs {geo.ambient(D)} finite coordinates (got {point!r})") from None
    if p.size != geo.ambient(D) or not np.all(np.isfinite(p)):
        raise Refused(f"{what}: point needs {geo.ambient(D)} finite coordinates (got {point!r})")
    return n, p, float(n @ (p - c0)), _half_extent(D, n)


def _signed(n, p, X):
    P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
    return (P - p) @ n


def _number(name, v):
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        raise Refused(f"{name} must be a real number (got {v!r})") from None


def _positive(name, v):
    x = _number(name, v)
    if not (math.isfinite(x) and x > 0):
        raise Refused(f"{name} must be positive and finite (got {v!r})")
    return x


def _real(name, v):
    x = _number(name, v)
    if not math.isfinite(x):
        raise Refused(f"{name} must be finite (got {v!r})")
    return x


# --------------------------------------------------------------------------------------------- step
def _axis(D):
    """A normal whose plane through the centre the contract's quadrature integrates exactly by symmetry: the x axis of a
    box, the polar axis of a ball and a cylinder; for a disk a direction half a node spacing (pi/160) off the y axis, so
    that no node of the 160 equispaced angles lies on the plane."""
    k = geo.kind(D)
    if k == "disk":
        return (-math.sin(math.pi / 160), math.cos(math.pi / 160))
    e = np.zeros(geo.ambient(D))
    e[0 if k == "box" else -1] = 1.0
    return tuple(e)


def _step_examples(D):
    size = geo.volume(D) ** (1.0 / geo.dimension(D))
    e, d = _axis(D), geo.ambient(D)
    diag = tuple([1.0, 2.0] + [0.0] * (d - 2))
    c = geo.centre(D)
    shifted = tuple(c + 0.1 * size * np.array(e))
    out = [{"normal": e}, {"normal": diag, "lower": 0.5, "upper": 2.0, "width": 0.15 * size},
           {"normal": e, "point": shifted, "width": 0.1 * size}]
    if geo.kind(D) == "cylinder":
        out.append({"normal": (1.0, 0.5, 0.3), "point": tuple(c + 0.05 * size), "lower": 1.0, "upper": 3.0})
    return out


@register("step_across_plane", KINDS, "a jump (sharp) or tanh transition (smooth) across a plane with a given normal",
          references=(GRISVARD,), examples=_step_examples)
def step_across_plane(D, *, normal, point=None, lower=-1.0, upper=1.0, width=None):
    """f = lower + (upper - lower) s(t) with t = n . (x - point) the signed distance to the plane through `point`
    (default: the domain's centre) with unit normal n = normal/|normal| (`normal`: a nonzero vector, no default).
    width None (sharp): s = 1 for t > 0, 0 for t < 0, 1/2 on the plane; width > 0 (smooth): s = (1 + tanh(t/width))/2.
    lower != upper are the two plateau values (defaults -1 and 1); the plane must cut the interior of the domain.
    Support: the whole domain, except a sharp step with lower or upper = 0 (the closed half-space of the other side,
    which meets the boundary). Regularity: sharp: bounded with a jump across the plane (order -1); smooth: C-infinity.
    Closed forms. The plane through the centre (a point of symmetry): f(x) + f(2c - x) = lower + upper, so the integral is
    (lower + upper) V / 2 (sharp and smooth). Sharp: integral = lower V_- + upper V_+ and l2_norm_sq = lower^2 V_- +
    upper^2 V_+ with the volume V_- of the side t < 0 (cap, segment, inclusion-exclusion; see _below). Smooth, with a
    constant cross-section A orthogonal to n (box with an axis normal; cylinder with the axis as normal): the integral
    is lower V + (upper - lower) A (E + (width/2)[ln cosh((E - d)/width) - ln cosh((-E - d)/width)]), E the half-extent
    and d the offset of the plane from the centre (integrate tanh)."""
    n, p, d0, E = _plane(D, normal, point, "step_across_plane")
    lower, upper = _real("lower", lower), _real("upper", upper)
    if lower == upper:
        raise Refused("step_across_plane: lower and upper are equal, there is no step")
    smooth = width is not None
    w = _positive("width", width) if smooth else None
    if abs(d0) >= E:
        raise Refused(f"step_across_plane: the plane misses the interior of the domain (offset {d0:g} from the centre, "
                      f"half-extent {E:g} along the normal)")
    V, jump = geo.volume(D), upper - lower

    def fn(*X):
        t = _signed(n, p, X)
        s = 0.5 * (1 + np.tanh(t / w)) if smooth else np.where(t > 0, 1.0, np.where(t < 0, 0.0, 0.5))
        return lower + jump * s

    grad = None
    if smooth:
        def grad(*X):
            return (jump / (2 * w) * _sech2(_signed(n, p, X) / w))[..., None] * n

    closed, sym = {}, abs(d0) <= 1e-12 * E
    if sym:
        closed["integral"] = ((lower + upper) * V / 2, DERIVED,
                              "the plane passes through the centre of symmetry: f(x) + f(2c - x) = lower + upper")
    if not smooth:
        vm = V / 2 if sym else _below(D, n, d0)
        if vm is not None:
            closed["integral"] = (lower * vm + upper * (V - vm), DERIVED, "lower V_- + upper V_+, volumes of the two sides")
            closed["l2_norm_sq"] = (lower ** 2 * vm + upper ** 2 * (V - vm), DERIVED, "lower^2 V_- + upper^2 V_+")
    elif not sym and (A := _constant_section(D, n)) is not None:
        mass = E + 0.5 * w * (_logcosh((E - d0) / w) - _logcosh((-E - d0) / w))
        closed["integral"] = (lower * V + jump * A * mass, DERIVED,
                              "constant cross-section A: A int_(-E)^(E) of lower + jump (1 + tanh((u - d)/width))/2 du")
    elif not sym:
        knots = d0 + w * np.array([-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0])
        mass = _through_plane(D, n, lambda u: lower + jump * 0.5 * (1 + math.tanh((u - d0) / w)),
                              lambda u: jump / (2 * w) * _sech2((u - d0) / w), knots, E)
        if mass is not None:
            closed["integral"] = (mass, CHECKED, THROUGH_WHY)
    support = Support("full", "the whole domain")
    if not smooth and (lower == 0.0 or upper == 0.0):
        sign = 1.0 if lower == 0.0 else -1.0
        support = Support("touches", "the closed half-space of the nonzero side, cut by the domain",
                          contains=lambda *X: sign * _signed(n, p, X) >= 0)
    reg = Regularity(math.inf) if smooth else Regularity(-1, piecewise=True)
    why = SMOOTH_WHY if smooth else "constant on each side of the plane, jumping by upper - lower; " + GRISVARD
    return BankFunction("step_across_plane", D, {"normal": tuple(n), "point": tuple(p), "lower": lower, "upper": upper,
                                                 "width": w}, fn, support, reg, closed, grad,
                        reasons={"regularity": why})


# --------------------------------------------------------------------------------------------- dyke
def _dyke_examples(D):
    size = geo.volume(D) ** (1.0 / geo.dimension(D))
    e, d, c = _axis(D), geo.ambient(D), geo.centre(D)
    tilt = tuple([1.0, 1.0] + [0.0] * (d - 2))
    out = [{"normal": e, "thickness": 0.3 * size, "width": 0.05 * size},
           {"normal": e, "thickness": 0.2 * size, "point": tuple(c + 0.15 * size * np.array(e)), "amplitude": 2.0,
            "background": 0.5, "width": 0.08 * size},
           {"normal": tilt, "thickness": 0.25 * size, "width": 0.07 * size}]
    if geo.kind(D) == "cylinder":
        out.append({"normal": (1.0, 0.5, 0.3), "thickness": 0.2 * size, "amplitude": -1.5})
    return out


@register("dyke", KINDS, "a slab of given thickness and normal (sharp indicator or tanh-smoothed) on a background",
          references=(GRISVARD,), examples=_dyke_examples)
def dyke(D, *, normal, thickness, point=None, amplitude=1.0, background=0.0, width=None):
    """f = background + amplitude g(t), t = n . (x - point) the signed distance to the mid-plane of the slab through
    `point` (default: the domain's centre) with unit normal n = normal/|normal|, h = thickness/2. width None (sharp):
    g = 1 for |t| < h, 0 for |t| > h, 1/2 on the faces; width > 0 (smooth): g = (tanh((t + h)/width) - tanh((t - h)/width))/2,
    which tends to the indicator as width -> 0 (its peak is tanh(h/width) < 1). `amplitude` is nonzero (default 1),
    `background` the value away from the dyke (default 0). The slab must meet the interior of the domain.
    Support: sharp with background 0: the closed slab cut by the domain (it meets the boundary); else the whole domain.
    Regularity: sharp: order -1 (jumps across the two faces); smooth: C-infinity.
    Closed forms. Sharp: with V_s = _slab(d - h, d + h) the volume of the slab in D, integral = background V +
    amplitude V_s and l2_norm_sq = background^2 (V - V_s) + (background + amplitude)^2 V_s. Smooth with a constant
    cross-section A orthogonal to n (box with an axis normal, cylinder with the axis as normal): integral = background V +
    amplitude A (width/2) [ln cosh((u - d + h)/width) - ln cosh((u - d - h)/width)] evaluated from u = -E to u = E,
    d the offset of the mid-plane from the centre and E the half-extent."""
    n, p, d0, E = _plane(D, normal, point, "dyke")
    h = 0.5 * _positive("thickness", thickness)
    amplitude, background = _real("amplitude", amplitude), _real("background", background)
    if amplitude == 0.0:
        raise Refused("dyke: amplitude must be nonzero")
    smooth = width is not None
    w = _positive("width", width) if smooth else None
    if d0 - h >= E or d0 + h <= -E:
        raise Refused(f"dyke: the slab (offset {d0:g}, half-thickness {h:g}) misses the domain (half-extent {E:g})")
    V = geo.volume(D)

    def fn(*X):
        t = _signed(n, p, X)
        if smooth:
            g = 0.5 * (np.tanh((t + h) / w) - np.tanh((t - h) / w))
        else:
            g = np.where(np.abs(t) < h, 1.0, np.where(np.abs(t) > h, 0.0, 0.5))
        return background + amplitude * g

    grad = None
    if smooth:
        def grad(*X):
            t = _signed(n, p, X)
            return (amplitude / (2 * w) * (_sech2((t + h) / w) - _sech2((t - h) / w)))[..., None] * n

    closed = {}
    if not smooth:
        vs = _slab(D, n, d0 - h, d0 + h)
        if vs is not None:
            closed["integral"] = (background * V + amplitude * vs, DERIVED, "background V + amplitude (volume of the slab in D)")
            closed["l2_norm_sq"] = (background ** 2 * (V - vs) + (background + amplitude) ** 2 * vs, DERIVED,
                                    "the values are background and background + amplitude")
    elif (A := _constant_section(D, n)) is not None:
        def prim(u):
            return 0.5 * w * (_logcosh((u - d0 + h) / w) - _logcosh((u - d0 - h) / w))
        closed["integral"] = (background * V + amplitude * A * (prim(E) - prim(-E)), DERIVED,
                              "constant cross-section A: A int_(-E)^(E) g(u - d) du, int tanh = width ln cosh")
    else:
        knots = np.r_[d0 - h + w * np.array([-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0]), d0 + h + w * np.array([-16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0])]
        mass = _through_plane(D, n, lambda u: background + amplitude * 0.5 * (math.tanh((u - d0 + h) / w) - math.tanh((u - d0 - h) / w)),
                              lambda u: amplitude / (2 * w) * (_sech2((u - d0 + h) / w) - _sech2((u - d0 - h) / w)), knots, E)
        if mass is not None:
            closed["integral"] = (mass, CHECKED, THROUGH_WHY)
    support = Support("full", "the whole domain")
    if not smooth and background == 0.0:
        support = Support("touches", f"the closed slab |t| <= {h:g} cut by the domain",
                          contains=lambda *X: np.abs(_signed(n, p, X)) <= h)
    reg = Regularity(math.inf) if smooth else Regularity(-1, piecewise=True)
    why = SMOOTH_WHY if smooth else "constant outside and inside the slab, jumping across its two faces; " + GRISVARD
    return BankFunction("dyke", D, {"normal": tuple(n), "thickness": 2 * h, "point": tuple(p), "amplitude": amplitude,
                                    "background": background, "width": w}, fn, support, reg, closed, grad,
                        reasons={"regularity": why})
