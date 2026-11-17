"""Compactly supported radial basis functions (plan 021, WP-B; EXP-037 DESIGN.md 5, 6).

    wendland(D, centre=None, radius=None, k=1, d=None)   Wendland's psi_{d,k}(t) = I^k (1-t)_+^l, l = floor(d/2) + k + 1,
                                                         t = distance / radius, value 1 at the centre: C^(2k)
    wu(D, centre=None, radius=None, ell=2, k=1)            Wu's psi_{l,k}: D^k of the autoconvolution of (1-x^2)_+^l: C^(2(l-k))
    buhmann(D, centre=None, radius=None)                 12 t^4 log t - 21 t^4 + 32 t^3 - 12 t^2 + 1 (Buhmann type): C^2

All three are polynomial profiles (Buhmann: with one logarithm) of the geodesic distance, supported in the closed
geodesic ball of `radius` inside the domain (at positive distance from the boundary, as for `bump`). On the sphere and
the circle the centre must lie on the manifold (| |c - o| - R | <= 1e-9 R) and the geodesic distance is smooth in normal
coordinates for radius < pi R; on the torus the centre must lie in the fundamental box and the minimum-image distance is
the Euclidean distance within a ball of radius < half the shortest period, which `injectivity` guards. The regularity
claims hold there; positive definiteness is claimed only for the Euclidean kinds, where the cited theorems are about R^m. The exact
polynomial coefficients come from wendland_poly.py. Positive definiteness is an extra claim ("positive_definite"),
carried by PDFunction, a BankFunction whose claims list adds it.

From wendland_poly.py (merged here by plan 063):
Exact polynomial profiles of the compactly supported radial functions (plan 021, WP-B; no family is registered here).

A profile is a polynomial in t = distance / radius on [0, 1], stored with exact rational coefficients as
(m, p): profile(t) = (1 - t)^m p(t), p a list of Fractions in increasing powers. The factorised form keeps the accuracy
near t = 1, where the profile is a tiny difference of large terms in the expanded form.

    wendland_profile(d, k)   Wendland's psi_{d,k}: I^k (1-t)_+^l with l = floor(d/2) + k + 1, scaled to value 1 at 0
                             (I f(t) = int_t^1 s f(s) ds)
    wu_profile(l, k)         Wu's psi_{l,k}: D^k of the autoconvolution of (1-x^2)_+^l, D f = -f'(t)/t, support [0, 2]
                             rescaled to [0, 1], value 1 at 0
    horner(p, t)             evaluation in floats
"""


import math
import operator
from fractions import Fraction
from functools import cache
from math import comb

import numpy as np

from ...core.claims import CHECKED, DERIVED, THEOREM, Claim, Claims
from ...core.errors import Refused
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register


# ---------------------------------------------------------------- from wendland_poly.py (plan 063)
def _trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _add(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def _scale(p, c):
    return [c * a for a in p]


def derivative(p):
    """Coefficients of p'."""
    return _trim([i * a for i, a in enumerate(p)][1:] or [Fraction(0)])


def divide_by_t(p):
    """p / t for p(0) = 0 (the radial derivative over the radius, a polynomial when the profile is flat at 0)."""
    if p[0] != 0:
        raise ValueError("p(0) != 0: not divisible by t")
    return p[1:] or [Fraction(0)]


def _divide_one_minus(p):
    """(q, True) with p = (1 - t) q when p(1) = 0, else (p, False)."""
    if sum(p) != 0:
        return p, False
    q = [Fraction(0)] * (len(p) - 1)
    acc = Fraction(0)
    for i in range(len(p) - 1, 0, -1):          # p = (1 - t) q: q_{i-1} = -(p_i - ... ) built from the top
        acc = acc - p[i]
        q[i - 1] = acc
    return _trim(q), True


def factorise(p):
    """(m, q) with p = (1 - t)^m q and q(1) != 0."""
    m, q = 0, _trim(p)
    while True:
        r, ok = _divide_one_minus(q)
        if not ok or all(a == 0 for a in q):
            return m, q
        m, q = m + 1, r


def horner(p, t):
    """The polynomial p (coefficients in increasing powers) at the float array t."""
    t = np.asarray(t, float)
    out = np.zeros_like(t) + float(p[-1])
    for a in reversed(p[:-1]):
        out = out * t + float(a)
    return out


def expanded(m, p):
    """The coefficients of (1 - t)^m p(t)."""
    base = [Fraction((-1) ** j * comb(m, j)) for j in range(m + 1)]
    return _mul(base, p)


def _integrate_t_times(p):
    """The coefficients of Q(t) = int_t^1 s p(s) ds."""
    anti = [Fraction(0), Fraction(0)] + [a / (i + 2) for i, a in enumerate(p)]             # int_0^t s p(s) ds
    return _add([sum(anti)], _scale(anti, -1))


@cache
def wendland_profile(d, k):
    """(m, p, ell): Wendland's psi_{d,k}(t) = (1 - t)^m p(t) with m = ell + k, deg p = k, psi(0) = 1."""
    ell = d // 2 + k + 1
    poly = [Fraction((-1) ** j * comb(ell, j)) for j in range(ell + 1)]
    for _ in range(k):
        poly = _trim(_integrate_t_times(poly))
    m, q = factorise(_scale(poly, 1 / poly[0]))
    return m, tuple(q), ell


def _bivariate_power(base, n):
    """base^n for a dict {(i, j): c} in x^i r^j."""
    out = {(0, 0): Fraction(1)}
    for _ in range(n):
        new = {}
        for (a, b), c in out.items():
            for (e, f), g in base.items():
                new[(a + e, b + f)] = new.get((a + e, b + f), Fraction(0)) + c * g
        out = new
    return out


@cache
def _autoconvolution(l):
    """g(r) = int (1-x^2)^l (1-(r-x)^2)^l dx over [r-1, 1], r in [0, 2], as a polynomial in r."""
    first = _bivariate_power({(0, 0): Fraction(1), (2, 0): Fraction(-1)}, l)                    # (1 - x^2)^l
    second = _bivariate_power({(0, 0): Fraction(1), (0, 2): Fraction(-1), (1, 1): Fraction(2), (2, 0): Fraction(-1)}, l)
    prod = {}
    for (a, b), c in first.items():
        for (e, f), g in second.items():
            prod[(a + e, b + f)] = prod.get((a + e, b + f), Fraction(0)) + c * g
    total = [Fraction(0)]
    for (i, j), c in prod.items():                    # int x^i dx = x^(i+1)/(i+1); at 1 and at r - 1
        at_one = [Fraction(0)] * j + [c / (i + 1)]
        shifted = [Fraction((-1) ** (i + 1 - n) * comb(i + 1, n)) for n in range(i + 2)]        # (r - 1)^(i+1)
        at_low = _scale([Fraction(0)] * j + shifted, c / (i + 1))
        total = _add(total, _add(at_one, _scale(at_low, -1)))
    return tuple(_trim(total))


@cache
def wu_profile(l, k):
    """(m, p, expanded) for Wu's psi_{l,k} on t = r/2: psi(t) = (1 - t)^m p(t), psi(0) = 1; 0 <= k <= l."""
    g = list(_autoconvolution(l))
    poly = [a * Fraction(2) ** i for i, a in enumerate(g)]                    # g(2t)
    for _ in range(k):
        poly = _trim(divide_by_t(_scale(derivative(poly), -1)))
    poly = _scale(poly, 1 / poly[0])
    m, q = factorise(poly)
    return m, tuple(q), tuple(poly)


FLAT = ("interval", "box", "disk", "ball", "shell", "cylinder")
KINDS = FLAT + ("sphere", "circle", "torus")
ON_MANIFOLD = 1e-9          # relative tolerance for a sphere/circle centre: | |c - o| - R | <= ON_MANIFOLD * R
RADIAL = "radial reduction of the integral over a geodesic ball, composite Gauss-Legendre (8 x 64 nodes) in r"
WENDLAND_1995 = (
    ("Wendland 1995, Piecewise polynomial, positive definite and compactly supported radial functions of minimal "
     "degree, Adv. Comput. Math. 4, 389-396"),
    ("Wendland 2005, Scattered Data Approximation, ch. 9 (psi_{d,k} is strictly positive definite on R^d, a "
     "polynomial of degree floor(d/2) + 3k + 1 on [0, 1], in C^(2k))"))
WU_1995 = (
    "Wu 1995, Compactly supported positive definite radial functions, Adv. Comput. Math. 4, 283-292",
    ("Wendland 2005, Scattered Data Approximation, ch. 9 (Wu's functions; the operator D = -(1/t) d/dt maps a "
     "positive definite function on R^m to one on R^(m+2))"))
MAX_WHY = ("|psi(t)| <= psi(0) = 1: the profile is positive definite on R^1 (Bochner), a property of the profile alone, "
           "so it bounds the function on every kind")
BUHMANN_1998 = (
    ("Buhmann 1998, Radial functions on compact support, Proc. Edinb. Math. Soc. 41, 33-46 (compactly supported "
     "positive definite radial functions with logarithmic terms)"),)


class PDFunction(BankFunction):
    """A BankFunction that also states the positive definiteness of its radial profile (`extra`: a tuple of Claims)."""

    def __init__(self, *args, extra=(), **kw):
        super().__init__(*args, **kw)
        self.extra = tuple(extra)

    @property
    def claims(self):
        return Claims(tuple(super().claims) + self.extra)

    def scaled(self, c, name=None):
        out = super().scaled(c, name)
        closed = {key: val for key, val in out.closed.items() if key != "max" or float(c) > 0}   # max(-f) is not |c| max f
        return PDFunction(out.family, out.domain, out.params, out.fn, out.support, out.regularity, closed,
                          out.gradient, out.name, out.reasons, extra=self.extra if float(c) > 0 else ())


# ----------------------------------------------------------------------------------------------- parameters
def _centre(D, centre):
    c = geo.centre(D) if centre is None else np.asarray(centre, float).reshape(-1)
    if c.size != geo.ambient(D) or not np.all(np.isfinite(c)):
        raise Refused(f"a centre on {D!r} has {geo.ambient(D)} finite Cartesian coordinates (got {centre!r})")
    k = geo.kind(D)
    if k in ("sphere", "circle"):
        o = np.asarray(D.centre, float)
        off = float(np.linalg.norm(c - o))
        if abs(off - D.radius) > ON_MANIFOLD * D.radius:
            raise Refused(f"a centre on a {k} of radius {D.radius:g} must lie on it: its distance {off:g} from the "
                          f"centre {tuple(o)} differs from the radius (got {centre!r})")
        c = o + (c - o) * (D.radius / off)                 # remove the rounding of the input: exactly on the manifold
    elif k == "torus":
        tol = 1e-12
        if any(not (s.a - tol <= x <= s.b + tol) for x, s in zip(c, D.sides)):
            raise Refused(f"a centre on a torus must lie in the fundamental box {[(s.a, s.b) for s in D.sides]} "
                          f"(got {centre!r}); the minimum-image distance is taken from a centre in the box")
    return c


def _integer(family, name, value, lo, hi):
    if isinstance(value, bool):
        raise Refused(f"{family}: {name} must be an integer in [{lo}, {hi}] (got {value!r})")
    try:
        n = operator.index(value)
    except TypeError:
        raise Refused(f"{family}: {name} must be an integer in [{lo}, {hi}] (got {value!r})") from None
    if not lo <= n <= hi:
        raise Refused(f"{family}: {name} must be an integer in [{lo}, {hi}] (got {n})")
    return n


def _support_radius(family, D, c, radius):
    room, inj = geo.room(D, c), geo.injectivity(D)
    if radius is None:
        radius = 0.5 * min(room, inj)
    if isinstance(radius, (bool, np.bool_)):
        raise Refused(f"{family}: radius must be a positive number, not a bool (got {radius!r})")
    try:
        radius = float(radius)
    except (TypeError, ValueError):
        raise Refused(f"{family}: radius must be a positive number (got {radius!r})") from None
    if not (math.isfinite(radius) and radius > 0):
        raise Refused(f"{family}: radius must be positive and finite (got {radius})")
    if radius >= room or radius >= inj:
        raise Refused(f"{family}: radius {radius:g} leaves the domain (room to the boundary {room:g}, largest geodesic "
                      f"ball {inj:g}); the support is the closed geodesic ball of that radius and lies inside")
    return radius


def _sphere_note(D):
    if geo.kind(D) in FLAT:
        return ""
    if geo.kind(D) == "torus":
        return ("; on the torus the profile of the minimum-image distance is the same function of the Euclidean "
                "distance (the ball of radius < half the shortest period is isometric to a Euclidean ball)")
    return ("; on the " + geo.kind(D) + " the profile of the geodesic distance is the same function in normal "
            "coordinates (the exponential map is a diffeomorphism from the ball of radius < pi R)")


def _examples(D):
    c = geo.centre(D)
    room = min(geo.room(D, c), geo.injectivity(D))
    return c, room


# ----------------------------------------------------------------------------------------------- the common build
def _radial(D, family, c, radius, params, phi, dphi_over_t, order, extra, reasons):
    """The bank function t -> phi(t) of t = distance / radius (phi supported in [0, 1], phi(0) = 1, |phi| <= 1)."""
    def profile(r):
        t = np.asarray(r, float) / radius
        out = np.zeros_like(t)
        m = t < 1
        out[m] = phi(t[m])
        return out

    def fn(*X):
        return profile(geo.distance(D, c, *X))

    grad = None
    if dphi_over_t is not None and geo.kind(D) in FLAT:
        def grad(*X):
            P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
            t = np.linalg.norm(P - c, axis=-1) / radius
            s = np.zeros_like(t)
            m = t < 1
            s[m] = dphi_over_t(t[m])
            return s[..., None] * (P - c) / radius ** 2

    closed = {"integral": (geo.radial_mass(D, profile, radius), DERIVED, RADIAL),
              "l2_norm_sq": (geo.radial_mass(D, lambda r: profile(r) ** 2, radius), DERIVED, RADIAL)}
    closed["max"] = (1.0, DERIVED, reasons["max"])
    sup = Support("interior", f"the closed geodesic ball of radius {radius:g} about {tuple(round(float(x), 6) for x in c)}",
                  tuple(c), radius, breaks=())
    reg = Regularity(order, piecewise=(order == 0), vanishes_at_boundary="all orders")
    why = {"regularity": reasons["regularity"],
           "support": f"t = distance/radius < 1 inside; room to the boundary {geo.room(D, c):g} > radius {radius:g}"}
    return PDFunction(family, D, params, fn, sup, reg, closed, grad, reasons=why, extra=extra)


def _poly_pieces(m, p):
    """phi and phi'/t for phi(t) = (1 - t)^m p(t); phi'/t is None when phi has a cone point at 0 (phi'(0) != 0)."""
    poly = expanded(m, list(p))

    def phi(t):
        return (1.0 - t) ** m * horner(p, t)
    dpoly = derivative(poly)
    if dpoly[0] != 0:
        return phi, None, poly
    m2, q2 = factorise(divide_by_t(dpoly))

    def dphi_over_t(t):
        return (1.0 - t) ** m2 * horner(q2, t)
    return phi, dphi_over_t, poly


def _pd_claim(family, D, m, reason, status=THEOREM):
    if geo.kind(D) not in FLAT:
        return ()
    return (Claim(f"the kernel (x, y) -> {family} profile(|x - y|/radius) is positive definite on R^m for m <= {m}",
                  status, reason, key="positive_definite"),)


# ----------------------------------------------------------------------------------------------- Wendland
def _wendland_examples(D):
    c, room = _examples(D)
    # k = 0 with d = ambient + 2: the edge contact (1-t)^3 is smooth enough for the contract's reference quadrature
    # (for (1-t)^2 in 3-D that quadrature itself is only accurate to 3e-7; the closed form is exact)
    # the torus has no flat-bounding-box reference in the contract: its fine domain quadrature is accurate to 1e-5 only
    # for profiles without the centre cone of k = 0, so the torus takes k = 1 here
    k0, r0 = (1, 0.5) if geo.kind(D) == "torus" else (0, 0.7)
    return [{}, {"k": k0, "d": geo.ambient(D) + 2, "radius": r0 * room}, {"k": 2, "radius": 0.3 * room, "d": 3},
            {"centre": tuple(c), "k": 3, "radius": 0.6 * room, "d": 1}]


@register("wendland", KINDS, "Wendland's compactly supported psi_{d,k}(t) = I^k (1-t)_+^l, C^(2k), t = distance/radius",
          references=WENDLAND_1995, examples=_wendland_examples)
def wendland(D, *, centre=None, radius=None, k=1, d=None):
    """Wendland's function psi_{d,k}(t), t = distance / radius, psi(0) = 1: psi_{d,k} = I^k phi_l with phi_l(t) =
    (1 - t)_+^l, l = floor(d/2) + k + 1, (I f)(t) = int_t^inf s f(s) ds. On [0, 1] it is (1 - t)^(l+k) p(t) with p a
    polynomial of degree k (d = 3: (1-t)^2, (1-t)^4 (4t+1), (1-t)^6 (35t^2+18t+3)/3, (1-t)^8 (32t^3+25t^2+8t+1)).

    Parameters (keyword-only): `centre` (default: the domain's centre); `radius`, the support radius (default: half the
    room to the boundary; the closed geodesic ball must lie inside the domain); `k` in 0..3 (default 1) the smoothness
    index; `d` in 1..30 (default: the ambient dimension of the domain) the dimension it is built for.
    Support: the closed geodesic ball (centre, radius). Regularity: C^(2k) (k = 0: continuous and Lipschitz, a
    cone at the centre, piecewise smooth); the first odd power of t at 0 is t^(2k+1), so no more at the centre; vanishes to order
    l + k >= 2k + 1 at t = 1. On spheres and circles the same holds in normal coordinates (radius < pi R). Positive
    definiteness (Euclidean kinds): strictly positive definite on R^m for every m <= d."""
    c = _centre(D, centre)
    k = _integer("wendland", "k", k, 0, 3)
    d = geo.ambient(D) if d is None else _integer("wendland", "d", d, 1, 30)
    radius = _support_radius("wendland", D, c, radius)
    m, p, _ = wendland_profile(d, k)
    phi, dphi, _ = _poly_pieces(m, p)
    reg = (WENDLAND_1995[0] + f"; psi_{{{d},{k}}} = (1-t)^{m} p(t), deg p = {k}, in C^({2 * k})" + _sphere_note(D))
    pd = _pd_claim("wendland", D, d, WENDLAND_1995[1] + "; Askey 1973 (sufficiency: phi_l is PD on R^n for "
                   "l >= floor(n/2)+1; the converse is a later result and is not used) and the dimension walk (I maps "
                   "PD on R^(n+2) to PD on R^n); the restriction of a PD function to R^m, m <= d, is PD")
    return _radial(D, "wendland", c, radius, {"centre": tuple(c), "radius": radius, "k": k, "d": d}, phi,
                   dphi, 2 * k, pd,
                   {"regularity": reg, "max": MAX_WHY})


# ----------------------------------------------------------------------------------------------- Wu
def _wu_examples(D):
    c, room = _examples(D)
    return [{}, {"ell": 1, "k": 0, "radius": 0.7 * room}, {"ell": 3, "k": 1, "radius": 0.4 * room},
            {"centre": tuple(c), "ell": 2, "k": 2, "radius": 0.6 * room}]


@register("wu", KINDS, "Wu's compactly supported psi_{l,k} = D^k((1-x^2)_+^l * (1-x^2)_+^l), C^(2(l-k)), PD on R^(2k+1)",
          references=WU_1995, examples=_wu_examples)
def wu(D, *, centre=None, radius=None, ell=2, k=1):
    """Wu's function psi_{l,k}: psi_l(r) = int (1-x^2)_+^l (1-(r-x)^2)_+^l dx (support [0, 2]), psi_{l,k} = D^k psi_l with
    D f(r) = -f'(r)/r, scaled so that the support radius is `radius` (t = distance / radius) and psi(0) = 1.
    l = 2, k = 0: (1-t)^5 (1+5t+9t^2+5t^3+t^4); l = 2, k = 1: (1-t)^4 (4+16t+12t^2+3t^3)/4.

    Parameters (keyword-only): `centre`, `radius` as for wendland; ell in 1..6 (default 2) and k in 0..ell
    (default 1); the profile is psi_{ell,k}.
    Regularity: C^(2(l-k)) (k = l: continuous, Lipschitz, piecewise smooth): the first odd power of t at 0 is
    t^(2(l-k)+1); the profile vanishes to order 2l+1-k >= l+1 at t = 1. Positive definiteness (Euclidean kinds):
    positive definite on R^m for every m <= 2k+1 (and not on R^(2k+3) in general: see the tests)."""
    c = _centre(D, centre)
    ell = _integer("wu", "ell", ell, 1, 6)
    k = _integer("wu", "k", k, 0, ell)
    radius = _support_radius("wu", D, c, radius)
    m, p, _ = wu_profile(ell, k)
    phi, dphi, _ = _poly_pieces(m, p)
    order = 2 * (ell - k)
    reg = (WU_1995[0] + f"; psi_{{{ell},{k}}} in C^({order}) (the first odd power of t at 0 is t^{order + 1}, vanishing "
           f"order {m} at t = 1)" + _sphere_note(D))
    pd = _pd_claim("wu", D, 2 * k + 1, WU_1995[1] + "; psi_l is the autoconvolution of an even function, so its 1-D "
                   "Fourier transform is |F f|^2 >= 0, and D^k raises the dimension by 2 per application")
    return _radial(D, "wu", c, radius, {"centre": tuple(c), "radius": radius, "ell": ell, "k": k}, phi,
                   dphi, order, pd,
                   {"regularity": reg, "max": MAX_WHY})


# ----------------------------------------------------------------------------------------------- Buhmann
def _buhmann_phi(t):
    t = np.asarray(t, float)
    safe = np.where(t > 0, t, 1.0)
    return 12 * t ** 4 * np.log(safe) - 21 * t ** 4 + 32 * t ** 3 - 12 * t ** 2 + 1


def _buhmann_dphi_over_t(t):
    t = np.asarray(t, float)
    safe = np.where(t > 0, t, 1.0)
    return -24 + 96 * t - 72 * t ** 2 + 48 * t ** 2 * np.log(safe)


def _buhmann_examples(D):
    c, room = _examples(D)
    return [{}, {"radius": 0.7 * room}, {"centre": tuple(c), "radius": 0.3 * room}]


@register("buhmann", KINDS, "Buhmann-type 12 t^4 log t - 21 t^4 + 32 t^3 - 12 t^2 + 1 on [0, 1], C^2, PD on R^3 (checked)",
          references=BUHMANN_1998, examples=_buhmann_examples)
def buhmann(D, *, centre=None, radius=None):
    """The compactly supported function phi(t) = 12 t^4 log t - 21 t^4 + 32 t^3 - 12 t^2 + 1 on t = distance / radius in
    [0, 1], 0 beyond (phi(0) = 1; phi, phi', phi'', phi''' vanish at t = 1, phi'''' = 96 there).

    Parameters (keyword-only): `centre`, `radius` as for wendland. Regularity: C^2: phi = 1 - 12t^2 + 32t^3 + O(t^4 log
    t) at 0, so the third derivative of the radial function jumps at the centre (t^3 = |x|^3 is C^2 only), while the
    contact with 0 at t = 1 is of order 4 (phi''' = 192 - 192t + 288 t log t). It decreases from 1 to 0: phi'(t)/t =
    h(t) = -24 + 96t - 72t^2 + 48 t^2 log t has h(1) = 0 and h'(t) = 96 (1 - t + t log t) >= 0, so h <= 0.
    Positive definiteness (Euclidean kinds): of the type constructed in Buhmann 1998; for this function it is only
    checked here, not proved: the radial Fourier transform in R^1, R^2, R^3 is positive on the frequencies sampled in
    the tests and its large-frequency term (from 32 t^3) is positive. Not claimed beyond R^3 (negative in R^5)."""
    c = _centre(D, centre)
    radius = _support_radius("buhmann", D, c, radius)
    reg = (BUHMANN_1998[0] + "; the expansion at 0 and the contact order at t = 1 are derived in the docstring (C^2 on "
           "the whole space, C^3 at the edge)" + _sphere_note(D))
    pd = _pd_claim("buhmann", D, 3, "numerical positivity of the radial Fourier transforms on R^1..R^3 (tests); "
                   "no proof recorded", status=CHECKED)
    return _radial(D, "buhmann", c, radius, {"centre": tuple(c), "radius": radius}, _buhmann_phi,
                   _buhmann_dphi_over_t, 2, pd,
                   {"regularity": reg, "max": "phi decreases on [0, 1] (phi'/t = h <= 0, see the docstring), phi(0) = 1"})
