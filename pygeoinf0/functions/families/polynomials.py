"""Polynomial families on intervals and boxes (plan 021 P1, work package D; EXP-037 DESIGN.md 5, 6).

    monomial(D, *, exponents, origin=None)   prod_i (x_i - o_i)^e_i                       interval, box
    legendre(D, *, index, norm="standard")   prod_i P_{n_i}(t_i), t_i = (2 x_i - a_i - b_i)/(b_i - a_i)   interval, box
    chebyshev(D, *, index)                   prod_i T_{n_i}(t_i), the first kind                     interval, box

Index layouts (explicit; they never depend on how many functions are used): `exponents` and `index` are tuples with one
non-negative integer per axis, in the order of the coordinates (x, y[, z]); on an interval a bare integer is the
one-tuple. [a_i, b_i] is the i-th side of the box. `norm="standard"` is P_n(1) = 1 (the classical normalisation);
`norm="orthonormal"` multiplies by prod_i sqrt((2 n_i + 1)/(b_i - a_i)), so that the squared L2 norm over the domain
(Lebesgue measure) is 1. The helpers `index_tuple`, `axes` and `tensor` are shared with the other polynomial modules.

From polynomials_angular.py (merged here by plan 063):
Orthogonal bases with angular structure (plan 021 P1, work package D; EXP-037 DESIGN.md 6).

    zernike(D, *, n, m, norm="standard")                 real Zernike polynomials on a disk
    spherical_harmonic(D, *, degree, j, convention=...)  real spherical harmonics on a sphere, the package's labels

Zernike layout: n >= 0 is the radial degree, m the signed angular index with |m| <= n and n - |m| even; m >= 0 is the
cosine mode, m < 0 the sine mode of |m|: Z_n^m = R_n^|m|(rho) cos(m phi), Z_n^-m = R_n^m(rho) sin(m phi) (m > 0),
rho = |x - c|/R, phi the angle about the centre c from the x-axis. Spherical harmonic layout: exactly
spaces/harmonic.py (packed label (degree l, j), 0 <= j <= 2 l; j = 0 zonal, j = 2m - 1 the cos(m phi) and
j = 2m the sin(m phi) member of order m), orthonormal on the unit sphere, no Condon-Shortley phase by default.

From polynomials_fourier.py (merged here by plan 063):
Real Fourier modes (plan 021 P1, work package D; EXP-037 DESIGN.md 6).

    fourier(D, *, index, norm="standard")    interval, box, circle, torus

Index layout (explicit, independent of how many modes are used): `index` has one signed integer k per axis (one for an
interval and a circle; a bare integer is the one-tuple). On an axis of period L starting at a (the side [a, a + L] of
an interval, box or torus; the circle's angle phi from the x-axis about its centre, L = 2 pi), with t = (x - a)/L:

    k = 0: 1        k > 0: cos(2 pi k t)        k < 0: sin(2 pi |k| t)

and the mode is the product over the axes. These are the real modes of the periodic extension; on an interval or a box
they are ordinary smooth functions of the closed domain, orthogonal in L2 (not periodic-boundary conditions).
"""


import math
import numbers

import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import legendre as leg
from scipy.special import eval_jacobi

from ...core.claims import DERIVED
from ...core.errors import Refused
from ...spaces import harmonic as harmonics
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register

MAX_DEGREE = 60
POLY_KINDS = ("interval", "box")
POLYNOMIAL = "a polynomial in the Cartesian coordinates, hence C-infinity (and entire) on the closed domain"
REF_AS = "Abramowitz & Stegun 1964, Handbook of Mathematical Functions, ch. 22 (orthogonal polynomials)"
REF_SZEGO = "Szego 1939, Orthogonal Polynomials, 2.1 and 3.3 (Legendre and Chebyshev polynomials)"


# --------------------------------------------------------------------------------------------- shared helpers
def index_tuple(value, length, name, low=0, high=MAX_DEGREE):
    """`value` as a tuple of `length` integers in [low, high] (a bare integer is the one-tuple), else Refused."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        value = (value,)
    try:
        items = tuple(value)
    except TypeError:
        raise Refused(f"{name} must be a tuple of {length} integers (got {value!r})") from None
    if len(items) != length:
        raise Refused(f"{name} must have one integer per coordinate axis ({length}), got {len(items)}: {value!r}")
    for v in items:
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise Refused(f"{name} must contain integers (got {v!r})")
        if not low <= v <= high:
            raise Refused(f"{name} entries must lie in [{low}, {high}] (got {int(v)})")
    return tuple(int(v) for v in items)


def axes(D):
    """[(a_i, b_i)] of the sides of an interval, box or torus (bounded), in coordinate order."""
    sides = [D] if geo.kind(D) == "interval" else list(D.sides)
    out = [(float(s.a), float(s.b)) for s in sides]
    if not all(math.isfinite(a) and math.isfinite(b) for a, b in out):
        raise Refused(f"the sides of {D!r} must be bounded")
    return out


def tensor(axis_fns, scale=1.0):
    """(fn, gradient) of scale * prod_i g_i(x_i) from per-axis callables g_i(x) -> (value, derivative)."""
    def parts(X):
        return [g(np.asarray(x, float)) for g, x in zip(axis_fns, X)]

    def fn(*X):
        out = scale
        for v, _ in parts(X):
            out = out * v
        return out

    def gradient(*X):
        vs, cols = parts(X), []
        for i in range(len(vs)):
            term = scale * vs[i][1]
            for j, (v, _) in enumerate(vs):
                if j != i:
                    term = term * v
            cols.append(np.broadcast_to(term, np.broadcast(*X).shape))
        return np.stack(cols, axis=-1)
    return fn, gradient


def _norm(norm):
    if norm not in ("standard", "orthonormal"):
        raise Refused(f"norm must be 'standard' or 'orthonormal' (got {norm!r})")
    return norm


def _tensor_function(family, D, params, axis_fns, scale, closed, reasons, vanishes=None):
    fn, grad = tensor(axis_fns, scale)
    return BankFunction(family, D, params, fn,
                        Support("full", "the whole domain (a polynomial vanishes on no open set unless it is zero)"),
                        Regularity(math.inf, vanishes_at_boundary=vanishes), closed, grad, reasons=reasons)


# --------------------------------------------------------------------------------------------- monomial
def _monomial_examples(D):
    d = geo.dimension(D)
    c = tuple(float(v) for v in geo.centre(D))
    sets = {1: [(0,), (3,), (5,)], 2: [(0, 0), (1, 2), (3, 1)], 3: [(1, 0, 2), (2, 2, 1)]}[d]
    return [{"exponents": sets[0]}, {"exponents": sets[1], "origin": c}] + [{"exponents": s} for s in sets[2:]]


@register("monomial", POLY_KINDS, "monomial prod_i (x_i - o_i)^e_i with an exponent tuple e and origin o",
          references=("Stein & Shakarchi 2003, Complex Analysis, 1.2 (polynomials are entire)",),
          examples=_monomial_examples)
def monomial(D, *, exponents, origin=None):
    """The monomial prod_i (x_i - o_i)^e_i of the Cartesian coordinates, e = `exponents` (one integer in [0, 60] per
    axis, in the order of the coordinates) about the `origin` o (default: the zero vector, i.e. the plain monomial
    x^e; give the domain's centre for a centred monomial). 0^0 = 1.

    Support: the whole domain. Regularity: C-infinity (a polynomial), no vanishing claim. Closed forms (elementary):
    integral = prod_i ((b_i - o_i)^(e_i + 1) - (a_i - o_i)^(e_i + 1))/(e_i + 1) and l2_norm_sq the same with 2 e_i.
    """
    sides = axes(D)
    e = index_tuple(exponents, len(sides), "exponents")
    if origin is None:
        o = (0.0,) * len(sides)
    else:
        try:
            o = tuple(float(v) for v in np.asarray(origin, float).reshape(-1))
        except (TypeError, ValueError):
            raise Refused(f"origin must be {len(sides)} real numbers (got {origin!r})") from None
        if len(o) != len(sides) or not all(math.isfinite(v) for v in o):
            raise Refused(f"origin must be {len(sides)} finite real numbers (got {origin!r})")

    def axis(ei, oi):
        def g(x):
            u = x - oi
            return u ** ei, (ei * u ** (ei - 1) if ei > 0 else np.zeros_like(u))
        return g

    def mass(p):
        v = 1.0
        for (a, b), ei, oi in zip(sides, e, o):
            v *= ((b - oi) ** (p * ei + 1) - (a - oi) ** (p * ei + 1)) / (p * ei + 1)
        return v

    why = "prod_i of the elementary integrals of (x - o_i)^(p e_i) over [a_i, b_i]"
    closed = {"integral": (mass(1), DERIVED, why), "l2_norm_sq": (mass(2), DERIVED, why + ", p = 2")}
    return _tensor_function("monomial", D, {"exponents": e, "origin": o}, [axis(ei, oi) for ei, oi in zip(e, o)], 1.0,
                            closed, {"regularity": POLYNOMIAL + "; Stein & Shakarchi 2003, Complex Analysis, 1.2"})


# --------------------------------------------------------------------------------------------- legendre
def _poly_axis(val, der, n, a, b):
    """g(x) = (value, derivative) of the degree-n polynomial (val: legval or chebval, der: legder or chebder) on [a, b]
    mapped affinely to [-1, 1]."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    dc = der(c)
    s = 2.0 / (b - a)

    def g(x):
        t = s * (x - a) - 1.0
        return val(t, c), s * val(t, dc)
    return g


def _legendre_examples(D):
    d = geo.dimension(D)
    idx = {1: [0, 2, 5], 2: [(0, 0), (2, 1), (1, 3)], 3: [(1, 0, 2), (2, 2, 1)]}[d]
    return [{"index": idx[0]}, {"index": idx[1], "norm": "orthonormal"}] + [{"index": i} for i in idx[2:]]


@register("legendre", POLY_KINDS, "Legendre polynomial P_n of an interval mapped to [a, b]; tensor products on boxes",
          references=(REF_AS, REF_SZEGO), examples=_legendre_examples)
def legendre(D, *, index, norm="standard"):
    """prod_i c_i P_{n_i}(t_i) with t_i = (2 x_i - a_i - b_i)/(b_i - a_i) in [-1, 1], n = `index` (one integer in
    [0, 60] per axis; a bare integer on an interval), P_n(1) = 1. norm = "standard": c_i = 1; "orthonormal":
    c_i = sqrt((2 n_i + 1)/(b_i - a_i)), so that the L2 norm over D (Lebesgue measure) is 1.

    Support: the whole domain. Regularity: C-infinity (a polynomial), no vanishing claim. Closed forms (orthogonality
    int_{-1}^{1} P_n P_k = 2 delta_nk/(2n + 1), A&S 22.2.10, with P_0 = 1; the affine map gives the factor
    (b_i - a_i)/2 per axis): integral = prod_i (b_i - a_i) [n_i = 0] (times prod c_i), l2_norm_sq =
    prod_i c_i^2 (b_i - a_i)/(2 n_i + 1), max = prod_i c_i (|P_n| <= 1 on [-1, 1], P_n(1) = 1: A&S 22.14.7).
    """
    sides, norm = axes(D), _norm(norm)
    n = index_tuple(index, len(sides), "index")
    c = 1.0
    if norm == "orthonormal":
        for (a, b), ni in zip(sides, n):
            c *= math.sqrt((2 * ni + 1) / (b - a))
    zero = all(ni == 0 for ni in n)
    length = math.prod(b - a for a, b in sides)
    why = "orthogonality of the Legendre polynomials (A&S 22.2.10), P_0 = 1, and the affine map to [-1, 1]"
    closed = {"integral": (c * length if zero else 0.0, DERIVED, why),
              "l2_norm_sq": (c * c * math.prod((b - a) / (2 * ni + 1) for (a, b), ni in zip(sides, n)), DERIVED, why),
              "max": (c, DERIVED, "|P_n| <= 1 on [-1, 1] with P_n(1) = 1 (A&S 22.14.7): attained at the upper corner")}
    axis_fns = [_poly_axis(leg.legval, leg.legder, ni, a, b) for (a, b), ni in zip(sides, n)]
    return _tensor_function("legendre", D, {"index": n, "norm": norm}, axis_fns, c, closed,
                            {"regularity": POLYNOMIAL + "; " + REF_AS + ", 22.3"})


# --------------------------------------------------------------------------------------------- chebyshev
def _chebyshev_examples(D):
    d = geo.dimension(D)
    idx = {1: [0, 1, 4], 2: [(0, 0), (2, 1), (3, 3)], 3: [(1, 0, 2), (2, 2, 1)]}[d]
    return [{"index": i} for i in idx]


@register("chebyshev", POLY_KINDS, "Chebyshev polynomial of the first kind T_n of an interval mapped to [a, b]; tensor "
          "products on boxes", references=(REF_AS, REF_SZEGO), examples=_chebyshev_examples)
def chebyshev(D, *, index):
    """prod_i T_{n_i}(t_i), T_n(cos u) = cos(n u), t_i = (2 x_i - a_i - b_i)/(b_i - a_i), n = `index` (one integer in
    [0, 60] per axis; a bare integer on an interval). The T_n are orthogonal for the weight (1 - t^2)^(-1/2)
    (constants pi for n = 0, pi/2 otherwise); the closed forms below are for Lebesgue measure on D.

    Support: the whole domain. Regularity: C-infinity (a polynomial), no vanishing claim. Closed forms (substituting
    t = cos u): int_{-1}^{1} T_n dt = 2/(1 - n^2) for even n, 0 for odd n; int_{-1}^{1} T_n^2 dt = 1 + 1/(1 - 4 n^2);
    each axis contributes the factor (b_i - a_i)/2; max = 1 (|T_n| <= 1, T_n(1) = 1, at the upper corner).
    """
    sides = axes(D)
    n = index_tuple(index, len(sides), "index")
    integral = math.prod(0.5 * (b - a) * (2.0 / (1 - ni * ni) if ni % 2 == 0 else 0.0) for (a, b), ni in zip(sides, n))
    l2 = math.prod(0.5 * (b - a) * (1.0 + 1.0 / (1 - 4 * ni * ni)) for (a, b), ni in zip(sides, n))
    why = "t = cos u: int T_n dt = int_0^pi cos(n u) sin u du, int T_n^2 dt = int_0^pi cos^2(n u) sin u du; affine map"
    closed = {"integral": (integral, DERIVED, why), "l2_norm_sq": (l2, DERIVED, why),
              "max": (1.0, DERIVED, "T_n(cos u) = cos(n u), so |T_n| <= 1 on [-1, 1] and T_n(1) = 1")}
    axis_fns = [_poly_axis(cheb.chebval, cheb.chebder, ni, a, b) for (a, b), ni in zip(sides, n)]
    return _tensor_function("chebyshev", D, {"index": n}, axis_fns, 1.0, closed,
                            {"regularity": POLYNOMIAL + "; " + REF_AS + ", 22.3"})


# ---------------------------------------------------------------- from polynomials_angular.py (plan 063)
MAX_ZERNIKE, MAX_HARMONIC_DEGREE = 40, 40
REF_BORN = "Born & Wolf 1999, Principles of Optics, 7th ed., 9.2 and Appendix VII (Zernike polynomials)"
REF_DT = "Dahlen & Tromp 1998, Theoretical Global Seismology, Appendix C (real spherical harmonics)"


# --------------------------------------------------------------------------------------------- zernike
def _zernike_examples(D):
    return [{"n": 0, "m": 0}, {"n": 2, "m": 0, "norm": "orthonormal"}, {"n": 3, "m": 1}, {"n": 4, "m": -2}]


@register("zernike", ("disk",), "real Zernike polynomial Z_n^m = R_n^|m|(rho) cos(m phi) or sin(|m| phi) on a disk",
          references=(REF_BORN, REF_AS), examples=_zernike_examples)
def zernike(D, *, n, m, norm="standard"):
    """The real Zernike polynomial of radial degree n and signed angular index m on the disk (centre c, radius R):
    Z = R_n^|m|(rho) cos(m phi) for m >= 0 and R_n^|m|(rho) sin(|m| phi) for m < 0, where rho = |x - c|/R and
    R_n^a(rho) = (-1)^k rho^a P_k^(a,0)(1 - 2 rho^2), k = (n - a)/2 (Jacobi polynomial), R_n^a(1) = 1. Requires
    0 <= n <= 40, |m| <= n and n - |m| even. norm = "standard": as above (max |Z| = 1 on the disk is not claimed);
    "orthonormal": multiplied by sqrt((n + 1)/(pi R^2)) (m = 0) or sqrt(2 (n + 1)/(pi R^2)) (m != 0), so that the
    squared L2 norm over the disk is 1.

    Evaluated as the polynomial (-1)^k P_k^(a,0)(1 - 2 s) Re/Im((dx + i dy)/R)^a, s = rho^2, which is exact at the
    centre (no angle is taken). Support: the whole disk. Regularity: C-infinity (a polynomial in x and y). Closed
    forms (derived): int_0^1 (R_n^a)^2 rho d rho = 1/(2 (n + 1)) and the angular integral of cos^2 / sin^2 give
    l2_norm_sq = R^2 pi/(n + 1) (m = 0), R^2 pi/(2 (n + 1)) (m != 0); the integral is pi R^2 for n = 0 and 0 otherwise
    (m != 0: the angular integral vanishes; m = 0, n = 2k: int_0^1 R_n^0 rho d rho = (1/4) int_-1^1 P_k = delta_k0/2).
    """
    norm = _norm(norm)
    (n,) = index_tuple(n, 1, "n", 0, MAX_ZERNIKE)
    (m,) = index_tuple(m, 1, "m", -MAX_ZERNIKE, MAX_ZERNIKE)
    a = abs(m)
    if a > n or (n - a) % 2:
        raise Refused(f"zernike needs |m| <= n and n - |m| even (got n = {n}, m = {m})")
    k, (cx, cy), R = (n - a) // 2, D.centre, float(D.radius)
    scale = 1.0
    if norm == "orthonormal":
        scale = math.sqrt((n + 1) / (math.pi * R * R)) * (1.0 if m == 0 else math.sqrt(2.0))
    sine = m < 0
    sign = -1.0 if k % 2 else 1.0

    def w(x, y):
        return ((np.asarray(x, float) - cx) + 1j * (np.asarray(y, float) - cy)) / R

    def part(z):
        return z.imag if sine else z.real

    def fn(x, y):
        z = w(x, y)
        return scale * sign * eval_jacobi(k, a, 0, 1.0 - 2.0 * np.abs(z) ** 2) * part(z ** a)

    def gradient(x, y):
        z = w(x, y)
        s = np.abs(z) ** 2
        g = sign * eval_jacobi(k, a, 0, 1.0 - 2.0 * s)
        dg = -sign * (k + a + 1) * eval_jacobi(k - 1, a + 1, 1, 1.0 - 2.0 * s) if k > 0 else np.zeros_like(s)
        # d/dx, d/dy of s = |z|^2 are 2 Re z/R, 2 Im z/R; of Re z^a: (Re, -Im)(a z^(a-1))/R; of Im z^a: (Im, Re)(...)/R
        zp = a * z ** (a - 1) if a > 0 else np.zeros_like(z)
        dwx, dwy = (zp.imag, zp.real) if sine else (zp.real, -zp.imag)
        W = part(z ** a)
        gx = dg * 2.0 * z.real / R * W + g * dwx / R
        gy = dg * 2.0 * z.imag / R * W + g * dwy / R
        return scale * np.stack([gx, gy], axis=-1)

    angular = 2.0 if m == 0 else 1.0
    why = ("int_0^1 (R_n^a)^2 rho d rho = 1/(2 (n + 1)) (Born & Wolf, Appendix VII) and int_0^2pi cos^2(m phi) dphi = "
           "pi (2 pi for m = 0)")
    closed = {"integral": (scale * math.pi * R * R if n == 0 else 0.0, DERIVED,
                           ("m != 0: the angular integral vanishes; m = 0, n = 2k: int_0^1 R_n^0 rho d rho = "
                            "(1/4) int_-1^1 P_k(t) dt = delta_k0/2 (orthogonality to P_0)")),
              "l2_norm_sq": (scale ** 2 * R * R * math.pi * angular / (2 * (n + 1)), DERIVED, why)}
    return BankFunction("zernike", D, {"n": n, "m": m, "norm": norm}, fn,
                        Support("full", "the whole disk (a nonzero polynomial vanishes on no open set)"),
                        Regularity(math.inf), closed, gradient,
                        reasons={"regularity": POLYNOMIAL + "; " + REF_BORN})


# --------------------------------------------------------------------------------------------- spherical harmonics
def _convention(convention):
    if isinstance(convention, harmonics.HarmonicConvention):
        return convention
    if convention == "package":
        return harmonics.PACKAGE
    if convention in ("condon-shortley", "thesis"):
        return harmonics.THESIS
    raise Refused(f"convention must be 'package', 'condon-shortley' or a HarmonicConvention (got {convention!r})")


def _harmonic_examples(D):
    return [{"degree": 0, "j": 0}, {"degree": 1, "j": 0}, {"degree": 3, "j": 2},
            {"degree": 2, "j": 3, "convention": "condon-shortley"}]


@register("spherical_harmonic", ("sphere",), "real spherical harmonic Y_lj in the package's packed labels (l, j)",
          references=(REF_DT, "spaces/harmonic.py (the package's conventions)"), examples=_harmonic_examples)
def spherical_harmonic(D, *, degree, j, convention="package"):
    """The real spherical harmonic of degree l = `degree` and packed position j on the sphere (centre c, radius R),
    evaluated at the direction (x - c)/|x - c| with spaces/harmonic.py: Y_l0 = p_l0(cos theta), Y_l,2m-1 =
    sqrt 2 p_lm(cos theta) cos(m phi), Y_l,2m = sqrt 2 p_lm(cos theta) sin(m phi), orthonormal on the unit sphere,
    0 <= l <= 40, 0 <= j <= 2 l. convention: "package" (default), "condon-shortley" (the thesis basis, (-1)^m Y) or
    a HarmonicConvention (a signed rescaling of the package's basis).

    Support: the whole sphere. Regularity: C-infinity (the restriction of a harmonic polynomial). Closed forms
    (derived; f = factor_lj Y_lj, factor = the convention's sign times scale): l2_norm_sq = factor^2 R^2 (orthonormality
    on the unit sphere, surface measure R^2 dOmega); integral = factor R^2 sqrt(4 pi) for l = 0 (Y_00 = 1/sqrt(4 pi))
    and 0 for l >= 1 (orthogonality to Y_00).
    """
    (l,) = index_tuple(degree, 1, "degree", 0, MAX_HARMONIC_DEGREE)
    (jj,) = index_tuple(j, 1, "j", 0, 2 * l)
    conv, R = _convention(convention), float(D.radius)
    centre = np.asarray(D.centre, float)
    col = int(harmonics.index(l, jj))
    factor = float(conv.factors(l, jj))

    def fn(*X):
        P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1) - centre
        shape = P.shape[:-1]
        return factor * harmonics.values(l, P.reshape(-1, 3))[:, col].reshape(shape)

    why = "orthonormality of the real spherical harmonics on the unit sphere (spaces/harmonic.py), area R^2 dOmega"
    closed = {"integral": (factor * R * R * math.sqrt(4 * math.pi) if l == 0 else 0.0, DERIVED,
                           "Y_00 = 1/sqrt(4 pi) is the constant; the other Y_lj are orthogonal to it"),
              "l2_norm_sq": (factor ** 2 * R * R, DERIVED, why)}
    params = {"degree": l, "j": jj, "convention": convention if isinstance(convention, str) else conv.name}
    return BankFunction("spherical_harmonic", D, params, fn, Support("full", "the whole sphere"),
                        Regularity(math.inf), closed, None,
                        reasons={"regularity": "the restriction to the sphere of a harmonic polynomial of degree l; " + REF_DT})


# ---------------------------------------------------------------- from polynomials_fourier.py (plan 063)
MAX_MODE = 40
REF = "Katznelson 2004, An Introduction to Harmonic Analysis, I.1 (the trigonometric system)"
REF_ZYG = "Zygmund 2002, Trigonometric Series, I.4 (orthogonality of cos and sin on a period)"
ENTIRE = "cos and sin of an affine function of the coordinates: entire, hence C-infinity on the closed domain"


def _axis(k, a, L):
    """g(x) = (value, derivative) of the mode k on the axis starting at a with period L."""
    w = 2.0 * math.pi * abs(k) / L

    def g(x):
        u = w * (np.asarray(x, float) - a)
        if k == 0:
            return np.ones_like(u), np.zeros_like(u)
        if k > 0:
            return np.cos(u), -w * np.sin(u)
        return np.sin(u), w * np.cos(u)
    return g


def _examples(D):
    d = geo.dimension(D)
    idx = {1: [0, 1, -2, 3], 2: [(0, 0), (1, 0), (-1, 2), (2, -3)], 3: [(0, 0, 0), (1, -1, 0), (2, 1, -1)]}[d]
    out = [{"index": i} for i in idx[:-1]]
    return out + [{"index": idx[-1], "norm": "orthonormal"}]


@register("fourier", ("interval", "box", "circle", "torus"), "real Fourier mode (1, cos 2 pi k t, sin 2 pi |k| t per "
          "axis, signed index k) on an interval, box, circle or torus", references=(REF, REF_ZYG), examples=_examples)
def fourier(D, *, index, norm="standard"):
    """The product over the axes of the real Fourier modes with the signed `index` k (module docstring; |k_i| <= 40):
    k = 0 gives 1, k > 0 gives cos(2 pi k t), k < 0 gives sin(2 pi |k| t), t = (x - a)/L on the side [a, a + L], and on
    a circle (centre c, radius R) the angle phi = atan2(y - c_y, x - c_x) in [-pi, pi] replaces 2 pi t (a mode of the
    angle, k integer, so the branch is irrelevant). norm = "standard": amplitude 1; "orthonormal": divided by the root
    of l2_norm_sq below, so that the L2 norm over the domain (Lebesgue / arc-length measure) is 1.

    Support: the whole domain. Regularity: C-infinity. A sine mode in every axis (all k_i < 0) vanishes on the whole
    boundary of an interval or box (claimed order 1: the function itself); other modes make no vanishing claim, and
    circles and tori have no boundary. Closed forms (orthogonality on a full period, Zygmund I.4): each axis of period L
    contributes L (k = 0) or L/2 (k != 0) to l2_norm_sq, the integral is prod L_i when all k_i = 0 and 0 otherwise,
    and max = 1 (standard) when all k_i >= 0 (attained at the lower corner; for sine modes max is not claimed).
    """
    norm = _norm(norm)
    kind = geo.kind(D)
    if kind == "circle":
        n = index_tuple(index, 1, "index", -MAX_MODE, MAX_MODE)
        weights = [2.0 * math.pi * D.radius]            # the arc length of the circle
    else:
        sides = axes(D)
        n = index_tuple(index, len(sides), "index", -MAX_MODE, MAX_MODE)
        weights = [b - a for a, b in sides]             # the periods
    sq = math.prod(w if ki == 0 else 0.5 * w for w, ki in zip(weights, n))
    scale = 1.0 / math.sqrt(sq) if norm == "orthonormal" else 1.0
    zero = all(ki == 0 for ki in n)
    why = "orthogonality of 1, cos, sin over a full period (Zygmund I.4); the measure of each axis is its period"
    closed = {"integral": (scale * math.prod(weights) if zero else 0.0, DERIVED, why),
              "l2_norm_sq": (scale ** 2 * sq, DERIVED, why)}
    if all(ki >= 0 for ki in n):
        closed["max"] = (scale, DERIVED, "cos <= 1 with equality at the lower corner (k = 0: constant 1)")
    reasons = {"regularity": ENTIRE + "; " + REF}
    if kind == "circle":
        cx, cy = D.centre
        (k,) = n

        def fn(x, y):
            phi = np.arctan2(np.asarray(y, float) - cy, np.asarray(x, float) - cx)
            return scale * (np.ones_like(phi) if k == 0 else np.cos(k * phi) if k > 0 else np.sin(-k * phi))
        grad, vanishes = None, None
    else:
        fn, grad = tensor([_axis(ki, a, L) for ki, (a, _), L in zip(n, sides, weights)], scale)
        vanishes = 1 if kind in ("interval", "box") and all(ki < 0 for ki in n) else None
    return BankFunction("fourier", D, {"index": n, "norm": norm}, fn,
                        Support("full", "the whole domain"), Regularity(math.inf, vanishes_at_boundary=vanishes),
                        closed, grad, reasons=reasons)
