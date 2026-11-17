"""Eigenfunctions of the Laplacian with boundary conditions (plan 021 P1, work package E; EXP-037 DESIGN.md 6).

    laplace_eigen(D, ...)    one family on every kind it is defined on, dispatched by the kind of D (plan 070 P6: the
                             name no longer repeats the domain, review E2); its parameters are the kind's:
        interval [a, b]  n, bc, beta     Dirichlet, Neumann, DN, ND, Robin
        box              n, bc, beta     tensor product of interval eigenfunctions, one index per axis
        disk             n, m, trig, bc  (below, from eigenfunctions_radial.py)
        ball             n, l, j, bc

Every function u solves -Laplace u = lambda u in the domain with the stated boundary condition and is normalised in
L2(D) (the Lebesgue measure): ||u||^2 = 1. The eigenvalue lambda is on record as the closed form "eigenvalue".

One interval mode covers every condition. With x' = x - a, L = b - a and a phase phi_a in [0, pi/2] at a,

    u(x) = N cos(k x' - phi_a),     tan(phi_a) = u'(a) / (k u(a)),

so phi_a = pi/2 is the Dirichlet and phi_a = 0 the Neumann condition at a; likewise phi_b at b, where the condition
holds when k L - phi_a = phi_b + (g - 1) pi for an integer g >= 1 (the g-th eigenvalue, lambda = k^2). Robin conditions (outward normal
derivative + beta u = 0, beta >= 0) mean tan(phi_a) = beta_a / k and tan(phi_b) = beta_b / k: the phases depend on k and
the equation k L = arctan(beta_a / k) + arctan(beta_b / k) + (g - 1) pi (the left side increases, the right side
decreases in k) has exactly one root in (0, g pi / L], found by Brent's method to machine precision. Then

    N^-2 = int_0^L cos^2(k x' - phi_a) dx' = L/2 + (sin 2 phi_a + sin 2 phi_b) / (4 k)       (k > 0)
    int_0^L u dx = N [(-1)^(g-1) sin phi_b + sin phi_a] / k                                  (k > 0)

and k = 0 (Neumann at both ends, g = 1, or Robin with both betas 0) is the constant 1 / sqrt(L).

From eigenfunctions_radial.py (merged here by plan 063):
Laplacian eigenfunctions of the disk and the ball (plan 021 P1, work package E; EXP-037 DESIGN.md 6).

    disk (n, m, trig, bc)    J_m(k r) cos(m phi) or sin(m phi), k = x / R, x a zero of J_m or J_m'
    ball (n, l, j, bc)       j_l(k r) Y_lj(theta, phi), k = x / R, x a zero of j_l or j_l'

(r, phi) and (r, theta, phi) are polar and spherical coordinates about the centre of the domain. Eigenvalues are
lambda = k^2 for -Laplace u = lambda u. The boundary condition is u = 0 (dirichlet) or du/dr = 0 (neumann) on the
boundary. The radial functions are evaluated through scipy.special; at r = 0 they take their limits (J_m(0) = delta_m0,
j_l(0) = delta_l0), so the value at the centre is the limit of the function there (not 0 for the constant l = 0 / m = 0
modes, 0 for the others), whatever direction the undefined angle takes.

The radial L2 norms are Lommel integrals: with x = k R,

    int_0^R J_m(k r)^2 r dr     = R^2/2 [J_m'(x)^2 + (1 - m^2/x^2) J_m(x)^2]
    int_0^R j_l(k r)^2 r^2 dr   = R^3/2 [j_l'(x)^2 + j_l(x) j_l'(x)/x + (1 - l(l+1)/x^2) j_l(x)^2]

(the second from (R^3/2)[j_l^2 - j_(l-1) j_(l+1)] and the recurrences of j_l), with j_l' = 0 or j_l = 0 simplifying them.
"""


import functools
import inspect
import math
import operator

import numpy as np
from scipy.special import j1, jn_zeros, jnp_zeros, jv, spherical_jn

from ...core.claims import DERIVED
from ...core.errors import Refused
from ...spaces.harmonic import values as harmonics_values
from ...realisations.spectra.laplace_spectra import mode as laplace_mode
from .. import geometry as geo
from ..base import BankFunction, Regularity, Support
from ..registry import register

COURANT = "Courant & Hilbert 1953, Methods of Mathematical Physics I, V.5 and VI.2 (eigenfunctions of the Laplacian)"
D_, N_ = ("dirichlet", 0.0), ("neumann", 0.0)
BCS = {"dirichlet": (D_, D_), "neumann": (N_, N_), "dn": (D_, N_), "nd": (N_, D_)}     # (left, right) conditions
EIGEN_REGULARITY = ("an eigenfunction of the Laplacian is real-analytic in the interior (elliptic regularity); these "
                    "are restrictions of entire functions of the Cartesian coordinates, so analytic up to the "
                    "boundary: " + COURANT)
LIMIT_INDEX = 100000


class EigenFunction(BankFunction):
    """A bank function whose "eigenvalue" closed form survives scaling (BankFunction.scaled would scale it by |c|)."""

    def scaled(self, c, name=None):
        out = super().scaled(c, name)
        closed = dict(out.closed)
        closed["eigenvalue"] = self.closed["eigenvalue"]
        return EigenFunction(out.family, out.domain, out.params, out.fn, out.support, out.regularity, closed,
                             out.gradient, out.name, out.reasons)


def int_index(name, v, lo, hi=LIMIT_INDEX):
    """v as a Python int in [lo, hi]; Refused with the reason otherwise."""
    if v is None:
        raise Refused(f"{name} is required (an integer >= {lo})")
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise Refused(f"{name} must be an integer (got {v!r})")
    try:
        v = operator.index(v)
    except TypeError:
        raise Refused(f"{name} must be an integer (got {v!r})") from None
    if not lo <= v <= hi:
        raise Refused(f"{name} must lie in [{lo}, {hi}] (got {v})")
    return v


def eigen_function(family, D, params, fn, support, regularity, eigenvalue, integral, why, gradient=None, note=""):
    """The EigenFunction with the closed forms eigenvalue, integral and l2_norm_sq = 1 (all derived); `note` extends the
    reason of the regularity theorem."""
    closed = {"eigenvalue": (float(eigenvalue), DERIVED, why["eigenvalue"]),
              "l2_norm_sq": (1.0, DERIVED, why["norm"]),
              "integral": (float(integral), DERIVED, why["integral"])}
    reasons = {"regularity": EIGEN_REGULARITY + (f"; {note}" if note else ""), "support": "analytic and not identically zero: its zero set is a null set"}
    return EigenFunction(family, D, params, fn, support, regularity, closed, gradient, reasons=reasons)


# --------------------------------------------------------------------------------------------- interval modes
# The modes are those of the Laplacian's spectrum on an interval (realisations/spectra/laplace_spectra.py, plan 068 P3): the bank
# names its conditions (dirichlet, neumann, dn, nd, robin with a pair of betas) and its index n, and is a view on them.
def _beta(beta):
    """(beta_a, beta_b) validated: finite and >= 0 (the sign that makes -Laplace non-negative)."""
    if isinstance(beta, (str, bytes)):
        raise Refused(f"beta must be a pair (beta_a, beta_b) of numbers (got {beta!r})")
    try:
        ba, bb = (float(b) for b in beta)
    except (TypeError, ValueError):
        raise Refused(f"beta must be a pair (beta_a, beta_b) of numbers (got {beta!r})") from None
    if not (math.isfinite(ba) and math.isfinite(bb) and ba >= 0 and bb >= 0):
        raise Refused(f"Robin coefficients must be finite and >= 0 (got beta = ({ba!r}, {bb!r}); infinity is the "
                      "Dirichlet condition: use bc='dirichlet')")
    return ba, bb


def interval_mode(a, b, bc, n, beta=None):
    """The Mode of [a, b] for the condition bc and index n (see _laplace_eigen_interval); validated. The g-th mode of
    pg.Laplacian(Interval(a, b), bc={"left": ..., "right": ...}) (realisations/spectra/laplace_spectra.mode), g = n + 1 for
    Neumann and n otherwise."""
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise Refused(f"the interval [{a}, {b}] must be bounded")
    if not isinstance(bc, str) or (bc not in BCS and bc != "robin"):
        raise Refused(f"bc must be one of dirichlet, neumann, dn, nd, robin (got {bc!r})")
    if (bc == "robin") != (beta is not None):
        raise Refused("beta = (beta_a, beta_b) is required for bc='robin' and not allowed otherwise")
    n = int_index("n", n, 0 if bc == "neumann" else 1)
    g = n + 1 if bc == "neumann" else n
    if bc == "robin":
        ba, bb = _beta(beta)
        ends = [("robin", x) if x > 0 else ("neumann", 0.0) for x in (ba, bb)]
    else:
        ends = BCS[bc]
    mode = laplace_mode(a, b, ends[0], ends[1], g)
    if bc == "robin" and mode.residual is None:
        mode.residual = 0.0                                # both betas 0: the Neumann closed form
    return mode


INTERVAL_WHY = {
    "norm": "u = N cos(k x' - phi_a) with N^-2 = L/2 + (sin 2 phi_a + sin 2 phi_b)/(4k): the L2 norm is 1 by construction",
    "integral": "int_0^L N cos(k x' - phi_a) dx' = N [(-1)^(g-1) sin phi_b + sin phi_a] / k (k = 0: N L)"}


def _interval_examples(D):
    return [{"n": 2, "bc": "dirichlet"}, {"n": 3, "bc": "neumann"}, {"n": 2, "bc": "nd"},
            {"n": 2, "bc": "robin", "beta": (1.5, 0.7)}]


def _laplace_eigen_interval(D, *, n=None, bc="dirichlet", beta=None):
    """The n-th eigenfunction u of -u'' = lambda u on [a, b], L2-normalised (int u^2 = 1), for the condition `bc`.

    With x' = x - a and L = b - a (module docstring for the general form):
      bc='dirichlet' (u(a) = u(b) = 0), n >= 1:   sqrt(2/L) sin(n pi x'/L),            lambda = (n pi / L)^2
      bc='neumann' (u'(a) = u'(b) = 0), n >= 0:   sqrt(2/L) cos(n pi x'/L), n = 0: 1/sqrt(L); lambda = (n pi / L)^2
      bc='dn' (u(a) = 0, u'(b) = 0), n >= 1:      sqrt(2/L) sin((n - 1/2) pi x'/L),     lambda = ((n - 1/2) pi / L)^2
      bc='nd' (u'(a) = 0, u(b) = 0), n >= 1:      sqrt(2/L) cos((n - 1/2) pi x'/L),     lambda = ((n - 1/2) pi / L)^2
      bc='robin', beta = (beta_a, beta_b) >= 0: -u'(a) + beta_a u(a) = 0, u'(b) + beta_b u(b) = 0, n >= 1 (n = 1 the
          lowest; n - 1 is the Neumann index it tends to as the betas go to 0): u = N cos(k x' - arctan(beta_a/k)),
          lambda = k^2, k the unique root in (0, n pi/L] of k L = arctan(beta_a/k) + arctan(beta_b/k) + (n - 1) pi.
    Closed forms: eigenvalue, l2_norm_sq = 1, integral. The gradient is the derivative, shape (..., 1). Only the
    Dirichlet condition makes the function vanish on the boundary (vanishes_at_boundary = 1)."""
    mode = interval_mode(float(D.a), float(D.b), bc, n, beta)
    k2 = mode.k ** 2
    if bc == "robin":
        form = (f"k^2 with k the root of k L = arctan({beta[0]:g}/k) + arctan({beta[1]:g}/k) + (n - 1) pi, found by "
                f"Brent's method (boundary residual {mode.residual:.1e})")
    else:
        form = "(n pi / L)^2" if bc in ("dirichlet", "neumann") else "((n - 1/2) pi / L)^2"
    why = {**INTERVAL_WHY, "eigenvalue": f"-u'' = lambda u with lambda = {form}, L = {mode.L:g}"}
    params = {"n": n, "bc": bc, **({"beta": tuple(float(b) for b in beta)} if bc == "robin" else {})}
    c = geo.centre(D)
    return eigen_function(
        "laplace_eigen", D, params, lambda x: mode.value(x),
        Support("full", "the whole interval (analytic, not identically zero)", tuple(c), 0.5 * mode.L),
        Regularity(math.inf, vanishes_at_boundary=1 if bc == "dirichlet" else None), k2, mode.integral, why,
        lambda x: mode.deriv(x)[..., None])


# --------------------------------------------------------------------------------------------- box
def _box_axes(D, n, bc, beta):
    """The Modes along the axes of the box D, validated."""
    d = len(D.sides)
    if not isinstance(n, (tuple, list, np.ndarray)) or len(n) != d:
        raise Refused(f"n must be a tuple of {d} indices, one per axis (got {n!r})")
    if isinstance(bc, str):
        bcs = (bc,) * d
    else:
        try:
            bcs = tuple(bc)
        except TypeError:
            raise Refused(f"bc must be a name or a tuple of {d} names, one per axis (got {bc!r})") from None
    if len(bcs) != d:
        raise Refused(f"bc must be a name or a tuple of {d} names, one per axis (got {bc!r})")
    if beta is None:
        betas = (None,) * d
    elif isinstance(beta, (tuple, list)) and len(beta) == d:
        betas = tuple(beta)
    else:
        raise Refused(f"beta must be None or a tuple of {d} entries (None or (beta_a, beta_b)), got {beta!r}")
    modes = [interval_mode(float(s.a), float(s.b), bcs[i], n[i], betas[i]) for i, s in enumerate(D.sides)]
    return modes, bcs, betas


def _box_examples(D):
    d = geo.dimension(D)
    mixed = ("neumann", "dn", "dirichlet")
    return [{"n": (1,) * d, "bc": "dirichlet"},
            {"n": tuple(range(1, d + 1)), "bc": tuple(mixed[i % 3] for i in range(d))},
            {"n": (2,) * d, "bc": ("robin",) + ("nd",) * (d - 1), "beta": ((1.0, 2.0),) + (None,) * (d - 1)}]


def _laplace_eigen_box(D, *, n=None, bc="dirichlet", beta=None):
    """The product u(x) = prod_i u_{n_i}(x_i) of interval eigenfunctions (laplace_eigen on an interval) of the sides of the
    box, so -Laplace u = lambda u with lambda = sum_i lambda_i and ||u|| = 1 (the Lebesgue measure of a product).

    n: a tuple of one index per axis, meaning on axis i what it means for laplace_eigen on an interval for that axis'
    condition (axes in the order of D.sides; no packing). bc: one name for every axis or a tuple of names, one per axis
    (dirichlet, neumann, dn, nd, robin; default dirichlet). beta: None, or a tuple with, per axis, None or the pair
    (beta_a, beta_b) (required exactly on the robin axes). Closed forms: eigenvalue (the sum), l2_norm_sq = 1,
    integral (the product of the factors'). The gradient has shape (..., d). The function vanishes on the whole
    boundary only if every axis is Dirichlet (vanishes_at_boundary = 1 then, else no claim)."""
    modes, bcs, betas = _box_axes(D, n, bc, beta)

    def parts(X):
        return [m.value(x) for m, x in zip(modes, X)]

    def fn(*X):
        out = 1.0
        for v in parts(X):
            out = out * v
        return out

    def gradient(*X):
        vals = parts(X)
        comps = []
        for i, (m, x) in enumerate(zip(modes, X)):
            term = m.deriv(x)
            for j, v in enumerate(vals):
                if j != i:
                    term = term * v
            comps.append(term)
        return np.stack(np.broadcast_arrays(*comps), axis=-1)

    integral = math.prod(m.integral for m in modes)
    params = {"n": tuple(int(i) for i in n), "bc": bcs,
              **({"beta": betas} if any(b is not None for b in betas) else {})}
    why = {"eigenvalue": "-Laplace u = (sum_i lambda_i) u for a product of one-dimensional eigenfunctions, lambda_i = "
                         f"k_i^2 with k = {tuple(round(m.k, 12) for m in modes)}",
           "norm": "the Lebesgue measure of a box is the product measure: ||u||^2 = prod ||u_i||^2 = 1",
           "integral": "Fubini: the product of the one-dimensional integrals of the factors"}
    c = geo.centre(D)
    r = 0.5 * math.sqrt(sum(m.L ** 2 for m in modes))
    return eigen_function(
        "laplace_eigen", D, params, fn, Support("full", "the whole box (analytic, not identically zero)", tuple(c), r),
        Regularity(math.inf, vanishes_at_boundary=1 if all(b == "dirichlet" for b in bcs) else None),
        sum(m.k ** 2 for m in modes), integral, why, gradient)


# ---------------------------------------------------------------- from eigenfunctions_radial.py (plan 063)
WATSON = ("Watson 1944, A Treatise on the Theory of Bessel Functions, ch. 5 (Lommel integrals) and ch. 15 (zeros); "
          "Abramowitz & Stegun 1964, Handbook of Mathematical Functions, 9.1 and 10.1")
DISK_RADIAL = "the Bessel functions are analytic; J_m(k r) e^{i m phi} = (x + i y)^m times an entire function of r^2"
BALL_RADIAL = "j_l(k r) Y_lj = (the solid harmonic r^l Y_lj) times an entire function of r^2"
MAX_ORDER, MAX_RADIAL = 60, 300
RADIAL_BCS = ("dirichlet", "neumann")


def _bc(bc):
    if bc not in RADIAL_BCS:
        raise Refused(f"bc must be 'dirichlet' or 'neumann' (got {bc!r})")
    return bc


# --------------------------------------------------------------------------------------------- disk
def disk_wavenumber(R, m, n, bc):
    """k = x/R, x the n-th positive zero of J_m (dirichlet) or J_m' (neumann); 0 for the constant (neumann, m = n = 0)."""
    if bc == "neumann" and m == 0 and n == 0:
        return 0.0
    return float((jn_zeros(m, n) if bc == "dirichlet" else jnp_zeros(m, n))[-1]) / R


def _disk_norm(R, m, k, bc):
    """N with N^2 c_m int_0^R J_m(k r)^2 r dr = 1, c_0 = 2 pi (cos), c_m = pi."""
    c = 2 * math.pi if m == 0 else math.pi
    if k == 0.0:
        return 1.0 / math.sqrt(math.pi * R * R)
    x = k * R
    radial = 0.5 * R * R * (jv(m + 1, x) ** 2 if bc == "dirichlet" else (1 - m * m / x ** 2) * jv(m, x) ** 2)
    return 1.0 / math.sqrt(c * radial)


def _disk_examples(D):
    return [{"n": 1, "m": 0}, {"n": 2, "m": 1, "trig": "sin"}, {"n": 1, "m": 2, "bc": "neumann"},
            {"n": 0, "m": 0, "bc": "neumann"}]


def _laplace_eigen_disk(D, *, n=None, m=None, trig="cos", bc="dirichlet"):
    """The eigenfunction N J_m(k r) T(m phi) of -Laplace on the disk of radius R about its centre, (r, phi) polar about
    the centre, T = cos (trig='cos', default) or sin (trig='sin', needs m >= 1), L2-normalised.

    bc='dirichlet' (u = 0 on the circle): k = j_{m,n} / R, the n-th positive zero of J_m, n >= 1.
    bc='neumann' (du/dr = 0 on the circle): k = j'_{m,n} / R, the n-th positive zero of J_m', n >= 1, and for m = 0 also
    n = 0, the constant 1 / sqrt(pi R^2) (lambda = 0). lambda = k^2. Normalisation: N^-2 = c_m int_0^R J_m(k r)^2 r dr
    (module docstring), c_0 = 2 pi, c_m = pi. Closed forms: eigenvalue, l2_norm_sq = 1, integral (2 pi N R J_1(kR)/k for
    m = 0, 0 for m >= 1 by the angular integral; N pi R^2 for the constant). The gradient follows from
    (d_x -+ i d_y) J_m(k r) e^{i m phi} = k (J_{m-1} e^{i(m-1)phi} or -J_{m+1} e^{i(m+1)phi}). Only the Dirichlet
    condition makes the function vanish on the boundary (vanishes_at_boundary = 1)."""
    m = int_index("m", m, 0, MAX_ORDER)
    bc = _bc(bc)
    n = int_index("n", n, 0 if (bc == "neumann" and m == 0) else 1, MAX_RADIAL)
    if trig not in ("cos", "sin"):
        raise Refused(f"trig must be 'cos' or 'sin' (got {trig!r})")
    if trig == "sin" and m == 0:
        raise Refused("trig='sin' needs m >= 1: sin(0 phi) is identically 0")
    R, cx, cy = float(D.radius), *(float(c) for c in D.centre)
    k = disk_wavenumber(R, m, n, bc)
    N = _disk_norm(R, m, k, bc)
    T = np.cos if trig == "cos" else np.sin

    def polar(x, y):
        dx, dy = np.asarray(x, float) - cx, np.asarray(y, float) - cy
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def fn(x, y):
        r, phi = polar(x, y)
        return N * jv(m, k * r) * T(m * phi)

    def gradient(x, y):
        r, phi = polar(x, y)
        jm, jp = jv(m - 1, k * r), jv(m + 1, k * r)
        cm, sm, cp, sp = jm * np.cos((m - 1) * phi), jm * np.sin((m - 1) * phi), jp * np.cos((m + 1) * phi), \
            jp * np.sin((m + 1) * phi)
        if trig == "cos":
            g = np.stack(np.broadcast_arrays(cm - cp, -(sp + sm)), axis=-1)
        else:
            g = np.stack(np.broadcast_arrays(sm - sp, cp + cm), axis=-1)
        return 0.5 * N * k * g

    if k == 0.0:
        integral = N * math.pi * R * R
    elif m == 0:
        integral = 2 * math.pi * N * R * float(j1(k * R)) / k
    else:
        integral = 0.0
    rootname = "J_m" if bc == "dirichlet" else "J_m'"
    why = {"eigenvalue": f"lambda = (x/R)^2 with x the n-th positive zero of {rootname}, m = {m}, n = {n}"
                         f" ({'constant mode, lambda = 0' if k == 0.0 else f'x = {k * R:.15g}'}); R = {R:g}",
           "norm": "N^-2 = c_m int_0^R J_m(k r)^2 r dr (Lommel integral), so the L2(disk) norm is 1",
           "integral": "int_0^R J_0(k r) r dr = R J_1(k R)/k for m = 0; for m >= 1 the angular integral of "
                       "cos(m phi) or sin(m phi) over [0, 2 pi] is 0; the constant has mass N pi R^2"}
    return eigen_function(
        "laplace_eigen", D, {"n": n, "m": m, "trig": trig, "bc": bc}, fn,
        Support("full", "the whole disk (analytic, not identically zero)", (cx, cy), R),
        Regularity(math.inf, vanishes_at_boundary=1 if bc == "dirichlet" else None), k * k, integral, why, gradient,
        note=DISK_RADIAL)


# --------------------------------------------------------------------------------------------- ball
@functools.cache
def _ball_spectrum(R, l, bc):
    """The spectrum of L_l, the degree-l radial operator of pg.Laplacian(Ball(R), bc, centre="regular")
    (realisations/spectra/bessel.RadialBessel, plan 068 P3): the bank's ball eigenfunctions are a view on it."""
    from ...operators.unbounded import Laplacian
    from ...domains import Ball
    return Laplacian(Ball(R), bc={"boundary": bc}, centre="regular").radial(l).spectrum()


def ball_wavenumber(R, l, n, bc):
    """k = x/R, x the n-th positive zero of j_l (dirichlet) or j_l' (neumann); 0 for the constant (neumann, l = n = 0):
    the wavenumber of the Laplacian's spectrum (the constant counts as the first mode of Neumann, l = 0)."""
    idx = n + 1 if (bc == "neumann" and l == 0) else n
    return float(_ball_spectrum(float(R), int(l), bc).wavenumbers(idx)[-1])


def _ball_norm(R, l, k, bc):
    """N with N^2 int_0^R j_l(k r)^2 r^2 dr = 1 (the real Y_lj are orthonormal on the unit sphere)."""
    if k == 0.0:
        return math.sqrt(3.0 / R ** 3)                          # N Y_00 = 1 / sqrt(vol), Y_00 = 1/sqrt(4 pi)
    x = k * R
    j, dj = spherical_jn(l, x), spherical_jn(l, x, True)
    radial = 0.5 * R ** 3 * (dj ** 2 if bc == "dirichlet" else (1 - l * (l + 1) / x ** 2) * j * j)
    return 1.0 / math.sqrt(radial)


def _ball_examples(D):
    return [{"n": 1, "l": 0, "j": 0}, {"n": 1, "l": 1, "j": 2}, {"n": 2, "l": 2, "j": 3, "bc": "neumann"},
            {"n": 0, "l": 0, "j": 0, "bc": "neumann"}]


def _laplace_eigen_ball(D, *, n=None, l=None, j=None, bc="dirichlet"):
    """The eigenfunction N j_l(k r) Y_lj(theta, phi) of -Laplace on the ball of radius R about its centre, L2-normalised
    (j_l the spherical Bessel function, Y_lj the package's real orthonormal harmonics of spaces/harmonic.py:
    j = 0 is Y_l0, j = 2m - 1 the cos(m phi) and j = 2m the sin(m phi) harmonic, 0 <= j <= 2l; no Condon-Shortley phase).

    bc='dirichlet' (u = 0 on the sphere): k = x_{l,n} / R, the n-th positive zero of j_l, n >= 1.
    bc='neumann' (du/dr = 0): k = y_{l,n} / R, the n-th positive zero of j_l', n >= 1; for l = 0 also n = 0, the constant
    1 / sqrt(4 pi R^3 / 3) (lambda = 0). lambda = k^2, with multiplicity 2l + 1 over j. The radial limit at r = 0 is
    exact: j_l(0) = delta_l0, so the function is 0 at the centre for l >= 1 and N Y_00 for l = 0 (the direction there
    is irrelevant). Closed forms: eigenvalue, l2_norm_sq = 1 (the Lommel norm of the module docstring), integral
    (N sqrt(4 pi) R^2 j_1(kR) / k for l = 0, 0 for l >= 1 since Y_lj is orthogonal to the constants). No gradient. Only
    the Dirichlet condition makes the function vanish on the boundary (vanishes_at_boundary = 1)."""
    l = int_index("l", l, 0, MAX_ORDER)
    j = int_index("j", j, 0, 2 * l)
    bc = _bc(bc)
    n = int_index("n", n, 0 if (bc == "neumann" and l == 0) else 1, MAX_RADIAL)
    R, c = float(D.radius), np.asarray(D.centre, float)
    k = ball_wavenumber(R, l, n, bc)
    N = _ball_norm(R, l, k, bc)

    def fn(x, y, z):
        P = np.stack(np.broadcast_arrays(*[np.asarray(a, float) for a in (x, y, z)]), axis=-1) - c
        flat = P.reshape(-1, 3)
        r = np.linalg.norm(flat, axis=1)
        Y = harmonics_values(l, flat)[:, l * l + j]
        return (N * spherical_jn(l, k * r) * Y).reshape(P.shape[:-1])

    if k == 0.0:
        integral = math.sqrt(4 * math.pi * R ** 3 / 3.0)
    elif l == 0:
        integral = N * math.sqrt(4 * math.pi) * R * R * float(spherical_jn(1, k * R)) / k
    else:
        integral = 0.0
    rootname = "j_l" if bc == "dirichlet" else "j_l'"
    why = {"eigenvalue": f"lambda = (x/R)^2 with x the n-th positive zero of {rootname}, l = {l}, n = {n}"
                         f" ({'constant mode, lambda = 0' if k == 0.0 else f'x = {k * R:.15g}'}); R = {R:g}",
           "norm": "N^-2 = int_0^R j_l(k r)^2 r^2 dr (Lommel integral) with Y_lj orthonormal on the unit sphere",
           "integral": "int_0^R j_0(k r) r^2 dr = R^2 j_1(k R)/k times int Y_00 = sqrt(4 pi) for l = 0; for l >= 1 "
                       "int_S Y_lj = 0 (orthogonal to Y_00); the constant has mass sqrt(vol)"}
    return eigen_function(
        "laplace_eigen", D, {"n": n, "l": l, "j": j, "bc": bc}, fn,
        Support("full", "the whole ball (analytic, not identically zero)", tuple(c), R),
        Regularity(math.inf, vanishes_at_boundary=1 if bc == "dirichlet" else None), k * k, integral, why,
        note=BALL_RADIAL)


# ------------------------------------------------------------------------------------------------- one family
_BY_KIND = {"interval": (_laplace_eigen_interval, _interval_examples), "box": (_laplace_eigen_box, _box_examples),
            "disk": (_laplace_eigen_disk, _disk_examples), "ball": (_laplace_eigen_ball, _ball_examples)}


def _examples(D):
    return _BY_KIND[geo.kind(D)][1](D)


@register("laplace_eigen", tuple(_BY_KIND),
          "L2-normalised eigenfunction of -Laplace on the domain with its boundary condition: interval and box "
          "(Dirichlet, Neumann, DN, ND or Robin, index n), disk (J_m(k r) cos or sin(m phi)), ball (j_l(k r) Y_lj)",
          references=(COURANT, WATSON), examples=_examples)
def laplace_eigen(D, **params):
    """The L2-normalised eigenfunction u of -Laplace u = lambda u on D with the stated boundary condition, by the kind of
    D (module docstring): on an interval (n, bc, beta), a box (n a tuple, bc and beta per axis), a disk (n, m, trig, bc)
    or a ball (n, l, j, bc). The eigenvalue lambda is on record as the closed form "eigenvalue"."""
    k = geo.kind(D)
    builder = _BY_KIND[k][0]
    names = tuple(inspect.signature(builder).parameters)[1:]
    unknown = sorted(set(params) - set(names))
    if unknown:
        raise Refused(f"laplace_eigen on a {k} takes {', '.join(names)}; not {', '.join(unknown)}")
    return builder(D, **params)
