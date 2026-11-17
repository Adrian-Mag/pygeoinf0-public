"""Kernels of the Sobolev constructions of spaces/constructions.py on intervals and boxes (plan 006): the free-space
Matern kernel plus a correction that carries the boundary. Derivations: experiments/EXP-026-sobolev-constructions/
NOTES.md (§ numbers below).

    finite rank   integer powers on an interval [a, b] (forms, their operators' integer powers): the Green function of
                  (1 - l^2 d^2)^n with n conditions at each end is G(x, y) = k_n(x - y) + phi(x)^T C phi(y), phi the 2n
                  homogeneous solutions g_j(t_e) = (t_e / l)^j e^(-t_e / l), t_e the distance to end e, j < n (§4).
                  C = -M^-1 N from the conditions (polynomials in the outward normal derivative). Energies are
                  E_matern + Phi_A C Phi_B^T with the moments Phi = int phi dmu in closed form: exact up to the 2n x 2n
                  solve, whose condition number grows like (l / (b - a))^(2n - 1) when l exceeds the interval.
    images        spectral spaces with Neumann (+1) or Dirichlet (-1) at every end or face, any order s: G is the
                  signed sum of the Matern kernel over the reflection group (§5), truncated at a reach R where the
                  omitted images are provably below a tolerance tau of k_ref |mu| |nu| (tail_bound, §14): tau = 1e-16
                  (rounding) under settings.images = "exact", accuracy x 0.05 under "accuracy" (energy.ExactEngine with
                  Reflections)
    formula norms integer orders on an interval: the inner product of smooth elements, sum_i w_i int (P_i f)(P_i g) +
                  sum_e jet_e(f)^T Q_e jet_e(g) (§8), from the derivatives of the piecewise Legendre realisation, and
                  the conditions every element satisfies at the ends (checked on the realisation)
"""
import collections
import itertools
import math

import numpy as np
from numpy.polynomial import legendre as npleg
from scipy import special

from ...core.claims import DERIVED, Claim
from ...spaces.constructions import L_poly, poly_mul
from ...core.errors import Unsupported, ViolatesBoundaryCondition
from ...quadrature import gauss
from ..series import legendre as L
from .atoms import Kind

ROUNDING = 1e-16         # tau of the image sum under settings.images = "exact": below rounding of k_ref |mu| |nu|
Image = collections.namedtuple("Image", "sign R t")
Image.__doc__ = "y -> R y + t (R a vector of +-1 per flat axis) with the sign of the image (energy.ExactEngine)"
CONDITION_TOL = 1e-8     # relative size of a boundary condition on a realised formula above which it is violated


# ------------------------------------------------------------------------------------------ exponential polynomials

def dg(n, l):
    """d/dt on coefficients over g_j(t) = (t/l)^j e^(-t/l), j < n: g_j' = (j/l) g_(j-1) - g_j / l."""
    D = -np.eye(n) / l
    for j in range(1, n):
        D[j - 1, j] = j / l
    return D


def g_values(t, n, l):
    t = np.asarray(t, float)
    u = t / l
    return np.stack([u ** j * np.exp(-u) for j in range(n)], axis=-1)


def matern_coefficients(n, l):
    """k_n(t) = sum_m alpha_m g_m(t) for t >= 0: the Matern kernel of H^n_l(R) (nu = n - 1/2) as exponential polynomial,
    z^nu K_nu(z) = sqrt(pi/2) e^-z sum_q (n' + q)! / (q! (n' - q)! 2^q) z^(n' - q), n' = n - 1."""
    c = (2 * math.pi) ** -0.5 * 2 ** (1 - n) / math.gamma(n) / l * math.sqrt(math.pi / 2)
    m1 = n - 1
    return np.array([c * math.factorial(2 * m1 - m) / (math.factorial(m1 - m) * math.factorial(m) * 2 ** (m1 - m))
                     for m in range(n)])


def g_integrals(t0, t1, n, l):
    """int_t0^t1 g_j(t) dt for j < n (arrays t0 <= t1): l (Gamma(j+1, u0) - Gamma(j+1, u1)), u = t/l, with the
    upper incomplete gamma j! e^-u sum_(i<=j) u^i/i!; 16-point Gauss on pieces with u1 - u0 <= 1 (no cancellation)."""
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    u0, u1 = t0 / l, t1 / l
    out = np.zeros(t0.shape + (n,))
    for j in range(n):
        Gam = lambda u: math.factorial(j) * np.exp(-u) * sum(u ** i / math.factorial(i) for i in range(j + 1))  # noqa
        out[..., j] = l * (Gam(u0) - Gam(u1))
    short = (u1 - u0) <= 1.0
    if np.any(short):
        x, w = gauss(16)
        a, b = t0[short], t1[short]
        T = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * x[None]
        out[short] = 0.5 * (b - a)[:, None] * np.einsum("q,mqj->mj", w, g_values(T, n, l))
    return out


class EndBasis:
    """phi = (g_j(x - a), j < n; g_j(b - x), j < n) on [a, b]: the homogeneous solutions of (1 - l^2 d^2)^n, bounded by
    1 on [a, b] (no overflow for small l)."""

    def __init__(self, a, b, n, l):
        self.a, self.b, self.n, self.l = float(a), float(b), int(n), float(l)

    def values(self, X):
        X = np.asarray(X, float).ravel()
        return np.hstack([g_values(X - self.a, self.n, self.l), g_values(self.b - X, self.n, self.l)])

    def moments(self, A):
        """(A.n, 2n): int phi dmu for the measures of a 1-D atom set (points and segments, with weights)."""
        P, E = A.P[:, 0], A.E1[:, 0]
        lo, hi = np.minimum(P, P + E), np.maximum(P, P + E)
        seg = A.kind == Kind.SEGMENT
        V = self.values(P)
        if np.any(seg):
            V[seg] = np.hstack([g_integrals(lo[seg] - self.a, hi[seg] - self.a, self.n, self.l),
                                g_integrals(self.b - hi[seg], self.b - lo[seg], self.n, self.l)])
        for q in np.flatnonzero(A.kind == Kind.WEIGHTED):         # weighted subintervals (EXP-031): Gauss with w
            X, W = A.piece[q].formula_nodes()
            V[q] = W @ self.values(X[:, 0])
        out = np.zeros((A.n, 2 * self.n))
        np.add.at(out, A.owner, A.weight[:, None] * V)
        return out

    def correction(self, conditions):
        """C with G = k_n(x - y) + phi(x)^T C phi(y) for the conditions {"left": [c, ...], "right": [...]} (n each,
        polynomials in d_n): C = -M^-1 N, M the conditions applied to phi, N to k_n(|x - y|) as functions of y (§4)."""
        n, l, Lng = self.n, self.l, self.b - self.a
        D = dg(n, l)
        Dr = [np.linalg.matrix_power(D, r) for r in range(2 * n + 1)]
        alpha = matern_coefficients(n, l)
        gL = g_values(Lng, n, l)
        rows_M, rows_N = [], []
        for e, end in enumerate(("left", "right")):
            cs = conditions[end]
            if len(cs) != n:
                raise ValueError(f"{len(cs)} conditions at {end} for a kernel of order {n}: need {n}")
            for c in cs:
                m, nn = np.zeros(2 * n), np.zeros(2 * n)
                for r, cr in enumerate(c):
                    if cr == 0:
                        continue
                    m[e * n:(e + 1) * n] += cr * (-1) ** r * Dr[r][0]          # own end: d_n = -d/dt at t = 0
                    m[(1 - e) * n:(2 - e) * n] += cr * (gL @ Dr[r])           # other end: d_n = +d/dt at t = b - a
                    nn[e * n:(e + 1) * n] += cr * (Dr[r] @ alpha)             # d_n^r k(|x - y|) = k^(r)(t_e(y))
                rows_M.append(m)
                rows_N.append(nn)
        M = np.array(rows_M)
        C = -np.linalg.solve(M, np.array(rows_N))
        return 0.5 * (C + C.T), float(np.linalg.cond(M))


class CorrectedEngine:
    """The energies of the kernel K + phi^T C phi: those of the inner engine (exact or H-matrix, Matern) plus the
    finite-rank part from the moments of the measures. The cached energies of an atom set (A.self_energy) are the
    corrected ones."""

    def __init__(self, inner, basis, C):
        self.inner, self.basis, self.C = inner, basis, C
        self.name = f"{inner.name} + rank-{len(C)} boundary correction (exact)"

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def _corr(self, A, B):
        return self.basis.moments(A) @ self.C @ self.basis.moments(B).T

    def energy(self, A, B):
        if A is B and A.self_energy is not None:
            return A.self_energy[0]
        out = self.inner.energy(A, B) + self._corr(A, B)
        if A is B:
            out = 0.5 * (out + out.T)
            A.self_energy = (out, A.self_energy[1])
        return out

    def apply(self, A, V):
        from ...settings import settings
        V = np.asarray(V, float)
        if A.self_energy is not None or A.n <= settings.dense_max:
            return self.energy(A, A) @ V
        Phi = self.basis.moments(A)
        return self.inner.apply(A, V) + Phi @ (self.C @ (Phi.T @ V))

    def describe(self, A):
        text = self.inner.describe(A)
        return None if text is None else f"{text}, + rank-{len(self.C)} boundary correction (exact)"

    def self_energies(self, A, idx):
        E, err = self.inner.self_energies(A, idx)
        Phi = self.basis.moments(A)[np.asarray(idx, int)]
        return E + np.einsum("ij,jk,ik->i", Phi, self.C, Phi), err

    def self_energies_all(self, A):
        Phi = self.basis.moments(A)
        return self.inner.self_energies_all(A) + np.einsum("ij,jk,ik->i", Phi, self.C, Phi)

    def correction_values(self, F, A):
        """(len(F), A.n): the correction's part of the representers k_mu_i at flat points F."""
        return self.basis.values(F[:, 0]) @ self.C @ self.basis.moments(A).T


# ------------------------------------------------------------------------------------------ images

class Reflections:
    """The reflection group of an interval or box [lo, hi] (flat coordinates) with a sign per face (+1: Neumann, the
    even reflection; -1: Dirichlet, the odd one). Per axis the images of y are y + 2 n L with sign (s_lo s_hi)^n and
    2 lo - y + 2 n L with sign s_lo (s_lo s_hi)^n (§5); a box takes all products."""

    def __init__(self, lo, hi, sign_lo, sign_hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.s_lo, self.s_hi = np.asarray(sign_lo, float), np.asarray(sign_hi, float)
        self._cuts = {}

    def tolerance(self):
        """(tau, setting): the tolerance of the image sum relative to k_ref |mu| |nu| under settings.images."""
        from ...settings import settings
        if settings.images == "accuracy":
            from .spectral import target
            return target(settings.accuracy), f"settings.images = 'accuracy' ({settings.accuracy:g} x 0.05)"
        return ROUNDING, "settings.images = 'exact' (rounding)"

    def cut(self, kernel):
        """(R, tau, bound, setting): the reach of the image sum for the kernel under settings.images, its tolerance tau,
        the proven bound T(R) / k_ref <= tau on the omitted images, and the setting."""
        tau, setting = self.tolerance()
        key = (kernel.s, kernel.d, kernel.l, tau)
        if key not in self._cuts:
            R = image_reach(kernel, self.hi - self.lo, tau)
            self._cuts[key] = (R, tau, tail_bound(kernel, self.hi - self.lo, R) / reference_value(kernel), setting)
        return self._cuts[key]

    def transforms(self, reach):
        """The images (Image(sign, R, t)) of the box within distance reach of it (the identity first)."""
        axes = []
        for lo, hi, a, b in zip(self.lo, self.hi, self.s_lo, self.s_hi):
            Lx, m = hi - lo, int(math.ceil(reach / (2 * (hi - lo)))) + 1
            items = []
            for k in range(-m, m + 1):
                p = (a * b) ** abs(k)
                for sign, R, t in ((p, 1.0, 2 * k * Lx), (a * p, -1.0, 2 * lo + 2 * k * Lx)):
                    ilo, ihi = sorted((R * lo + t, R * hi + t))
                    gap = max(0.0, ilo - hi, lo - ihi)
                    if gap <= reach:
                        items.append((gap, sign, R, t))
            axes.append(items)
        out = []
        for combo in itertools.product(*axes):
            if math.sqrt(sum(c[0] ** 2 for c in combo)) <= reach:
                out.append(Image(float(np.prod([c[1] for c in combo])), np.array([c[2] for c in combo]),
                                 np.array([c[3] for c in combo])))
        out.sort(key=lambda g: (np.any(g.R < 0), float(np.abs(g.t).sum())))
        return out


def reference_value(kernel):
    """k_ref, the scale of the tolerance of image sums: k(0) when finite (s > d/2, the prior variance of the free
    kernel), else k(l)."""
    return kernel.k0 if math.isfinite(kernel.k0) else float(kernel(kernel.l))


def tail_bound(kernel, sides, R):
    """A proven upper bound T(R) >= sup over x, y in the box of sum |k(|x - g y|)| over the images g y at distance
    >= R (EXP-026 NOTES §14). The images of y are 2^d translates of the lattice 2 L Z^d (L = sides); disjoint cells
    of half-diameter h = |L| give N(r) <= V_d (r + h)^d / prod L images within r of x, and k > 0 decreases, so
        sum k <= int_R^inf N(r) (-k'(r)) dr = k(R) N(R) + int_R^inf k(r) N'(r) dr.
    In the integral k(r) <= rho k(R) e^(-(r - R)/l) (r/R)^(nu - 1/2): e^z sqrt(z) K_mu(z) decreases for mu >= 1/2 and
    increases to sqrt(pi/2) for mu < 1/2 (DLMF 10.32.8 with t = 1 + tau/z), rho = 1 or sqrt(pi/2) / (e^Z sqrt(Z)
    K_mu(Z)); with (r + h)^(d-1) <= (1 + h/R)^(d-1) r^(d-1) it is an incomplete gamma function."""
    L = np.asarray(sides, float)
    d, l, nu = len(L), kernel.l, kernel.nu
    h, vol = float(np.linalg.norm(L)), float(np.prod(L))
    Vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    Z, mu = R / l, abs(nu)
    if Z > 600:
        return 0.0
    rho = 1.0 if mu >= 0.5 else math.sqrt(math.pi / 2) / (math.exp(Z) * math.sqrt(Z) * float(special.kv(mu, Z)))
    q = d - 1.5 + nu
    tail = math.exp(Z) * l ** (q + 1) * float(special.gammaincc(q + 1, Z)) * math.gamma(q + 1)
    integral = rho * d * Vd * (1 + h / R) ** (d - 1) * R ** (0.5 - nu) * tail / vol
    return float(kernel(R)) * (Vd * (R + h) ** d / vol + integral)


def image_reach(kernel, sides, tau):
    """The first R on the grid l (1 + j / 8) with tail_bound(R) <= tau k_ref."""
    ref, l = reference_value(kernel), kernel.l
    for j in range(8 * 600):
        R = l * (1 + j / 8)
        if tail_bound(kernel, sides, R) <= tau * ref:
            return R
    raise Unsupported(f"no reach of the image sum within 600 l meets the tolerance {tau:g}")


def images_claim(images, kernel):
    """The claim of an image sum, at the current settings.images."""
    R, tau, bound, setting = images.cut(kernel)
    ref = "k(0)" if math.isfinite(kernel.k0) else "k(l)"
    return Claim(f"kernel: the Matern kernel summed over the signed reflection images within R = {R / kernel.l:.4g} l; "
                 f"each energy within {bound:.1e} {ref} |mu| |nu| (total variations) of the full image sum "
                 f"(tolerance {tau:.0e}, {setting})", DERIVED,
                 "method of images for Neumann/Dirichlet faces (EXP-026 NOTES §5); the omitted images by a lattice "
                 "tail bound for the Matern kernel (NOTES §14, green.tail_bound)")


# ------------------------------------------------------------------------------------------ norms of formulas

def derivative(pl, r):
    """The r-th derivative of a piecewise Legendre realisation (piecewise; exact on the polynomials)."""
    if r == 0:
        return pl
    coefs = []
    for i, C in enumerate(pl.coefs):
        h = pl.breaks[i + 1] - pl.breaks[i]
        coefs.append(npleg.legder(C, r, scl=2.0 / h, axis=0) if C.shape[0] > r else np.zeros((1, C.shape[1])))
    return L.PiecewiseLegendre(pl.breaks, coefs, np.zeros(pl.N))


def apply_poly(pl, c):
    """sum_r c_r d^r f for a polynomial c in d/dx."""
    terms = [(float(cr), derivative(pl, r)) for r, cr in enumerate(c) if cr != 0]
    return L.lincomb(terms) if terms else L.lincomb([(0.0, pl)])


class IntervalForm:
    """The inner product of smooth elements of a construction of integer order on [a, b] (§8):
        <f, g> = sum_i w_i int (P_i f)(P_i g) dx + sum_e jet_e(f)^T Q_e jet_e(g),
    P_i polynomials in d/dx, jet_e = (u, d_n u, ..., d_n^(m-1) u) at end e; with the conditions every element
    satisfies at each end (polynomials in d_n)."""

    def __init__(self, a, b, interior, boundary, conditions):
        self.a, self.b = float(a), float(b)
        self.interior, self.boundary, self.conditions = interior, boundary, conditions

    def _jets(self, pl, m):
        """(2, m, N): d_n^r f at a (d_n = -d/dx) and b (d_n = d/dx), r < m."""
        V = np.stack([L.evaluate(derivative(pl, r), np.array([self.a, self.b])) for r in range(m)], axis=1)
        V[0] *= ((-1.0) ** np.arange(m))[:, None]
        return V

    def check(self, pl):
        """Raise ViolatesBoundaryCondition when a realised formula does not satisfy the conditions of its space."""
        for e, end in enumerate(("left", "right")):
            for c in self.conditions.get(end, []):
                J = self._jets(pl, len(c))[e]
                val = np.asarray(c) @ J
                scale = sum(abs(cr) * derivative(pl, r).sup_bound() for r, cr in enumerate(c))
                bad = np.abs(val) > CONDITION_TOL * np.maximum(scale, 1e-300)
                if np.any(bad):
                    from ...spaces.constructions import describe_condition
                    raise ViolatesBoundaryCondition(
                        f"a formula violates {describe_condition(c)} at the {end} end ({float(np.max(np.abs(val))):.3g}): "
                        "every element of this space satisfies it, so the function is not an element (choose a space "
                        "without the condition, or a function that satisfies it)")

    def gram(self, F, G):
        self.check(F)
        self.check(G)
        out = 0.0
        for w, c in self.interior:
            out = out + w * L.gram(apply_poly(F, c), apply_poly(G, c))
        for e, end in enumerate(("left", "right")):
            Q = self.boundary.get(end)
            if Q is not None:
                m = len(Q)
                out = out + self._jets(F, m)[e].T @ np.asarray(Q) @ self._jets(G, m)[e]
        return out


def exterior_form(s, l):
    """Q_s: the minimal exterior energy of H^s_l(R) on a half-line as a quadratic form in the jet (u, ..., d_n^(s-1) u)
    at its end (§3): V = sum_j c_j tau^j e^-tau (tau = distance / l), jet = T c, energy = l c^T W_s c."""
    D = dg(s, 1.0)
    T = np.array([np.linalg.matrix_power(D, r)[0] * l ** -r for r in range(s)])     # d_n = (1/l) d/dtau outward
    W = np.array([[math.factorial(i + j) / 2.0 ** (i + j + 1) for j in range(s)] for i in range(s)])
    E = l * sum(math.comb(s, i) * np.linalg.matrix_power(D, i).T @ W @ np.linalg.matrix_power(D, i) for i in range(s + 1))
    Ti = np.linalg.inv(T)
    Q = Ti.T @ E @ Ti
    return 0.5 * (Q + Q.T)


def interval_form(C, a, b):
    """The IntervalForm of a construction of integer order on [a, b], or None (fractional orders)."""
    s, l = C.s, C.length_scale
    if C.kind == "restriction":
        if s != int(s):
            return None
        k = int(s)
        Q = exterior_form(k, l)
        return IntervalForm(a, b, [(math.comb(k, j) * l ** (2 * j), (0.0,) * j + (1.0,)) for j in range(k + 1)],
                            {"left": Q, "right": Q}, {})
    q, k = C.power(), C.form.k
    if q is None:
        return None
    conds = {e: [c for c, _ in C.element_conditions(e)] for e in ("left", "right")}
    Lk = L_poly(l, k)
    P = (1.0,)
    for _ in range(q // 2):
        P = poly_mul(P, Lk)
    if q % 2 == 0:
        return IntervalForm(a, b, [(1.0, P)], {}, conds)
    interior = [(math.comb(k, j) * l ** (2 * j), poly_mul((0.0,) * j + (1.0,), P)) for j in range(k + 1)]
    boundary = {}
    for end, beta in C.form.robin:
        cvec = np.asarray(P)                                 # (P u)(end) over the d_n-jet: P has even powers of d only
        boundary[end] = beta * np.outer(cvec, cvec)
    return IntervalForm(a, b, interior, boundary, conds)


def face_signs(C, parts):
    """+1 (Neumann) or -1 (Dirichlet) per boundary part, for images; Unsupported for Robin parts and other forms."""
    if C.kind != "spectral" or C.form.k != 1:
        raise Unsupported(f"{C.label()}: only the operator 1 - l^2 Delta of an order-1 form is realised by images")
    signs = {}
    for part in parts:
        (c, essential), = C.form.conditions(part)
        if not essential and c[0] != 0:
            raise Unsupported(f"{C.label()}: images need Neumann or Dirichlet conditions; Robin parts are realised only "
                              "on intervals with integer orders (finite rank)")
        signs[part] = -1.0 if essential else 1.0
    return signs


def box_realisation(C, D):
    """Spectral spaces of 1 - l^2 Delta on a box with Neumann/Dirichlet faces, any s: signed reflection images."""
    signs = face_signs(C, D.boundary_parts)
    names = D.internal_names
    return dict(images=Reflections([s.a for s in D.sides], [s.b for s in D.sides], [signs[f"{n}-"] for n in names],
                                   [signs[f"{n}+"] for n in names]), claims=[])


def interval_realisation(C, D):
    """What the registry needs for a construction on an interval D: dict(order, correction=(basis, C), images,
    formula, claims) for RKHSBackend, or Unsupported."""
    a, b, l = D.a, D.b, C.length_scale
    form = interval_form(C, a, b)
    if C.kind == "restriction":
        return dict(order=C.s, formula=form)
    q = C.power()
    if q is not None:
        n = int(C.s)
        basis = EndBasis(a, b, n, l)
        Cmat, cond = basis.correction({e: [c for c, _ in C.kernel_conditions(e)] for e in ("left", "right")})
        claim = Claim(f"kernel: the Matern kernel plus a rank-{2 * n} boundary correction from the conditions at the "
                      f"ends, exact up to its {2 * n} x {2 * n} solve (condition number {cond:.1e}: relative error "
                      f"~ {cond * 1.1e-16:.0e})", DERIVED, "Green function of (1 - l^2 d^2)^n (EXP-026 NOTES §4)")
        return dict(order=C.s, correction=(basis, Cmat), formula=form, claims=[claim])
    try:
        signs = face_signs(C, ("left", "right"))
    except Unsupported as e:
        raise Unsupported(f"{e} (fractional powers are realised by images; Robin eigenfunctions are not explicit)")
    return dict(order=C.s, images=Reflections([a], [b], [signs["left"]], [signs["right"]]), formula=None,
                claims=[])
