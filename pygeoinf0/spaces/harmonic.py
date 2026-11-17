"""Real spherical harmonics as functions on the sphere, their conventions, and zonal (Funk-Hecke) coefficients (plan
009 G1, G4; moved from realisations/harmonics.py by plan 069 P4: these are the continuum's definitions, which every
realisation of a sphere, ball or shell uses, not a realisation).

The package's harmonics are orthonormal on the unit sphere, without the Condon-Shortley phase, packed degree by degree at
row l^2 + j:

    Y_l0 = p_l0(cos theta),  Y_l,2m-1 = sqrt(2) p_lm(cos theta) cos(m phi),  Y_l,2m = sqrt(2) p_lm(cos theta) sin(m phi)

with p_lm the associated Legendre functions normalised so that 2 pi int p_lm^2 d(cos theta) = 1 (p_lm > 0 near the
north pole for m > 0; assoc(m, L, mu, s) by the three-term recurrence in l; rows(L) and order_rows(L) the packed
layout). A HarmonicConvention (pg.HarmonicConvention) names another real basis by its phase and scale relative to this
one, Y^conv_lj = sign_lj scale_lj Y_lj, so a conversion is a signed permutation-free diagonal scaling, exact:

    THESIS = HarmonicConvention("condon-shortley")       the thesis's basis (scipy's sph_harm_y with the
                                                         Condon-Shortley phase, sqrt(2) Re / Im): (-1)^m Y_lj
    THESIS.convert(C, to=PACKAGE)                        coefficients in the thesis basis -> the package's
    HarmonicConvention("condon-shortley", scale=SQRT2_NONZONAL)   a basis whose non-zonal members are Y / sqrt(2)

values(L, X) gives every Y_lj at points (Cartesian unit vectors or (theta, phi)); labels(L) the (l, j) of every packed
row; legendre_P(L, x) the Legendre polynomials P_l; funk_hecke(profile, psi0, L) the coefficients
lambda_l = 2 pi int_0^pi w(psi) P_l(cos psi) sin(psi) dpsi of a zonal weight, so that the average
int w(xi . eta) h(eta) d eta = sum_l lambda_l sum_j Y_lj(xi) h_lj (Funk-Hecke). The degree-wise operations on
realisations in harmonics are realisations/series/harmonic_degrees.py.
"""
import math
from functools import lru_cache

import numpy as np

from ..quadrature import gauss


# ------------------------------------------------------------------------------------------ Legendre functions

@lru_cache(maxsize=None)
def _start(m):
    """p_mm = c_m sin^m(theta): c_m, from c_0 = 1/sqrt(4 pi), c_m = c_{m-1} sqrt((2m + 1) / (2m))."""
    if m == 0:
        return 1.0 / math.sqrt(4 * math.pi)
    return _start(m - 1) * math.sqrt((2 * m + 1) / (2 * m))


def assoc(m, L, mu, s):
    """p_lm(mu) for l = m..L at points mu = cos(theta) (s = sin(theta) >= 0): array (L - m + 1, len(mu)).
    p_mm in logarithms (sin^m underflows harmlessly to 0 near the poles at high order), then
    p_{m+1,m} = sqrt(2m + 3) mu p_mm, p_lm = a_lm (mu p_{l-1,m} - b_lm p_{l-2,m})."""
    out = np.empty((L - m + 1, mu.size))
    if m == 0:
        out[0] = _start(0)
    else:
        with np.errstate(divide="ignore"):
            out[0] = np.exp(math.log(_start(m)) + m * np.log(s))
    if L > m:
        out[1] = math.sqrt(2 * m + 3) * mu * out[0]
    for l in range(m + 2, L + 1):
        a = math.sqrt((4.0 * l * l - 1) / (l * l - m * m))
        b = math.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1) ** 2 - 1))
        out[l - m] = a * (mu * out[l - m - 1] - b * out[l - m - 2])
    return out


def rows(L):
    """For the packed layout of degree L: (l, m, part) of every row, part 0 = m == 0 or cos, 1 = sin."""
    l = np.repeat(np.arange(L + 1), 2 * np.arange(L + 1) + 1)
    j = np.arange((L + 1) ** 2) - l * l
    m = (j + 1) // 2
    part = np.where(j == 0, 0, (j + 1) % 2)
    return l, m, part


@lru_cache(maxsize=None)
def order_rows(L):
    """Per order m: the packed row indices of the cos part (l = m..L) and of the sin part (m >= 1)."""
    out = []
    for m in range(L + 1):
        ls = np.arange(m, L + 1)
        if m == 0:
            out.append((ls * ls, None))
        else:
            out.append((ls * ls + 2 * m - 1, ls * ls + 2 * m))
    return out


SQRT2_NONZONAL = "sqrt2-nonzonal"


def index(l, j):
    """The packed row of (l, j), 0 <= j <= 2l."""
    l, j = np.asarray(l, int), np.asarray(j, int)
    if np.any(l < 0) or np.any(j < 0) or np.any(j > 2 * l):
        raise ValueError("harmonic labels are (l, j) with l >= 0 and 0 <= j <= 2l")
    return l * l + j


def labels(L):
    """(l, j) of every packed row up to degree L, shape ((L + 1)^2, 2)."""
    l, m, part = rows(L)
    return np.stack([l, np.where(m == 0, 0, 2 * m - 1 + part)], axis=1)


def order_of(j):
    """The order m of the packed position j (j = 0: m = 0; j = 2m - 1, 2m)."""
    j = np.asarray(j, int)
    return (j + 1) // 2


def _angles(X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[None]
    if X.shape[-1] == 2:
        return X[:, 0], X[:, 1]
    n = np.linalg.norm(X, axis=1)
    theta = np.arccos(np.clip(X[:, 2] / np.where(n > 0, n, 1.0), -1.0, 1.0))
    return theta, np.mod(np.arctan2(X[:, 1], X[:, 0]), 2 * math.pi)


def values(L, X, theta_phi=None):
    """Every Y_lj, l <= L, at the points X ((m, 3) directions; or theta_phi = (theta, phi)): shape (m, (L + 1)^2)."""
    theta, phi = (np.asarray(theta_phi[0], float).ravel(), np.asarray(theta_phi[1], float).ravel()) \
        if theta_phi is not None else _angles(X)
    mu, s = np.cos(theta), np.abs(np.sin(theta))
    out = np.empty((theta.size, (L + 1) ** 2))
    for m, (rc, rs) in enumerate(order_rows(L)):
        P = assoc(m, L, mu, s).T                                   # (points, l = m..L)
        if m == 0:
            out[:, rc] = P
        else:
            out[:, rc] = math.sqrt(2) * P * np.cos(m * phi)[:, None]
            out[:, rs] = math.sqrt(2) * P * np.sin(m * phi)[:, None]
    return out


class HarmonicConvention:
    """A real orthonormal-up-to-scale basis of spherical harmonics, Y^conv_lj = sign_lj scale_lj Y_lj in terms of the
    package's (module docstring). phase: "package" (none) or "condon-shortley" ((-1)^m); scale: "orthonormal" or
    SQRT2_NONZONAL (1 for m = 0, 1/sqrt(2) otherwise), or a callable (l, j) -> scale."""

    def __init__(self, phase="package", scale="orthonormal", name=None):
        if phase not in ("package", "condon-shortley"):
            raise ValueError("phase is 'package' or 'condon-shortley'")
        if not (callable(scale) or scale in ("orthonormal", SQRT2_NONZONAL)):
            raise ValueError("scale is 'orthonormal', SQRT2_NONZONAL or a callable (l, j) -> scale")
        self.phase, self.scale = phase, scale
        self.name = name or (phase if scale == "orthonormal" else f"{phase}, {scale if isinstance(scale, str) else 'scaled'}")

    def factors(self, l, j):
        """sign_lj scale_lj for arrays of labels: Y^conv = factors * Y."""
        l, j = np.asarray(l, int), np.asarray(j, int)
        m = order_of(j)
        sign = np.where(m % 2 == 1, -1.0, 1.0) if self.phase == "condon-shortley" else np.ones(np.shape(m))
        if callable(self.scale):
            sc = np.asarray(self.scale(l, j), float)
        elif self.scale == SQRT2_NONZONAL:
            sc = np.where(m == 0, 1.0, 1.0 / math.sqrt(2))
        else:
            sc = np.ones(np.shape(m))
        return sign * sc

    def convert(self, C, to=None, L=None):
        """Coefficients C (packed rows (L + 1)^2 first) of a function in this basis, expressed in the basis `to`
        (default: the package's). f = sum a^conv Y^conv = sum a Y gives a = factors * a^conv."""
        to = PACKAGE if to is None else to
        C = np.asarray(C, float)
        L = int(round(math.sqrt(C.shape[0]))) - 1 if L is None else L
        lab = labels(L)
        f = self.factors(lab[:, 0], lab[:, 1]) / to.factors(lab[:, 0], lab[:, 1])
        return f.reshape((-1,) + (1,) * (C.ndim - 1)) * C

    def values(self, L, X, theta_phi=None):
        """The basis functions of this convention at points: values(L, X) times the factors."""
        lab = labels(L)
        return values(L, X, theta_phi) * self.factors(lab[:, 0], lab[:, 1])[None, :]

    def __repr__(self):
        return f"HarmonicConvention({self.name})"


PACKAGE = HarmonicConvention("package", name="package")
THESIS = HarmonicConvention("condon-shortley", name="condon-shortley (thesis)")


# ------------------------------------------------------------------------------------------------ zonal weights
def profile_function(profile, psi0):
    """(w0(psi), support end) of a named profile with angular radius psi0, or of a callable of psi."""
    psi0 = float(psi0)
    if callable(profile):
        return profile, math.pi
    if profile == "raised-cosine":
        return (lambda p: np.where(p <= psi0, np.cos(0.5 * math.pi * p / psi0) ** 2, 0.0)), psi0
    if profile == "spherical-cap":
        return (lambda p: np.where(p <= psi0, 1.0, 0.0)), psi0
    if profile == "gaussian":
        return (lambda p: np.exp(-0.5 * (p / psi0) ** 2)), math.pi
    raise ValueError("profile: 'raised-cosine', 'spherical-cap', 'gaussian' or a callable of the angle psi")


def legendre_P(L, x):
    """P_l(x), l = 0..L (unnormalised Legendre polynomials) by the three-term recurrence: shape (L + 1, len(x))."""
    x = np.asarray(x, float)
    out = np.empty((L + 1, x.size))
    out[0] = 1.0
    if L:
        out[1] = x
    for l in range(1, L):
        out[l + 1] = ((2 * l + 1) * x * out[l] - l * out[l - 1]) / (l + 1)
    return out


def funk_hecke(profile, psi0, L, normalise=True, nodes=None):
    """lambda_l (l = 0..L) of the zonal weight w = w0 / int w0 (normalise) or w0, and the squared norm int w^2 d eta,
    by Gauss-Legendre in psi over the support (2L + 64 nodes; the integrands are analytic there). Returns
    (lambda (L + 1,), norm2)."""
    w0, end = profile_function(profile, psi0)
    n = nodes or (2 * L + 64)
    t, wt = gauss(n)
    psi = 0.5 * end * (t + 1.0)
    dw = 0.5 * end * wt * np.sin(psi)
    v = np.asarray(w0(psi), float)
    mass = 2 * math.pi * float(dw @ v)
    if normalise:
        if not mass > 0:
            raise ValueError("the profile has no positive mass")
        v = v / mass
    lam = 2 * math.pi * (legendre_P(L, np.cos(psi)) @ (dw * v))
    return lam, 2 * math.pi * float(dw @ (v * v))
