"""Circular complex Gaussian measures (plan 020 P3c; experiments/EXP-035-mineral-exploration/COMPLEX.md section 3).

    circular(X, covariance=S, mean=None)     CN(m, S) on a complex space X with coordinates z: E (z - m)(z - m)^H = S
                                             and E (z - m)(z - m)^T = 0 (proper: no pseudo-covariance)
    X.standard_gaussian()                    CN(0, W^-1) on pg.Euclidean(n, field="C") (X.gaussian(covariance=S) refuses
                                             since plan 069 P6: call circular(X, S) on it and on pg.Complexified of a
                                             cell-realised L2)

The coordinates are those of pg.port.Coordinates: the numbers of C^n, the cell values of a complex cell space. S is a
Hermitian positive semidefinite matrix, a vector (its diagonal) or a number (S = s I). As a measure on the real space
X_R (<x, y>_R = Re <x, y>, the structure the rest of pygeoinf0 uses) it is a GaussianMeasure whose covariance is

    C_R u = (1/2) S W u          (W the metric of the coordinates: E <z, u>_R <z, v>_R = (1/2) Re v^H W S W u)

so variances of real functionals, push-forwards, sums and conditioning work unchanged; the complex variance
E |<z, u>|^2 = u^H W S W u is twice the real one (cvariance). The measure is circular iff C_R commutes with J; that
holds here by construction (C_R is complex-linear: multiplication by a complex matrix), a theorem recorded as the claim
"circular". A push-forward by an operator with the claim complex-linear stays circular (derived); otherwise the claim
is dropped (a real-linear image of a circular law need not be circular, e.g. Re z). The posterior of a circular prior
with circular noise through a complex-linear operator is circular (derived).
"""
import numpy as np

from ..core.claims import CHECKED, DERIVED, THEOREM, Claim
from ..core.errors import Refused, SpaceMismatch

CIRCULAR = "circular (proper): E (z - m)(z - m)^T = 0"


class _Coords:
    """The coordinates of a complex space: get(x) -> (n,) or (n, N), make(V) -> element or family, W applied."""

    def __init__(self, X):
        from ..spaces.base import CoordinateSpace
        from ..spaces.complex import CellsComplexified
        from ..spaces.elements import Family, Function, Vector, VectorFamily
        if isinstance(X, CoordinateSpace) and X.field == "C":
            self.n = X.dim
            self.get = lambda x: x.values.T if isinstance(x, VectorFamily) else x.values
            self.make = lambda V: Vector(X, V) if V.ndim == 1 else VectorFamily(X, V.T)
            self.W = X.apply_metric
        elif isinstance(X, CellsComplexified):
            from ..realisations.piecewise_constant import CellValues
            vol = X.backend.cells.volumes
            self.n = len(vol)
            self.get = lambda x: x.realisation.V if isinstance(x, Family) else x.realisation.V[:, 0]
            self.make = lambda V: Function(X, pl=CellValues(V)) if V.ndim == 1 else Family(X, CellValues(V))
            self.W = lambda V: (vol[:, None] * V) if np.ndim(V) == 2 else vol * V
        else:
            raise SpaceMismatch(f"circular Gaussian measures are defined on pg.Euclidean(n, field='C') and on pg.Complexified "
                                f"of a cell-realised L2, not on {X}")


def _matrix(S, n):
    S = np.asarray(S)
    if S.ndim == 0:
        S = np.full(n, S)
    if S.ndim == 1:
        if S.shape != (n,) or np.any(S.real < 0) or np.any(S.imag != 0):
            raise Refused(f"a diagonal covariance is n = {n} non-negative numbers")
        return np.diag(S.real).astype(complex)
    S = np.asarray(S, complex)
    if S.shape != (n, n) or not np.allclose(S, S.conj().T, atol=1e-13 * max(1.0, np.abs(S).max())):
        raise Refused(f"a covariance of {n} complex coordinates is a Hermitian {n} x {n} matrix")
    S = 0.5 * (S + S.conj().T)
    if np.linalg.eigvalsh(S).min() < -1e-12 * max(1.0, np.abs(S).max()):
        raise Refused("the covariance is not positive semidefinite")
    return S


def circular(X, covariance=None, mean=None, name="μ"):
    """CN(mean, covariance) on X (module docstring)."""
    from ..operators.linear import LinearOperator
    from .gaussian import GaussianMeasure
    co = _Coords(X)
    if covariance is None:
        raise Refused(f"a circular Gaussian on {X} needs its covariance S of the coordinates", needs=("covariance=S",))
    S = _matrix(covariance, co.n)
    from .gaussian import coordinate_factor
    L = coordinate_factor(S, name)                                # L L^H = S to rounding, nothing added

    C = LinearOperator(X, X, lambda u: co.make(0.5 * (S @ co.W(co.get(u)))), name="C_R")
    C._claims = [Claim("complex-linear (commutes with J)", THEOREM, "multiplication of the coordinates by S W / 2",
                       key="complex-linear")]
    C._sa = (True, THEOREM, "C_R = S W / 2 with S Hermitian: <C_R u, v>_R = Re v^H W S W u / 2 is symmetric")
    lam = np.linalg.eigvalsh(S)
    floor = np.finfo(float).eps * len(lam) * float(np.abs(lam).max(initial=0.0))
    C._pos = ("pd", CHECKED, f"S is positive definite: its smallest eigenvalue {lam.min():.3g} is above the rounding "
              f"floor eps n max|lam| = {floor:.3g}") if lam.size and lam.min() > floor else \
        ("psd", THEOREM, "S is positive semidefinite")       # plan 065 A5: definiteness is checked, not presumed

    def sampler(N, rng):
        xi = (rng.standard_normal((co.n, N)) + 1j * rng.standard_normal((co.n, N))) / np.sqrt(2)
        Z = L @ xi                                                    # E Z Z^H = S, E Z Z^T = 0
        return co.make(Z[:, 0] if N == 1 else Z)

    mu = GaussianMeasure(X, C, mean, name=name, _sampler=sampler,
                         _claims=[Claim(CIRCULAR, THEOREM, "z = L xi with xi standard circular and L L^H = S (to rounding: "
                                        "probability/gaussian.coordinate_factor, no jitter)",
                                        key="circular"),
                                  Claim("the covariance is trace class", THEOREM, "finitely many coordinates")])
    mu.coordinate_covariance = S
    return mu


def cvariance(mu, u):
    """E |<z - m, u>|^2 for a circular measure: twice the variance of the real functional <., u>_R."""
    if not mu.claims.find("circular"):
        raise Refused(f"{mu.name} carries no circularity claim: its complex variance is not twice the real one")
    return 2.0 * mu.variance(u)


def _complex_linear(A):
    """A's stored complex-linearity claim (not A.claims, which may form a dense matrix to check self-adjointness)."""
    from ..core.claims import Claims
    return Claims(getattr(A, "_claims", ())).find("complex-linear")


def pushforward_claims(mu, A):
    """The circularity claim of A(mu): derived when mu is circular and A is complex-linear (A J = J A)."""
    c = mu.claims.find("circular")
    if c is None or not c.holds:                                 # the common case: nothing to propagate, no work
        return []
    a = _complex_linear(A)
    if a is not None and a.holds:
        return [Claim(CIRCULAR, DERIVED, f"the image of a circular measure under the complex-linear {A.name} "
                      f"({a.status})", key="circular")]
    return []


def posterior_claims(prior, G, noise):
    """The circularity claim of the posterior: derived when prior and noise are circular and G is complex-linear (then
    the mean is complex-linear in d and the covariance C - C G* S^-1 G C commutes with J)."""
    cs = [m.claims.find("circular") for m in (prior, noise)]
    if not all(c is not None and c.holds for c in cs):
        return []
    a = _complex_linear(G)
    if a is not None and a.holds:
        return [Claim(CIRCULAR, DERIVED, f"circular prior and noise, complex-linear {G.name}: the posterior covariance "
                      "commutes with J", key="circular")]
    return []
