"""Measures on finitely many points of R^d: CountingMeasure (weighted) and DiracMeasure, the unit point mass of a
domain (the evaluation functionals' measure; the point mass on a model space is pygeoinf0.probability.Dirac). Plan 067
P5: moved from core/measures.py.
"""
import numpy as np

from ..core.claims import THEOREM, Claim
from ..core.logic import Truth
from ..sets import FiniteSet, Region
from .base import Measure, _as_interval


def _points(S):
    """The points of a finite set as an (n, d) array."""
    return S.points if S.points.ndim == 2 else S.points[:, None]


class CountingMeasure(Measure):
    """Counting measure on a finite set, optionally weighted: mu(A) = sum of the weights of the points in A.
    L2 of a weighted counting measure on n points is R^n with the weighted dot product (see Euclidean)."""

    sigma_algebra = "all subsets"

    def __init__(self, points, weights=None):
        self.set = points if isinstance(points, FiniteSet) else FiniteSet(points)
        w = np.ones(len(self.set)) if weights is None else np.asarray(weights, float)
        if w.shape != (len(self.set),) or np.any(w <= 0):
            raise ValueError("counting-measure weights must be positive, one per point")
        self.weights = w

    def __call__(self, A):
        A = _as_interval(A)
        P = _points(self.set)
        if not (isinstance(A, Region) and A.decides_membership and A.cartesian_dim == P.shape[1]):
            raise TypeError(f"cannot measure {A!r}")
        return float(self.weights[np.atleast_1d(A.contains(P))].sum())

    def mass(self):
        return float(self.weights.sum()), 0.0

    def integrate(self, f):
        """(sum_i w_i f(p_i), 0.0) for f a function of the Cartesian coordinates (as for LebesgueMeasure.integrate)."""
        from .weights import _call
        return float(self.weights @ _call(f, _points(self.set))), 0.0

    def _sign(self):
        return Truth.FALSE, THEOREM, "positive weights"

    def _own_claims(self):
        return [Claim("a measure", THEOREM, "non-negative weights, countably additive on a finite set")]

    def __repr__(self):
        return f"counting measure on {self.set}" + ("" if np.all(self.weights == 1) else " (weighted)")


class DiracMeasure(Measure):
    """The unit point mass at x0, a number or a point of R^d: mu(A) = 1 if x0 is in A, else 0."""

    sigma_algebra = "all subsets"

    def __init__(self, x0):
        x = np.asarray(x0, float)
        self.x0 = float(x) if x.ndim == 0 else x.copy()
        self.set = FiniteSet([x0] if x.ndim else [float(x)])

    def __call__(self, A):
        A = _as_interval(A)
        return 1.0 if np.atleast_1d(self.x0) in A else 0.0        # raises when membership is not decided

    def mass(self):
        return 1.0, 0.0

    def integrate(self, f):
        """(f(x0), 0.0) for f a function of the Cartesian coordinates (as for LebesgueMeasure.integrate)."""
        from .weights import _call
        return float(_call(f, np.atleast_1d(np.asarray(self.x0, float))[None, :])[0]), 0.0

    def _properties(self):
        why = "the unit point mass at a point"
        return {"finite": (Truth.TRUE, THEOREM, why), "probability": (Truth.TRUE, THEOREM, why),
                "signed": (Truth.FALSE, THEOREM, why)}

    def __repr__(self):
        x = np.atleast_1d(self.x0)
        return f"Dirac measure at {x[0]:g}" if x.size == 1 else f"Dirac measure at ({', '.join(f'{v:g}' for v in x)})"


