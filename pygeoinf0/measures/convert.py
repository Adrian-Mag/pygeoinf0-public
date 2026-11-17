"""Reading measures from what callers have (a measure, a set, points) and the quantities of a measure that the spaces
and functionals use: measure_dim, as_measure, as_measures, measure_of, total_mass, total_variation. Plan 067 P5: moved
from core/measures.py.
"""
import numpy as np

from ..sets import Region
from .base import Measure, SignedMeasure, _as_interval
from .lebesgue import DensityMeasure, LebesgueMeasure, carries_hausdorff
from .points import CountingMeasure, DiracMeasure


def measure_dim(mu):
    """The dimension k of the set carrying mu: 0 for point masses, 1 for intervals, paths and curves, 2 for surfaces,
    and so on (a k-dimensional Hausdorff measure); for a signed measure, the lowest dimension among its terms (the one
    that decides boundedness)."""
    if isinstance(mu, SignedMeasure):
        return min(measure_dim(m) for _, m in mu.terms)
    if isinstance(mu, (DiracMeasure, CountingMeasure)):
        return 0
    return int(mu.set.dim)


def as_measure(x):
    """A measure from a measure, a set (its Lebesgue/Hausdorff measure) or a point of R^d (its Dirac measure)."""
    if isinstance(x, Measure):
        return x
    if carries_hausdorff(x) or isinstance(x, str):
        return measure_of(x)
    a = np.asarray(x, float)
    if a.ndim <= 1:
        return DiracMeasure(a)
    raise TypeError(f"expected a measure, a set or a point, got {type(x).__name__}")


def as_measures(mu, cartesian_dim=None):
    """(list of measures, many) for M.riesz. Measures and sets are one item each; a list or tuple of them is many.
    Points are decided by shape against the Cartesian dimension d of the domain: numbers of shape (d,) are one point,
    (m, d) are m points (on an interval, d = 1: a number is one point, a sequence of numbers m points)."""
    from ..core.errors import SpaceMismatch
    if isinstance(mu, (Measure, Region, str)):
        return [as_measure(mu)], False
    if isinstance(mu, (list, tuple)) and len(mu) and not all(np.isscalar(v) and not isinstance(v, str) for v in mu) \
            and not all(isinstance(v, np.ndarray) and v.ndim == 0 for v in mu):
        if all(isinstance(v, (Measure, Region, str)) for v in mu):
            return [as_measure(m) for m in mu], True
        a = np.asarray(mu, float) if all(np.ndim(v) == 1 for v in mu) else None
        if a is None:                                # a mixed list: (m, d) arrays inside it are m points
            out = []
            for v in mu:
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    out += as_measures(v, cartesian_dim)[0]
                else:
                    out.append(as_measure(v))
            return out, True
        mu = a
    a = np.asarray(mu, float)
    d = cartesian_dim
    if d is None:
        return ([DiracMeasure(p) for p in a], True) if a.ndim == 2 else ([DiracMeasure(a)], False)
    if d == 1:
        if a.ndim == 0:
            return [DiracMeasure(float(a))], False
        return [DiracMeasure(float(v)) for v in a.ravel()], True
    if a.ndim == 1 and a.size == d:
        return [DiracMeasure(a)], False
    if a.ndim == 2 and a.shape[1] == d:
        return [DiracMeasure(p) for p in a], True
    raise SpaceMismatch(f"points of shape {a.shape} do not fit a domain with {d} Cartesian coordinates")


def measure_of(I):
    """Lebesgue measure for an interval (Interval or string) or a domain; measures pass through."""
    if isinstance(I, Measure):
        return I
    I = _as_interval(I)
    if carries_hausdorff(I):
        return LebesgueMeasure(I)
    raise TypeError(f"expected a measure, an interval or a domain, got {I!r}")


def total_mass(mu):
    """mu(S) for the set S carrying mu: mu.mass()[0] (exact, or by quadrature for callable densities, whose mass()
    gives the estimate); for a signed measure the signed total."""
    return mu.mass()[0]


def total_variation(mu):
    """|mu|(S): the norm of integration against mu on spaces of continuous functions (Riesz-Markov)."""
    if isinstance(mu, SignedMeasure):
        return float(sum(abs(c) * total_variation(m) for c, m in mu.terms))   # an upper bound when terms overlap
    if isinstance(mu, DensityMeasure):
        return mu.variation()
    w = getattr(mu, "weights", None)
    return float(np.sum(np.abs(w))) if w is not None else float(total_mass(mu))


