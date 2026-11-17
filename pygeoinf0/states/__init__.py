"""States of knowledge on a space (plan 008; plan 067 P4): the State protocol, the kinds set, measure and ordering
(core/kinds.py), and the conversions between them (conversions.py). Where each kind's classes live:

    set        pygeoinf0.sets (Subset, PredicateSubset) and pygeoinf0.convex (ConvexSet, NormBall, Ellipsoid, the
               subspaces Whole, Zero, Span, Complement, FeasibleSet); the prior "none" is NoPrior(X) = Whole(X)
    measure    pygeoinf0.probability (ProbabilityMeasure, GaussianMeasure, Dirac, DiscreteMeasure, CylindricalGaussian)
    ordering   Ordering (states/base.py; declared, not developed)
"""
from .base import NoPrior, Ordering, State, is_element, state_kind
from .conversions import as_measure, as_set, harden

__all__ = ["State", "NoPrior", "Ordering", "state_kind", "is_element", "as_set", "as_measure", "harden"]
