"""The property space and the model-property relation (plan 070 P1; architecture review D5, decision Q10).

Two of the seven elements of an inference problem (inference/problem.py: M, D, mdr, P, mpr, prior, calibration) are
here:

- P, the property space: the space of the properties one wants to know about the model. The families here map into a
  finite-dimensional P = R^p with the dot product (pg.Euclidean(p)), one coordinate per property.
- mpr, the model-property relation: a relation from M to P. The families here are single-valued: linear operators
  T: M -> P whose rows are bounded functionals on M (their claims state linearity and boundedness, with the reason).
  A Problem takes one as its mpr (pg.Problem(M=M, P=T.codomain, mpr=T)).

The families are factored, T = Phi tau (families.py): tau a separable family of functionals on M (operators/
separable.py) and Phi a fixed p x B matrix, so the methods work on the B blocks and never form a p x p matrix
(solvers/dataspace/factored.py; their lazy covariances are probability/lazy_covariance.py).

    pg.CapAverages(M, radius, centres, ...)     averages of a field on a sphere over caps (Funk-Hecke)       families.py
    pg.RotatedFamily(T0, centres)               the rotations of a zonal functional (the addition theorem)   families.py
    pg.harmonic_target(T, degree)               a target kept symbolic, by its harmonic coefficients          harmonic_targets.py

The numerics they use but lower layers use too stay below: the diagonal rowquad with the lazy covariance in probability/,
and the harmonic coefficients of a symbolic density and the box holding a measure's support in
realisations/series/symbolic_harmonics.py.
"""
from .families import CapAverages, FactoredFamily, RotatedFamily, directions
from .harmonic_targets import harmonic_target

__all__ = ["CapAverages", "RotatedFamily", "FactoredFamily", "harmonic_target", "directions"]
