"""Solvers for A x = y, and a recommendation that says why.

Cholesky()      self-adjoint positive definite operators between Euclidean spaces (direct, n^3/3 flops)
Direct()        self-adjoint operators between Euclidean spaces, ill conditioned: nothing is dropped; the condition
                and the digits it costs are stated, and a direction with no correct digit is refused (below)
Eigh(rcond)     the pseudo-inverse, asked for with its threshold: eigenvalues below rcond * max are dropped
LU()            any invertible square operator between Euclidean spaces
CG(rtol)        self-adjoint positive definite operators on any Hilbert space; it only applies A and takes
                inner products, so it also works on function spaces

All of them work in the inner product of the space: for R^n with metric W the self-adjoint form is W A.

A.inv is the inverse, never a substitute (plan 054). An eigenvalue with |lambda| <= eps n max |lambda| has no correct digit
in double precision: the inverse is refused and the refusal names the choices. A.pseudo_inverse(rcond) and a shift are
mathematical choices for the caller to make. The digits stated count rounding only; errors in A itself come on top.

damping_for_misfit(A, d, target) is the damping alpha at which the damped residual has the stated squared norm;
choosing the target from the datum (the discrepancy principle) is a meta rule, pg.meta.discrepancy (plan 070 P2).

dataspace/ is the data-space engine (plan 070 P3): one decomposition of the whitened data Gram matrix and its closed
forms, internal to the methods, its metric stated explicitly; pg.decompose is its public route (solvers/dataspace/).
"""
from .backends import backends
from .choice import Alpha, Recommendation, damping_for_misfit, recommend, solve_damping, usage
from .direct import (
                     LU,
                     BlockDirect,
                     Cholesky,
                     Direct,
                     Eigh,
                     KroneckerEigh,
                     Solver,
                     inherited,
                     inherited_refusal,
                     pseudo_solver,
                     rounding_claim,
                     stated_error,
)
from .iterative import CG

__all__ = ["Alpha", "BlockDirect", "CG", "Cholesky", "Direct", "Eigh", "KroneckerEigh", "LU", "Recommendation", "Solver",
           "backends", "damping_for_misfit", "inherited", "inherited_refusal", "pseudo_solver", "recommend",
           "rounding_claim", "solve_damping", "stated_error", "usage"]
