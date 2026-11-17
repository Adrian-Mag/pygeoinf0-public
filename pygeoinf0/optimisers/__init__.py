"""Optimisers (plan 063): the methods, separate from the problems that call them.

    gauss_newton(G, d, theta, alpha=, theta0=)   Levenberg-Marquardt for 1/2 ||G(theta) - d||^2 + alpha/2 ||theta - theta0||^2
    lbfgs(J, theta, alpha=, theta0=)             L-BFGS in the model space's inner product for J + alpha/2 ||theta - theta0||^2
    lp_max(c, A, b, backend="simplex")           max c.x s.t. A x <= b; backends "simplex" (exact to rounding) and "highs"

smooth.py holds the first two (written only against the port and the spaces' inner products), lp.py the linear programs,
scalar.py the certified one-dimensional searches of the support-function routes (bisect, trs, trs_columns), minimise.py
the one wrapper of scipy.optimize (stated precision), dual/ the dual solvers of BGP (plan 070 P4: moved out of convex/,
which calls them through optimisers.dual.__all__). Convex sets, support functions, the set algebra and the feasible set
stay in pygeoinf0.convex. Plan 055's bundle, primal-dual and proximal methods come here. Robust
losses, sampling and other optimisers are plugins (the author, 2026-10-30).
"""
from .lp import lp_max
from .smooth import NonlinearResult, gauss_newton, lbfgs

__all__ = ["NonlinearResult", "gauss_newton", "lbfgs", "lp_max"]
