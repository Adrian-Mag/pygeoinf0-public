"""The one wrapper of scipy.optimize (plan 070 P4, architecture review D6): generic minimisation and root finding with
the precision stated by the caller.

    minimise(f, x0, method, precision, jac=, bounds=, constraints=)    scipy.optimize.minimize
    minimise_scalar(f, bounds, xatol, maxiter=)                         scipy.optimize.minimize_scalar, bounded
    least_squares_box(A, b, bounds, max_iter)                           scipy.optimize.lsq_linear, trust-region reflective
    root(f, lo, hi, xtol, rtol, maxiter)                                scipy.optimize.brentq

Every call states its precision: the tolerances are required arguments (no scipy defaults), passed to scipy unchanged,
and the result records them (res.precision, res.method). The result is the optimiser's candidate, not a certified
value: the callers in pygeoinf0 certify it themselves (a dual bound is valid at any multiplier, a witness is checked
against its constraints; optimisers/dual/), or state its precision in their claims. Exceptions raised by f (a budget
of evaluations, say) propagate. Special-function root finding (the zeros of Bessel functions, chi-square laws) stays
where it is (D6).
"""


def _stated(res, method, precision):
    res.method, res.precision = method, dict(precision)
    return res


def minimise(f, x0, method, precision, jac=None, bounds=None, constraints=None):
    """scipy.optimize.minimize(f, x0, method=method, options=precision) with jac, bounds and constraints passed only
    when given: the OptimizeResult with .method and .precision (the options as stated)."""
    from scipy.optimize import minimize
    kw = {k: v for k, v in (("jac", jac), ("bounds", bounds), ("constraints", constraints)) if v is not None}
    return _stated(minimize(f, x0, method=method, options=dict(precision), **kw), method, precision)


def minimise_scalar(f, bounds, xatol, maxiter=None):
    """The bounded scalar minimisation of f on bounds (Brent's method with golden sections) to xatol (and at most maxiter
    iterations when given): the OptimizeResult with .method and .precision."""
    from scipy.optimize import minimize_scalar
    precision = {"xatol": xatol} if maxiter is None else {"xatol": xatol, "maxiter": maxiter}
    return _stated(minimize_scalar(f, bounds=bounds, method="bounded", options=dict(precision)), "bounded", precision)


def least_squares_box(A, b, bounds, max_iter):
    """min ||A v - b|| over a box of v (bounds = (lo, hi)) by the trust-region reflective method with matrix-free LSMR
    steps (lsmr_tol="auto"), at most max_iter iterations: the OptimizeResult with .method and .precision."""
    from scipy.optimize import lsq_linear
    precision = {"lsmr_tol": "auto", "max_iter": max_iter}
    return _stated(lsq_linear(A, b, bounds=bounds, method="trf", lsmr_tol="auto", max_iter=max_iter), "trf", precision)


def root(f, lo, hi, xtol, rtol, maxiter):
    """A root of f in [lo, hi] (f(lo) and f(hi) of opposite signs) by Brent's method to xtol + rtol |x|, at most maxiter
    iterations (scipy.optimize.brentq; a RuntimeError if it does not converge)."""
    from scipy.optimize import brentq
    return brentq(f, lo, hi, xtol=xtol, rtol=rtol, maxiter=maxiter)
