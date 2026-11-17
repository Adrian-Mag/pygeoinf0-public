"""Iterative solvers: conjugate gradients in the domain's inner product, matrix-free (only products with the operator)."""
import math

import numpy as np

from ..spaces.elements import Family, Function, Vector, VectorFamily
from ..core.errors import Refused
from .direct import Solver, _about


class CG(Solver):
    """Conjugate gradients in the domain's inner product (self-adjoint positive definite operators), matrix-free.
    Stops when the residual norm is at most max(rtol ||y||, atol), or with tighter=True the smaller of the two. Refuses
    when it stops short of that: a result it did not reach is not returned."""

    about = _about("the inverse, to a stated residual", "positive definite, n above settings.dense_max or a function space: only "
                   "products with the operator", "the residual is bounded and stated; the error in x is at most cond times the "
                   "relative residual, and the condition is not known to it; refused if it stops short",
                   "a few products per digit, depending on the spectrum; no matrix",
                   "stopping at a tolerance: a residual bound, not an error bound")

    def __init__(self, rtol=1e-10, maxiter=None, atol=0.0, tighter=False):
        self.rtol, self.maxiter, self.atol, self.tighter = rtol, maxiter, float(atol), tighter
        self.name = f"CG(rtol={rtol:g}" + (f", atol={atol:.3g}" if atol else "") + (", the tighter)" if tighter else ")")

    def factor(self, A):
        return _CGSolve(A, self.rtol, self.maxiter, self.atol, self.tighter)


class _CGSolve:
    def __init__(self, A, rtol, maxiter, atol=0.0, tighter=False):
        self.A, self.rtol, self.maxiter, self.atol, self.tighter = A, rtol, maxiter, atol, tighter
        self.info = ["conjugate gradients in the inner product of the space"]

    def _one(self, y):
        A, sp = self.A, self.A.domain
        x = sp.zero
        r, p = y, y
        rr = sp.inner(r, r)
        stop = (min if self.tighter else max)(self.rtol * math.sqrt(rr), self.atol) ** 2
        from ..spaces.base import Euclidean
        maxiter = self.maxiter or (10 * sp.dim if isinstance(sp, Euclidean) else 1000)   # rounding: n steps are not enough
        it = 0
        while rr > stop and it < maxiter:
            Ap = A(p)
            pAp = sp.inner(p, Ap)
            if not pAp > 0:
                raise Refused(f"{A.name}: conjugate gradients broke down, <p, A p> = {pAp:.3g} after {it} iterations: it is "
                              "not numerically positive definite here", needs=["a direct solve", "A.pseudo_inverse(rcond=...)"])
            alpha = rr / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = sp.inner(r, r)
            p = r + (rr_new / rr) * p
            rr = rr_new
            it += 1
        if rr > stop:
            raise Refused(f"{A.name}: conjugate gradients stopped at residual {math.sqrt(rr):.3g} after {it} iterations, "
                          f"short of {math.sqrt(stop):.3g}", needs=["a larger maxiter", "a direct solve"])
        self.info = [f"conjugate gradients: {it} iterations, residual {math.sqrt(rr):.3g} (stop at {math.sqrt(stop):.3g}); "
                     "the residual is bounded, the error in x also depends on the condition number"]
        return x

    def solve(self, y):
        if isinstance(y, (Vector, Function)):
            return self._one(y)
        if isinstance(y, VectorFamily):
            return VectorFamily(self.A.domain, np.array([self._one(v).values for v in y]), name="X")
        if isinstance(y, Family):
            return self.A.domain.family([self._one(f) for f in y])
        raise TypeError(type(y).__name__)
