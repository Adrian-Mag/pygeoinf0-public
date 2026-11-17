"""Linear programming for the support functions of polytopes in R^n: max c.x s.t. A x <= b with x free (moved from
convex/ by plan 063). Two backends:
    "simplex" (default)  a two-phase dense tableau simplex with Bland's rule (no cycling), exact to rounding for the small
                         problems of the support functions; what every certified route uses
    "highs"              scipy's HiGHS, for large sparse problems: its result is optimal to HiGHS's feasibility and optimality
                         tolerances (1e-7 by default), stated, not to rounding. HiGHS starts native threads on first use, and
                         forking worker processes after that (PYGEOINF0_WORKERS) risks deadlocks: it is refused while
                         settings.workers > 1."""
import math

import numpy as np

TOL = 1e-11


def _pivot(T, basis, i, j):
    T[i] /= T[i, j]
    for k in range(T.shape[0]):
        if k != i and T[k, j] != 0.0:
            T[k] -= T[k, j] * T[i]
    basis[i] = j


def _simplex(T, basis, obj, allowed, max_iter=10000):
    """Maximise obj . z over the tableau (rows: constraints, last column: values); True when optimal, False when
    unbounded. Bland's rule: the entering column of least index with positive reduced cost, ties in the ratio test by
    the least basic index."""
    scale = max(1.0, float(np.abs(obj).max(initial=0.0)))
    for _ in range(max_iter):
        reduced = obj - obj[basis] @ T[:, :-1]
        enter = next((j for j in allowed if reduced[j] > TOL * scale), None)
        if enter is None:
            return True
        col = T[:, enter]
        rows = [i for i in range(T.shape[0]) if col[i] > TOL]
        if not rows:
            return False
        ratios = [(T[i, -1] / col[i], basis[i], i) for i in rows]
        best = min(r for r, _, _ in ratios)
        i = min((bi, i) for r, bi, i in ratios if r <= best + TOL * max(1.0, abs(best)))[1]
        _pivot(T, basis, i, enter)
    raise ArithmeticError("the simplex did not terminate")


def lp_max(c, A, b, backend="simplex"):
    """(max c.x s.t. A x <= b, x): +inf if unbounded, -inf if infeasible (module docstring)."""
    if backend == "highs":
        return _highs(c, A, b)
    if backend != "simplex":
        raise ValueError(f"no LP backend {backend!r}; the backends are 'simplex' and 'highs'")
    c, A, b = np.asarray(c, float).ravel(), np.atleast_2d(np.asarray(A, float)), np.asarray(b, float).ravel()
    m, n = A.shape
    sign = np.where(b < 0, -1.0, 1.0)[:, None]
    T = np.zeros((m, 2 * n + 2 * m + 1))                       # columns: u, v (x = u - v), slacks, artificials, b
    T[:, :n], T[:, n:2 * n] = sign * A, -sign * A
    T[:, 2 * n:2 * n + m] = np.diag(sign.ravel())
    T[:, 2 * n + m:2 * n + 2 * m] = np.eye(m)
    T[:, -1] = sign.ravel() * b
    basis = list(range(2 * n + m, 2 * n + 2 * m))
    art = set(basis)
    obj1 = np.zeros(2 * n + 2 * m)
    obj1[2 * n + m:] = -1.0
    _simplex(T, basis, obj1, range(2 * n + 2 * m))
    if -obj1[basis] @ T[:, -1] > TOL * max(1.0, float(np.abs(b).max(initial=0.0))):
        return -math.inf, None                                  # phase I could not remove the artificials
    keep = []
    for i in range(m):                                          # drive the artificials out of the basis
        if basis[i] in art:
            j = next((j for j in range(2 * n + m) if abs(T[i, j]) > TOL), None)
            if j is None:
                continue                                        # a redundant row
            _pivot(T, basis, i, j)
        keep.append(i)
    T, basis = T[keep], [basis[i] for i in keep]
    obj2 = np.zeros(2 * n + 2 * m)
    obj2[:n], obj2[n:2 * n] = c, -c
    if not _simplex(T, basis, obj2, range(2 * n + m)):
        return math.inf, None
    z = np.zeros(2 * n + 2 * m)
    z[basis] = T[:, -1]
    x = z[:n] - z[n:2 * n]
    return float(c @ x), x


def _highs(c, A, b):
    from ..core.errors import Refused
    from ..execution import workers
    if workers() > 1:
        raise Refused("HiGHS starts native threads, and forking the worker processes after that risks deadlocks",
                      needs=["backend='simplex'", "PYGEOINF0_WORKERS=1"])
    from scipy.optimize import linprog
    c = np.asarray(c, float).ravel()
    r = linprog(-c, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
    if r.status == 0:
        return float(c @ r.x), r.x
    if r.status == 2:
        return -math.inf, None
    if r.status == 3:
        return math.inf, None
    raise Refused(f"HiGHS stopped without a solution: {r.message}", needs=["backend='simplex'"])
