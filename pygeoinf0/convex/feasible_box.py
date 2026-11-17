"""The feasible set's route for a pointwise-box prior (plans 009 P5, 057, 059, 060, 061; moved out of convex/nonsmooth.py
by plan 070 P4): K = {m in B : e - G m in N} to its Bound and its emptiness. This module reads K (its prior, data map,
noise set and datum) and the noise set's form (convex/forms.py); the certified non-smooth dual, the cell witnesses and
the one-slab dual are optimisers/dual/nonsmooth.py, sphere_cells.py and slabs.py, called through optimisers.dual.

The noise side is a ball, an ellipsoid or a coordinate box (forms.noise_side), or slab-type (forms.slab_rows: the cells
in the slabs' coordinates, and one slab on L2 of an interval by its exact 1-D dual). Interval cells are kept per edges
and Bounds per direction on K (plan 059): K's data and noise are fixed after construction.
"""
import math

import numpy as np

from ..core.logic import Truth
from ..optimisers.dual import (CERTIFIED, Bound, CellDual, NoiseSide, auto_cells, emptiness, one_slab, plain_interval,
                               reduced, solve_cells)
from .forms import noise_side, slab_rows


def noise_of(K):
    """(NoiseSide, e, A) of K's noise: A None for a ball, an ellipsoid or a coordinate box; for slab-type noise the
    rows A (convex/forms.py slab_rows), the noise side then the coordinate box of A (e - G m) and e = A d (plan 060)."""
    if "_noise" not in K.__dict__:
        D = K.G.codomain
        e = np.asarray(K.d.values, float)
        rows = slab_rows(K.noise, D)
        if rows is None:
            K._noise = (noise_side(K.noise, D), e, None)
        else:
            A, lo, hi = rows
            K._noise = (NoiseSide.box(lo, hi), A @ e, A)
    return K._noise


def feasible_bound(K, c, cells=None, sub=None, rounds=10):
    """sup <c, m> over a feasible set K whose prior is a pointwise box, with the certified bracket (module docstring).
    cells: a CellProblem, a callable c -> CellProblem, or None (automatic: interval cells, `sub` per piece, or on
    sphere summands latitude-longitude cells, sub (L + 1) bands; default K.sub). One
    slab of noise on L2 of an interval takes the exact 1-D dual (optimisers/dual/slabs.py one_slab); other slab-type noise the
    cells in the slabs' coordinates."""
    D, G = K.G.codomain, K.G
    box = K.prior
    noise, e, A = noise_of(K)
    sub = getattr(K, "sub", 4) if sub is None else sub
    memo, key = None, None
    if cells is None:                                         # interval cells: remembered per direction (plan 059)
        R = getattr(c, "realisation", None)
        if R is not None and hasattr(R, "coefs"):
            memo = K.__dict__.setdefault("_bound_memo", {})
            key = (sub, rounds, R.breaks.tobytes(), b"".join(np.ascontiguousarray(C).tobytes() for C in R.coefs))
            if key in memo:
                return memo[key]
    if A is not None and len(A) == 1 and cells is None and plain_interval(box):
        b = one_slab(box, G, K.__dict__.setdefault("_cell_cache", {}), c, A[0], noise.lo[0], noise.hi[0],
                     np.asarray(K.d.values, float))
    else:
        cp = auto_cells(box, G, c, sub, K.__dict__.setdefault("_cell_cache", {})) if cells is None else \
            (cells(c) if callable(cells) else cells)
        if A is not None:
            cp = reduced(cp, A)

        def upper(z):
            zz = z if A is None else A.T @ z
            return box.support(c - G.adjoint(D(D.solve_metric(zz)))).upper
        vf = None
        if getattr(cp, "key", None) is not None:             # the interior point depends on the cells, not on c
            cache = K.__dict__.setdefault("_cell_cache", {})
            vf = cache.get(("vf", cp.key))
            if vf is None:
                vf = cache["vf", cp.key] = CellDual(cp, noise, e).interior()
        b = solve_cells(cp, noise, e, upper, interior=vf, rounds=rounds, note=cp.note.partition("; ")[2])
        b.cells = cp
    if memo is not None:
        if len(memo) > 4096:
            memo.clear()
        memo[key] = b
    return b


def box_empty(K):
    """Truth of K = {} for a box prior: FALSE with a witness, TRUE with a certificate in K.emptiness (a certified
    Farkas certificate, or the one-slab route's exact supports), else UNKNOWN."""
    D, G = K.G.codomain, K.G
    zero = K.space.zero
    b = feasible_bound(K, zero, K.cells)
    if math.isfinite(b.lower):
        return Truth.FALSE
    if b.status == "EMPTY":
        K.emptiness = b
        return Truth.TRUE
    noise, e, A = noise_of(K)
    cp = auto_cells(K.prior, G, zero, K.sub, K.__dict__.setdefault("_cell_cache", {})) if K.cells is None else \
        (K.cells(zero) if callable(K.cells) else K.cells)
    if A is not None:
        cp = reduced(cp, A)
    far = emptiness(cp, noise, e, lambda z: K.prior.support(
        -G.adjoint(D(D.solve_metric(z if A is None else A.T @ z)))).upper)
    if far is not None:
        K.emptiness = Bound(-math.inf, -math.inf, -math.inf, "EMPTY", far[0], None,
                            f"K(d) is empty: phi(lambda, 0) <= {far[1]:.3g} < 0 (certified Farkas certificate)", claim=CERTIFIED)
        return Truth.TRUE
    return Truth.UNKNOWN
