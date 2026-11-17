"""Slab-type noise in box BGP (plan 060; moved from convex/slabs.py by plan 070 P4, the rows of a set in
convex/forms.py slab_rows): slabs, intersections of slabs and polytopes with paired half-spaces.

Noise {e : lo_i <= A_i . e <= hi_i, i = 1..k} (A_i the plain-coordinate row of a slab's normal a_i, A_i . r =
<a_i, r>_D; translates shift lo, hi): e - G m in N is a coordinate box on R^k for A (e - G m), so box BGP runs in R^k
with G' = A G and e' = A e (convex/forms.py slab_rows, reduced). Refused with the reason: a zero normal, a negative or infinite
width, a polytope half-space without its opposite (a one-sided constraint). Dependent rows are kept (redundant or
contradictory constraints: the latter make the set empty, which the dual certifies).

One slab on L2 of an interval (one_slab): with a~ = G*(W^-1 A^T), <a~, m> = A . G m in [alpha, beta],

    U = sup <c, m> = min over lambda of phi(lambda) = sigma_B(c - lambda a~) + max(lambda beta, lambda alpha)

(THEOREM, weak duality: for alpha <= g <= beta, lambda g <= max(lambda beta, lambda alpha); phi convex with the
monotone subgradient -<a~, m_lambda> + (beta or alpha), m_lambda = u on {c - lambda a~ > 0}, l on {< 0}). lambda by
bracketing and bisection; each phi(lambda) is certified (sigma_B exact on intervals). The witness mixes the
bang-bang fields at the two ends of the final bracket so that <a~, m> meets the active side, which also covers a
zero set of c - lambda* a~ of positive measure; it is an element of the space, checked in the box and in the slab
(the space's exact inner product). EXACT when the bracket closes. A target with no constraint active (lambda* = 0)
takes the box's own extremal field.
"""
import math

import numpy as np

from ...core.errors import Unsupported
from ...core.precision import TOL
from .bound import CERTIFIED, Bound


def reduced(cp, A):
    """The cell problem for the residual A (e - G m) (cells unchanged, data images through A)."""
    from .nonsmooth import CellProblem
    out = CellProblem(cp.lo, cp.hi, cp.t, apply=lambda v: A @ cp.apply(v), adjoint=lambda z: cp.adjoint(A.T @ z),
                      fixed_data=0.0, fixed_target=cp.fixed_target, note=cp.note)
    out.images, out.key = cp.images, None if cp.key is None else ("slab", cp.key)
    return out


# ------------------------------------------------------------------------------------------------ one slab, exact
def _bang(M, q, l, u):
    """The element u on {q > 0}, l on {q <= 0} (roots of q per piece, exact; the bounds restricted per piece)."""
    from ...spaces.elements import Function
    from ...realisations.series.legendre import PiecewiseLegendre, refine, restrict, union_breaks
    from ...realisations.bounds.box_integrals import signs
    pls = [q.realisation] + [b.realisation for b in (l, u) if not isinstance(b, float)]
    nb = union_breaks(*[p.breaks for p in pls])
    qc = refine(q.realisation, nb)

    def coef(b):
        return [None] * (len(nb) - 1) if isinstance(b, float) else refine(b.realisation, nb)
    lc, uc = coef(l), coef(u)
    br, cs = [nb[0]], []
    for i in range(len(nb) - 1):
        x0, x1 = nb[i], nb[i + 1]
        c = np.trim_zeros(qc[i][:, 0], "b")
        pts, sg = signs(c) if len(c) else (np.array([-1.0, 1.0]), np.array([0.0]))
        for k, s in enumerate(sg):
            if pts[k + 1] <= pts[k]:
                continue
            b = x0 + 0.5 * (x1 - x0) * (pts[k + 1] + 1)
            if b <= br[-1]:
                continue
            bound, C = (u, uc) if s > 0 else (l, lc)
            cs.append(np.array([[bound]]) if isinstance(bound, float) else restrict(C[i], x0, x1, br[-1], b))
            br.append(b)
    br[-1] = nb[-1]
    return Function(M, pl=PiecewiseLegendre(br, cs, np.zeros(1)), name="bang-bang")


def one_slab(box, G, cache, c, row, lo, hi, e, iters=200):
    """The Bound of sup <c, m> over the box and one slab (module docstring), on L2 of an interval: box the pointwise
    box prior (its space, bounds, support and membership), G the data map, cache a dict owned by the feasible set."""
    M, D = box.space, G.codomain
    if ("slab", "a") not in cache:
        cache["slab", "a"] = G.adjoint(D(D.solve_metric(row)))
    at = cache["slab", "a"]
    s = float(row @ e)
    alpha, beta = s - hi, s - lo                                # alpha <= <a~, m> <= beta
    l, u = box.lower[0], box.upper[0]

    def field(lam):
        m = _bang(M, c - at * lam, l, u)
        return m, float(M.inner(at, m))

    def phi(lam):
        return box.support(c - at * lam).upper + max(lam * beta, lam * alpha)

    gmin, gmax = -box.support(at * -1.0).upper, box.support(at).upper
    if gmin > beta + TOL * max(1.0, abs(beta)) or gmax < alpha - TOL * max(1.0, abs(alpha)):
        side = beta if gmin > beta else alpha
        return Bound(-math.inf, -math.inf, -math.inf, "EMPTY", None, None,
                     f"K(d) is empty: the box gives <a, G m> in [{gmin:.6g}, {gmax:.6g}], the slab needs "
                     f"[{alpha:.6g}, {beta:.6g}] ({side:.6g} out of reach; exact supports)", claim=CERTIFIED)
    m0, g0 = field(0.0)
    if alpha <= g0 <= beta:                                     # no constraint active
        up = phi(0.0)
        return _result(M, c, m0, up, 0.0, "no slab side active")
    sgn = 1.0 if g0 > beta else -1.0                           # lambda > 0 lowers g, lambda < 0 raises it
    target = beta if sgn > 0 else alpha
    scale = max(float(M.norm(c)) / max(float(M.norm(at)), 1e-300), 1e-300)
    a, ga, ma = 0.0, g0, m0
    b = sgn * scale
    mb, gb = field(b)
    for _ in range(iters):                                      # bracket: sgn (g - target) changes sign
        if sgn * (gb - target) <= 0:
            break
        a, ga, ma = b, gb, mb
        b *= 2.0
        mb, gb = field(b)
    else:
        raise Unsupported("the one-slab dual did not bracket its root (the slab side is reached only in the limit)")
    for _ in range(iters):                                      # bisection on the monotone subgradient
        if abs(b - a) <= 4 * np.finfo(float).eps * max(abs(a), abs(b)):
            break
        mid = 0.5 * (a + b)
        mm, gm = field(mid)
        if sgn * (gm - target) > 0:
            a, ga, ma = mid, gm, mm
        else:
            b, gb, mb = mid, gm, mm
    # the witness: theta ma + (1 - theta) mb with <a~, m> on the active side, just inside it
    inside = target - sgn * 8 * np.finfo(float).eps * max(abs(target), abs(ga), abs(gb), 1.0)
    th = 0.0 if sgn * (gb - inside) >= 0 else min(max((inside - gb) / (ga - gb), 0.0), 1.0)
    m = ma * th + mb * (1.0 - th) if th > 0 else mb
    up = min(phi(a), phi(b), phi(0.0))
    return _result(M, c, m, up, b, f"lambda in [{min(a, b):.12g}, {max(a, b):.12g}]", box, at, alpha, beta)


def _result(M, c, m, up, lam, note, box=None, at=None, alpha=None, beta=None):
    from ...core.logic import Truth
    ok = True
    if box is not None:
        g = float(M.inner(at, m))
        ok = alpha <= g <= beta and box.member(m) is Truth.TRUE
    lower = float(M.inner(c, m)) if ok else -math.inf
    status = ("EXACT" if up - lower <= TOL * max(1.0, abs(up)) else "BRACKET") if ok else "OUTER"
    return Bound(up if status == "OUTER" else lower, lower, up, status, np.array([lam]), m if ok else None,
                 f"box prior, one slab of noise: exact 1-D dual ({note}) and a bang-bang witness; gap {up - lower:.3g}", claim=CERTIFIED)
