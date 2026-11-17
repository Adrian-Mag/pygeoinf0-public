"""Quadrature rules adapted to a weight (plan 022 P2): the weight decides where a rule is refined, so that a narrow
or compactly supported density is integrated to the stated accuracy by every consumer (pairings with formulas,
the pieces of weighted segments) instead of being sampled on a rule fixed in advance.

    adapted_panels(edges, n, w)    composite Gauss on an interval, panels bisected where w is not resolved
    adapted_edges(edges, n, w)     the accepted panels themselves
    adapted_square(n, w)           the 2-D analogue on the reference square (cells quartered)
    adapted_nodes_on(S, n, w)      weights.nodes_on refined for w: intervals, 2-D boxes, other sets by order doubling

A cell is accepted when its rule and its children's give int w and the first moments of w within ADAPT_TOL times
the total int |w|: a rule, not a claim; cells where w vanishes at every node are accepted at once (a feature
between the starting nodes can be missed, as with any sampling of a callable). Cells still unresolved after
ADAPT_DEPTH bisections (or n_max) are kept, and where their largest disagreement exceeds TOL = 1e-9 of int |w| (fewer
than 9 digits, the general rule) a RealisationWarning gives it with their count (an estimate of what is lost there, not a
bound): the flag was returned and every caller discarded it (plan 054, wrong row Q6; measured: |x - 0.3|^-0.9 undeclared, 3.8e-2 relative lost without a word).
"""
import warnings

import numpy as np

from ..measures.weights import nodes_on
from ..quadrature import gauss01

ADAPT_TOL = 1e-13            # weight-adapted rules (plan 022 P2): per cell, relative to the total int |w|
ADAPT_DEPTH = 40             # bisections at most; a cell still unresolved there is kept and reported


def _unresolved(where, count, worst):
    """The warning of a rule left unresolved, by the general rule: where what is lost may exceed TOL of int |w| (fewer
    than 9 digits); below that the rule's result keeps 9 digits and the flag alone reports it."""
    from ..core.precision import TOL
    from .errors import RealisationWarning
    if not worst > TOL:
        return
    warnings.warn(f"{where}: {count} cell(s) of a weight-adapted rule are not resolved (largest disagreement {worst:.2g} "
                  "of int |w|, an estimate of what is lost there, not a bound): the weight has a feature the rule cannot "
                  "follow, such as an undeclared singular point (declare it: singular_at=, exponent=) or a jump "
                  "(breaks=)", RealisationWarning, stacklevel=3)


def adapted_panels(edges, n, w, tol=ADAPT_TOL):
    """A composite Gauss rule on the panels between edges (n points each), bisected where the weight w (a vectorised
    function of the coordinate) is not resolved (plan 022 P2): a panel is accepted when its rule and the rules of its
    halves give int w and int w s (s the panel's own coordinate) within tol times the total int |w|; accepted panels
    keep the halves' nodes. Panels where w vanishes agree at once, so a compact support costs nothing beyond the
    starting panels. Returns (nodes, weights without w, resolved): resolved is False when ADAPT_DEPTH was reached."""
    A, B, resolved = adapted_edges(edges, n, w, tol)
    g, gw = gauss01(n)
    H, C = B - A, 0.5 * (A + B)
    x = np.concatenate([(A[:, None] + 0.5 * H[:, None] * g).ravel(), (C[:, None] + 0.5 * H[:, None] * g).ravel()])
    W = np.concatenate([(0.5 * H[:, None] * gw).ravel()] * 2)
    return x, W, resolved


def adapted_edges(edges, n, w, tol=ADAPT_TOL):
    """The accepted panels (A, B, resolved) of adapted_panels: the partition of [edges[0], edges[-1]] on which an
    n-point Gauss rule resolves w (plan 022 P2)."""
    g, gw = gauss01(n)
    e = np.asarray(edges, float)
    A, B = e[:-1][np.diff(e) > 0], e[1:][np.diff(e) > 0]
    W0 = (B - A)[:, None] * gw
    scale = max(float(np.sum(np.abs(w((A[:, None] + (B - A)[:, None] * g).ravel())) * W0.ravel())), 1e-300)
    accA, accB, resolved = [], [], True
    for depth in range(ADAPT_DEPTH + 1):
        if not len(A):
            break
        H, C = B - A, 0.5 * (A + B)
        Xc = A[:, None] + H[:, None] * g
        Xl, Xr = A[:, None] + 0.5 * H[:, None] * g, C[:, None] + 0.5 * H[:, None] * g
        fc, fl, fr = (w(X.ravel()).reshape(X.shape) for X in (Xc, Xl, Xr))
        Wc, Wh = H[:, None] * gw, 0.5 * H[:, None] * gw
        bad, gap = np.zeros(len(A), bool), np.zeros(len(A))
        for phi in (lambda X: 1.0, lambda X, C=C, H=H: (X - C[:, None]) / H[:, None]):
            coarse = np.sum(fc * Wc * phi(Xc), axis=1)
            fine = np.sum(fl * Wh * phi(Xl), axis=1) + np.sum(fr * Wh * phi(Xr), axis=1)
            gap = np.maximum(gap, np.abs(coarse - fine))
            bad |= np.abs(coarse - fine) > tol * scale
        if depth == ADAPT_DEPTH:
            resolved = not bad.any()
            if not resolved:
                _unresolved(f"adapted_edges ({ADAPT_DEPTH} bisections)", int(bad.sum()), float(gap[bad].max()) / scale)
            bad = np.zeros(len(A), bool)
        ok = ~bad
        accA.append(A[ok])
        accB.append(B[ok])
        A, B = np.concatenate([A[bad], C[bad]]), np.concatenate([C[bad], B[bad]])
    A, B = np.concatenate(accA), np.concatenate(accB)
    order = np.argsort(A)
    return A[order], B[order], resolved


def adapted_square(n, w, cells=1, tol=ADAPT_TOL):
    """The 2-D analogue on the reference square [0, 1]^2 (tensor Gauss n x n per cell, quartered where w is not
    resolved; moments 1, s, t), starting from cells x cells. w is a vectorised function of (a, b). Returns
    (a, b, weights without w, resolved)."""
    g, gw = gauss01(n)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    GW = np.outer(gw, gw).ravel()
    G1, G2 = G1.ravel(), G2.ravel()
    t = np.arange(cells) / cells
    A0, B0 = (x.ravel() for x in np.meshgrid(t, t, indexing="ij"))
    H = np.full(len(A0), 1.0 / cells)
    scale = max(float(np.sum(np.abs(w(A0[:, None] + H[:, None] * G1, B0[:, None] + H[:, None] * G2)) *
                             H[:, None] ** 2 * GW)), 1e-300)
    xa, xb, ws, resolved = [], [], [], True
    a0, b0, h = A0, B0, H
    for depth in range(ADAPT_DEPTH + 1):
        if not len(a0):
            break
        ca, cb = a0 + 0.5 * h, b0 + 0.5 * h
        Pa, Pb = a0[:, None] + h[:, None] * G1, b0[:, None] + h[:, None] * G2
        fc = w(Pa, Pb)
        kids = [(a0 + da * h, b0 + db * h) for da in (0.0, 0.5) for db in (0.0, 0.5)]
        Ka = [ka[:, None] + 0.5 * h[:, None] * G1 for ka, _ in kids]
        Kb = [kb[:, None] + 0.5 * h[:, None] * G2 for _, kb in kids]
        fk = [w(a, b) for a, b in zip(Ka, Kb)]
        bad, gap = np.zeros(len(a0), bool), np.zeros(len(a0))
        for phi in (lambda a, b: 1.0, lambda a, b, ca=ca, h=h: (a - ca[:, None]) / h[:, None],
                    lambda a, b, cb=cb, h=h: (b - cb[:, None]) / h[:, None]):
            coarse = np.sum(fc * phi(Pa, Pb) * GW, axis=1) * h ** 2
            fine = sum(np.sum(f * phi(a, b) * GW, axis=1) for f, a, b in zip(fk, Ka, Kb)) * (0.5 * h) ** 2
            gap = np.maximum(gap, np.abs(coarse - fine))
            bad |= np.abs(coarse - fine) > tol * scale
        if depth == ADAPT_DEPTH:
            resolved = not bad.any()
            if not resolved:
                _unresolved(f"adapted_square ({ADAPT_DEPTH} quarterings)", int(bad.sum()), float(gap[bad].max()) / scale)
            bad = np.zeros(len(a0), bool)
        ok = ~bad
        for a, b in zip(Ka, Kb):
            xa.append(a[ok].ravel())
            xb.append(b[ok].ravel())
            ws.append((np.broadcast_to((0.5 * h[ok]) ** 2, (len(GW), ok.sum())).T * GW).ravel())
        a0 = np.concatenate([ka[bad] for ka, _ in kids])
        b0 = np.concatenate([kb[bad] for _, kb in kids])
        h = np.tile(0.5 * h[bad], 4)
    return np.concatenate(xa), np.concatenate(xb), np.concatenate(ws), resolved


def adapted_nodes_on(S, n, w, points=(), breaks=(), tol=ADAPT_TOL, n_max=512, exponent=None):
    """nodes_on(S, n, points, breaks) refined where the weight w (a function of Cartesian points (m, d) -> (m,)) is
    not resolved (plan 022 P2): on an interval the panels of nodes_on bisected (adapted_panels); on a 2-D box the
    square quartered (adapted_square, starting from the cells nodes_on's order implies); on other sets quadrature.rule at
    doubled orders until the moments 1 and x_k of w agree (up to n_max). Declared singular points keep the graded rule
    of nodes_on (exponent: the declared one, for its innermost cell). Returns (X, W, resolved); W does not include w."""
    from ..domains import Box
    from ..sets import Interval
    if points or (isinstance(S, Interval) is False and not isinstance(S, Box) and getattr(S, "charts", lambda: None)() is None):
        X, W = nodes_on(S, n, points, breaks, exponent)
        return X, W, True
    if isinstance(S, Interval):
        edges = sorted({S.a, S.b} | {float(b) for b in breaks if S.a < b < S.b})
        x, W, ok = adapted_panels(edges, n, lambda t: w(t[:, None]), tol)
        return x[:, None], W, ok
    if isinstance(S, Box) and S.dim == 2:
        (a0, a1), (b0, b1) = [(s.a, s.b) for s in S.sides]
        m = 16
        u, v, W, ok = adapted_square(m, lambda a, b: w(np.stack([a0 + (a1 - a0) * a.ravel(), b0 + (b1 - b0) * b.ravel()],
                                                                 axis=1)).reshape(a.shape),
                                     cells=max(1, -(-n // m)), tol=tol)
        return np.stack([a0 + (a1 - a0) * u, b0 + (b1 - b0) * v], axis=1), W * (a1 - a0) * (b1 - b0), ok
    prev, k = None, n
    while True:
        X, W = nodes_on(S, k, points, breaks)
        wx = w(X)
        mom = np.concatenate([[W @ wx], (W * wx) @ X])
        scale = max(float(np.abs(W) @ np.abs(wx)), 1e-300)
        if prev is not None and np.max(np.abs(mom - prev)) <= tol * scale * max(1.0, float(np.max(np.abs(X)))):
            return X, W, True
        if 2 * k > n_max:
            _unresolved(f"adapted_nodes_on (order doubled to {k})", 1,
                        float(np.max(np.abs(mom - prev))) / scale if prev is not None else float("inf"))
            return X, W, False
        prev, k = mom, 2 * k
