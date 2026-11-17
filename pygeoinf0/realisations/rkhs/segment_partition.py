"""Weighted segments split where their weight needs it (plan 022 P2): a density on a segment or subinterval given by a
callable or an element, whose weight 16 starting panels do not resolve, becomes one SegmentPiece per panel of a
partition on which a 12-point rule (the energy routes' order) resolves it; every such measure on the same segment
shares the union partition, so that their pieces are identical (the same-piece route, two weights) or disjoint
and touching. Panels where the weight vanishes are dropped. Used by atoms.atoms_from / _pieces_of.
"""
import itertools

import numpy as np


def adapts(mu):
    """Whether the pieces of mu follow its weight (plan 022 P2): a density given by a callable or an element."""
    from ...measures import DensityMeasure, PieceDensity
    return isinstance(mu, DensityMeasure) and not isinstance(mu, PieceDensity) and \
        mu.weight.kind in ("callable", "function") and not mu.weight.singular


START = 16                   # starting panels of a weighted segment's adapted partition


def _segment_edges(p, q, values):
    """The panels (in arc length from p) on which a 12-point Gauss rule (the energy routes' order) resolves the weight
    (core/adaptive.adapted_edges), starting from START panels; and whether any panel had to be bisected."""
    from ...core.adaptive import adapted_edges
    L = float(np.linalg.norm(q - p))
    u = (q - p) / L
    A, B, _ = adapted_edges(np.linspace(0.0, L, START + 1), 12, lambda t: values(p + np.asarray(t).reshape(-1, 1) * u))
    return A, B, len(A) > START


def common_segment_edges(items):
    """{(p, q): edges} from items (ends, values): the segment or subinterval (p, q) of each measure whose pieces follow
    its weight (adapts) and the weight's values at Cartesian points. For the segments where some such weight is not
    resolved by the starting panels: the union of the panel edges of every such measure there, so that their pieces are
    identical (the same-piece route, with two weights) or disjoint and touching, never partly overlapping. Segments
    where every weight is resolved as they are keep one piece per measure, as before."""
    edges, refined = {}, set()
    for ends, values in items:
        A, B, split = _segment_edges(*ends, values)
        key = (tuple(ends[0]), tuple(ends[1]))
        edges.setdefault(key, set()).update(A.tolist() + B.tolist())
        if split:
            refined.add(key)
    return {k: np.array(sorted(v)) for k, v in edges.items() if k in refined}


def segment_pieces(p, q, w, wt, common=None):
    """w ds on the segment p q: one SegmentPiece, or, for a weight given by a callable or an element (kinds callable
    and function), one piece per panel of the common partition of the segment (common_segment_edges; this weight's
    own when it is alone), panels where w vanishes at every node dropped: the weight decides the resolution of every
    energy route, and a compact support costs nothing outside it (plan 022 P2)."""
    from ...quadrature import gauss01
    from .weighted import SegmentPiece
    p, q = np.atleast_1d(np.asarray(p, float)), np.atleast_1d(np.asarray(q, float))
    if wt is None or wt.kind not in ("callable", "function"):
        return [SegmentPiece(p, q, w)]
    L = float(np.linalg.norm(q - p))
    u = (q - p) / L
    if common is None:
        A, B, split = _segment_edges(p, q, lambda X: w(X))
        common = {(tuple(p), tuple(q)): np.unique(np.concatenate([A, B]))} if split else {}
    edges = common.get((tuple(p), tuple(q)))
    if edges is None:                                  # resolved by the starting panels: one piece, as before
        return [SegmentPiece(p, q, w)]
    g, _ = gauss01(12)
    out = []
    for a, b in itertools.pairwise(edges):
        if b > a and np.any(w(p + (a + (b - a) * g).reshape(-1, 1) * u) != 0):
            out.append(SegmentPiece(p + a * u, p + b * u, w))
    return out or [SegmentPiece(p, q, w)]
