"""The data space (plan 068 P5; the author's decisions Q1 and Q9): a named finite-dimensional Hilbert space with
labelled coordinates.

    D = pg.DataSpace(n)                                   R^n with the dot product, the space the data live in
    D = pg.DataSpace(n, labels={"mode": [...], "s": [...]})
                                                          labelled coordinates: D.select(s=2), D.groups("s") (labels.py)
    W = D.whitened(C)                                     the whitened coordinates of a noise covariance C: the isometry
                                                          d -> L^-1 d (C = L L^T) of D's vectors with the C^-1 inner
                                                          product onto the plain R^n (operators/whitening.py)
    pg.spaces.euclidean_sum([D_1, ..., D_K])               a direct sum of Euclideans is a Euclidean with block labels
                                                          (block, coordinate) (spaces/direct_sum.py); of plain data spaces
                                                          with unit weights, a data space

D stays plain (Q1): the noise law is not D's metric but lives in the relation from the model to the data
(pg.Additive(G, noise)); methods that want the C^-1 inner product name it with D.whitened(C). A data space is a
Euclidean, so every route written for R^n takes it, and it equals Euclidean(n) when it has no labels (labels name
coordinates: two data spaces with different labels are different spaces). The same vectors with another inner product
(D.with_metric(W), D.with_covariance(C)) form a Euclidean, not a data space. pg.Euclidean(n, covariance=C) remains for
the routes that still state the noise as D's metric (plan 070 P3 makes D plain throughout).
"""
from ..core.errors import Refused
from .base import CoordinateSpace, Euclidean

_CHOICES = ["the noise law in the relation: pg.Additive(G, noise) with noise a measure or a set on D",
            "the whitened coordinates of a noise covariance C: D.whitened(C)",
            "R^n with that inner product, not a data space: pg.Euclidean(n, weights=..., metric=... or covariance=...)"]


class DataSpace(Euclidean):
    """R^n with the dot product and labelled coordinates: the data space (see the module docstring)."""

    def __new__(cls, n=None, labels=None, **metric):
        return CoordinateSpace.__new__(cls)

    def __init__(self, n, labels=None, **metric):
        given = sorted(k for k, v in metric.items() if v is not None)
        unknown = sorted(set(metric) - {"weights", "metric", "covariance"})
        if unknown:
            raise TypeError(f"DataSpace(n, labels=...) takes no {', '.join(unknown)}")
        if given:
            raise Refused(f"a data space is plain R^n (the author's decision Q1): its inner product is the dot product, "
                          f"so {', '.join(given)} is not given to it; the noise is not D's metric", needs=_CHOICES)
        super().__init__(n, labels=labels)

    def __repr__(self):
        return f"R^{self.dim} (data space{', labelled' if self._labels is not None else ''})"
