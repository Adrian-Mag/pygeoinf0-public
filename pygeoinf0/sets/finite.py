"""Finite sets of points of R^d (plan 067 P2: from core/sets.py), Regions of dimension 0: their Hausdorff measure H^0
counts the points.

    FiniteSet([0.1, 0.5])               numbers (points of R^1)
    FiniteSet([(0, 0), (1, 0.5)])       points of R^d, as rows
"""
import math

import numpy as np

from .base import EmptySet
from .regions import Region


def _fmt(x):
    if np.ndim(x):
        return "(" + ", ".join(_fmt(v) for v in np.ravel(x)) + ")"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.12g}"


class FiniteSet(Region):
    """A finite set of real numbers, FiniteSet([0.1, 0.5]) or FiniteSet(range(5)), or of points of R^d,
    FiniteSet([(0, 0), (1, 0.5)]) (points as rows)."""

    __slots__ = ("points",)
    dim = 0

    def __init__(self, points):
        arr = np.asarray(list(points), dtype=float)
        pts = np.unique(arr, axis=0) if arr.ndim == 2 and arr.shape[1] > 1 else np.unique(arr.ravel())
        pts.flags.writeable = False
        object.__setattr__(self, "points", pts)

    def __setattr__(self, *_):
        raise AttributeError("finite sets are immutable")

    def __len__(self):
        return len(self.points)

    @property
    def cartesian_dim(self):
        return 1 if self.points.ndim == 1 else self.points.shape[1]

    @property
    def volume(self):
        """H^0: the number of points."""
        return len(self.points)

    def _inside_points(self, X):
        if self.points.ndim == 1:
            return np.isin(X[:, 0], self.points)
        return (X[:, None, :] == self.points[None]).all(axis=-1).any(axis=1)

    def __and__(self, other):
        if isinstance(other, Region) and other.decides_membership and int(other.cartesian_dim) == self.cartesian_dim:
            P = self.points if self.points.ndim == 2 else self.points[:, None]
            keep = np.atleast_1d(other.contains(P))
            if not keep.any():
                return EmptySet(self.ambient, why=f"no point of {self!r} lies in {other!r}")
            return self if keep.all() else FiniteSet(self.points[keep])
        return Region.__and__(self, other)

    __rand__ = __and__

    def __eq__(self, other):
        return isinstance(other, FiniteSet) and np.array_equal(self.points, other.points)

    def __hash__(self):
        return hash(self.points.tobytes())

    def __repr__(self):
        if len(self.points) <= 6:
            return "{" + ", ".join(_fmt(p) for p in self.points) + "}"
        return "{" + ", ".join(_fmt(p) for p in self.points[:3]) + f", ... ({len(self.points)} points)" + "}"
