"""Labelled coordinates of R^n (plan 009 G9): data indexed by, e.g., (mode, degree, order).

    D = pg.Euclidean(n, labels=[...])              n distinct hashable labels (tuples, strings, numbers)
    D = pg.Euclidean(n, labels={"mode": [...], "s": [...], "t": [...]})
                                                   named fields: the labels are named tuples (D.labels[i].s)
    D.select(predicate, **where)                   the indices of the coordinates whose labels satisfy a predicate and
                                                   field conditions (s=2, t=[0, 1], s=lambda s: s > 4)
    D.groups("s", "t")                             {(s, t): indices}: blocks of rows (plan 009 G2)
    R.select(rows)                                 a relation into D of some coordinates only; the selected
                                                   space keeps their labels (relations/selection.py)

Labels name coordinates: two spaces with different labels are different spaces (an unlabelled space compares by
dimension and metric only). Spaces derived from a labelled one (its standard space, with_metric, with_covariance)
keep its labels.
"""
import collections

import numpy as np


class Labels:
    """The labels of the coordinates of R^n: items (one per coordinate) and, for named fields, their names."""

    def __init__(self, labels, n):
        if isinstance(labels, Labels):
            self.items, self.names = labels.items, labels.names
            return
        if isinstance(labels, dict):
            names = tuple(labels)
            cols = [list(np.asarray(v).tolist()) if isinstance(v, np.ndarray) else list(v) for v in labels.values()]
            if any(len(c) != n for c in cols):
                raise ValueError(f"every label field needs {n} values")
            Label = collections.namedtuple("Label", names)
            self.items, self.names = tuple(Label(*row) for row in zip(*cols)), names
        else:
            items = tuple(labels)
            if len(items) != n:
                raise ValueError(f"{len(items)} labels for {n} coordinates")
            self.items, self.names = items, getattr(items[0], "_fields", None) if items else None
        try:
            unique = len(set(self.items)) == n
        except TypeError:
            raise TypeError("labels must be hashable (tuples, strings, numbers)") from None
        if not unique:
            raise ValueError("labels must be distinct")

    def take(self, rows):
        out = Labels.__new__(Labels)
        out.items, out.names = tuple(self.items[int(i)] for i in rows), self.names
        return out

    def select(self, predicate, where):
        def ok(lab):
            if predicate is not None and not predicate(lab):
                return False
            for f, want in where.items():
                if self.names is None or f not in self.names:
                    raise KeyError(f"no label field {f!r} (fields: {self.names})")
                v = getattr(lab, f)
                if callable(want):
                    if not want(v):
                        return False
                elif isinstance(want, (list, tuple, set, frozenset, range, np.ndarray)):
                    if v not in set(np.asarray(want).tolist() if isinstance(want, np.ndarray) else want):
                        return False
                elif v != want:
                    return False
            return True
        return np.array([i for i, lab in enumerate(self.items) if ok(lab)], dtype=int)

    def groups(self, fields):
        if self.names is None:
            raise KeyError("groups by field need labels with named fields")
        idx = [self.names.index(f) for f in fields]
        out = {}
        for i, lab in enumerate(self.items):
            out.setdefault(tuple(lab[k] for k in idx), []).append(i)
        return {k: np.array(v, dtype=int) for k, v in out.items()}


class LabelledCoordinates:
    """What Euclidean spaces get from labels (module docstring): labels, fields, select, groups."""

    _labels = None

    @property
    def labels(self):
        """The labels of the coordinates (a tuple), or None."""
        return None if self._labels is None else self._labels.items

    @property
    def fields(self):
        """The names of the label fields (labels given as named fields), or None."""
        return None if self._labels is None else self._labels.names

    def select(self, predicate=None, **where):
        """The indices (ascending) of the coordinates whose labels satisfy predicate(label) and every condition
        field=value (a value, a collection of values, or a test of the value)."""
        if self._labels is None:
            raise AttributeError(f"{self} has no labels: give Euclidean(n, labels=...), or select rows by index")
        return self._labels.select(predicate, where)

    def groups(self, *fields):
        """{value of the fields: indices} in the order of first appearance (e.g. the blocks of rows of a data space
        by degree and order: D.groups("s", "t"))."""
        if self._labels is None:
            raise AttributeError(f"{self} has no labels")
        return self._labels.groups(fields)
