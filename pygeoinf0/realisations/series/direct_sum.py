"""The realisation of elements of a direct sum X = X_1 (+) ... (+) X_K (spaces/direct_sum.py): the tuple of the summands'
own realisations, nothing chosen here.

A realisation DirectSumRealisation holds N members. For every summand k it stores (R_k, idx_k): the summand's realisation of the members
idx_k (sorted, distinct) and nothing for the others, which are zero in that summand. So a family of travel-time
representers keeps, per layer, only the rays that enter the layer, realised as that layer's own family (for a Sobolev
layer: one identity family of representers, which the exact and compressed energy engines handle as they do any
other), and

    gram(A, B) = sum_k w_k G_k[idx_k^A, idx_k^B],   G_k the summand's Gram of its parts

which is every summand's existing Gram, scattered. Euclidean summands are realised by their coordinate arrays (n, N).
Every operation is exact on the realisations; errors combine as sqrt(sum_k w_k e_k^2), the norm of X.

A part may also be Symbolic: the representers in an L2 summand of measures whose sets the summand's backend cannot
realise without fitting across their boundary (spaces/representers.py: a bump cut by a layer, S cap D). They stay what
they are, measures, and pair exactly as in the summand (its inner product: Gauss quadrature on S, exact intersections),
combine into signed measures, and are realised only where an operation needs the backend (adding a formula), which
then refuses as the summand's representer does (plan 035 P1).
"""
import numpy as np

from ...core.errors import NotDefined
from ..backends import Backend


class DirectSumRealisation:
    """N members: parts[k] = (R_k, idx_k) or None (zero in summand k)."""

    __slots__ = ("parts", "N")

    def __init__(self, parts, N):
        self.parts, self.N = tuple(parts), int(N)

    @property
    def size(self):
        return sum(getattr(p[0], "size", np.size(p[0])) for p in self.parts if p is not None)


class Symbolic:
    """The members of a part kept symbolic: one measure each (their representers d mu / d lambda in the summand)."""

    __slots__ = ("measures",)
    CAPABILITIES = ("symbolic_measures",)

    def __init__(self, measures):
        self.measures = list(measures)

    @property
    def N(self):
        return len(self.measures)

    @property
    def size(self):
        return 0


class SymbolicPart:
    """The operations of a backend on Symbolic parts of an L2 summand H (its inner product pairs them exactly)."""

    errors_known = True
    realisation_name = "symbolic"

    def __init__(self, H):
        self.H = H

    def wrap(self, R):
        from ...spaces.elements import Family
        from ...spaces.representers import DensityRepresenters
        return DensityRepresenters(self.H, R.measures) if isinstance(R, Symbolic) else Family(self.H, R)

    def lincomb(self, terms):
        from ...measures import SignedMeasure
        if not all(isinstance(R, Symbolic) for _, R in terms):
            B = self.H.backend
            return B.lincomb([(a, realised(self.H, R)) for a, R in terms])
        return Symbolic([SignedMeasure([(a, R.measures[j]) for a, R in terms]) for j in range(terms[0][1].N)])

    def combine(self, R, A):
        return Symbolic(self.wrap(R).combine(A).measures)

    def take(self, R, idx):
        return Symbolic([R.measures[int(i)] for i in np.atleast_1d(idx)])

    def hstack(self, Rs):
        if all(isinstance(R, Symbolic) for R in Rs):
            return Symbolic([m for R in Rs for m in R.measures])
        return self.H.backend.hstack([realised(self.H, R) for R in Rs])

    def gram(self, A, B):
        a = self.wrap(A)
        return np.atleast_2d(np.asarray(self.H.inner(a, a if B is A else self.wrap(B)), float))

    def sqnorms(self, A):
        return np.asarray(self.H.norm(self.wrap(A)), float) ** 2

    def error(self, R):
        return np.zeros(R.N)

    def summary(self, R):
        return f"symbolic: {R.N} measure(s), d mu / d lambda"


def realised(H, R):
    """R as a realisation in the backend of the summand H: a Symbolic part is fitted there, or refused (Refused) as its
    representers are when their sets are not realisable (spaces/representers.py)."""
    if not isinstance(R, Symbolic):
        return R
    from ...spaces.representers import DensityRepresenters
    return DensityRepresenters(H, R.measures).realisation


class EuclideanPart:
    """The operations of a backend on coordinate arrays (n, N) of a Euclidean summand E (its metric in the Gram)."""

    errors_known = True
    realisation_name = "coordinates"

    def __init__(self, E):
        self.E = E

    def lincomb(self, terms):
        return sum(a * V for a, V in terms)

    def combine(self, V, A):
        return V @ A

    def take(self, V, idx):
        return V[:, np.atleast_1d(idx)]

    def hstack(self, Vs):
        return np.hstack(Vs)

    def gram(self, A, B):
        return A.T @ self.E.apply_metric(B)

    def sqnorms(self, A):
        return np.einsum("ij,ij->j", A, self.E.apply_metric(A))

    def sqnorms_or(self, A, unavailable):
        return self.sqnorms(A)

    def error(self, V):
        return np.zeros(V.shape[1])

    def summary(self, V):
        return f"{V.shape[0]} coordinates"


class SumBackend(Backend):
    """Backend of a direct sum: per summand the summand's backend (or EuclideanPart), weights w_k of the inner product."""

    realisation_name = "direct sum"

    def __init__(self, space):
        from ...spaces.base import Euclidean
        self.space = space
        self.parts = [EuclideanPart(H) if isinstance(H, Euclidean) else H.backend for H in space.summands]
        self.weights = np.asarray(space.weights, float)
        self.errors_known = all(getattr(B, "errors_known", True) for B in self.parts)
        self.dim = None

    def ops(self, k, *Rs):
        """The operations for part k: the summand's backend, or SymbolicPart when one of the parts Rs is Symbolic."""
        if any(isinstance(R, Symbolic) for R in Rs):
            return SymbolicPart(self.space.summands[k])
        return self.parts[k]

    # ---------------------------------------------------------------- members
    @staticmethod
    def _pos(idx, N):
        inv = np.full(N, -1)
        inv[idx] = np.arange(len(idx))
        return inv

    def _scatter(self, k, part, N, onto):
        """The part (R, idx) re-expressed on the members `onto` (a superset of idx): zero columns for the others."""
        R, idx = part
        if len(idx) == len(onto) and np.array_equal(idx, onto):
            return R
        S = np.zeros((len(idx), len(onto)))
        S[np.arange(len(idx)), self._pos(onto, N)[idx]] = 1.0
        return self.ops(k, R).combine(R, S)

    def lincomb(self, terms):
        N = max(R.N for _, R in terms)
        terms = [(a, R if R.N == N else self.combine(R, np.ones((1, N)))) for a, R in terms]    # broadcast one member
        out = []
        for k in range(len(self.parts)):
            live = [(a, R.parts[k]) for a, R in terms if R.parts[k] is not None and a != 0]
            if not live:
                out.append(None)
                continue
            onto = np.unique(np.concatenate([p[1] for _, p in live]))
            Rs = [(a, self._scatter(k, p, N, onto)) for a, p in live]
            out.append((self.ops(k, *[R for _, R in Rs]).lincomb(Rs), onto))
        return DirectSumRealisation(out, N)

    def combine(self, R, A):
        A = np.asarray(A, float)
        out = []
        for k, p in enumerate(R.parts):
            Ak = None if p is None else A[p[1]]
            if p is None or not np.any(Ak):
                out.append(None)
                continue
            out.append((self.ops(k, p[0]).combine(p[0], Ak), np.arange(A.shape[1])))
        return DirectSumRealisation(out, A.shape[1])

    def take(self, R, idx):
        sel = np.atleast_1d(np.asarray(idx, int))
        out = []
        for k, p in enumerate(R.parts):
            if p is None:
                out.append(None)
                continue
            pos = self._pos(p[1], R.N)[sel]
            keep = np.flatnonzero(pos >= 0)
            out.append(None if not len(keep) else (self.ops(k, p[0]).take(p[0], pos[keep]), keep))
        return DirectSumRealisation(out, len(sel))

    def hstack(self, Rs):
        offs = np.cumsum([0] + [R.N for R in Rs])
        out = []
        for k in range(len(self.parts)):
            live = [(R.parts[k], o) for R, o in zip(Rs, offs) if R.parts[k] is not None]
            if not live:
                out.append(None)
                continue
            R = live[0][0][0] if len(live) == 1 else self.ops(k, *[p[0] for p, _ in live]).hstack([p[0] for p, _ in live])
            out.append((R, np.concatenate([p[1] + o for p, o in live])))
        return DirectSumRealisation(out, offs[-1])

    # ---------------------------------------------------------------- geometry
    def gram(self, A, B):
        out = np.zeros((A.N, B.N))
        for k, (pa, pb) in enumerate(zip(A.parts, B.parts)):
            if pa is None or pb is None:
                continue
            G = self.weights[k] * np.asarray(self.ops(k, pa[0], pb[0]).gram(pa[0], pb[0]))
            if len(pa[1]) == A.N and len(pb[1]) == B.N:          # every member present (idx sorted: 0, ..., N-1)
                out += G
            else:
                out[np.ix_(pa[1], pb[1])] += G
        return out

    def sqnorms(self, A):
        out = np.zeros(A.N)
        for k, p in enumerate(A.parts):
            if p is not None:
                out[p[1]] += self.weights[k] * self.ops(k, p[0]).sqnorms(p[0])
        return out

    def error(self, R):
        e2 = np.zeros(R.N)
        for k, p in enumerate(R.parts):
            if p is not None:
                e2[p[1]] += self.weights[k] * np.asarray(self.ops(k, p[0]).error(p[0]), float) ** 2
        return np.sqrt(e2)

    # ---------------------------------------------------------------- not defined on a tuple
    def fit(self, fun, N=1, breaks=None):
        raise NotDefined("an element of a direct sum is given by its parts, X({key: element, ...}), not by a formula")

    def constant(self, values):
        raise NotDefined("a number is not an element of a direct sum; give the parts, X({key: c, ...})")

    def evaluate(self, R, *X):
        raise NotDefined("an element of a direct sum is a tuple: evaluate its parts, x[key](points)")

    def colsum(self, R):
        return self.combine(R, np.ones((R.N, 1)))

    # ---------------------------------------------------------------- description
    def describe(self, R):
        return dict(kind="direct sum", size=R.size, parts=self.summary(R))

    def summary(self, R):
        names = self.space.names
        live = [f"{names[k]}: {self.ops(k, p[0]).summary(p[0])}" for k, p in enumerate(R.parts) if p is not None]
        return "; ".join(live) if live else "zero"
