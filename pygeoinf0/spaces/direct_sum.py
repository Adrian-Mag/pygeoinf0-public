"""Direct sums of Hilbert spaces.

    X = DirectSum({"vp": V, "moho": H, "station": pg.Euclidean(4)})     X = V (+) H (+) R^4, named summands
    X = DirectSum([V, H], weights=[w1, w2])                            <x, y>_X = sum_k w_k <x_k, y_k>_k (w_k > 0)
    x = X({"vp": f, "moho": h})                                        an element from its parts (missing parts are 0)
    x["vp"], X.part(x, "vp")                                           a part: the projection P_k x
    X.embed("vp", f)                                                   the injection (0, ..., f, ..., 0)
    X.riesz({"vp": mu, "moho": -c * pg.DiracMeasure(p)})               the representer of x -> sum_k int x_k dmu_k
    X.riesz([{...}, {...}, ...])                                       one functional per dict: a Family (the data)
    pygeoinf0.probability.cylindrical(X, alpha)                       N(0, I/alpha) on X (independent summands)

An element is a tuple (x_1, ..., x_K) and X is complete with the sum of the inner products, so the summands are
independent: knowing x_1 says nothing about x_2, and a field made of parts on adjacent layers may jump between them.
A bounded functional on X is a sum of bounded functionals on the summands, l(x) = sum_k l_k(x_k); its representer
in X is (r_1 / w_1, ..., r_K / w_K) with r_k the representer of l_k in X_k, so
||l||^2 = sum_k ||r_k||_k^2 / w_k, and the Gram matrix of many functionals is sum_k G_k / w_k with G_k the Gram matrix of
their parts in X_k (realisations/series/direct_sum.py: summands a functional does not touch cost nothing). The cylindrical
Gaussian N(0, I/alpha) of X is the product of N(0, I_k / (alpha w_k)) on the summands: the weights set the prior scale
of each part, alpha the overall one.

Elements are Function objects whose realisation is the tuple of the parts' realisations, so operators, Functionals,
G G*, solvers, settings and Gaussian measures treat X like any other Hilbert space; x(points) is refused (evaluate a
part: x[key](points)).

A direct sum of Euclideans is a Euclidean (plan 068 P5, the author's decision Q9):

    E = euclidean_sum({"P": D1, "S": D2}, weights=None)               R^(n_1 + n_2) with the block metric (w_k W_k)_k
    E.select(block="S"), E.groups("block")                            its coordinates carry block labels (block,
                                                                      coordinate): the summand's name and the
                                                                      coordinate's label in it (or its index)

Its inner product is the direct sum's, <x, y>_E = sum_k w_k <x_k, y_k>_k for x the concatenation of the parts; of plain
data spaces with unit weights it is a plain DataSpace. Its elements are vectors (the parts concatenated), not tuples:
DirectSum keeps the tuple elements for sums with function spaces and for callers that build elements part by part.
The column operator x -> (G_k x)_k (operators/blocks.ColumnOperator) maps into it.
"""
import math

import numpy as np

from ..core import fits
from ..core.claims import ASSERTED, THEOREM, Claim, Claims
from .elements import Family, Function, Vector, VectorFamily
from ..core.errors import SpaceMismatch
from .base import Euclidean, HilbertSpace
from .identity import require, require_realised


class DirectSum(HilbertSpace):
    """X_1 (+) ... (+) X_K with <x, y> = sum_k w_k <x_k, y_k>_k (see the module docstring)."""

    def __init__(self, summands, weights=None):
        if isinstance(summands, dict):
            names, spaces = list(summands), list(summands.values())
        else:
            spaces = list(summands)
            names = list(range(len(spaces)))
        if not spaces or not all(isinstance(H, HilbertSpace) for H in spaces):
            raise TypeError("a direct sum needs one or more Hilbert spaces")
        if len(set(names)) != len(names):
            raise ValueError("the summands need distinct names")
        if weights is None:
            w = np.ones(len(spaces))
        elif isinstance(weights, dict):
            unknown = set(weights) - set(names)
            if unknown:
                raise KeyError(f"weights for summands that do not exist: {sorted(map(str, unknown))}")
            w = np.array([float(weights.get(n, 1.0)) for n in names])
        else:
            w = np.asarray(weights, float).ravel()
        if w.shape != (len(spaces),) or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("one positive, finite weight per summand")
        self.summands, self.names, self.weights = tuple(spaces), tuple(names), tuple(float(x) for x in w)
        from ..realisations.registry import choose_sum
        self.backend = choose_sum(self)

    # ---------------------------------------------------------------- structure
    def __len__(self):
        return len(self.summands)

    def index(self, key):
        """The position of a summand, by name (or by position when the names are not integers)."""
        if key in self.names:
            return self.names.index(key)
        if isinstance(key, (int, np.integer)) and not any(isinstance(n, (int, np.integer)) for n in self.names):
            return int(key) % len(self.summands)
        raise KeyError(f"{key!r} is not a summand of {self} (summands: {', '.join(map(repr, self.names))})")

    def summand(self, key):
        return self.summands[self.index(key)]

    @property
    def domain(self):
        raise AttributeError("a direct sum has no domain: its elements are tuples; evaluate a part, x[key](points)")

    # ---------------------------------------------------------------- parts <-> realisations
    def _part_of(self, k, u, resolution=None):
        """(realisation, N) of an element or family u of summand k (numbers and callables become elements, a callable
        at the declared resolution if one is given: core/fits.py)."""
        H = self.summands[k]
        if isinstance(H, Euclidean):
            if isinstance(u, Vector):
                V = H._values(u)[:, None]
            elif isinstance(u, VectorFamily):
                V = H._values(u).T
            else:
                V = H(np.asarray(u, float)).values[:, None]
            return np.array(V, float), V.shape[1]
        if not isinstance(u, (Function, Family)):
            u = H.element(u, resolution=resolution) if callable(u) and resolution is not None else H.element(u)
        require(u.space, H, f"{getattr(u, 'name', 'the part')} is in {u.space}, not in the summand "
                            f"{self.names[k]!r} = {H}", getattr(u, "name", None))
        from .representers import representable, symbolic
        if symbolic(u) and not representable(u):          # kept symbolic: paired exactly, never fitted (plan 035 P1)
            from ..realisations.series.direct_sum import Symbolic
            ms = [u.measure] if isinstance(u, Function) else u.measures
            return Symbolic(ms), len(ms)
        return u.realisation, (len(u) if isinstance(u, Family) else 1)

    def _from_parts(self, items, name, family=None, resolution=None):
        """An element (family=False) or family of X from {position: element or family of that summand}; one
        ResolutionWarning for all the parts given as bare callables (core/fits.py)."""
        with fits.gathered(name):
            return self._from_parts_(items, name, family, resolution)

    def _from_parts_(self, items, name, family, resolution):
        parts, N, made = [None] * len(self.summands), None, {}
        for k, u in items.items():
            if u is None or (isinstance(u, (int, float)) and u == 0 and not isinstance(self.summands[k], Euclidean)):
                continue
            if callable(u) and not isinstance(u, (Function, Family)) and not isinstance(self.summands[k], Euclidean):
                u = self.summands[k].element(u, resolution=resolution) if resolution is not None else \
                    self.summands[k].element(u)
            if getattr(u, "_res", None) is not None and isinstance(u, Function):
                made[k] = u                                   # its part keeps u's resolution (core/fits.py)
            R, n = self._part_of(k, u, resolution)
            if N is not None and n != N:
                raise ValueError(f"parts of {N} and {n} members cannot form one family")
            N = n
            parts[k] = (R, np.arange(n))
        from ..realisations.series.direct_sum import DirectSumRealisation
        if family is None:
            family = N is not None and any(isinstance(u, (Family, VectorFamily)) for u in items.values())
        pl = DirectSumRealisation(parts, 1 if N is None else N)
        if family:
            return Family(self, pl, name=name)
        res = fits.Record("inherited", parent=tuple(made.values()), name=name, parts=made) if made else None
        return Function(self, pl=pl, name=name, res=res)

    def _items(self, parts):
        if isinstance(parts, dict):
            return {self.index(k): v for k, v in parts.items()}
        parts = list(parts)
        if len(parts) != len(self.summands):
            raise ValueError(f"{len(parts)} parts for {len(self.summands)} summands (or give a dict of the nonzero ones)")
        return dict(enumerate(parts))

    def element(self, parts, name="x", resolution=None):
        """The element with the given parts: a dict {name: part} (missing parts are zero) or a sequence of K parts.
        A part is an element of the summand, or anything the summand turns into one (a function, a number).
        resolution=w: the narrowest feature width of the parts given as callables (core/fits.py)."""
        if isinstance(parts, Function) and parts.space == self:
            require_realised(parts.space, self, parts.name)
            return parts
        return self._from_parts(self._items(parts), name, family=False, resolution=resolution)

    def __call__(self, parts, name="x", resolution=None):
        return self.element(parts, name, resolution)

    def family(self, items, name="F"):
        """Many elements: a list of elements of X, or a dict (or sequence) of families of the summands, one per part,
        all with the same number of members."""
        if isinstance(items, dict) or (isinstance(items, (list, tuple)) and len(items) == len(self.summands)
                                       and any(isinstance(u, (Family, VectorFamily)) for u in items)):
            return self._from_parts(self._items(items), name, family=True)
        items = [self.element(x) for x in items]
        return Family(self, self.backend.hstack([x.realisation for x in items]), name=name)

    @property
    def zero(self):
        return self.element({}, name="0")

    def part(self, x, key):
        """The part of x (an element or a family of X) in the summand key: P_k x."""
        k = self.index(key)
        H, R = self.summands[k], self._pl_of(x).parts[k]
        many = isinstance(x, Family)
        name = f"{x.name}[{self.names[k]!r}]"
        if R is None:
            if isinstance(H, Euclidean):
                return VectorFamily(H, np.zeros((len(x), H.dim)), name=name) if many else Vector(H, np.zeros(H.dim), name=name)
            return H.family([H.zero] * len(x), name=name) if many else H.zero
        Rk, idx = R
        N = len(x) if many else 1
        if len(idx) != N:                          # some members are zero in this summand
            S = np.zeros((len(idx), N))
            S[np.arange(len(idx)), idx] = 1.0
            Rk = self.backend.ops(k, Rk).combine(Rk, S)
        from ..realisations.registry import provides
        if provides(Rk, "symbolic_measures"):
            from .representers import DensityRepresenter, DensityRepresenters
            return DensityRepresenters(H, Rk.measures, name=name) if many else \
                DensityRepresenter(H, Rk.measures[0], name=name)
        if isinstance(H, Euclidean):
            return VectorFamily(H, Rk.T, name=name) if many else Vector(H, Rk[:, 0], name=name)
        rec = getattr(x, "_res", None)
        u = None if many or rec is None or rec.parts is None else rec.parts.get(k)
        return Family(H, Rk, name=name) if many else Function(H, pl=Rk, name=name, res=None if u is None else fits.inherited(u, name))

    def embed(self, key, u, name=None):
        """The injection of u (an element or family of the summand key) into X: (0, ..., u, ..., 0)."""
        k = self.index(key)
        return self._from_parts({k: u}, name or f"ι({getattr(u, 'name', 'u')})")

    def functionals_of(self, family, name="G"):
        """pg.Functionals(family): the operator into R^n of representers in this sum, row i x -> <x, family[i]> (what
        PiecewiseSpace.functional returns for a list of measures). Like projection, a method of the space returning an
        operator: the late import is this module's crossing spaces -> operators of test_structure."""
        from ..operators.functionals import Functionals
        require(family.space, self, f"{family.name} is in {family.space}, not in {self}", family.name)
        return Functionals(family, name=name)

    def projection(self, key):
        """P_k: X -> X_k as an operator; its adjoint is u -> embed(key, u) / w_k."""
        from ..operators.linear import LinearOperator
        k = self.index(key)
        w = self.weights[k]
        P = LinearOperator(self, self.summands[k], apply=lambda x: self.part(x, key),
                           adjoint=lambda u: self.embed(key, u) * (1.0 / w), name=f"P[{self.names[k]!r}]")
        P._claims = [Claim("linear", THEOREM, "a coordinate projection of a direct sum"),
                     Claim(f"bounded, norm 1/sqrt(w_k) = {1 / math.sqrt(w):.4g}", THEOREM,
                           "||x_k||^2 <= ||x||^2 / w_k, with equality on the summand")]
        return P

    # ---------------------------------------------------------------- geometry
    def _pl_of(self, u):
        if isinstance(u, (Function, Family)):
            require(u.space, self, f"{u.name} is in {u.space}, not in {self}", u.name)
            return u.realisation
        raise TypeError(f"expected an element or family of {self}, got {type(u).__name__}")

    def inner(self, f, g):
        G = self.backend.gram(self._pl_of(f), self._pl_of(g))
        if isinstance(f, Function) and isinstance(g, Function):
            return float(G[0, 0])
        if isinstance(f, Function):
            return G[0]
        if isinstance(g, Function):
            return G[:, 0]
        return G

    def norm(self, f):
        r = np.sqrt(np.maximum(self.backend.sqnorms(self._pl_of(f)), 0.0))
        return float(r[0]) if isinstance(f, Function) else r

    def _realisation_error(self, pl):
        return self.backend.error(pl)

    # ---------------------------------------------------------------- functionals and measures
    def riesz(self, spec):
        """Representers of functionals l(x) = sum_k l_k(x_k), each given by a dict {summand: what that summand's riesz
        takes} (a measure, a signed measure such as -R * mu or c1 * DiracMeasure(p1) + c2 * DiracMeasure(p2), a set,
        points; for a Euclidean summand the coefficients a of x_k -> a . x_k). A dict whose parts are single
        functionals gives an element, one whose parts are m functionals each a family of m; a list of dicts gives a
        family with one member per dict (the parts are realised summand by summand, each in one call). A part may also
        be a representer in its summand already (an element or family of X_k). Symbolic L2 representers whose sets the
        summand cannot realise (a measure restricted to a layer: S cap D) stay symbolic and pair exactly, as in the
        summand itself (realisations/series/direct_sum.Symbolic)."""
        if isinstance(spec, dict):
            parts = {self.index(k): self._riesz_part(self.index(k), v) for k, v in spec.items()}
            many = {isinstance(u, (Family, VectorFamily)) for u in parts.values()}
            if len(many) > 1:
                raise ValueError("the parts of one dict must all be single functionals or all families of one size")
            return self._from_parts(parts, "k", family=many.pop() if many else False)
        spec = list(spec)
        groups = {}
        for i, d in enumerate(spec):
            for key, v in d.items():
                groups.setdefault(self.index(key), ([], []))
                groups[self.index(key)][0].append(i)
                groups[self.index(key)][1].append(v)
        parts = [None] * len(self.summands)
        for k, (idx, items) in groups.items():
            F = self._riesz_part(k, items, many=True)
            parts[k] = (self._part_of(k, F)[0], np.asarray(idx))
        from ..realisations.series.direct_sum import DirectSumRealisation
        return Family(self, DirectSumRealisation(parts, len(spec)), name="k")

    def _riesz_part(self, k, v, many=False):
        """The representer in X of a functional on summand k alone (its representer in X_k, divided by w_k)."""
        H, w = self.summands[k], self.weights[k]
        if isinstance(H, Euclidean):
            A = np.atleast_2d(np.asarray(v, float)) if many else np.asarray(v, float)[None]
            V = H.solve_metric(A.T) / w                                  # (n, m): W^-1 a / w
            return VectorFamily(H, V.T, name="k") if many or np.ndim(v) == 2 else Vector(H, V[:, 0], name="k")
        if isinstance(v, (Function, Family)) and v.space == H:      # a representer in X_k already
            require_realised(v.space, H, v.name)
            r = v
        elif many and all(isinstance(x, Function) and x.space == H for x in v):
            r = H.family(list(v), name="k")                              # (realised as H, or refused there)
        elif many:
            r = H.riesz(list(v)) if all(_single(x) for x in v) else H.family([H.riesz(x) for x in v], name="k")
        else:
            r = H.riesz(v)
        return r if w == 1 else r * (1.0 / w)

    def standard_gaussian(self):
        """N(0, I) on a direct sum of finite-dimensional spaces (Euclideans): the product of the standard Gaussians
        N(0, I_k / w_k) of the summands (a GaussianMeasure: the identity is trace class there). Refused on a sum with an
        infinite-dimensional summand (plan 069 P6): name a prior there, e.g. pygeoinf0.probability.cylindrical(X, alpha),
        the product of the cylindrical Gaussians of the summands."""
        from .base import Euclidean
        from ..probability import GaussianMeasure

        def finite(X):
            return isinstance(X, Euclidean) or bool(getattr(X, "summands", None)) and all(finite(Y) for Y in X.summands)
        if not finite(self):
            return super().standard_gaussian()
        return GaussianMeasure.cylindrical(self, 1.0)

    # ---------------------------------------------------------------- identity
    @property
    def claims(self):
        w = "" if all(x == 1 for x in self.weights) else " w_k"
        return Claims([Claim(f"<x, y> = sum_k{w} <x_k, y_k>_k over {len(self)} summands", ASSERTED, "the definition"),
                       Claim("Hilbert space (complete)", THEOREM,
                             "a finite product of complete spaces is complete; positive weights give an equivalent norm"),
                       Claim("the summands are orthogonal and independent (a part may change without the others)",
                             THEOREM, "the inner product is the sum of the summands' ones")])

    def __eq__(self, other):
        return other is self or (isinstance(other, DirectSum) and self.names == other.names
                                 and self.weights == other.weights
                                 and all(a == b for a, b in zip(self.summands, other.summands)))

    def __hash__(self):
        return hash(("DirectSum", self.names, self.weights))

    def __repr__(self):
        parts = [f"{n}: {H!r}" if not isinstance(n, int) else repr(H) for n, H in zip(self.names, self.summands)]
        w = "" if all(x == 1 for x in self.weights) else ", weighted"
        return f"DirectSum({'; '.join(parts)}{w})" if len(parts) <= 4 else \
            f"DirectSum({'; '.join(parts[:3])}; … {len(parts)} summands{w})"


def euclidean_sum(summands, weights=None):
    """The direct sum of real Euclidean spaces as a Euclidean with block labels (the module docstring); summands a list
    (blocks named 0, 1, ...) or a dict by name, weights one positive number per summand (default 1)."""
    from .data import DataSpace
    names, spaces = (list(summands), list(summands.values())) if isinstance(summands, dict) else \
        (list(range(len(summands))), list(summands))
    if not spaces or not all(isinstance(D, Euclidean) for D in spaces):
        raise TypeError("euclidean_sum sums real Euclidean spaces (R^n); for function spaces or C^n use pg.DirectSum, "
                        "whose elements are tuples of parts")
    if len(set(names)) != len(names):
        raise ValueError("the summands need distinct names")
    w = np.ones(len(spaces)) if weights is None else np.asarray(weights, float).ravel()
    if w.shape != (len(spaces),) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("one positive, finite weight per summand")
    labels = {"block": [n for n, D in zip(names, spaces) for _ in range(D.dim)],
              "coordinate": [lab for D in spaces for lab in (D.labels or range(D.dim))]}
    n = sum(D.dim for D in spaces)
    metric = _sum_metric(spaces, w)
    if not metric:
        return DataSpace(n, labels=labels)
    return Euclidean(n, labels=labels, **metric)


def _sum_metric(spaces, w):
    """The metric of the sum as Euclidean's keyword (none for the dot product): the blocks w_k W_k, diagonal where every
    W_k is."""
    if all(D.is_standard for D in spaces) and np.all(w == 1.0):
        return {}
    if all(D.weights is not None or D.is_standard for D in spaces):
        return {"weights": np.concatenate([wk * (np.ones(D.dim) if D.is_standard else D.weights)
                                           for wk, D in zip(w, spaces)])}
    from scipy.linalg import block_diag
    return {"metric": block_diag(*[wk * D.metric_matrix() for wk, D in zip(w, spaces)])}


def _single(x):
    """Whether x describes one functional for a summand's riesz (a measure, a set, or one point)."""
    from ..measures import Measure
    from ..sets import Region
    if isinstance(x, (Measure, Region, str, dict)):                 # a dict: one functional on a nested direct sum
        return True
    a = np.asarray(x, float) if not isinstance(x, (list, tuple)) or all(np.isscalar(v) for v in x) else None
    return a is not None and a.ndim <= 1
