"""Coordinates of a space for external code (plan 019 F1; experiments/EXP-034-external-implementations/DESIGN.md 2.1).

    Coordinates(M, export, imp=None)       E: M -> R^n, bounded and linear (what external code is given), and an import
                                           P: R^n -> M; by default the canonical import P = E* (E E*)^-1, the element of
                                           least norm in M with the given numbers
    Coordinates.nodal(M, points)           u -> (u(x_1), ..., u(x_n)): bounded only where point evaluation is (H^s with
                                           s > d/2); refused on L2 with the reason
    Coordinates.averages(M, cells)         u -> the mean of u over each cell: bounded on L2; for disjoint cells of L2 the
                                           canonical import is the piecewise-constant function
    Coordinates.functionals(M, measures)   u -> (int u dmu_1, ..., int u dmu_n): bounded functionals given by measures
    Coordinates.identity(D)                the numbers of a Euclidean space D (its metric W: E* c = W^-1 c)
    Coordinates.cells(M)                   the cell values of M = pg.L2(D, realisation=pg.realisations.Cells(...)):
                                           the cell averages, with diagonal maps (E* c = c / vol, P c = c): no
                                           Riesz representer per cell, so it scales with the cells
    Coordinates.direct_sum(X, {key: C_k})  the numbers of a direct sum: the parts' numbers in the order given; summands
                                           without coordinates are not exported (and imported as 0)

    c.numbers(m)    the external code's array E m          c.element(x)    the element P x
    c.error(m)      ||m - P E m||_M, what the round trip loses
    c.functional(g) h -> g . E h, a pg.Functional (spaces/dual.py): what a tool's "gradient" g in these coordinates
                    really is (jax.grad of a function of the numbers gives g); its gradient() is the Riesz representer
                    E* g in the space's inner product (plan 068 P2: was Covector.from_numbers(C, g))
    c.subset(predicate, name)   the elements whose numbers satisfy an external predicate (a dom given by external
                    code, e.g. "a first arrival exists"; plan 020 P5b): a pygeoinf0.sets.PredicateSubset, not convex

E (c.export) and P (c.imp) are LinearOperators with their Hilbert adjoints; the numbers live in the standard R^n (dot
product). The adjoint of the export is where M's inner product enters a derivative or a gradient (DESIGN 2.3): the
export must therefore be bounded on M, and the constructors here refuse coordinates that are not. A nonlinear change of
variables (slowness to velocity, a logarithm) is not a coordinate system: compose a pg.Map with one.
"""
import itertools

import numpy as np

from ..core.claims import DERIVED, THEOREM, Claim, Claims
from ..core.errors import Refused, SpaceMismatch
from ..spaces.base import CoordinateSpace, Euclidean, HilbertSpace
from ..operators.functionals import Functionals
from ..operators.linear import LinearOperator
from ..spaces.identity import require


class Coordinates:
    """A coordinate system of a space for external code (module docstring)."""

    def __init__(self, space, export, imp=None, name="E", claims=()):
        if not isinstance(export, LinearOperator):
            raise SpaceMismatch(f"the export must be a LinearOperator on {space}")
        require(export.domain, space, f"the export must be a LinearOperator on {space}", export.name)
        R = export.codomain
        if not (isinstance(R, CoordinateSpace) and R.is_standard):
            raise SpaceMismatch("the export maps into the standard R^n (external code is given plain numbers), "
                                f"not into {R}")
        self.space, self.export, self.dim, self.name = space, export, R.dim, name
        self._claims = list(claims)
        if imp is None:
            self.imp = self._canonical_import()
        else:
            if not isinstance(imp, LinearOperator) or imp.domain != R:
                raise SpaceMismatch(f"the import must be a LinearOperator from R^{self.dim} to {space}")
            require(imp.codomain, space, f"the import must be a LinearOperator from R^{self.dim} to {space}", imp.name)
            self.imp = imp

    def _canonical_import(self):
        """P = E* (E E*)^-1: the least-norm element with the given numbers; P E is the orthogonal projection of M onto
        the span of the coordinates' Riesz representers, and E P = I."""
        if not isinstance(self.space, HilbertSpace):
            raise Refused(f"the canonical import needs an inner product, and {self.space} has none",
                          needs=("an explicit import (imp=)",))
        E, R = self.export, self.export.codomain
        gram = np.asarray((E @ E.adjoint).matrix(), float)
        try:
            chol = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            raise Refused(f"the {self.dim} coordinate functionals are linearly dependent in {self.space}: no import "
                          "recovers numbers they cannot tell apart", needs=("independent coordinates",)) from None

        def solve(x):
            return np.linalg.solve(chol.T, np.linalg.solve(chol, x))

        P = LinearOperator(R, self.space, lambda x: E.adjoint(R(solve(x.values))),
                           adjoint=lambda m: R(solve(E(m).values)), name=f"{self.name}⁺")
        P._claims = [Claim("linear", THEOREM, "E* (E E*)^-1 is a composition of linear maps", key="linear")]
        self._claims += [Claim("E P = I", DERIVED, "the canonical import E* (E E*)^-1 (to the Gram solve)", key="E P = I"),
                         Claim("P E is the orthogonal projection onto the span of the coordinates' representers",
                               DERIVED, "P E = E* (E E*)^-1 E", key="projection")]
        return P

    # ---------------------------------------------------------------------------------------------- constructors
    @classmethod
    def functionals(cls, space, measures, name="E"):
        """u -> (int u dmu_i): the measures' Riesz representers in the space, so bounded by Cauchy-Schwarz; refused with
        the space's reason (pg.errors.UnboundedFunctional) where a measure is not bounded there."""
        if not hasattr(space, "riesz"):
            raise Refused(f"coordinates need Riesz representers, and {space} has none", needs=("a Hilbert space",))
        E = Functionals(space.riesz(list(measures)), name=name)
        bounded = Claim("bounded", DERIVED, "functionals given by Riesz representers in the space (Cauchy-Schwarz)",
                        key="bounded")
        return cls(space, E, name=name, claims=[bounded])

    @classmethod
    def nodal(cls, space, points, name="E"):
        """u -> (u(x_1), ..., u(x_n)). Point evaluation must be bounded on the space (H^s with s > d/2): on L2 it is
        not, and the space's refusal (pg.PointEvaluationUndefined) says so; use Coordinates.averages there."""
        pts = np.asarray(points, float)
        pts = pts[:, None] if pts.ndim == 1 else pts
        return cls.functionals(space, [p for p in pts], name=name)

    @classmethod
    def averages(cls, space, cells, name="E"):
        """u -> the mean of u over each cell (each cell's normalised Lebesgue measure)."""
        from ..measures import LebesgueMeasure
        return cls.functionals(space, [LebesgueMeasure(c).normalised() for c in cells], name=name)

    @classmethod
    def identity(cls, space, name="E"):
        """The numbers of a Euclidean space, real or complex (plan 020 P3a): E v = v, P x = x; with the metric W of
        the space, E* c = W^-1 c (the same formula in C^n with its real inner product Re <., .>)."""
        if not isinstance(space, CoordinateSpace):
            raise SpaceMismatch(f"identity coordinates are those of a Euclidean space, not of {space}")
        R = Euclidean(space.dim, field=space.field)
        E = LinearOperator(space, R, lambda v: R(np.asarray(v.values)),
                           adjoint=lambda c: space(space.solve_metric(np.asarray(c.values))), name=name)
        P = LinearOperator(R, space, lambda c: space(np.asarray(c.values)),
                           adjoint=lambda v: R(space.apply_metric(np.asarray(v.values))), name=f"{name}⁻¹")
        for op, why in ((E, "the identity of the coordinates"), (P, "the identity of the coordinates")):
            op._claims = [Claim("linear", THEOREM, why, key="linear")]
        exact = [Claim("bounded", THEOREM, "a linear map on a finite-dimensional space", key="bounded"),
                 Claim("E P = I", THEOREM, "both are the identity of the coordinates", key="E P = I"),
                 Claim("P E = I", THEOREM, "both are the identity of the coordinates", key="projection")]
        return cls(space, E, imp=P, name=name, claims=exact)

    @classmethod
    def cells(cls, space, name="cells"):
        """The cell averages of a cell-realised L2 (module docstring): E f = (f_i), the averages; E* c = c / vol and
        P c = c on the cells, both exact; P E is the orthogonal projection onto the cell-constant functions, which is
        what the realisation stores, so the numbers lose nothing the realisation holds. On pg.Complexified(M) of a
        cell-realised M the numbers are complex (C^n); the same formulas hold in the real structure Re <., .>."""
        from ..spaces.elements import Family, Function, VectorFamily
        from ..realisations.piecewise_constant import CellValues
        from ..realisations.registry import provides
        B = getattr(space, "backend", None)
        if not provides(B, "cell_values"):
            raise SpaceMismatch(f"cell coordinates need a cell-realised L2 (realisation=pg.realisations.Cells(...)), "
                                f"not {space}")
        vol, R = B.cells.volumes, Euclidean(B.cells.n)
        if getattr(space, "field", "R") == "C":           # complex cell values: numbers in C^n (plan 020 P3b)
            R = Euclidean(B.cells.n, field="C")

        def numbers(f, w=1.0):                    # a function -> its cell averages; a family -> a family of them
            if isinstance(f, Family):
                return VectorFamily(R, (w * f.realisation.V.T))
            return R(w * f.realisation.V[:, 0])

        def function(c, w=1.0):                   # numbers (or a family of them) -> cell-constant functions
            if isinstance(c, VectorFamily):
                return Family(space, CellValues((c.values * w).T))
            return Function(space, pl=CellValues(c.values * w))

        E = LinearOperator(space, R, numbers, adjoint=lambda c: function(c, 1.0 / vol), name=name)
        P = LinearOperator(R, space, function, adjoint=lambda f: numbers(f, vol), name=f"{name}⁺")
        for op in (E, P):
            op._claims = [Claim("linear", THEOREM, "cell averages and cell-constant functions", key="linear")]
            if getattr(space, "field", "R") == "C":
                op._claims.append(Claim("complex-linear (commutes with J)", THEOREM, "averages of complex values",
                                        key="complex-linear"))
        claims = [Claim("bounded", DERIVED, "|f_i| <= ||f|| / sqrt(vol_i) (Cauchy-Schwarz on each cell)", key="bounded"),
                  Claim("E P = I", DERIVED, "the average of a cell-constant function is its value", key="E P = I"),
                  Claim("P E is the orthogonal projection onto the cell-constant functions", DERIVED,
                        "the realisation stores exactly that projection", key="projection")]
        return cls(space, E, imp=P, name=name, claims=claims)

    @classmethod
    def direct_sum(cls, space, parts, name="E"):
        """The numbers of a direct sum X = X_1 (+) ... (+) X_K (pg.DirectSum, weights w_k): E x = (E_k x_k) over the
        given summands, in order. Adjoints in X's inner product: E* c = (E_k* c_k / w_k), and the import is the parts'
        imports (with canonical parts it is X's canonical import, since E E* = blockdiag(E_k E_k* / w_k))."""
        from ..spaces.direct_sum import DirectSum
        if not isinstance(space, DirectSum):
            raise SpaceMismatch(f"direct_sum coordinates are those of a pg.DirectSum, not of {space}")
        keys = list(parts)
        for k in keys:
            require(parts[k].space, space.summand(k), f"the coordinates given for {k!r} are of {parts[k].space}, not of "
                    f"the summand {space.summand(k)}", repr(k))
        if any(parts[k].export.codomain.field == "C" for k in keys):
            raise Refused("direct sums with complex parts come with complex function spaces (plan 020 P3b)")
        sizes = [parts[k].dim for k in keys]
        cuts = np.cumsum([0] + sizes)
        names = [space.names[space.index(k)] for k in keys]
        w = [space.weights[space.index(k)] for k in keys]
        R = Euclidean(int(cuts[-1]))

        def split(c):
            return [np.asarray(c, float)[a:b] for a, b in itertools.pairwise(cuts)]

        def export(x):
            return R(np.concatenate([parts[k].numbers(space.part(x, k)) for k in keys]))

        def export_adj(c):
            return space.element({n: (1.0 / wk) * parts[k].export.adjoint(parts[k].export.codomain(ck))
                                  for k, n, wk, ck in zip(keys, names, w, split(c.values))})

        def imp(c):
            return space.element({n: parts[k].element(ck) for k, n, ck in zip(keys, names, split(c.values))})

        def imp_adj(x):
            return R(np.concatenate([np.asarray(parts[k].imp.adjoint(wk * space.part(x, k)).values, float)
                                     for k, wk in zip(keys, w)]))

        E = LinearOperator(space, R, export, adjoint=export_adj, name=name)
        P = LinearOperator(R, space, imp, adjoint=imp_adj, name=f"{name}⁺")
        E._claims = [Claim("linear", THEOREM, "a stack of linear maps of the parts", key="linear")]
        P._claims = [Claim("linear", THEOREM, "a stack of linear maps of the parts", key="linear")]
        claims = []
        for key in ("bounded", "E P = I"):
            got = [parts[k].claims.find(key) for k in keys]
            if all(c is not None and c.holds for c in got):
                claims.append(Claim(key, DERIVED, f"from the parts ({', '.join(map(str, keys))})", key=key))
        return cls(space, E, imp=P, name=name, claims=claims)

    # ---------------------------------------------------------------------------------------------- use
    def numbers(self, m):
        """E m as a NumPy array (float, or complex for complex coordinates): what the external code is given."""
        return np.asarray(self.export(m).values)

    def element(self, x):
        """P x: the element of the space for the external code's numbers x."""
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"{self.name} has {self.dim} coordinates; got an array of shape {x.shape}")
        return self.imp(self.export.codomain(x))

    def functional(self, g, name="ω"):
        """h -> g . E h for components g of the numbers (module docstring): a Functional of the space's dual."""
        from ..spaces.dual import Functional
        from ..operators.linear import Matrix
        g = np.asarray(g, float)
        if g.shape != (self.dim,):
            raise ValueError(f"{self.name} has {self.dim} coordinates; got components of shape {g.shape}")
        A = Matrix(g[None, :], name="gᵀ") @ self.export
        return Functional.from_operator(A, name=name, claims=[Claim("bounded", DERIVED,
                                                                    f"g . E with E bounded ({self.name})", key="bounded")])

    def subset(self, predicate, name="S"):
        """{m : predicate(E m)}: a domain of definition given by external code on the numbers (module docstring)."""
        from ..sets.subsets import PredicateSubset
        return PredicateSubset(self.space, lambda m: bool(predicate(self.numbers(m))), name=name)

    def error(self, m):
        """||m - P E m||_M: what the round trip through the numbers loses for this element."""
        return self.space.norm(m - self.element(self.numbers(m)))

    @property
    def claims(self):
        return Claims(self._claims)

    def __repr__(self):
        return f"Coordinates({self.name}: {self.space} -> R^{self.dim})"
