"""Selection of some of the data of a relation into R^n (plan 009 G9; EXP-030 NOTES.md section 9; named by plan 068 P7).

    R.select(rows)          rows: indices, a boolean mask, a slice, a predicate on the labels of the data space, or a
                            dict of label conditions (pg.Euclidean.select)

The result is the relation of the retained data alone. For a kernel it is the marginal of the kernel on those rows,
R_J D ~ N(R_J G m, R_J C R_J*): exact deletion, which the inflation of the other rows' noise by a factor f only
approaches, with an O(f^-2) error in the posterior. For a set-valued relation G m + N it is R_J G m + R_J(N), the data on
the rows explained by some noise of the set (a ball or ellipsoid projects to the ellipsoid of the marginal shape). For
an operator it is R_J G. The selected data space keeps the labels of its rows, and its standard Gaussian is the
marginal of the original's, so Euclidean(n, covariance=C) gives Euclidean(|J|, covariance=C_JJ).

    CoordinateProjection(D, idx)   R_J: D -> D_J, y -> (y_i)_(i in J), the operator behind the selection
"""
import numpy as np

from ..core.claims import DERIVED, THEOREM, Claim
from ..spaces.elements import Vector, VectorFamily
from ..core.errors import NotDefined, Unsupported
from ..spaces.base import Euclidean
from ..operators.linear import LinearOperator
from ..spaces.identity import identical


def rows_of(D, rows):
    """The indices of the retained coordinates of D (module docstring)."""
    if callable(rows):
        idx = D.select(rows)
    elif isinstance(rows, dict):
        idx = D.select(**rows)
    elif isinstance(rows, slice):
        idx = np.arange(D.dim)[rows]
    else:
        a = np.asarray(rows)
        if a.dtype == bool:
            if a.shape != (D.dim,):
                raise ValueError(f"a mask of the data has {D.dim} entries")
            idx = np.flatnonzero(a)
        else:
            idx = a.astype(int).ravel()
    if idx.size == 0:
        raise ValueError("no data retained")
    if idx.min() < -D.dim or idx.max() >= D.dim:
        raise IndexError(f"rows outside 0..{D.dim - 1}")
    idx = idx % D.dim
    if len(np.unique(idx)) != len(idx):
        raise ValueError("rows must be distinct")
    return idx


def marginal_space(D, idx):
    """The coordinates idx of D as a space whose standard Gaussian is the marginal of D's, with their labels."""
    if D.is_standard:
        out = Euclidean(len(idx))
    elif D.weights is not None:
        out = Euclidean(len(idx), weights=D.weights[idx])
    else:
        C = D.solve_metric(np.eye(D.dim)[:, idx])[idx]
        out = Euclidean(len(idx), covariance=0.5 * (C + C.T))
    out._labels = None if D._labels is None else D._labels.take(idx)
    return out


class CoordinateProjection(LinearOperator):
    """R_J: D -> D_J, y -> (y_i)_(i in J): the coordinate projection onto retained data (its adjoint scatters back
    through the metrics, <R_J y, z>_J = <y, R_J* z>)."""

    why_linear = "a coordinate projection"

    def __init__(self, D, idx, DJ=None, name="R_J"):
        idx = np.asarray(idx, int)
        super().__init__(D, marginal_space(D, idx) if DJ is None else DJ, name=name)
        self.idx = idx
        self._claims = [Claim("linear", THEOREM, "a coordinate projection", key="linear"),
                        Claim("bounded", THEOREM, "finite dimension", key="bounded")]

    def _apply(self, y):
        if isinstance(y, Vector):
            return Vector(self.codomain, y.values[self.idx], name=f"{y.name}_J")
        if isinstance(y, VectorFamily):
            return VectorFamily(self.codomain, y.values[:, self.idx], name=f"{y.name}_J")
        raise TypeError(type(y).__name__)

    def _make_adjoint(self):
        D, DJ, idx = self.domain, self.codomain, self.idx

        def adj(z):
            Z = DJ.apply_metric(np.atleast_2d(z.values).T)
            Y = np.zeros((D.dim, Z.shape[1]))
            Y[idx] = Z
            Y = D.solve_metric(Y)
            return Vector(D, Y[:, 0]) if isinstance(z, Vector) else VectorFamily(D, Y.T)
        return LinearOperator(DJ, D, apply=adj, adjoint=self._apply, name=f"{self.name}*")


def _selected_operator(S, G):
    """R_J G, keeping the structure of Functionals and matrices (and a Gram matrix already computed), with its ranges
    recorded (core/estimable.py: ran((R_J G)*) in ran G*)."""
    from ..core.estimable import within
    return within(_selected(S, G), S, G)


def _selected(S, G):
    from ..operators.functionals import Functionals
    from ..operators.linear import Matrix
    idx, DJ = S.idx, S.codomain
    if type(G) is Functionals:
        out = Functionals(G.kernels[idx], into=DJ, name=f"{G.name}_J")
        if G._gram is not None and isinstance(G._gram, np.ndarray):
            g = G._gram[np.ix_(idx, idx)]
            g.flags.writeable = False
            out._gram = g
        return out
    if hasattr(G, "select_rows"):                     # separable functionals: the structure kept
        return G.select_rows(idx, DJ)
    if type(G) is Matrix:
        return Matrix(G.A[idx], domain=G.domain, codomain=DJ, name=f"{G.name}_J")
    return S @ G


def _project(N, S):
    """R_J(N) for a noise set N of D (exact for balls and ellipsoids; otherwise the lazy image)."""
    from ..convex.algebra import Translate, image
    from ..convex.sets import Ellipsoid, NormBall
    DJ, idx = S.codomain, S.idx
    if isinstance(N, Translate):
        return Translate(_project(N.base, S), S(N.shift))
    if isinstance(N, NormBall) and identical(N.space, S.domain):
        return NormBall(DJ, N.radius, centre=None if N.centre is None else S(N.centre), name=f"{N.name}_J")
    if isinstance(N, Ellipsoid):
        return Ellipsoid(DJ, N.radius, covariance=N.shape[np.ix_(idx, idx)], centre=N.centre_values[idx], name=f"{N.name}_J")
    return image(S, N)


def select(R, rows):
    """The relation R into R^n of the data rows only, R.select(rows) (module docstring)."""
    from .additive import AdditiveSet
    from .kernels import AdditiveMarkovKernel
    from .linear import LinearRelation
    from ..convex.subspaces import image as sub_image
    D = R.codomain
    if not isinstance(D, Euclidean):
        raise NotDefined(f"a selection of data rows needs a finite-dimensional data space with coordinates, not {D}")
    idx = rows_of(D, rows)
    S = CoordinateProjection(D, idx)
    DJ = S.codomain
    why = "EXP-030 NOTES.md section 9: R_J D ~ N(R_J G m, R_J C R_J*), the marginal; with a diagonal C, deletion"
    claim = Claim(f"{len(idx)} of {D.dim} data selected: the marginal on those rows (exact deletion)", DERIVED, why,
                  key="selection")
    name = f"{getattr(R, 'name', 'R')}_J"
    if isinstance(R, AdditiveMarkovKernel):
        if R.nuisance is not None:
            raise Unsupported("a selection of data from a kernel with free nuisances: select from the relation before the quotient")
        mu = R.noise
        from ..operators.linear import Identity
        if isinstance(mu.covariance, Identity):
            nJ = DJ.standard_gaussian() if mu.zero_mean else DJ.standard_gaussian() + S(mu.mean)
        else:
            nJ = S(mu)
        out = AdditiveMarkovKernel(_selected_operator(S, R.G), nJ, name=name)
        out._claims += [c for c in R._claims if c.key == "offset"] + [claim]
        out.offset = None if getattr(R, "offset", None) is None else S(R.offset)
        out.rows, out.parent = idx, R
        return out
    if isinstance(R, AdditiveSet):
        cl = Claim(f"{len(idx)} of {D.dim} data selected: R_J G m + R_J(N), the data on those rows explained by "
                   "some noise of the set (a ball or ellipsoid projects to its marginal shape)", DERIVED,
                   "the projection of the noise set (EXP-030 NOTES.md section 9)", key="selection")
        out = AdditiveSet(_selected_operator(S, R.G), _project(R.noise, S), name=name,
                          claims=[c for c in R._claims if c.key == "offset"] + [cl])
        out.rows, out.parent = idx, R
        return out
    if isinstance(R, LinearRelation):
        if R.G is None:
            raise Unsupported(f"{R.name} has no operator part to select from")
        return LinearRelation(_selected_operator(S, R.G), mul=sub_image(S, R.mul), dom=R.dom, name=name,
                              claims=[claim])
    if isinstance(R, LinearOperator):
        out = _selected_operator(S, R)
        out._claims.append(claim)
        return out
    raise Unsupported(f"a selection of data rows from a {type(R).__name__} is not implemented (kernels, additive sets, "
                      "linear relations and operators are)")
