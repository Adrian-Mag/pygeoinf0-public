"""The approximation role of a finite subspace (plan 068 P6, review B10; plan 025 P2 for the numbers): a Galerkin
discretisation of a Hilbert space M by the span V of N functions, D = pg.Discretisation(V), V = pg.Span(basis, space=M).

A finite subspace has two roles. As a set (convex/subspaces.Span) it is a ClosedSubspace of M with its generators, Gram
matrix, coefficient space, synthesis S and analysis S*, and orthogonal projection P = S S*: a grid taken as the model
space. As a discretisation (here) it stands for M in a problem that stays posed on M, and it says what it approximates
and how well, in the norm of M and never in coefficient coordinates:

    D.subspace              V, the Span (D.space = M)
    D.claims                V's claims, and the approximation claims:
                            "uniform approximation": sup over the unit ball of M of ||f - P f||_M = 1 when V is smaller
                            than M (THEOREM, holds False: a unit element orthogonal to V exists), V = M otherwise
                            "approximation": ||f - P f||_M <= e for every f in a class; UNKNOWN until D.state(e, over)
                            records the caller's statement (ASSERTED); a check never promotes it
                            "checked approximation": from D.check(fs), about those elements only (CHECKED)
    D.residual(f)           ||f - P f||_M
    D.check(fs)             one Enclosure of ||f - P f||_M per element: the residual on the realisations, widened by the
                            realisation errors of f and of the basis functions (CHECKED)
    D.state(e, over, how)   the caller's statement ||f - P f||_M <= e for f in `over` (a subset of M, or words)
    D.matrix(T)             T o S for an operator T from M into a Euclidean space (Functionals: the data x basis matrix),
                            a Matrix from the coefficient space
    D.pairings(measures)    the data x basis matrix (int phi_i dmu_j) of points, segments (rays) and boxes: exact and
                            sparse for local bases (realisations/series/local.py); densities w dx on full-dimensional
                            subsets (averaging kernels, bumps): L2 inner products of the local realisations of phi_i and
                            w (exact to their realisation errors, stated; plan 028); defined on the span even where
                            the functionals are unbounded on M (a ray on L2 with pixels), and then claimed as such
    D.galerkin(A, test=D)   K_ji = <psi_j, A phi_i> for A: M -> M' and a discretisation of M' by psi_j
    D.operator(A, test=D)   the Galerkin operator P' A S in coefficients: G'^-1 K, a Matrix between the two
                            coefficient spaces (its adjoint is the adjoint in the two metrics); exact on the span, and
                            nothing is claimed about ||A - P' A P|| on M (not computed here)
    D.gram_in(N)            the Gram of the same functions in another space N (bank functions and callables are made
                            again in N), e.g. the stiffness of the form space SobolevForm(B, 1, l).space()

    Synthesis(V), Analysis(V)   S: c -> sum_i c_i phi_i and S* = G^-1 (<phi_i, .>), the maps of a Span (V.synthesis)

The space M is unchanged (AGENTS.md: a space is never identified with a truncation). Every number of D is a number of M
on the span: the Gram, loads and pairings are M's inner products and functionals of the basis functions (exact on their
realisations, with the realisations' errors). With realisation="local", bases of compactly supported functions (pixels,
hats, bumps) have sparse Grams of their overlaps, and translates share their blocks (realisations/series/local.py). A
discretisation needs independent generators (V.coefficient_space); dependent ones are refused with their numerical rank.
"""
import math

import numpy as np
import scipy.sparse as sp

from ..core.claims import ASSERTED, CHECKED, DERIVED, THEOREM, UNKNOWN, Claim, Claims
from ..core.errors import Refused, SpaceMismatch, Unsupported
from ..core.numbers import Enclosure
from ..spaces.base import Euclidean
from ..spaces.elements import Function, Vector, VectorFamily
from .linear import LinearOperator, Matrix, synthesise
from ..spaces.identity import identical, require


def _dense(G):
    return G.toarray() if sp.issparse(G) else np.asarray(G, float)


def _full_density(m, dom):
    """A density measure w dx on a full-dimensional subset of the box `dom` with a sampled (non-singular) weight."""
    from ..measures import DensityMeasure
    S = getattr(m, "set", None)
    if not isinstance(m, DensityMeasure) or getattr(m.weight, "singular", False):
        return False
    if getattr(S, "dim", None) != getattr(dom, "dim", -1) or not hasattr(S, "_inside") or not hasattr(dom, "sides"):
        return False
    try:
        bb = S.bounding_box
    except Exception:
        return False
    return all(a >= lo - 1e-12 * (hi - lo) and b <= hi + 1e-12 * (hi - lo)
               for (a, b), (lo, hi) in zip(bb, dom.bounding_box))


def _finite(x):
    """Realisation errors with NaN (not on record) read as +inf."""
    return np.nan_to_num(np.asarray(x, float), nan=np.inf)


class Discretisation:
    """A Galerkin discretisation of M by a finite subspace V = pg.Span(basis, space=M) (module docstring)."""

    def __init__(self, subspace, *rest, name=None):
        if rest or not all(hasattr(subspace, a) for a in ("vectors", "coefficient_space", "project", "gram")):
            if rest:
                raise TypeError("pg.Discretisation takes the finite subspace (plan 068 P6): "
                                "pg.Discretisation(pg.Span(basis, space=M)) for pg.Discretisation(M, basis)")
            raise TypeError(f"pg.Discretisation takes a finite subspace (a pg.Span), not {type(subspace).__name__}")
        self.subspace = V = subspace
        self.space, self.name = V.space, name or getattr(V, "name", "D")
        V.coefficient_space                                    # independent generators, or Refused with the rank
        k = V.dim
        dim = V.space.dim if isinstance(V.space, Euclidean) else math.inf
        full = k == dim
        self._uniform = Claim(
            "V = M: every element of M is in the span" if full else
            f"sup over the unit ball of M of ||f - P f||_M = 1: the span (dimension {k}) does not approximate M "
            "uniformly", THEOREM,
            "a span of the dimension of M is M" if full else
            ("M is infinite-dimensional" if dim == math.inf else f"M has dimension {dim} > {k}") +
            ": a unit element orthogonal to the span exists, at distance 1 from it", key="uniform approximation",
            holds=full)
        self._approximation = Claim(
            "how well the span approximates the elements of M a problem needs is not stated", UNKNOWN,
            "state it with D.state(e, over) (asserted) or check given elements with D.check(fs) (checked, about those "
            "elements only)", key="approximation")
        self._checked = []

    @property
    def claims(self):
        return Claims(list(self.subspace.claims) + [self._uniform, self._approximation] + self._checked)

    def __len__(self):
        return len(self.subspace.vectors)

    # ---------------------------------------------------------------- the approximation claim
    def residual(self, f):
        """||f - P f||_M: the distance of f to the span, in the norm of M."""
        return float(self.space.norm(f - self.subspace.project(f)))

    def check(self, fs, name=None):
        """One Enclosure of ||f - P f||_M per element (an element, a family or a list of elements of M), CHECKED: the
        residual of the realisations, widened by the realisation errors of f and sum_i |c_i| err(phi_i) of P f (the
        distance of f to the span of the exact basis functions differs from the realised one by at most these). It
        records a CHECKED claim about these elements only; the claim about a class (D.state) is unchanged."""
        V = self.subspace
        items = [fs] if isinstance(fs, (Function, Vector)) else list(fs)
        if not items:
            raise ValueError("check needs at least one element")
        basis_err = _finite(V.vectors.errors()) if hasattr(V.vectors, "errors") else np.zeros(len(V.vectors))
        out = []
        for f in items:
            if getattr(f, "space", None) != self.space:
                raise SpaceMismatch(f"{getattr(f, 'name', f)} is not an element of {self.space}")
            r = self.residual(f)
            c = np.abs(np.asarray(V.coefficients(f), float)).ravel()
            margin = float(_finite(getattr(f, "error", 0.0))) + float(c @ basis_err)
            unknown = not math.isfinite(margin)
            how = ("the residual of the realisations; realisation errors not on record, so the ends are those of the "
                   "realisations" if unknown else
                   f"the residual of the realisations +- {margin:.2g} (realisation errors of f and of the basis)")
            m = 0.0 if unknown else margin
            out.append(Enclosure(r, max(r - m, 0.0), r + m, CHECKED, how))
        e = max(x.upper for x in out)
        what = name or (f"{len(items)} element" + ("s" if len(items) > 1 else ""))
        self._checked.append(Claim(f"||f - P f||_M <= {e:.3g} for the {what} checked", CHECKED,
                                   "computed on the realisations (D.check): it says nothing of other elements",
                                   key="checked approximation"))
        return out

    def state(self, e, over, how="stated by the caller"):
        """Record the caller's statement ||f - P f||_M <= e for every f in `over` (a subset of M, or words naming the
        class), ASSERTED: nothing here verifies it. It replaces an earlier statement."""
        e = float(e)
        if not (e >= 0 and math.isfinite(e)):
            raise ValueError(f"an approximation error is a finite number >= 0, not {e}")
        if not isinstance(over, str):
            if getattr(over, "space", None) != self.space:
                raise SpaceMismatch(f"the class must be a subset of {self.space} (or words naming it), not {over!r}")
            over = repr(over)
        self._approximation = Claim(f"||f - P f||_M <= {e:g} for every f in {over}", ASSERTED, how,
                                    key="approximation")
        return self._approximation

    # ---------------------------------------------------------------- operators
    def matrix(self, T):
        """T o S as a Matrix from the coefficient space, for T: M -> Euclidean (columns T(phi_i))."""
        require(T.domain, self.space, f"{T.name} acts on {T.domain}, not on {self.space}", T.name)
        if not isinstance(T.codomain, Euclidean):
            raise Unsupported(f"{T.name} maps into {T.codomain}: use D.galerkin(T, test) or D.operator(T, test) for "
                              "operators between function spaces")
        V = self.subspace
        A = np.asarray(T(V.vectors).values, float).reshape(len(self), -1).T
        return Matrix(A, domain=V.coefficient_space, codomain=T.codomain, name=f"{T.name}{self.name}")

    def pairings(self, measures, name="E"):
        """The (n, N) matrix of int phi_i dmu_j as a SparseMatrix from the coefficient space to R^n (module docstring).
        Local bases: exact (points, segments split at the pieces' breaks, coordinate boxes), and densities w dx on
        full-dimensional subsets S of the domain as the L2 inner products of the local realisations of phi_i and w 1_S
        (exact up to the realisation error of w 1_S, stated in the claim); otherwise the space's functionals (bounded ones
        only)."""
        from ..measures import as_measures
        from .sparse import SparseMatrix
        empty = isinstance(measures, (list, tuple)) and not len(measures)
        items = [] if empty else as_measures(measures, self.space.domain.cartesian_dim)[0]
        if not items:                                # a (0, N) matrix would need a zero-dimensional codomain
            raise Refused("pairings of no measures: give at least one (pygeoinf0 has no zero-dimensional spaces)")
        dens = [k for k, m in enumerate(items) if _full_density(m, self.space.domain)]
        parts, hows = [], []
        if dens:
            try:
                A, err = self._density_pairings([items[k] for k in dens])
                parts.append(sp.csr_matrix(A))
                hows.append(f"densities: L2 inner products of the local realisations of the basis and of w 1_S "
                            f"(realisation errors of these functions at most {err:.1e} in L2)")
            except Unsupported:
                dens = []
        rest = [k for k in range(len(items)) if k not in set(dens)]
        if rest:
            E, how = self._other_pairings([items[k] for k in rest])
            parts.insert(0, E)
            hows.insert(0, how)
        inv = np.argsort(np.array(rest + dens))
        E, how = sp.vstack(parts).tocsr()[inv], "; ".join(hows)
        out = SparseMatrix(E, domain=self.subspace.coefficient_space, codomain=Euclidean(len(items)), name=name)
        out._claims.append(Claim("int phi_i dmu_j", DERIVED, how + "; the functionals restricted to the span (on the "
                                 "space they may be unbounded)", key="pairings"))
        return out

    def _other_pairings(self, items):
        """(n, N) sparse pairings of measures other than full-dimensional densities, and how they were computed."""
        fn = getattr(self.space.backend, "pairings", None)
        if fn is not None:
            try:
                return sp.csr_matrix(fn(self.subspace.vectors.realisation, items).T), "exact on the realisations (realisations/series/local.py)"
            except Unsupported:
                pass
        from .functionals import Functionals
        try:
            T = Functionals(self.space, self.space.riesz(items))
        except Refused as e:
            raise Refused(f"{self.name}: these measures are not bounded functionals on {self.space} and the basis "
                          f"has no exact pairing with them ({e}); a local basis (realisation='local') has one") from e
        return sp.csr_matrix(self.matrix(T).A), ("the space's functionals of the basis functions (their Riesz "
                                                 "representers, with those representers' realisation errors)")

    def _density_pairings(self, mus):
        """(k, N) int phi_i w dx over S for density measures w dx on full-dimensional subsets S of the domain, and the
        largest L2 realisation error of the functions w 1_S (plan 028: Riesz representers of such measures in a Sobolev
        space carry their own fit error, 1e-4 relative for bumps in H^1)."""
        from ..core.errors import RealisationError
        from ..spaces.base import L2
        from .functionals import Functionals
        from .inclusion import Inclusion
        from ..measures import space_factor
        dom = self.space.domain
        c = space_factor(self.space) if isinstance(self.space, L2) else 1.0
        # L2 of the space's own measure when it is c dx (plan 064): <phi, w 1_S / c>_(c dx) = int phi w dx over S
        L = L2(dom if c in (1.0, None) else self.space.measure, realisation="local")
        c = 1.0 if c is None else c
        fs = []
        for m in mus:
            S, wt = m.set, m.weight
            whole = S == dom
            box = tuple((max(a, lo), min(b, hi)) for (a, b), (lo, hi) in zip(S.bounding_box, dom.bounding_box))

            def f(*X, S=S, wt=wt, whole=whole):
                P = np.stack(np.broadcast_arrays(*[np.asarray(x, float) for x in X]), axis=-1)
                v = wt.values(P.reshape(-1, P.shape[-1])).reshape(P.shape[:-1]) / c
                return v if whole else np.where(S._inside(*X), v, 0.0)
            try:
                fs.append(L(f, support=box))
            except RealisationError as e:
                raise Unsupported(f"the density of {m} has no local realisation ({e})") from e
        err = max(float(np.nan_to_num(np.asarray(x.error, float), nan=np.inf).max()) for x in fs)
        sources = self.subspace.sources
        if sources and all(hasattr(b, "element") and hasattr(b, "family") for b in sources):
            # bank functions: realised once in L2 on the box (the same function on every realisation), then one sparse
            # L2 Gram of the two local realisations (6241 hats, 100 bumps: seconds; through Inclusion: minutes)
            if getattr(self, "_l2_basis", None) is None or not identical(self._l2_basis[0], L):   # realised as L
                self._l2_basis = (L, L.family([b.element(L) for b in sources]))
            H = self._l2_basis[1]
            err = max(err, float(np.nan_to_num(H.errors(), nan=np.inf).max()))
            G = L.backend.gram(L.family(fs).realisation, H.realisation)
            return (G.toarray() if sp.issparse(G) else np.asarray(G, float)), err
        try:
            T = Functionals(L, fs) @ Inclusion(self.space, L) if not identical(self.space, L) else Functionals(L, fs)
        except (TypeError, ValueError, Unsupported, Refused) as e:
            raise Unsupported(f"no inclusion of {self.space} into L2 for density pairings ({e})") from e
        return np.asarray(self.matrix(T).A, float), err

    def galerkin(self, A=None, test=None):
        """(N', N) array K_ji = <psi_j, A phi_i> in the codomain of A (default: the identity, K = G when test is D)."""
        W = (self if test is None else test).subspace
        image = self.subspace.vectors if A is None else A(self.subspace.vectors)
        require(image.space, W.space, f"the test discretisation is of {W.space}; the images lie in {image.space}",
                "the images")
        return _dense(W.space.inner(W.vectors, image))

    def operator(self, A=None, test=None):
        """P' A S in coefficients: the Matrix G'^-1 K from this coefficient space to the test's. It is exact on the span
        (the spaces' inner products); its error as an approximation of A on M, ||A - P' A P||, is not claimed."""
        W = (self if test is None else test).subspace
        K = self.galerkin(A, test)
        out = Matrix(W.solve(K), domain=self.subspace.coefficient_space, codomain=W.coefficient_space,
                     name=f"{self.name}[{getattr(A, 'name', 'I')}]")
        out._claims.append(Claim("P' A S: the Galerkin operator of A between the two spans", DERIVED,
                                 "K_ji = <psi_j, A phi_i> and G'^-1 K, in the spaces' inner products; as an approximation "
                                 "of A on M its error is not claimed here", key="galerkin"))
        return out

    def gram_in(self, other):
        """The Gram of the same functions in another space (bank functions and callables are made again there)."""
        sources = self.subspace.sources
        if sources is None or any(s is None for s in sources):
            raise Unsupported("gram_in needs the basis functions themselves (bank functions or callables); this "
                              "span was given realised elements or a family")
        items = [s.element(other) if hasattr(s, "element") and hasattr(s, "support") else other(s) for s in sources]
        fam = other.family(items, name="phi")
        return _dense(other.inner(fam, fam))

    def __repr__(self):
        kind = "sparse" if sp.issparse(self.subspace.gram) else "dense"
        return f"{self.name}: a discretisation of {self.space} by the span of {len(self)} functions  [{kind} Gram]"


class Synthesis(LinearOperator):
    """S: c -> sum_i c_i phi_i from the coefficient space of a span V (independent generators) to its space
    (V.synthesis)."""

    why_linear = "a combination of fixed elements with the given coefficients"

    def __init__(self, V):
        super().__init__(V.coefficient_space, V.space, name=f"S_{V.name}")
        self.V = V

    def _apply(self, c):
        if isinstance(c, Vector):
            return synthesise(self.V.vectors, c.values, single=True)
        if isinstance(c, VectorFamily):
            return synthesise(self.V.vectors, c.values.T, single=False)
        raise TypeError(type(c).__name__)

    def _make_adjoint(self):
        return Analysis(self.V)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [synthesis from {len(self.V.vectors)} generators]"


class Analysis(LinearOperator):
    """S* = G^-1 (<phi_i, .>): the coefficients of the orthogonal projection onto the span (V.analysis)."""

    why_linear = "inner products with fixed elements, then a fixed Gram solve"

    def __init__(self, V):
        super().__init__(V.space, V.coefficient_space, name=f"S_{V.name}*")
        self.V = V

    def _apply(self, f):
        c = self.V.coefficients(f)
        if isinstance(f, (Function, Vector)):
            return Vector(self.codomain, c, name=f"c[{f.name}]")
        return VectorFamily(self.codomain, np.asarray(c).T, name="C")

    def _make_adjoint(self):
        return self.V.synthesis

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [analysis onto {len(self.V.vectors)} generators]"
