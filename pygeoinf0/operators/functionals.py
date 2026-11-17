"""Stacks of bounded linear functionals given by kernels, and their adjoints. One functional is a pg.Functional, an
element of the dual space (spaces/dual.py, plan 068 P2): Functional.from_representer(k) is m -> <k, m>.

Functionals(kernels)        m -> (<k_1, m>, ..., <k_n, m>): n functionals as one operator into R^n
                            (with integral kernels: the stack of integrals (G m)_i = int k_i m dmu)
Functionals(M, kernels, against=nu)   m -> int k_i m dnu on M = L2(mu): the representers k_i dnu/dmu (Radon-Nikodym),
                            refused (pg.Refused) when they are not square integrable, unless taper= (a declared
                            modification) makes them so (plan 009 G13: operators/separable.py, spaces/dual.py)
Functionals.separable(M, radial, harmonics)   k_i (x) Y_(l_i j_i) on spheres, balls, shells and their direct sums, with
                            structured Grams (plan 009 G2: operators/separable.py)
G[i]                        the i-th functional, a pg.Functional (Integrals: from its measure)
G.adjoint                   y -> sum_i (W y)_i k_i, W the metric of the data space
G @ G.adjoint               the Gram matrix, with derived claims (self-adjoint, positive semidefinite)

On a space of continuous functions (pg.Continuous) functionals are measures, not kernels:
Functional.from_measure(X, mu)   f -> int f dmu, bounded with norm |mu|(K) (Riesz-Markov: the dual of C(K) is the measures)
Integrals(X, measures)      f -> (int f dmu_1, ..., int f dmu_n); no adjoint into C, no G G*
"""

import numpy as np
import scipy.sparse as sp

from ..core.claims import CHECKED, DERIVED, THEOREM, UNKNOWN, Claim
from ..spaces.elements import Family, Function, Vector, VectorFamily
from ..measures import as_measures
from ..spaces.base import Euclidean, HilbertSpace
from ..spaces.dual import Functional
from .linear import LinearOperator, Matrix
from .sparse import KroneckerMatrix, SparseMatrix
from .refinement import _recompressed, _refined, chain_text  # a-posteriori Gram errors
from ..spaces.identity import identical


class Functionals(LinearOperator):
    """G m = (<k_1, m>, ..., <k_n, m>) for kernels k_i in a Hilbert function space (their Riesz representers).

    Functionals([k1, k2, ...]) maps into R^n; Functionals(kernels, into=D) into a given n-dimensional space D
    (D's inner product changes the adjoint, not G). With integral kernels this is the stack of integrals
    (G m)_i = int k_i m dmu."""

    why_linear = "inner products with fixed kernels"

    def __init__(self, kernels, *rest, into=None, name="G", against=None, taper=None):
        extra, sources = [], None
        if isinstance(kernels, HilbertSpace):               # Functionals(M, kernels, against=...): density kernels
            space, kernels, rest = kernels, rest[0], rest[1:]
            if against is not None or taper is not None:
                from .separable import density_family
                kernels, extra = density_family(space, kernels, against, taper)
            elif not isinstance(kernels, Family):
                kernels = [k if isinstance(k, Function) else space.element(k) for k in kernels]
                sources = _sources(kernels)
                kernels = space.family(kernels, name="k")
        elif against is not None or taper is not None:
            raise TypeError("against= and taper= need the space first: pg.Functionals(M, kernels, against=nu)")
        if rest:
            into = rest[0]
        if len(rest) > 1:
            name = rest[1]
        if isinstance(kernels, Family):
            K = kernels
        else:
            items = [_representer(k) if isinstance(k, Functional) else k for k in kernels]
            space = items[0].space
            sources = _sources(items)
            K = space.family(items, name="k")
        if not isinstance(K.space, HilbertSpace):
            raise TypeError(f"{K.space} has no inner product; functionals given by kernels need a Hilbert space")
        n = len(K)
        D = Euclidean(n) if into is None else into
        if not isinstance(D, Euclidean) or D.dim != n:
            raise ValueError(f"{n} functionals need an {n}-dimensional Euclidean codomain")
        super().__init__(K.space, D, name=name)
        self.kernels = K
        self._sources = sources                                # callables and breaks, for an a-posteriori Gram error
        self._knorm_cache, self._kerr_cache = None, None      # computed on first use (plan 022 P1)
        zero = self._zero_kernels(K)
        if np.any(zero):
            from ..core.errors import ZeroFunctional
            raise ZeroFunctional(f"functionals {np.flatnonzero(zero).tolist()} are zero on {K.space} (e.g. "
                                 "point values on a Dirichlet part, where every element vanishes): a datum there carries "
                                 "no information, and any value other than 0 contradicts the space. Leave them out, or "
                                 "use a space without the condition." + _realised_as(K, zero))
        self._gram = None
        self._claims.append(Claim("bounded", DERIVED,
                                  f"each kernel is in {K.space} (finite norm), so |<k_i, m>| <= ||k_i|| ||m||"))
        self._claims.append(Claim("linear", THEOREM, "inner products are linear"))
        self._claims += extra

    @classmethod
    def separable(cls, space, radial, harmonics, *, against=None, taper=None, into=None, name="G", quadrature=None,
                  convention=None):
        """Functionals k_i (x) Y_(l_i j_i) with structured Grams (operators/separable.py)."""
        from .separable import separable
        return separable(space, radial, harmonics, against=against, taper=taper, into=into, name=name,
                         quadrature=quadrature, convention=convention)

    def _zero_kernels(self, K):
        """Which kernels are the zero functional: read off the representers' atoms when the backend can (no energy
        work: plan 022 P1), else from the kernels' norms (then kept)."""
        exact = getattr(K.space.backend, "exact_zeros", None)
        zero = exact(K.realisation) if exact is not None else None
        if zero is None:
            zero = np.asarray(self._knorm) == 0
        return zero

    @property
    def _knorm(self):
        """||k_i||, on first use (it needs the energies of representers)."""
        if self._knorm_cache is None:
            self._knorm_cache = self.kernels.space.norm(self.kernels)
        return self._knorm_cache

    @property
    def _kerr(self):
        """The kernels' estimated errors, on first use (for surface and weighted measures they need self-energies)."""
        if self._kerr_cache is None:
            self._kerr_cache = self.kernels.errors()
        return self._kerr_cache

    def _apply(self, x):
        M = self.domain
        if isinstance(x, Function) and x._terms is not None and x._pl is None:     # a combination: by linearity
            parts = [(c, self._apply(t)) for c, t in x._terms]                    # (spaces/elements.py, plan 022 P7)
            v = sum(c * p.values for c, p in parts)
            err = sum(abs(c) * np.asarray(p.err, float) for c, p in parts)       # the triangle inequality
            return Vector(self.codomain, v, err=err, name=f"{self.name}({x.name})")
        if isinstance(x, Function):
            v = M.inner(self.kernels, x)
            pe = getattr(M.backend, "last_pairing_error", None)       # weighted pieces: the two orders (EXP-031 O4)
            xe = np.asarray(x.error, float)
            # ||k_i|| x.error: not needed when x.error is 0 (exact) or unknown (NaN, a formula in H^s)
            known = np.isfinite(xe) & (xe != 0)
            err = (self._knorm * xe if np.any(known) else xe + np.zeros(len(v))) + \
                (pe[:, 0] if pe is not None and pe.shape == (len(v), 1) else 0.0)
            ask = getattr(M.backend, "errors_pending", None)          # representers' errors not yet computed
            if (ask is not None and ask(self.kernels.realisation)) or np.any(self._kerr):   # ||x|| matters only when the kernels carry errors
                xn = float(np.sqrt(max(M.backend.sqnorms_or(x.realisation, np.nan)[0], 0.0)))   # NaN: formula in H^s
                err = err + (np.nan if np.isnan(xn) else xn * self._kerr)    # unknown ||x||: the errors are not needed
            return Vector(self.codomain, v, err=err, name=f"{self.name}({x.name})")
        if isinstance(x, Family):
            G = M.inner(self.kernels, x)          # (k, N); scipy.sparse from a local realisation with large N
            G = G.toarray() if sp.issparse(G) else G
            return VectorFamily(self.codomain, G.T, name=f"{self.name}({x.name})")
        raise TypeError(type(x).__name__)

    def gram(self):
        """The Gram matrix <k_i, k_j> (computed once)."""
        if self._gram is None:
            self._gram = self.domain.inner(self.kernels, self.kernels)
            if isinstance(self._gram, np.ndarray):
                self._gram.flags.writeable = False
        return self._gram

    def _make_adjoint(self):
        return FunctionalsAdjoint(self)

    def _compose(self, other):
        if isinstance(other, FunctionalsAdjoint) and identical(other.G.domain, self.domain):
            H = other.G
            from ..realisations.registry import provides
            from ..settings import settings
            n = max(self.codomain.dim, H.codomain.dim)
            if n > settings.dense_max and (symbolic_representers(self.kernels) or
                                           not provides(self.kernels.realisation, "kronecker_gram")):
                out = super()._compose(other)                    # products G(H* y): no n x n matrix
                out._claims.append(Claim(f"applied as {self.name}({H.name}*(y)) without forming the {n} x {n} matrix",
                                         DERIVED, f"n = {n} > settings.dense_max = {settings.dense_max}"))
                out._claims.append(_execution_claim(self.domain, kernels=self.kernels))
                if H is self:
                    out._sa = (True, DERIVED, f"of the form {self.name} {self.name}*")
                    out._pos = ("psd", DERIVED, f"<{self.name}{self.name}* y, y> = ||{self.name}* y||^2 >= 0")
                return out
            Gr = self.gram() if H is self else self.domain.inner(self.kernels, H.kernels)
            if provides(Gr, "kronecker_product"):
                if not H.codomain.is_standard and H.codomain.weights is None:
                    raise NotImplementedError("a non-diagonal data metric with a Kronecker Gram matrix")
                out = KroneckerMatrix(Gr, weights=H.codomain.weights, domain=H.codomain, codomain=self.codomain,
                                      name=f"{self.name}{H.name}*")
            elif isinstance(Gr, np.ndarray):
                A = Gr if H.codomain.is_standard else H.codomain.apply_metric(Gr.T).T
                out = Matrix(A, domain=H.codomain, codomain=self.codomain, name=f"{self.name}{H.name}*")
            else:
                import scipy.sparse as sp
                if not H.codomain.is_standard:
                    if H.codomain.weights is None:
                        raise NotImplementedError("a non-diagonal data metric with a sparse Gram matrix")
                    Gr = Gr @ sp.diags(H.codomain.weights)
                out = SparseMatrix(Gr, domain=H.codomain, codomain=self.codomain, name=f"{self.name}{H.name}*")
            if H is self:
                out._sa = (True, DERIVED, f"of the form {self.name} {self.name}*")
                out._pos = ("psd", DERIVED, f"<{self.name}{self.name}* y, y> = ||{self.name}* y||^2 >= 0")
            out._claims.append(_execution_claim(self.domain, dense=True, kernels=self.kernels))
            if H is self:                                   # what a later inverse inherits (plan 054 P3), on first use
                out._stated_error = lambda G=self, Gr=Gr, D=H.codomain: _gram_error(G, Gr, D)
                out._sharper_error = lambda G=self, Gr=Gr, D=H.codomain: _gram_error(G, Gr, D, refine=True)
            return out
        return super()._compose(other)

    def __getitem__(self, i):
        """The i-th datum's functional; its kernel k_i = G*(W^-1 e_i) is estimable by construction (core/estimable.py)."""
        from ..core import estimable
        return Functional.from_representer(estimable.mark(self.kernels[i], self.adjoint))


def _representer(l):
    """The Riesz representer of a Functional in a stack of kernels (refused for functionals without one)."""
    if l.representer is None:
        raise TypeError(f"{l.name} has no Riesz representer to stack: pg.Functionals takes functionals given by "
                        "representers (Functional.from_representer, X.functional on a Hilbert space)")
    return l.representer


def symbolic_representers(kernels):
    """Symbolic representers (spaces/representers.py): no realisation in the backend is needed or made."""
    return hasattr(kernels, "combine")


def _sources(items):
    """[(callable, breaks)] of kernels that are Functions built from callables, else None."""
    if not items or not all(isinstance(k, Function) and getattr(k, "_src", None) is not None for k in items):
        return None
    return [(k._src, k._breaks) for k in items]


def _gram_error(G, Gr, D, refine=False):
    """(e, status, how) for A = G G* on D (plan 054 P3): e bounds, or estimates, ||A - A_true|| in D's norm, from what
    pygeoinf0 states about the Gram's inputs; None when it states nothing beyond rounding.
      - the kernels' realisations: ||k~_i - k_i|| <= e_i (Family.errors(), estimates), so |G~_ij - G_ij| <=
        e_i n_j + e_j n_i + e_i e_j entrywise (n_i = ||k~_i||) and ||G~ - G||_2 <= 2 ||e|| ||n|| + ||e||^2 (a matrix
        dominated entrywise by a nonnegative B has ||X||_2 <= ||B||_2);
      - compressed energies (settings.dense_gram = "compressed"): the H-matrix's estimate rel_error ||G||_F;
      - D's metric W: ||W^(1/2) dG W^(1/2)|| <= ||W|| ||dG|| (D's orthonormal coordinates)."""
    from ..core.claims import weakest
    from ..settings import settings
    statuses = []
    e_k, how = 0.0, []
    if not symbolic_representers(G.kernels):
        err = np.asarray(G._kerr, float)
        if err.size and np.all(np.isfinite(err)) and err.max(initial=0.0) > 0:
            nrm = np.asarray(G._knorm, float)
            e_k = 2.0 * float(np.linalg.norm(err) * np.linalg.norm(nrm)) + float(err @ err)
            ref = _refined(G, Gr) if refine else None
            if ref is None:
                how.append(f"the kernels' realisations (errors up to {err.max():.2g}, estimates): <= {e_k:.3g}")
            else:
                lev = "" if len(ref) < 4 else f"; {chain_text(*ref[3])}"
                how.append(f"the kernels' realisations: {ref[0] + ref[2]:.3g} a posteriori (||G~ - G'|| = {ref[0]:.3g}, G' "
                           f"the Gram of the kernels realised again at tol {ref[1]:.3g}, plus its own first-order bound "
                           f"{ref[2]:.3g}{lev}; a bound where the kernels' stated errors are, which are estimates; the "
                           f"first-order bound from the first kernels' errors, up to {err.max():.2g}, is {e_k:.3g})")
                e_k = ref[0] + ref[2]
            statuses.append(CHECKED)
        elif err.size and not np.all(np.isfinite(err)):
            how.append("the kernels' realisation errors: unknown for some kernels")
            statuses.append(UNKNOWN)
    kx = None if symbolic_representers(G.kernels) else getattr(getattr(G.kernels, "realisation", None), "kx", None)
    Hm = getattr(getattr(kx, "atoms", None), "hmatrix", None)
    e_c = 0.0
    if Hm is not None and settings.dense_gram == "compressed" and Hm.rel_error > 0:     # 0: no far block
        GrA = Gr.toarray() if hasattr(Gr, "toarray") else np.asarray(Gr)
        e_c = float(Hm.rel_error) * float(np.linalg.norm(GrA))
        ref = _recompressed(G, GrA, kx.atoms, Hm) if refine else None
        if ref is None:
            how.append(f"compressed energies (settings.compression = {Hm.compression:g} when the H-matrix was built; "
                       f"estimated relative error {Hm.rel_error:.1e} Frobenius): about {e_c:.3g}")
        else:
            lev = "" if len(ref) < 4 else (f"; verification level {ref[3][0]}" + ("" if ref[3][0] == ref[3][1] else
                                                                                  f" of the {ref[3][1]} asked"))
            how.append(f"compressed energies: {ref[0] + ref[1]:.3g} a posteriori, an estimate (||G~ - G'|| = {ref[0]:.3g}, "
                       f"G' the Gram with the energies compressed again at {ref[2]:.3g}, plus its own estimate "
                       f"{ref[1]:.3g}{lev}; the first estimate, relative error {Hm.rel_error:.1e} Frobenius, is {e_c:.3g})")
            e_c = ref[0] + ref[1]
        statuses.append(CHECKED)
    if not how:
        return None
    w = 1.0 if D.is_standard else (float(np.max(D.weights)) if D.weights is not None else
                                   float(np.linalg.eigvalsh(D.apply_metric(np.eye(D.dim))).max()))
    parts = (e_k + e_c) * w
    return parts, weakest(*statuses), "; ".join(how) + ("" if w == 1.0 else f"; times ||W|| = {w:.3g} of D's metric")


def _execution_claim(space, dense=False, kernels=None):
    """How the energies behind a Gram operator are computed (time and rounding; with an H-matrix, its stated error;
    with another engine of settings.engine, its name and estimate: the spectral engine is prepared here)."""
    from ..settings import settings
    engine = getattr(space.backend, "engine", None)
    kx = None if symbolic_representers(kernels) else getattr(getattr(kernels, "realisation", None), "kx", None)
    atoms = getattr(kx, "atoms", None)
    H = getattr(atoms, "hmatrix", None)
    other = None if dense or atoms is None or not hasattr(engine, "describe") else engine.describe(atoms)
    if engine is None:
        how = f"{space.backend.realisation_name} realisation"
    elif dense and (H is None or settings.dense_gram != "compressed"):
        how = "exact energies"
    elif other is not None:
        how = other
    else:
        how = (f"{engine.name} energies (settings.compression = "
               f"{settings.compression if H is None else H.compression:g}"
               + (f" when the H-matrix was built; estimated relative error {H.rel_error:.1e})" if H is not None else ")"))
    return Claim(f"computed with {how}, {settings.workers} worker process(es)", DERIVED, "pg.settings at construction")


class FunctionalsAdjoint(LinearOperator):
    """G* y = sum_i (W y)_i k_i, so that <G m, y>_D = <m, G* y>_M (W is the metric of the data space)."""

    why_linear = "a combination of fixed kernels with the coefficients W y"

    def __init__(self, G):
        super().__init__(G.codomain, G.domain, name=f"{G.name}*")
        self.G = G

    def _apply(self, y):
        Ks = self.G.kernels
        if symbolic_representers(Ks):                                # symbolic representers stay symbolic
            C = self.domain.apply_metric(y.values if isinstance(y, Vector) else y.values.T)
            out = Ks.combine(C.reshape(len(Ks), -1))
            return out[0] if isinstance(y, Vector) else out
        K = Ks.realisation
        if isinstance(y, Vector):
            c = self.domain.apply_metric(y.values)
            return Function(self.codomain, pl=self.codomain.backend.combine(K, c[:, None]), name=f"{self.name}({y.name})")
        if isinstance(y, VectorFamily):
            C = self.domain.apply_metric(y.values.T)
            return Family(self.codomain, self.codomain.backend.combine(K, C), name=f"{self.name}({y.name})")
        raise TypeError(type(y).__name__)

    def _make_adjoint(self):
        return self.G

    def _compose(self, other):
        out = super()._compose(other)
        if other is self.G:
            out._sa = (True, DERIVED, f"of the form {self.G.name}* {self.G.name}")
            out._pos = ("psd", DERIVED, f"<{self.G.name}*{self.G.name} m, m> = ||{self.G.name} m||^2 >= 0")
        return out


class Integrals(LinearOperator):
    """G f = (int f dmu_1, ..., int f dmu_n) on a space of continuous functions (pg.Continuous): n functionals given by
    finite measures (points, subintervals, weighted points, densities) as one operator into R^n (or into=D).

    The space has no inner product, so G has no adjoint into it: G* maps R^n into the dual of C, the measures
    (y -> sum_i y_i mu_i). There is no G G* and no least squares on C; for those use a Hilbert space (pg.Sobolev,
    where points and intervals have Riesz representers, or pg.L2)."""

    why_linear = "integrals against fixed finite measures"

    def __init__(self, space, measures, into=None, name="G"):
        if not hasattr(space, "integrate_against"):
            raise TypeError(f"Integrals act on spaces of continuous functions (pg.Continuous); on {space} use "
                            "pg.Functionals(M.riesz(measures))")
        items, _ = as_measures(measures, 1)
        D = Euclidean(len(items)) if into is None else into
        if not isinstance(D, Euclidean) or D.dim != len(items):
            raise ValueError(f"{len(items)} functionals need an {len(items)}-dimensional Euclidean codomain")
        super().__init__(space, D, name=name)
        self.measures = items
        self._claims.append(Claim("linear", THEOREM, "integrals are linear"))
        self._claims.append(Claim("bounded", THEOREM, "|int f dmu_i| <= mu_i(K) max |f|"))

    def _apply(self, f):
        v = self.domain.integrate_against(f, self.measures)
        if isinstance(f, Function):
            return Vector(self.codomain, v, name=f"{self.name}({f.name})")
        if isinstance(f, Family):
            return VectorFamily(self.codomain, v.T, name=f"{self.name}({f.name})")
        raise TypeError(type(f).__name__)

    def norms(self):
        """The norm of each functional: the total mass of its measure."""
        from ..measures import total_variation
        return np.array([total_variation(mu) for mu in self.measures])

    def __getitem__(self, i):
        return Functional.from_measure(self.domain, self.measures[i])


def _realised_as(K, zero):
    """The realisation of the zero kernels, named (a zero from a fit whose samples missed a narrow density is not the
    zero functional: plan 035 P2)."""
    from ..spaces.representers import symbolic
    if symbolic(K):
        return ""
    B = K.space.backend
    out = [f"{i}: {B.summary(B.take(K.realisation, int(i)))}" for i in np.flatnonzero(zero)[:3]]
    return (f" Their representers are realised as zero by the {getattr(B, 'realisation_name', 'backend')} "
            f"realisation ({'; '.join(out)}): if a fit of a density produced them, its samples may have missed a "
            "narrow or off-axis support; give breaks= at the support (a bank function supplies them), or keep the "
            "measure symbolic (f.measure()).")
