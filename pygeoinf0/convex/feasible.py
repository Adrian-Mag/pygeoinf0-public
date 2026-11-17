"""Feasible sets of set-valued inversions and their images under a linear model-property relation (plan 008 cells B2,
B5, B6; inv:399-426, inf:512-529):

    K = FeasibleSet(prior, G, noise, d)    {m in prior : d - G m in noise}; empty is a finding, not an error
    K.member(m)                            three-valued
    K.bound(c), K.support(c)               sup over K of <c, m> for c in the model space (optimisers/dual/): a Bound, an
                                           Enclosure whose lower end is a witness's value and upper end the dual at a
                                           certificate (support returns it: plan 067 P3); estimable=True declares c in
                                           the data functionals' span (core/estimable.py)
    K.is_empty()                           Truth, with a Farkas certificate in K.emptiness when TRUE
    P = T(K) (K.image(T))                  FeasibleImage: P.bound(q), P.support(q), P.interval(k) (the certified outer
                                           ends: the lower end of the inf, the upper end of the sup), P.bounds(k) (both
                                           Bounds with their values), P.outer_polytope(Q); K.image(T, estimable=True)
                                           or P.estimable = True declares T's directions so

With no prior a target in the data's span only to rounding cannot be told from one outside it (bound +inf): it is
estimable when declared or built from the data functionals (core/estimable.py), not by its numbers. K.dependent
declares the data exactly dependent at the rounding floor; it says nothing about targets.

The prior is a NormBall of the model space (Gram-exact), an operator Ellipsoid {<A m, m> <= a^2} (through the
decomposition with the middle operator A^-1), a PointwiseBox (convex/feasible_box.py and optimisers/dual/nonsmooth.py:
a certified dual and cell witnesses, statuses with the gap; K.cells may hold a CellProblem factory, K.sub sets the
interval cells per piece) or none (pg.NoPrior(X), which is Whole(X)); other priors give membership only. The noise set is
any subset of the data space with a support function (balls, ellipsoids, slabs, spans of free nuisances and their sums,
boxes, polytopes; convex/forms.py turns them into the solvers' forms).

A ball or ellipsoid of noise (without nuisances) is solved through a data-space decomposition (plan 009 G3,
solvers/dataspace/decomposition.py): one eigendecomposition of the whitened Gram matrix for the set and all its images, then
O(n) per direction, with the certificate evaluated in the same eigenbasis. FeasibleSet(..., decomposition=dec) uses a
given one (pg.decompose(G, noise)), decomposition=False the reduction of the joint Gram matrix per direction below.
"""
import math

import numpy as np

from ..core import estimable as _how
from ..core.claims import DERIVED, THEOREM, Claim
from ..core.errors import Unsupported
from ..core.precision import TOL
from ..core.logic import Truth
from ..optimisers.dual import CERTIFIED, Bound, exactness, exactness_reduce, reduction
from . import forms
from .sets import ConvexSet, NormBall, OperatorEllipsoid, coordinate_direction
from ..spaces.identity import identical


class FeasibleSet(ConvexSet):
    """{m in prior : d - G m in noise} (module docstring)."""

    def __init__(self, prior, G, noise, d, name="K(d)", decomposition=None, cache=None, dependent=False):
        from ..operators.claim_rules import status
        from .subspaces import Whole
        self.prior = Whole(G.domain) if prior is None else prior
        self.G, self.noise, self.d = G, noise, d
        cl = [Claim("may be empty: then prior, relation and datum are jointly inconsistent (a finding)", THEOREM,
                    "inv:426")]
        lin = status(G, "linear")[0] is Truth.TRUE
        if lin and all(S.claims.status("convex")[0] is Truth.TRUE for S in (self.prior, noise)):
            cl.append(Claim("convex", DERIVED, "prior and noise convex, G linear (inf:529)", key="convex"))
        if status(G, "bounded")[0] is Truth.TRUE and all(S.claims.status("closed")[0] is Truth.TRUE
                                                        for S in (self.prior, noise)):
            cl.append(Claim("closed", DERIVED, "prior and noise closed, G bounded", key="closed"))
        if self.prior.claims.status("bounded")[0] is Truth.TRUE:
            cl.append(Claim("bounded", DERIVED, "a subset of a bounded prior set", key="bounded"))
        super().__init__(G.domain, name, cl)
        self._GG = self.emptiness = None
        self.decomposition, self._cache, self._sp = decomposition, cache, None
        self.cells = None                                    # box priors: a CellProblem or a factory c -> CellProblem
        self.sub = 4                                         # box priors: interval cells per piece (optimisers/dual/nonsmooth.py)
        self.dependent, self._dep = dependent, None          # data declared exactly dependent at the rounding floor

    # membership -----------------------------------------------------------------------------------------------
    def member(self, m):
        return self.prior.member(m) & self.noise.member(self.d - self.G(m))

    # the problem in coordinates ---------------------------------------------------------------------------------
    def _prior(self):
        from .subspaces import Whole
        P = self.prior
        if isinstance(P, (NormBall, OperatorEllipsoid)):
            return P.radius, P.centre
        if isinstance(P, Whole):
            return math.inf, None
        raise Unsupported(f"support functions of feasible sets are computed for a norm-ball prior or none; the prior "
                          f"{P.name} gives membership only")

    def gram(self):
        if self._GG is None:
            self._GG = reduction.gram_blocks(self.G)[0]
        return self._GG

    def dependency(self):
        """(unexplained, at the floor, structural, n): correlation eigenvalues of the data Gram at most eps n max (data
        representers dependent to rounding), less those that are exactly zero by structure (n - dim M for a
        finite-dimensional model space; structural_zeros). The unexplained ones cannot be told from exact
        dependence (plan 054 fix round)."""
        if self._dep is None:
            J = self.gram()
            d = np.sqrt(np.clip(np.diag(J), 0.0, None))
            d = np.where(d > 0.0, d, 1.0)
            lam = np.linalg.eigvalsh(0.5 * (J + J.T) / np.outer(d, d))
            n = len(lam)
            low = int(np.sum(lam <= reduction.EPS * n * max(lam.max(initial=0.0), 1e-300)))
            st = reduction.structural_zeros(n, self.G)
            self._dep = (max(low - st, 0), low, st, n)
        return self._dep

    def _spectral(self):
        """The BGP problem in the eigenbasis of a data-space decomposition, or None where that route does not apply
        (other priors or noise sets, nuisances, a partial decomposition): module docstring."""
        if self._sp and self._sp.dependent != self.dependent:     # K.dependent set after a first bound
            self._sp = None
        if self._sp is None:
            from .subspaces import Whole
            self._sp = False
            shape = ball_shape(self.noise, self.G.codomain)
            prior_ok = isinstance(self.prior, (NormBall, OperatorEllipsoid, Whole))
            if self.decomposition is not False and shape is not None and prior_ok:
                from ..solvers.dataspace import cached, decomposition_key
                from .forms import noise_metric
                base, centre, rho = shape
                dec, S = self.decomposition, self._middle()
                try:
                    metric = noise_metric(base, self.G.codomain)              # the noise set's shape, stated
                    if dec is None:
                        dec = cached(self._cache, self.G, metric, S)
                    elif not (dec.G is self.G and dec.middle is S and
                              decomposition_key(self.G, metric, S)[1] == dec.noise.key):
                        raise ValueError("the decomposition is not that of this G with the noise set's shape and the "
                                         "prior's metric (middle operator)")
                    if not dec.partial:
                        r, m0 = self._prior()
                        e = self.d.values - centre - (0.0 if m0 is None else self.G(m0).values)
                        self._sp = dec.ball_problem(e, r, rho, self.dependent, centre=m0)
                except (Unsupported, MemoryError, np.linalg.LinAlgError):
                    self._sp = False
        return self._sp or None

    def _middle(self):
        """The prior's metric operator for the decomposition: A^-1 of an operator ellipsoid, else None (identity)."""
        return self.prior.inverse if isinstance(self.prior, OperatorEllipsoid) else None

    def _box(self):
        from .pointwise import PointwiseBox
        return isinstance(self.prior, PointwiseBox)

    def bound(self, c, estimable=False):
        """sup over K of <c, m> for an element c of the model space, with its bracket and certificate. estimable=True
        declares c exactly in the span of the data functionals (the caller's responsibility; core/estimable.py): with
        no prior it decides whether a target there to rounding is bounded. Built from them, c is so by construction."""
        M = self.space
        if self._box():
            from .feasible_box import feasible_bound
            return feasible_bound(self, c, self.cells)
        r, m0 = self._prior()
        how = _how.target(c, self.G, estimable)
        sp = self._spectral()
        if sp is not None:
            S = self._middle()
            Sc = c if S is None else S(c)
            return sp.bound(sp.dec._coef(self.G(Sc).values), float(M.inner(c, Sc)),
                            0.0 if m0 is None else float(M.inner(c, m0)), how, exactness.element_error(c))
        if isinstance(self.prior, OperatorEllipsoid):
            raise Unsupported("an operator-ellipsoid prior is solved through a data-space decomposition (ball or "
                              "ellipsoid noise); none applies here")
        gc = np.asarray(self.G(c).values, float)
        J = np.block([[np.array([[M.inner(c, c)]]), gc[None, :]], [gc[:, None], self.gram()]])
        info = {}
        Yd, B = exactness_reduce.reduce_stated(J, 1, info, [exactness.element_error(c)], self.G, r, how)
        const = 0.0 if m0 is None else float(M.inner(c, m0))
        return solve(self, Yd[:, 0], B, const, r, m0, info, np.ones(1), how=how)

    def support(self, c, estimable=False):
        """The Bound of sigma_K(c): an Enclosure with certified ends (bound's docstring); float(K.support(c)) is the
        solver's value, which may sit at either end."""
        return self.bound(c, estimable)

    def is_empty(self):
        """Truth of K = {} (a direction-0 problem); TRUE comes with a Farkas certificate in self.emptiness."""
        if self._box():
            return self._box_empty()
        r, m0 = self._prior()
        sp = self._spectral()
        if sp is not None:                                   # direction 0 = G* 0: in the data's span by construction
            b = sp.bound(np.zeros(len(sp.delta)), 0.0, 0.0, _how.BUILT)
        else:
            info = {}
            B = reduction.reduce(self.gram(), 0, info)[1]
            b = solve(self, np.zeros(B.shape[1]), B, 0.0, r, m0, info)
        if b.status == "EMPTY":
            self.emptiness = b
            return Truth.TRUE
        return Truth.FALSE if math.isfinite(b.lower) or b.status == "EXACT" else Truth.UNKNOWN

    def image(self, T, estimable=False):
        return FeasibleImage(self, T, estimable)

    def _box_empty(self, enclosing=None):
        """A box prior: TRUE with a certificate in self.emptiness, FALSE with a witness, else UNKNOWN
        (optimisers/dual/nonsmooth.py)."""
        from .feasible_box import box_empty
        return box_empty(self)


class FeasibleImage(ConvexSet):
    """T(K) for a feasible set K and a linear T into R^p: sigma(q) = sigma_K(T* q) by the dual (inf:699-911).
    estimable (settable): T's directions declared in the data functionals' span (core/estimable.py)."""

    def __init__(self, K, T, estimable=False):
        cl = [Claim("convex", DERIVED, "the image of a convex set under a linear map", key="convex")] \
            if K.claims.status("convex")[0] is Truth.TRUE else []
        if K.claims.status("bounded")[0] is Truth.TRUE:
            cl += [Claim("bounded", DERIVED, "the image of a bounded set under a bounded map", key="bounded"),
                   Claim("closed", DERIVED, "the image of a weakly compact set (inf:529)", key="closed")]
        super().__init__(T.codomain, f"{T.name}({K.name})", cl)
        self.K, self.T, self._red, self._pert, self.estimable = K, T, None, None, estimable

    def _estimability(self):
        """T's estimability for K's data: DECLARED, BUILT or None (core/estimable.py)."""
        return _how.property_map(self.T, self.K.G, self.estimable)

    def member(self, p):
        return Truth.UNKNOWN if self.K.is_empty() is not Truth.TRUE else Truth.FALSE

    def _reduction(self):
        if self._red is None:
            GG, GT, TT = reduction.gram_blocks(self.K.G, self.T)
            self._red_info = {}
            self._red = reduction.reduce(np.block([[TT, GT.T], [GT, GG]]), TT.shape[0], self._red_info)
        return self._red

    def bound(self, q):
        """sup over T(K) of <q, p>_P, with its bracket and certificate."""
        P = self.space
        qv = np.asarray(getattr(q, "values", q), float).ravel()
        if self.K._box():                                   # <q, T m>_P = <T* q, m>: a direction of the model space
            return self.K.bound(self.T.adjoint(P(qv)), self.estimable)
        Wq = P.apply_metric(qv)
        r, m0 = self.K._prior()
        sp = self.K._spectral()
        if sp is not None:
            dec = sp.dec
            tg = dec.targets(self.T)
            const, te = 0.0 if m0 is None else float(Wq @ self.T(m0).values), exactness.direction_error(tg, Wq)
            if tg.tt is None or getattr(tg, "Phi", None) is not None:      # many properties or a factored family
                from ..solvers.dataspace.factored import direction
                return sp.bound(*direction(dec, tg, Wq), const, self._estimability(), te)
            return sp.bound(dec._beta(tg) @ Wq, float(Wq @ tg.tt @ Wq), const, self._estimability(), te)
        Yt, B = self._reduction()
        const = 0.0 if m0 is None else float(Wq @ self.T(m0).values)
        how = exactness_reduce.columns(self._red_info, self.T, self.K.G, r, self._estimability())   # stated errors
        return solve(self.K, Yt @ Wq, B, const, r, m0, self._red_info, Wq, lambda: self._perturbed(how), how)

    def _perturbed(self, how):
        """The general rule's perturbed reductions of the joint Gram matrix, once per estimability (convex/
        exactness_reduce.py)."""
        if self._pert is None or self._pert[0] != how:
            perturbed = exactness_reduce.perturbed
            red = self._red_info["reduction"]
            self._pert = (how, perturbed(red, _stated_zeros(self.K, len(red[1]), red[5], how)))
        return self._pert[1]

    def support(self, q):
        """The Bound of sigma_P(q) (an Enclosure with certified ends; its value is the solver's)."""
        return self.bound(q)

    def bounds(self, k=0):
        """(lower, upper) Bounds of the k-th property over the feasible set."""
        e = coordinate_direction(self.space, k).values
        return self.bound(-e).negated(), self.bound(e)

    def interval(self, k=0):
        """The certified outer interval of the k-th property: the lower end of the inf's Bound and the upper end of the
        sup's (plan 067 P3, review A7; equal to the values to TOL where both are EXACT). bounds(k) gives the Bounds with
        the solver's values (a witness end in a BRACKET)."""
        lo, hi = self.bounds(k)
        return float(lo.lower), float(hi.upper)

    def intervals(self, brackets=False):
        """(lower, upper, status_lower, status_upper, claims) of every property at once: vectorised over the
        directions in the eigenbasis of a data-space decomposition (optimisers/dual/spectral.py; plan 009 G4), each bound
        certified as by bounds(k); otherwise bounds(k) property by property. brackets=True appends a dict:
        lower_bracket, upper_bracket ((p, 2): each end's certified bracket, widened where the general rule made it
        BRACKET) and error_lower, error_upper (E of the rule, nan where it did not judge; optimisers/dual/exactness.py)."""
        from ..core.claims import Claims
        p = self.space.dim
        sp = self.K._spectral()
        if sp is None or not self.space.is_standard:
            pairs = [self.bounds(k) for k in range(p)]
            out = (np.array([lo.value for lo, _ in pairs]), np.array([hi.value for _, hi in pairs]),
                   np.array([lo.status for lo, _ in pairs], object), np.array([hi.status for _, hi in pairs], object),
                   Claims([Claim("bounds direction by direction", DERIVED, "no data-space decomposition applies")]))
            if not brackets:
                return out
            E = np.array([[np.nan if b.error is None else b.error for b in pr] for pr in pairs]).reshape(p, 2)
            return out + (dict(lower_bracket=np.array([[lo.lower, lo.upper] for lo, _ in pairs]).reshape(p, 2),
                               upper_bracket=np.array([[hi.lower, hi.upper] for _, hi in pairs]).reshape(p, 2),
                               error_lower=E[:, 0], error_upper=E[:, 1]),)
        from ..solvers.dataspace.factored import interval_claims, intervals
        dec = sp.dec
        tg = dec.targets(self.T)
        r, m0 = self.K._prior()
        Tm0 = np.zeros(p) if m0 is None else self.T(m0).values
        how = self._estimability()
        lo, hi, sl, su, redo, br = intervals(sp, dec._beta(tg), tg, Tm0, how)
        cl = interval_claims(dec, p, redo)
        if not math.isfinite(r):                                       # no prior: which estimability applied
            cl.append(Claim(_how.note(how), _how.status(how), "core/estimable.py"))
        return (lo, hi, sl, su, Claims(cl)) + ((br,) if brackets else ())

    def is_empty(self):
        return self.K.is_empty()


def ball_shape(N, D):
    """(base, centre, radius) for a noise set that is a translated NormBall or Ellipsoid of D, else None."""
    from .algebra import Translate
    from .sets import Ellipsoid
    centre = np.zeros(D.dim)
    while isinstance(N, Translate):
        centre = centre + N.shift.values
        N = N.base
    if isinstance(N, NormBall) and identical(N.space, D):
        return N, centre + (0.0 if N.centre is None else N.centre.values), N.radius
    if isinstance(N, Ellipsoid) and identical(N.space, D):
        return N, centre + N.centre_values, N.radius
    return None


def _stated_zeros(K, q, n_dirs, how=None):
    """How many of the q directions of a joint Gram matrix of n_dirs targets and q - n_dirs data representers are
    stated exactly zero, for the general rule's lifted re-solve (optimisers/dual/exactness_reduce.py): q - (a bound on its rank).
    The rank is at most dim M (a finite-dimensional model space: structure), and at most the data's rank plus n_dirs,
    or plus 0 where the targets are declared or built estimable (core/estimable.py: in the data's span); the data's
    rank is n less their floor directions where they are declared exactly dependent (K.dependent), else n."""
    n = q - n_dirs
    rank = n - (K.dependency()[1] if K.dependent else 0) + (0 if how else n_dirs)
    return max(q - rank, reduction.structural_zeros(q, K.G), 0)


def _cut_note(info):
    """The rounding-floor cuts of the reduction and of the data map, when they dropped anything (plan 054 P2)."""
    out = []
    if info.get("dropped"):
        out.append(f"span reduction: {info['dropped']} of {info['size']} directions at the rounding floor dropped "
                   "(correlation eigenvalues <= eps q max: dependent representers, or no correct digit)")
    if info.get("data_dropped"):
        out.append(f"{info['data_dropped']} direction(s) of the data map at the rounding floor dropped (singular values "
                   "<= eps max(n, k) s_max): the datum along them is not imposed (exact if the data are exactly "
                   "dependent there)")
    if info.get("declared"):
        out.append("the data are declared exactly dependent at the rounding floor (K.dependent): the value is exact "
                   "for that declaration")
    return "; ".join(out)


def _noise_support(N, v, kind):
    """The upper end of sigma_N(v) for the dual certificate. For the ball and exact forms the multiplier is built in the
    annihilator of the noise set's subspaces (reduction.whiten: lambda = Phi^T Z l with Z an orthonormal basis of
    (Phi F)^perp, to that QR's stated floor), so a component of v in them at the rounding level is the rounding of that
    construction, not a direction of the noise: their support is 0 there, the lower end of the subspace Enclosure
    [0, +inf] (plan 071 P1: a subspace's support no longer reads a tolerance as orthogonality; the knowledge is the
    construction's). A component above the rounding level (+inf at both ends) stays +inf; other forms take the upper end."""
    s = N.support(v)
    if kind in ("ball", "exact") and not math.isfinite(s.upper) and math.isfinite(s.lower):
        return float(s.lower)
    return float(s.upper)


def solve(K, a, B, const, r, m0, info=None, wdir=None, cases=None, how=None):
    """The Bound of max const + a.y over the reduced feasible set of K (module docstring of optimisers/dual/reduction.py). info: the
    reduction's count of dropped directions (reduce), stated in the Bound's note. wdir: the direction's weights on
    the reduction's direction columns (a = Yd @ wdir), for the general rule (optimisers/dual/exactness_reduce.py; cases: its
    perturbed reductions, or a callable giving them); None: the rule is not applied (a re-solve, an emptiness test).
    how: the targets' estimability (core/estimable.py), for the lifted re-solve's stated zeros and the no-prior note;
    UNBOUNDED for a target declared or built estimable is refused (the numbers contradict it)."""
    info = {} if info is None else dict(info)
    D, N = K.G.codomain, K.noise
    e = K.d.values - (0.0 if m0 is None else K.G(m0).values)

    def phi(lam, a=a, const=const):                             # the general dual (second route)
        v = a - B.T @ D.apply_metric(lam)
        nv = float(np.linalg.norm(v))
        if math.isfinite(r):
            sp = r * nv
        else:
            sp = 0.0 if nv <= 1e-10 * max(np.linalg.norm(a), np.linalg.norm(B), 1e-300) else math.inf
        return const + float(lam @ D.apply_metric(e)) + sp + _noise_support(N, D(-lam), kind)   # an upper bound

    kind, form = forms.noise_form(N, D)
    lam = ell = None
    extra = 0.0
    if kind in ("ball", "exact"):
        dt, Bt, Mw = reduction.whiten(form, e, B)
        if kind == "exact" or not math.isfinite(r):          # the floor decides these (reduction.floor_refusal)
            unexplained, low, _, n = K.dependency()
            info["data_dependent"], info["data_rank"] = unexplained, n - low
        if kind == "exact":                                 # each equation to unit length: a datum's units
            rn = np.linalg.norm(Bt, axis=1)                  # must not decide ball_exact's rank (plan 037)
            rw = 1.0 / np.where(rn > 0, rn, 1.0)
            val, y, lt, st = reduction.ball_exact(a, rw[:, None] * Bt, rw * dt, r, info, K.dependent)
            lt = None if lt is None else rw * lt
        elif math.isfinite(r):
            val, y, lt, st = reduction.ball_ball(a, Bt, dt, r, form["rho"])
        else:
            val, y, lt, st = reduction.none_ball(a, Bt, dt, form["rho"], info, K.dependent)
        lam = None if lt is None else D.solve_metric(Mw.T @ lt)
        if lt is not None:                                  # the general rule: data weights, the datum's rounding
            ell = Mw.T @ lt
            extra = reduction.EPS * (len(dt) * float(np.abs(dt) @ np.abs(lt)) + form["rho"] * float(np.linalg.norm(lt)))
    elif kind == "slab":
        n_, c_ = form["normal"], form["centre"]
        s, beta = float(n_ @ D.apply_metric(e - c_)), B.T @ D.apply_metric(n_)
        if not math.isfinite(r):
            raise Unsupported("a slab without a prior bounds only the estimator's own direction (use oCSOLA)")
        val, y, st = reduction.ball_slab(a, beta, s, form["w"], r)
        if st != "EMPTY":
            tau = reduction.slab_dual(a, beta, s, form["w"], r)[1]
            lam, ell = tau * n_, tau * D.apply_metric(n_)
            extra = reduction.EPS * abs(tau) * (abs(s) + form["w"])
    else:
        cons, starts = _general_parts(N, D, e, B, a, r)
        lower, y, lam, upper = reduction.general(a, B, r, cons, phi, starts)
        lower = const + lower                                # the witness's value: <c, m0> + a . y (phi has const)
        st = "EXACT" if upper - lower <= TOL * max(1.0, abs(upper)) else ("BRACKET" if math.isfinite(lower)
                                                                             else "OUTER")
        return Bound(upper if st == "OUTER" else lower, lower, upper, st, lam, y,
                          "; ".join(filter(None, ["general noise set: SLSQP witness, numerical dual", _cut_note(info)])), claim=CERTIFIED)
    if st == "EMPTY":
        z = phi(lam, a=np.zeros_like(a), const=0.0) if lam is not None else math.nan
        return Bound(-math.inf, -math.inf, -math.inf, "EMPTY", lam, y,
                          f"K(d) is empty: phi(lambda, 0) = {z:.3g} < 0 (Farkas certificate)" if z < 0 else
                          "K(d) is empty", claim=CERTIFIED)
    if st == "UNBOUNDED":
        free = kind in ("ball", "exact") and form["F"].shape[1] > 0      # free nuisances: ran G* is not enough
        if how and wdir is not None and not free:
            raise exactness.contradicted(how, *_outside(a, B))
        why = f"{_how.note(how)}, but not in the span of the data combinations the free nuisances leave" if how \
            else _how.note(None)
        return Bound(math.inf, -math.inf, math.inf, "UNBOUNDED", None, None,
                          "; ".join(filter(None, ["the direction is not estimable and nothing bounds it (inf:122)",
                                                  _cut_note(info), why])), claim=CERTIFIED)
    value = const + val
    feasible = y is not None and (not math.isfinite(r) or float(y @ y) <= r * r * (1 + 1e-9))
    if feasible and kind != "exact":
        feasible = N.member(D(e - B @ y)) is not Truth.FALSE
    lower = value if feasible else -math.inf
    upper = phi(lam) if lam is not None else math.inf
    gap_ok = upper - lower <= TOL * max(1.0, abs(value))
    status = "EXACT" if gap_ok else ("BRACKET" if math.isfinite(lower) and math.isfinite(upper) else "PRIMAL")
    note, E = _cut_note(info), None
    if wdir is not None and info.get("reduction") is not None:           # the general rule (optimisers/dual/exactness_reduce.py)
        xr = exactness_reduce
        red = info["reduction"]
        z = None if ell is None else np.r_[np.asarray(wdir, float), -ell]
        E1 = xr.first_order(red, z, y, r, exactness.C * extra)
        if cases is None:
            def cases():
                return xr.perturbed(red, _stated_zeros(K, len(red[1]), red[5], how))
        E = xr.judge(value, E1, lambda Yd, B2, i2: solve(K, Yd @ wdir, B2, const, r, m0, i2, how=how).value, cases,
                     not red[4].all())
        lower, upper, status, note = xr.apply(value, lower, upper, status, note, E, not red[4].all(), red)
    if wdir is not None and not math.isfinite(r):                        # no prior: which estimability applied
        note = "; ".join(filter(None, [note, _how.note(how)]))
    return Bound(value, lower, upper, status, lam, y, note, E, claim=CERTIFIED)


def _outside(a, B):
    """(|a_perp|^2, the threshold) of the direction a against the row space of the reduced data map B, as
    reduction.none_ball decided UNBOUNDED (|a_perp| > 1e-12 |a|, after the reduction dropped the floor directions), for the
    refusal of a target declared or built estimable (exactness.contradicted)."""
    _, _, V, _, rank = reduction.data_svd(B)
    perp = a - V[:, :rank] @ (V[:, :rank].T @ a)
    return float(perp @ perp), float((1e-12 * np.linalg.norm(a)) ** 2)


def _general_parts(N, D, e, B, a, r):
    """SLSQP constraints for the primal witness and starting certificates for the dual, for boxes, polytopes and
    products of balls and boxes (stack(..., rule='all')); (None, ()) for other sets."""
    from .algebra import CartesianProduct
    from .sets import CoordinateBox, Polytope

    def linear(A, b, rows=slice(None)):
        fin = np.isfinite(b)
        return [dict(type="ineq", fun=lambda y, A=A[fin], b=b[fin], rows=rows: b - A @ (e[rows] - B[rows] @ y))]

    def box(S, rows=slice(None)):
        n = S.space.dim
        return linear(np.r_[np.eye(n), -np.eye(n)], np.r_[S.hi, -S.lo], rows)
    if isinstance(N, CoordinateBox):
        return box(N), ()
    if isinstance(N, Polytope):
        return linear(N.A, N.b), ()
    if not (isinstance(N, CartesianProduct) and N.blocks is not None):
        return None, ()
    cons, starts = [], []
    for S, (lo, hi) in zip(N.parts, N.blocks):
        rows = slice(lo, hi)
        kind, form = forms.noise_form(S, S.space)
        if kind == "ball":
            dt, Bt, Mw = reduction.whiten(form, e[rows], B[rows])
            cons.append(dict(type="ineq", fun=lambda y, dt=dt, Bt=Bt, rho=form["rho"]: rho ** 2 - np.sum((dt - Bt @ y) ** 2)))
            if math.isfinite(r):
                val, yk, lt, st = reduction.ball_ball(a, Bt, dt, r, form["rho"])
                if lt is not None and st == "EXACT":                 # the block alone: already a certificate
                    lam = np.zeros(D.dim)
                    lam[rows] = S.space.solve_metric(Mw.T @ lt)
                    starts.append(lam)
        elif isinstance(S, CoordinateBox):
            cons += box(S, rows)
        else:
            return None, starts
    return cons, starts
