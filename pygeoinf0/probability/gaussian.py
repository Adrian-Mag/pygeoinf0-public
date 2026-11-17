"""Gaussian measures on Hilbert spaces.

    mu = GaussianMeasure(space, covariance=C, mean=m)    N(m, C): C self-adjoint, positive semidefinite, trace class; a
                                                         ProbabilityMeasure (probability/measures.py, plan 067 P5)
    Dirac(x)                                             the point mass delta_x on x's space: the degenerate Gaussian
                                                         N(x, 0), named (the cells build it); a zero covariance spelled
                                                         out stays a valid Dirac
    GaussianMeasure.cylindrical(H, alpha)                N(0, I/alpha) on a Hilbert space: in infinite dimension a
                                                         CylindricalGaussian (decision Q11), its own class and not a
                                                         measure on H: no samples in H, exact variances of bounded
                                                         functionals; on a finite-dimensional space a GaussianMeasure
                                                         (probability.cylindrical(H, alpha); CylindricalGaussian(H, alpha))
    D.standard_gaussian()                                the standard Gaussian of a Euclidean space (covariance I)
    probability.legendre_decay(M, decay), random_features(M, l, F, seed)
                                                         named priors on a function space (probability/priors.py; plan
                                                         069 P6: no space picks one, X.gaussian() refuses)
    mu.sample(size, seed)                                exact draws
    G(mu), G @ mu                                        the push-forward N(G m, G C G*)
    mu + nu, mu + x                                      independent sum, shift
    mu.condition(G, d, noise)                            the posterior given d = G x + e, e ~ noise (default: the standard
                                                         Gaussian of G's codomain)

Covariances are operators on the space, so a measure on a weighted R^n is defined through the space's inner product:
E <x, u> <x, v> = <C u, v>. On D = Euclidean(n, covariance=C_d) the standard Gaussian has coordinates N(0, C_d): the noise
of a data space whose metric is the inverse noise covariance is that space's standard Gaussian.

Sampling: standard Gaussians through the metric's factor, finite-rank measures through their generators, push-forwards
by applying the operator to samples, sums by adding samples, posteriors by Matheron's rule (x + C G* S^-1 (d - G x - e),
with x, e drawn from prior and noise); none of these factorises a covariance. A measure on R^n with no such route (for
example the push-forward of a cylindrical measure) is sampled from a factor L of its dense coordinate covariance S,
L L^T = S to rounding (coordinate_factor: Cholesky, or a clipped eigendecomposition where S is singular to rounding; no
jitter, plan 054 P3; n <= 4000, beyond that Unsupported). Cylindrical measures have no samples.
"""
import numpy as np

from ..core.claims import ASSERTED, DERIVED, THEOREM, UNKNOWN, Claim, Claims, derived
from ..core.kinds import MEASURE
from ..spaces.elements import Family, Function, Vector, VectorFamily
from .measures import ProbabilityMeasure
from ..spaces.identity import require, require_realised

FAMILY_BLOCK = 256   # members per block when the variances of a family are computed (diagonal of C(F_b), F_b)


def _zero(space):
    return space.zero


def _representer_of(u):
    """A functional given by its Riesz representer stands for that representer (spaces/dual.Functional)."""
    from ..spaces.dual import Functional
    return u.representer if isinstance(u, Functional) and u.representer is not None else u


def coordinate_factor(S, name="the covariance"):
    """L with L L^H = S to rounding for a positive semidefinite S (real or Hermitian), with nothing added (plan 054 P3).

    Cholesky where it succeeds (backward stable: L L^H = S + dS with |dS| about n eps |L| |L^H|). Where S is singular to
    rounding, S = V diag(lam) V^H and L = V diag(sqrt(lam)), the eigenvalues in [-eps n max|lam|, 0) (rounding: no
    correct digit, the floor of solvers.direct) set to 0. An eigenvalue below that floor means S is not positive
    semidefinite: refused. The former 1e-14 tr(S)/n jitter drew from S + 1e-14 tr(S)/n I, a covariance error of 10 to 50
    times the rounding along the smallest eigenvectors (measured, plan 054 P3)."""
    S = np.asarray(S)
    S = 0.5 * (S + S.conj().T)
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        pass
    lam, V = np.linalg.eigh(S)
    top = float(np.abs(lam).max(initial=0.0))
    floor = np.finfo(float).eps * len(lam) * top
    if lam.size and lam.min() < -floor:
        from ..core.errors import Refused
        raise Refused(f"{name}: the coordinate covariance has an eigenvalue {lam.min():.3g}, below the rounding floor "
                      f"-eps n max|lam| = {-floor:.3g}: it is not positive semidefinite, so it is the covariance of no "
                      "measure and has no samples")
    return V * np.sqrt(np.maximum(lam, 0.0))[None, :]


def premise(mu, which):
    """(status, note) of the statement that mu's covariance is self-adjoint ("sa"), positive semidefinite ("psd") or
    positive definite ("pd"), as a premise of a construction (plan 065 A5): the covariance's own record when it holds,
    else the measure's statement (made when the measure was built, named in the note), else UNKNOWN."""
    rec = mu.covariance._sa if which == "sa" else mu.covariance._pos
    if rec and rec[0] and (which != "pd" or rec[0] == "pd"):
        return rec[1], ""
    if which == "pd":
        return UNKNOWN, ""
    statement = "the covariance is self-adjoint" if which == "sa" else "the covariance is positive semidefinite"
    for c in getattr(mu, "claims", ()):
        if c.statement == statement and c.holds:
            return c.status, f"; premise: {statement} ({mu.name}, {c.status}: {c.reason})"
    return UNKNOWN, ""


def _finite_dimensional(space):
    """Whether the space is finite-dimensional here: a Euclidean space, or a direct sum of such spaces."""
    from ..spaces.base import Euclidean
    if isinstance(space, Euclidean):
        return True
    summands = getattr(space, "summands", None)
    return bool(summands) and all(_finite_dimensional(X) for X in summands)


def _isonormal(space, alpha):
    """N(0, I/alpha): its covariance, name and the claim naming it (alpha positive and finite, 1/alpha finite)."""
    from ..operators.linear import Identity
    alpha = float(alpha)
    if not (0 < alpha < np.inf and np.isfinite(1.0 / alpha)):
        raise ValueError("alpha (the inverse prior variance scale) must be positive and finite, with 1/alpha finite")
    C = Identity(space) if alpha == 1 else (1.0 / alpha) * Identity(space)
    name = "γ" if alpha == 1 else f"γ/{alpha:g}"
    named = Claim(f"the isonormal Gaussian N(0, I/{alpha:g}) with Cameron-Martin space {space!r}", THEOREM,
                  "variances of bounded functionals are their squared norms over alpha")
    return C, name, named


def _law(cylindrical):
    """The class of a Gaussian law built from others: cylindrical when one of its inputs is (and nothing makes it a
    measure on the space, as a push-forward to R^n does)."""
    return CylindricalGaussian if cylindrical else GaussianMeasure


class _Gaussian:
    """The Gaussian law N(mean, covariance) on a Hilbert space: what GaussianMeasure and CylindricalGaussian share (its
    variances, push-forwards, sums and conditioning)."""

    mean_error = None   # a posterior's: the stated error of its mean in the space's norm (posterior_covariance.py)
    variance_error = None   # the stated errors of the last variances from stated input errors (input_error.py)
    _input = None       # a posterior's inherited input error (probability/input_error.py)
    _cnorm = None       # a bound of ||covariance||, kept once formed (probability/input_error.cov_norm)

    def __init__(self, space, covariance=None, mean=None, name="μ", _sampler=None, _claims=()):
        from ..spaces.base import Euclidean
        from ..operators.linear import Identity
        self.space, self.set, self.name = space, space, name
        if covariance is None:
            if not isinstance(space, Euclidean):
                raise ValueError(f"{space} is infinite-dimensional: it has no standard Gaussian measure (the identity is "
                                 "not trace class, white noise is not an element). Give a covariance operator, or name "
                                 "a prior: pygeoinf0.probability.legendre_decay, random_features, cylindrical "
                                 "(probability/priors.py).")
            covariance = Identity(space)
        if covariance.domain != space or covariance.codomain != space:
            raise ValueError(f"a covariance on {space} must map {space} to itself")
        require_realised(covariance.domain, space, covariance.name)
        require_realised(covariance.codomain, space, covariance.name)
        self.covariance = covariance
        self.mean = _zero(space) if mean is None else mean
        self.zero_mean = mean is None                # mean exactly zero (kept out of conditioning: no formula part)
        self._sampler = _sampler
        sa, pos = covariance.is_selfadjoint(), covariance.positivity()
        if sa[0] is False:
            from ..core.errors import NotACovariance
            raise NotACovariance(f"{covariance.name} is not self-adjoint ({sa[1]}: {sa[2]}): it is not the covariance of a "
                          "Gaussian measure", needs=("a self-adjoint covariance, e.g. (C + C*) / 2 if that is what you "
                                                     "mean",))
        cl = list(_claims)
        assumed = "not verified: assumed by building a Gaussian measure with this covariance"
        cl.append(Claim("the covariance is self-adjoint", sa[1] if sa[0] else ASSERTED, sa[2] if sa[0] else assumed))
        cl.append(Claim("the covariance is positive semidefinite", pos[1] if pos[0] else ASSERTED,
                        pos[2] if pos[0] else assumed))
        if isinstance(space, Euclidean):
            cl.append(Claim("the covariance is trace class", THEOREM, "finite dimension"))
        elif not any("trace class" in c.statement for c in cl):
            tc = [c for c in covariance.claims if "trace class" in c.statement]
            cl.append(tc[0] if tc else Claim("the covariance is trace class", ASSERTED, "not verified"))
        if getattr(covariance, "rest", None) is not None:               # a pg.Covariance applied as its known part
            cl += [c for c in covariance.claims if c.key == "truncation"]
        self.claims = Claims(cl)

    @property
    def is_cylindrical(self):
        """isinstance(self, CylindricalGaussian): no samples in the space (decision Q11)."""
        return isinstance(self, CylindricalGaussian)

    @classmethod
    def cylindrical(cls, space, alpha=1.0, mean=None):
        """N(mean, I/alpha) (mean 0 by default) on a Hilbert space H: the Gaussian whose Cameron-Martin space is H with
        the norm scaled by sqrt(alpha) (the isonormal Gaussian for alpha = 1). On an infinite-dimensional space it is a
        CylindricalGaussian, not a measure on H (the identity is not trace class): no samples in H, but the variance of
        every bounded functional is exact, Var <x, u> = ||u||^2 / alpha, its push-forwards to R^n are Gaussian measures
        with samples, and conditioning on finitely many data is exact. On a finite-dimensional space (a Euclidean space
        or a direct sum of them) the identity is trace class and N(0, I/alpha) is a GaussianMeasure (plan 067 P5)."""
        if _finite_dimensional(space):
            C, name, named = _isonormal(space, alpha)
            from ..spaces.base import Euclidean
            trace = [] if isinstance(space, Euclidean) else [Claim(    # (a Euclidean space states it when built)
                "the covariance is trace class", THEOREM, "finite dimension: N(0, I/alpha) is a Gaussian measure")]
            return GaussianMeasure(space, C, mean, name=name, _claims=[named] + trace)
        return CylindricalGaussian(space, alpha, mean)

    @classmethod
    def product(cls, measures, space=None):
        """The product of independent Gaussian measures mu_k on X_k: a Gaussian measure on the direct sum X (given, or
        pg.DirectSum of the spaces of the measures; measures a list, or a dict by summand name). Its mean is the tuple of
        means and its covariance the block-diagonal operator (w_k C_k)_k, w_k the weights of X, since
        E <x, u>_X <x, v>_X = sum_k w_k^2 <C_k u_k, v_k>_k for independent parts. It is cylindrical when one factor is."""
        from ..spaces.direct_sum import DirectSum
        from ..operators.blocks import BlockDiagonal
        names = list(measures) if isinstance(measures, dict) else None
        mus = list(measures.values()) if names else list(measures)
        X = space if space is not None else DirectSum(dict(zip(names, [m.space for m in mus])) if names
                                                       else [m.space for m in mus])
        if [X.summand(n) for n in (names or X.names)] != [m.space for m in mus]:
            raise ValueError(f"the measures live on {[m.space for m in mus]}, not on the summands of {X}")
        order = [X.index(n) for n in names] if names else list(range(len(mus)))
        blocks = [None] * len(X.summands)
        for k, m in zip(order, mus):
            blocks[k] = m.covariance if X.weights[k] == 1 else X.weights[k] * m.covariance
        mean = None if all(m.zero_mean for m in mus) else X._from_parts(dict(zip(order, [m.mean for m in mus])),
                                                                           "mean", family=False)

        def sampler(N, rng):
            return X._from_parts({k: m._centred(N, rng) for k, m in zip(order, mus)}, "sample", family=N > 1)
        cyl = any(m.is_cylindrical for m in mus)
        tc = [c for m in mus for c in m.claims if "trace class" in c.statement and "not" not in c.statement]
        trace = (Claim("the covariance is not trace class: a cylindrical factor", THEOREM, "see that factor") if cyl else
                 Claim("the covariance is trace class", DERIVED, "a finite sum of trace-class blocks")
                 if len(tc) == len(mus) and all(c.status in (THEOREM, DERIVED) for c in tc) else
                 Claim("the covariance is trace class", ASSERTED, "not verified for every factor"))
        return _law(cyl)(X, BlockDiagonal(X, blocks, name="C"), mean, name=" x ".join(m.name for m in mus),
                         _sampler=None if cyl else sampler,
                   _claims=[Claim("a product of independent Gaussian measures", DERIVED,
                                  "the covariance is block diagonal: (w_k C_k)_k on the direct sum"), trace])

    # sampling ---------------------------------------------------------------------------------------
    def sample(self, size=None, seed=None):
        """One draw (an element) or `size` draws (a Family / VectorFamily)."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        X = self._centred(1 if size is None else int(size), rng)
        out = X + self.mean if size is None else _shift(X, self.mean)
        if size is None and isinstance(out, (Function, Vector)):
            out.name = f"sample of {self.name}"
        return out

    def _centred(self, N, rng):
        """N draws from N(0, covariance), as a family (or one element when N == 1 and the route returns one)."""
        from ..operators.linear import FiniteRank, Identity
        if self._sampler is not None:
            return self._sampler(N, rng)
        C = getattr(self.covariance, "realised", self.covariance)     # a pg.Covariance: its realised (known) part
        from ..spaces.base import Euclidean
        if isinstance(C, Identity) and isinstance(self.space, Euclidean):   # the standard Gaussian of R^n
            Z = rng.standard_normal((self.space.dim, N))
            X = self.space.from_orthonormal(Z)
            return Vector(self.space, X[:, 0]) if N == 1 else VectorFamily(self.space, X.T)
        if isinstance(C, FiniteRank):
            from ..operators.linear import synthesise
            A = np.sqrt(np.maximum(C.lam, 0.0))[:, None] * rng.standard_normal((C.rank, N))
            return synthesise(C.generators, A, single=N == 1)
        if isinstance(self.space, Euclidean) and self.space.dim <= 4000:
            # coordinates have covariance S = C_mat W^-1 (E <x,u> <x,v> = <C u, v>); a dense factor
            S = self.space.solve_metric(C.matrix().T).T
            L = coordinate_factor(S, self.name)
            X = L @ rng.standard_normal((self.space.dim, N))
            return Vector(self.space, X[:, 0]) if N == 1 else VectorFamily(self.space, X.T)
        from ..core.errors import NotDefined, Unsupported
        if self.is_cylindrical:
            raise NotDefined(f"{self.name} is the cylindrical Gaussian of {self.space} (its Cameron-Martin space): it is not "
                              "a measure on that space, whose samples would have infinite norm, so draws are not "
                              "available; variances of bounded functionals and conditioning are exact")
        raise Unsupported(f"no sampler for a covariance of type {type(C).__name__} on {self.space}")

    # moments -----------------------------------------------------------------------------------------
    def variance(self, u, exact=False):
        """Var <x, u> = <C u, u> for an element u (or the linear form <u, .>). A family gives the array of its members'
        variances; on a function space with riesz, a point, an (m, d) array of points, a set or a measure stands for the
        functional it integrates (Var u(x0), Var int x dmu), and a list or array of them gives an array. A posterior
        above settings.dense_max computes them from a randomised decomposition with a bound per functional, and exactly
        where the bound exceeds settings.accuracy (or everywhere with exact=True); self.variance_claims says how
        (probability/scale.py)."""
        u = _representer_of(u)
        if isinstance(u, (np.ndarray, list, tuple, Family, VectorFamily)) and len(u) == 0:
            return np.zeros(0)
        if not isinstance(u, (Function, Vector, Family, VectorFamily)) and hasattr(self.space, "riesz"):
            u = self.space.riesz(u)
        scale = getattr(self, "_scale", None)
        if scale is not None:
            single = not isinstance(u, (Family, VectorFamily))
            fam = self.space.family([u]) if single and not isinstance(u, Vector) else \
                VectorFamily(self.space, u.values[None, :]) if single else u
            var, _ = scale.variance(fam, exact)
            self.variance_claims = scale.last
            from .input_error import variance_errors                 # stated input errors at scale (plan 054)
            e = variance_errors(self, fam, fam[0] if len(fam) else fam, var, scale.last_yn, refuse=False)
            self.variance_error = None if e is None else (float(e[0]) if single else e)
            return float(var[0]) if single else var
        from .input_error import variance_errors
        C = self.covariance
        if isinstance(u, (Family, VectorFamily)):
            out, errs = [], []
            for b in (u[i:i + FAMILY_BLOCK] for i in range(0, len(u), FAMILY_BLOCK)):
                Cb = C(b)
                out.append(np.diag(np.atleast_2d(self.space.inner(Cb, b))))
                errs.append(variance_errors(self, b, Cb, out[-1]))     # stated input errors (plan 054)
            self.variance_error = None if all(e is None for e in errs) else \
                np.concatenate([np.zeros(len(v)) if e is None else e for v, e in zip(out, errs)])
            return np.concatenate(out)
        Cu = C(u)
        v = float(self.space.inner(Cu, u))
        e = variance_errors(self, u, Cu, v)
        self.variance_error = None if e is None else float(e[0])
        return v

    def integrate(self, u):
        """(E <x, u>, its stated error) = (<mean, u>, ||u|| mean_error) for a bounded linear functional, given by its
        representer u (an element of the space; a pg.Functional by its representer). Other integrands are refused: the integral
        of a nonlinear function under a Gaussian is not computed here (mu.sample(n) and an average is a Monte Carlo
        estimate, with its own error)."""
        u = _representer_of(u)
        if not isinstance(u, (Function, Vector)):
            from ..core.errors import Unsupported
            raise Unsupported(f"integrating {type(u).__name__} against {self.name}: only bounded linear functionals "
                              "<x, u> are integrated exactly (their mean <m, u>); a nonlinear integrand needs a Monte "
                              "Carlo estimate (mu.sample(n) and an average, with its own error), a further choice")
        value = 0.0 if self.zero_mean else float(self.space.inner(self.mean, u))
        err = 0.0 if not self.mean_error else float(self.mean_error) * float(self.space.norm(u))
        return value, err

    def std(self, u, exact=False):
        """The standard deviation: the square root of variance(u) (an array when variance(u) is one)."""
        v = self.variance(u, exact)
        return np.sqrt(np.maximum(v, 0.0)) if np.ndim(v) else float(np.sqrt(max(v, 0.0)))

    # algebra ----------------------------------------------------------------------------------------
    def pushforward(self, A):
        """The law of A x for x ~ this measure: N(A m, A C A*)."""
        from ..operators.linear import FiniteRank
        if A.domain != self.space:
            raise ValueError(f"{A.name} acts on {A.domain}, not on {self.space}")
        require_realised(A.domain, self.space, A.name)
        C = self.covariance
        if isinstance(C, FiniteRank) or hasattr(C, "congruence"):
            cov = C.pushforward(A)                                    # finite rank, or a pg.Covariance: A C A*
            sampler = None
        else:
            cov = A @ C @ A.adjoint
            (sa, n1), (pos, n2) = premise(self, "sa"), premise(self, "psd")
            cov._sa = derived(True, f"of the form {A.name} C {A.name}*{n1}", sa)
            cov._pos = derived("psd", f"a congruence of a positive semidefinite operator{n2}", pos)
            from ..spaces.base import Euclidean

            def sampler(N, rng, base=self):
                return A(base._centred(N, rng))
            if self.is_cylindrical and isinstance(A.codomain, Euclidean):
                sampler = None                           # no samples in the space: the law on R^n from its matrix
        cyl = self.is_cylindrical and sampler is not None
        mean = None if self.zero_mean else A(self.mean)                # a zero mean stays exactly zero (not integrated)
        from .circular import pushforward_claims
        return _law(cyl)(A.codomain, cov, mean, name=f"{A.name}({self.name})", _sampler=sampler,
                         _claims=[Claim("a push-forward", DERIVED, f"the law of {A.name} x, x ~ {self.name}")]
                         + pushforward_claims(self, A))

    def __add__(self, other):
        if isinstance(other, _Gaussian):                 # independent sum
            if other.space != self.space:
                raise ValueError("measures on different spaces")
            require_realised(other.space, self.space, f"{self.name} + {other.name}")
            a, b = self, other

            def sampler(N, rng):
                return a._centred(N, rng) + b._centred(N, rng)
            return _law(self.is_cylindrical or other.is_cylindrical)(
                self.space, self.covariance + other.covariance, self.mean + other.mean, name=f"{self.name} * {other.name}",
                _sampler=sampler, _claims=[Claim("an independent sum", ASSERTED, "the two measures are independent")])
        return _law(self.is_cylindrical)(self.space, self.covariance, self.mean + other, name=self.name,
                                         _sampler=self._sampler)

    __radd__ = __add__

    def condition(self, G, d, noise=None):
        """The posterior of x given d = G x + e, e ~ noise independent of x (noise=None: on a covariance-weighted
        codomain, the standard Gaussian of it, unit noise in its metric; on a plain codomain, NoiseUnstated, plan 071
        P8: its dot product is not a noise model, and D.standard_gaussian() states unit noise).

            mean       m + C G* S^-1 (d - G m),    S = G C G* + C_e
            covariance C - C G* S^-1 G C
            samples    x + C G* S^-1 (d - G x - e)   (Matheron's rule, x and e drawn from prior and noise)

        The mean is refined with the operators' residual and carries its stated error, .mean_error, in the space's norm
        (an a-posteriori estimate; a PrecisionWarning below 9 digits, a refusal at none: probability/posterior_covariance.py).

        The model-data relation is built explicitly, pg.Additive(G, noise), and the backward cell B10 computes the
        posterior (pygeoinf0.cells, plan 008): the same numbers, with claims that say which relation was used."""
        from ..cells import update
        from ..relations.additive import Additive, require_noise
        require_noise(noise, G.codomain, "mu.condition(G, d)")
        return update(self, Additive(G, G.codomain.standard_gaussian() if noise is None else noise), d, policy="tagged")

    def _condition(self, G, d, noise):
        """The linear Gaussian update itself (condition's formulas): cell B10 calls it."""
        D = G.codomain
        noise = D.standard_gaussian() if noise is None else noise
        C, Ce = self.covariance, noise.covariance
        GC = G(self).covariance
        S = GC + Ce
        Gmu = G(self)
        (s1, n1), (s2, n2) = premise(Gmu, "sa"), premise(noise, "sa")
        S._sa = derived(True, f"G C G* + C_e{n1}{n2}", s1, s2)
        (p1, n1), (p2, n2), (p3, _) = premise(Gmu, "psd"), premise(noise, "psd"), premise(noise, "pd")
        S._pos = derived("pd", "G C G* is positive semidefinite and C_e positive definite", p1, p3) if p3 != UNKNOWN \
            else derived("psd", f"G C G* + C_e, both positive semidefinite{n1}{n2}", p1, p2)
        Sinv = S.inv
        K = C @ G.adjoint                                # the operator C G*: D -> space
        from .posterior_covariance import EPS, PosteriorCovariance, judge_mean, refined_mean
        r = d - noise.mean if self.zero_mean else d - G(self.mean) - noise.mean
        mean, err = refined_mean(self.space, C, K, G, Ce, Sinv, r)      # plan 054: refined, the operators' residual
        mean = mean if self.zero_mean else self.mean + mean
        mean_error, size = None, None
        if err is not None:                              # an inverse to rounding (not CG's stated tolerance)
            size = float(self.space.norm(mean))
            mean_error = err + EPS * size
            judge_mean(f"{self.name} | d", size, mean_error)
        from .input_error import InputError, judge, operator_error
        inp, eG = None, operator_error(G)
        if eG is not None:                               # the kernels' stated errors (plan 054, input_error.py)
            size = float(self.space.norm(mean)) if size is None else size
            inp = InputError(self.space, C, K, G, GC, Sinv, Sinv(r), mean, size, *eG)
            mean_error = (mean_error or EPS * size) + inp.mean
            judge("mean", f"{self.name} | d", size, mean_error, eG[1] + " (in the space's norm)", refuse=False)
        cov = PosteriorCovariance(C, G, Ce, Sinv, prior=self, noise=noise, S=S)   # Joseph's form (plan 054), claims
        prior = self
        from .circular import posterior_claims

        def sampler(N, rng):
            # Matheron: x + K S^-1 (d - G x - e) has the posterior law; centred, it is x0 - K S^-1 (G x0 + e0)
            X0, E0 = prior._centred(N, rng), noise._centred(N, rng)
            return X0 - K(Sinv(G(X0) + E0))

        post = _law(self.is_cylindrical)(
            self.space, cov, mean, name=f"{self.name} | d", _sampler=sampler,
            _claims=[Claim("the posterior of a linear Gaussian model", DERIVED, "Bayes' rule for Gaussians"),
                     Claim("noise independent of the model", ASSERTED, "a modelling assumption")]
            + posterior_claims(self, G, noise))
        post.solver = Sinv
        post.mean_error = mean_error                    # in the space's norm: probability/posterior_covariance.py
        post._input = inp
        from .scale import AtScale, applies
        if applies(G, noise, self):                      # plan 005 P5: variances and samples at scale
            post._scale = AtScale(self, G, noise)
            if not self.is_cylindrical:
                post._sampler = post._scale.sampler
        return post

    def harden(self, form, mass, kind="norm", law=None, method="auto", name=None):
        """The set {m : ||B (m - mean)||^2 <= t} (kind="quadratic": <A (m - mean), m - mean> <= t) with mass >= mass
        under this measure, t from the weighted chi-square law of the form with a proved bracket (plan 009 G7;
        probability/hardening.py). The Cameron-Martin ball is refused (mass 0 in infinite dimension)."""
        from .hardening import harden
        return harden(self, form, mass, kind=kind, law=law, method=method, name=name)

    def __repr__(self):
        return f"{self.name} = N(mean, {self.covariance.name}) on {self.space}"


def _shift(X, m):
    """Add the element m to every member of the family X."""
    if isinstance(X, VectorFamily):
        return VectorFamily(X.space, X.values + m.values[None, :], name=X.name)
    return X + m


class GaussianMeasure(_Gaussian, ProbabilityMeasure):
    """N(mean, covariance) on a Hilbert space, a probability measure on it (see the module docstring).

    The covariance is trace class; the claims carry the keys finite, probability and signed of the Measure root."""

    def __init__(self, space, covariance=None, mean=None, name="μ", _sampler=None, _claims=()):
        super().__init__(space, covariance, mean, name=name, _sampler=_sampler, _claims=_claims)
        self.claims = Claims(self.property_claims() + list(self.claims))   # first: positions of the others kept


class Dirac(GaussianMeasure):
    """The point mass delta_x at an element x of a Hilbert space: the degenerate Gaussian N(x, 0).

    Architecture review B2: the zero-covariance Gaussian is delta_x, and that spelling stays valid. Dirac(x) on x's space; Dirac(space=X)
    is delta_0 (its mean exactly zero). Its samples are x; its push-forward under A is N(A x, 0) = delta_(A x)."""

    def __init__(self, x=None, space=None, name="δ", _claims=()):
        from ..operators.linear import Identity
        if x is None and space is None:
            raise TypeError("Dirac(x) needs the point x (an element of a space), or space= for delta_0")
        X = x.space if space is None else space
        if x is not None and space is not None:
            require(x.space, space, f"the point lies in {x.space!r}, not in {space!r}", getattr(x, "name", None))

        def sampler(N, rng):
            return _zeros(X, N)
        super().__init__(X, 0.0 * Identity(X), x, name=name, _sampler=sampler, _claims=[Claim(
            "a Dirac measure: the degenerate Gaussian N(x, 0)", THEOREM,
            "a Gaussian with zero covariance is the point mass at its mean")] + list(_claims))
        self.point = X.zero if x is None else x


def _zeros(X, N):
    """N zero elements of X: one element for N = 1, else a family."""
    if N == 1:
        return X.zero
    from ..spaces.base import Euclidean
    if isinstance(X, Euclidean):
        return VectorFamily(X, np.zeros((N, X.dim)))
    return X.family([X.zero] * N)


class CylindricalGaussian(_Gaussian):
    """A Gaussian law that speaks about bounded functionals only, not a measure on its space (decision Q11).

    N(mean, C) with a covariance that is not known to be trace class, e.g. N(0, I/alpha) on an infinite-dimensional
    Hilbert space. It is not a Measure on its space
    (no countably additive extension to its Borel sets in general, no samples in it): variances of bounded functionals
    are exact, its push-forwards to R^n are GaussianMeasures with samples, conditioning on finitely many data is exact
    and gives a CylindricalGaussian. The cells check it by this class (cells/forward.py F10, cells/backward.py B8, B10);
    the kind of the state is still a measure (declared here; pygeoinf0.states.state_kind)."""

    kind = MEASURE

    def __init__(self, space, covariance=1.0, mean=None, name=None, _sampler=None, _claims=()):
        """CylindricalGaussian(H, alpha=1.0, mean=None): N(mean, I/alpha) on an infinite-dimensional Hilbert space H
        (plan 069 P6; probability.cylindrical(H, alpha)); refused on a finite-dimensional space, where N(0, I/alpha) is
        a GaussianMeasure. A covariance operator in place of alpha: the law the package builds from others (posteriors,
        sums, push-forwards of cylindrical laws)."""
        if covariance is None or (np.ndim(covariance) == 0 and not hasattr(covariance, "domain")):
            if _finite_dimensional(space):
                from ..core.errors import Refused
                raise Refused(f"{space} is finite-dimensional: N(0, I/alpha) is a Gaussian measure there, not a "
                              "cylindrical one", needs=("X.standard_gaussian()", "pygeoinf0.probability.cylindrical(X, "
                                                        "alpha), a GaussianMeasure in finite dimension"))
            C, nm, named = _isonormal(space, 1.0 if covariance is None else covariance)
            _claims = [Claim(f"the cylindrical Gaussian N(0, I/{float(covariance or 1.0):g}) with Cameron-Martin space "
                             f"{space!r}", THEOREM, "variances of bounded functionals are their squared norms over alpha"),
                       Claim("the covariance is not trace class: a cylindrical measure, not a measure on the space (no "
                             "samples in it)", THEOREM, "the identity of an infinite-dimensional space is not trace "
                             "class; its push-forwards under finitely many functionals are Gaussian measures on R^n")]
            super().__init__(space, C, mean, name=name or nm, _claims=_claims)
            return
        super().__init__(space, covariance, mean, name=name or "μ", _sampler=_sampler, _claims=_claims)

    def sample(self, size=None, seed=None):
        from ..core.errors import NotDefined
        raise NotDefined(f"{self.name} is the cylindrical Gaussian of {self.space} (its Cameron-Martin space): it is not "
                          "a measure on that space, whose samples would have infinite norm, so draws are not available; "
                          "variances of bounded functionals and conditioning are exact")

    def __repr__(self):
        return f"{self.name} = cylindrical N(mean, {self.covariance.name}) on {self.space}"


def noise_metric(noise, D, centre=None):
    """The data-space engine's metric (solvers.dataspace.NoiseMetric) of a Gaussian noise law on D (plan 070 P3,
    decision Q1: the law's own covariance, stated by the law, not inferred by the engine): its covariance in coordinates
    and its mean (plus centre) as the offset."""
    from ..core.errors import SpaceMismatch
    from ..operators.linear import Identity
    from ..solvers.dataspace import NoiseMetric
    from ..solvers.dataspace.whitening import covariance_metric
    if not is_gaussian(noise):
        raise TypeError(f"a Gaussian noise law, not {type(noise).__name__}")
    if noise.space != D:
        raise SpaceMismatch(f"the noise is a measure on {noise.space}, not on {D}")
    off = None if noise.zero_mean else noise.mean.values
    off = off if centre is None else (centre if off is None else centre + off)
    C = noise.covariance
    if isinstance(C, Identity):
        return NoiseMetric("standard", None, D, off, f"{noise.name}, the standard Gaussian of the data space")
    out = covariance_metric(C, D)
    out.offset, out.what = off, f"the covariance of {noise.name}"
    return out


def is_gaussian(obj):
    """Whether obj is a Gaussian law: a GaussianMeasure (Dirac included) or a CylindricalGaussian, by class (plan 067
    P4, D8: never by the attributes covariance or sample)."""
    return isinstance(obj, (GaussianMeasure, CylindricalGaussian))
