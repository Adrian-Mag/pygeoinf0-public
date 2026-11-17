"""Measure-valued relations: observation (Markov) kernels x -> L(x, .) (el:214-221, inv:167-175). Named MarkovKernel
(plan 068 P7, review E2): "kernel" alone also means integral and reproducing kernels and ker.

    MarkovKernel(X, Y, law)               a kernel from a rule (law(x) is a probability measure on Y); measurability
                                          asserted with your reason (measurable="why")
    AdditiveMarkovKernel(G, noise)        x -> G x + noise (the translated noise law, app-add), built by
                                          pg.Additive(G, gaussian); with nuisance=E the kernel lives on Y / E (the
                                          quotient: free nuisance parameters such as origin times, DESIGN 1.7)
    L.converse()                          refused: a kernel has no converse without a prior (inv:177-183)
    L.converse(prior=mu)                  the posterior kernel d -> mu^d (a stated version)
    L.acceptance(alpha, shape)            acceptance regions S_alpha(x) with L(x, S_alpha(x)) = 1 - alpha: a set-valued
                                          relation (confidence-set inversion, inv:203-246); shape "ellipsoid" (the
                                          chi-square region of the whole noise vector) or an estimator X: D -> R (its
                                          slab, inf:400-407)
    L.likelihood(d)                       the ordering of models by the Gaussian likelihood of d (not a density on X)
    L.standardised()                      (G', datum map): the additive kernel read where its noise is the standard
                                          Gaussian (plan 071 P4, Q1 as amended: pg.Additive(G, N(b, C)) on a plain D
                                          is D_C = Euclidean(n, covariance=C) with unit noise, shifted by b)
"""
import math

import numpy as np

from ..core.claims import ASSERTED, DERIVED, THEOREM, Claim, Claims
from ..core.errors import Refused, Unsupported
from ..core.logic import Truth
from ..core.kinds import MEASURE, is_state
from ..core.relation import Relation
from ..operators.claim_rules import status
from ..spaces.identity import require_realised


class MarkovKernel(Relation):
    """x -> law(x), a probability measure on the codomain (module docstring)."""

    kind = MEASURE

    def __init__(self, domain, codomain, law, name="L", measurable=None, claims=()):
        self.domain, self.codomain, self.law, self.name = domain, codomain, law, name
        self._claims = list(claims)
        if measurable:
            self._claims.append(Claim("measurable", ASSERTED, measurable, key="measurable"))

    @property
    def claims(self):
        return Claims(self._claims)

    def __call__(self, x):
        if is_state(x):
            from ..cells import push
            return push(x, self)
        out = self.law(x)
        if isinstance(out, dict):
            raise TypeError(f"the law of {self.name} at {x!r} is a dict: a law on a finite set is "
                            "pg.DiscreteMeasure({point: probability}, on=codomain) (plan 067 P5)")
        return out

    def converse(self, prior=None):
        if prior is None:
            raise Refused(f"the kernel {self.name} has no converse without a prior measure on {self.domain!r}: it gives "
                          "probabilities of data given a model, not of models given data (inv:177-183)",
                          needs=[f"{self.name}.converse(prior=mu) for the posterior kernel",
                                 f"{self.name}.acceptance(alpha) for confidence sets",
                                 f"{self.name}.likelihood(d) to rank models"])
        return PosteriorKernel(self, prior)


class AdditiveMarkovKernel(MarkovKernel):
    """x -> G x + noise, with noise a GaussianMeasure on G's codomain independent of x; with nuisance E (a span in the
    codomain) the kernel x -> law of q(G x + e) on the quotient Y / E (module docstring)."""

    def __init__(self, G, noise, nuisance=None, name=None):
        if noise.space != G.codomain:
            raise ValueError(f"the noise must be a measure on {G.codomain!r}")
        require_realised(noise.space, G.codomain, "the noise")
        self.G, self.noise, self.nuisance, self.q = G, noise, nuisance, None
        Ge, ne = G, noise
        if nuisance is not None:
            from .linear import LinearRelation
            self.q = LinearRelation(G, mul=nuisance).quotient_map()
            Ge, ne = self.q @ G, self.q(noise)
        self.effective = (Ge, ne)                               # the operator and noise law of the kernel's codomain
        cl = [Claim("x -> G x + e, e ~ noise independent of x", ASSERTED, "an additive noise model", key="independent noise"),
              Claim("additive (fibre-constant: depends on x through G x)", THEOREM, "the construction", key="additive")]
        held, st = status(G, "bounded")
        if held is Truth.TRUE:
            cl.append(Claim("measurable", DERIVED, f"G bounded ({st}), translation measurable (app-add)", key="measurable"))
        if nuisance is not None:
            cl.append(Claim(f"free nuisance in a span of dimension {nuisance.dim}: the kernel lives on Y / E", THEOREM,
                            "the canonical quotient kernel; equal to the flat-prior limit (DESIGN 1.7)", key="quotient"))
        super().__init__(G.domain, Ge.codomain, lambda x: ne + Ge(x), name=name or f"{G.name} + {noise.name}", claims=cl)

    def datum(self, d):
        """The observed datum as an element of the kernel's codomain (q(d) on a quotient)."""
        return d if self.q is None else self.q(d)

    def standardised(self):
        """(G', d -> d'): the kernel read in the inner product of its noise (plan 071 P4, Mac review item 4, Q1 as
        amended: both routes are one model). For the noise N(b, C) on D (C an operator on D, its coordinate covariance
        S = C W^-1 checked by the data-space engine's rule: probability.gaussian.noise_metric), D_C is D's vectors with
        the S^-1 inner product, J: D -> D_C the identity on coordinates (operators.whitening.SameVectors), G' = J G and
        d' = J(q(d) - b); in D_C the noise is the standard Gaussian, so -log l(d | m) = (1/2)||G' m - d'||^2 up to a
        constant: route A (a plain D with the noise in the relation) is route B (Euclidean(n, covariance=C)). A noise
        with covariance the identity of D (route B already) gives G' = G."""
        from ..operators.linear import Identity
        G, ne = self.effective
        b = None if ne.zero_mean else ne.mean
        J = None
        if not isinstance(ne.covariance, Identity):
            from ..operators.whitening import SameVectors
            from ..probability.gaussian import noise_metric
            from ..spaces.base import Euclidean
            D = G.codomain
            if not isinstance(D, Euclidean):
                raise Unsupported(f"reading the noise of {self.name} as an inner product needs a Euclidean data space, "
                                  f"not {D}")
            nm = noise_metric(ne, D)                      # NotACovariance for a covariance that is not one
            J = SameVectors(D, D.standard.with_covariance(nm.data), name="J_C")
            G = J @ G

        def datum(d):
            y = self.datum(d)
            y = y if b is None else y - b
            return y if J is None else J(y)
        return G, datum

    def _standard_noise(self):
        G, ne = self.effective
        from ..operators.linear import Identity
        return isinstance(ne.covariance, Identity)

    def acceptance(self, alpha, shape="ellipsoid"):
        """The set-valued relation x -> G x + S_alpha with L(x, G x + S_alpha) = 1 - alpha (module docstring)."""
        from scipy.stats import chi2, norm
        from ..convex.sets import Ellipsoid, NormBall, Slab
        from .additive import AdditiveSet
        if self.q is not None:
            raise Unsupported("acceptance regions of a quotient kernel: build them on the quotient (pg.Additive(q G, ...))")
        D, mu = self.G.codomain, self.noise
        centre = None if mu.zero_mean else mu.mean
        if isinstance(shape, str):
            if shape != "ellipsoid":
                raise ValueError("shape is 'ellipsoid' or an estimator X: D -> R (its slab)")
            r = math.sqrt(chi2.ppf(1.0 - alpha, D.dim))
            if self._standard_noise():
                S = NormBall(D, r, centre=centre, name=f"S_{alpha:g}")
            else:
                C = D.solve_metric(mu.covariance.matrix().T).T            # the coordinate covariance (sampling code)
                S = Ellipsoid(D, r, covariance=C, centre=centre, name=f"S_{alpha:g}")
            why = f"||e||^2 in the noise metric is chi^2 with {D.dim} degrees of freedom"
        else:
            X = shape
            w = norm.ppf(1.0 - alpha / 2.0) * float(np.sqrt(X(mu).covariance.matrix()[0, 0]))
            S = Slab(D, X.adjoint(X.codomain([1.0])), w, centre=centre, name=f"slab_{alpha:g}")
            why = f"X e is N(0, X C X*): |X e| <= z_(1 - alpha/2) sd with probability 1 - alpha"
        return AdditiveSet(self.G, S, name=f"{self.G.name} + {S.name}",
                           claims=[Claim(f"L(x, S_alpha(x)) = {1 - alpha:g}", THEOREM, why, key="acceptance"),
                                   Claim("region chosen before the data", ASSERTED,
                                         "built by acceptance(alpha) from the noise law alone", key="fixed before data")])

    def likelihood(self, d):
        """The ordering of models by -log l(d | x) = (1/2) ||q(d) - G x - mean||^2 in the noise metric (continuous
        Gaussian density version; standard noise of the codomain): not a probability density on X (inv:187-201)."""
        from ..states.base import Ordering
        if not self._standard_noise():
            raise Unsupported("likelihood orderings are implemented for the standard Gaussian noise of the data space")
        G, ne = self.effective
        y = self.datum(d) if ne.zero_mean else self.datum(d) - ne.mean
        Y = G.codomain
        return Ordering(G.domain, cost=lambda x: 0.5 * Y.norm(y - G(x)) ** 2, name=f"-log l({getattr(d, 'name', 'd')} | x)",
                        claims=[Claim("an ordering by likelihood, not a probability on models", THEOREM, "inv:187-201")])


class PosteriorKernel(Relation):
    """d -> mu^d, the converse of a kernel L given a prior mu (inv:516-558): defined almost everywhere; the version is a
    claim (the affine one for linear Gaussian models)."""

    kind = MEASURE

    def __init__(self, L, prior):
        self.L, self.prior = L, prior
        self.domain, self.codomain, self.name = L.codomain, L.domain, f"{L.name}⁻¹[{prior.name}]"
        self._claims = [Claim("the posterior kernel: a version, defined almost everywhere", THEOREM, "inv:537-558")]

    @property
    def claims(self):
        return Claims(self._claims)

    def __call__(self, d):
        from ..cells import update
        return update(self.prior, self.L, d)
