"""The public route to the data-space engine (plan 070 P3, architecture review D4, decisions Q1 and Q7): pg.decompose.

The engine (solvers/dataspace/) takes its metric as an explicit argument. This route reads it from what a caller
states about the noise, each by its owner's rule:

    a model-data relation       its noise law (pg.Additive(G, noise): the law lives in the relation, Q1)
    a Gaussian measure on D     its covariance and mean (probability.gaussian.noise_metric)
    a NormBall or Ellipsoid of D, translated or not
                                its shape and centre (convex.forms.noise_metric)
    a covariance, D.whitened(C) or D
                                as stated (solvers.dataspace.metric_of)
    nothing                     on a covariance-weighted D, unit noise in D's metric; on a plain D, NoiseUnstated
                                (plan 071 P8: D's dot product is not a noise model; D.standard_gaussian() states it)

and gives the engine the rule for targets kept symbolic (their degree from the data when there is no prior, as
pg.harmonic_target(T, degree) states it). pg.Problem's methods reach the engine through cached().
"""
import math

from ..core.kinds import SET, stated_kind


def noise_metric(noise, D):
    """The engine's metric (solvers.dataspace.NoiseMetric) for a noise statement on D (module docstring)."""
    from ..convex.algebra import Translate
    from ..convex.sets import Ellipsoid, NormBall
    from ..probability import is_gaussian
    from ..solvers.dataspace import metric_of
    centre = None
    while isinstance(noise, Translate):
        centre = (0.0 if centre is None else centre) + noise.shift.values
        noise = noise.base
    if noise is None:
        return metric_of(D, D, centre, "the standard Gaussian of the data space")
    if isinstance(noise, (NormBall, Ellipsoid)):
        from ..convex.forms import noise_metric as set_metric
        return set_metric(noise, D, centre)
    if is_gaussian(noise):
        from ..probability.gaussian import noise_metric as law_metric
        return law_metric(noise, D, centre)
    from ..operators.linear import LinearOperator
    if isinstance(noise, LinearOperator):                        # a covariance operator or D.whitened(C): no centre
        return metric_of(noise, D)
    return metric_of(noise, D, centre)


def symbolic_targets(G, T):
    """The public routes' rule for a target kept symbolic without a prior: its harmonic coefficients up to the data's
    degree (pg.harmonic_target(T, degree), plan 044)."""
    from ..properties.harmonic_targets import harmonic_target
    return harmonic_target(T, math.isqrt(int(G.lm.max())))


def cached(cache, G, noise=None, middle=None, **kw):
    """The decomposition of (G, the metric of noise, middle) from a cache (pg.Problem's), built on first use."""
    from ..solvers.dataspace import cached as engine
    return engine(cache, G, noise_metric(noise, G.codomain), middle, target_rule=symbolic_targets, **kw)


def decompose(G, noise=None, middle=None, **kw):
    """The data-space decomposition of a linear map G: M -> R^n (or a model-data relation: pg.Additive(G, noise) gives
    G and its noise), a noise statement (module docstring) and a middle operator (solvers/dataspace/decomposition.py):

        dec = pg.decompose(G, D.standard_gaussian())          SOLA's and a norm-ball BGP's (S = the model metric)
        dec = pg.decompose(G, noise, middle=prior)            PLI's (S = the prior covariance)
        dec.sola(T, np.logspace(-3, 3, 200), d)               every gamma at once; dec.posterior(T, d, a, nu), ...

    Keywords: gram= (the Gram matrix <g_i, S g_j> given, e.g. a pg.BlockDiagonal with shared blocks), blocks= ("auto":
    found from the values; None: one block; or a list of row sets), tail= (the partial routes' cut), route= ("randomised"
    or "lanczos" above settings.dense_max), rank_cap= (the randomised route's largest rank)."""
    from ..solvers.dataspace import DataDecomposition
    if hasattr(G, "effective") and hasattr(G, "noise") and noise is None:
        G, noise = G.effective
    elif hasattr(G, "noise") and hasattr(G, "G") and noise is None and stated_kind(G.noise) is SET:
        G, noise = G.G, G.noise
    if kw.get("normal") is not None:
        D = kw["normal"].domain
    elif G is not None:
        D = G.codomain
    else:
        D = kw["space"] if kw.get("space") is not None else getattr(kw.get("gram"), "domain", None)
    from ..relations.additive import require_noise
    require_noise(noise, D, "pg.decompose")
    return DataDecomposition(G, noise_metric(noise, D), middle, target_rule=symbolic_targets, **kw)
