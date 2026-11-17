"""The data-space engine (plan 070 P3, architecture review D4, decisions Q1 and Q7): one decomposition of the whitened
data Gram matrix per problem and the closed forms in its eigenbasis, internal to the methods. It sits in the numerical
column below convex/ and probability/, which call it, and above optimisers/ (the dual solvers of BGP run in its
eigenbasis).

    decomposition.py   DataDecomposition(G, metric, middle): the routes (blocks, dense, randomised, Lanczos), the
                       coordinates, the pairings of a property map, the cache (cached, decomposition_key)
    closed_forms.py    SOLA for arrays of gamma, the PLI posterior for arrays of (a, nu), BGP with an ellipsoid prior,
                       feasibility, the evidence, the damping for a misfit target, the prior-predictive check
    whitening.py       the metric, stated (metric_of: D.whitened(C), the noise covariance C, or D; never inferred from
                       a set, a measure or nothing), and the per-block factors F with F C F^T = I
    partial.py         the randomised (Nystrom) and Lanczos routes, their stated bounds, the deflated block CG
    factored.py        the closed forms of a factored property family T = Phi tau and of many properties at once
    cancellation.py    the mismatch and variance formed again where the closed form's subtraction loses digits
    quadratic.py       rowquad: diag(Phi V Phi^T) without the p x p matrix

The public route is pg.decompose (inference/dataspace.py: it reads the noise law of a model-data relation, a measure
or a noise set and states the metric here); the methods (pg.Problem.run, pg.inference.*) are the public routes to the
closed forms. DataDecomposition is not a pg name.
"""
from .decomposition import DataDecomposition, cached, decomposition_key
from .whitening import NoiseMetric, metric_of

__all__ = ["DataDecomposition", "NoiseMetric", "cached", "decomposition_key", "metric_of"]
