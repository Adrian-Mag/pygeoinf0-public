"""Named priors (plan 069 P6; the architecture review A13 and E3; decision Q2): Gaussian laws on a function space, chosen
by the caller by name. No space picks a prior: X.gaussian() refuses (core.errors.PriorUnstated) and names these.

    legendre_decay(X, decay, modes=None, mean=None)     on L2 of an interval (Lebesgue measure): N(mean, C) with
                                                        C phi_k = decay^(2k) phi_k for every k >= 0, phi_k the
                                                        orthonormal Legendre functions; trace class, tr C = 1/(1 - q),
                                                        q = decay^2. Applied and sampled as its first K modes (modes=K;
                                                        default the smallest K with decay^K < 1e-16, the old default),
                                                        the omitted part's trace q^K/(1 - q) and norm q^K stated
                                                        (theorem): OperatorCovariance.spectral
    random_features(X, length_scale=None, features=128, seed=0, mean=None)
                                                        on L2 of a domain in R^d (a box, a sphere, a ball, ...): the
                                                        finite-rank Gaussian N(mean, C_F), C_F the integral operator of
                                                        k_F(x, y) = (2/F) sum_j psi_j(x) psi_j(y), psi_j(x) =
                                                        cos(w_j . x + b_j), w_j ~ N(0, I/l^2), b_j ~ U(0, 2 pi) (seeded;
                                                        l by default a fifth of the diagonal of the bounding box, the old
                                                        default); what it approximates is claimed (below)
    cylindrical(H, alpha=1.0, mean=None)                N(mean, I/alpha) on a Hilbert space H: a CylindricalGaussian
                                                        (decision Q11: exact variances of bounded functionals, no samples
                                                        in H) in infinite dimension, a GaussianMeasure in finite
                                                        dimension (GaussianMeasure.cylindrical); pg.CylindricalGaussian(H,
                                                        alpha) is the same law, refused on a finite-dimensional H
    D.standard_gaussian()                               the standard Gaussian of a finite-dimensional inner product
                                                        (spaces: Euclidean, data and complex coordinate spaces, finite
                                                        direct sums)
    circular(X, covariance, mean)                       a circular complex Gaussian (probability/circular.py)

Function space first (plan 066): each prior is an operator on X stated through the Covariance root (probability/
covariance.py), with its self-adjointness, positivity and trace class claimed by their rule, never a law of coefficient
vectors of a realisation.

What random_features approximates (Rahimi and Recht 2007, "Random features for large-scale kernel machines", NIPS,
claim 1; Bochner's theorem for the Gaussian spectral measure): 2 cos(w . x + b) cos(w . y + b) = cos(w . (x - y)) +
cos(w . (x + y) + 2 b), so over the draw of (w, b), E k_F(x, y) = E cos(w . (x - y)) = exp(-|x - y|^2 / (2 l^2)), the
squared-exponential covariance; k_F is the mean of F independent terms in [-2, 2], so by Hoeffding's inequality
P(|k_F(x, y) - exp(-|x - y|^2 / (2 l^2))| >= eps) <= 2 exp(-F eps^2 / 8) for each fixed x, y. Both are statements about
the random draw: the error of the draw a seed gives is not derived (UNKNOWN). The covariance applied is that of the
realised (fitted) features, for which self-adjointness, positivity and trace class hold exactly by its form.
"""
import math

import numpy as np

from ..core.claims import THEOREM, UNKNOWN, Claim
from ..core.errors import Refused

__all__ = ["legendre_decay", "random_features", "cylindrical"]

RAHIMI_RECHT = ("Bochner's theorem for the Gaussian spectral measure (Rahimi and Recht 2007, Random features for "
                "large-scale kernel machines, claim 1) and Hoeffding's inequality for F independent terms in [-2, 2]")


def _require_l2(X, what):
    from ..spaces.base import L2
    if not isinstance(X, L2):
        raise Refused(f"{what} is a Gaussian measure on an L2 space; {X} is not one (a covariance operator needs the "
                      "inner product)", needs=(f"{what} on pg.L2 of the domain, pushed to {X} by an operator if that is "
                                               "what you mean",))


def _legendre_tail(decay, K):
    """The omitted part of legendre_decay after K modes: its trace sum_{k >= K} decay^(2k) = q^K / (1 - q) and its norm
    q^K (the largest omitted eigenvalue), q = decay^2."""
    q = decay * decay
    return q ** K / (1.0 - q), q ** K


def legendre_decay(X, decay=0.8, modes=None, mean=None):
    """N(mean, C) on L2 of an interval with C phi_k = decay^(2k) phi_k for the orthonormal Legendre functions phi_k of
    Lebesgue measure, every k >= 0 (module docstring): applied and sampled as its first K modes, the omitted part stated
    (OperatorCovariance.spectral). modes: K (default the smallest K with decay^K < 1e-16, plus one)."""
    from ..measures import LebesgueMeasure
    from ..spaces.elements import Family
    from .gaussian import GaussianMeasure
    from .operator_covariance import OperatorCovariance
    _require_l2(X, "legendre_decay")
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay = {decay!r}: the eigenvalues decay^(2k) are summable (C trace class) for 0 < decay < 1")
    if type(X.measure) is not LebesgueMeasure:
        raise Refused(f"legendre_decay uses the Legendre functions orthonormal for Lebesgue measure; {X} has the measure "
                      f"{X.measure}, for which they are not orthonormal (so they are not eigenfunctions of the stated C)",
                      needs=("pg.Covariance.spectral(eigenvalues, eigenvectors) with eigenvectors orthonormal in X",))
    K = int(math.ceil(math.log(1e-16) / math.log(decay))) + 1 if modes is None else int(modes)
    if K < 1:
        raise ValueError("modes >= 1")
    from ..realisations.registry import capability
    try:                                       # the capability of the realisation (an interval's 'legendre')
        R, lam = capability(X.backend, "legendre_decay_generators", "legendre_decay")(decay, K)
    except NotImplementedError as e:
        raise Refused(f"legendre_decay is a prior on L2 of an interval, whose 'legendre' realisation has the orthonormal "
                      f"Legendre functions; {X} realised {X.realisation!r} has none ({e})",
                      needs=("random_features(X, length_scale, features, seed) on a box, sphere or ball",
                             "pg.GaussianMeasure(X, covariance=C) with a covariance of your own")) from e
    phi = Family(X, R, name="φ")
    tail, top = _legendre_tail(decay, K)
    eps = np.finfo(float).eps                  # so that [sum lam, sum lam + tail] contains tr C despite the rounding of
    tail = tail * (1 + 4 * eps) + 2 * K * eps * float(np.sum(lam))      # both sums (a bound on the omitted trace)
    a, b = X.domain.a, X.domain.b               # an interval: the realisation 'legendre' exists only there
    q = decay * decay
    named = Claim(f"legendre_decay: N(0, C), C phi_k = {decay:g}^(2k) phi_k for every k >= 0, phi_k the orthonormal "
                  f"Legendre functions of Lebesgue measure on [{a:g}, {b:g}]; tr C = 1/(1 - {q:g})", THEOREM,
                  "the definition (named by the caller; plan 069 P6)", key="prior")
    C = OperatorCovariance.spectral(lam, phi, tail=tail, tail_max=top, status=THEOREM,
                                    reason=f"the omitted eigenvalues {decay:g}^(2k), k >= {K}: their sum q^K/(1 - q), "
                                           "their largest q^K (a geometric series)", name="C")
    return GaussianMeasure(X, C, mean, name=f"legendre_decay({decay:g})", _claims=[named])


def _box_diagonal(D):
    box = getattr(D, "bounding_box", None)
    if box is None:
        return None
    box = box() if callable(box) else box
    return math.sqrt(sum((hi - lo) ** 2 for lo, hi in box))


def _feature_weights(F):
    """The weights 2/F of the F random features (each term of k_F is 2 cos cos / F)."""
    return np.full(F, 2.0 / F)


def random_features(X, length_scale=None, features=128, seed=0, mean=None):
    """The finite-rank Gaussian N(mean, C_F) of F random Fourier features on L2 of a domain in R^d (module docstring),
    with what it approximates claimed: the squared-exponential covariance exp(-|x - y|^2 / (2 l^2)) in expectation over
    the features and with Hoeffding's bound for each pair of points (theorem), the error of this draw UNKNOWN."""
    from ..spaces.elements import Family
    from .gaussian import GaussianMeasure
    from ..operators.linear import FiniteRank
    from .operator_covariance import OperatorCovariance
    _require_l2(X, "random_features")
    F = int(features)
    if F < 1:
        raise ValueError("features >= 1")
    D = X.domain
    cd = int(getattr(D, "cartesian_dim", 0) or 0)
    if cd < 2:
        raise Refused(f"random_features draws frequencies in the Cartesian coordinates of a domain in R^d, d >= 2; {D} "
                      "is not one (the old default had none on an interval either)",
                      needs=("legendre_decay(X, decay) on an interval",
                             "pg.GaussianMeasure(X, covariance=C) with a covariance of your own"))
    ell = length_scale
    if ell is None:
        diag = _box_diagonal(D)
        if diag is None:
            raise Refused(f"random_features on {D}: no bounding box for the default length scale",
                          needs=("length_scale=l",))
        ell = 0.2 * diag
    ell = float(ell)
    if not (0.0 < ell < math.inf):
        raise ValueError("length_scale > 0")
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((F, cd)) / ell
    b = rng.uniform(0, 2 * math.pi, F)

    def fun_idx(*X_and_idx):
        *P, idx = X_and_idx
        return np.cos(sum(W[idx, k] * np.asarray(P[k], float)[..., None] for k in range(cd)) + b[idx])
    from ..realisations.registry import capability
    R = capability(X.backend, "fit_family", "random_features")(fun_idx, F, name="random features")
    psi = Family(X, R, name="ψ")
    w = _feature_weights(F)
    try:
        fit = float(np.max(psi.errors))
    except Exception:                                       # the realisation states no errors
        fit = math.nan
    fit_text = f"at most {fit:.3g} each in the norm of {X}" if math.isfinite(fit) else "not stated by the realisation"
    C = OperatorCovariance(FiniteRank(psi, w, name="C_F"), name="C_F", _finite=(psi, w, False),
                           selfadjoint=(True, THEOREM, "sum_j (2/F) psi_j (x) psi_j: a sum of self-adjoint rank-one "
                                                       "operators"),
                           positive=("psd", THEOREM, "a sum of rank-one operators psi_j (x) psi_j with weights 2/F > 0"))
    claims = [
        Claim(f"random_features: N(0, C_F), C_F u = sum_j (2/{F}) <psi_j, u> psi_j, psi_j(x) = cos(w_j . x + b_j), "
              f"w_j ~ N(0, I/l^2), b_j ~ U(0, 2 pi), l = {ell:.6g}, seed {seed}", THEOREM,
              "the definition (named by the caller; plan 069 P6)", key="prior"),
        Claim(f"approximates the squared-exponential covariance exp(-|x - y|^2 / (2 l^2)), l = {ell:.6g}: over the draw "
              f"of the features E k_F(x, y) = exp(-|x - y|^2 / (2 l^2)), and for each fixed x, y "
              f"P(|k_F(x, y) - exp(-|x - y|^2 / (2 l^2))| >= eps) <= 2 exp(-F eps^2 / 8), F = {F}", THEOREM,
              RAHIMI_RECHT, key="approximates"),
        Claim(f"the error of this draw (seed {seed}, {F} features) against exp(-|x - y|^2 / (2 l^2)), in any norm",
              UNKNOWN, "not derived: the bound above is a probability over draws, not a bound for one seed",
              key="approximation error"),
        Claim(f"the features are realised by a fit, errors {fit_text}; the covariance applied is that of the realised "
              "features (self-adjoint, positive semidefinite and of finite rank exactly)", THEOREM,
              "the form sum_j w_j psi_j (x) psi_j with w_j > 0 holds for any generators", key="realised features")]
    return GaussianMeasure(X, C, mean, name=f"random_features({F})", _claims=claims)


def cylindrical(H, alpha=1.0, mean=None):
    """N(mean, I/alpha) on a Hilbert space H (module docstring): GaussianMeasure.cylindrical."""
    from .gaussian import GaussianMeasure
    return GaussianMeasure.cylindrical(H, alpha, mean)
