"""The data-dependent choice rules (plan 070 P2, review D3; thesis el:59, el:524-526): each takes (problem, method,
hyperparameter family) and returns a meta record (a Choice) that states its family, the values chosen and whether the
datum entered the choice. The cells and methods take the chosen values as stated values and pass the record on.

    ch = pg.meta.discrepancy(pb, "least_squares", pg.meta.damping("least_squares", centre=c), target=None)
    pb.run("least_squares", **ch.options())             damped least squares at the gamma with chi^2 = target (n)
    ev = pg.meta.maximum_evidence(pb, "pli", pg.meta.scales())     (a, nu) maximising the Gaussian evidence p(d | a, nu)
    fit = pg.meta.fit_noise(pb, "pli", pg.meta.NoiseFamily(D, groups="s"))   theta (and a) maximising p(d | theta, a)
    k = pg.meta.knee(pb, "sola", pg.meta.damping("sola", lo, hi), rule=pg.meta.Knee())   the knee of SOLA's curve
    pg.meta.RULES                                        {name: rule}: the rules above (tests/test_laws.py checks each)

discrepancy. For d = G m + e with the noise standard Gaussian in D's metric (either route of Q1 as amended, plan 071 P4: a
plain D with pg.Additive(G, N(b, C)) is read in the noise's inner product, R.standardised(); a covariance-weighted D
with its standard Gaussian, or with a point relation, is that already; a point relation into a plain D states no noise
and is refused unless target= is stated), gamma with ||G m_gamma - d||^2 = target
(default n = dim D: E chi^2 = n), m_gamma = c + G* (G G* + gamma I)^-1 (d - G c) the minimiser of Phi + gamma J,
J = (1/2)||m - c||^2 (inv:289-364). The root is pg.solvers.damping_for_misfit (eigendecomposition up to
settings.dense_max, Lanczos above): the same numbers as before plan 070.

maximum_evidence and fit_noise. The maximiser of the Gaussian evidence p(d | theta) for d = G m + offset + e,
m ~ N(m0, a^2 S), e ~ N(0, C(theta)) (appQ:173-182): over the scales (a, nu) with C(theta) = nu^2 C (C the MDR's
noise covariance), or over a NoiseFamily C(theta) (the MDR's noise is replaced; a fitted too unless a= is given).
Route: when the groups of a noise family are unions of the blocks of the data-space decomposition of (G, the base
noise, the prior) and there are no extra terms, the evidence and its gradient are closed forms from that one
decomposition (DataDecomposition.evidence with block_scale, plan 009 G3); otherwise every evaluation factorises
a^2 G S G* + C(theta) (n <= settings.dense_max). The scales use the problem's decomposition with the prior covariance
(pb.decomposition("prior")). The claims say which. A fitted noise law is never the inference noise unless it is
passed explicitly, pg.Additive(G, fit.measure()) (appQ:252).

knee. The knee of SOLA's (sigma, TRM) curve over the family's range of gamma by a declared rule (pg.meta.Knee;
meta/tradeoff.py): from the curve's axes, which do not depend on the datum, so "used the datum" does not hold.
"""
import math

import numpy as np

from ..core.claims import DERIVED, Claim
from ..core.errors import Refused, Unsupported
from ..probability import is_gaussian
from ..spaces.identity import require
from .families import HyperparameterFamily, NoiseFamily, damping, scales
from .record import Choice
from .tradeoff import Knee, TradeOff


def _needs(problem, rule, *fields):
    missing = [f for f in fields if getattr(problem, f, None) is None]
    if missing:
        raise Refused(f"{rule} needs the problem's {', '.join(missing)}", needs=missing)


def _family(family, default, rule, method):
    family = default if family is None else family
    if not isinstance(family, HyperparameterFamily):
        raise TypeError(f"{rule}: the family is a pg.meta hyperparameter family (damping, scales, NoiseFamily), not "
                        f"{type(family).__name__}")
    if family.method != method:
        raise Refused(f"{rule}: the family {family.name!r} parameterises {family.method}, not {method}",
                      needs=[f"a family of {method}"])
    return family


def _forward(mdr, datum, rule, noise=True):
    """(G, d, claims) with the noise the standard Gaussian of G's codomain (plan 071 P4, Q1 as amended): an additive
    Gaussian kernel read in its noise's inner product (R.standardised(): a plain D with N(b, C) in the relation, or a
    covariance-weighted D with its standard Gaussian); a linear point relation into a covariance-weighted D (unit noise
    in D's metric, claimed) or, when noise is False (a stated target reads no noise), into any R^n. A point relation into
    a plain D with the noise needed is refused (NoiseUnstated, naming both routes)."""
    from ..core.kinds import MEASURE
    from ..operators.linear import LinearOperator
    from ..relations.additive import point_noise
    from ..relations.kernels import AdditiveMarkovKernel
    from ..spaces.base import Euclidean
    if getattr(mdr, "kind", None) is MEASURE and isinstance(mdr, AdditiveMarkovKernel):
        G, d = mdr.standardised()
        return G, d(datum), []
    if isinstance(mdr, LinearOperator) and isinstance(mdr.codomain, Euclidean):
        return mdr, datum, ([point_noise(mdr, rule)[1]] if noise else [])
    raise Refused(f"{rule}: the model-data relation is a linear operator into R^n or an additive Gaussian kernel, not "
                  f"{type(mdr).__name__}", needs=["a linear MDR"])


class Discrepancy(Choice):
    kind = "discrepancy principle"


def discrepancy(problem, method="least_squares", family=None, target=None):
    """gamma by the discrepancy principle, ||G m_gamma - d||^2 = target (default n), for damped least squares at the
    family's centre (module docstring): a Choice with .gamma."""
    from ..solvers.choice import damping_for_misfit
    if method != "least_squares":
        raise Refused(f"the discrepancy principle chooses the damping of least_squares, not of {method}",
                      needs=["method='least_squares'"])
    _needs(problem, "the discrepancy principle", "mdr", "datum")
    family = _family(family, damping("least_squares"), "discrepancy", method)
    G, d, read = _forward(problem.mdr, problem.datum, "the discrepancy principle", noise=target is None)
    c = family.options.get("centre")
    y = d if c is None else d - G(c)
    D = G.codomain
    n = D.dim if target is None else float(target)
    alpha = damping_for_misfit(G @ G.adjoint, y, target=n)
    out = Discrepancy()
    out._choose("the discrepancy principle", family, {"gamma": float(alpha)}, True,
                "gamma solves ||G m_gamma - d||^2 = target for this datum", [
                    Claim(f"discrepancy principle: chi^2 = {n:.6g}" + (" (= n: the expected chi^2 of the noise)" if
                          target is None else " (a stated target)"), DERIVED, "inv:289-364, el:524")] +
                read + list(alpha.claims))
    out.target = n
    return out


class EvidenceChoice(Choice):
    """A maximum of the Gaussian evidence: the values, log_evidence, route and the claims."""

    kind = "evidence maximisation"

    def _show(self):
        return list(self.values) + ["log_evidence"]


def maximum_evidence(problem, method="pli", family=None, start=None, optimiser="L-BFGS-B"):
    """(a, nu) maximising log p(d | a, nu) (module docstring): a Choice with .a, .nu, .log_evidence."""
    from ..optimisers.minimise import minimise
    if method != "pli":
        raise Refused(f"the evidence is that of the Gaussian model of pli, not of {method}", needs=["method='pli'"])
    _needs(problem, "evidence maximisation", "mdr", "prior", "datum")
    family = _family(family, scales(), "maximum_evidence", method)
    if isinstance(family, NoiseFamily):
        return fit_noise(problem, method, family, start=start, optimiser=optimiser)
    if family.parameters != ("a", "nu"):
        raise Refused("maximum_evidence searches the scales (a, nu) or a NoiseFamily", needs=["pg.meta.scales()"])
    if not is_gaussian(problem.prior):
        raise Refused("the evidence needs a Gaussian prior", needs=["a GaussianMeasure prior"])
    dec = problem.decomposition("prior")
    d = problem.mdr.datum(problem.datum) if hasattr(problem.mdr, "datum") else problem.datum
    m0 = None if problem.prior.zero_mean else problem.prior.mean
    x0 = np.zeros(2) if start is None else np.log(np.asarray(start, float))

    def f(x):
        a, nu = np.exp(x)
        r = dec.evidence(d, a=a, nu=nu, prior_mean=m0, gradient=True)
        return -r.log_evidence, -np.array([r.d_a * a, r.d_nu * nu])
    res = minimise(f, x0, optimiser, dict(maxiter=2000, gtol=1e-10), jac=True)
    a, nu = map(float, np.exp(res.x))
    out = EvidenceChoice()
    out._choose("evidence maximisation", family, {"a": a, "nu": nu}, True,
                "the evidence p(d | a, nu) of this datum", [
                    Claim("the maximiser of the Gaussian evidence over (a, nu)", DERIVED,
                          f"closed form from one data-space decomposition ({dec.describe()}); {optimiser}: "
                          f"{res.message} ({res.nit} iterations)")])
    out.log_evidence, out.route = -float(res.fun), "closed"
    return out


class NoiseFit(EvidenceChoice):
    """A fitted noise law (a meta record): theta by name, the prior scale a, the log evidence and the claims."""

    kind = "noise law fitted by the evidence"

    def measure(self, name=None):
        """The fitted law as a measure on D: use it as noise only by passing it explicitly (module docstring)."""
        return self.family.measure(self.theta_vector, name=name)

    def __repr__(self):
        th = ", ".join(f"{k} = {v:.4g}" for k, v in self.theta.items())
        return f"NoiseFit({th}; a = {self.a:.4g}; log evidence {self.log_evidence:.6g}; {self.route})"


def fit_noise(problem, method="pli", family=None, a=None, start=None, optimiser="L-BFGS-B"):
    """Maximise the Gaussian evidence over a NoiseFamily's parameters (and the prior scale a when a is None): module
    docstring. The problem's MDR gives G (its noise is replaced by the family's); a NoiseFit."""
    from ..optimisers.minimise import minimise
    from ..solvers.dataspace import DataDecomposition
    if method != "pli":
        raise Refused(f"the evidence is that of the Gaussian model of pli, not of {method}", needs=["method='pli'"])
    if not isinstance(family, NoiseFamily):
        raise Refused("fit_noise needs the noise family it searches: family=pg.meta.NoiseFamily(D, ...)",
                      needs=["family=pg.meta.NoiseFamily(D, ...)"])
    _needs(problem, "fit_noise", "mdr", "prior", "datum")
    prior, datum = problem.prior, problem.datum
    G = getattr(problem.mdr, "G", problem.mdr)
    D = family.D
    require(G.codomain, D, f"G maps into {G.codomain}, the family lives on {D}", "G")
    if not is_gaussian(prior):
        raise TypeError("the evidence needs a prior measure on the model space (a GaussianMeasure)")
    d = np.asarray(getattr(datum, "values", datum), float)
    m0 = None if prior.zero_mean else prior.mean
    ng = len(family.groups)
    th0 = np.ones(family.size) if start is None else np.asarray(start, float)
    x0 = np.log(np.r_[th0, 1.0 if a is None else []])
    dec = None
    if not family.extra:
        dec = DataDecomposition(G, family.base, middle=prior.covariance)
        bg = [np.unique(family.group_of[p]) for p in dec._rows]
        if dec.partial or any(len(u) != 1 for u in bg):
            dec = None
        else:
            gb = np.array([u[0] for u in bg])                   # the group of each block
    if dec is not None:
        route = f"closed form from one data-space decomposition ({dec.describe()}), {len(gb)} blocks in {ng} groups"

        def f(x):
            th, av = np.exp(x[:ng]), (math.exp(x[ng]) if a is None else float(a))
            r = dec.evidence(d, a=av, nu=1.0, block_scale=th[gb], prior_mean=m0, gradient=True)
            gth = np.bincount(gb, weights=r.d_block_scale, minlength=ng) * th
            grad = -np.r_[gth, [r.d_a * av] if a is None else []]
            return -r.log_evidence, grad
        res = minimise(f, x0, optimiser, dict(maxiter=2000, gtol=1e-10), jac=True)
    else:
        from ..settings import settings
        if D.dim > settings.dense_max:
            raise Unsupported("a noise family whose groups split the decomposition's blocks, or with extra terms, is "
                              f"fitted densely: n = {D.dim} > settings.dense_max")
        C = prior.covariance
        K = D.solve_metric(np.asarray((G @ C @ G.adjoint).matrix(), float).T).T
        K = 0.5 * (K + K.T)
        e = d - (0.0 if m0 is None else G(m0).values)
        route = f"dense: a Cholesky factorisation of a^2 K + C(theta) per evaluation (n = {D.dim})"

        def f(x):
            th, av = np.exp(x[:family.size]), (math.exp(x[family.size]) if a is None else float(a))
            S = family.covariance(th)
            A = av * av * K + (np.diag(S) if S.ndim == 1 else S)
            L = np.linalg.cholesky(A)
            z = np.linalg.solve(L, e)
            return 0.5 * (2.0 * np.sum(np.log(np.diag(L))) + z @ z + D.dim * math.log(2 * math.pi))
        res = minimise(f, x0, optimiser, dict(maxiter=2000))
    th = np.exp(res.x[:family.size])
    av = math.exp(res.x[family.size]) if a is None else float(a)
    out = NoiseFit()
    out._choose("evidence maximisation over a noise family", family, {**dict(zip(family.names, th)), "a": av}, True,
                "the evidence p(d | theta, a) of this datum", [
                    Claim("a fitted noise law: the maximiser of the Gaussian evidence over the family" +
                          ("" if a is not None else " and the prior scale a"), DERIVED,
                          f"{route}; {optimiser}: {res.message} ({res.nit} iterations)"),
                    Claim("not the inference noise unless passed to pg.Additive explicitly", DERIVED, "appQ:252")])
    out.theta_vector, out.a = np.asarray(th, float), av
    out.theta = dict(zip(family.names, out.theta_vector))
    out.log_evidence = -float(res.fun if np.ndim(res.fun) == 0 else res.fun[0])
    out.route = "closed" if dec is not None else "dense"
    return out


def knee(problem, method="sola", family=None, rule=None, T=None, column=0):
    """The knee of the trade-off curve of the problem's MDR and MPR (or T) over the family's range of gamma by a
    declared rule (default pg.meta.Knee()): a KneeResult (a Choice with .gamma). method "sola" (the model metric) or
    "pli" (the prior covariance as the metric: gamma = (nu / a)^2)."""
    if method not in ("sola", "pli"):
        raise Refused(f"the knee is chosen on the trade-off curve of sola or pli, not of {method}",
                      needs=["method='sola'", "method='pli'"])
    _needs(problem, "the knee", "mdr")
    family = _family(family, None, "knee", method) if family is not None else None
    if family is None or family.ranges["gamma"][1] is None:
        raise Refused("the knee needs the range of gamma it searches: family=pg.meta.damping(method, lo=, hi=)",
                      needs=["lo", "hi"])
    T = problem.mpr if T is None else T
    if T is None:
        raise Refused("the knee needs the property map: the problem's mpr or T=", needs=["mpr"])
    lo, hi = family.ranges["gamma"]
    curve = TradeOff(problem.decomposition("prior" if method == "pli" else None), T)
    return (rule or Knee()).find(curve, lo, hi, column, family=family)


RULES = {"discrepancy": discrepancy, "maximum_evidence": maximum_evidence, "fit_noise": fit_noise, "knee": knee}

