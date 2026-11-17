"""Inference methods as recipes over the cells (plan 008; EXP-028 DESIGN.md 2.4, 3.9). Each is a plain function of
the objects it needs, and declares them (its natural level, el:62) so that pg.Problem runs it when they exist:

    minimum_norm(mdr, datum)                     the summary of B5 with the whole space: G^+ d
    least_squares(mdr, datum, gamma)             B13/B14's summary: damped least squares; gamma stated (a number,
                                                 or pg.meta.discrepancy(problem).gamma: chosen from the datum, recorded)
    sola(mdr, mpr, datum, gamma)                 X = T G* (G G* + gamma C_e)^-1: an estimate of the DELIVERED property
                                                 X G m with its noise law (inf:75-350); prior none; gamma stated
    csola(prior, mdr, mpr, datum, X)             X d + (T - X G)(S_M) + (-X)(S_N): an outer set for the target
                                                 (inf:352-394)
    ocsola(prior, mdr, mpr, datum, X, alpha)     BGP with the slab of the fixed estimator X: the one-dimensional dual
                                                 inf over tau of tau s + |tau| w + sigma_S(t - tau r) (inf:427-465);
                                                 a ball in closed form, a pointwise box per property with tau from a
                                                 quadrature and one certified support per end (inference/ocsola_box.py)
    bgp(prior, mdr, mpr, datum, directions)      the feasible property set T(K(d)) with certificates (inf:476-1092)
    pli(prior, mdr, mpr, datum)                  the property posterior T# mu^d (inf:1095-1292)

Where the data space allows it (n <= settings.dense_max, or a structured Gram operator), sola, pli, bgp, ocsola and
csola work through one data-space decomposition (plan 009 G3, solvers/dataspace/decomposition.py), shared through pg.Problem's
cache: SOLA for an array of gamma in one call, PLI from the closed form, BGP and oCSOLA in O(n) per direction or gamma.
The results' claims name the decomposition.

Each method declares (plan 070 P5, review D9) the calibrations it supports (calibrations: the classes of
inference/calibration.py whose guarantee its result is about) and the options it takes (options); Problem.run refuses a
calibration or an option it does not declare, naming the declared ones, and adds the calibration claims to the result.
A pg.meta choice's options() are a method's options for these hyperparameter families only (plan 070 P2's open
decision 6):

    damping of least_squares  -> gamma, centre          damping of sola -> gamma (SOLA's X: also csola's, ocsola's)
    scales (a, nu), damping of pli, a noise family -> no option of pli: refused, its hint saying how the choice is
    stated through the elements (the prior N(m0, a^2 S) and noise N(b, nu^2 C); a fitted law as the noise explicitly)
"""
import math

import numpy as np

from ..core.claims import DERIVED, THEOREM, Claim, Claims
from ..core.errors import Refused
from ..core.kinds import MEASURE, SET, stated_kind
from ..spaces.base import Euclidean
from ..operators.linear import Identity, LinearOperator
from .calibration import Bayesian, Deterministic, Frequentist, Hardened, Unstated

METHODS = {}


class Method:
    """A recipe: its function, the elements it needs (with tests and wording), what it produces, the calibrations it
    supports (classes; Unstated among them), the options it takes and a hint for the options it refuses."""

    def __init__(self, name, fn, needs, produces, calibrations, options=(), hint=None):
        self.name, self.fn, self.needs, self.produces = name, fn, needs, produces
        self.calibrations, self.options, self.hint = tuple(calibrations), tuple(options), hint

    def missing(self, fields):
        return [what for field, test, what in self.needs if fields.get(field) is None or not test(fields[field])]

    def supports(self, calibration):
        """Refused unless the calibration (None: unstated) is one this method declares."""
        if calibration is not None and not isinstance(calibration, self.calibrations):
            names = ", ".join(c.__name__ for c in self.calibrations)
            raise Refused(f"{self.name} supports the calibrations {names}, not {calibration!r}: its result is not the "
                          "object that guarantee is about", needs=[c.__name__ for c in self.calibrations])

    def takes(self, options):
        """Refused for an option the method does not declare (a pg.meta choice's family claim named)."""
        bad = [k for k in options if k != "cache" and k not in self.options]
        if bad:
            fams = {c.statement for k in bad for c in (getattr(options[k], "claims", None) or ())
                    if getattr(c, "key", None) == "hyperparameter family"}
            raise Refused(f"{self.name} takes no option{'s' if len(bad) > 1 else ''} {', '.join(bad)}; its options: "
                          f"{', '.join(self.options) or 'none'}" + (f". {self.hint}" if self.hint else "") +
                          "".join(f" (the choice's {f})" for f in sorted(fams)), needs=list(self.options))


def method(name, needs, produces, *, calibrations, options=(), hint=None):
    def wrap(fn):
        METHODS[name] = Method(name, fn, needs, produces, calibrations, options, hint)
        return fn
    return wrap


def _point_linear(R):
    return isinstance(R, LinearOperator) and isinstance(R.codomain, Euclidean)


def _gaussian_kernel(R):
    """An additive Gaussian kernel: a relation of kind measure that is an AdditiveMarkovKernel (its class or a subclass)."""
    from ..relations.kernels import AdditiveMarkovKernel
    return getattr(R, "kind", None) is MEASURE and isinstance(R, AdditiveMarkovKernel)


def _additive_set(R):
    """G m + N with a noise set N: a relation of kind set that is an AdditiveSet (its class or a subclass)."""
    from ..relations.additive import AdditiveSet
    return getattr(R, "kind", None) is SET and isinstance(R, AdditiveSet)


def _set_mdr(R):
    """A set-valued relation with a set cell for its update (additive, linear or voting), or a linear point relation."""
    from ..relations.algebra import Voting
    from ..relations.linear import LinearRelation
    return (getattr(R, "kind", None) is SET and isinstance(R, (LinearRelation, Voting))) or _additive_set(R) \
        or _point_linear(R)


def _set_prior(S):
    """A state of kind set (the whole space, NoPrior(X), included)."""
    return stated_kind(S) is SET


def _convex_prior(S):
    """A set state with a support function (a ConvexSet)."""
    from ..convex.sets import ConvexSet
    return _set_prior(S) and isinstance(S, ConvexSet)


def _ball_or_box_prior(S):
    """A set state that is a norm ball or a pointwise box (their supports are in closed form here)."""
    from ..convex.pointwise import PointwiseBox
    from ..convex.sets import NormBall
    return _set_prior(S) and isinstance(S, (NormBall, PointwiseBox))


def _gaussian_prior(mu):
    """A state of kind measure that is a Gaussian law (GaussianMeasure, Dirac or CylindricalGaussian)."""
    from ..probability import is_gaussian
    return stated_kind(mu) is MEASURE and is_gaussian(mu)


def _anything(x):
    return True


DATUM = ("datum", _anything, "datum: the observed data")
MPR = ("mpr", _point_linear, "mpr: a linear property map into R^p")


def _forward(mdr, what="the method"):
    """(G, the noise law): a Gaussian kernel's; a point relation's by relations.additive.point_noise (unit noise in the
    metric of a covariance-weighted D; NoiseUnstated on a plain D: plan 071 P4, Q1 as amended)."""
    if _gaussian_kernel(mdr):
        return mdr.effective
    from ..relations.additive import point_noise
    return mdr, point_noise(mdr, what)[0]


def _decomposable(G):
    """Whether the data-space decomposition route applies: a dense or structured data Gram matrix."""
    from ..settings import settings
    return isinstance(G.codomain, Euclidean) and (G.codomain.dim <= settings.dense_max
                                                  or hasattr(G, "structured_gram"))


def _decomposition(cache, G, noise, middle=None):
    from .dataspace import cached
    return cached(cache, G, noise, middle)


def _centred(noise):
    from ..probability import GaussianMeasure
    return noise if noise.zero_mean else GaussianMeasure(noise.space, noise.covariance, name=noise.name)


@method("minimum_norm", [("mdr", _point_linear, "mdr: a linear point relation into R^n (exact data)"), DATUM],
        "point: a summary of the fibre (no guarantee)", calibrations=(Unstated,))
def minimum_norm(mdr, datum, **_):
    """The element of least norm of the fibre G^-1(d) (cell B5 with the prior none, the whole space)."""
    from ..cells import update
    from ..states.base import NoPrior
    return update(NoPrior(mdr.domain), mdr, datum).summary()


@method("least_squares", [("mdr", lambda R: _point_linear(R) or _gaussian_kernel(R), "mdr: a linear point relation "
                           "or an additive Gaussian kernel"), DATUM], "ordering: Phi + gamma J, with summary()",
        calibrations=(Unstated,), options=("gamma", "centre"))
def least_squares(mdr, datum, gamma=None, centre=None, **_):
    """The updated ordering Phi + gamma (1/2)||m - c||^2 (cells B13/B14); .summary() is damped least squares. gamma is
    stated: a number, or the gamma of a meta rule (pg.meta.discrepancy(problem, "least_squares", family)), whose
    record the ordering carries; without one the update refuses (review D3: the cells take a stated alpha)."""
    from ..cells import update
    from ..states.base import Ordering
    return update(Ordering.norm(mdr.domain, centre), mdr, datum, rule=("soft", gamma))


class SOLA:
    """A SOLA estimate: estimate = X d of the DELIVERED property X G m (not T m), its noise law X# noise, the delivered
    map X G and the operator X (inf:62, inf:296-350)."""

    def __init__(self, X, delivered, estimate, noise_law, claims):
        self.X, self.delivered, self.estimate, self.noise_law, self.claims = X, delivered, estimate, noise_law, claims

    def __repr__(self):
        return f"SOLA estimate {np.round(self.estimate.values, 6)} of the delivered property (std " \
               f"{np.round(np.sqrt(np.diag(np.atleast_2d(self.noise_law.covariance.matrix()))), 6)})"


@method("sola", [("mdr", lambda R: _point_linear(R) or _gaussian_kernel(R), "mdr: a linear MDR into R^n"), MPR, DATUM],
        "point: an estimate of the delivered property with its noise law", calibrations=(Unstated,),
        options=("gamma",))
def sola(mdr, mpr, datum, gamma=None, cache=None, **_):
    """X = T G* (G G* + gamma C_e)^-1 applied to the datum (module docstring). gamma stated (plan 071 P4, Mac review
    item 5: no default): a number, or an array for the closed forms at every gamma at once (DataDecomposition.sola:
    estimates, std, delivered mismatch, ...), or a meta rule's choice (pg.meta.knee(problem, "sola", family).gamma)."""
    if gamma is None:
        raise Refused("sola needs gamma, stated: a number, an array (every gamma at once), or the gamma of a meta rule "
                      "(pg.meta.knee(problem, 'sola', pg.meta.damping('sola', lo=, hi=)).gamma)", needs=["gamma"])
    if _gaussian_kernel(mdr):
        (G, noise), read = mdr.effective, []
    else:
        from ..relations.additive import point_noise
        (noise, claim), G = point_noise(mdr, "sola"), mdr
        read = [claim]
    d = mdr.datum(datum) if _gaussian_kernel(mdr) else datum
    if _decomposable(G) or np.ndim(gamma):
        dec = _decomposition(cache, G, noise)
        if np.ndim(gamma) or _many(mpr):
            return dec.sola(mpr, gamma, d)
        X = dec.sola_operator(mpr, gamma)
        cl = [Claim("estimates the delivered property X G m, not the target T m", THEOREM, "inf:62, inf:324-350"),
              Claim("prior: none (SOLA skips the update of a state on the model space)", THEOREM, "el:57"),
              Claim(f"X from the data-space decomposition ({dec.describe()})", DERIVED, "plan 009 G3")] + read
        if not noise.zero_mean:
            cl.append(Claim("the noise mean (a known offset) is subtracted from the datum", DERIVED, "E X e = X b"))
        return SOLA(X, X @ G, X(d if noise.zero_mean else d - noise.mean), X(_centred(noise)), Claims(cl))
    A = G @ G.adjoint
    X = mpr @ G.adjoint @ (A + gamma * noise.covariance).inv
    return SOLA(X, X @ G, X(d), X(noise), Claims([
        Claim("estimates the delivered property X G m, not the target T m", THEOREM, "inf:62, inf:324-350"),
        Claim("prior: none (SOLA skips the update of a state on the model space)", THEOREM, "el:57")] + read))


@method("csola", [("prior", _convex_prior, "prior: a convex set with a support function"),
                  ("mdr", _additive_set, "mdr: G m + N with a noise set N"), MPR, DATUM],
        "set: an outer set for the target property (deterministic bound)", calibrations=(Unstated, Deterministic),
        options=("X", "gamma"))
def csola(prior, mdr, mpr, datum, X=None, gamma=None, cache=None, **_):
    """E_X = X d + (T - X G)(S_M) + (-X)(S_N) (inf:366-382): forward cells only; X the fixed weights (e.g. sola().X),
    or SOLA's X for gamma from the data-space decomposition (noise covariance: the noise set's shape)."""
    from ..convex.algebra import image, minkowski
    if X is None:
        if gamma is None:
            raise Refused("csola needs the weights X, or gamma (SOLA's X)", needs=["X", "gamma"])
        from ..convex.feasible import ball_shape
        shape = ball_shape(mdr.noise, mdr.G.codomain)
        X = _decomposition(cache, mdr.G, None if shape is None else shape[0]).sola_operator(mpr, gamma)
    out = minkowski(minkowski(image(mpr - X @ mdr.G, prior), image(-1.0 * X, mdr.noise)), X(datum))
    out._claims.append(Claim("an outer set: P(d) is contained in it (the two error terms vary separately)", THEOREM,
                             "inf:373-383"))
    return out


@method("ocsola", [("prior", _ball_or_box_prior,
                    "prior: a norm ball or a pointwise box"),
                   ("mdr", _gaussian_kernel, "mdr: an additive Gaussian kernel (its slab)"), MPR, DATUM],
        "interval: oCSOLA bounds of one property (or of each property, with a box)",
        calibrations=(Unstated, Frequentist, Hardened), options=("X", "alpha", "gamma", "refine", "oversample", "rtol"))
def ocsola(prior, mdr, mpr, datum, X=None, alpha=0.05, gamma=None, cache=None, refine=None, oversample=4, rtol=None,
           **_):
    """[lower, upper] of the scalar property T m by the one-dimensional dual over the scale of X (module docstring).
    Without X, the SOLA estimator for gamma (a number or an array: arrays of bounds), with every pairing in closed form
    from the data-space decomposition.

    A pointwise box prior (pg.PointwiseBox on intervals, spheres, R^n and direct sums of them; plan 062,
    inference/ocsola_box.py): every property of T (one row of X each), tau searched on a quadrature approximation of
    sigma_B (oversample) and certified once per end (refine: further certified trials; None: to convergence where
    sigma_B is exact, else none; rtol: the certified bracket's, default the box's). Returns a BoxOCSOLA, the tuple
    (lower, upper) with tau_lower/tau_upper, the brackets, support counts and claims. gamma: a number."""
    from .sola_slabs import ocsola_ends
    from scipy.stats import norm
    G, noise = _forward(mdr)
    from ..convex.pointwise import PointwiseBox
    if isinstance(prior, PointwiseBox):
        from .ocsola_box import box_ocsola
        if X is None:
            if gamma is None:
                raise Refused("ocsola needs the estimator X, or gamma (SOLA's X)", needs=["X", "gamma"])
            if np.ndim(gamma):
                raise Refused("ocsola with a box takes one gamma (a number); loop over an array of gamma",
                              needs=["gamma"])
            X = _decomposition(cache, G, noise).sola_operator(mpr, float(gamma))
        d = mdr.datum(datum) if _gaussian_kernel(mdr) else datum
        return box_ocsola(prior, G, noise, mpr, d, X, alpha, refine=refine, oversample=oversample, rtol=rtol)
    if X is None:
        if gamma is None:
            raise Refused("ocsola needs the estimator X, or gamma (SOLA's X)", needs=["X", "gamma"])
        return _ocsola_closed(prior, mdr, mpr, datum, alpha, gamma, cache)
    w = norm.ppf(1 - alpha / 2) * math.sqrt(float(X(noise).covariance.matrix()[0, 0]))
    s = float(X(datum).values[0]) - (0.0 if noise.zero_mean else float(X(noise.mean).values[0]))
    m0 = prior.centre
    # pairings: t = T* 1, r~ = G* X* 1 (one property, one estimator); sigma_S(v) = <v, m0> + r ||v||
    one = mpr.codomain([1.0])
    t, rr = mpr.adjoint(one), G.adjoint(X.adjoint(X.codomain([1.0])))
    M = mpr.domain
    J = np.array([[M.inner(t, t), M.inner(t, rr)], [M.inner(t, rr), M.inner(rr, rr)]])
    shift_t = 0.0 if m0 is None else M.inner(t, m0)
    shift_r = 0.0 if m0 is None else M.inner(rr, m0)
    # coordinates of t and r~ in an orthonormal basis (reduction.reduce), with the general rule (optimisers/dual/exactness_reduce.py)
    return ocsola_ends(J, s - shift_r, w, prior.radius, shift_t,
                       _ocsola_errors(mpr, G, lambda: G.codomain.apply_metric(X.adjoint(X.codomain([1.0])).values)))


def _ocsola_errors(mpr, G, wy):
    """The stated realisation errors of oCSOLA's two columns (t = T* 1 and r~ = G* y, y = X* 1: ||d r~|| <= sum_i
    |(W y)_i| e_i for the kernels' errors e_i, W the data metric; wy() gives W y, the row of X's matrix), or None where
    none is stated (optimisers/dual/exactness_reduce.py)."""
    from ..optimisers.dual.exactness import kernel_errors
    te, ge = kernel_errors(mpr), kernel_errors(G)
    if te is None and ge is None:
        return None
    er = 0.0 if ge is None else float(np.abs(np.ravel(np.asarray(wy(), float))) @ ge)
    return np.array([0.0 if te is None else float(te[0]), er])


def _ocsola_closed(prior, mdr, mpr, datum, alpha, gamma, cache):
    """oCSOLA with SOLA's X for each gamma: <t, t>, <t, G* X* 1>, ||G* X* 1||^2 and the noise std from the closed forms
    (NOTES.md section 4)."""
    from .sola_slabs import ocsola_ends
    from scipy.stats import norm
    G, noise = _forward(mdr)
    if mpr.codomain.dim != 1:
        raise Refused("oCSOLA bounds one property: give a property map into R^1")
    dec = _decomposition(cache, G, noise)
    res = dec.sola(mpr, np.atleast_1d(np.asarray(gamma, float)), mdr.datum(datum) if _gaussian_kernel(mdr) else datum,
                   refuse=None)                         # the mismatch is not used here (solvers/dataspace/cancellation.py)
    z = norm.ppf(1 - alpha / 2)
    m0 = prior.centre
    out = []
    for i, g in enumerate(res.gamma):
        tt, rr = float(res.target[0]), float(res.delivered[i, 0])
        tr = float(res.cross[i, 0])
        shift_t = shift_r = 0.0
        if m0 is not None:
            shift_t = float(mpr(m0).values[0])
            shift_r = float(dec.sola_operator(mpr, g)(G(m0)).values[0])
        s, w = float(res.estimate[i, 0]), z * float(res.std[i, 0])
        cerr = _ocsola_errors(mpr, G, lambda: dec.sola_operator(mpr, g).matrix()[0])
        out.append(ocsola_ends(np.array([[tt, tr], [tr, rr]]), s - shift_r, w, prior.radius, shift_t, cerr))
    return out[0] if np.ndim(gamma) == 0 else (np.array([o[0] for o in out]), np.array([o[1] for o in out]))


class BGP:
    """BGP output: the feasible property set (a FeasibleImage), coordinate bounds with certificates, an outer polytope.
    For a property family or many properties (_many): arrays lower, upper, status_lower, status_upper instead of a
    list of Bound pairs (bounds is None; claims say how they were computed), with lower_bracket, upper_bracket ((p, 2))
    and error_lower, error_upper: each end's certified bracket and E of the general rule (optimisers/dual/exactness.py), as a
    Bound's lower, upper and error."""

    def __init__(self, P, bounds, polytope, arrays=None):
        self.set, self.bounds, self.polytope = P, bounds, polytope
        self.lower = self.upper = self.status_lower = self.status_upper = self.claims = None
        self.lower_bracket = self.upper_bracket = self.error_lower = self.error_upper = None
        if arrays is not None:
            self.lower, self.upper, self.status_lower, self.status_upper, self.claims = arrays[:5]
            for k, v in (arrays[5] if len(arrays) > 5 else {}).items():
                setattr(self, k, v)

    def interval(self, k=0):
        if self.bounds is None:
            return float(self.lower[k]), float(self.upper[k])
        lo, hi = self.bounds[k]
        return lo.value, hi.value

    def bracket(self, k=0):
        """((lower, upper) of the lower end, (lower, upper) of the upper end) of property k: the certified brackets,
        widened where the general rule made them BRACKET (optimisers/dual/exactness.py); interval(k) gives the computed ends."""
        if self.bounds is None:
            return tuple(map(float, self.lower_bracket[k])), tuple(map(float, self.upper_bracket[k]))
        lo, hi = self.bounds[k]
        return (lo.lower, lo.upper), (hi.lower, hi.upper)

    def plot(self, pair=(0, 1), directions=90, ax=None, **kw):
        """The outer polygon of the feasible property set projected onto properties pair = (i, j): its support in a
        direction q of the pair is the whole set's at q embedded (certified upper ends; plotting/sets.py)."""
        from ..convex.sets import Polytope
        from ..spaces.base import Euclidean
        from ..plotting.sets import directions_for, plot_set
        i, j = pair
        Q2 = directions_for(2, int(directions))
        Q = np.zeros((len(Q2), self.set.space.dim))
        Q[:, i], Q[:, j] = Q2[:, 0], Q2[:, 1]
        P2 = Polytope(Euclidean(2), Q2, self.set.outer_polytope(Q)[1], f"properties {i}, {j}", "outer", self.set)
        kw.setdefault("label", f"properties ({i}, {j}) of {self.set.name} (outer)")
        return plot_set(P2, ax=ax, **kw)

    def __repr__(self):
        if self.bounds is None:
            ok = int(np.sum((self.status_lower == "EXACT") & (self.status_upper == "EXACT")))
            return f"BGP intervals of {len(self.lower)} properties ({ok} certified EXACT at both ends)"
        return "BGP intervals " + ", ".join(f"[{lo.value:.6g}, {hi.value:.6g}] ({hi.status})" for lo, hi in self.bounds)


def _many(T):
    """A property family (T = Phi tau) or more properties than COVARIANCE_DENSE: results as arrays, covariances lazy."""
    from ..solvers.dataspace.factored import COVARIANCE_DENSE
    return getattr(T, "factors", None) is not None or T.codomain.dim > COVARIANCE_DENSE


@method("bgp", [("prior", _set_prior, "prior: a set (pg.NoPrior(X), the whole space, included)"),
                ("mdr", _set_mdr, "mdr: a set-valued relation (G m + N; try mdr.acceptance(alpha) for a kernel)"),
                MPR, DATUM], "set: the feasible property set with certified bounds",
        calibrations=(Unstated, Deterministic, Frequentist, Hardened), options=("directions", "estimable"))
def bgp(prior, mdr, mpr, datum, directions=None, cache=None, estimable=False, **_):
    """The feasible property set T(K(d)), its coordinate bounds and an outer polytope (module docstring).
    estimable=True declares the properties in the data functionals' span (core/estimable.py; with no prior it decides)."""
    from ..cells import push, update
    K = update(prior, mdr, datum)
    if cache is not None and getattr(K, "_cache", False) is None:
        K._cache = cache                                    # FeasibleSet: share the problem's decompositions
    P = push(K, mpr)
    if estimable:
        P.estimable = True
    if _many(mpr) and hasattr(P, "intervals"):
        return BGP(P, None, None if directions is None else P.outer_polytope(directions),
                   arrays=P.intervals(brackets=True))
    bounds = [P.bounds(k) for k in range(mpr.codomain.dim)]
    return BGP(P, bounds, None if directions is None else P.outer_polytope(directions))


PLI_HINT = ("A pg.meta choice for pli is stated through the elements: the scales (a, nu) as the prior N(m0, a^2 S) and the "
            "noise N(b, nu^2 C) (gamma = (nu / a)^2 fixes their ratio only), a fitted noise law as the noise, "
            "pg.Additive(G, fit.measure()) (appQ:252): pb.with_(prior=..., mdr=...)")


@method("pli", [("prior", _gaussian_prior, "prior: a Gaussian measure"),
                ("mdr", _gaussian_kernel, "mdr: an additive Gaussian kernel"), MPR, DATUM],
        "measure: the property posterior", calibrations=(Unstated, Bayesian), hint=PLI_HINT)
def pli(prior, mdr, mpr, datum, cache=None, **_):
    """T# of the posterior (cells B10 then F10); through the data-space decomposition with the prior covariance as
    the middle operator where it applies (the same measure, from the closed form)."""
    from ..cells import push, update
    G, noise = mdr.effective
    if not (_decomposable(G) and _gaussian_prior(prior)):
        return push(update(prior, mdr, datum), mpr)
    from ..core.claims import ASSERTED
    from ..operators.linear import Matrix
    from ..probability import Covariance, GaussianMeasure
    from ..probability.covariance import posterior_records
    dec = _decomposition(cache, G, noise, prior.covariance)
    res = dec.posterior(mpr, mdr.datum(datum), prior_mean=None if prior.zero_mean else prior.mean)
    P = mpr.codomain
    sa, pos = posterior_records(prior, noise, "a posterior covariance T (C - C G* S^-1 G C) T*")
    if isinstance(res.covariance, LinearOperator):                        # lazy: a family or many properties, a
        C = res.covariance.conclude(positive=pos)                         # Covariance with the posterior's rule added
    else:                                                                 # the operator of coordinate covariance S: S W
        C = Covariance(Matrix(P.apply_metric(res.covariance.T).T, P, P, name="C"), name="C", selfadjoint=sa,
                       positive=pos)
    cl = [Claim("the property posterior T#mu^d (cells B10 then F10, Gaussian conditioning)", DERIVED, "inf:1095-1292"),
          Claim(f"noise independent of the model: {noise.name}", ASSERTED, "a modelling assumption"),
          Claim(f"computed in closed form from the data-space decomposition ({dec.describe()})", DERIVED, "plan 009 G3")]
    if prior.is_cylindrical:
        cl.append(Claim("the prior is cylindrical: a Gaussian measure on the finite-dimensional codomain, statements "
                        "about bounded functionals of the model only (EXP-028 DESIGN 1.9)", THEOREM, "inv:659-669"))
    if isinstance(res.covariance, LinearOperator):
        cl.append(Claim("the covariance is kept lazy (no p x p matrix); .variances holds the variances", DERIVED,
                        "a property family or many properties (plan 009 G4)"))
    seen = {c.statement for c in cl}
    cl += [c for c in getattr(res, "claims", ()) if c.statement not in seen]      # the decomposition's (plan 054, I17)
    out = GaussianMeasure(P, C, P(res.mean), name=f"{mpr.name}#({prior.name} | d)", _claims=cl)
    out.variances = np.asarray(res.variance if getattr(res, "variance", None) is not None else
                               np.diagonal(res.covariance), float)
    for k in ("variance_error", "std_error", "variance_bound", "std_bound"):         # the stated errors, delivered
        v = getattr(res, k, None)
        if v is not None:
            setattr(out, k, np.asarray(v, float))
    return out


def run(name, fields, **options):
    """Run a method on the given elements, refusing with what is missing."""
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(METHODS)}")
    m = METHODS[name]
    miss = m.missing(fields)
    if miss:
        raise Refused(f"{name} needs: " + "; ".join(miss), needs=miss)
    m.takes(options)
    return m.fn(**{k: v for k, v in fields.items() if v is not None}, **options)


def decomposition_of(mdr, middle=None, cache=None):
    """The data-space decomposition a method uses for this relation: its operator and noise (a Gaussian kernel's law,
    a noise set's shape, or the standard Gaussian of the data space for an operator)."""
    if _gaussian_kernel(mdr):
        G, noise = mdr.effective
    elif hasattr(mdr, "noise") and hasattr(mdr, "G"):
        from ..convex.feasible import ball_shape
        shape = ball_shape(mdr.noise, mdr.G.codomain)
        G, noise = mdr.G, (None if shape is None else shape[0])
    else:
        from ..relations.additive import point_noise
        point_noise(mdr, "the data-space decomposition of a problem")    # NoiseUnstated on a plain D (plan 071 P4)
        G, noise = mdr, None
    return _decomposition(cache, G, noise, middle)
