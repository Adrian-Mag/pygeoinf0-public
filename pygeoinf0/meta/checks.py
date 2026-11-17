"""Prior-predictive checks and coverage at a fixed truth (plan 009 G11 (3), (4); EXP-030 DESIGN.md G11, S21).

    pc = pg.meta.PredictiveCheck(dec, d, a, nu, bins=4)      dec = pg.decompose(G, noise, middle=prior)
    pc.chi2, pc.dof, pc.p_value                              chi^2_pred and P(chi^2_n >= chi2) (an Enclosure)
    pc.residual_chi2, pc.residual_law, pc.residual_p_value   the posterior residual and its weighted chi^2 law
    pc.bins                                                  the same per bin of rho^2 Lam (edges fixed before the data)
    cv = pg.meta.Coverage(dec, T, truth, a=a, nu=nu, level=0.95)        PLI credible intervals; gamma=...: SOLA's
    cv.probability, cv.bias, cv.sigma_post, cv.sigma_alt, cv.frequency(estimates)

Predictive check (appQ:74-88, pli:151-217; NOTES.md section 4). Under the prior predictive d ~ N(G m0 + b,
a^2 G S G* + nu^2 C) the whitened eigen coordinates z_k = delta_k / (a^2 Lam_k + nu^2)^(1/2) are iid N(0, 1): chi2 =
sum z_k^2 has the law chi^2_n, and the residual of the posterior mean in noise units, sum delta_k^2 nu^2 /
(a^2 Lam_k + nu^2)^2 = sum w_k z_k^2 with w_k = 1 / (1 + rho^2 Lam_k) (rho = a / nu), has the weighted chi^2 law
sum w_k Z_k^2 (pg.WeightedChiSquare: p-values as Enclosures). Binning the coordinates by their signal-to-noise ratio
lambda_k = rho^2 Lam_k (bins=: edges in lambda, or a number of equal-count bins of the spectrum) gives the same laws per
bin, since the z_k are independent; the edges depend on Lam, a and nu only, so they are fixed before the data. When a
and nu were chosen from the same datum (e.g. by the evidence), the laws are those of fixed hyperparameters and the
p-values are conditional on that choice: hyperparameters_from_datum=True records it (el:524-526).

Coverage at a fixed truth (appQ:311-319). For a truth m (or its clean datum G m + b and value T m) the estimate p_hat
is affine in d and the noise Gaussian, so p_hat ~ N(T m + eps, sigma_alt^2) exactly, with the bias eps = p_hat(clean
datum) - T m and sigma_alt^2 the variance of the propagated noise. The interval p_hat +- z sigma_post covers T m with
probability

    P = Phi((z sigma_post - eps) / sigma_alt) - Phi((-z sigma_post - eps) / sigma_alt).

PLI: p_hat the posterior mean, sigma_post the posterior std, sigma_alt^2 = nu^2 X C X* for the posterior-mean map
X = T S G* (G S G* + (nu / a)^2 C)^-1 (VarianceSplit's propagated noise). SOLA (gamma=): p_hat = X_gamma d,
sigma_post = sigma_alt = the noise std of X_gamma d. The thesis counted hits in 3 noise draws per truth (S21);
frequency(estimates) does that for given draws, as an illustration of P.

From lanes.py (merged here by plan 063):
One-at-a-time sensitivity lanes (plan 009 G11 (7); EXP-030 DESIGN.md G11).

    lane = pg.meta.Lane("prior", "length scale", {"l = 0.1": v1, "l = 0.2": v2}, baseline="l = 0.1")
    lane = pg.meta.Lane.run("noise", "noise scale", lambda s: result(s), {"x1": 1.0, "x2": 2.0}, baseline="x1")
    lane.values, lane.differences, lane.relative, lane.element

A lane changes one element of the problem between a baseline and its variants and records what a quantity does. The
element is named among the seven of the thesis (el: M, D, the MDR, P, the MPR, the prior state, the calibration) or
"realisation" (a numerical choice, not an element): "noise" is the MDR's noise law and "target" the MPR. That only the
named element changes is the user's statement (ASSERTED); a variant chosen from the datum is recorded as
data-dependent (el:524-526).

From shares.py (merged here by plan 063):
Leakage by direct-sum summand and the PLI variance split (plan 009 G11 (5), (6); EXP-030 DESIGN.md G11).

    lk = pg.meta.Leakage(dec, T, gamma, target=("h",))      dec = pg.decompose(G, noise) on a pg.DirectSum model space
    lk.components[key]                                      delivered, mismatch, target per summand
    lk.rrm, lk.rcm, lk.share                                resolution and contamination parts of the TRM, shares
    lk.apply(m)                                             X G_f m_f per summand: the delivered property by summand
    vs = pg.meta.VarianceSplit(dec, T, a, nu)               dec = pg.decompose(G, noise, middle=prior)
    vs.noise, vs.mismatch, vs.total, vs.noise_share, vs.mismatch_share

Leakage (sola:46-67, 202-269). On a direct sum M = (+)_f M_f with weights w_f, <x, y> = sum_f w_f <x_f, y_f>_f, the
delivered averaging kernel a = G* X* (the representer of A = X G) splits into its parts a_f, and every squared norm is
the sum of its summands' (orthogonality of the summands): ||t - a||^2 = sum_f ||t_f - a_f||^2. dec.sola(T, gamma,
fields=True) gives delivered_f = ||a_f||^2, mismatch_f = ||t_f - a_f||^2 and target_f = ||t_f||^2 from the one
decomposition (NOTES.md section 4: beta* D_gamma (U* W^(f) U) D_gamma beta). With the target summands F (declared, or
those where t_f != 0), relative to ||t||:
    rrm = (sum_{f in F} mismatch_f)^(1/2) / ||t||          the resolution misfit: the kernel's error on the target's
                                                           own summands
    rcm = (sum_{f not in F} mismatch_f)^(1/2) / ||t||      the contamination: a_f on the other summands (t_f = 0 there,
                                                           so mismatch_f = delivered_f)
    rrm^2 + rcm^2 = trm^2, and share[f] = mismatch_f / mismatch (the parts of the TRM^2).
These are the definitions of this module (DERIVED from the direct-sum norm); the thesis's RRM / RCM names are read as
these two parts of the TRM. For a model m, A m = sum_f X G (iota_f m_f) by linearity: apply(m) returns the parts (the
paired control of sola:279-296, estimate(full) - estimate(control) = the leaked part, without a second solve).

Variance split (verd:45). For the PLI posterior with prior N(m0, a^2 S) and noise N(b, nu^2 C), the posterior-mean map
X = T S G* (G S G* + (nu / a)^2 C)^-1 is SOLA's X_gamma in the metric S at gamma = nu^2 / a^2, and (Joseph's form of
the Gaussian update, valid for the optimal gain)

    T C_post T* = nu^2 X C X* + a^2 (T - X G) S (T - X G)*,     diagonal: variance = noise + mismatch,

the propagated noise and the prior-weighted mismatch ||t - G* X*||^2_(a^2 S). Both come from one dec.sola call on the
PLI decomposition; their sum equals dec.posterior's variance (tests/test_meta.py).
"""


import numpy as np

from ..core.claims import ASSERTED, DERIVED, Claim
from ..core.errors import Unsupported
from ..probability.chisquare import WeightedChiSquare
from .families import damping, scales_claim
from .record import Record


class PredictiveCheck(Record):
    """The prior-predictive check of a datum at (a, nu) (module docstring), from dec.predictive."""

    kind = "prior-predictive check"

    def __init__(self, dec, datum, a=1.0, nu=1.0, bins=None, prior_mean=None, hyperparameters_from_datum=False):
        r = dec.predictive(datum, a, nu, prior_mean)
        self.a, self.nu = float(a), float(nu)
        w = np.asarray(r.residual_weights, float)
        k, extra = len(r.z), len(w) - len(r.z)
        self.z, self.chi2, self.dof, self.trace_hat = r.z, r.chi2, r.dof, r.trace_hat
        self.residual_chi2, self.residual_weights = r.residual_chi2, w
        self.snr = np.r_[1.0 / w[:k] - 1.0, np.zeros(extra)]          # rho^2 Lam_k (0 for unseen coordinates)
        self._z2 = np.r_[r.z ** 2, np.zeros(extra)]
        self._res = np.r_[r.residual_terms, np.zeros(extra)]
        self.law = WeightedChiSquare([1.0], multiplicities=[self.dof], name="chi2_pred")
        self.residual_law = WeightedChiSquare(w, name="residual chi2")
        self._p = self._rp = None
        cl = [Claim("under the prior predictive the whitened coordinates z_k are iid N(0, 1): chi2 ~ chi^2_n, the "
                    "residual ~ sum w_k Z_k^2, w_k = 1 / (1 + rho^2 Lam_k)", DERIVED,
                    "d ~ N(G m0 + b, a^2 G S G* + nu^2 C) in the eigenbasis (appQ:74-88, pli:201-206; NOTES.md section 4)")]
        self.edges, self.bins = None, None
        if bins is not None:
            self._bin(bins)
            cl.append(Claim(f"{len(self.bins)} bins of rho^2 Lam with edges {np.array2string(self.edges, precision=3)}, "
                            "fixed before the data", DERIVED, "the edges depend on Lam, a and nu only"))
        if hyperparameters_from_datum:
            cl.append(Claim("the laws are those of fixed (a, nu): with (a, nu) chosen from this datum the p-values are "
                            "conditional on that choice", ASSERTED, "el:524-526"))
        self._record(True, "a check of the datum against its prior predictive",
                     [scales_claim(a, nu)] + cl + list(r.claims),
                     choice_from_datum=hyperparameters_from_datum, choice="the hyperparameters (a, nu)")

    def _bin(self, bins):
        s = self.snr
        if np.ndim(bins) == 0:
            e = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(bins) + 1)))
            e = np.r_[-np.inf, e[1:-1], np.inf]
        else:
            e = np.asarray(bins, float)
            if np.any(np.diff(e) <= 0):
                raise ValueError("bin edges are increasing")
        self.edges = e
        idx = np.clip(np.searchsorted(e, s, side="right") - 1, 0, max(len(e) - 2, 0))
        self.bins = []
        for b in range(len(e) - 1):
            sel = idx == b
            if not sel.any():
                continue
            self.bins.append(Bin((float(e[b]), float(e[b + 1])), int(sel.sum()), float(self._z2[sel].sum()),
                                 float(self._res[sel].sum()), self.residual_weights[sel]))

    @property
    def p_value(self):
        """P(chi^2_n >= chi2) under the prior predictive (an Enclosure)."""
        if self._p is None:
            self._p = self.law.sf(self.chi2)
        return self._p

    @property
    def residual_p_value(self):
        """P(sum w_k Z_k^2 >= residual_chi2) (an Enclosure, pg.WeightedChiSquare)."""
        if self._rp is None:
            self._rp = self.residual_law.sf(self.residual_chi2)
        return self._rp

    def _show(self):
        return ["chi2", "dof", "trace_hat", "residual_chi2"]


class Bin:
    """One bin of a PredictiveCheck: snr (its edges in rho^2 Lam), dof, chi2 and law (chi^2_dof), residual_chi2 and
    residual_law (sum w_k Z_k^2 over the bin); p_value and residual_p_value (Enclosures) are computed on first use."""

    def __init__(self, snr, dof, chi2, residual_chi2, weights):
        self.snr, self.dof, self.chi2, self.residual_chi2 = snr, dof, chi2, residual_chi2
        self.law = WeightedChiSquare([1.0], multiplicities=[dof])
        self.residual_law = WeightedChiSquare(weights)
        self._p = self._rp = None

    @property
    def p_value(self):
        if self._p is None:
            self._p = self.law.sf(self.chi2)
        return self._p

    @property
    def residual_p_value(self):
        if self._rp is None:
            self._rp = self.residual_law.sf(self.residual_chi2)
        return self._rp

    def __repr__(self):
        return f"Bin(rho^2 Lam in [{self.snr[0]:.3g}, {self.snr[1]:.3g}), dof {self.dof}, chi2 {self.chi2:.6g})"


class Coverage(Record):
    """The probability that the interval estimate +- z sigma_post covers the value at a fixed truth (module docstring):
    probability, bias, sigma_post, sigma_alt, z and the value, one per property. PLI by default (a, nu; dec built with
    middle=prior); SOLA with gamma=. The truth is a model element, or clean= (its datum G m + b) and value= (T m)."""

    kind = "coverage at a fixed truth"

    def __init__(self, dec, T, truth=None, *, a=1.0, nu=1.0, gamma=None, level=0.95, z=None, prior_mean=None,
                 clean=None, value=None, pairings=None, generating=None):
        from scipy.special import ndtr, ndtri
        if truth is not None:
            clean = np.asarray(dec.G(truth).values, float)
            if dec.noise.offset is not None:
                clean = clean + dec.noise.offset
            value = np.asarray(T(truth).values, float)
        if clean is None or value is None:
            raise ValueError("give the truth (a model element) or clean= and value=")
        self.level = None if z is not None else float(level)
        self.z = float(z) if z is not None else float(ndtri(0.5 + 0.5 * float(level)))
        if gamma is not None:
            r = dec.sola(T, float(gamma), clean, pairings=pairings)
            est, self.sigma_post = r.estimate, r.std
            self.sigma_alt, self.method = r.std, "SOLA"
            claims = [damping("sola").claim({"gamma": float(gamma)})] + list(r.claims)
        else:
            a, nu = float(a), float(nu)
            post = dec.posterior(T, clean, a, nu, prior_mean=prior_mean, pairings=pairings)
            est, self.sigma_post = post.mean, post.std
            self.sigma_alt = nu * dec.sola(T, (nu / a) ** 2, pairings=pairings).std
            claims, self.method = [scales_claim(a, nu)] + list(post.claims), "PLI"
        if generating is not None:                    # a data-generating noise law other than the MDR's (appQ)
            g_ = float(gamma) if gamma is not None else (nu / a) ** 2
            X = np.atleast_2d(np.asarray(dec.sola_operator(T, g_, pairings=pairings).matrix(), float))
            Cg = np.asarray(generating, float)
            var = (X * X) @ Cg if Cg.ndim == 1 else np.einsum("pi,ij,pj->p", X, Cg, X)
            self.sigma_alt = np.sqrt(np.maximum(var, 0.0))
            claims.append(Claim("the generating noise law is given (generating=): sigma_alt^2 = X C_gen X* for the "
                                "estimator's map X; the interval's sigma_post is the method's own", ASSERTED,
                                "generating="))
        self.value = np.asarray(value, float)
        self.bias = np.asarray(est, float) - self.value
        h = self.z * self.sigma_post
        self.probability = ndtr((h - self.bias) / self.sigma_alt) - ndtr((-h - self.bias) / self.sigma_alt)
        self._record(False, "a fixed truth and the noise law; no datum", [Claim(
            f"{self.method}: P(|p_hat - T m| <= z sigma_post) = Phi((z sigma_post - eps) / sigma_alt) - Phi((-z "
            f"sigma_post - eps) / sigma_alt), z = {self.z:.6g}", DERIVED, "p_hat is affine in d and the noise is "
            "Gaussian (the MDR's law): p_hat ~ N(T m + eps, sigma_alt^2) exactly (appQ:311-319)")] + claims)

    def frequency(self, estimates):
        """The fraction of given estimates (draws x properties) within z sigma_post of the value, and its binomial
        standard error under the exact probability: (fraction, std)."""
        e = np.atleast_2d(np.asarray(estimates, float))
        hit = np.abs(e - self.value) <= self.z * self.sigma_post
        return hit.mean(axis=0), np.sqrt(self.probability * (1 - self.probability) / len(e))

    def _show(self):
        return ["probability", "z"]


# ---------------------------------------------------------------- from lanes.py (plan 063)
def _elements():
    """The elements a lane may change: the seven of pg.Problem (inference/problem.py FIELDS without the datum, plan 070
    P2), their upper-case and part aliases, and the realisation (numerics; not an element)."""
    from ..inference.problem import ELEMENTS as SEVEN, FIELDS
    out = {k: SEVEN[k] for k in FIELDS if k != "datum"}
    out.update({"MDR": SEVEN["mdr"], "noise": f"{SEVEN['mdr']} (its noise law)", "MPR": SEVEN["mpr"],
                "target": f"{SEVEN['mpr']} (the target)", "realisation": "the realisation (numerics; not an element)"})
    return out


ELEMENTS = _elements()


class Lane(Record):
    """A one-at-a-time sensitivity lane (module docstring): element, name, values {label: value}, baseline (a label),
    differences and relative changes against the baseline."""

    kind = "sensitivity lane"

    def __init__(self, element, name, values, baseline=None, used_datum=False, chosen_from_datum=False, note=""):
        if element not in ELEMENTS:
            raise ValueError(f"element: one of {sorted(ELEMENTS)}")
        if not values:
            raise ValueError("a lane has at least one value")
        self.element, self.changes, self.name = element, ELEMENTS[element], name
        self.values = {k: np.asarray(v, float) for k, v in values.items()}
        self.baseline = next(iter(self.values)) if baseline is None else baseline
        if self.baseline not in self.values:
            raise ValueError("the baseline is one of the labels")
        b = self.values[self.baseline]
        self.differences = {k: v - b for k, v in self.values.items() if k != self.baseline}
        with np.errstate(divide="ignore", invalid="ignore"):
            self.relative = {k: d / np.abs(b) for k, d in self.differences.items()}
        self._record(used_datum, "stated by the lane's author" + (f" ({note})" if note else ""), [Claim(
            f"one at a time: only {ELEMENTS[element]} changes between the baseline {self.baseline!r} and "
            f"{len(self.differences)} variant(s) ({name})", ASSERTED, "the lane's declaration", key="lane")],
            choice_from_datum=chosen_from_datum, choice=f"the variants of {name}")

    @classmethod
    def run(cls, element, name, compute, settings, baseline=None, **kw):
        """The lane of compute(setting) over {label: setting} (each call one variant)."""
        return cls(element, name, {k: compute(s) for k, s in settings.items()}, baseline=baseline, **kw)

    def _show(self):
        return ["element", "name", "baseline"]


# ---------------------------------------------------------------- from shares.py (plan 063)
class Leakage(Record):
    """The delivered averaging kernel split by direct-sum summand (module docstring): components, rrm, rcm, trm,
    share (dicts or arrays per property, one row per gamma for an array of gamma). target: the target's summands."""

    kind = "leakage by summand"

    def __init__(self, dec, T, gamma, target=None):
        r = dec.sola(T, gamma, fields=True)
        self.dec, self.T, self.gamma = dec, T, r.gamma
        self.components = r.components
        tt = np.asarray(r.target, float)
        keys = list(r.components)
        if target is None:
            target = [k for k in keys if np.any(np.asarray(r.components[k]["target"]) > 1e-12 * tt)]
            how = "the summands where the target is not zero (||t_f||^2 > 1e-12 ||t||^2)"
        else:
            target = [target] if isinstance(target, str) else list(target)
            if any(k not in keys for k in target):
                raise ValueError(f"target summands among {keys}")
            how = "declared"
        self.target = target
        mis = {k: np.asarray(r.components[k]["mismatch"], float) for k in keys}
        total = sum(mis.values())
        res = sum(mis[k] for k in target) if target else 0.0 * total
        con = total - res
        self.rrm = np.sqrt(np.maximum(res, 0.0) / tt)
        self.rcm = np.sqrt(np.maximum(con, 0.0) / tt)
        self.trm = np.sqrt(np.maximum(total, 0.0) / tt)
        self.share = {k: mis[k] / total for k in keys}
        self.mismatch = total
        self._record(False, "the delivered kernel depends on G, the noise covariance, T and gamma only", [
            Claim("||t - a||^2 = sum_f ||t_f - a_f||^2 over the summands; rrm^2 + rcm^2 = trm^2", DERIVED,
                  "the norm of a weighted direct sum (module docstring)"),
            Claim(f"target summands {target}", DERIVED if how != "declared" else ASSERTED, how)] + list(r.claims))

    def apply(self, m, G=None):
        """{summand: X G (iota_f m_f)} for a model element m of the direct sum and one gamma (the delivered property
        by summand; the parts sum to X G m). G defaults to the decomposition's."""
        if np.ndim(self.gamma):
            raise Unsupported("apply needs one gamma")
        G = self.dec.G if G is None else G
        M = G.domain
        X = self.dec.sola_operator(self.T, float(self.gamma))
        return {k: np.asarray(X(G(M.embed(k, M.part(m, k)))).values, float) for k in M.names}

    def _show(self):
        return ["gamma", "target"]


class VarianceSplit(Record):
    """The PLI posterior variance split into propagated noise and prior-weighted mismatch (module docstring), per
    property: noise, mismatch, total, noise_share, mismatch_share. dec: the PLI decomposition (middle = the prior
    covariance S at a = 1)."""

    kind = "PLI variance split"

    def __init__(self, dec, T, a=1.0, nu=1.0, pairings=None):
        a, nu = float(a), float(nu)
        if not (a > 0 and nu > 0):
            raise ValueError("a > 0 and nu > 0")
        r = dec.sola(T, (nu / a) ** 2, pairings=pairings)
        self.a, self.nu = a, nu
        self.noise = nu * nu * np.asarray(r.variance, float)
        self.mismatch = a * a * np.maximum(np.asarray(r.mismatch, float), 0.0)
        self.total = self.noise + self.mismatch
        self.noise_share, self.mismatch_share = self.noise / self.total, self.mismatch / self.total
        self._record(False, "the posterior variance does not depend on the datum", [scales_claim(a, nu), Claim(
            "posterior variance = nu^2 X C X* + a^2 ||t - G* X*||^2_S, X the posterior-mean map (SOLA at gamma = "
            f"nu^2 / a^2 = {(nu / a) ** 2:.6g} in the metric of the prior covariance)", DERIVED,
            "Joseph's form of the Gaussian update for the optimal gain; NOTES.md section 4"),
            Claim("the decomposition's middle operator is the prior covariance at a = 1" if dec.middle is not None else
                  "no middle operator: the prior covariance is a^2 times the model metric's identity",
                  DERIVED, dec.describe())] + list(r.claims))

    def _show(self):
        return ["a", "nu", "noise_share"]
