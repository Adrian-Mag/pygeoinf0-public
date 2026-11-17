"""The closed forms of a data-space decomposition (plan 009 G3; EXP-030 NOTES.md section 4): SOLA for arrays of
gamma, the PLI posterior for arrays of (a, nu), BGP with an ellipsoid prior, feasibility, the evidence, the
damping for a misfit target (the numerics of the discrepancy principle) and the prior-predictive check. ClosedForms
is the part of DataDecomposition that uses the eigenbasis; solvers/dataspace/decomposition.py builds it."""
import math

import numpy as np

from ...core.claims import CHECKED, DERIVED, THEOREM, Claim, Claims
from ...core.errors import Unsupported
from ...spaces.base import Euclidean
from .whitening import _representers

RCOND = 1e-15           # eigenvalues below RCOND * (their block's max, about 5 eps) count as zero where 1 / Lam appears
# (feasibility: the count is in its claims; plan 054 P2)


class Result:
    """What a closed form returns: named arrays (attributes), the hyperparameters and claims (which decomposition and
    which formula)."""

    def __init__(self, kind, claims, **values):
        self.kind, self.claims = kind, Claims(claims)
        self._names = list(values)
        for k, v in values.items():
            setattr(self, k, v)

    def __repr__(self):
        def show(v):
            a = np.asarray(v) if not isinstance(v, dict) else None
            if a is None:
                return f"{{{', '.join(map(str, v))}}}"
            return f"{a.item():.6g}" if a.size == 1 else f"array{a.shape}"
        vals = ", ".join(f"{k}={show(getattr(self, k))}" for k in self._names if getattr(self, k) is not None)
        return f"{self.kind}({vals})"


class ClosedForms:
    """The methods of DataDecomposition that evaluate closed forms (module docstring); they need lam, _coef,
    _back, _spec, _delta, targets and the claims of the decomposition."""

    # ---------------------------------------------------------------------------------------------- SOLA
    def sola(self, T=None, gamma=1.0, datum=None, fields=False, pairings=None, refuse=True):
        """SOLA X = T G* (G G* + gamma C)^-1 (with S: T S G* (G S G* + gamma C)^-1) for every gamma of an array, in
        closed form (module docstring): estimate (if a datum is given), std and variance of X d's noise, the delivered
        mismatch ||t - G* X*||^2 (in the metric S) with its stated error mismatch_error, the delivered norm
        ||G* X*||^2, their derivatives in gamma (d mismatch / d variance = -gamma), and with fields=True the same per
        summand of a direct-sum model space. The mismatch is a difference of nearly equal numbers where the target is
        well resolved: where the closed form's stated error exceeds TOL it is formed again from the residual element,
        and what is left is warned of, or refused where no digit is correct (refuse=False: warned of;
        solvers/dataspace/cancellation.py)."""
        tg = self.targets(T, pairings)
        if getattr(tg, "Phi", None) is not None:
            if fields:
                return self._sola_fields_factored(tg, gamma, datum)
            from .factored import sola as factored_sola
            return factored_sola(self, tg, gamma, datum)
        g = np.asarray(gamma, float)
        scalar = g.ndim == 0
        g = np.atleast_1d(g).ravel()
        if np.any(~(g > 0)):
            raise ValueError("gamma must be positive")
        p = tg.cross.shape[1]
        cols = tg.cross if datum is None else np.c_[tg.cross, self._delta(datum)]
        lam, C, _ = self._spec(cols)
        beta = C[:, :p]
        b2, tt = beta ** 2, tg.diag()
        est, var, dlv, mis, dvar, dmis, crs = ([] for _ in range(7))
        step = max(1, int(2e7 // max(len(lam), 1)))
        for a in range(0, len(g), step):
            R = 1.0 / (lam[:, None] + g[None, a:a + step])
            R2 = R * R
            if datum is not None:
                est.append(R.T @ (beta * C[:, p][:, None]))
            S1, S2, S3 = R.T @ b2, R2.T @ b2, (R2 * R).T @ b2
            d = (R2 * lam[:, None]).T @ b2
            var.append(S2)
            crs.append(S1)
            dlv.append(d)
            mis.append(tt[None, :] - 2.0 * S1 + d)
            dvar.append(-2.0 * S3)
            dmis.append(2.0 * g[a:a + step, None] * S3)
        from .cancellation import sola_mismatch
        mis, mis_err = sola_mismatch(self, tg, g, lam, beta, tt, np.concatenate(mis), refuse)
        mis, mis_err = [mis], [mis_err]
        cat = (lambda L: (np.concatenate(L)[0] if scalar else np.concatenate(L)))
        comps = self._sola_fields(tg, g, beta, scalar) if fields else None
        return Result("SOLA", self._result_claims("SOLA in closed form in gamma", "NOTES.md section 4 (EXP-030)"),
                      gamma=g[0] if scalar else g, estimate=cat(est) if est else None, std=np.sqrt(cat(var)),
                      variance=cat(var), mismatch=cat(mis), mismatch_error=cat(mis_err), delivered=cat(dlv), target=tt,
                      cross=cat(crs), d_variance=cat(dvar),
                      d_mismatch=cat(dmis), components=comps)

    def sola_operator(self, T=None, gamma=1.0, pairings=None):
        """X: D -> P (a Matrix) for one gamma: X^T = F^T (W + gamma)^-1 F <g, S t>."""
        from ...operators.linear import Matrix
        tg = self.targets(T, pairings)
        g = float(gamma)
        if getattr(tg, "Phi", None) is not None and not self.partial:     # a factored family: X = Phi X_tau
            Xt = self._back(self._beta(tg) / (self.lam[:, None] + g))
            Xtau = Matrix(Xt.T, domain=self.D, codomain=Euclidean(Xt.shape[1]), name="X_tau")
            X = Matrix(tg.Phi, domain=Xtau.codomain, codomain=tg.T.codomain, name="Phi") @ Xtau
            X._claims.append(Claim(f"X = Phi X_tau, X_tau = tau S G* (G S G* + {g:g} C)^-1 (never p x n)", DERIVED,
                                   self.describe()))
            return X
        if not self.partial:
            Xt = self._back(self._beta(tg) / (self.lam[:, None] + g))
        else:
            wt, U = self._white[0], self._U[0]
            Fc = wt.fwd(tg.cross)
            H = U.T @ Fc
            Xt = wt.back(U @ (H / (self.lam[:, None] + g)) + (Fc - U @ H) / g)
        P = tg.T.codomain if tg.T is not None else Euclidean(Xt.shape[1])
        X = Matrix(Xt.T, domain=self.D, codomain=P, name="X")
        X._claims.append(Claim(f"X = T S G* (G S G* + {g:g} C)^-1 from the data-space decomposition", DERIVED,
                               self.describe()))
        return X

    def _sola_fields_factored(self, tg, gamma, datum):
        from .factored import fields, sola as factored_sola
        res = factored_sola(self, tg, gamma, datum)
        if self.middle is not None or self.partial:
            raise Unsupported("per-summand components need the model metric (no middle) and a complete decomposition")
        res.components = fields(self, tg, gamma, self._beta(tg), self.G, tg.T)
        if res.components is None:
            raise Unsupported("per-summand components of a factored family need separable functionals (structured Grams)")
        return res

    def _posterior_many(self, tg, datum, a, nu, prior_mean):
        from .factored import posterior_many
        return posterior_many(self, tg, datum, a, nu, prior_mean)

    def _sola_fields(self, tg, g, beta, scalar):
        from ...spaces.direct_sum import DirectSum
        M = getattr(self.G, "domain", None)
        if self.middle is None and tg.T is not None and not self.partial:
            from .factored import fields
            out = fields(self, tg, g[0] if scalar else g, beta, self.G, tg.T)
            if out is not None:
                return out
        if not isinstance(M, DirectSum) or self.middle is not None or tg.T is None or self.partial:
            raise Unsupported("per-summand components need a direct-sum model space, the model metric (no middle), a "
                              "property map and a complete decomposition")
        Rg, Rt = _representers(self.G), _representers(tg.T)
        out = {}
        for k, key in enumerate(M.names):
            H, w = M.summands[k], M.weights[k]
            pg_, pt = M.part(Rg, key), M.part(Rt, key)
            Kf = w * np.atleast_2d(H.inner(pg_, pg_))
            cf = w * np.atleast_2d(H.inner(pg_, pt))
            tf = w * np.diag(np.atleast_2d(H.inner(pt, pt)))
            dl, ms = [], []
            for gv in g:
                x = self._back(beta / (self.lam[:, None] + gv))          # the plain weights of G* X* (n x p)
                d = np.einsum("ip,ij,jp->p", x, Kf, x)
                dl.append(d)
                ms.append(tf - 2.0 * np.einsum("ip,ip->p", cf, x) + d)
            out[key] = dict(delivered=dl[0] if scalar else np.array(dl), mismatch=ms[0] if scalar else np.array(ms),
                            target=tf)
        return out

    # ---------------------------------------------------------------------------------------------- PLI
    def posterior(self, T=None, datum=None, a=1.0, nu=1.0, prior_mean=None, pairings=None, refuse=True):
        """The property posterior for prior N(m0, a^2 S) and noise N(offset, nu^2 C) for arrays of (a, nu)
        (broadcast): mean, covariance, and the prior covariance a^2 T S T*. The mean depends on (a, nu) only through
        rho = a / nu (pli:255); the covariance scales with a^2. The variances carry their stated error
        (variance_error, std_error): a^2 tt - a^4 beta^T S^-1 beta loses digits by subtraction for a property the data
        resolve well under a wide prior, so where that error exceeds TOL the covariance is formed again without the
        subtraction (solvers/dataspace/cancellation.py); what is left is warned of, or refused where no digit is correct
        (refuse=False: warned of)."""
        tg = self.targets(T, pairings)
        if getattr(tg, "Phi", None) is not None:
            from .factored import posterior as factored_posterior
            return factored_posterior(self, tg, datum, a, nu, prior_mean)
        if tg.tt is None:
            return self._posterior_many(tg, datum, a, nu, prior_mean)
        A, N = np.broadcast_arrays(np.asarray(a, float), np.asarray(nu, float))
        shape = A.shape
        A, N = A.ravel(), N.ravel()
        p = tg.cross.shape[1]
        lam, C, _ = self._spec(np.c_[tg.cross, self._delta(datum, prior_mean)])
        beta, delta = C[:, :p], C[:, p]
        S = (A ** 2)[None, :] * lam[:, None] + (N ** 2)[None, :]          # a^2 Lam + nu^2
        mean = (A ** 2)[:, None] * (beta.T @ (delta[:, None] / S)).T
        if prior_mean is not None and tg.T is not None:
            mean = mean + tg.T(prior_mean).values[None, :]
        cov = (A ** 2)[:, None, None] * tg.tt[None] - (A ** 4)[:, None, None] * np.einsum("kp,kh,kq->hpq", beta, 1.0 / S,
                                                                                            beta)
        cov = 0.5 * (cov + cov.transpose(0, 2, 1))
        from .cancellation import posterior_covariance, std_error as cancel_std
        cov, var_err = posterior_covariance(self, tg, A, N, lam, beta, cov, refuse)
        prior = (A ** 2)[:, None, None] * tg.tt[None]
        out = Result("posterior", self._result_claims(
            "the property posterior in closed form for each (a, nu)", "Gaussian conditioning in the eigenbasis, NOTES.md "
            "section 4 (EXP-030)"), a=A.reshape(shape), nu=N.reshape(shape), mean=mean.reshape(shape + (p,)),
            covariance=cov.reshape(shape + (p, p)), prior_covariance=prior.reshape(shape + (p, p)))
        out.std = np.sqrt(np.maximum(np.diagonal(out.covariance, axis1=-2, axis2=-1), 0.0))
        out.variance_error = var_err.reshape(shape + (p,))
        out.std_error = cancel_std(out.std ** 2, out.variance_error)
        out.T = tg.T
        if getattr(self, "error", None) is not None:          # the randomised route (solvers/dataspace/partial.py)
            # with R = (rho^2 W_k + I)^-1, E = W - W_k of either sign (plan 054 P3): |w^T [R - (rho^2 W + I)^-1] w| <=
            # rho^2 ||E|| ||R w||^2 / (1 - rho^2 ||E||) (partial.two_sided), so the variance error (a^4 / nu^2) times
            # that, and R = nu^2 / S on the eigen coordinates (extended spectrum): a^6 ||E|| sum_k beta_k^2 / S_k^2
            # times the two-sided factor
            from .partial import std_error, two_sided
            bound = ((A ** 6) * two_sided(self.error, (A / N) ** 2))[:, None] * self.error * \
                np.einsum("kp,kh->hp", beta * beta, 1.0 / (S * S))
            out.variance_bound = bound.reshape(shape + (p,))
            out.std_bound = std_error(out.std ** 2, out.variance_bound)
            out.claims = Claims(list(out.claims) + [Claim(
                f"posterior variances from the rank-{len(self.lam)} Nystrom approximation: each within variance_bound "
                f"of the exact one, on either side (largest std bound {float(np.max(out.std_bound, initial=0.0)):.3g})",
                CHECKED, f"||W - W_k|| <= {self.error:.3g} with probability >= 1 - 1e-10 (solvers/dataspace/partial.py), the "
                         "resolvent identity per functional; W_k <= W holds only to rounding in double (plan 054 P3), "
                         "so the bound is two-sided")])
        return out

    # ---------------------------------------------------------------------------------------------- BGP
    def ball_problem(self, e, r, rho, dependent=False, centre=None):
        """The BGP problem sup <c, u> over {||u||_(S^-1) <= r} ∩ {||F(e - G u)|| <= rho} for the plain residual e
        (d - offset - G m0), solved per direction in the eigenbasis (optimisers/dual/spectral.py); dependent: the data declared
        exactly dependent at the rounding floor (FeasibleSet.dependent); centre: the prior centre m0 that e subtracts
        (its norm enters the datum's stated error where the kernels state errors, optimisers/dual/exactness.py)."""
        if self.partial:
            raise Unsupported("BGP needs a complete decomposition (a partial one does not bound the remaining spectrum)")
        from ...optimisers.dual import SpectralBall
        sp = SpectralBall(self, np.asarray(e, float), r, rho, dependent)
        if centre is not None and self.G is not None:
            sp._m0n = float(self.G.domain.norm(centre))
        return sp

    def ellipsoid_bound(self, T=None, a=1.0, r=1.0, datum=None, q=None, centre=None, pairings=None):
        """sup <q, T m>_P over {||m - m0||_(S^-1) <= a} ∩ {||F(d - offset - G m)|| <= r}: a Bound with certificate
        (EXACT when primal and dual agree to core.precision.TOL). q: one direction (p,), several (J, p), or None for every
        coordinate (a list of (lower, upper) per property)."""
        from ...optimisers.dual.exactness import direction_error
        tg = self.targets(T, pairings)
        prob = self.ball_problem(self._delta(datum, centre), a, r, centre=centre)
        beta = self._beta(tg)
        P = tg.T.codomain if tg.T is not None else Euclidean(tg.cross.shape[1])
        Tm0 = tg.T(centre).values if (centre is not None and tg.T is not None) else np.zeros(P.dim)

        factored = getattr(tg, "Phi", None) is not None
        if factored:
            from .quadratic import rowquad
            extra = np.maximum(tg.tt_diag - rowquad(tg.Phi, tg.tt), 0.0)

        def one(w):
            if factored:                                  # T = Phi tau: the direction's block coordinates Phi^T w
                z = tg.Phi.T @ w
                return prob.bound(beta @ z, float(z @ tg.tt @ z + extra @ (w * w)), float(w @ Tm0), None,
                                  direction_error(tg, w, T))
            ww = float(w @ tg.tt @ w) if tg.tt is not None else float(w @ tg.tt_op(P(w)).values)
            return prob.bound(beta @ w, ww, float(w @ Tm0), None, direction_error(tg, w, T))
        if q is None:
            out = []
            for k in range(P.dim):
                e = np.zeros(P.dim)
                e[k] = 1.0
                out.append((one(-e).negated(), one(e)))
            return out
        Q = np.asarray(q, float)
        if Q.ndim == 1:
            return one(P.apply_metric(Q))
        return [one(P.apply_metric(v)) for v in Q]

    def ellipsoid_intervals(self, T=None, a=1.0, r=1.0, datum=None, centre=None, pairings=None):
        """[lower, upper] of every property of T over the same set as ellipsoid_bound, vectorised over the directions
        (optimisers/dual/spectral.py; a factored family never forms its p x n pairings): a Result with lower, upper and
        the certificate statuses."""
        from .factored import ellipsoid_intervals
        return ellipsoid_intervals(self, self.targets(T, pairings), a, r, datum, centre)

    def feasibility(self, datum, a=None, r=None, centre=None):
        """rho_min(a) = min ||F(d - offset - G m)|| over {||m - m0||_(S^-1) <= a} for an array of a, and a*(r), the
        least a with rho_min(a) <= r, for an array of r (inf: no prior radius fits; 0: m0 fits). The trust-region
        secular equation in the eigenbasis (appC:60-83)."""
        if self.partial:
            raise Unsupported("feasibility needs a complete decomposition")
        delta = self._coef(self._delta(datum, centre))
        lam = self.lam
        pos = self.positive(RCOND)
        f0 = float(delta[~pos] @ delta[~pos])
        dd = float(delta @ delta)
        q0 = float(np.sum(delta[pos] ** 2 / lam[pos]))
        L, dp = np.maximum(lam, 0.0), delta          # for mu > 0 no 1 / Lam appears: every eigenvalue counts
        scale = float(np.median(lam[pos])) if pos.any() else 1.0

        def misfit2(mu):
            return float(np.sum(dp ** 2 * (mu / (L + mu)) ** 2))

        def radius2(mu):
            return float(np.sum(L * dp ** 2 / (L + mu) ** 2))
        out = {}
        if a is not None:
            av = np.atleast_1d(np.asarray(a, float))
            rm = []
            for x in av:
                if q0 <= x * x:
                    rm.append(math.sqrt(f0))
                else:
                    mu = _root(lambda m: radius2(m) - x * x, decreasing=True, scale=scale)
                    rm.append(math.sqrt(misfit2(mu)))
            out["rho_min"] = rm[0] if np.ndim(a) == 0 else np.array(rm)
        if r is not None:
            rv = np.atleast_1d(np.asarray(r, float))
            ast = []
            for x in rv:
                if f0 > x * x:
                    ast.append(math.inf)
                elif dd <= x * x:
                    ast.append(0.0)
                else:
                    mu = _root(lambda m: misfit2(m) - x * x, decreasing=False, scale=scale)
                    ast.append(math.sqrt(radius2(mu)))
            out["a_star"] = ast[0] if np.ndim(r) == 0 else np.array(ast)
        zero, below, n = self.zero_count(RCOND)
        return Result("feasibility", self._result_claims(
            "rho_min(a) and a*(r) from the trust-region secular equation", "the least misfit over the prior set is "
            "y = sqrt(Lam) delta / (Lam + mu) with ||y|| = a (appC:60-83, NOTES.md section 4)") + [Claim(
            f"{zero} of {n} eigenvalues <= {RCOND:g} x their block's largest counted as zero in the floor and q0 (where "
            "1 / Lam appears)" + (f"; {below} kept at most eps n_block x it (no correct digit)" if below else ""),
            DERIVED, "plan 054 P2")],
            rho_min=out.get("rho_min"), a_star=out.get("a_star"), floor=math.sqrt(f0))

    # ---------------------------------------------------------------------------------------------- evidence etc.
    def evidence(self, datum, a=1.0, nu=1.0, block_scale=None, prior_mean=None, gradient=False):
        """log p(d | a, nu) for d = G m + offset + e, m ~ N(m0, a^2 S), e ~ N(0, nu^2 C) (arrays of a, nu broadcast);
        block_scale multiplies nu per block of the decomposition (noise laws with one multiplier per block, e.g. per
        degree). With gradient=True also d/da, d/dnu (and d/d block_scale for a single (a, nu))."""
        lam, C, extra = self._spec(self._delta(datum, prior_mean))
        A, N = np.broadcast_arrays(np.asarray(a, float), np.asarray(nu, float))
        shape = A.shape
        A, N = A.ravel(), N.ravel()
        bs = np.ones(len(lam)) if block_scale is None else np.r_[np.asarray(block_scale, float)[self._block_of],
                                                                 np.ones(len(lam) - len(self._block_of))]
        if block_scale is not None and self.partial:
            raise Unsupported("block_scale needs a complete decomposition")
        nk2 = (N ** 2)[None, :] * (bs ** 2)[:, None]                          # per component, per (a, nu)
        S = (A ** 2)[None, :] * lam[:, None] + nk2
        d2 = (C ** 2)[:, None]
        logp = -0.5 * (np.sum(np.log(S) + d2 / S, axis=0) + extra * np.log(N ** 2) + self.logdet_noise()
                       + self.n * math.log(2 * math.pi))
        out = dict(log_evidence=logp.reshape(shape) if shape else float(logp[0]))
        if gradient:
            Q = 1.0 / S - d2 / S ** 2
            ga = -A * np.sum(lam[:, None] * Q, axis=0)
            gn = -np.sum(nk2 / N[None, :] * Q, axis=0) - extra / N
            out["d_a"] = ga.reshape(shape) if shape else float(ga[0])
            out["d_nu"] = gn.reshape(shape) if shape else float(gn[0])
            if block_scale is not None and len(A) == 1:
                g = -(nk2[:, 0] / bs) * Q[:, 0]
                out["d_block_scale"] = np.bincount(self._block_of, weights=g[:len(self._block_of)],
                                                   minlength=len(self._rows))
        return Result("evidence", self._result_claims(
            "the Gaussian evidence: -2 log p = sum log(a^2 Lam + nu^2) + delta^2 / (a^2 Lam + nu^2) + log det C + "
            "n log 2 pi", "d ~ N(G m0 + offset, a^2 K + nu^2 C) in the eigenbasis (NOTES.md section 4)"), **out)

    def damping_for_misfit(self, datum, target):
        """The damping alpha with sum delta_k^2 (alpha / (Lam_k + alpha))^2 = target (a stated target: the discrepancy
        principle's choice of it is pg.meta.discrepancy), as solvers.damping_for_misfit computes it: a float with
        claims."""
        from ..choice import Alpha, solve_damping
        n = float(target)
        lam, C, _ = self._spec(self._delta(datum))
        alpha, _ = solve_damping(lam, C ** 2, n)
        return Alpha(alpha, Claims([Claim("damping for the misfit target solved exactly (the root bracketed above the "
                                          "rounding floor; solvers.solve_damping)",
                                          THEOREM if not self.partial else CHECKED,
                                          f"the data-space decomposition ({self.describe()}), bisection in log alpha")]
                                   + list(self._claims)))

    def block_coordinates(self, rows):
        """The eigen coordinates (indices into lam and the per-coordinate results) of the blocks that contain the given
        data rows; the rows must make up whole blocks (a ValueError otherwise)."""
        rows = np.unique(np.asarray(rows, int))
        if not hasattr(self, "_row_block"):
            self._row_block = np.empty(self.n, int)
            for b, p in enumerate(self._rows):
                self._row_block[p] = b
        bs = np.unique(self._row_block[rows])
        if sum(len(self._rows[b]) for b in bs) != len(rows):
            raise ValueError("the rows do not make up whole blocks of the decomposition")
        return np.concatenate([np.arange(self.offsets[b], self.offsets[b + 1]) for b in bs]) if len(bs) else \
            np.zeros(0, int)

    def predictive(self, datum, a=1.0, nu=1.0, prior_mean=None):
        """The prior-predictive check at (a, nu): z_k = delta_k / (a^2 Lam_k + nu^2)^(1/2), iid N(0, 1) under the
        prior predictive; chi2 = sum z^2 (n degrees of freedom); trace_hat = tr H = sum rho^2 Lam / (1 + rho^2 Lam)
        (rho = a / nu); the residual of the posterior mean in noise units, residual_chi2, whose law is
        sum w_k Z_k^2 with weights w_k = 1 / (1 + rho^2 Lam_k) (pli:201-206; a weighted chi^2), mean n - tr H;
        residual_terms its parts per eigen coordinate (sums over blocks: block_coordinates)."""
        lam, C, extra = self._spec(self._delta(datum, prior_mean))
        a, nu = float(a), float(nu)
        S = a * a * lam + nu * nu
        z = C / np.sqrt(S)
        w = nu * nu / S
        return Result("predictive", self._result_claims(
            "the prior-predictive law of the whitened data and of the posterior residual", "appQ:74-88, pli:201-206; "
            "NOTES.md section 4"), z=z, chi2=float(z @ z), dof=self.n, trace_hat=float(np.sum(a * a * lam / S)),
            residual_chi2=float(np.sum(C ** 2 * nu * nu / S ** 2)), residual_weights=np.r_[w, np.ones(extra)],
            residual_terms=C ** 2 * nu * nu / S ** 2)


def _root(f, decreasing, scale):
    """mu > 0 with f(mu) = 0 for a monotone f (bracketed geometrically, then Brent in log mu)."""
    from ...optimisers.minimise import root
    s = -1.0 if decreasing else 1.0
    lo = hi = max(scale, 1e-300)
    while s * f(hi) < 0 and hi < 1e300:
        hi *= 10.0
    while s * f(lo) > 0 and lo > 1e-300:
        lo /= 10.0
    if lo == hi:
        return lo
    t = root(lambda x: f(math.exp(x)), math.log(lo), math.log(hi), xtol=1e-15, rtol=4 * np.finfo(float).eps,
             maxiter=500)
    return math.exp(t)
