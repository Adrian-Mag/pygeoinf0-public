"""Weighted chi-square laws Q = sum_j lam_j Z_j^2 with error bounds (plan 009 G7; EXP-030 DESIGN G7, NOTES.md section 7).

    Q = pg.WeightedChiSquare(weights, multiplicities=None, tail=None)
    Q.cdf(x), Q.sf(x)                   Enclosures containing P(Q <= x), P(Q > x)
    Q.quantile(p)                       an Enclosure containing the p-quantile inf{x : P(Q <= x) >= p}
    method="auto" (default)             a closed form for one distinct weight, the polar formula for two weights of
                                        multiplicity one, Davies' inversion otherwise
    method="davies"                     the trapezoidal inversion of the characteristic function, with proved bounds
    method="imhof"                      Imhof's integral by adaptive quadrature (the quadrature's estimate: checked)
    method="saddlepoint"                Lugannani-Rice (an approximation without a bound: asserted; the thesis realisation)

lam_j >= 0 with multiplicities m_j > 0 (Z_j independent standard normal; m_j copies of each weight), sum m_j lam_j < inf.

Davies' inversion (Davies 1973, Biometrika 60:415; Davies 1980, Appl. Stat. 29:323). With phi(u) = E exp(iuQ) =
prod_j (1 - 2i lam_j u)^(-m_j/2) and a step D > 0,

    S(x) = 1/2 - (1/pi) sum_{k >= 0} Im[phi((k + 1/2) D) exp(-i (k + 1/2) D x)] / (k + 1/2)
         = E[1/2 - (1/2) sign(sin(D (Q - x) / 2))]                       (the square-wave Fourier series)

and the bracket equals 1{Q < x} whenever |Q - x| < 2 pi / D, so |S(x) - P(Q < x)| <= P(Q >= x + h) + P(Q <= x - h),
h = 2 pi / D (aliasing). Both tails are bounded by Chernoff's inequality, P(Q >= c) <= exp(-t c) E exp(tQ) for
0 < t < 1 / (2 lam_max) and P(Q <= c) <= exp(t c) E exp(-tQ) for t > 0, minimised over t (any t gives a valid bound).
The series is cut after K terms; since |phi(u)| <= prod_{j in S} (2 lam_j u)^(-m_j/2) for any subset S of the weights
(r = sum_S m_j), the remainder is at most c_S D^(-r/2) (2/r) (K + 1/2)^(-r/2) / pi (an integral comparison for the
decreasing summand). Rounding adds at most 64 eps (number of weights + 8) sum_k |term_k|. The three parts make the
bracket (derived). P(Q = x) = 0 for x > 0 (Q has a density when some lam_j > 0), so P(Q < x) = P(Q <= x).

The series converges like k^(-1 - r/2): fast for many weights, slowly for r <= 2 (one or two weights of multiplicity
one). The number of terms is capped (max_terms); the bracket then states what was reached.

An unknown part (tail). Q = Q_N + T with T >= 0 independent of Q_N and E T <= tau (the omitted weights of a truncated
spectrum): {Q <= x} is in {Q_N <= x} and {Q_N <= x - delta} is in {Q <= x} or {T > delta}, so
P(Q_N <= x - delta) - P(T > delta) <= P(Q <= x) <= P(Q_N <= x), and q_N(p) <= q(p) <= min_delta [q_N(p + P(T > delta))
+ delta]. P(T > delta) <= tau / delta (Markov); when every omitted weight is <= tail_max = mu as well, Chernoff gives
P(T > delta) <= min_s exp(-s delta) (1 - 2 s mu)^(-tau / (2 mu)), since -log(1 - 2 s l) / l increases with l (EXP-030
NOTES.md section 7, sharpened).
"""
import math
import warnings

import numpy as np

from ..core.numbers import Enclosure
from ..core.claims import ASSERTED, CHECKED, DERIVED, Claim, Claims

EPS = np.finfo(float).eps
DAVIES = "Davies 1973 (Biometrika 60:415), trapezoidal inversion with aliasing, truncation and rounding bounds"


class WeightedChiSquare:
    """The law of sum_j lam_j Z_j^2 (module docstring)."""

    def __init__(self, weights, multiplicities=None, tail=None, tail_max=None, tail_status=DERIVED,
                 tail_reason="given", max_terms=4_000_000, name="Q"):
        lam = np.asarray(weights, float).ravel()
        m = np.ones_like(lam) if multiplicities is None else np.broadcast_to(np.asarray(multiplicities, float),
                                                                             lam.shape).copy()
        if np.any(lam < 0) or not np.all(np.isfinite(lam)):
            raise ValueError("the weights of a weighted chi-square law are finite and >= 0")
        if np.any(m <= 0):
            raise ValueError("multiplicities are positive")
        keep = lam > 0
        lam, m = lam[keep], m[keep]
        if len(lam) == 0:
            raise ValueError("at least one positive weight")
        # merge equal weights (exactly equal values: multiplicities add)
        u, inv = np.unique(lam, return_inverse=True)
        mm = np.zeros(len(u))
        np.add.at(mm, inv, m)
        order = np.argsort(-u)
        self.weights, self.multiplicities = u[order], mm[order]
        self.tail = 0.0 if tail is None else float(tail)
        if self.tail < 0:
            raise ValueError("the bound on the mean of the omitted part is >= 0")
        self.tail_max = None if tail_max is None else float(tail_max)
        self.tail_status, self.tail_reason, self.max_terms, self.name = tail_status, tail_reason, int(max_terms), name

    # moments -----------------------------------------------------------------------------------------
    @property
    def mean(self):
        """E Q_N (the known part; the full mean is in [mean, mean + tail])."""
        return float(self.weights @ self.multiplicities)

    @property
    def variance(self):
        return float(2 * (self.weights ** 2) @ self.multiplicities)

    @property
    def dof(self):
        return float(self.multiplicities.sum())

    # Chernoff tails ----------------------------------------------------------------------------------
    def _log_mgf(self, t):
        return -0.5 * float(self.multiplicities @ np.log1p(-2.0 * t * self.weights))

    def upper_tail_bound(self, c):
        """A bound on P(Q_N >= c) (Chernoff)."""
        if c <= self.mean:
            return 1.0
        from scipy.optimize import minimize_scalar
        tmax = 0.5 / self.weights[0]
        f = lambda s: -s * tmax * c + self._log_mgf(s * tmax)            # noqa: E731
        r = minimize_scalar(f, bounds=(0.0, 1.0 - 1e-12), method="bounded", options={"xatol": 1e-12})
        return float(min(1.0, math.exp(min(f(r.x), 0.0))))

    def lower_tail_bound(self, c):
        """A bound on P(Q_N <= c) (Chernoff)."""
        if c <= 0:
            return 0.0
        if c >= self.mean:
            return 1.0
        from scipy.optimize import minimize_scalar
        f = lambda logt: math.exp(logt) * c + self._log_mgf(-math.exp(logt))   # noqa: E731
        r = minimize_scalar(f, bounds=(-60.0, 60.0), method="bounded", options={"xatol": 1e-10})
        return float(min(1.0, math.exp(min(f(r.x), 0.0))))

    # Davies -------------------------------------------------------------------------------------------
    def _davies(self, x, tol):
        """(value, error bound, terms) for P(Q_N <= x)."""
        if x <= 0:
            return 0.0, 0.0, 0
        lam, m = self.weights, self.multiplicities
        h = max(x, 8.0 * math.sqrt(self.variance), self.mean)
        while self.upper_tail_bound(x + h) + self.lower_tail_bound(x - h) > tol / 2:
            h *= 1.5
            if h > 1e8 * max(x, self.mean):
                break
        alias = self.upper_tail_bound(x + h) + self.lower_tail_bound(x - h)
        D = 2 * math.pi / h
        # number of terms: the smallest over prefixes S of the sorted weights
        r = np.cumsum(m)
        logc = np.cumsum(-0.5 * m * np.log(2 * lam))
        target = max(tol / 4, 1e-300)
        logK = (2 / r) * (logc - 0.5 * r * math.log(D) + np.log(2 / r) - math.log(math.pi * target))
        K = int(min(self.max_terms, max(16.0, math.exp(min(float(logK.min()), 40.0)))))
        Kh = K + 0.5
        trunc = math.exp(min(float(np.min(logc - 0.5 * r * math.log(D) + np.log(2 / r) - 0.5 * r * math.log(Kh))),
                             700.0)) / math.pi
        total, absum = 0.0, 0.0
        block = max(1, int(4_000_000 // max(len(lam), 1)))
        for s in range(0, K + 1, block):
            k = np.arange(s, min(K + 1, s + block), dtype=float) + 0.5
            u = (k * D)[:, None]
            rho = -0.25 * (np.log1p(4.0 * (lam * u) ** 2) @ m)
            theta = 0.5 * (np.arctan(2.0 * lam * u) @ m) - k * D * x
            t = np.exp(rho) * np.sin(theta) / k
            total += float(t.sum())
            absum += float(np.abs(t).sum())
        val = 0.5 - total / math.pi
        rnd = 64 * EPS * (len(lam) + 8) * (absum / math.pi + 1.0)
        return val, alias + trunc + rnd, K

    def _imhof(self, x):
        """(value, estimated error) of P(Q_N <= x) by Imhof's integral, cut at U <= 1e6 / lam_min."""
        from scipy.integrate import quad
        lam, m = self.weights, self.multiplicities
        r = float(m.sum())
        logp = 0.5 * float(m @ np.log(lam))
        logU = (2 / r) * (math.log(2 / (math.pi * r * 1e-14)) - logp)  # the cut-off bound 2/(pi r U^(r/2) prod) <= 1e-14
        U = min(math.exp(min(logU, 700.0)), 1e6 / lam[-1])

        def f(u):
            if u == 0:
                return 0.5 * (float(m @ lam) - x)
            th = 0.5 * float(m @ np.arctan(lam * u)) - 0.5 * x * u
            return math.sin(th) * math.exp(-0.25 * float(m @ np.log1p((lam * u) ** 2))) / u
        edges = [0.0]
        step = min(1.0 / lam[0], 2 * math.pi / max(x, 1e-300))
        while edges[-1] < U:
            edges.append(min(U, edges[-1] + step))
            step *= 1.25
        val, err = 0.0, 0.0
        with warnings.catch_warnings():                                  # the estimate is reported, not hidden
            warnings.simplefilter("ignore")
            for lo, hi in zip(edges[:-1], edges[1:]):
                v, e = quad(f, lo, hi, limit=200, epsabs=1e-16, epsrel=1e-13)
                val, err = val + v, err + e
        cut = 2 / (math.pi * r) * math.exp(-0.5 * r * math.log(U) - logp)
        return 0.5 - val / math.pi, err / math.pi + cut

    def _saddlepoint(self, x):
        """The Lugannani-Rice approximation of P(Q_N <= x) (Kuonen 1999, Biometrika 86:929)."""
        from scipy.optimize import brentq
        from scipy.stats import norm
        lam, m = self.weights, self.multiplicities
        if x <= 0:
            return 0.0
        K1 = lambda t: float(m @ (lam / (1 - 2 * t * lam))) - x          # noqa: E731
        hi = 0.5 / lam[0] * (1 - 1e-13)                                  # K'(t) -> inf as t -> 1/(2 lam_max)
        lo = -1.0
        while K1(lo) > 0:
            lo *= 2
        t = brentq(K1, lo, hi, xtol=1e-300, rtol=4 * EPS, maxiter=500)
        with np.errstate(over="ignore", divide="ignore"):
            Kt = -0.5 * float(m @ np.log1p(-2 * t * lam))
            K2 = float(m @ (2 * lam ** 2 / (1 - 2 * t * lam) ** 2))
        w = math.copysign(math.sqrt(max(2 * (t * x - Kt), 0.0)), t)
        v = t * math.sqrt(K2)
        if abs(w) < 1e-7 or v == 0:                                      # the limit at the mean
            k2, k3 = self.variance, 8 * float(m @ lam ** 3)
            return 0.5 + k3 / (6 * math.sqrt(2 * math.pi) * k2 ** 1.5)
        return float(norm.cdf(w) + norm.pdf(w) * (1 / w - 1 / v))

    def _closed(self, x):
        """One distinct weight: lam chi^2_m (the regularised incomplete gamma function, relative accuracy ~1e-15)."""
        from scipy.special import gammainc
        v = float(gammainc(0.5 * self.multiplicities[0], 0.5 * x / self.weights[0]))
        e = 1e-14 * v + 1e-300
        return Enclosure(v, max(v - e, 0.0), min(v + e, 1.0), DERIVED, "closed form: lam chi^2_m "
                         "(scipy.special.gammainc, relative accuracy 1e-14 stated)")

    def _polar(self, x):
        """Two weights of multiplicity one: P = (1/2 pi) int_0^2pi 1 - exp(-x / (2 g(t))) dt, g = l1 cos^2 + l2 sin^2
        (polar coordinates of (Z_1, Z_2): rho^2 ~ chi^2_2 independent of the uniform angle), by the periodic trapezoidal
        rule, which converges geometrically for this analytic integrand; the difference of n and 2n points is the
        estimate (checked)."""
        l1, l2 = self.weights

        def rule(n):
            t = (np.arange(n) + 0.5) * (0.5 * math.pi / n)               # a quarter period suffices (symmetry)
            g = l1 * np.cos(t) ** 2 + l2 * np.sin(t) ** 2
            return float(np.mean(-np.expm1(-x / (2 * g))))
        n, prev = 16, rule(8)
        while True:
            cur = rule(n)
            if abs(cur - prev) <= 1e-15 or n > 1 << 22:
                break
            prev, n = cur, 2 * n
        e = abs(cur - prev) + 1e-15
        return Enclosure(cur, max(cur - e, 0.0), min(cur + e, 1.0), CHECKED, "the polar formula for two weights, "
                         f"periodic trapezoidal rule ({n} points; doubling difference {e:.2g})")

    def _known_cdf(self, x, method, tol):
        if method == "auto" and len(self.weights) == 1:
            return self._closed(float(x))
        if method == "auto" and len(self.weights) == 2 and self.dof == 2:
            return self._polar(float(x))
        if method in ("auto", "davies"):
            v, e, K = self._davies(float(x), tol)
            return Enclosure(min(max(v, 0.0), 1.0), max(v - e, 0.0), min(v + e, 1.0), DERIVED,
                             f"{DAVIES}: {K} terms, error <= {e:.3g}")
        if method == "imhof":
            v, e = self._imhof(float(x))
            return Enclosure(min(max(v, 0.0), 1.0), max(v - e, 0.0), min(v + e, 1.0), CHECKED,
                             f"Imhof 1961 (Biometrika 48:419) by adaptive quadrature: estimated error {e:.3g}")
        if method == "saddlepoint":
            v = self._saddlepoint(float(x))
            return Enclosure(v, 0.0, 1.0, ASSERTED, "the Lugannani-Rice saddlepoint approximation: no error bound")
        raise ValueError("method is 'auto', 'davies', 'imhof' or 'saddlepoint'")

    # the law ------------------------------------------------------------------------------------------
    def tail_probability(self, delta):
        """A bound on P(T > delta) for the omitted part: Markov (E T <= tail), and with tail_max (every omitted weight
        <= tail_max) Chernoff: log E exp(sT) <= -(tail / (2 tail_max)) log(1 - 2 s tail_max), as -log(1 - 2 s l) / l
        increases with l."""
        if self.tail == 0:
            return 0.0
        b = min(1.0, self.tail / delta)
        mu = self.tail_max
        if mu:
            from scipy.optimize import minimize_scalar
            f = lambda z: -z / (2 * mu) * delta - self.tail / (2 * mu) * math.log1p(-z)     # noqa: E731
            r = minimize_scalar(f, bounds=(0.0, 1 - 1e-12), method="bounded", options={"xatol": 1e-12})
            b = min(b, math.exp(min(f(r.x), 0.0)))
        return b

    def _tail_status(self, st):
        if st == DERIVED and self.tail_status == DERIVED:
            return DERIVED
        return ASSERTED if ASSERTED in (st, self.tail_status) else CHECKED

    def _tail_reason(self):
        return (f"an omitted part T >= 0 with E T <= {self.tail:.3g}" + (f", weights <= {self.tail_max:.3g}"
                if self.tail_max else "") + f" ({self.tail_reason}): P(Q_N <= x - d) - P(T > d) <= P(Q <= x) <= "
                "P(Q_N <= x)")

    def cdf(self, x, method="auto", tol=1e-12):
        """An Enclosure containing P(Q <= x) (module docstring)."""
        F = self._known_cdf(x, method, tol)
        if self.tail == 0 or method == "saddlepoint":
            return F
        lo = 0.0
        for d in float(x) * np.geomspace(1e-6, 0.5, 16):
            t = self.tail_probability(d)
            if t < 1:
                lo = max(lo, self._known_cdf(float(x) - d, method, tol).lower - t)
        return Enclosure(float(F), lo, F.upper, self._tail_status(F.status), f"{F.reason}; {self._tail_reason()}")

    def sf(self, x, method="auto", tol=1e-12):
        F = self.cdf(x, method, tol)
        return Enclosure(1 - float(F), 1 - F.upper, 1 - F.lower, F.status, F.reason)

    def _hi_start(self, p):
        c = self.mean + self.tail + 1.0 * math.sqrt(self.variance)
        while 1 - self.upper_tail_bound(c) < p + 1e-3 * (1 - p):
            c = 2 * c
        return 2 * c

    @staticmethod
    def _root(g, lo, hi, side):
        """A point x with g(x) < 0 (side "neg") or g(x) >= 0 (side "pos") next to the root of g on [lo, hi]."""
        from scipy.optimize import brentq
        x = brentq(g, lo, hi, xtol=1e-300, rtol=8 * EPS, maxiter=300)
        step = 4 * EPS * max(abs(x), 1e-300)
        for _ in range(200):
            if (g(x) < 0) == (side == "neg"):
                return x
            x = x - step if side == "neg" else x + step
            step *= 2
        raise ArithmeticError("no sign change next to the root")

    def _near(self, g, v, lo, hi, side):
        """_root of g bracketed by stepping from v towards lo (side "neg") or hi (side "pos")."""
        if (g(v) < 0) == (side == "neg"):
            return v
        d = 1e-10 * max(v, 1e-300)
        while True:
            x = max(lo, v - d) if side == "neg" else min(hi, v + d)
            if (g(x) < 0) == (side == "neg") or x in (lo, hi):
                return self._root(g, *sorted((v, x)), side)
            d *= 10

    def _known_quantile(self, p, method, tol, upper_only=False):
        hi0 = self._hi_start(p)
        if method == "saddlepoint":
            from scipy.optimize import brentq
            v = brentq(lambda x: self._saddlepoint(x) - p, 1e-300, hi0, xtol=1e-300, rtol=8 * EPS)
            return Enclosure(v, 0.0, math.inf, ASSERTED, "the Lugannani-Rice saddlepoint approximation, inverted: no "
                             "error bound")
        F = lambda x: self._known_cdf(x, method, tol)                     # noqa: E731
        if upper_only:
            u = self._root(lambda x: F(x).lower - p, 0.0, hi0, "pos")
            return Enclosure(u, 0.0, u, F(u).status, "")
        v = self._root(lambda x: float(F(x)) - p, 0.0, hi0, "pos")
        lower = self._near(lambda x: F(x).upper - p, v, 0.0, hi0, "neg")       # F(lower) <= F.upper(lower) < p
        upper = self._near(lambda x: F(x).lower - p, v, 0.0, hi0, "pos")       # F(upper) >= F.lower(upper) >= p
        st = F(v)
        return Enclosure(v, lower, upper, st.status, f"the cdf bracket inverted ({st.reason})")

    def quantile(self, p, method="auto", tol=1e-12):
        """An Enclosure containing the p-quantile (0 < p < 1; module docstring)."""
        p = float(p)
        if not 0 < p < 1:
            raise ValueError("0 < p < 1")
        q = self._known_quantile(p, method, tol)
        if self.tail == 0 or method == "saddlepoint":
            return q
        best = math.inf
        for d in np.geomspace(1e-6, 0.3, 10) * max(float(q), 1e-300):
            pp = p + self.tail_probability(d)
            if pp < 1 - 1e-9:
                best = min(best, self._known_quantile(pp, method, tol, upper_only=True).upper + d)
        return Enclosure(float(q), q.lower, best, self._tail_status(q.status), f"{q.reason}; {self._tail_reason()}")

    @property
    def claims(self):
        cl = [Claim(f"{self.name} = sum lam_j Z_j^2 with {len(self.weights)} distinct weights (dof {self.dof:g})",
                    DERIVED, "the law of a Gaussian quadratic form in its eigenbasis")]
        if self.tail:
            cl.append(Claim(f"an omitted part T >= 0, independent, E T <= {self.tail:.6g}", self.tail_status,
                            self.tail_reason, key="tail"))
        return Claims(cl)

    def __repr__(self):
        return (f"WeightedChiSquare({len(self.weights)} weights, dof {self.dof:g}, mean {self.mean:.6g}"
                + (f", tail <= {self.tail:.3g})" if self.tail else ")"))
