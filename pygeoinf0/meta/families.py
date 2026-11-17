"""Hyperparameter families: the named sets a data-dependent choice rule searches (plan 070 P2, review B14: named records
carried in claims, not positional floats), and the noise families fitted by the evidence (plan 009 G9; EXP-030
DESIGN.md G9, appQ:173-182; moved here from relations/noise.py by plan 070 P2).

    F = pg.meta.damping("least_squares", centre=c)   gamma > 0, the weight of J = (1/2)||m - c||^2 in Phi + gamma J
    F = pg.meta.damping("sola", lo=1e-4, hi=1e4)     gamma in [lo, hi], SOLA's trade-off weight (the knee's range)
    F = pg.meta.scales()                             (a, nu): prior N(m0, a^2 S), noise N(b, nu^2 C); gamma = (nu / a)^2
    F = pg.meta.NoiseFamily(D, sigma=s, groups="s")  C(theta) = diag(theta_g(i)^2 s_i^2): one multiplier per group of
                                                     data (here per value of the label field s: per degree), base
                                                     standard deviations s (default: those of D's standard Gaussian)
    F = pg.meta.NoiseFamily(D, s, groups, extra={"core": C1})  + phi^2 C1 for named extra terms (coordinates of D)
    F.claim(values)                                  the family and the values as a claim (key "hyperparameter family")

A family names its parameters, the method they parameterise and that method's other options (a centre, a range), so
a record built at chosen values states which family they belong to and what they mean.
"""
import numpy as np

from ..core.claims import ASSERTED, DERIVED, Claim
from ..core.errors import SpaceMismatch, Unsupported
from ..spaces.base import Euclidean


class HyperparameterFamily:
    """A named family of hyperparameters (module docstring): name, parameters (their names, in order), method (the
    method they parameterise), options (the method's other options, fixed), meaning (what the parameters are) and the
    range of each parameter ({name: (lo, hi)}, open where None)."""

    def __init__(self, name, parameters, method, meaning, options=None, ranges=None):
        self.name, self.parameters, self.method, self.meaning = name, tuple(parameters), method, meaning
        self.options = dict(options or {})
        self.ranges = {p: (ranges or {}).get(p, (0.0, None)) for p in self.parameters}

    def claim(self, values=None):
        """The family (and, given, the values {name: value}) as a claim with key "hyperparameter family"."""
        def rng(p):
            lo, hi = self.ranges[p]
            return f"{p} > 0" if (lo == 0.0 and hi is None) else f"{p} in [{lo:g}, {hi:g}]"
        at = "" if values is None else "; at " + ", ".join(f"{k} = {float(v):.6g}" for k, v in values.items())
        opts = "".join(f", {k} = {'given' if v is not None else None}" for k, v in self.options.items())
        return Claim(f"hyperparameter family {self.name!r} of {self.method}: {', '.join(map(rng, self.parameters))} "
                     f"({self.meaning}{opts}){at}", ASSERTED, "the family the choice searched or the record was built "
                     "at (review B14)", key="hyperparameter family")

    def __repr__(self):
        return f"HyperparameterFamily({self.name!r}, {self.parameters}, method={self.method!r})"


def damping(method="least_squares", centre=None, lo=None, hi=None):
    """gamma > 0 (or in [lo, hi]): for least_squares the weight of J = (1/2)||m - c||^2 in Phi + gamma J (centre=c),
    for sola the trade-off weight of X = T G* (G G* + gamma C)^-1 (module docstring)."""
    meaning = {"least_squares": "the weight of J = (1/2)||m - c||^2 in Phi + gamma J",
               "sola": "SOLA's weight: X = T G* (G G* + gamma C)^-1, mismatch + gamma variance",
               "pli": "the PLI estimator family at gamma = (nu / a)^2"}.get(method, "a damping weight")
    if (lo is None) != (hi is None) or (lo is not None and not 0 < lo < hi):
        raise ValueError("a range of gamma is 0 < lo < hi (both, or neither)")
    options = {"centre": centre} if method == "least_squares" else {}
    return HyperparameterFamily("damping", ("gamma",), method, meaning, options,
                                {"gamma": (0.0, None) if lo is None else (float(lo), float(hi))})


def scales(method="pli"):
    """(a, nu): the prior N(m0, a^2 S) and the noise N(b, nu^2 C) of PLI; the posterior mean depends on gamma =
    (nu / a)^2 only (module docstring)."""
    return HyperparameterFamily("scales", ("a", "nu"), method, "prior N(m0, a^2 S), noise N(b, nu^2 C); "
                                "gamma = (nu / a)^2")


def scales_claim(a, nu, method="pli"):
    """The claim of the scales family at (a, nu), with gamma = (nu / a)^2 (the PLI records of pg.meta)."""
    c = scales(method).claim({"a": a, "nu": nu})
    return Claim(f"{c.statement}; gamma = (nu / a)^2 = {(float(nu) / float(a)) ** 2:.6g}", c.status, c.reason,
                 key=c.key)


class NoiseFamily(HyperparameterFamily):
    """C(theta) on D = R^n in coordinates (module docstring): group multipliers on a diagonal base, plus scaled extra
    terms. Parameters, in order: one theta per group, one phi per extra term (all positive), then the prior scale a
    (fitted unless given)."""

    def __init__(self, D, sigma=None, groups=None, extra=None, name="C(θ)"):
        if not isinstance(D, Euclidean):
            raise SpaceMismatch("a noise family lives on a data space R^n")
        self.D, self.label, n = D, name, D.dim
        if sigma is None:
            if D.is_standard:
                var = np.ones(n)
            elif D.weights is not None:
                var = 1.0 / D.weights
            else:
                raise Unsupported("give sigma: the data space's covariance is not diagonal")
        else:
            var = np.broadcast_to(np.asarray(sigma, float) ** 2, (n,)).copy()
        if np.any(~(var > 0)):
            raise ValueError("the base standard deviations must be positive")
        self.base = var
        if groups is None:
            groups = {"all": np.arange(n)}
        elif isinstance(groups, str):
            groups = D.groups(groups)
        elif isinstance(groups, tuple) and all(isinstance(x, str) for x in groups):
            groups = D.groups(*groups)
        elif not isinstance(groups, dict):
            groups = {k: np.asarray(g, int) for k, g in enumerate(groups)}
        self.groups = {k: np.asarray(v, int) for k, v in groups.items()}
        cover = np.concatenate(list(self.groups.values()))
        if not np.array_equal(np.sort(cover), np.arange(n)):
            raise ValueError("the groups must partition the data")
        self.group_of = np.empty(n, int)
        for j, idx in enumerate(self.groups.values()):
            self.group_of[idx] = j
        self.extra = {}
        for k, C in (extra or {}).items():
            C = np.asarray(C, float)
            if C.shape not in ((n,), (n, n)):
                raise ValueError(f"the extra term {k!r} is a diagonal (n,) or a matrix (n, n)")
            self.extra[k] = C
        self.names = [f"theta[{k}]" for k in self.groups] + [f"phi[{k}]" for k in self.extra]
        super().__init__("noise", self.names + ["a"], "pli", f"noise N(0, {name}) with {len(self.groups)} group "
                         f"multipliers and {len(self.extra)} extra terms; prior N(m0, a^2 S)")

    @property
    def size(self):
        return len(self.names)

    def covariance(self, theta):
        """The coordinate covariance C(theta): a vector (diagonal) or a matrix."""
        th = np.asarray(theta, float)
        g = len(self.groups)
        out = th[:g][self.group_of] ** 2 * self.base
        for phi, C in zip(th[g:], self.extra.values()):
            if out.ndim == 1 and C.ndim == 1:
                out = out + phi ** 2 * C
            else:
                out = (np.diag(out) if out.ndim == 1 else out) + phi ** 2 * (np.diag(C) if C.ndim == 1 else C)
        return out

    def measure(self, theta, name=None):
        """N(0, C(theta)) as a measure on D (its covariance operator is C(theta) W in coordinates)."""
        from ..operators.linear import Diagonal, Matrix
        from ..probability import GaussianMeasure
        D, S = self.D, self.covariance(theta)
        if S.ndim == 1 and (D.is_standard or D.weights is not None):
            C = Diagonal(S if D.is_standard else S * D.weights, D, name="C")
        else:
            S = np.diag(S) if S.ndim == 1 else S
            C = Matrix(D.apply_metric(S.T).T, D, D, name="C")
            C._sa = (True, DERIVED, "a covariance matrix in coordinates")
        C._pos = ("pd", DERIVED, "a positive combination of positive definite terms")
        return GaussianMeasure(D, C, name=name or f"{self.label} fitted")

    def __repr__(self):
        return f"NoiseFamily({self.D}, {len(self.groups)} groups, {len(self.extra)} extra terms)"
