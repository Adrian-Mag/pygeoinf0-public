"""Checks of a supplied derivative (plan 019 F4; experiments/EXP-034-external-implementations/DESIGN.md 2.4).

    taylor_test(G, theta, directions=4, eps=..., seed=0)  ->  TaylorResult

For each direction h (scaled to the norm of theta, or to 1 at theta = 0): r(eps) = ||G(theta + eps h) - G(theta) -
eps DG(theta) h||_D for decreasing eps, and the observed rate log(r_j / r_j+1) / log(eps_j / eps_j+1), the median over
the steps above the rounding floor. A correct derivative of a twice differentiable map gives rate 2 (or a remainder at
rounding level everywhere, for an affine map); a kink, or a wrong derivative, gives rate 1 in some direction, and that
is reported, not hidden. Also the adjoint identity <DG h, y>_D = <h, DG* y>_M on random y.

directions: a number k (random directions: through the domain's coordinates for a pg.port.External, from M.random where the
space has its one law, a finite-dimensional inner product; on L2 of an interval from legendre_decay(M, 0.8), on L2 of a
domain in R^d, d >= 2, from random_features(M), named here as the directions of a check, not a prior: plan 069 P6) or
a list of elements. Steps that leave the map's declared dom are dropped and counted.

    complex_linearity(A, directions=4, seed=0)  ->  ComplexLinearityResult     (plan 020 P3b)

The companion check for complex spaces: a real-linear A between complex spaces (pg.Euclidean(n, field="C"), pg.Complexified)
is complex-linear iff A J = J A. For each random x: ||A(J x) - J(A x)|| / max(||A x||, ||A J x||), checked against a
tolerance; a map into a real space is refused (complex-linearity needs J on both sides).

The results are evidence with status checked, never a proof of differentiability or complex-linearity (AGENTS.md).
"""
from dataclasses import dataclass, field

import numpy as np

from ..core.claims import CHECKED, Claim, Claims
from ..core.errors import NotDefined, PriorUnstated, Refused
from ..core.logic import Truth

EPS = tuple(10.0 ** -k for k in range(1, 8))
FLOOR = 1e-11          # remainders below FLOOR * ||G(theta)|| are rounding: no rate is read from them
SMOOTH = 1.8           # a direction whose rate is at least SMOOTH is consistent with a correct derivative


@dataclass
class TaylorResult:
    eps: tuple
    remainders: list            # one array per direction (NaN where the step left dom)
    rates: list                 # one observed rate per direction (inf: remainder at rounding level)
    adjoint_gap: float          # max relative |<DG h, y> - <h, DG* y>| (NaN without an adjoint)
    left_dom: int = 0
    claims: Claims = field(default_factory=Claims)

    def __repr__(self):
        r = ", ".join("exact" if np.isinf(x) else f"{x:.2f}" for x in self.rates)
        return f"TaylorResult(rates [{r}], adjoint gap {self.adjoint_gap:.1e})\n{self.claims!r}"


def random_element(X, seed):
    """A random element of X for a check (module docstring): X.random(seed) where X has its one law; on an L2 space,
    which picks no prior (plan 069 P6), a draw from a named one: legendre_decay(X, 0.8) on an interval,
    random_features(X) on a domain in R^d, d >= 2. These are directions of a check, not a prior: the check's claims
    hold for whatever directions were drawn."""
    try:
        return X.random(seed=seed)
    except PriorUnstated:
        from ..spaces.base import L2
        if not isinstance(X, L2):
            raise
        from ..probability.priors import legendre_decay, random_features
        mu = legendre_decay(X, 0.8) if X.domain.cartesian_dim == 1 else random_features(X)
        return mu.sample(seed=seed)


def _directions(G, theta, k, rng):
    if hasattr(G, "coords"):
        cin = G.coords[0]
        N = cin.export.codomain
        if getattr(N, "field", "R") == "C":             # complex numbers: complex directions (plan 020 P3c)
            return [cin.element(N.random(seed=int(rng.integers(2 ** 31))).values) for _ in range(k)]
        return [cin.element(rng.standard_normal(cin.dim)) for _ in range(k)]
    if hasattr(G.domain, "random"):
        return [random_element(G.domain, int(rng.integers(2 ** 31))) for _ in range(k)]
    raise Refused(f"no random directions in {G.domain}", needs=("directions= (a list of elements)",))


def _random_in(G, D, seed, DG, h):
    """A random element of the codomain for the adjoint identity: D.random; where D has no default measure (a
    cell-realised L2, plan 020 P4), random numbers through G's codomain coordinates; otherwise DG h."""
    if hasattr(D, "random"):
        try:
            return random_element(D, seed)
        except (NotImplementedError, NotDefined, PriorUnstated):    # not built, no draws (cylindrical), no named law
            pass
    cout = G.coords[1] if hasattr(G, "coords") else None
    if cout is not None:
        N = cout.export.codomain
        if getattr(N, "field", "R") == "C":
            return cout.element(N.random(seed=seed).values)
        return cout.element(np.random.default_rng(seed).standard_normal(cout.dim))
    return DG(h)


def _rate(eps, r, scale):
    ok = np.isfinite(r) & (r > FLOOR * scale)
    if not ok.any():
        return np.inf
    steps = [np.log(r[j] / r[j + 1]) / np.log(eps[j] / eps[j + 1])
             for j in range(len(r) - 1) if ok[j] and ok[j + 1]]
    return float(np.median(steps)) if steps else np.nan


def taylor_test(G, theta, directions=4, eps=EPS, seed=0):
    """The Taylor test and the adjoint identity of G's derivative at theta (module docstring)."""
    rng = np.random.default_rng(seed)
    hs = _directions(G, theta, directions, rng) if np.isscalar(directions) else list(directions)
    M, D = G.domain, G.codomain
    size = M.norm(theta) or 1.0
    hs = [h * (size / M.norm(h)) for h in hs]
    eps = tuple(float(e) for e in eps)
    G0, DG = G(theta), G.derivative(theta)
    scale = max(D.norm(G0), 1e-300)
    dom = getattr(G, "_dom", None)
    remainders, rates, left = [], [], 0
    for h in hs:
        Dh = DG(h)
        r = np.full(len(eps), np.nan)
        for j, e in enumerate(eps):
            t = theta + e * h
            if dom is not None and dom.member(t) is Truth.FALSE:
                left += 1
                continue
            r[j] = D.norm(G(t) - G0 - e * Dh)
        remainders.append(r)
        rates.append(_rate(np.array(eps), r, scale))
    try:
        A = DG.adjoint
        ys = [_random_in(G, D, int(rng.integers(2 ** 31)), DG, h) for h in hs]
        gaps = []
        for h, y in zip(hs, ys):
            lhs, rhs = D.inner(DG(h), y), M.inner(h, A(y))
            gaps.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))
        gap = max(gaps)
    except (NotImplementedError, TypeError):
        gap = np.nan
    finite = [x for x in rates if not np.isnan(x)]
    if not finite:
        raise Refused("every step left the declared dom: no rate can be read", needs=("smaller eps",))
    bad = [i for i, x in enumerate(rates) if not np.isnan(x) and x < SMOOTH]
    prec = G.claims.find("precision") if hasattr(G, "claims") else None
    stated = f"; the provider states: {prec.statement} ({prec.reason})" if prec is not None and not prec.holds else ""
    note = stated if bad else ""
    claims = [Claim("Taylor remainder O(eps^2) in every direction" if not bad else
                    f"Taylor remainder not O(eps^2) in direction(s) {bad}: a kink or a wrong derivative there",
                    CHECKED, f"{len(hs)} directions, eps {eps[0]:.0e}..{eps[-1]:.0e}, observed rates "
                    + ", ".join("exact" if np.isinf(x) else f"{x:.2f}" for x in rates)
                    + (f"; {left} steps left dom" if left else "") + note, key="derivative", holds=not bad)]
    if not np.isnan(gap):
        claims.append(Claim("adjoint identity <DG h, y> = <h, DG* y>", CHECKED,
                            f"max relative gap {gap:.1e} over {len(hs)} pairs" + (stated if gap >= 1e-8 else ""),
                            key="adjoint", holds=gap < 1e-8))
    return TaylorResult(eps, remainders, rates, gap, left, Claims(claims))


@dataclass
class ComplexLinearityResult:
    defects: list               # one relative defect ||A J x - J A x|| / max(||A x||, ||A J x||) per direction
    claims: Claims = field(default_factory=Claims)

    @property
    def holds(self):
        return self.claims.find("complex-linear").holds

    def __repr__(self):
        return f"ComplexLinearityResult(max defect {max(self.defects):.1e})\n{self.claims!r}"


def _complex_direction(X, seed):
    from ..core.errors import Unsupported
    try:
        return X.random(seed=seed)
    except (AttributeError, Unsupported, NotImplementedError, NotDefined):
        H = getattr(X, "real_space", None)                       # Complexified: random real and imaginary parts
        if H is None:
            raise Refused(f"no random directions in {X}", needs=("directions= (a list of elements)",)) from None
        return X.element((random_element(H, seed), random_element(H, seed + 1)))


def complex_linearity(A, directions=4, seed=0, tol=1e-10):
    """Whether the linear operator A commutes with the complex structures J of its domain and codomain (module
    docstring): checked on random directions, or on a list of elements."""
    X, Y = A.domain, A.codomain
    for S in (X, Y):
        if getattr(S, "field", "R") != "C" or not hasattr(S, "J"):
            raise Refused(f"{S} is not a complex space: complex-linearity needs J on the domain and the codomain",
                          needs=("pg.Euclidean(n, field='C') or pg.Complexified on both sides",))
    rng = np.random.default_rng(seed)
    xs = ([_complex_direction(X, int(rng.integers(2 ** 30))) for _ in range(directions)]
          if np.isscalar(directions) else list(directions))
    defects = []
    for x in xs:
        a, b = A(X.J(x)), Y.J(A(x))
        defects.append(Y.norm(a - b) / max(Y.norm(a), Y.norm(b), 1e-300))
    ok = max(defects) <= tol
    claim = Claim("complex-linear (A J = J A)" if ok else "not complex-linear: A J != J A", CHECKED,
                  f"{len(xs)} directions, max relative defect {max(defects):.1e} (tolerance {tol:.0e})",
                  key="complex-linear", holds=ok)
    return ComplexLinearityResult(defects, Claims([claim]))
