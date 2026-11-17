"""BGP for a norm ball (or operator ellipsoid) prior and a noise ball in the eigenbasis of a data-space
decomposition (plan 009 G3; solvers/dataspace/decomposition.py): no factorisation per direction, the dual evaluated
exactly in the same coordinates (optimisers/dual/reduction.py has the solvers).

From spectral_many.py (merged here by plan 063):
BGP for many directions at once in the eigenbasis of a data-space decomposition (plan 009 G4): the ball-ball problem
of optimisers/dual/spectral.py, max a.y over ||y|| <= r and ||delta - diag(s) y|| <= rho, solved for J directions as arrays.

Every direction has its own a; the data map, the datum and the radii are shared, so the problem in the right singular
basis (reduction.ball_ball_spectral) differs between directions only through ab. The two multipliers (mu for the prior,
omega for the data) minimise the Lagrange dual, convex in (mu, omega) and an upper bound for every mu, omega >= 0:

    g(mu, omega) = sum_k (ab_k + omega bb_k)^2 / (2 D_k) + mu r^2 / 2 + omega (rho^2 - ||delta||^2) / 2,
    D_k = mu + omega s_k^2,  y_k = (ab_k + omega bb_k) / D_k,
    grad g = ((r^2 - ||y||^2) / 2, (rho^2 - misfit(y)) / 2),  Hess g = sum_k [y_k, -(bb_k - s_k^2 y_k)]^T [...] / D_k.

The joint route (P4a speed): Newton's method on g in log mu and log omega (a multiplicative step, so the multipliers
stay positive across the eigenvalues' range of 1e-6 to 5e9), safeguarded by backtracking on g (Armijo; g is convex, so
every accepted step decreases an upper bound), started from the previous chunk's solution (neighbouring directions
have neighbouring multipliers). About ten passes over the coordinates per step and a few steps per direction, against
a bracket on omega with a trust-region solve for mu at every trial. Columns that do not converge take the bracketing
route (omega by safeguarded Newton steps in log omega inside a bracket [lo, hi] with misfit(lo) > rho^2 >= misfit(hi),
mu by Newton's method on 1/||y(mu)|| - 1/r from a lower bound, Mor\'e-Sorensen, the iterate kept on the feasible side).

Either way the result is then certified exactly as for one direction: the primal value a.y at the feasible end of the bracket is a lower bound (a
witness checked against both constraints), the dual phi(x) at x = -omega (delta - s y) an upper bound by weak duality,
evaluated exactly. Directions whose gap does not close to core.precision.TOL (and the shared EMPTY or TOUCH cases) are solved again
one by one (SpectralBall.bound), so every result is certified the same way as before; the claims count both routes.

The general rule for EXACT (optimisers/dual/exactness.py) is applied per chunk: E1 for every column, and where E1 exceeds GRAY x
TOL the RESOLVES perturbed problems (exactness.perturbed, built once per decomposition) solve those columns again as
arrays, by this module's own route (plan 054 fix2: one by one they made a 256-centre map 40x slower). E per direction
is returned with the bracket, as Bound.error is for one direction.
"""


import math

import numpy as np

from ...core import estimable
from ...core.precision import TOL
from ..scalar import trs, trs_columns, trs_floor
from . import bound, exactness, reduction

RCOND = 1e-15           # eigenvalues below RCOND * (their block's max, about 5 eps) count as zero where 1 / Lam appears;
# the count is in every Bound's note, with those kept below the floor eps n_block (plan 054 P2: measured against exact
# intervals at cond >= 1e16, moving the cut to the floor or to 0 changed the endpoints within their rounding error).
# Plan 054 fix round: with no prior, eigenvalues at the floor not explained by structure (dec.at_floor) refuse the bound
# (reduction.floor_refusal); FeasibleSet.dependent declares the data exactly dependent there: those eigenvalues are then zero.


class SpectralBall:
    """BGP in the eigenbasis (DataDecomposition.ball_problem, FeasibleSet): with u = sum_k y_k v_k + zeta w (v_k the model
    directions that F G maps to sqrt(Lam_k) u_k, orthonormal in the metric S^-1; w the part of the direction the data do
    not see), the data map is diag(sqrt Lam), so reduction.ball_ball_spectral solves each direction in O(n) and the dual
    phi(x) = <x, delta> + r ||c - G* F^T U x||_S + rho ||x|| is evaluated exactly in the same coordinates."""

    def __init__(self, dec, e, r, rho, dependent=False, delta=None):
        self.dec, self.r, self.rho, self.dependent = dec, float(r), float(rho), dependent
        self._check, self._dir, self._how, self._terr = True, None, None, 0.0     # the exactness rule
        # (optimisers/dual/exactness.py), off on its re-solves; _how: the current target's estimability (core/estimable.py);
        # _terr: its representer's stated error; _m0n: ||m0|| of a prior centre (the datum's stated error)
        self._m0n = 0.0
        lam = dec.lam
        self.delta = dec._coef(e) if delta is None else np.asarray(delta, float)
        self.pos = dec.positive(RCOND) & (dec.above_floor() if dependent else True)
        self.s = np.sqrt(lam[self.pos])
        self.dp = self.delta[self.pos]
        dz = self.delta[~self.pos]
        self.f0 = float(dz @ dz)
        self.s2 = np.r_[lam[self.pos], 0.0]
        self.bb = np.r_[self.s * self.dp, 0.0]
        self.dd = float(self.delta @ self.delta)
        self.res = (np.r_[self.dp, 0.0], self.f0)                 # the misfit from the residual (reduction.residual_misfit)
        self.misfit = reduction.residual_misfit(self.bb, self.s2, self.dd, self.res)
        self.bnorm = float(np.sqrt(np.sum(lam[self.pos])))
        zero, below, n = dec.zero_count(RCOND)
        self.note = f"data-space decomposition ({dec.describe()}); {zero} of {n} eigenvalues <= {RCOND:g} x their " \
                    "block's largest counted as zero" + (f", {below} kept at most eps n_block x it (no correct digit)"
                                                         if below else "")
        if dependent:
            self.note += f"; declared exactly dependent (K.dependent): the {zero + below} at the rounding floor taken as zero"

    def _x(self, pos_part, zero_part):
        x = np.zeros(len(self.delta))
        x[self.pos] = pos_part
        x[~self.pos] = zero_part
        return x

    def phi(self, x, a_pos, a_perp, const):
        """The dual value at the multiplier x (eigen coordinates): an upper bound for every x (weak duality)."""
        v = math.sqrt(a_perp ** 2 + float(np.sum((a_pos - self.s * x[self.pos]) ** 2)))
        if math.isfinite(self.r):
            sp = self.r * v
        else:
            sp = 0.0 if v <= 1e-10 * max(math.sqrt(a_perp ** 2 + float(a_pos @ a_pos)), self.bnorm, 1e-300) else math.inf
        return const + float(x @ self.delta) + sp + self.rho * float(np.linalg.norm(x))

    def _cert(self, x):
        dec = self.dec
        return lambda: dec.D.solve_metric(dec._back(x))

    def bound(self, beta, cc, const=0.0, how=None, terr=0.0):
        """how: the target's estimability (core/estimable.py: DECLARED, BUILT or None); it decides with no prior only.
        terr: the target representer's stated realisation error (optimisers/dual/exactness.py, "The representers' own stated
        errors"; 0: none stated)."""
        self._dir, self._how, self._terr = (beta, float(cc), const), how, float(terr)
        a_pos = beta[self.pos] / self.s
        a_perp = math.sqrt(max(cc - float(a_pos @ a_pos), 0.0))
        r, rho = self.r, self.rho
        if not math.isfinite(r):
            return self._none(a_pos, a_perp, const)
        ab = np.r_[a_pos, a_perp]
        case, val, yb, om = reduction.ball_ball_spectral(ab, self.bb, self.s2, self.dd, r, rho, self.res)
        if case == "EMPTY":
            res = self._x(self.dp - self.s * yb[:-1], self.delta[~self.pos])
            x = -res / np.linalg.norm(res)
            z = self.phi(x, np.zeros_like(a_pos), 0.0, 0.0)
            return bound.LazyBound(-math.inf, -math.inf, -math.inf, "EMPTY", self._cert(x), yb,
                                  f"K(d) is empty: phi(lambda, 0) = {z:.3g} < 0 (Farkas certificate)" if z < 0 else
                                  "K(d) is empty", claim=bound.CERTIFIED)
        if case == "TOUCH":
            y0 = yb[:-1]
            n0 = float(np.linalg.norm(y0))
            if n0 > r * (1 + 1e-12):
                x = self._x(-y0 / self.s, 0.0)
                return bound.LazyBound(-math.inf, -math.inf, -math.inf, "EMPTY", self._cert(x), yb, "K(d) is empty", claim=bound.CERTIFIED)
            if a_perp <= 1e-12 * max(float(np.linalg.norm(ab)), 1e-300):
                val, y, x = float(a_pos @ y0), np.r_[y0, 0.0], self._x(a_pos / self.s, 0.0)
            else:
                rem = math.sqrt(max(r * r - n0 * n0, 0.0))
                y = np.r_[y0, rem]
                val = float(ab @ y)
                x = None if rem == 0 else self._x((a_pos - (a_perp / rem) * y0) / self.s, 0.0)
        elif case in ("ZERO", "PRIOR"):
            y, x = yb, np.zeros(len(self.delta))
        else:
            y = yb
            x = -om * self._x(self.dp - self.s * yb[:-1], self.delta[~self.pos])
        return self._finish(const + val, y, x, a_pos, a_perp, const)

    def _finish(self, value, y, x, a_pos, a_perp, const):
        from ...core.logic import Truth, decide
        r, rho = self.r, self.rho
        feasible = y is not None and (not math.isfinite(r) or float(y @ y) <= r * r * (1 + 1e-9))
        if feasible:
            yp = y[:len(self.s)]
            m = math.sqrt(max(float(np.sum((self.dp - self.s * yp) ** 2)) + self.f0, 0.0))
            feasible = decide(max(m - rho, 0.0), max(rho, m, 1e-300)) is not Truth.FALSE
        lower = value if feasible else -math.inf
        upper = self.phi(x, a_pos, a_perp, const) if x is not None else math.inf
        gap_ok = upper - lower <= TOL * max(1.0, abs(value))
        status = "EXACT" if gap_ok else ("BRACKET" if math.isfinite(lower) and math.isfinite(upper) else "PRIMAL")
        note, E = self.note, None
        undecided = None if math.isfinite(r) else getattr(self, "_undecided", None)
        if undecided is not None:                                  # no prior, estimable only within the stated error
            status, lower, upper = ("BRACKET" if status == "EXACT" else status), -math.inf, math.inf
            note += exactness.note_estimable(*undecided)
        elif self._check and self._dir is not None:              # the general rule (optimisers/dual/exactness.py)
            beta, cc, c0 = self._dir
            self._E1 = None
            if x is None:
                E1 = math.inf
            elif math.isfinite(r):
                v2 = a_perp ** 2 + float(np.sum((a_pos - self.s * x[self.pos]) ** 2))
                E1 = float(exactness.first_order(self, x, v2, cc, self._terr)[0])
            else:
                E1 = exactness.first_order_none(self, x, y[:len(self.s)], cc, self._terr)
            Es = 0.0 if self._E1 is None else float(np.ravel(self._E1[1])[0])
            E = exactness.judge(self, beta, cc, c0, value, E1, Es)
            if E > TOL * max(1.0, abs(value)):
                status = "BRACKET" if status == "EXACT" else status
                lower, upper = lower - exactness.WIDEN * E, upper + exactness.WIDEN * E
                note += exactness.note(E, value, exactness.what(self, self._terr))
        if undecided is None and not math.isfinite(r):              # no prior: which estimability applied
            note += "; " + estimable.note(self._how)
        return bound.LazyBound(value, lower, upper, status, None if x is None else self._cert(x), y, note, E, claim=bound.CERTIFIED)

    def _none(self, a_pos, a_perp, const):
        """No prior (r = inf): finite iff the direction is estimable (reduction.none_ball in the eigenbasis); refused where
        eigenvalues at the floor are not explained by structure, unless the data are declared dependent (K.dependent,
        reduction.floor_refusal). Estimability is decided by the stated error of a_perp^2 (optimisers/dual/exactness.py, "No prior"):
        UNBOUNDED only beyond it (refused for a target declared or built estimable: the numbers contradict it); within
        it the finite value with an infinite bracket, or, for a target declared or built estimable (core/estimable.py),
        the finite value judged by first order."""
        unexplained = 0 if self.dependent else self.dec.at_floor(RCOND)[0]
        if unexplained:
            raise reduction.floor_refusal("no prior (data-space decomposition)", unexplained, len(self.delta),
                                     "eigenvalues of the whitened data Gram <= eps n_block x their block's largest")
        mu0 = a_pos / self.s
        cc = self._dir[1] if self._dir is not None else a_perp ** 2 + float(a_pos @ a_pos)
        e2 = exactness.perp_error(self, mu0, cc, a_pos, self._terr)
        if a_perp ** 2 > e2:                                      # decided by the stated error (optimisers/dual/exactness.py)
            if self._how:
                raise exactness.contradicted(self._how, a_perp ** 2, e2)
            return bound.Bound(math.inf, -math.inf, math.inf, "UNBOUNDED", None, None,
                              "the direction is not estimable and nothing bounds it (inf:122): its part outside the "
                              f"data's span, a_perp^2 = {a_perp ** 2:.3g}, exceeds its stated error {e2:.3g}; "
                              + estimable.note(None), claim=bound.CERTIFIED)
        self._undecided = None if self._how else (a_perp ** 2, e2)
        a_perp = 0.0                                              # estimable within the stated error, or declared/built
        rho = self.rho
        dperp = self.delta[~self.pos]
        rp2 = rho ** 2 - self.f0
        if rp2 < -1e-12 * max(rho ** 2, 1e-300):
            x = self._x(0.0, -dperp / np.linalg.norm(dperp))
            z = self.phi(x, np.zeros_like(a_pos), 0.0, 0.0)
            return bound.LazyBound(-math.inf, -math.inf, -math.inf, "EMPTY", self._cert(x), None,
                                  f"K(d) is empty: phi(lambda, 0) = {z:.3g} < 0 (Farkas certificate)" if z < 0 else
                                  "K(d) is empty", claim=bound.CERTIFIED)
        rp = math.sqrt(max(rp2, 0.0))
        nm = float(np.linalg.norm(mu0))
        z = self.dp + (rp * mu0 / nm if nm > 0 else 0.0)
        y = np.r_[z / self.s, 0.0]
        x = self._x(mu0, -(nm / rp) * dperp) if rp > 0 else None
        return self._finish(const + float(mu0 @ self.dp) + rp * nm, y, x, a_pos, a_perp, const)


# ---------------------------------------------------------------- from spectral_many.py (plan 063)
CHUNK = 64                # directions per array pass (n x CHUNK arrays stay in cache: 2.3x faster than 512 at n = 8002)
NEWTON = 40               # joint Newton steps before a column takes the bracketing route
SPAN = 8                  # chunks per task when the chunks run in worker processes (pg.settings.workers > 1)


def _solve(sp, ab, omega):
    """y(omega) for every direction (columns of ab), with the multipliers mu."""
    c = ab + sp.bb[:, None] * omega[None, :]
    D0 = sp.s2[:, None] * omega[None, :]
    mu = trs_floor(c, D0, sp.r)
    y, mu = trs_columns(c, D0, sp.r, mu, np.ones(ab.shape[1], bool))
    return y, mu


def _misfit(sp, y):
    """||d - B y||^2 from the residual (reduction.residual_misfit; SpectralBall.misfit)."""
    return sp.misfit(y)


def bounds(sp, Beta, cc, const, how=None, error=False, terr=None):
    """(value, lower, upper, status, fallback count) of sup over the feasible set in the directions whose eigen
    coordinates are the columns of Beta (n_eff, J), with <c, S c> = cc and constants const (module docstring); how:
    the directions' estimability (core/estimable.py), deciding with no prior; error=True appends E of the general
    rule per direction (optimisers/dual/exactness.py; nan where it did not judge); terr: the targets' stated representer errors
    per direction (optimisers/dual/exactness.py; None: none stated)."""
    return signed_bounds(sp, Beta, cc, const, signs=(1.0,), how=how, error=error, terr=terr)[0]


class Directions:
    """One fork pool (execution.Pool) for every call of signed_bounds on the problem sp with at most `columns`
    directions of `rows` eigen coordinates each: a map's batches and both signs share it, instead of forking a pool per
    call (16 forks for a 1 degree map). The directions of a call reach the workers through a shared buffer of rows x
    columns doubles (32 MB at 8002 x 512; 262 MB at 8002 x 4096), allocated only when there are several workers."""

    def __init__(self, sp, rows, columns):
        from ...execution import Pool, shared, workers
        self.buf = shared((rows, columns)) if workers() > 1 else None
        self.pool = Pool(_span_task, (sp, self.buf))

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def signed_bounds(sp, Beta, cc, const, signs=(1.0, -1.0), directions=None, how=None, error=False, terr=None):
    """[bounds(sp, s Beta, cc, s const, error=error) for s in signs], the spans of every sign in one round of worker
    processes (a Directions pool when given, else one forked for this call). The chunks, their starts and the arithmetic
    are those of separate bounds calls, so the results are the same to the bit."""
    J = Beta.shape[1]
    terr = np.zeros(J) if terr is None else np.broadcast_to(np.asarray(terr, float), (J,))
    outs = [(np.empty(J), np.empty(J), np.empty(J), np.empty(J, dtype=object), np.full(J, np.nan)) for _ in signs]
    if not math.isfinite(sp.r):
        return [_one_by_one(sp, s * Beta, cc, s * np.asarray(const, float), np.arange(J), *o, how, terr)
                [:6 if error else 5] for s, o in zip(signs, outs)]
    yls, _ = trs(sp.bb, sp.s2, sp.r)
    mmin = _misfit(sp, yls)
    if mmin >= sp.rho ** 2 * (1 - 1e-10):                    # EMPTY or TOUCH: shared; the per-direction route decides
        return [_one_by_one(sp, s * Beta, cc, s * np.asarray(const, float), np.arange(J), *o, how, terr)
                [:6 if error else 5] for s, o in zip(signs, outs)]
    cc, const = np.asarray(cc, float), np.asarray(const, float)
    first = min(J, CHUNK)                                        # the first chunk sets the start of every other
    spans = [(a, min(J, a + SPAN * CHUNK)) for a in range(first, J, SPAN * CHUNK)]
    tasks, redo = [], []
    for k, (s, (value, lower, upper, status, err)) in enumerate(zip(signs, outs)):
        bad, start = _chunk(sp, _signed(s, Beta[:, :first]), cc[:first], s * const[:first], value[:first], lower[:first],
                            upper[:first], status[:first], None, err[:first], terr[:first])
        redo.append(list(np.flatnonzero(bad)))
        tasks += [(k, s, start, a, b, cc[a:b], s * const[a:b], terr[a:b]) for a, b in spans]
    own = directions is None
    if own:
        directions = _Own(sp, Beta)
    try:
        if directions.buf is not None and directions.pool.parallel() and len(spans) > 1:   # spans in forked workers
            if directions.buf is not Beta:
                directions.buf[:, :J] = Beta
            done = directions.pool.map(tasks)
        else:
            done = [(t, _span(sp, _signed(t[1], Beta[:, t[3]:t[4]]), t[5], t[6], t[2], t[7])) for t in tasks]
    finally:
        if own:
            directions.close()
    for t, out in done:
        k, a, b = t[0], t[3], t[4]
        for o, part in zip(outs[k], out[:5]):
            o[a:b] = part
        redo[k] += list(a + out[5])
    return [(_one_by_one(sp, s * Beta, cc, s * const, np.array(r, int), *o, how, terr) if r else o[:4] + (0, o[4]))
            [:6 if error else 5] for s, o, r in zip(signs, outs, redo)]


class _Own(Directions):
    """A pool for one call: the workers inherit Beta itself (no copy)."""

    def __init__(self, sp, Beta):
        from ...execution import Pool
        self.buf = Beta
        self.pool = Pool(_span_task, (sp, Beta))


def _span_task(sp, buf, k, s, start, a, b, cc, const, terr=None):
    return _span(sp, _signed(s, buf[:, a:b]), cc, const, start, terr)


def _signed(s, B):
    """s B for s = +1 or -1: the view itself for +1 (no copy of n_eff x span doubles), its negation for -1."""
    return B if s == 1.0 else -B


def _span(sp, Beta, cc, const, start, terr=None):
    """The directions Beta (columns) in chunks, each started from the previous one (the first from `start`): (value,
    lower, upper, status, error, indices to redo)."""
    m = Beta.shape[1]
    value, lower, upper, error = np.empty(m), np.empty(m), np.empty(m), np.full(m, np.nan)
    status = np.empty(m, dtype=object)
    redo = []
    for c in range(0, m, CHUNK):
        sl = slice(c, min(m, c + CHUNK))
        bad, start = _chunk(sp, Beta[:, sl], cc[sl], const[sl], value[sl], lower[sl], upper[sl], status[sl], start,
                            error[sl], None if terr is None else terr[sl])
        redo += list(c + np.flatnonzero(bad))
    return value, lower, upper, status, error, np.array(redo, int)


def _one_by_one(sp, Beta, cc, const, idx, value, lower, upper, status, error, how=None, terr=None):
    for j in idx:
        b = sp.bound(Beta[:, j], float(cc[j]), float(const[j]), how, 0.0 if terr is None else float(terr[j]))
        value[j], lower[j], upper[j], status[j] = b.value, b.lower, b.upper, b.status
        error[j] = np.nan if b.error is None else b.error
    return value, lower, upper, status, len(idx), error


def values(sp, B, cc, const):
    """The values alone of sup over sp in the directions B (columns), by this module's route (no rule: sp._check is
    off on the perturbed problems of optimisers/dual/exactness.py); inf where a direction is refused (another answer is
    admitted)."""
    from ...core.errors import Refused
    J = B.shape[1]
    out = np.empty(J)

    def single(j):
        try:
            return sp.bound(B[:, j], float(cc[j]), float(const[j]), sp._how).value
        except Refused:
            return math.inf
    if not math.isfinite(sp.r):
        return np.array([single(j) for j in range(J)])
    yls, _ = trs(sp.bb, sp.s2, sp.r)
    if _misfit(sp, yls) >= sp.rho ** 2 * (1 - 1e-10):     # EMPTY or TOUCH
        return np.array([single(j) for j in range(J)])
    v, _, _, _, _, redo = _span(sp, B, np.asarray(cc, float), np.asarray(const, float), None)
    out[:] = v
    for j in redo:
        out[j] = single(j)
    return out


def _chunk(sp, B, cc, const, value, lower, upper, status, start=None, error=None, terr=None):
    r, rho = sp.r, sp.rho
    a_pos = B[sp.pos] / sp.s[:, None]
    a_perp = np.sqrt(np.maximum(cc - np.einsum("kj,kj->j", a_pos, a_pos), 0.0))
    ab = np.vstack([a_pos, a_perp[None, :]])
    na = np.sqrt(np.einsum("kj,kj->j", ab, ab))
    J = B.shape[1]
    omega = np.zeros(J)
    y = np.where(na[None, :] > 0, r * ab / np.where(na > 0, na, 1.0)[None, :], 0.0)
    prior = (na == 0) | (_misfit(sp, y) <= rho ** 2)
    live = np.flatnonzero(~prior)
    if len(live):
        abl = ab[:, live]
        if start is None:                                         # a few columns by the bracketing route set the scale
            few = np.linspace(0, len(live) - 1, min(8, len(live))).astype(int)
            yf, of = _outer(sp, abl[:, few])
            _, mf = _solve(sp, abl[:, few], of)
            start = (float(np.exp(np.median(np.log(np.maximum(mf, 1e-300))))), float(np.exp(np.median(np.log(of)))))
        mu0, om0 = np.full(len(live), start[0]), np.full(len(live), start[1])
        yl, om, mu, conv = _joint(sp, abl, mu0, om0)
        if (~conv).any():                                         # the bracketing route for the rest
            idx = np.flatnonzero(~conv)
            yl[:, idx], om[idx] = _outer(sp, abl[:, idx])
        y[:, live], omega[live] = yl, om
        if conv.any():
            start = (float(np.exp(np.median(np.log(mu[conv])))), float(np.exp(np.median(np.log(om[conv])))))
    # certificates and witnesses, exactly as SpectralBall._finish does for one direction
    yp = y[:-1]
    val = const + np.einsum("kj,kj->j", ab, y)
    res = sp.dp[:, None] - sp.s[:, None] * yp
    m = np.sqrt(np.maximum(np.einsum("kj,kj->j", res, res) + sp.f0, 0.0))
    feas = (np.einsum("kj,kj->j", y, y) <= r * r * (1 + 1e-9)) & (np.maximum(m - rho, 0.0) <= 1e-7 * np.maximum(rho, m))
    xpos = -omega[None, :] * res                                   # x on the positive eigenvalues
    xzero = -omega[None, :] * sp.delta[~sp.pos][:, None]
    v = np.sqrt(a_perp ** 2 + np.einsum("kj,kj->j", a_pos - sp.s[:, None] * xpos, a_pos - sp.s[:, None] * xpos))
    xd = sp.delta[sp.pos] @ xpos + sp.delta[~sp.pos] @ xzero
    xn = np.sqrt(np.einsum("kj,kj->j", xpos, xpos) + np.einsum("kj,kj->j", xzero, xzero))
    up = const + xd + r * v + rho * xn
    lo = np.where(feas, val, -math.inf)
    scale = TOL * np.maximum(1.0, np.abs(val))
    ok = (up - lo) <= scale
    st = np.where(ok, "EXACT", "BRACKET").astype(object)
    if sp._check and ok.any():                         # the general rule for EXACT (optimisers/dual/exactness.py)
        X = np.zeros((len(sp.delta), J))
        X[sp.pos], X[~sp.pos] = xpos, xzero
        E1 = exactness.first_order(sp, X, v * v, cc, terr)
        E1r, Es = sp._E1
        E = np.where(ok, E1, np.nan)
        js = np.flatnonzero(ok & np.isfinite(val) & (E1 > exactness.GRAY * scale) & np.isfinite(Es))
        if len(js):                                    # the re-solves, as arrays (exactness.judge for one direction)
            s = exactness.spread_many(sp, B[:, js], cc[js], const[js], val[js])
            E[js] = np.maximum(np.where(np.isfinite(E1r[js]), E1r[js], 0.0), s) + Es[js]
        for j in np.flatnonzero(ok & (E > scale)):
            st[j], lo[j], up[j] = "BRACKET", lo[j] - exactness.WIDEN * E[j], up[j] + exactness.WIDEN * E[j]
        if error is not None:
            error[:] = E
    value[:], lower[:], upper[:] = val, lo, up
    status[:] = st
    return ~ok, start


def _joint(sp, ab, mu, om, steps=NEWTON):
    """Newton's method on the dual g(mu, omega) in (log mu, log omega) for every column of ab (module docstring):
    (y, omega, mu, converged)."""
    J = ab.shape[1]
    r2, rho2 = sp.r ** 2, sp.rho ** 2
    b, s2 = sp.bb[:, None], sp.s2[:, None]
    lin = 0.5 * (rho2 - sp.dd)

    def at(m, w, cols):
        D = s2 * w[None, :]
        D += m[None, :]
        c = b * w[None, :]
        c += ab if cols is None else ab[:, cols]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            y = c / D
            g = 0.5 * np.einsum("kj,kj->j", c, y) + 0.5 * m * r2 + w * lin
        return D, y, g

    D, y, g = at(mu, om, None)
    conv = np.zeros(J, bool)
    for _ in range(steps):
        act = np.flatnonzero(~conv & np.isfinite(g))
        if not len(act):
            break
        whole = len(act) == J                                        # no copies while every column is active
        Da, ya = (D, y) if whole else (D[:, act], y[:, act])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            yy = ya * ya
            g1 = 0.5 * (r2 - yy.sum(axis=0))
            g2 = 0.5 * (rho2 - _misfit(sp, ya))
            q = ya / Da
            e = s2 * ya
            np.subtract(b, e, out=e)                                 # e = bb - s^2 y
            h11 = np.einsum("kj,kj->j", ya, q)
            h12 = -np.einsum("kj,kj->j", e, q)
            e /= np.sqrt(Da)
            h22 = np.einsum("kj,kj->j", e, e)
            det = h11 * h22 - h12 * h12
            d1 = -(h22 * g1 - h12 * g2) / det                        # the Newton step in (mu, omega)
            d2 = -(h11 * g2 - h12 * g1) / det
            dec = -(g1 * d1 + g2 * d2)                               # the Newton decrement squared: g - g* ~ dec / 2
        m, w, ga = mu[act], om[act], g[act]
        good = np.isfinite(dec) & (dec >= 0)
        done = good & (np.abs(g1) <= 0.5e-13 * r2) & (np.abs(g2) <= 0.5e-13 * rho2)   # both constraints active
        conv[act[done]] = True
        go = good & ~done
        if not go.any():
            continue
        du = np.clip(np.where(go, d1 / m, 0.0), -4.0, 4.0)          # log mu and log omega steps (bounded)
        dv = np.clip(np.where(go, d2 / w, 0.0), -4.0, 4.0)
        slope = g1 * m * du + g2 * w * dv                            # directional derivative (< 0)
        t = np.ones(len(act))
        todo = np.flatnonzero(go)
        for _ in range(40):                                          # backtracking (Armijo) on the convex g
            if not len(todo):
                break
            mt, wt = m[todo] * np.exp(t[todo] * du[todo]), w[todo] * np.exp(t[todo] * dv[todo])
            Dt, yt, gt = at(mt, wt, None if len(todo) == J else act[todo])
            slack = 1e-13 * np.abs(ga[todo])       # g's own rounding: near the optimum the full step is taken
            accept = np.isfinite(gt) & (gt <= ga[todo] + 1e-4 * t[todo] * slope[todo] + slack)
            k = act[todo[accept]]
            mu[k], om[k], g[k] = mt[accept], wt[accept], gt[accept]
            if len(k) == J:
                D, y = Dt, yt
            else:
                D[:, k], y[:, k] = Dt[:, accept], yt[:, accept]
            t[todo[~accept]] *= 0.5
            todo = todo[~accept]
    return y, om, mu, conv


def _outer(sp, ab):
    """omega and y(omega) with misfit(y) = rho^2 (from the feasible side) for every column of ab."""
    J = ab.shape[1]
    rho2 = sp.rho ** 2
    hi = np.ones(J)
    for _ in range(300):                                      # misfit(hi) <= rho^2
        y, _ = _solve(sp, ab, hi)
        bad = _misfit(sp, y) > rho2
        if not bad.any():
            break
        hi = np.where(bad, hi * 4.0, hi)
    lo = hi / 4.0
    for _ in range(1100):                                     # misfit(lo) > rho^2
        y, _ = _solve(sp, ab, lo)
        good = (_misfit(sp, y) <= rho2) & (lo > 1e-300)
        if not good.any():
            break
        hi = np.where(good, lo, hi)
        lo = np.where(good, lo / 4.0, lo)
    om = np.sqrt(lo * hi)
    for _ in range(200):
        y, mu = _solve(sp, ab, om)
        g = _misfit(sp, y) - rho2                             # decreasing in omega
        hi = np.where(g <= 0, om, hi)
        lo = np.where(g > 0, om, lo)
        dg = _dmisfit(sp, ab, y, om, mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = om - g / dg
        inside = np.isfinite(newton) & (newton > lo) & (newton < hi)
        om = np.where(inside, newton, np.sqrt(lo * hi))
        if np.all(((hi - lo) <= 1e-15 * hi) | (np.abs(g) <= 1e-15 * rho2)):
            break
    # Newton may converge from the infeasible side (g > 0 at every iterate), leaving hi at the bracket's end: step just
    # past the converged iterate and keep it as hi where the misfit is feasible there
    for eps in (1e-13, 1e-11, 1e-9):
        cand = np.minimum(lo * (1.0 + eps), hi)
        far = cand < hi
        if not far.any():
            break
        y, _ = _solve(sp, ab, cand)
        hi = np.where(far & (_misfit(sp, y) <= rho2), cand, hi)
    y, _ = _solve(sp, ab, hi)                                 # the feasible end of the bracket
    return y, hi


def _dmisfit(sp, ab, y, om, mu):
    """d misfit(y(omega)) / d omega (implicit differentiation through the trust-region condition)."""
    D = sp.s2[:, None] * om[None, :] + mu[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(D > 0, 1.0 / D, 0.0)
    g = sp.bb[:, None] - sp.s2[:, None] * y                        # (bb - s2 y)
    act = mu > 0
    num = np.einsum("kj,kj->j", y * g, inv)
    den = np.einsum("kj,kj->j", y * y, inv)
    with np.errstate(divide="ignore", invalid="ignore"):
        dmu = np.where(act & (den > 0), num / den, 0.0)
    dy = (g - y * dmu[None, :]) * inv
    return -2.0 * np.einsum("kj,kj->j", g, dy)
