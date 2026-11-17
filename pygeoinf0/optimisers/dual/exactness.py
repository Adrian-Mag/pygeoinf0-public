"""When a bound in the eigenbasis of a data-space decomposition is EXACT (plan 054, the author's general rule): only if
(the sensitivity of the bound to its input errors) x (the stated size of those errors) <= core.precision.TOL max(1, |value|).

The bracket [lower, upper] of optimisers/dual/spectral.py is certified for the decomposition in hand. That decomposition is the
exact one of a whitened Gram W + dW with dW the stated input error, not of W itself, so the bound of the problem
meant can differ from the computed one by E, the change that dW makes. EXACT needs both the gap and E within TOL.

The input error (stated, per block b of the decomposition, n_b rows, largest eigenvalue L_b):
    ||dW_b|| <= C eps n_b L_b               the eigendecomposition's backward error, which covers the rounding of the
                                            Gram's entries; exact zeros between blocks stay exact
    ||d beta_b|| <= C eps n_b sqrt(L_b cc)   the cross pairings <g, S c> in eigen coordinates
    ||d delta_b|| <= C eps n_b ||delta_b||   the whitened datum; |d cc| <= C eps cc
A decomposition that does not state blocks (a given spectrum) has the componentwise rounding of its stored entries:
every coordinate is a block of one, ||dW_k|| = eps Lam_k. Nothing here depends on what the blocks are: an operator
with genuinely smaller errors (exact zeros, blocks it states) passes because its stated error is smaller.

The sensitivity (first order, the envelope theorem): the bound is min over the multiplier x of
    phi(x) = <x, delta> + r v(x) + rho ||x||,   v(x)^2 = cc - 2 beta.x + x^T Lam x,
so at the optimal x a change dW, d beta, d delta, d cc moves it by r d(v^2) / (2 v) + <x, d delta> + ..., at most
    E1 = r e2 / v + sum_b C eps n_b ||delta_b|| ||x_b|| + C eps rho ||x||,
    e2 = sum_b C eps n_b (L_b ||x_b||^2 + 2 sqrt(L_b cc) ||x_b||) + C eps cc,
and r e2 / v becomes r sqrt(e2) where e2 >= v^2 (v itself has no correct digit). Both bound the change of r v
exactly: sqrt(v^2 - e2) >= v - e2 / v while e2 <= v^2, and |sqrt(v'^2) - v| <= sqrt(e2) for any |v'^2 - v^2| <= e2,
so the certificate side is first order only through x. The former E1 = inf there left E to the re-solves alone: an
estimable target (v ~ 0, a_perp^2 = cc - |a_pos|^2 a few ulps either side of 0) then got E 1e-14 or 6e-7 by which side
the three re-solves' rounding fell, EXACT or BRACKET by linear-algebra library (plan 054, the Mac's Accelerate;
tests/test_error_account_blas.py).

Near degeneracy first order fails (eigenvalues at the floor change which directions count): measured, the change under
random perturbations of the stated size was up to 160 x E1 (P2's Gaussian kernels, cond(W) 2.6e3 to beyond double;
/disks/data/tmp/mission-054-report.md, "General rule"). So wherever E1 > GRAY x TOL max(1, |value|) the bound is solved
again on RESOLVES problems whose blocks are the exact eigendecompositions of W_b + dW_b (dW_b random symmetric of the
stated size, a fixed seed: reproducible), and E = max(E1, the largest change). Below GRAY nothing more is done: the
largest measured ratio, 160, leaves E below 0.2 TOL there.

No prior (r = inf; plan 054, pre-existing bugs). The dual has the constraint G* x = c instead of the term r v(x), so
the bound is finite iff the direction is estimable, a_perp = 0 with a_perp^2 = cc - beta^T Lam^-1 beta. That is decided
by the same rule: a_perp^2 is UNBOUNDED's evidence only where it exceeds its stated error (perp_error: e2 at the
multiplier mu0 = Lam^-1 beta, plus the subtraction's rounding). Within it the stated error admits an estimable target
(the finite bound) and a non-estimable one (+inf), so E = inf: the finite value is computed, the bracket is infinite and
the note names the declaration (the reduce route's lifted re-solve gives the same, optimisers/dual/exactness_reduce.py). No method
can tell the two from rounded numbers, so it is decided by how the target was made (core/estimable.py, the author's
decision of 2026-11-06): built inside pygeoinf0 from the data functionals (G.adjoint(y), a datum's functional, their
combinations, a map L @ G), or declared by the caller (estimable=True; the caller's responsibility). K.dependent
declares the data exactly dependent and says nothing about targets. Declared or built, a_perp is taken as 0 where it is
within its stated error, and refused beyond it (exactness.contradicted: the numbers contradict the declaration or the
construction; neither: UNBOUNDED); the finite bound is judged by first order through the constraint: with the model m* = R w (w = y / sqrt(Lam) on the data representers) and z = (1, -x)
on the joint Gram J = [[cc, beta^T], [beta, Lam]], a change dJ moves the value by w^T dJ z exactly to first order (no
factor 1/2 here: R z = 0), at most
    E1 = sum_b C eps n_b ||w_b|| (L_b ||x_b|| + sqrt(L_b cc)) + (the datum's term, as above),
with the re-solves above GRAY as for a prior ball. The former test, a_perp > 1e-12 |a|, met a_perp = sqrt(eps) |t| from
the subtraction's rounding at any conditioning (/disks/data/tmp/mission-054-report.md, "Pre-existing bugs and
consistency"). Where data directions sit at the rounding floor the bound is refused first (SpectralBall._none,
reduction.floor_refusal), the same rule's outcome where the stated error admits exactly dependent and independent data with
different bounds (E = inf).

Where E > TOL max(1, |value|): EXACT becomes BRACKET, the bracket (of any status) is widened by WIDEN x E on both
sides, and the note names the choices (a Gram in higher precision, not in pygeoinf0 yet; a bound that does not need
these digits, such as a wider noise or prior ball, a mathematical choice).

The representers' own stated errors (plan 054, truncation round). The data's kernels k_i and the target's t may carry
stated realisation errors ||k~_i - k_i|| <= e_i, ||t~ - t|| <= e_t (Family.errors(), Function.error: estimates where a
realisation truncates, as the sphere's harmonic degree cap does; tabulated fits). With the whitening F (F C F^T = I) the
whitened representers h = F k move by ||dH|| <= ||F|| ||e|| =: E_h (|<dH a, m>| <= sum_j |(F^T a)_j| e_j ||m||), and
with s = ||S^(1/2)|| (the middle operator; 1 for the model metric), A = s E_h, B = s e_t and L = ||W|| = max Lam:
    ||dW|| <= 2 A sqrt(L) + A^2,   ||d beta|| <= A sqrt(cc) + B sqrt(L) + A B,   |d cc| <= 2 B sqrt(cc) + B^2,
and a prior centre m0 moves the whitened datum by at most E_h ||m0||. With no prior, a target built from the data
functionals (core/estimable.py: by construction, t = G* y of the realised kernels themselves) has the bound y.d + (the
noise term) whatever the kernels are, so their stated errors do not enter there (a declared target's realisation is the
caller's, so they do). They enter e2 (d(v^2) <= ||dW|| ||x||^2 + 2 ||d beta||
||x|| + |d cc|), the datum's term, perp_error and first_order_none, added to the rounding terms (the two errors are of
different origin and both present). Not stated (no error, or NaN: a formula in H^s): only rounding. The re-solves
perturb the rounding part only (first order carries the rest); the reduce route states the same with the columns'
errors in J's correlation units (optimisers/dual/exactness_reduce.py).

The constants are measured choices, kept by the author's decision (2026-11-06), not derived:
    C = 1           the stated errors in units of eps n_b (eps q for the reduce route, optimisers/dual/exactness_reduce.py)
    GRAY = 1e-3     first order was measured up to 160 x below the re-solves' change, which leaves E < 0.2 TOL below it
    RESOLVES = 3    perturbed problems: E is a 3-sample estimate of the change, not a bound
    WIDEN = 2       the safety factor of the widened bracket: widened by E alone, 1 of 13 ends missed the exact value
                    by 4.4e-3 TOL (P2's Gaussian kernels; /disks/data/tmp/mission-054-report.md, "General rule")
The same constants and the same rule apply in the reduce route (optimisers/dual/exactness_reduce.py)."""
import copy
import math

import numpy as np

from ...core.precision import TOL

EPS = np.finfo(float).eps
C = 1.0             # the rounding constant of the stated errors (unit: eps n_b)
GRAY = 1e-3         # E1 above GRAY x TOL max(1, |value|): checked by re-solves
RESOLVES = 3        # perturbed problems
SEED = 54
WIDEN = 2.0         # the widened bracket: WIDEN x E on both sides


def blocks(sp):
    """(starts, n_b, L_b, ||delta_b||) of the problem's decomposition, cached on the problem."""
    out = getattr(sp, "_blocks", None)
    if out is None:
        dec = sp.dec
        lam_b = getattr(dec, "_lam_b", None)
        if lam_b is not None and hasattr(dec, "offsets"):
            starts = np.asarray(dec.offsets[:-1], int)
            nb = np.array([len(b) for b in lam_b], float)
            L = np.array([float(np.abs(b).max(initial=0.0)) for b in lam_b])
        else:                                                  # a given spectrum: componentwise
            lam = np.asarray(dec.lam, float)
            starts, nb, L = np.arange(len(lam)), np.ones(len(lam)), np.abs(lam)
        d2 = np.add.reduceat(sp.delta ** 2, starts) if len(starts) else np.zeros(0)
        out = sp._blocks = (starts, nb, L, np.sqrt(d2))
    return out


def kernel_errors(A):
    """The stated realisation errors of the representers of a map A (Functionals: its kernels, Family.errors()), an
    array, or None where none is stated (no error beyond rounding, symbolic representers, or unknown: NaN)."""
    from ...operators.functionals import Functionals, symbolic_representers
    if not isinstance(A, Functionals) or symbolic_representers(A.kernels):
        return None
    e = np.asarray(A._kerr, float)
    return e if e.size and np.all(np.isfinite(e)) and e.max(initial=0.0) > 0 else None


def element_error(c):
    """The stated error of one model element c (Function.error), or 0.0."""
    e = getattr(c, "error", None)
    try:
        e = float(e)
    except (TypeError, ValueError):
        return 0.0
    return e if math.isfinite(e) and e > 0 else (math.inf if e == math.inf else 0.0)


def target_errors(tg, T=None):
    """The stated realisation errors of a target pairing's representers (p,) (tg.T's kernels, or the model element T
    itself: [e]), or None where none is stated; cached on tg."""
    if "err" not in vars(tg):
        from ...operators.linear import LinearOperator
        src = tg.T if tg.T is not None else T
        if isinstance(src, LinearOperator):
            tg.err = kernel_errors(src)
        else:
            e = element_error(src) if src is not None else 0.0
            tg.err = np.array([e]) if e else None
    return tg.err


def direction_error(tg, w=None, T=None):
    """The stated representer error of the direction with property weights w (|w| @ the targets' errors; 0.0 where none
    is stated), or of every property (w None: an array, or None)."""
    err = target_errors(tg, T)
    if w is None:
        return err
    return 0.0 if err is None else float(np.abs(np.asarray(w, float)) @ err)


def whitening_norm(wt, n):
    """||F_b||_2 of one block's whitening (solvers/dataspace/whitening._White)."""
    if wt.kind == "diag":
        return float(np.max(np.abs(wt.data), initial=0.0))
    if wt.kind == "chol":
        sv = np.linalg.svd(np.asarray(wt.data, float), compute_uv=False)
        return 1.0 / float(sv.min()) if sv.size and sv.min() > 0 else math.inf
    D = wt.D
    if D.is_standard:
        return 1.0
    if D.weights is not None:
        return float(np.sqrt(np.max(D.weights)))
    return float(np.sqrt(np.linalg.eigvalsh(D.metric_matrix()).max()))


def stated(dec):
    """(E_h, s, L) of the data representers' stated errors for a decomposition (module docstring), or None where none is
    stated; cached on dec."""
    out = getattr(dec, "_stated", 0)
    if out != 0:
        return out
    out = None
    G = getattr(dec, "G", None)
    e = kernel_errors(G) if G is not None and getattr(dec, "normal", None) is None else None
    if e is not None and len(e) == dec.n:
        D = dec.D
        kappa = 1.0                                          # D's metric W: the plain Gram's kernels against F
        if not D.is_standard:
            w = np.asarray(D.weights, float) if D.weights is not None else np.linalg.eigvalsh(D.metric_matrix())
            kappa = float(w.max() / max(w.min(), 1e-300))
        Eh = math.sqrt(sum((whitening_norm(wt, len(p)) * float(np.linalg.norm(e[p]))) ** 2
                           for p, wt in zip(dec._rows, dec._white))) * kappa
        out = (Eh, middle_norm(dec), float(np.max(dec.lam, initial=0.0)))
    dec._stated = out
    return out


def middle_norm(dec):
    """s = ||S^(1/2)|| of the decomposition's middle operator (1 for the model metric; exact for c I, else a power
    iteration estimate, SAFETY times), cached on dec."""
    s = getattr(dec, "_stated_s", None)
    if s is None:
        S = dec.middle
        if S is None:
            s = 1.0
        else:
            from ...operators.error import power_estimate, scaled_identity
            c = scaled_identity(S)
            G = dec.G
            s = math.sqrt(abs(c)) if c is not None else \
                math.sqrt(power_estimate(S, G.adjoint(dec.D(np.ones(dec.n))), G.domain))
        dec._stated_s = s
    return s


def built(sp):
    """A target built from the data functionals with no prior (core/estimable.py: by construction): t = G* y for the
    realised kernels themselves, so the bound y.d + (the noise term) does not depend on the kernels and their stated
    errors do not move it (module docstring)."""
    from ...core.estimable import BUILT
    return not math.isfinite(sp.r) and getattr(sp, "_how", None) == BUILT


def coefficients(sp, terr):
    """(E_h, s, L) for the stated terms, or None where neither the data's representers nor the targets (terr) state
    an error."""
    st = stated(sp.dec)
    if st is not None:
        return st
    if not np.any(np.asarray(0.0 if terr is None else terr, float)):
        return None
    return 0.0, middle_norm(sp.dec), float(np.max(sp.dec.lam, initial=0.0))


def stated_terms(sp, nx, nx2, cc, terr):
    """(e2, datum term) of the representers' stated errors (module docstring) for multiplier norms nx (J), their
    squares nx2, cc (J) and the targets' errors terr (J or a number); zeros where nothing is stated."""
    co = coefficients(sp, terr)
    if co is None:
        return 0.0, 0.0
    Eh, s, L = co
    terr = np.broadcast_to(np.asarray(0.0 if terr is None else terr, float), np.shape(nx))
    A, B, rL = s * Eh, s * terr, math.sqrt(max(L, 0.0))
    rc = np.sqrt(np.maximum(cc, 0.0))
    with np.errstate(invalid="ignore"):
        dW = 2.0 * A * rL + A * A
        db = A * rc + B * rL + A * B
        dc = 2.0 * B * rc + B * B
        e2 = dW * nx2 + 2.0 * db * nx + dc
        e2 = np.where(np.isnan(e2), math.inf, e2)
    return e2, Eh * getattr(sp, "_m0n", 0.0) * nx


def first_order(sp, X, v2, cc, terr=None):
    """E1 for the multipliers X (n_eff, J) in eigen coordinates (every coordinate, the zero ones too), v^2 (J), cc (J)
    and the targets' stated errors terr (J, or None) (module docstring)."""
    starts, nb, L, dn = blocks(sp)
    X = np.asarray(X, float).reshape(len(sp.delta), -1)
    v2, cc = np.atleast_1d(np.asarray(v2, float)), np.maximum(np.atleast_1d(np.asarray(cc, float)), 0.0)
    nx2 = np.add.reduceat(X * X, starts, axis=0)
    nx = np.sqrt(nx2)
    w = C * EPS * nb
    e2 = (w * L) @ nx2 + 2.0 * np.sqrt(cc) * ((w * np.sqrt(L)) @ nx) + C * EPS * cc
    nxt = np.sqrt(nx2.sum(axis=0))
    rest = (w * dn) @ nx + C * EPS * sp.rho * nxt
    t2, tr = stated_terms(sp, nxt, nxt * nxt, cc, terr)
    with np.errstate(divide="ignore", invalid="ignore"):
        sv = np.sqrt(np.where(v2 > 0, v2, 1.0))
        # min(x / |v|, sqrt(x)) bounds |sqrt(v^2 + x) - |v||: where v has no correct digit E1 is r sqrt(e2), a bound
        r_round = sp.r * np.where(e2 < v2, e2 / sv, np.sqrt(e2)) + rest
        et = e2 + t2
        total = sp.r * np.where(et < v2, et / sv, np.sqrt(et)) + rest + tr
    none = (np.asarray(t2) + np.asarray(tr)) == 0                   # nothing stated: E1 is the rounding one
    sp._E1 = (r_round, np.where(none, 0.0, total - r_round))
    return total


def perp_error(sp, mu0, cc, a_pos, terr=None):
    """The stated error of a_perp^2 = cc - |a_pos|^2 (no prior): e2 of first_order at the multiplier mu0 = Lam^-1 beta
    (d(a_perp^2) = d cc - 2 mu0 . d beta + mu0^T dW mu0), plus the rounding of the subtraction itself,
    C eps (cc + |a_pos|^2). Where a_perp^2 is within it, the stated input error admits an estimable target and a
    non-estimable one (module docstring, "No prior")."""
    starts, nb, L, _ = blocks(sp)
    x = np.zeros(len(sp.delta))
    x[sp.pos] = mu0
    nx2 = np.add.reduceat(x * x, starts) if len(starts) else np.zeros(0)
    w = C * EPS * nb
    cc = max(float(cc), 0.0)
    e2 = float((w * L) @ nx2 + 2.0 * math.sqrt(cc) * ((w * np.sqrt(L)) @ np.sqrt(nx2)) + C * EPS * cc)
    nm = float(np.linalg.norm(x))
    t2, _ = stated_terms(sp, np.array([nm]), np.array([nm * nm]), np.array([cc]), terr)
    return e2 + float(np.ravel(t2)[0]) + C * EPS * (cc + float(a_pos @ a_pos))


def first_order_none(sp, x, yp, cc, terr=None):
    """E1 with no prior (module docstring, "No prior"): the model's weights on the data representers w = y / sqrt(Lam)
    (yp: the witness on the positive eigenvalues), the multiplier x (every coordinate): sum_b C eps n_b ||w_b||
    (L_b ||x_b|| + sqrt(L_b cc)) + the datum's term, as in first_order."""
    starts, nb, L, dn = blocks(sp)
    W = np.zeros(len(sp.delta))
    W[sp.pos] = np.asarray(yp, float) / sp.s
    X = np.asarray(x, float)
    nw = np.sqrt(np.add.reduceat(W * W, starts))
    nx = np.sqrt(np.add.reduceat(X * X, starts))
    w = C * EPS * nb
    cc = max(float(cc), 0.0)
    lead = float((w * nw) @ (L * nx + np.sqrt(L * cc)))
    out = r_round = lead + float((w * dn) @ nx) + C * EPS * sp.rho * float(np.linalg.norm(X))
    tv = 0.0 if terr is None else float(terr)
    co = None if built(sp) else coefficients(sp, tv)
    if co is not None:                          # the representers' stated errors: |w . (d beta - dW x)| + the datum's
        Eh, s, Lm = co
        A, B, rL, nX = s * Eh, s * tv, math.sqrt(max(Lm, 0.0)), float(np.linalg.norm(X))
        dW, db = 2.0 * A * rL + A * A, A * math.sqrt(cc) + B * rL + A * B
        out += float(np.linalg.norm(W)) * (db + dW * nX) + Eh * getattr(sp, "_m0n", 0.0) * nX
    sp._E1 = (np.array([r_round]), np.array([out - r_round]))
    return out


def _perturbed(sp):
    """RESOLVES problems of the stated input error's size around sp, with the map of eigen coordinates into each. The
    perturbed decompositions depend on the decomposition alone (a fixed seed), so they are built once per decomposition
    and shared by every problem on it (a map's centres, a run of data draws); only the datum is moved."""
    out = getattr(sp, "_perturbed", None)
    if out is not None:
        return out
    from .spectral import SpectralBall
    out = []
    for d2, move in _perturbed_decompositions(sp.dec):
        p = SpectralBall(d2, None, sp.r, sp.rho, sp.dependent, delta=move(sp.delta))
        p._check = False
        out.append((p, move))
    sp._perturbed = out
    return out


def _perturbed_decompositions(dec):
    """RESOLVES decompositions whose blocks are the exact eigendecompositions of W_b + dW_b (module docstring), with the
    map of eigen coordinates into each; cached on dec."""
    out = getattr(dec, "_perturbed_decs", None)
    if out is not None:
        return out
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(RESOLVES):
        d2 = copy.copy(dec)
        if getattr(dec, "_U", None) is not None:
            lam_b, U_b, rot = [], [], []
            for lam, U in zip(dec._lam_b, dec._U):
                nb, Lb = len(lam), float(np.abs(lam).max(initial=0.0))
                Z = rng.standard_normal((nb, nb))
                Z = Z + Z.T
                nz = np.linalg.norm(Z, 2)
                Z *= C * EPS * nb * Lb / (nz if nz > 0 else 1.0)
                W = (U * lam) @ U.T + Z
                l2, U2 = np.linalg.eigh(0.5 * (W + W.T))
                lam_b.append(l2)
                U_b.append(U2)
                rot.append(U2.T @ U)
            d2._lam_b, d2._U, d2.lam = lam_b, U_b, np.concatenate(lam_b)
            d2._targets = {}
            offs = dec.offsets

            def move(v, rot=rot, offs=offs):
                v = np.asarray(v, float)
                return np.concatenate([R @ v[a:b] for R, a, b in zip(rot, offs[:-1], offs[1:])])
        else:
            lam = np.asarray(dec.lam, float)
            d2.lam = lam * (1.0 + C * EPS * rng.choice([-1.0, 1.0], len(lam)))

            def move(v):
                return np.asarray(v, float)
        out.append((d2, move))
    dec._perturbed_decs = out
    return out


def spread(sp, beta, cc, const, value):
    """The largest change of the bound over the perturbed problems (inf where one is not finite)."""
    from ...core.errors import Refused
    worst = 0.0
    for p, move in _perturbed(sp):
        try:
            b = p.bound(move(beta), cc, const, sp._how)
        except Refused:                     # the perturbed problem is at the rounding floor: another answer is admitted
            return math.inf
        worst = max(worst, abs(b.value - value)) if math.isfinite(b.value) else math.inf
    return worst


def spread_many(sp, B, cc, const, value):
    """spread for the directions B (columns) at once, each perturbed problem solved as arrays
    (optimisers/dual/spectral.values): the largest change of each bound, inf where one is not finite or refused."""
    from .spectral import values
    worst = np.zeros(B.shape[1])
    for p, move in _perturbed(sp):
        p._how = sp._how
        v = values(p, move(B), cc, const)
        worst = np.where(np.isfinite(v), np.maximum(worst, np.abs(v - value)), math.inf)
    return worst


def judge(sp, beta, cc, const, value, E1, stated_part=0.0):
    """E (module docstring) for one direction whose first-order estimate is E1, of which stated_part comes from the
    representers' stated errors; re-solves where needed. The re-solves perturb the rounding part only, so they replace
    that part's first-order estimate and the stated part is added (inf stays inf: the stated error admits v = 0)."""
    scale = TOL * max(1.0, abs(value))
    if not math.isfinite(value) or E1 <= GRAY * scale:
        return E1
    if not math.isfinite(stated_part):
        return math.inf
    E1r = E1 - stated_part if math.isfinite(E1) else E1
    s = spread(sp, beta, cc, const, value)
    return max(E1r if math.isfinite(E1r) else 0.0, s) + stated_part


CHOICES = ("The choices: a Gram in higher precision (not available in pygeoinf0 yet), or a bound that does not need "
           "these digits (a wider noise or prior ball, a mathematical choice)")


DECLARE = ("where the target is exactly in the data's span (an estimable target, a datum's functional), build it from "
           "the data functionals (G.adjoint(y), G[k].kernel, a map L @ G: estimable by construction) or declare it: "
           "K.bound(c, estimable=True), K.image(T, estimable=True) or P.estimable = True (core/estimable.py; "
           "K.dependent declares the data dependent, not the target); or a prior bound")


def note_estimable(perp2, e2):
    return (f"; not EXACT: the direction lies in the data's span only within the stated input errors (its part outside, "
            f"a_perp^2 = {perp2:.3g}, is at most its stated error {e2:.3g}; optimisers/dual/exactness.py, 'No prior'), which "
            "admit a non-estimable direction, bound +inf: the finite value assumes it estimable, the bracket is "
            "infinite; the target is neither declared estimable nor built from the data functionals. The choices: "
            + DECLARE)


def contradicted(how, perp2, e2):
    """The refusal of a no-prior bound whose target is declared or built estimable (core/estimable.py) while its part
    outside the data's span exceeds its stated error: the numbers contradict the declaration or the construction."""
    from ...core.errors import Refused
    return Refused(f"no prior: the target is {how} estimable, but its part outside the data's span, a_perp^2 = "
                   f"{perp2:.3g}, exceeds its stated error {e2:.3g} (rounding): the numbers contradict it",
                   needs=["a target exactly in the data's span; without the declaration this bound is +inf "
                          "(UNBOUNDED)", "a prior bound (a mathematical choice)",
                          "built from the data functionals: the target's and the representers' realisations differ "
                          "beyond rounding; refine them (their stated errors)"])


def what(sp, terr=None):
    """The input errors the note names: rounding, and the representers' stated errors where there are any."""
    base = "the decomposition's backward error, eps n_b ||W_b|| per block"
    st = stated(sp.dec)
    tv = float(np.max(np.atleast_1d(0.0 if terr is None else terr), initial=0.0))
    parts = ([f"the data kernels' stated realisation errors (||dH|| <= {st[0]:.2g} whitened, estimates)"]
             if st is not None and st[0] > 0 else []) + ([f"the target's stated realisation error {tv:.2g}"] if tv else [])
    return base + ("; " + "; ".join(parts) if parts else "")


def note(E, value, what="the decomposition's backward error, eps n_b ||W_b|| per block", where="optimisers/dual/exactness.py"):
    tol = TOL * max(1.0, abs(value))
    return (f"; not EXACT: the stated input errors ({what}, and the rounding of the pairings and the datum) move this "
            f"bound by up to {E:.2g} > TOL max(1, |value|) = {tol:.2g} (first order through the multiplier, checked "
            f"by {RESOLVES} re-solves; {where}), and the bracket is widened by {WIDEN:g} x that. " + CHOICES
            + ("; or the representers realised more accurately (a smaller tol of the space, or a space whose "
               "realisation reaches the needed degree)" if "stated realisation" in what else ""))
