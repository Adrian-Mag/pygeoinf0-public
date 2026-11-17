"""The direct solvers (Cholesky, Direct, Eigh, LU, KroneckerEigh, BlockDirect): each computes the inverse or refuses, says what
it lost and warns when digits are at stake. What each is for: pg.solvers.usage(). Below them, what an inverse inherits from
the stated error of its operator (plan 054 P3 and the general rule): stated_error (first order, or a posteriori at the depth
pg.settings.verification sets), inherited (the statement an Inverse carries, and whether it refuses) and inherited_refusal."""
import math
import warnings

import numpy as np
import scipy.linalg as sla

from ..spaces.elements import Vector, VectorFamily
from ..core.errors import PrecisionWarning, Refused
from ..spaces.identity import require_realised

EPS = np.finfo(float).eps
WELL = 1e10                     # below this condition estimate Cholesky alone is enough; above it the spectrum is looked at


def _floor(lam):
    """(the count of eigenvalues with no correct digit, the condition number of the rest of them, the largest |lam|)."""
    a = np.abs(lam)
    top = float(a.max(initial=0.0))
    bad = a <= EPS * a.size * top
    return int(bad.sum()), (top / float(a[~bad].min()) if (~bad).any() else math.inf), top


def _digits(cond):
    return f"condition {cond:.3g}, so about {min(math.log10(max(cond, 1.0)), 16):.0f} of 16 digits lost"


def _refuse(name, n, bad, where="", raw="Direct"):
    return Refused(f"{name}: {bad} of {n} directions have an eigenvalue at most eps n times the largest{where}: in double "
                   "precision they hold no correct digit, so the inverse cannot be computed",
                   needs=["A.pseudo_inverse(rcond=...): the pseudo-inverse with a threshold you choose",
                          "a shift you choose, (A + s I).inv: a mathematical regularisation",
                          f"A.inverse(pg.{raw}(refuse=False)): the numbers double precision gives, with a warning that "
                          "those directions hold no correct digit",
                          "the operator in higher precision"])


def _warn(name, cond, bad=0):
    """The warning of a result whose digits are at stake: from a condition of WELL, and at the floor."""
    if bad:
        warnings.warn(f"{name}: {bad} directions hold no correct digit in double precision, and they are in the result; "
                      f"the others: {_digits(cond)}", PrecisionWarning, stacklevel=3)
    elif cond >= WELL:
        warnings.warn(f"{name}: {_digits(cond)} (rounding only; an error in the operator itself comes on top)",
                      PrecisionWarning, stacklevel=3)


def _about(computes, when, error, cost, simplification="none"):
    return dict(computes=computes, when=when, error=error, cost=cost, simplification=simplification)


class Solver:
    name = "solver"

    def factor(self, A):
        raise NotImplementedError

    def __repr__(self):
        return self.name


def _square_matrix(A):
    from ..spaces.base import Euclidean
    if not (isinstance(A.domain, Euclidean) and isinstance(A.codomain, Euclidean)):
        raise TypeError(f"{A.name} is not a map between Euclidean spaces; use CG for operators on function spaces")
    M = A.matrix()
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{A.name} is not square ({M.shape}); it has no inverse")
    return M


def orthonormal_form(A, M):
    """The matrix of A in orthonormal coordinates of its (common) domain and codomain: symmetric iff A is self-adjoint."""
    D = A.domain
    if D.is_standard:
        return M
    return D.to_orthonormal(M @ D.from_orthonormal(np.eye(D.dim)))


class _Factorisation:
    def __init__(self, A, solve_values, info):
        self.A, self._solve, self.info = A, solve_values, info

    def solve(self, y):
        if isinstance(y, Vector):
            return Vector(self.A.domain, self._solve(y.values), name="x")
        if isinstance(y, VectorFamily):
            return VectorFamily(self.A.domain, self._solve(y.values.T).T, name="X")
        raise TypeError(type(y).__name__)


class Cholesky(Solver):
    """Dense Cholesky factorisation (self-adjoint positive definite operators); with a condition estimate."""

    name = "Cholesky"
    about = _about("the inverse", "self-adjoint positive definite with a condition below 1e10: the cheapest direct method, and "
                   "what A.inv chooses there", "relative error near cond eps; the estimate is stated, and warned from 1e10",
                   "n^3/3 flops; dense: two n x n arrays (the matrix and its factor); sparse: the factor's fill-in")

    def __init__(self, factor=None, backend="dense"):
        from .backends import backend_named
        self._pre = factor             # (cho_factor result, operator) from recommend(), reused once
        self.backend = backend_named(backend)
        self.name = "Cholesky" if self.backend.name == "dense" else f"Cholesky(backend={self.backend.name!r})"

    def factor(self, A):
        D = A.domain
        if A.codomain != D:
            raise ValueError("Cholesky needs an operator from a space to itself")
        require_realised(A.codomain, D, A.name)
        if self.backend.name != "dense":
            return self.backend.cholesky(A)
        if self._pre is not None and self._pre[1] is A:
            c = self._pre[0]
        else:
            c = _cholesky(A, _square_matrix(A))
            if c is None:
                raise np.linalg.LinAlgError(f"Cholesky failed: {A.name} is not numerically positive definite; "
                                            "use A.inv, or A.pseudo_inverse(rcond) with a threshold you choose")
        cond = _cond_estimate(c)
        _warn(A.name, cond)
        return _Factorisation(A, lambda Y: sla.cho_solve(c, D.apply_metric(Y), check_finite=False),
                              [f"n = {D.dim}", _digits(cond)])


class _CholFactor(tuple):
    """cho_factor's (factor, lower), with the 1-norm of the factorised matrix for the condition estimate."""

    def __new__(cls, c, anorm):
        out = tuple.__new__(cls, c)
        out.anorm = anorm
        return out


def _norm1(S, block=512):
    """max_j sum_i |S_ij|, a block of columns at a time (no second n x n array)."""
    return max((float(np.abs(S[:, j:j + block]).sum(axis=0).max()) for j in range(0, S.shape[1], block)), default=0.0)


def _cholesky(A, M):
    """The Cholesky factor of the symmetric part of W M, or None when it fails. One n x n array besides M: the symmetric
    part is formed once and factorised in place; M itself is not changed."""
    S = A.domain.apply_metric(M)
    S = S + S.T
    S *= 0.5
    anorm = _norm1(S)
    try:
        return _CholFactor(sla.cho_factor(S, lower=True, overwrite_a=True, check_finite=False), anorm)
    except np.linalg.LinAlgError:
        return None


def _cond_estimate(c, S=None):
    """1-norm condition number estimate from a Cholesky factor (LAPACK pocon, O(n^2)), with the 1-norm of the factorised
    matrix S (carried by the factor from _cholesky)."""
    from scipy.linalg import lapack
    anorm = c.anorm if S is None else _norm1(S)
    rcond, info = lapack.dpocon(c[0], anorm, uplo="L" if c[1] else "U")
    return math.inf if rcond == 0 else 1.0 / rcond


class Direct(Solver):
    """The inverse of a self-adjoint operator of any conditioning: its spectrum is checked, then Cholesky where it succeeds
    (backward stable) or LU. Nothing is dropped; a direction with no correct digit is refused (module docstring)."""

    name = "Direct"
    about = _about("the inverse", "self-adjoint of any conditioning: what A.inv chooses from a condition of 1e10",
                   "relative error near cond eps, stated and warned; refused where a direction has no correct digit "
                   "(refuse=False returns the numbers, with a warning)", "the spectrum and a factorisation, about 2 n^3 flops")

    def __init__(self, refuse=True):
        self.refuse = refuse

    def factor(self, A):
        M = _square_matrix(A)
        D = A.domain
        Ah = orthonormal_form(A, M)
        lam = sla.eigvalsh(0.5 * (Ah + Ah.T), check_finite=False)
        bad, cond, _ = _floor(lam)
        if bad and self.refuse:
            raise _refuse(A.name, D.dim, bad)
        _warn(A.name, cond, bad)
        info = [f"n = {D.dim}", _digits(cond), "nothing dropped"] + ([f"{bad} directions hold no correct digit"] if bad else [])
        c = _cholesky(A, M)
        if c is not None:
            out = _Factorisation(A, lambda Y: sla.cho_solve(c, D.apply_metric(Y), check_finite=False), info)
        else:
            lu = sla.lu_factor(M, check_finite=False)
            out = _Factorisation(A, lambda Y: sla.lu_solve(lu, Y, check_finite=False), info)
        out.smallest = float(np.abs(lam).min(initial=math.inf))
        return out


class Eigh(Solver):
    """The pseudo-inverse: eigenvalues below rcond * max are treated as zero, a choice the caller makes. rcond = "rounding"
    drops only those at most eps n max (for an operator singular by construction)."""

    about = _about("the pseudo-inverse with a stated threshold, not the inverse", "the pseudo-inverse is what you want, or an "
                   "operator singular by construction", "near cond eps on the directions kept, cond counting only those; "
                   "the part of the datum in a dropped direction is lost, and the count is stated", "about 9 n^3 flops",
                   "the threshold: a mathematical choice that changes the object")

    def __init__(self, rcond):
        if rcond != "rounding" and not (isinstance(rcond, (int, float)) and rcond >= 0):
            raise ValueError(f"rcond must be a number >= 0 or 'rounding', not {rcond!r}")
        self.rcond = rcond
        self.name = "Eigh(rcond=rounding)" if rcond == "rounding" else f"Eigh(rcond={rcond:g})"

    def factor(self, A):
        M = _square_matrix(A)
        D = A.domain
        Ah = orthonormal_form(A, M)
        lam, V = np.linalg.eigh(0.5 * (Ah + Ah.T))
        cut = (EPS * lam.size if self.rcond == "rounding" else self.rcond) * np.abs(lam).max()
        keep = np.abs(lam) > cut
        inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
        info = [f"n = {D.dim}", f"rank {int(keep.sum())} of {D.dim}",
                f"eigenvalues {lam.min():.3g} to {lam.max():.3g}",
                f"{int((~keep).sum())} direction(s) below {cut:.3g} dropped ({self.name})"]

        def solve(Y):
            Z = D.to_orthonormal(Y)
            Z = V @ ((inv[:, None] if Z.ndim == 2 else inv) * (V.T @ Z))
            return D.from_orthonormal(Z)

        out = _Factorisation(A, solve, info)
        out.dropped, out.cut, out.n = int((~keep).sum()), float(cut), int(lam.size)
        return out


def rounding_claim(Ainv, what, extra=""):
    """The claim of a true pseudo-inverse taken with Eigh("rounding") (plan 054 P2): its threshold and its count."""
    from ..core.claims import DERIVED, Claim
    f = Ainv.factorisation()
    return Claim(f"{what}: the pseudo-inverse, eigenvalues with |lam| <= eps n max|lam| = {f.cut:.3g} dropped (no correct "
                 f"digit in double precision): {f.dropped} of {f.n}" + extra, DERIVED, "Eigh('rounding'), plan 054 P2")


class LU(Solver):
    """Dense LU factorisation (general square operators)."""

    name = "LU"
    about = _about("the inverse", "square and not self-adjoint; what A.inv chooses there, with refuse=True",
                   "relative error near cond eps with the singular values' condition, stated and warned from 1e10; given "
                   "explicitly it returns the numbers at the floor with a warning", "singular values and a factorisation, "
                   "about 5 n^3 flops")

    def __init__(self, refuse=False):
        self.refuse = refuse

    def factor(self, A):
        M = _square_matrix(A)
        bad, cond, _ = _floor(sla.svdvals(M, check_finite=False))
        if bad and self.refuse:
            raise _refuse(A.name, M.shape[0], bad, " (singular values)", raw="LU")
        _warn(A.name, cond, bad)
        lu = sla.lu_factor(M, check_finite=False)
        return _Factorisation(A, lambda Y: sla.lu_solve(lu, Y, check_finite=False), [f"n = {M.shape[0]}", _digits(cond)])


def _kron_parts(A):
    """(K, w, alpha) when A = c K W + alpha I with K a Kronecker Gram, W = w I a uniform diagonal metric; else None."""
    from ..operators.linear import Identity, Sum
    from ..operators.sparse import KroneckerMatrix

    def flat(op, c=1.0):
        if isinstance(op, Sum):
            out = []
            for a, B in op.terms:
                out += flat(B, c * a)
            return out
        return [(c, op)]

    terms = flat(A)
    km = [(c, B) for c, B in terms if isinstance(B, KroneckerMatrix)]
    ids = [(c, B) for c, B in terms if isinstance(B, Identity)]
    if len(km) != 1 or len(km) + len(ids) != len(terms):
        return None
    c, B = km[0]
    w = 1.0 if B.weights is None else B.weights
    if np.ndim(w) and not np.all(w == w[0]):
        return None
    return B.K, c * float(np.ravel(w)[0]), float(sum(a for a, _ in ids))


class KroneckerEigh(Solver):
    """(w K + alpha I) x = y for a Kronecker Gram K: exact, through the eigendecompositions of its small factors. Refused where
    a direction has no correct digit, unless rcond is given: the pseudo-inverse at alpha = 0, eigenvalues below rcond times
    the largest dropped."""

    about = _about("the inverse; with rcond at no shift, the pseudo-inverse", "a Kronecker Gram with uniform data weights, plus a "
                   "multiple of the identity", "near cond eps, stated and warned; refused at the floor unless rcond is given",
                   "O(N (n_1 + ... + n_d)) per solve, no N x N matrix", "none; rcond is a mathematical choice")

    def __init__(self, rcond=None):
        self.rcond = rcond
        self.name = "Kronecker eigendecomposition"

    def factor(self, A):
        parts = _kron_parts(A)
        if parts is None:
            raise TypeError(f"{A.name} is not (a multiple of) a Kronecker Gram operator plus a multiple of the identity "
                            "with uniform data weights")
        K, w, alpha = parts
        lam = 1.0
        for l, _ in K.eig():
            lam = np.multiply.outer(lam, l)
        info = [f"N = {K.N} = {' x '.join(map(str, K.dims))}", f"w = {w:.3g}, alpha = {alpha:.3g}"]
        bad, cond, _ = _floor(np.ravel(w * lam + alpha))
        if self.rcond is None:
            if bad:
                raise _refuse(A.name, K.N, bad, raw="Direct")
            _warn(A.name, cond)
            info.append(_digits(cond))
        else:
            info.append(f"pseudo-inverse with rcond {self.rcond:g}" if alpha <= 0 else _digits(cond))
        return _Factorisation(A, lambda Y: K.solve_shifted(Y, w, alpha, self.rcond or 0.0), info)


class BlockDirect(Solver):
    """Block by block, for a self-adjoint pg.BlockDiagonal of rows on standard R^n. The inverse: per block, Cholesky when its
    condition estimate is below WELL, else its spectrum is checked and it is solved with nothing dropped; a direction with no
    correct digit, relative to its own block, is refused (module docstring). With rcond: the pseudo-inverse, per block
    relative to that block, with the count stated. A block that repeats without a shift is factored once. Nothing n x n is
    formed (plan 031)."""

    name = "BlockDirect"
    about = _about("the inverse; with rcond, the pseudo-inverse", "blocks of rows on R^n or a diagonal metric, as for one harmonic "
                   "a block: what A.inv chooses there", "near cond eps of each block, the worst stated and warned; each block's "
                   "digits are its own, however far apart their scales; refused at the floor (refuse=False returns the numbers, "
                   "with a warning)", "the sum over the blocks of their cubes; nothing n x n",
                   "none; rcond is a mathematical choice")

    def __init__(self, rcond=None, refuse=True):
        self.rcond, self.refuse = rcond, refuse

    def _block(self, M):
        """(solve, directions with no correct digit, condition, directions dropped)."""
        c = None
        try:
            c = sla.cho_factor(M, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            pass
        if self.rcond is None and c is not None:
            cond = _cond_estimate(c, M)
            if cond < WELL:
                return (lambda Y: sla.cho_solve(c, Y, check_finite=False)), 0, cond, 0
        if self.rcond is not None:
            w, Q = sla.eigh(M, check_finite=False)
            keep = np.abs(w) > self.rcond * np.abs(w).max(initial=0.0)
            Qk, wk = Q[:, keep], w[keep]
            return (lambda Y: Qk @ ((Qk.T @ Y) / (wk[:, None] if Y.ndim == 2 else wk))), 0, 0.0, int((~keep).sum())
        bad, cond, _ = _floor(sla.eigvalsh(M, check_finite=False))
        if bad and self.refuse:
            return None, bad, cond, 0
        if bad:
            lost = bad
            if c is not None:
                return (lambda Y: sla.cho_solve(c, Y, check_finite=False)), lost, cond, 0
            lu = sla.lu_factor(M, check_finite=False)
            return (lambda Y: sla.lu_solve(lu, Y, check_finite=False)), lost, cond, 0
        if c is not None:
            return (lambda Y: sla.cho_solve(c, Y, check_finite=False)), 0, cond, 0
        lu = sla.lu_factor(M, check_finite=False)
        return (lambda Y: sla.lu_solve(lu, Y, check_finite=False)), 0, cond, 0

    def factor(self, A):
        n, seen, entries = A.domain.dim, {}, []
        for b in range(A.n_blocks):
            key = None if A._shift is not None else int(A.shared[b])
            if key is None or key not in seen:
                entry = self._block(0.5 * (A.block(b) + A.block(b).T))
                if key is not None:
                    seen[key] = entry
            else:
                entry = seen[key]
            entries.append(entry)
        bad = sum(e[1] for e in entries)
        where = f" of its block, in {sum(1 for e in entries if e[1])} of {A.n_blocks} blocks"
        if bad and self.refuse:
            raise _refuse(A.name, n, bad, where, raw="BlockDirect")
        if self.rcond is None:
            worst = max(e[2] for e in entries)
            _warn(A.name, worst, bad)
            info = [f"n = {n}, {A.n_blocks} blocks", f"worst block: {_digits(worst)}", "nothing dropped"]
            if bad:
                info.append(f"{bad} directions hold no correct digit{where}")
        else:
            info = [f"n = {n}, {A.n_blocks} blocks", f"pseudo-inverse with rcond {self.rcond:g} per block: "
                    f"{sum(e[3] for e in entries)} directions dropped, in {sum(1 for e in entries if e[3])} blocks"]
        solves, rows = [e[0] for e in entries], A.partition

        def solve(Y):
            out = np.empty_like(Y, dtype=float)
            for p, s in zip(rows, solves):
                out[p] = s(Y[p])
            return out
        return _Factorisation(A, solve, info)


def pseudo_solver(A, rcond):
    """The solver of A.pseudo_inverse(rcond): the pseudo-inverse with that threshold, on A's structure."""
    if not (isinstance(rcond, (int, float)) and rcond >= 0):
        raise ValueError(f"rcond must be a number >= 0, not {rcond!r}")
    if getattr(A, "rows", False) and A.domain == A.codomain and A.domain.is_standard:
        return BlockDirect(rcond)
    return KroneckerEigh(rcond) if _kron_parts(A) is not None else Eigh(rcond)


# What an inverse inherits from the stated error of its operator (plan 054 P3) -------------------------------------
def stated_error(A, sharp=False):
    """(e, status, how): a bound or estimate e on ||A - A_true|| in the norm of A's domain, from what pygeoinf0 states
    about how A was built (plan 054 P3), or None when nothing is stated beyond rounding. Sources: a statement on A
    (operators.error.state, A._stated_error), a Gram G G* (its kernels' realisation errors, compressed energies:
    operators/functionals._gram_error), a covariance applied as its known part (the omitted part R >= 0, ||R|| <= m);
    combined by the rules of operators/error.py for sums, compositions (structural shortcuts included) and inverses
    (unknown for pseudo-inverses and where no rule applies: never None when a part states an error). Status: the weakest
    of the sources (checked for estimates). sharp: the sharpest statement available, an a-posteriori estimate where the operator
    has one (A._sharper_error: realised kernels realised again, operators/refinement._refined; compressed energies
    compressed again, refinement._recompressed; a covariance with a refinement, Covariance.sharper_error), which may cost
    work (measured: about one more construction of the operator's Gram per level of pg.settings.verification, which sets
    how deep, at least 1 here; kept per level on A)."""
    from ..core.claims import weakest
    from ..operators.refinement import depth
    for attr in (("_sharper_error", "_stated_error") if sharp else ("_stated_error",)):
        own = getattr(A, attr, None)
        if callable(own) and attr == "_sharper_error":
            cache, key = A.__dict__.setdefault("_sharper_cache", {}), (depth(), own)
            if key not in cache:
                cache[key] = own()
            own = cache[key]
        elif callable(own):
            own = own()
            setattr(A, attr, own)
        if own is not None:
            return own
    rest = getattr(A, "rest", None)
    if isinstance(rest, dict) and hasattr(A, "realised"):
        inner = stated_error(A.realised, sharp)
        e = float(rest["norm"]) + (inner[0] if inner else 0.0)
        how = f"the covariance applied as its known part, the omitted part R >= 0 with ||R|| <= {rest['norm']:.3g}"
        return (e, weakest(rest["status"], *(inner[1:2] if inner else ())), how + (f"; {inner[2]}" if inner else ""))
    if hasattr(A, "realised") and A.realised is not A:
        return stated_error(A.realised, sharp)
    from ..operators import error as operror                         # the rules (plan 065 A6, plan 068 P3)
    err = lambda X: stated_error(X, sharp)                          # noqa: E731
    terms = getattr(A, "terms", None)
    if terms:
        return operror.sum_error(terms, err)
    ab = operror.factors(A)                                          # a Composition, or a shortcut that kept its factors
    if ab is not None:
        return operror.compose_error(ab[0], ab[1], err)
    from ..operators.inverses import Inverse, PseudoInverse
    if isinstance(A, Inverse):
        if isinstance(A, PseudoInverse) or getattr(A.solver, "rcond", None) is not None:
            return operror.pseudo_inverse_error(A.A, err)
        eA = err(A.A)
        return None if eA is None else operror.inverse_error(A.A.name, eA, _inverse_norm(A.A))
    return None


def _inverse_norm(A):
    """||A^-1|| = 1 / sigma_min of A in orthonormal coordinates (A square on one Euclidean space, within
    settings.dense_max), or None when it is not computed here."""
    from ..settings import settings
    from ..spaces.base import Euclidean
    if not (isinstance(A.domain, Euclidean) and A.domain == A.codomain and A.domain.dim <= settings.dense_max):
        return None
    sig = sla.svdvals(orthonormal_form(A, _square_matrix(A)), check_finite=False)
    return math.inf if not sig[-1] > 0 else 1.0 / float(sig[-1])


def _smallest(A):
    """The smallest |eigenvalue| of A in orthonormal coordinates of its domain (A self-adjoint), and the largest."""
    M = _square_matrix(A)
    Ah = orthonormal_form(A, M)
    lam = np.abs(sla.eigvalsh(0.5 * (Ah + Ah.T), check_finite=False))
    return float(lam.min(initial=math.inf)), float(lam.max(initial=0.0))


def inherited(A, fact):
    """(statement, status, reason, ratio, verified) of what A^-1 inherits from A's stated error (plan 054 P3), or None.

    With A~ the operator in hand and ||A~ - A|| <= e (A the operator meant), x = A^-1 y and x~ = A~^-1 y satisfy
    x~ - x = A~^-1 (A - A~) x, so ||x~ - x|| <= (e / |lambda|_min(A~)) ||x||: the relative error inherited by every
    solve is at most e / |lambda|_min, in addition to the rounding the factorisation states. A bound where e is one; an
    estimate where e is.

    The general rule (plan 054): where that ratio reaches 1 (no correct digit left by the stated error), the sharpest
    statement of A's error is used instead (stated_error(A, sharp=True): an a-posteriori estimate where A has one, at
    the depth pg.settings.verification sets), and Inverse refuses if it still reaches 1. verified is False only at
    pg.settings.verification = 0 ("quick"), where nothing is formed again: the first-order statement is given as it is,
    and where it reaches 1 the statement says it was not verified and Inverse warns instead of refusing.

    Measured (tests/test_error_account_rule.py, _rule2.py, _levels.py; the mission report, "Verification levels"): on
    12 and 16 Gaussian kernels in L2[0, 1] realised at tol 1e-3 .. 1e-7 against the worst direction's error at 40
    digits (3 of the 10 without a digit): level 0 refuses none and warns on 9 (the 3 among them), levels 1, 2, 3 refuse
    7, 4, 3 (false refusals 4, 1, 0), none gives a direction without a digit silently; truncated covariances with a
    refinement (false refusals 6, 4, 4 of 39) and compressed energies (5, 4, 4 of 23):
    the remaining false refusals have a true ||dA|| / |lambda|_min near or above 1 while the worst direction keeps a
    digit, a looseness of the sensitivity 1 / |lambda|_min (the worst direction, the author's decision 1), not of the
    error's estimate. A check per right-hand side would cost one more solve (not built)."""
    from ..settings import settings
    st = stated_error(A)
    if st is None:
        return None
    e, status, how = st
    if math.isnan(e):                                              # no rule bounds the error (plan 065 A6)
        return (f"inherits a stated error of {A.name} of unknown size ({how}): no bound on a solve's error from it, on "
                "top of the rounding above", "unknown", "no rule applies (plan 065 A6)", math.nan, True)
    small = getattr(fact, "smallest", None)
    from ..core.claims import weakest
    if small is None:
        from ..spaces.base import Euclidean
        if not isinstance(A.domain, Euclidean) or A.domain.dim > settings.dense_max:
            e, status, how = st
            return (f"inherits the stated error of {A.name}: ||dA|| <= {e:.3g} ({how}); its smallest eigenvalue is not "
                    f"computed here (no matrix), so the relative error of a solve is at most {e:.3g} / |lambda|_min",
                    weakest("derived", status), "x~ - x = A~^-1 (A - A~) x (plan 054 P3)", math.nan, True)
        small, _ = _smallest(A)
    quick = settings.verification == 0
    if not quick and (not small > 0 or st[0] / small >= 1):        # the sharpest statement decides (the general rule)
        st = stated_error(A, sharp=True) or st
    e, status, how = st
    r = e / small if small > 0 else math.inf
    head = (f"inherits the stated error of {A.name}: ||dA|| <= {e:.3g} ({how}); with |lambda|_min = {small:.3g}, every "
            f"solve's relative error from it is at most {r:.3g}")
    tail = (", so this bound leaves no correct digit in the worst direction (the stated error reaches the smallest "
            f"eigenvalue; at pg.settings.verification = {settings.verification}: a deeper level, 2 or 3, states it more "
            "sharply)" if r >= 1 else "")
    if quick and r >= 1:
        tail = (", so this first-order statement does not show a correct digit in the worst direction; NOT VERIFIED "
                "(pg.settings.verification = 0, quick): the a-posteriori check and the refusal it decides were skipped, "
                "the numbers may hold no correct digit; verification >= 1 checks it")
    return (head + tail + ", on top of the rounding above", weakest("derived", status),
            "x~ - x = A~^-1 (A - A~) x (plan 054 P3); an estimate where the stated error is one", r, not (quick and r >= 1))


def inherited_refusal(name, statement, fact=None):
    """The refusal of an inverse whose stated input error leaves no correct digit (plan 054, the general rule: the
    sensitivity 1 / |lambda|_min times the sharpest stated error of the operator reaches 1)."""
    return Refused(f"{name}: {statement}",
                   needs=["a deeper verification level, pg.settings.verification = 2 or 3 (or pg.settings.using("
                          "verification=3)): nested refinements state the inherited error more sharply, at about one "
                          "more refinement per level; it helps where the statement is loose, not the sensitivity "
                          "(measured, kernels: false refusals 4 / 1 / 0 of 10 at levels 1 / 2 / 3)",
                          "a smaller error in the operator: the kernels realised more accurately (the space's tol), less "
                          "compression (settings.compression), more of the covariance",
                          "a shift you choose, (A + s I).inv: a mathematical regularisation",
                          "A.inverse(pg.Direct(refuse=False)): the numbers, with a warning that the stated error leaves "
                          "no correct digit in the worst direction"])
