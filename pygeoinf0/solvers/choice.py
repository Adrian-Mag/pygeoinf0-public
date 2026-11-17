"""Choosing: recommend(A) (a solver for A and why), usage() (what each solver is for) and damping_for_misfit (the
damping alpha at which the damped residual has a stated squared norm: the numerics of the discrepancy principle, whose
choice of the target from the datum's noise law is the meta rule pg.meta.discrepancy; plan 070 P2)."""
import math

import numpy as np

from ..core.errors import Refused
from .direct import (
    EPS,
    LU,
    WELL,
    BlockDirect,
    Cholesky,
    Direct,
    Eigh,
    KroneckerEigh,
    _cholesky,
    _cond_estimate,
    _kron_parts,
)
from .iterative import CG
from ..spaces.identity import identical


class Recommendation:
    def __init__(self, solver, facts, reasons):
        self.solver, self.facts, self.reasons = solver, facts, reasons

    def __repr__(self):
        lines = [f"recommended solver: {self.solver!r}"]
        lines += [f"  - {f}" for f in self.facts]
        lines += [f"  why: {r}" for r in self.reasons]
        return "\n".join(lines)


def usage():
    """For people and agents: each solver's name with what it computes, when it is the right choice, the error it can deliver,
    its cost and any simplification it makes. A.inverse(solver).claims give the numbers for a given operator."""
    return {c.__name__: dict(c.about) for c in (Cholesky, Direct, LU, BlockDirect, KroneckerEigh, CG, Eigh)}


def recommend(A):
    """Choose a solver for A from what is known about A (claims) and, for matrices, from its spectrum."""
    from ..spaces.base import Euclidean, HilbertSpace
    D, C = A.domain, A.codomain
    if getattr(A, "rows", False) and D == C and D.is_standard and A.is_selfadjoint()[0]:
        return Recommendation(BlockDirect(), [f"{A.name}: {D} → {C}, n = {D.dim}, {A.n_blocks} blocks of rows"],
                              ["block diagonal: a direct solve per block, with each block's own conditioning, costs the "
                               "sum of the blocks' cubes; no n x n matrix"])
    parts = _kron_parts(A)
    if parts is not None:
        K, w, alpha = parts
        return Recommendation(KroneckerEigh(), [
            f"{A.name}: {D} → {C}, n = {K.N}",
            f"the kernels are all combinations of {' x '.join(map(str, K.dims))} distinct axis factors, so G G* is the "
            f"Kronecker product of {len(K.dims)} small Gram matrices; data weights uniform"],
            ["exact solve through the eigendecompositions of the factors: O(N (n_1 + ... + n_d)) per solve, "
             "no N x N matrix, no iterations"])
    sa = A.is_selfadjoint()
    if isinstance(D, Euclidean) and isinstance(C, Euclidean):
        n = D.dim
        if C.dim != n:
            raise ValueError(f"{A.name} maps R^{n} to R^{C.dim}, so it has no inverse")
        facts = [f"{A.name}: {D} → {C}, n = {n}"]
        if D == C and sa[0]:
            facts.append(f"self-adjoint ({sa[1]}: {sa[2]})")
            from ..settings import settings
            from .backends import Dense
            need, budget = Dense().cost(n)["bytes"], settings.memory_gb * 2 ** 30
            if n <= settings.dense_max and need > budget:
                facts.append(f"a dense factorisation needs about {need / 2 ** 30:.3g} GB (the matrix and its factor), more "
                             f"than settings.memory_gb = {settings.memory_gb:g}")
                if A.sparse() is not None:
                    return Recommendation(Cholesky(backend="sparse"), facts, [
                        "a sparse form is known (sparse matrices, diagonals and their sums): a sparse Cholesky holds only its "
                        "factor; the pivots check positive definiteness and the condition is estimated from the factor"])
                return Recommendation(CG(rtol=settings.accuracy, atol=settings.accuracy * math.sqrt(n), tighter=True), facts,
                                      ["conjugate gradients need only products with the operator and n numbers per vector"])
            if n <= settings.dense_max:
                M = A.matrix()
                c = _cholesky(A, M)
                cond = math.inf if c is None else _cond_estimate(c)
                facts.append("Cholesky factorisation " + ("failed (not numerically positive definite)" if c is None
                             else f"succeeded; condition number estimate {cond:.2g} (LAPACK pocon)"))
                if c is not None and cond < WELL:
                    return Recommendation(Cholesky(factor=(c, A)), facts, [
                        "positive definite and well conditioned: Cholesky is the cheapest direct method "
                        f"(n³/3 flops) and backward stable; expect a relative error near cond·eps ≈ {cond * EPS:.1g}. "
                        "The factor computed for this check is reused"])
                lost = "all" if not math.isfinite(cond) else f"about {math.log10(cond):.0f}"
                return Recommendation(Direct(), facts, [
                    f"Cholesky alone would lose {lost} of 16 digits or fail, so the spectrum is checked first: nothing "
                    "is dropped, the digits lost are stated, and a direction with no correct digit is refused"])
            atol = settings.accuracy * math.sqrt(n)
            return Recommendation(CG(rtol=settings.accuracy, atol=atol, tighter=True), facts, [
                f"n > settings.dense_max = {settings.dense_max}: a dense factorisation costs n³/3 flops and n² memory; "
                "conjugate gradients need only products with the operator and converge for positive definite operators",
                f"stopping rule (settings.accuracy = {settings.accuracy:g}): residual norm in the data space's metric at "
                f"most {settings.accuracy:g}·sqrt(n), i.e. its rms is {100 * settings.accuracy:g} % of a noise standard "
                "deviation when the metric is the inverse noise covariance, and at most that fraction of the datum's own "
                "norm whatever the metric; the error this leaves in predicted data is no larger"])
        return Recommendation(LU(refuse=True), facts, ["not known to be self-adjoint: LU with partial pivoting"])
    pos = A.positivity()
    if isinstance(D, HilbertSpace) and identical(D, C) and sa[0] and pos[0] == "pd":
        return Recommendation(CG(), [f"{A.name}: {D} → {D}", f"self-adjoint ({sa[1]}: {sa[2]})",
                                     f"positive definite ({pos[1]}: {pos[2]})"],
                              ["an operator on a function space has no matrix; conjugate gradients work in the "
                               "space itself, with its inner product"])
    raise NotImplementedError(f"no solver for {A.name} yet: on function spaces only self-adjoint positive definite "
                              "operators (conjugate gradients) are supported")


class Alpha(float):
    """A damping parameter: a float that records how it was computed (alpha.claims)."""

    def __new__(cls, value, claims):
        out = float.__new__(cls, value)
        out.claims = claims
        return out


def solve_damping(lam, b2, n):
    """alpha with sum (alpha / (lam + alpha))^2 b2 = n, by bisection in log alpha between the rounding floor of the
    eigenvalues, eps len(lam) max lam (below it chi^2 depends on eigenvalues with no correct digit), and 1e20 max lam
    (above it chi^2 equals the zero solution's to rounding). A root outside that range has no correct digit and is
    refused with the choices named (plan 054, wrong row S13: it was the bracket's end, returned as solved exactly)."""

    def chi2(a):
        return float(np.sum((a / (np.maximum(lam, 0.0) + a)) ** 2 * b2))
    if chi2(1e300) < n:
        raise ValueError("even the zero solution fits the data better than the target: the target is too large")
    top = max(float(np.max(lam, initial=0.0)), 1e-300)
    lo, hi = math.log10(max(EPS * len(lam) * top, 1e-300)), math.log10(top) + 20
    choices = ["a target you choose inside the range", "a damping alpha you choose", "the operator in higher precision"]
    if chi2(10 ** lo) > n:
        raise Refused(f"discrepancy principle: chi^2 = {chi2(10 ** lo):.6g} > target {n:.6g} already at the rounding floor "
                      f"alpha = eps n max lam = {10 ** lo:.3g}; the part of the datum the operator cannot fit (or fits only "
                      "through eigenvalues with no correct digit) exceeds the target, so no alpha with a correct digit "
                      "reaches it", needs=choices)
    if chi2(10 ** hi) < n:
        raise Refused(f"discrepancy principle: the target {n:.6g} equals the zero solution's chi^2 = {chi2(1e300):.6g} to "
                      f"rounding; alpha would exceed 1e20 max lam = {10 ** hi:.3g} and has no correct digit", needs=choices)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if chi2(10 ** mid) > n else (mid, hi)
    return 10 ** lo, chi2


def damping_for_misfit(A, d, target, decomposition=None):
    """The damping alpha with ||A_alpha-residual||^2 = target, for the data-space normal operator A = G G*
    (self-adjoint on the data space D) and data d in D, where the damped solution is m_alpha = G* (A + alpha I)^-1 d
    and its residual is alpha (A + alpha I)^-1 d; norms are D's (so with D = Euclidean(n, covariance=C) the target n
    is chi^2 = n). The target is stated: choosing it from the datum's noise law is the discrepancy principle, a meta
    rule (pg.meta.discrepancy, which calls this); a constraint ||G m - d|| <= rho gives target rho^2 (the Occam model).

    In D's orthonormal coordinates chi^2(alpha) = alpha^2 b^T (B + alpha I)^-2 b. For n <= settings.dense_max from the
    eigendecomposition of B (exact); above, from Lanczos on B started at b (products with A only): chi^2 is the Gauss
    quadrature of the tridiagonal T_k, and the iteration stops when the change of chi^2 at the computed alpha over the
    last 10 steps is at most settings.accuracy * sqrt(2 n) (the standard deviation of chi^2 itself is sqrt(2 n)). The
    result is a float with claims naming the route.

    The dense route is the data-space decomposition of A (solvers/dataspace/decomposition.py, plan 009 G3): decomposition=
    gives one already computed (pg.decompose(G), a pg.Problem's), so that the same eigendecomposition serves this
    root and what follows it (posterior moments, SOLA, BGP)."""
    from ..core.claims import CHECKED, Claim, Claims
    from ..core.errors import SpaceMismatch
    from ..spaces.base import Euclidean
    from ..settings import settings
    D = A.domain
    if A.codomain != D or not isinstance(D, Euclidean):
        raise SpaceMismatch(f"damping_for_misfit(A, d, target) needs the data-space normal operator A = G G* (from "
                            f"{D} to itself, a Euclidean space); {A.name} maps {D} to {A.codomain}")
    if getattr(d, "space", None) != D:
        raise SpaceMismatch(f"the data d must be an element of {D} (D(values)), not {getattr(d, 'space', type(d).__name__)}")
    n = float(target)
    if decomposition is not None:
        if decomposition.D != D:
            raise SpaceMismatch(f"the decomposition is of {decomposition.D}, not of {D}")
        return decomposition.damping_for_misfit(d, n)
    if D.dim <= settings.dense_max:
        from .dataspace import DataDecomposition
        return DataDecomposition(None, D, normal=A, blocks=None).damping_for_misfit(d, n)
    b = D.to_orthonormal(d.values)
    return _lanczos_alpha(A, D, b, n, settings.accuracy, Alpha, Claims, Claim, CHECKED)


def _lanczos_alpha(A, D, b, n, accuracy, Alpha, Claims, Claim, CHECKED, check=10, kmax=3000):
    """Lanczos (full reorthogonalisation) on B x = to_orthonormal(A(from_orthonormal(x))), started at b; see
    damping_for_misfit."""
    nb = float(np.linalg.norm(b))
    Q = [b / nb]
    diag, off = [], []
    prev, tol = None, accuracy * math.sqrt(2 * n)
    for k in range(min(D.dim, kmax)):
        w = D.to_orthonormal(A(D(D.from_orthonormal(Q[-1]))).values)
        diag.append(float(Q[-1] @ w))
        Qm = np.array(Q)
        w = w - Qm.T @ (Qm @ w)
        w = w - Qm.T @ (Qm @ w)                                      # twice is enough (Kahan, Parlett)
        beta = float(np.linalg.norm(w))
        done = beta <= 1e-14 * max(abs(diag[0]), 1e-300)
        if (k + 1) % check == 0 or done:
            T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
            th, V = np.linalg.eigh(T)
            alpha, chi2 = solve_damping(th, nb ** 2 * V[0] ** 2, n)
            if done or (prev is not None and abs(chi2(alpha) - prev(alpha)) <= tol):
                how = "the Krylov space is invariant (exact)" if done else \
                    f"chi^2 changed by at most {tol:.3g} = settings.accuracy * sqrt(2 n) over the last {check} steps"
                return Alpha(alpha, Claims([Claim("damping for the misfit target solved by Lanczos", CHECKED,
                                                  f"{k + 1} products with {A.name}; {how}")]))
            prev = chi2
        off.append(beta)
        Q.append(w / beta)
    raise ArithmeticError(f"Lanczos for the damping of a misfit target did not converge in {kmax} steps")
