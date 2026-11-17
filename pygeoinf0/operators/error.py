"""The operator-error claim ||A~ - A|| <= e and its combination rules (plan 065 A6, plan 068 P3; review A6). A~ is the
operator in hand, A the one meant; e is a bound (or an estimate, by its status) in the operator norm between the
spaces' norms. A statement is (e, status, how); e = NaN with status unknown means a statement of unknown size: an error
is stated somewhere inside and no rule bounds its effect, which is never reported as "nothing beyond rounding".

    state(A, e, how, status)  the caller states ||A~ - A|| <= e for an operator it built (a measured or truncated one);
                              A carries it as a claim with key "operator error"
    error_claim(st)           the claim of a statement (e, status, how)
    sums                      sum_error: ||sum c_j A~_j - sum c_j A_j|| <= sum |c_j| e_j
    compositions              compose_error: ||A~ B~ - A B|| <= e_A ||B~|| + ||A~|| e_B + e_A e_B (A = A~ - dA,
                              B = B~ - dB), with the norms of the operators in hand; UNKNOWN where a norm the rule needs is
                              not on record. Structural shortcuts (a Matrix product multiplied out, a Diagonal product)
                              keep their factors (carry_factors), so the rule applies to them too
    inverses                  inverse_error: with n = ||A~^-1|| and n e < 1, A = A~ (I - A~^-1 dA) is invertible and
                              ||A~^-1 - A^-1|| <= n^2 e / (1 - n e) (Neumann series); for n e >= 1, A may be singular:
                              UNKNOWN. pseudo_inverse_error: always UNKNOWN (a perturbation can move eigenvalues across
                              the threshold, and the perturbation bounds of pseudo-inverses need the rank unchanged)
    norm bounds               norm_bound(A): an operator's own norm_bound() (Identity and orthogonal projections 1, a
                              multiplication sup |m|); a diagonal max |d| (standard or diagonal metric); a matrix between
                              standard spaces its 2-norm; Functionals sqrt(||W||) ||(||k~_i||)_i||; products and sums by
                              sub-multiplicativity and the triangle inequality; otherwise None (not on record)
    norm estimates            scaled_identity(C): c where C = c I exactly; power_estimate(A, v, space): ||A|| of a
                              self-adjoint positive semidefinite A by power iteration from v, counted SAFETY times (an
                              estimate, not a bound; the factor measured in probability/posterior_covariance.py)

solvers.stated_error(A) is the entry point: it reads what A states (its own statement, a Gram's realised kernels, a
covariance applied as its known part) and applies these rules to sums, compositions and inverses. What the kernels' errors
of a raw G do is accounted where G is used (probability/input_error.py), not here.
"""
import math

import numpy as np

from ..core.claims import ASSERTED, DERIVED, UNKNOWN, Claim, weakest

KEY = "operator error"
SAFETY = 2.0        # an iteration's last step estimates its error, it does not bound it (probability/posterior_covariance.py)


def state(A, e, how, status=ASSERTED):
    """State ||A~ - A|| <= e for the operator A in hand (the caller's statement: a measured or truncated operator), as a
    claim on A (key "operator error") that solvers.stated_error and the rules here read."""
    e = float(e)
    if not e >= 0:
        raise ValueError(f"an operator error bound is a number >= 0, not {e!r}")
    A._stated_error = (e, status, how)
    cl = [c for c in list(getattr(A, "_claims", None) or ()) if c.key != KEY]
    A._claims = cl + [error_claim(A._stated_error)]
    return A


def error_claim(st):
    """The claim of a statement (e, status, how): ||A~ - A|| <= e, or "of unknown size" for e = NaN."""
    e, status, how = st
    if math.isnan(e):
        return Claim("operator error ||A~ - A||: of unknown size", UNKNOWN, how, key=KEY)
    return Claim(f"operator error ||A~ - A|| <= {e:.3g}", status, how, key=KEY)


def unknown(how):
    """A statement of unknown size: an error is stated inside and no rule bounds its effect."""
    return (math.nan, UNKNOWN, how)


def carry_factors(out, A, B):
    """Record on the result of A @ B, when a structural shortcut made it something other than a Composition, the factors
    it is the product of, so that the composition rule still applies to its stated error (review A6)."""
    from .linear import Composition
    if not isinstance(out, Composition) and out is not A and out is not B and "_product_of" not in out.__dict__:
        out._product_of = (A, B)
    return out


def factors(A):
    """(A1, A2) when A is the product A1 A2 (a Composition, or a shortcut that kept its factors), else None."""
    from .linear import Composition
    if isinstance(A, Composition):
        return A.A, A.B
    own = getattr(A, "factors", None)                       # an operator defined as a product (MinimumNorm)
    if callable(own):
        return own()
    return getattr(A, "__dict__", {}).get("_product_of")


def norm_bound(A):
    """A bound on ||A|| (the operator in hand) from its structure, or None when none is on record."""
    from ..spaces.base import Euclidean
    from .functionals import Functionals, FunctionalsAdjoint, symbolic_representers
    from .linear import Composition, Diagonal, Identity, Matrix, Sum
    own = getattr(A, "norm_bound", None)
    if callable(own):
        return float(own())
    if isinstance(A, Identity):
        return 1.0
    if isinstance(A, Diagonal) and (A.domain.is_standard or A.domain.weights is not None):
        return float(np.abs(A.d).max(initial=0.0))
    if isinstance(A, Matrix) and isinstance(A.domain, Euclidean) and A.domain.is_standard and A.codomain.is_standard:
        return float(np.linalg.norm(A.A, 2))
    if isinstance(A, (Functionals, FunctionalsAdjoint)):
        G = A.G if isinstance(A, FunctionalsAdjoint) else A
        if symbolic_representers(G.kernels):
            return None
        D = G.codomain
        w = 1.0 if D.is_standard else (float(np.max(D.weights)) if D.weights is not None else None)
        return None if w is None else math.sqrt(w) * float(np.linalg.norm(np.asarray(G._knorm, float)))
    if isinstance(A, Composition):
        a, b = norm_bound(A.A), norm_bound(A.B)
        return None if a is None or b is None else a * b
    if isinstance(A, Sum):
        parts = [norm_bound(B) for _, B in A.terms]
        return None if any(p is None for p in parts) else sum(abs(c) * p for (c, _), p in zip(A.terms, parts))
    return None


def sum_error(terms, error):
    """(e, status, how) of sum_j c_j A_j from the terms' stated errors (error(X) gives them), or None when none states
    one: sum |c_j| e_j (the triangle inequality); unknown when a term's error is of unknown size."""
    parts = [(abs(c), error(B)) for c, B in terms]
    parts = [(c, p) for c, p in parts if p is not None]
    if not parts:
        return None
    how = "; ".join((f"{c:g} x " if c != 1 else "") + p[2] for c, p in parts)
    e = sum(c * p[0] for c, p in parts)
    if math.isnan(e):
        return unknown(f"a sum with a term whose error is of unknown size ({how})")
    return e, weakest(*(p[1] for _, p in parts)), how


def compose_error(A, B, error):
    """(e, status, how) of A B from the factors' stated errors (error(X) gives them), or None when neither states one."""
    eA, eB = error(A), error(B)
    if eA is None and eB is None:
        return None
    how = "; ".join(f"{X.name}: {e[2]}" for X, e in ((A, eA), (B, eB)) if e is not None)
    if any(e is not None and math.isnan(e[0]) for e in (eA, eB)):
        return unknown(f"a composition {A.name} {B.name} with a factor's error of unknown size ({how})")
    need = ([B] if eA is not None else []) + ([A] if eB is not None else [])
    norms = {id(X): norm_bound(X) for X in need}
    missing = [X.name for X in need if norms[id(X)] is None]
    if missing:
        return unknown(f"a composition {A.name} {B.name} with a stated error ({how}) and no norm on record for "
                       f"{', '.join(missing)}: no rule bounds the product's error")
    a, b = (eA[0] if eA else 0.0), (eB[0] if eB else 0.0)
    nA, nB = norms.get(id(A), 0.0), norms.get(id(B), 0.0)
    e = a * nB + nA * b + a * b
    st = weakest(*(x[1] for x in (eA, eB) if x is not None))
    return e, st, f"{A.name} {B.name}: e_A ||B|| + ||A|| e_B + e_A e_B = {e:.3g} ({how})"


def inverse_error(name, eA, n):
    """(e, status, how) of A~^-1 from A~'s stated error eA = (e, status, how) and n = ||A~^-1|| (None: not computed):
    n^2 e / (1 - n e) when n e < 1; unknown otherwise (module docstring)."""
    e, status, how = eA
    if math.isnan(e):
        return unknown(f"the inverse of {name}, whose error is of unknown size ({how})")
    if n is None:
        return unknown(f"the inverse of {name} with a stated error ({how}) and ||{name}~^-1|| not computed here: no rule "
                       "bounds the inverse's error")
    if not n * e < 1:
        return unknown(f"the inverse of {name} with ||d{name}|| <= {e:.3g} ({how}) and ||{name}~^-1|| = {n:.3g}: n e = "
                       f"{n * e:.3g} >= 1, so {name} may be singular; no bound")
    out = n * n * e / (1 - n * e)
    return out, weakest(status, DERIVED), (f"({name})^-1: n^2 e / (1 - n e) = {out:.3g}, n = ||{name}~^-1|| = {n:.3g} "
                                           f"(Neumann series; {how})")


def pseudo_inverse_error(A, error):
    """The pseudo-inverse of A: unknown when A states an error, None when it states none (module docstring)."""
    eA = error(A)
    if eA is None:
        return None
    return unknown(f"inherits the stated error of {A.name} ({eA[2]}); through a pseudo-inverse its effect is not "
                   "bounded here: a perturbation of that size can move eigenvalues across the threshold, and the "
                   "perturbation bounds of pseudo-inverses need the rank unchanged, which is not checked")


# ------------------------------------------------------------------------------------------------ norm estimates
def scaled_identity(C):
    """c where C = c I (an Identity, or a sum of scaled ones), else None."""
    from .linear import Identity
    if isinstance(C, Identity):
        return 1.0
    terms = getattr(C, "terms", None)
    if terms:
        cs = [(c, scaled_identity(B)) for c, B in terms]
        if all(b is not None for _, b in cs):
            return float(sum(c * b for c, b in cs))
    return None


def power_estimate(A, v, space, its=8):
    """An estimate of ||A|| for self-adjoint positive semidefinite A from v (SAFETY times)."""
    from .linear import combine_outputs
    est = 0.0
    for _ in range(its):
        nv = float(space.norm(v))
        if nv == 0:
            break
        v = A(combine_outputs([(1.0 / nv, v)], space))
        est = float(space.norm(v))
    return SAFETY * est
