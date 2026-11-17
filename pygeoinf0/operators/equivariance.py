"""Rotation invariance derived from how an operator was built (plan 009 G2; EXP-030 DESIGN G2, Theorem G2.1; NOTES.md
section 3). Internal: the structured Grams of separable functionals (operators/separable.py) ask here.

Theorem (Schur). A bounded operator A on M = (+)_i N_i (x) L2(S^2) that commutes with every rotation is
A = (+)_l A_l (x) I_(H_l): it maps the degree-l harmonics of each multiplicity space into themselves and acts the same
way on every order. The cases derived here act by a number per degree and summand, A = (+)_k f_k(-Delta_S) on summand
k, and are recognised from their factors, never from their values:

    None, the identity                          f = 1
    AngularFunction, BandProjection             f (operators/band.py)
    sums, positive multiples, compositions      sums and products of the f's (all of them commute)
    inverses                                    1 / f (where the factor's f has no zero)
    block-diagonal operators between direct     one factor per summand
    sums
    an operator with degree_scalars(space)      its own statement (an extension point: the Sturm-Liouville constructions
                                                of plan 009 G5 act by a radial operator per degree instead)

degree_scalars(S, space) gives the per-summand functions, or None when the structure is not derived from the factors
(then nothing is claimed and the dense routes are used). degree_operators(S, space) is the wider statement of plan 009
G5: per summand a function of l giving ("scalar", c) or ("radial", P, lam), the radial operator
sum_n lam_n phi_n (x) phi_n with phi_n the columns of the legendre.PiecewiseLegendre P (a Bessel-potential covariance's known part,
realisations/spectra/laplacian_covariance.py), or None when degree l is absent (zero).
"""
import numpy as np

from ..core.claims import THEOREM, Claim


def _ones(l):
    return np.ones(np.shape(l))


def _n(space):
    return len(space.summands) if hasattr(space, "summands") else 1


def degree_scalars(S, space):
    """[f_k] per summand (one for a space that is not a direct sum) with S = (+)_k f_k(-Delta_S), or None."""
    from .band import AngularFunction
    from .blocks import BlockDiagonal
    from .inverses import Inverse
    from .linear import Composition, Identity, Sum
    n = _n(space)
    S = getattr(S, "covariance", S)
    if S is None:
        return [_ones] * n
    if hasattr(S, "degree_scalars"):
        return S.degree_scalars(space)
    if getattr(S, "domain", None) != space or getattr(S, "codomain", None) != space:
        return None
    if isinstance(S, Identity):
        return [_ones] * n
    if isinstance(S, AngularFunction):
        return [S.parts.get(k, _ones) for k in range(n)]
    if isinstance(S, Sum):
        parts = [(c, degree_scalars(A, space)) for c, A in S.terms]
        if any(p is None for _, p in parts):
            return None
        return [(lambda l, k=k: sum(c * p[k](l) for c, p in parts)) for k in range(n)]
    if isinstance(S, Composition):
        a, b = degree_scalars(S.A, space), degree_scalars(S.B, space)
        if a is None or b is None:
            return None
        return [(lambda l, k=k: a[k](l) * b[k](l)) for k in range(n)]
    if isinstance(S, Inverse):
        a = degree_scalars(S.A, space)
        if a is None:
            return None
        return [(lambda l, k=k: 1.0 / a[k](l)) for k in range(n)]
    if isinstance(S, BlockDiagonal) and not S.rows:
        out = []
        for k, A in enumerate(S.blocks):
            if A is None:
                out.append(lambda l: np.zeros(np.shape(l)))
                continue
            f = degree_scalars(A, space.summands[k])
            if f is None:
                return None
            out.append(f[0])
        return out
    return None


def degree_operators(S, space):
    """[f_k] per summand with f_k(l) = ("scalar", c) | ("radial", P, lam) | None (module docstring), or None."""
    from .blocks import BlockDiagonal
    n = _n(space)
    if hasattr(S, "degree_operators"):
        out = S.degree_operators(space)
        if out is not None:
            return out
    if isinstance(S, BlockDiagonal) and not S.rows and getattr(S, "domain", None) == space:
        out = []
        for k, A in enumerate(S.blocks):
            if A is None:
                out.append(lambda l: None)
                continue
            f = degree_operators(A, space.summands[k])
            if f is None:
                return None
            out.append(f[0])
        return out
    f = degree_scalars(S, space)
    if f is None:
        return None
    return [(lambda l, g=g: ("scalar", float(np.asarray(g(np.array([l])), float).ravel()[0]))) for g in f][:n]


def claim(S):
    """The claim that S acts degree by degree (Schur), for results that used degree_scalars."""
    what = "the model metric" if S is None else getattr(getattr(S, "covariance", S), "name", "S")
    return Claim(f"{what} commutes with rotations and acts by a number per degree and summand", THEOREM,
                 "derived from its factors (functions of the angular Laplacian, their sums, products and inverses); "
                 "Schur's lemma, EXP-030 NOTES.md section 3 (Theorem G2.1)", key="rotation invariant")
