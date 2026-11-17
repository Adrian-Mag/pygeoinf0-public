"""Whitening by a noise covariance and the eigendecompositions of the whitened data Gram matrix, per block
(solvers/dataspace/decomposition.py, plan 009 G3): the metric as the engine takes it, stated explicitly (NoiseMetric,
metric_of: D.whitened(C), the covariance C, or D; plan 070 P3, decision Q1), the factor F with F C F^T = I per block
(_White), the eigendecomposition with its probed residual (_eig), and small helpers."""
import hashlib
import math

import numpy as np
import scipy.linalg as sla

from ...core.errors import NotACovariance, Refused, SpaceMismatch, Unsupported
from ...spaces.metrics import SYMMETRY_RTOL, symmetric_within_rounding

PROBES = 3              # Gaussian probes of the eigen residual per block


def _digest(*items):
    h = hashlib.sha1()
    for a in items:
        if isinstance(a, np.ndarray):
            a = np.ascontiguousarray(a)
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        else:
            h.update(repr(a).encode())
    return h.hexdigest()[:20]


def _cols(A):
    """The matrix of an operator between Euclidean spaces, by applying it to the identity family of its domain (no
    refusal for a tall n x p matrix with n above settings.dense_max)."""
    P = A.domain
    return np.asarray(A(P.family(np.eye(P.dim), name="I")).values, float).T


def _values(D, d):
    if d is None:
        return None
    if hasattr(d, "space"):
        if d.space != D:
            raise SpaceMismatch(f"the datum is in {d.space}, not in the data space {D}")
        return np.asarray(d.values, float)
    v = np.asarray(d, float)
    if v.shape != (D.dim,):
        raise ValueError(f"a datum of R^{D.dim} has {D.dim} values")
    return v


FIX = ("a symmetric positive definite covariance: positive variances; a matrix symmetric to rounding (symmetrise it "
       "yourself if (C + C^T) / 2 is the covariance you mean)",)


def checked_variances(v, what):
    """v as positive finite variances, or NotACovariance naming the first coordinate that is not (plan 071 P4)."""
    v = np.asarray(v, float)
    bad = np.flatnonzero(~(np.isfinite(v) & (v > 0)))
    if bad.size:
        i = int(bad[0])
        raise NotACovariance(f"{what or 'the noise covariance'} is not a covariance: the variance at coordinate {i} is "
                             f"{v[i]:g} ({bad.size} of {v.size} not positive and finite)", needs=FIX)
    return v


def checked_covariance(S, what, rtol=SYMMETRY_RTOL):
    """A covariance matrix symmetric within rounding (spaces.metrics.symmetric_within_rounding at rtol), returned
    symmetrised to the last bit, with positive variances; else NotACovariance (plan 071 P4, Mac review item 6: never
    symmetrised silently). Positive definiteness is checked where it is factored (_white_block)."""
    S = np.asarray(S, float)
    what = what or "the noise covariance"
    if not np.all(np.isfinite(S)):
        raise NotACovariance(f"{what} has entries that are not finite", needs=FIX)
    if not symmetric_within_rounding(S, rtol):
        gap = float(np.abs(S - S.T).max())
        raise NotACovariance(f"{what} is not symmetric: max |C - C^T| = {gap:.3g} with max |C| = {np.abs(S).max():.3g}, "
                             f"beyond rounding (relative tolerance {rtol:.3g})", needs=FIX)
    checked_variances(np.diag(S), what)
    return 0.5 * (S + S.T)


class NoiseMetric:
    """The noise covariance in coordinates, the metric of the whitening as the engine takes it (kind standard: D's own
    inner product, the covariance of D's standard Gaussian; diag: a vector; dense: a matrix; blocks: a BlockDiagonal),
    its mean or centre (offset), a value key and a description. metric_of builds one from an explicit argument. A diag
    or dense covariance is checked (checked_variances, checked_covariance): a covariance that is not one is refused
    with NotACovariance, never symmetrised or clipped (plan 071 P4); rtol the symmetry tolerance of a dense one."""

    def __init__(self, kind, data, D, offset=None, what="", rtol=SYMMETRY_RTOL):
        if kind == "diag":
            data = checked_variances(data, what)
        elif kind == "dense":
            data = checked_covariance(data, what, rtol)
        elif kind == "blocks":
            for k, B in enumerate(data.matrices):
                checked_covariance(B, f"{what or 'the noise covariance'} (block {k})")
        self.kind, self.data, self.offset, self.what = kind, data, offset, what
        if kind == "standard":
            m = D._metric
            self.key = ("standard", "I" if m is None else _digest(D.weights) if D.weights is not None
                        else _digest(D.metric_matrix()))
        elif kind == "blocks":
            B = data
            self.key = ("blocks", _digest(*B.matrices, B.shared, B.order, B._shift if B._shift is not None else 0))
        else:
            self.key = (kind, _digest(data))


CHOICES = ("D.whitened(C)", "the noise covariance C", "D")


def metric_of(metric, D, offset=None, what=None):
    """The engine's metric from an explicit argument (plan 070 P3, review D4, decision Q1), never inferred:

        D.whitened(C)          the whitened coordinates of a noise covariance C on D's vectors (operators/whitening.py)
        C                      the noise covariance: an operator on D (self-adjoint in D's inner product), a number, a
                               vector of variances or a matrix of coordinates
        D                      D's own inner product (the covariance of D.standard_gaussian())
        a NoiseMetric          as given

    offset: the noise mean or centre (D's coordinates). Anything else (nothing, a measure, a set) is refused, naming
    these choices: reading a noise law is the public routes' (pg.decompose, the methods: the law lives in the
    model-data relation), and a set's shape is its owner's (convex/forms.py)."""
    from ...operators.linear import LinearOperator
    from ...operators.whitening import Whitening
    from ...spaces.base import Euclidean
    if isinstance(metric, NoiseMetric):
        return metric
    if isinstance(metric, Whitening) and metric.inverse_of is None:
        if metric.source != D:
            raise SpaceMismatch(f"the whitening is of {metric.source}, not of the data space {D}")
        C = metric.covariance
        C = getattr(C, "d", None) if getattr(C, "d", None) is not None else (
            np.asarray(C.matrix(), float) if isinstance(C, LinearOperator) else C)
        return _array_metric(C, D, offset, what or f"D.whitened(C): the whitened coordinates ({metric.name})")
    if isinstance(metric, Euclidean):
        if metric != D:
            raise SpaceMismatch(f"the metric is that of {metric}, not of the data space {D}")
        return NoiseMetric("standard", None, D, offset, what or "the standard Gaussian of the data space")
    if isinstance(metric, LinearOperator):
        out = covariance_metric(metric, D)
        out.offset = offset
        if what:
            out.what = what
        return out
    if isinstance(metric, (int, float, np.floating, np.integer, np.ndarray, list, tuple)) and not isinstance(metric, bool):
        return _array_metric(metric, D, offset, what)
    raise Refused(f"the data-space engine takes its metric explicitly, not from {type(metric).__name__}: "
                  "D.whitened(C) (C the noise covariance), the covariance C itself (an operator on D, a number, a "
                  "vector of variances or a matrix), or D for D's own inner product; pg.decompose and the methods read "
                  "the noise law of a model-data relation", needs=CHOICES)


def _array_metric(C, D, offset, what):
    C = np.asarray(C, float)
    if C.ndim == 0:
        return NoiseMetric("diag", np.full(D.dim, float(C)), D, offset, what or "a constant noise variance")
    if C.shape == (D.dim,):
        return NoiseMetric("diag", C.copy(), D, offset, what or "a diagonal noise covariance")
    if C.shape == (D.dim, D.dim):
        return NoiseMetric("dense", C, D, offset, what or "a noise covariance matrix")
    raise ValueError(f"a noise covariance of R^{D.dim}: a number, {D.dim} variances or a {D.dim} x {D.dim} matrix")


def covariance_metric(C, D):
    """A covariance operator on D (self-adjoint in D's inner product) in coordinates: C W^-1."""
    from ...operators.blocks import BlockDiagonal
    from ...operators.linear import Diagonal, Identity
    if isinstance(C, Identity):
        return NoiseMetric("standard", None, D, None, "the standard Gaussian of the data space")
    if isinstance(C, Diagonal) and (D.is_standard or D.weights is not None):
        return NoiseMetric("diag", C.d if D.is_standard else C.d / D.weights, D, None, f"the diagonal covariance {C.name}")
    if isinstance(C, BlockDiagonal) and C.rows and D.is_standard:
        return NoiseMetric("blocks", C, D, None, f"the block-diagonal covariance {C.name}")
    S = D.solve_metric(np.asarray(C.matrix(), float).T).T
    rtol = SYMMETRY_RTOL                    # C W^-1 through a dense metric solve: its rounding widens the tolerance
    if not (D.is_standard or D.weights is not None):
        rtol = max(SYMMETRY_RTOL, 10.0 * D.dim * np.finfo(float).eps * float(np.linalg.cond(D.metric_matrix())))
    return NoiseMetric("dense", S, D, None, f"the covariance {C.name}", rtol=rtol)


class _White:
    """Whitening of one block: F with F C_b F^T = I (diag: F = diag(s); chol: F = L^-1 with C_b = L L^T; metric: the
    orthonormal coordinates of D, F = L^T with W = L L^T, for the standard Gaussian of a space with a dense metric)."""

    def __init__(self, kind, data, D=None):
        self.kind, self.data, self.D = kind, data, D

    def fwd(self, V):
        if self.kind == "diag":
            return self.data[:, None] * V if V.ndim == 2 else self.data * V
        if self.kind == "chol":
            return sla.solve_triangular(self.data, V, lower=True, check_finite=False)
        return self.D.to_orthonormal(V)

    def back(self, X):
        """F^T X."""
        if self.kind == "diag":
            return self.data[:, None] * X if X.ndim == 2 else self.data * X
        if self.kind == "chol":
            return sla.solve_triangular(self.data, X, lower=True, trans="T", check_finite=False)
        return self.D.apply_metric(self.D.from_orthonormal(X))

    def both(self, K):
        """F K F^T for a symmetric K."""
        if self.kind == "diag":
            return self.data[:, None] * K * self.data[None, :]
        if self.kind == "chol":
            Z = sla.solve_triangular(self.data, K, lower=True, check_finite=False)
            return sla.solve_triangular(self.data, Z.T, lower=True, check_finite=False)
        return self.D.to_orthonormal(np.asarray(self.D.to_orthonormal(K)).T)

    def logdet(self):
        """log det C_b."""
        if self.kind == "diag":
            return -2.0 * float(np.sum(np.log(self.data)))
        if self.kind == "chol":
            return 2.0 * float(np.sum(np.log(np.diag(self.data))))
        D = self.D
        if D.is_standard:
            return 0.0
        if D.weights is not None:
            return -float(np.sum(np.log(D.weights)))
        return -float(np.linalg.slogdet(D.metric_matrix())[1])


def _white_block(noise, rows, D, full):
    if noise.kind == "standard":
        if full:
            return _White("metric", None, D)
        if D.is_standard:
            return _White("diag", np.ones(len(rows)))
        if D.weights is not None:
            return _White("diag", np.sqrt(D.weights[rows]))
        # the block of W^-1 (W block diagonal here); W was checked symmetric (spaces/metrics.py), so the average
        # removes the rounding of the inverse only
        C = np.linalg.inv(D.metric_matrix()[np.ix_(rows, rows)])
        return _White("chol", np.linalg.cholesky(0.5 * (C + C.T)))
    if noise.kind == "diag":
        return _White("diag", 1.0 / np.sqrt(noise.data[rows]))
    C = noise.data[np.ix_(rows, rows)] if noise.kind == "dense" else _dense_rows(noise.data, rows)
    if not np.any(C - np.diag(np.diag(C))):
        return _White("diag", 1.0 / np.sqrt(np.diag(C)))
    try:
        return _White("chol", np.linalg.cholesky(C))
    except np.linalg.LinAlgError:
        raise NotACovariance(f"{noise.what or 'the noise covariance'} is not a covariance: it is not positive definite "
                             f"(the Cholesky factorisation of its block of {len(rows)} rows failed)", needs=FIX) from None


def _dense_rows(B, rows):
    """The submatrix B[rows, rows] of a BlockDiagonal on R^n whose rows are one of its blocks."""
    for b, part in enumerate(B.partition):
        if len(part) == len(rows) and np.array_equal(np.sort(part), np.sort(rows)):
            pos = {int(i): k for k, i in enumerate(part)}
            idx = np.array([pos[int(i)] for i in rows])
            return B.block(b)[np.ix_(idx, idx)]
    raise Unsupported("the noise covariance's blocks do not match the Gram matrix's")


def _eig(W):
    """(Lam, U, relative residual) of a symmetric W; the residual ||W U - U Lam||_F / ||W||_F by Gaussian probes."""
    W = 0.5 * (W + W.T)
    lam, U = np.linalg.eigh(W)
    Z = np.random.default_rng(0).standard_normal((len(lam), PROBES))
    R = W @ (U @ Z) - U @ (lam[:, None] * Z)
    scale = max(float(np.sqrt(np.sum(lam ** 2))), 1e-300)
    return lam, U, float(np.linalg.norm(R) / math.sqrt(PROBES) / scale)


def _eig_task(make, b):
    return make(b)


def _cc():
    from scipy.sparse.csgraph import connected_components
    return connected_components


def _same_partition(A, B):
    pa, pb = A.partition, B.partition
    return len(pa) == len(pb) and all(np.array_equal(x, y) for x, y in zip(pa, pb))


def _split(sizes, order):
    off = np.r_[0, np.cumsum(sizes)]
    return [np.sort(order[off[b]:off[b + 1]]) for b in range(len(sizes))]


def _representers(A):
    """The representers of the rows of A: M -> R^p (A m = (<a_i, m>)_i): A.kernels, else A*(W^-1 e_i)."""
    if hasattr(A, "kernels"):
        return A.kernels
    P = A.codomain
    return A.adjoint(P.family(P.solve_metric(np.eye(P.dim)).T))
