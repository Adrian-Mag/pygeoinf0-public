"""Product (separated) families on tensor domains, kernel expansions, and lazy linear combinations.

ProductFamily   N product functions k_i(xi) = prod_k u_{i,k}(xi_k) in the natural coordinates of a domain. Each axis stores
     its distinct one-dimensional factors once (realised with the 1-D machinery, dense or sparse) and a map from
     members to factors: kernels on a 46 x 46 x 46 lattice have 46 distinct factors per axis, not 97,336.
KernelExpansion   an expansion sum_i c_i k_i over an ProductFamily family (what G* y is); exact, evaluated on demand.
LazyCombination   a linear combination of realisations of different kinds (tensor, expansion); exact, kept lazy.

Everything reduces to one-dimensional integrals against the axis weights J_k (1, r, r^2, sin theta):

    <k_i, m>    = sum_abc C_abc B1[i,a] B2[i,b] B3[i,c],    B_k[i,a] = int u_{i,k} phi_a J_k      (m tensor)
    <k_i, k_j>  = prod_k int u_{i,k} u_{j,k} J_k                                                (Gram)

For many kernels the Gram matrix is sparse: kernels whose supports do not overlap on every axis are orthogonal.
Candidate pairs come from a KD-tree on the support boxes; the matrix is returned as scipy.sparse.
"""
import inspect

import numpy as np
from numpy.polynomial import legendre as npleg
import scipy.sparse as sp

from . import legendre as L
from . import sparse as S
from .legendre import TINY
from ...quadrature import gauss, jacobian_nodes

DENSE_GRAM_MAX = 6000          # above this many members the Gram matrix is assembled sparsely


def _weight(name):
    """(the Jacobian named name, the extra Gauss points its integrals take): domains.JACOBIANS and
    quadrature.jacobian_nodes (plan 069 P2: the Gauss counts left the domains)."""
    from ...domains import JACOBIANS
    return JACOBIANS[name], jacobian_nodes(name)


# ------------------------------------------------------------------------------------------ one axis

def _pieces(U):
    """(breaks, list of (cols or None, coefficient block)) for a 1-D realisation, dense or sparse."""
    if S.is_sparse(U):
        return np.asarray(U.breaks), list(zip(U.cols, U.coefs))
    return np.asarray(U.breaks), [(None, C) for C in U.coefs]


def axis_projections(U, basis, weight):
    """(U.N, basis.S): int over the axis of u_j(t) phi_a(t) J(t) dt, phi_a the axis basis (piecewise Legendre or Fourier)."""
    wf, extra = _weight(weight)
    Ub, Up = _pieces(U)
    if basis.key[0] == "F":
        return _fourier_projections(U, basis, wf, extra, Ub, Up)
    bb = np.asarray(basis.breaks)
    nb = L.union_breaks(Ub, bb)
    out = np.zeros((U.N, basis.S))
    for q in range(len(nb) - 1):
        x0, x1 = nb[q], nb[q + 1]
        mid = 0.5 * (x0 + x1)
        pu = int(np.clip(np.searchsorted(Ub, mid, side="right") - 1, 0, len(Up) - 1))
        cols, Cu = Up[pu]
        if Cu.size == 0:
            continue
        pb = int(np.clip(np.searchsorted(bb, mid, side="right") - 1, 0, basis.P - 1))
        t, w = gauss((Cu.shape[0] + basis.n) // 2 + extra + 1)
        x = x0 + 0.5 * (x1 - x0) * (t + 1.0)
        Uv = npleg.legvander((2.0 * x - Ub[pu] - Ub[pu + 1]) / (Ub[pu + 1] - Ub[pu]), Cu.shape[0] - 1) @ Cu
        Pv = npleg.legvander((2.0 * x - bb[pb] - bb[pb + 1]) / (bb[pb + 1] - bb[pb]), basis.n - 1)
        blk = Uv.T @ (Pv * (0.5 * (x1 - x0) * w * wf(x))[:, None])
        rows = slice(None) if cols is None else cols
        out[rows, pb * basis.n:(pb + 1) * basis.n] += blk
    return out


def _fourier_projections(U, basis, wf, extra, Ub, Up):
    """axis_projections onto a Fourier basis: per piece of U, Gauss-Legendre with enough nodes for the polynomial times
    cos(K theta) over the piece (an angular span of 2 pi h / L, so a further ~ K pi h / L + 20 degrees)."""
    out = np.zeros((U.N, basis.S))
    span = basis.hi - basis.lo
    for p, (cols, Cu) in enumerate(Up):
        if Cu.size == 0:
            continue
        x0, x1 = Ub[p], Ub[p + 1]
        omega = basis.K * np.pi * (x1 - x0) / span
        t, w = gauss(int((Cu.shape[0] + omega + 20) // 2) + extra + 1)
        x = x0 + 0.5 * (x1 - x0) * (t + 1.0)
        Uv = npleg.legvander(t, Cu.shape[0] - 1) @ Cu
        blk = Uv.T @ (basis.vander(x) * (0.5 * (x1 - x0) * w * wf(x))[:, None])
        out[slice(None) if cols is None else cols] += blk
    return out


def _axis_blocks(U, V, weight, diag=False):
    """Yield (rows, cols, block) with block[i, j] = int over one piece of u_rows[i] v_cols[j] J."""
    wf, extra = _weight(weight)
    Ub, Up = _pieces(U)
    Vb, Vp = _pieces(V)
    nb = L.union_breaks(Ub, Vb)
    for q in range(len(nb) - 1):
        x0, x1 = nb[q], nb[q + 1]
        mid = 0.5 * (x0 + x1)
        pu = int(np.clip(np.searchsorted(Ub, mid, side="right") - 1, 0, len(Up) - 1))
        pv = int(np.clip(np.searchsorted(Vb, mid, side="right") - 1, 0, len(Vp) - 1))
        cu, Cu = Up[pu]
        cv, Cv = Vp[pv]
        if Cu.size == 0 or Cv.size == 0:
            continue
        t, w = gauss((Cu.shape[0] + Cv.shape[0]) // 2 + extra + 1)
        x = x0 + 0.5 * (x1 - x0) * (t + 1.0)
        Uv = npleg.legvander((2.0 * x - Ub[pu] - Ub[pu + 1]) / (Ub[pu + 1] - Ub[pu]), Cu.shape[0] - 1) @ Cu
        Vv = npleg.legvander((2.0 * x - Vb[pv] - Vb[pv + 1]) / (Vb[pv + 1] - Vb[pv]), Cv.shape[0] - 1) @ Cv
        ww = 0.5 * (x1 - x0) * w * wf(x)
        rows = np.arange(U.N) if cu is None else cu
        if diag:
            yield rows, None, (Uv * Uv * ww[:, None]).sum(axis=0)
        else:
            yield rows, (np.arange(V.N) if cv is None else cv), Uv.T @ (ww[:, None] * Vv)


def axis_gram(U, V, weight):
    """(U.N, V.N): int u_i v_j J over the axis."""
    G = np.zeros((U.N, V.N))
    for r, c, blk in _axis_blocks(U, V, weight):
        G[np.ix_(r, c)] += blk
    return G


def axis_sqnorms(U, weight):
    out = np.zeros(U.N)
    for r, _, v in _axis_blocks(U, U, weight, diag=True):
        np.add.at(out, r, v)
    return out


def axis_supports(U, rel=1e-12, samples=4097):
    """Per factor (lo, hi): an interval outside which it is below rel * its maximum. From the pieces where the factor
    is stored (sparse families), refined by values on a uniform grid (one grid spacing of margin each side).
    rel must sit above the rounding noise of the realisation (about 1e-14 relative for high-degree pieces)."""
    Ub = np.asarray(U.breaks)
    t = np.linspace(Ub[0], Ub[-1], samples)
    lo = np.empty(U.N)
    hi = np.empty(U.N)
    step = max(1, (1 << 23) // samples)
    for s in range(0, U.N, step):
        cols = np.arange(s, min(U.N, s + step))
        V = np.abs(np.nan_to_num(axis_values(U, t, cols)))
        live = V > rel * np.maximum(V.max(axis=0), TINY)
        any_ = live.any(axis=0)
        first = np.where(any_, live.argmax(axis=0), 0)
        last = np.where(any_, samples - 1 - live[::-1].argmax(axis=0), 0)
        lo[cols] = t[np.maximum(first - 1, 0)]
        hi[cols] = t[np.minimum(last + 1, samples - 1)]
    return lo, hi


def axis_values(U, t, cols=None):
    """(len(t), k) values of the factors cols (all if None) at points t of the axis; NaN outside."""
    R = U
    if cols is not None:
        R = S.take(U, np.asarray(cols)) if S.is_sparse(U) else U.take(np.asarray(cols))
    elif S.is_sparse(U):
        R = U.take(np.arange(U.N))
    return L.evaluate(R, np.asarray(t, float))


class AxisFamily:
    """The distinct factors of one axis (a 1-D realisation U) and the map member -> factor (idx)."""

    __slots__ = ("U", "idx", "weight", "_supp")

    def __init__(self, U, idx, weight):
        self.U, self.idx, self.weight = U, np.asarray(idx, np.intp), weight
        self._supp = None

    def supports(self):
        if self._supp is None:
            self._supp = axis_supports(self.U)
        return self._supp


# ------------------------------------------------------------------------------------------ families

class ProductFamily:
    """N product functions on the reference box of a domain (see the module docstring)."""

    __slots__ = ("axes", "N", "err", "_proj", "_gram")
    CAPABILITIES = ("kronecker_gram",)

    def __init__(self, axes, err=None):
        self.axes = tuple(axes)
        self.N = len(self.axes[0].idx)
        self.err = np.zeros(self.N) if err is None else np.asarray(err, float)
        self._proj, self._gram = {}, None

    @property
    def d(self):
        return len(self.axes)

    @property
    def size(self):
        return sum(a.U.size for a in self.axes) + self.N * self.d

    def take(self, idx):
        idx = np.atleast_1d(np.asarray(idx, np.intp))
        return ProductFamily([AxisFamily(a.U, a.idx[idx], a.weight) for a in self.axes], self.err[idx])


class KernelExpansion:
    """M expansions sum_i c[i, m] k_i over an ProductFamily family (c has shape (family.N, M))."""

    __slots__ = ("family", "c", "err")

    def __init__(self, family, c, err=None):
        self.family, self.c = family, np.asarray(c, float)
        self.err = np.abs(self.c).T @ family.err if err is None else np.asarray(err, float)

    @property
    def N(self):
        return self.c.shape[1]

    @property
    def size(self):
        return self.c.size


class LazyCombination:
    """N columns sum_j a_j R_j of realisations R_j of any kind (all with N columns)."""

    __slots__ = ("terms", "err")

    def __init__(self, terms):
        self.terms = tuple((float(a), R) for a, R in terms)
        self.err = sum(abs(a) * R.err for a, R in self.terms)

    @property
    def N(self):
        return self.terms[0][1].N

    @property
    def size(self):
        return sum(R.size for _, R in self.terms)


def fit_product_family(backend, factors, params, name="F"):
    """Realise the family k_i = prod_k factors[k](xi_k, **params_i). factors[k] is a function of the k-th natural
    coordinate (None: the constant 1); it receives only the parameters its signature names (or all, with **kw).
    Distinct factors are found from the parameter values and realised once per axis (1-D, dense or sparse)."""
    from ...measures import LebesgueMeasure
    from ...sets import Interval
    from ..backends import IntervalBackend
    dom = backend.domain
    if len(factors) != dom.dim:
        raise ValueError(f"{dom} has natural coordinates {dom.internal_names}: give {dom.dim} factors (None for 1)")
    arrays = {k: np.asarray(v, float).ravel() for k, v in params.items()}
    sizes = {len(v) for v in arrays.values()}
    if len(sizes) > 1:
        raise ValueError("all parameter arrays must have the same length")
    N = sizes.pop() if sizes else 1
    axes, norms, errs = [], [], []
    for k, f in enumerate(factors):
        lo, hi = dom.reference[k]
        ib = IntervalBackend(LebesgueMeasure(Interval(lo, hi)), backend.tol)
        if f is None:
            U, inv = ib.constant([1.0]), np.zeros(N, np.intp)
        else:
            sig = inspect.signature(f)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                names = sorted(arrays)
            else:
                names = [nm for nm in list(sig.parameters)[1:] if nm in arrays]
            if names:
                uniq, inv = np.unique(np.column_stack([arrays[nm] for nm in names]), axis=0, return_inverse=True)
                inv = inv.ravel()
            else:
                uniq, inv = np.zeros((1, 0)), np.zeros(N, np.intp)

            def fun_idx(t, j, f=f, names=names, uniq=uniq):
                kw = {nm: uniq[j, c][None, :] for c, nm in enumerate(names)}
                return np.broadcast_to(np.asarray(f(t[:, None], **kw), float), (len(t), len(j)))

            U = ib.fit_family(fun_idx, len(uniq), name=f"{name} ({dom.internal_names[k]} factors)")
        axes.append(AxisFamily(U, inv, dom.jacobian[k]))
        norms.append(np.sqrt(axis_sqnorms(U, dom.jacobian[k]))[inv])
        errs.append(U.err[inv])
    # |prod u_k - prod v_k| <= sum_k |u_k - v_k| prod_{j != k} |u_j| (to first order), in the weighted norm
    err = np.zeros(N)
    for k in range(dom.dim):
        others = np.prod([norms[j] for j in range(dom.dim) if j != k], axis=0)
        err += errs[k] * others
    return ProductFamily(axes, err * np.sqrt(max(backend.jac_max, 1.0)))


# ------------------------------------------------------------------------------------------ inner products

def projections(F, bases):
    """Per axis (F.N, S_k) projections of the members onto a tensor basis (cached per basis)."""
    out = []
    for a, b in zip(F.axes, bases):
        key = (id(a.U), b.key, a.weight)
        hit = F._proj.get(key)
        if hit is None:
            hit = axis_projections(a.U, b, a.weight)
            if len(F._proj) > 12:
                F._proj.clear()
            F._proj[key] = hit
        out.append(hit[a.idx])
    return out


def gram_sf_tr(F, T):
    """(F.N, T.N): <k_i, t_n> for a tensor realisation T."""
    Bs = projections(F, T.bases)
    C = T.C
    d = F.d
    out = np.empty((F.N, T.N))
    inner = int(np.prod(C.shape[1:]))
    step = max(1, (1 << 22) // max(inner, 1))
    for s in range(0, F.N, step):
        sl = slice(s, s + step)
        X = np.tensordot(Bs[0][sl], C, axes=(1, 0))                 # (m, S2, ..., Sd, N)
        for k in range(1, d):
            X = np.einsum("ib...,ib->i...", X, Bs[k][sl])
        out[sl] = X
    return out


def _boxes(F):
    lo = np.empty((F.N, F.d))
    hi = np.empty((F.N, F.d))
    for k, a in enumerate(F.axes):
        l, h = a.supports()
        lo[:, k], hi[:, k] = l[a.idx], h[a.idx]
    return lo, hi


def _candidates(F1, F2, chunk=512):
    """Yield index arrays (i, j) of pairs whose support boxes overlap on every axis, a chunk of F1 at a time."""
    from scipy.spatial import cKDTree
    lo1, hi1 = _boxes(F1)
    lo2, hi2 = _boxes(F2)
    width = np.maximum((hi1 - lo1).max(axis=0), (hi2 - lo2).max(axis=0))
    width = np.where(width > 0, width, 1.0)
    tree = cKDTree(0.5 * (lo2 + hi2) / width)
    c1 = 0.5 * (lo1 + hi1) / width
    for s in range(0, F1.N, chunk):
        rows = np.arange(s, min(F1.N, s + chunk))
        lists = tree.query_ball_point(c1[rows], 1.0 + 1e-12, p=np.inf, return_sorted=False, workers=1)
        counts = np.fromiter((len(x) for x in lists), int, len(lists))
        if counts.sum() == 0:
            continue
        i = np.repeat(rows, counts)
        j = np.concatenate([np.asarray(x, np.intp) for x in lists if len(x)])
        keep = np.all((lo1[i] <= hi2[j]) & (lo2[j] <= hi1[i]), axis=1)
        yield i[keep], j[keep]


def _axis_pair_values(a1, a2, i, j):
    """For pairs of members (i, j): int u_{idx1[i]} u_{idx2[j]} J along one axis."""
    u1, u2 = a1.idx[i], a2.idx[j]
    n1, n2 = a1.U.N, a2.U.N
    if n1 * n2 <= 4e7:
        return axis_gram(a1.U, a2.U, a1.weight)[u1, u2]
    rows, cols, vals = [], [], []
    for r, c, blk in _axis_blocks(a1.U, a2.U, a1.weight):
        rr, cc = np.meshgrid(r, c, indexing="ij")
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        vals.append(blk.ravel())
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n1, n2))
    return np.asarray(G[u1, u2]).ravel()


class KronGram:
    """The Gram matrix of a lattice product family (its members are all combinations of the distinct factors of each
    axis): K = P^T (G_1 kron ... kron G_d) P with G_k the n_k x n_k Gram matrices of the axis factors and P the order
    of the members on the lattice. Products cost O(N sum n_k); (w K + alpha I)^-1 is exact through the
    eigendecompositions of the G_k. No N x N matrix is formed."""
    CAPABILITIES = ("kronecker_product",)

    def __init__(self, factors, pos):
        self.factors = [np.asarray(G, float) for G in factors]
        self.dims = tuple(G.shape[0] for G in self.factors)
        self.pos = np.asarray(pos, np.intp)
        self.N = len(self.pos)
        self.shape = (self.N, self.N)
        self._eig = None

    @property
    def nnz(self):
        return sum(G.size for G in self.factors)

    def _lattice(self, X):
        X = np.asarray(X, float)
        one = X.ndim == 1
        X2 = X.reshape(self.N, -1)
        Y = np.empty_like(X2)
        Y[self.pos] = X2
        return Y.reshape(self.dims + (X2.shape[1],)), one

    def _members(self, Y, one):
        Y = Y.reshape(self.N, -1)[self.pos]
        return Y[:, 0] if one else Y

    @staticmethod
    def _modes(T, mats):
        for k, A in enumerate(mats):
            T = np.moveaxis(np.tensordot(A, T, axes=(1, k)), 0, k)
        return T

    def __matmul__(self, X):
        if sp.issparse(X):
            X = X.toarray()
        T, one = self._lattice(X)
        return self._members(self._modes(T, self.factors), one)

    def __rmatmul__(self, X):
        return (self @ np.asarray(X).T).T                    # K is symmetric

    def diagonal(self):
        d = np.ones(self.dims)
        for k, G in enumerate(self.factors):
            shape = [1] * len(self.dims)
            shape[k] = self.dims[k]
            d = d * np.diag(G).reshape(shape)
        return d.ravel()[self.pos]

    def toarray(self):
        if self.N > 6000:
            raise MemoryError(f"a {self.N} x {self.N} Kronecker Gram matrix is not formed densely")
        D = self.factors[0]
        for G in self.factors[1:]:
            D = np.kron(D, G)
        return D[np.ix_(self.pos, self.pos)]

    def eig(self):
        if self._eig is None:
            self._eig = [np.linalg.eigh(0.5 * (G + G.T)) for G in self.factors]
        return self._eig

    def apply_fn(self, V, f):
        """f(K) V through the eigendecompositions of the factors (f applied to the eigenvalues of K)."""
        E = self.eig()
        lam = np.ones(self.dims)
        for k, (l, _) in enumerate(E):
            shape = [1] * len(self.dims)
            shape[k] = self.dims[k]
            lam = lam * l.reshape(shape)
        T, one = self._lattice(V)
        T = self._modes(T, [Q.T for _, Q in E]) * f(lam)[..., None]
        return self._members(self._modes(T, [Q for _, Q in E]), one)

    def solve_shifted(self, D, w=1.0, alpha=0.0, rcond=1e-12):
        """X with (w K + alpha I) X = D (alpha > 0), or the pseudo-inverse solution for alpha = 0 (eigenvalues of K below
        rcond times the largest are dropped)."""
        E = self.eig()
        lam = np.ones(self.dims)
        for k, (l, _) in enumerate(E):
            shape = [1] * len(self.dims)
            shape[k] = self.dims[k]
            lam = lam * l.reshape(shape)
        T, one = self._lattice(D)
        T = self._modes(T, [Q.T for _, Q in E])
        den = w * lam + alpha
        if alpha <= 0:
            cut = rcond * np.abs(lam).max() * w
            den = np.where(np.abs(den) > cut, den, np.inf)
        T = T / den[..., None]
        return self._members(self._modes(T, [Q for _, Q in E]), one)


def lattice_positions(F):
    """For a product family whose members are exactly all combinations of the distinct axis factors: the position of
    each member on the lattice (C order); else None."""
    dims = tuple(a.U.N for a in F.axes)
    if int(np.prod(dims)) != F.N:
        return None
    pos = np.ravel_multi_index(tuple(a.idx for a in F.axes), dims)
    return pos if len(np.unique(pos)) == F.N else None


def gram_sf_sf(F1, F2):
    """(F1.N, F2.N) Gram matrix of two product families: dense below DENSE_GRAM_MAX members, else sparse."""
    if F1 is F2 and F1._gram is not None:
        return F1._gram
    if F1 is F2 and F1.N > DENSE_GRAM_MAX:
        pos = lattice_positions(F1)
        if pos is not None:
            F1._gram = KronGram([axis_gram(a.U, a.U, a.weight) for a in F1.axes], pos)
            return F1._gram
    if max(F1.N, F2.N) <= DENSE_GRAM_MAX:
        G = np.ones((F1.N, F2.N))
        for a1, a2 in zip(F1.axes, F2.axes):
            G *= axis_gram(a1.U, a2.U, a1.weight)[np.ix_(a1.idx, a2.idx)]
    else:
        # pairs whose supports (at 1e-12 of each factor's maximum) do not overlap are not formed, and entries below
        # 1e-15 sqrt(K_ii K_jj) are dropped; both are below the accuracy of the realisations (about 1e-13)
        n1 = np.sqrt(sqnorms_sf(F1))
        n2 = n1 if F2 is F1 else np.sqrt(sqnorms_sf(F2))
        I, J, V = [], [], []
        for i, j in _candidates(F1, F2):
            v = np.ones(len(i))
            for a1, a2 in zip(F1.axes, F2.axes):
                v *= _axis_pair_values(a1, a2, i, j)
            big = np.abs(v) > 1e-15 * n1[i] * n2[j]
            I.append(i[big])
            J.append(j[big])
            V.append(v[big])
        G = sp.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(F1.N, F2.N))
    if F1 is F2:
        F1._gram = G
    return G


def sqnorms_sf(F):
    out = np.ones(F.N)
    for a in F.axes:
        out *= axis_sqnorms(a.U, a.weight)[a.idx]
    return out


def integral_sf(F):
    from .tensor import AxisBasis
    out = np.ones(F.N)
    for a in F.axes:
        Ub = np.asarray(a.U.breaks)
        out *= axis_projections(a.U, AxisBasis([Ub[0], Ub[-1]], 1), a.weight)[:, 0][a.idx]
    return out


# ------------------------------------------------------------------------------------------ dispatch

def gram(A, B, tensor_gram):
    """Gram matrix of any two realisations; tensor_gram(TensorSeries, TensorSeries) handles the tensor-tensor case."""
    from .tensor import TensorSeries
    if isinstance(A, LazyCombination):
        return sum(a * _dense(gram(R, B, tensor_gram)) for a, R in A.terms)
    if isinstance(B, LazyCombination):
        return sum(b * _dense(gram(A, R, tensor_gram)) for b, R in B.terms)
    if isinstance(A, KernelExpansion):
        return np.asarray(A.c.T @ gram(A.family, B, tensor_gram))
    if isinstance(B, KernelExpansion):
        return np.asarray(gram(A, B.family, tensor_gram) @ B.c)
    if isinstance(A, ProductFamily) and isinstance(B, ProductFamily):
        return gram_sf_sf(A, B)
    if isinstance(A, ProductFamily) and isinstance(B, TensorSeries):
        return gram_sf_tr(A, B)
    if isinstance(A, TensorSeries) and isinstance(B, ProductFamily):
        return gram_sf_tr(B, A).T
    return tensor_gram(A, B)


def _dense(G):
    return G.toarray() if sp.issparse(G) else G


def sqnorms(A, tensor_sqnorms, tensor_gram):
    if isinstance(A, ProductFamily):
        return sqnorms_sf(A)
    if isinstance(A, KernelExpansion):
        K = gram_sf_sf(A.family, A.family)
        return np.einsum("im,im->m", A.c, np.asarray(K @ A.c))
    if isinstance(A, LazyCombination):
        out = 0.0
        for a, R in A.terms:
            for b, Q in A.terms:
                out = out + a * b * np.diag(_dense(gram(R, Q, tensor_gram)))
        return np.asarray(out)
    return tensor_sqnorms(A)


def integral(A, tensor_integral):
    if isinstance(A, ProductFamily):
        return integral_sf(A)
    if isinstance(A, KernelExpansion):
        return A.c.T @ integral_sf(A.family)
    if isinstance(A, LazyCombination):
        return sum(a * integral(R, tensor_integral) for a, R in A.terms)
    return tensor_integral(A)


def evaluate_sf_members(F, xi, members, coeffs):
    """sum over members m of coeffs[m, :] * k_m at internal points xi (arrays of one shape): (npts, ncols)."""
    shape = np.shape(xi[0])
    flat = [np.asarray(a, float).ravel() for a in xi]
    npts = flat[0].size
    out = np.zeros((npts, coeffs.shape[1]))
    ok = np.all([np.isfinite(a) for a in flat], axis=0)
    if not ok.any() or len(members) == 0:
        out[~ok] = np.nan
        return out.reshape(shape + (coeffs.shape[1],))
    pts = [a[ok] for a in flat]
    lo, hi = _boxes(F)
    plo = np.array([p.min() for p in pts])
    phi_ = np.array([p.max() for p in pts])
    hit = np.all((lo[members] <= phi_) & (hi[members] >= plo), axis=1)
    members, coeffs = members[hit], coeffs[hit]
    acc = np.zeros((len(pts[0]), coeffs.shape[1]))
    step = max(1, int(2e7 // max(len(pts[0]), 1)))
    for s in range(0, len(members), step):
        mm = members[s:s + step]
        P = np.ones((len(pts[0]), len(mm)))
        for k, a in enumerate(F.axes):
            u = a.idx[mm]
            uu, inv = np.unique(u, return_inverse=True)
            P *= axis_values(a.U, pts[k], uu)[:, inv.ravel()]
        acc += np.nan_to_num(P) @ coeffs[s:s + step]
    out[ok] = acc
    out[~ok] = np.nan
    return out.reshape(shape + (coeffs.shape[1],))


def evaluate_core(A, xi):
    """An expansion sum_i c_i prod_k u_{i,k} rewritten over the distinct factors: a core array C[a_1, ..., a_d] (the sum of
    the coefficients of the members using factors a_1, ..., a_d) contracted with the factor values at the points."""
    F = A.family
    dims = tuple(a.U.N for a in F.axes)
    core = np.zeros(dims + (A.N,))
    np.add.at(core, tuple(a.idx for a in F.axes), A.c)
    shape = np.shape(xi[0])
    flat = [np.asarray(a, float).ravel() for a in xi]
    ok = np.all([np.isfinite(a) for a in flat], axis=0)
    out = np.full((flat[0].size, A.N), np.nan)
    pts = np.nonzero(ok)[0]
    step = max(1, int((1 << 24) // max(1, int(np.prod(dims[1:])) * A.N)))
    for s in range(0, len(pts), step):
        sel = pts[s:s + step]
        V = [np.nan_to_num(axis_values(a.U, flat[k][sel])) for k, a in enumerate(F.axes)]
        T = np.tensordot(V[0], core, axes=(1, 0))
        for k in range(1, F.d):
            T = np.einsum("pa...,pa->p...", T, V[k])
        out[sel] = T
    return out.reshape(shape + (A.N,))


def grid_values(A, axes_points, tensor_grid):
    """Values on the tensor grid axes_points[0] x ... x axes_points[d-1] of natural coordinates: (m_1, ..., m_d, N).
    An expansion over a product family is contracted axis by axis through its core over the distinct factors
    (cost sum_k m_1..m_k n_k..n_d per column instead of N per grid point); otherwise the grid is evaluated as points."""
    if isinstance(A, LazyCombination):
        return sum(a * grid_values(R, axes_points, tensor_grid) for a, R in A.terms)
    if isinstance(A, KernelExpansion):
        F = A.family
        dims = tuple(a.U.N for a in F.axes)
        if int(np.prod(dims)) * A.N <= (1 << 26):
            core = np.zeros(dims + (A.N,))
            np.add.at(core, tuple(a.idx for a in F.axes), A.c)
            T = core
            for k, a in enumerate(F.axes):
                V = np.nan_to_num(axis_values(a.U, np.asarray(axes_points[k], float)))      # (m_k, n_k)
                T = np.moveaxis(np.tensordot(V, T, axes=(1, k)), 0, k)
            return T
    if isinstance(A, (ProductFamily, KernelExpansion)):
        G = np.meshgrid(*[np.asarray(t, float) for t in axes_points], indexing="ij")
        return evaluate(A, G, None)
    return tensor_grid(A, axes_points)


def evaluate(A, xi, tensor_eval):
    if isinstance(A, ProductFamily):
        return evaluate_sf_members(A, xi, np.arange(A.N), np.eye(A.N)) if A.N <= 64 else _too_many(A)
    if isinstance(A, KernelExpansion):
        dims = [a.U.N for a in A.family.axes]
        if int(np.prod(dims)) * A.N <= (1 << 24):
            return evaluate_core(A, xi)
        nz = np.nonzero(np.any(A.c != 0, axis=1))[0]
        return evaluate_sf_members(A.family, xi, nz, A.c[nz])
    if isinstance(A, LazyCombination):
        return sum(a * evaluate(R, xi, tensor_eval) for a, R in A.terms)
    return tensor_eval(A, xi)


def _too_many(A):
    raise ValueError(f"evaluating all {A.N} members of a product family at once; take some members first (F[i])")


def lincomb(terms, tensor_lincomb):
    """Exact linear combination: tensors are added as tensors, expansions over one family by their coefficients,
    and mixtures are kept lazily as LazyCombination."""
    from .tensor import TensorSeries
    flat = []
    for a, R in terms:
        if isinstance(R, LazyCombination):
            flat += [(a * b, Q) for b, Q in R.terms]
        else:
            flat.append((a, R))
    trs = [(a, R) for a, R in flat if isinstance(R, TensorSeries)]
    exs = {}
    rest = []
    for a, R in flat:
        if isinstance(R, KernelExpansion):
            key = id(R.family)
            if key in exs:
                fam, c = exs[key]
                exs[key] = (fam, c + a * R.c)
            else:
                exs[key] = (R.family, a * R.c)
        elif isinstance(R, ProductFamily):
            if R.N != 1 and len(flat) > 1:
                raise NotImplementedError("arithmetic on a product family: take its members, or apply operators to it")
            rest.append((a, R))
        elif not isinstance(R, TensorSeries):
            rest.append((a, R))
    out = []
    if trs:
        out.append((1.0, tensor_lincomb(trs)))
    out += [(1.0, KernelExpansion(fam, c)) for fam, c in exs.values()]
    out += rest
    if len(out) == 1 and out[0][0] == 1.0:
        return out[0][1]
    return LazyCombination(out)


def describe(A):
    if isinstance(A, ProductFamily):
        u = [a.U.N for a in A.axes]
        return dict(kind="product family", pieces=0, degree=0, size=A.size,
                    factors=f"{A.N} members; distinct factors per axis {u}")
    if isinstance(A, KernelExpansion):
        return dict(kind="kernel expansion", pieces=0, degree=0, size=A.size,
                    expansion=f"sum of {A.family.N} product kernels with coefficients (an exact, lazy realisation)")
    return dict(kind="linear combination", pieces=0, degree=0, size=A.size,
                terms=", ".join(type(R).__name__ for _, R in A.terms))


def summary(A):
    if isinstance(A, ProductFamily):
        return f"product family: {A.N} members, distinct factors per axis {[a.U.N for a in A.axes]}"
    if isinstance(A, KernelExpansion):
        return f"expansion in {A.family.N} product kernels"
    return f"lazy combination of {len(A.terms)} realisations"
