"""Block operators: block-diagonal operators between direct sums (spaces/direct_sum.py) and on R^n with blocks of rows,
and the column operator x -> (G_k x)_k (plan 068 P5).

ColumnOperator([G_1, ..., G_K])          x -> (G_1 x, ..., G_K x) from X (the G_k's common domain) into the direct sum of
ColumnOperator({"P": G_1, "S": G_2})    their codomains: a Euclidean with block labels when every codomain is one
                                        (spaces/direct_sum.euclidean_sum: the blocks' metrics, weighted by w_k), else
                                        a pg.DirectSum. Its adjoint is y -> sum_k w_k G_k* y_k, since
                                        <G x, y> = sum_k w_k <G_k x, y_k>_k. G.block(d, k) is the k-th block of d (an
                                        element of G_k's codomain); G.ranges the coordinates of each block (Euclidean
                                        codomains). Several observations of one model stack into one (pg.stack).

BlockDiagonal(X, [A_1, ..., A_K], Y)   x -> (A_1 x_1, ..., A_K x_K) from X = (+) X_k to Y = (+) Y_k, A_k: X_k -> Y_k
                                        (None: the zero map). With weights w_k on X and v_k on Y its adjoint is
                                        y -> ((v_1/w_1) A_1* y_1, ...), because <A x, y>_Y = sum_k v_k <x_k, A_k* y_k>_k.
It is the covariance operator of a product of Gaussian measures (GaussianMeasure.product) and the natural form of any
operator that acts layer by layer.

BlockDiagonal(D, [K_1, ..., K_m], shared=[s_1, ..., s_B], shift=..., order=...)
                                        on D = R^n: block b acts on the rows order[o_b : o_b + n_b] of D by the matrix
                                        K_(s_b) + diag(shift_b). Blocks that repeat are stored once (a Gram matrix
                                        G C G* whose blocks repeat across orders of a rotation-invariant problem, plan 009
                                        G2), the shifts give each block its own diagonal (a noise covariance that does
                                        not repeat), and the row order lets the blocks interleave in D's coordinates.
                                        Nothing n x n is stored; products go block by block, one matrix product per
                                        distinct block. BlockDiagonal.detect(A) finds the blocks of a dense matrix from
                                        its values (a checked claim; structure derived from how an operator was built is
                                        stated by its constructor).
"""
import numpy as np

from ..core.claims import CHECKED, DERIVED, Claim, by_rule
from ..spaces.elements import Family, Vector, VectorFamily
from ..core.errors import SpaceMismatch
from ..spaces.base import Euclidean
from ..spaces.direct_sum import DirectSum, euclidean_sum
from ..spaces.identity import require
from .linear import Diagonal, LinearOperator


class BlockDiagonal(LinearOperator):
    """x -> (A_k x_k)_k between direct sums with the same number of summands, or blocks of rows of R^n (see the
    module docstring)."""

    why_linear = "a block-diagonal stack of its blocks"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return list(self.blocks)

    def __init__(self, X, blocks, Y=None, name="A", shared=None, shift=None, order=None, claims=()):
        if isinstance(X, Euclidean):
            self._rows_init(X, blocks, Y, name, shared, shift, order, claims)
            return
        if shared is not None or shift is not None or order is not None:
            raise TypeError("shared, shift and order describe blocks of rows of R^n; between direct sums the blocks are "
                            "the summands")
        Y = X if Y is None else Y
        if len(blocks) != len(X.summands) or len(Y.summands) != len(X.summands):
            raise ValueError(f"{len(blocks)} blocks for direct sums of {len(X.summands)} and {len(Y.summands)} summands")
        for k, A in enumerate(blocks):
            if A is not None:
                msg = f"block {X.names[k]!r} maps {A.domain} to {A.codomain}, not {X.summands[k]} to {Y.summands[k]}"
                require(A.domain, X.summands[k], msg, f"block {X.names[k]!r}")
                require(A.codomain, Y.summands[k], msg, f"block {X.names[k]!r}")
        super().__init__(X, Y, name=name)
        self.rows = False
        self.blocks = list(blocks)
        live = [A for A in blocks if A is not None]
        if X == Y and all(A.domain == A.codomain and (A._sa or (None,))[0] for A in live):
            self._sa = (True, by_rule(*(A._sa[1] for A in live)),
                        "every block is self-adjoint (equal weights on both sides)")
            if all(A._pos for A in live):
                self._pos = ("psd", by_rule(*(A._pos[1] for A in live)), "every block is positive semidefinite")
        self._claims.append(Claim("linear", DERIVED, "every block is linear"))
        self._claims += list(claims)

    def _apply(self, x):
        if self.rows:
            return self._rows_apply(x)
        X, Y = self.domain, self.codomain
        parts = {k: A(X.part(x, X.names[k])) for k, A in enumerate(self.blocks) if A is not None}
        return Y._from_parts(parts, f"{self.name}({x.name})", family=isinstance(x, Family))

    def _make_adjoint(self):
        if self.rows:
            return self._rows_adjoint()
        X, Y = self.domain, self.codomain
        blocks = [None if A is None else (A.adjoint if Y.weights[k] == X.weights[k]
                                          else (Y.weights[k] / X.weights[k]) * A.adjoint)
                  for k, A in enumerate(self.blocks)]
        return BlockDiagonal(Y, blocks, X, name=f"{self.name}*")

    # ------------------------------------------------------------------------------------------- blocks of rows
    def _rows_init(self, X, blocks, Y, name, shared, shift, order, claims):
        Y = X if Y is None else Y
        if not isinstance(Y, Euclidean) or Y.dim != X.dim:
            raise SpaceMismatch(f"blocks of rows map R^{X.dim} to itself (or to a space of the same dimension), not to {Y}")
        mats = []
        for K in blocks:
            K = np.array(getattr(K, "A", K), float)
            if K.ndim != 2 or K.shape[0] != K.shape[1]:
                raise ValueError("the blocks of rows of R^n are square matrices")
            K.flags.writeable = False
            mats.append(K)
        sh = np.arange(len(mats)) if shared is None else np.asarray(shared, int).ravel()
        if sh.size == 0 or sh.min() < 0 or sh.max() >= len(mats):
            raise ValueError(f"shared indexes the {len(mats)} distinct blocks")
        sizes = np.array([mats[i].shape[0] for i in sh])
        n = X.dim
        if sizes.sum() != n:
            raise ValueError(f"the blocks cover {sizes.sum()} rows of R^{n}")
        order = np.arange(n) if order is None else np.asarray(order, int).ravel()
        if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
            raise ValueError(f"order must be a permutation of the {n} rows")
        off = np.r_[0, np.cumsum(sizes)]
        if shift is None:
            sp = None
        elif isinstance(shift, (list, tuple)):
            if len(shift) != len(sh):
                raise ValueError(f"{len(shift)} shifts for {len(sh)} blocks")
            sp = np.concatenate([np.broadcast_to(np.asarray(s, float), (int(m),)) for s, m in zip(shift, sizes)])
        else:
            s = np.asarray(shift, float)
            sp = np.full(n, float(s)) if s.ndim == 0 else s.ravel()[order]       # one entry per row of X
            if sp.shape != (n,):
                raise ValueError(f"shift: a number, one value per row of R^{n} or one array per block")
        LinearOperator.__init__(self, X, Y, name=name)
        self.rows, self.blocks = True, None
        self.matrices, self.shared, self.order = tuple(mats), sh, order
        self.sizes, self.offsets = sizes, off
        self._shift = None if sp is None else sp
        if sp is not None:
            self._shift.flags.writeable = False
        self._groups = []                          # per distinct block: the blocks using it and their rows (block order)
        for i, K in enumerate(mats):
            bs = np.flatnonzero(sh == i)
            if len(bs):
                m = K.shape[0]
                self._groups.append((i, len(bs), m, (off[bs][:, None] + np.arange(m)[None, :]).ravel()))
        self._claims.append(Claim("linear", DERIVED, "every block is a matrix"))
        self._claims += list(claims)
        if X == Y and X.is_standard and all(np.array_equal(K, K.T) for K in mats):
            self._sa = (True, CHECKED, "every block is symmetric (exact equality) and the metric is the dot product")

    @property
    def partition(self):
        """The rows of R^n of each block, in block order (indices into the domain's coordinates)."""
        if not self.rows:
            raise AttributeError("the blocks of a map between direct sums are its summands")
        return [self.order[self.offsets[b]:self.offsets[b + 1]] for b in range(len(self.shared))]

    @property
    def n_blocks(self):
        return len(self.shared) if self.rows else len(self.blocks)

    def shift(self, b):
        """The diagonal added to block b (zeros when there is none)."""
        m = int(self.sizes[b])
        return np.zeros(m) if self._shift is None else self._shift[self.offsets[b]:self.offsets[b + 1]]

    def block(self, b):
        """The dense matrix of block b: K_(shared[b]) + diag(shift_b)."""
        K = self.matrices[self.shared[b]]
        return K + np.diag(self.shift(b)) if self._shift is not None else K

    def _rows_values(self, V):
        """A V for V of shape (n, N), block by block (one matrix product per distinct block)."""
        P = V[self.order]
        out = np.empty_like(P)
        for i, nb, m, idx in self._groups:
            Z = P[idx].reshape(nb, m, -1).transpose(1, 0, 2).reshape(m, -1)            # (m, blocks x N)
            out[idx] = (self.matrices[i] @ Z).reshape(m, nb, -1).transpose(1, 0, 2).reshape(nb * m, -1)
        if self._shift is not None:
            out += self._shift[:, None] * P
        res = np.empty_like(out)
        res[self.order] = out
        return res

    def _rows_apply(self, x):
        if isinstance(x, Vector):
            return Vector(self.codomain, self._rows_values(x.values[:, None])[:, 0], name=f"{self.name}{x.name}")
        if isinstance(x, VectorFamily):
            return VectorFamily(self.codomain, self._rows_values(x.values.T).T, name=f"{self.name}{x.name}")
        raise TypeError(type(x).__name__)

    def _rows_adjoint(self):
        X, Y = self.domain, self.codomain
        if X.is_standard and Y.is_standard:
            sh = None if self._shift is None else [self.shift(b) for b in range(len(self.shared))]
            return BlockDiagonal(Y, [K.T for K in self.matrices], X, name=f"{self.name}*", shared=self.shared,
                                 shift=sh, order=self.order)

        def adj(y):                                # <A x, y>_Y = x . A^T W_Y y = <x, W_X^-1 A^T W_Y y>_X
            V = Y.apply_metric(np.atleast_2d(y.values).T)
            T = BlockDiagonal(Euclidean(X.dim), [K.T for K in self.matrices], shared=self.shared, order=self.order,
                              shift=None if self._shift is None else [self.shift(b) for b in range(len(self.shared))])
            W = X.solve_metric(T._rows_values(V))
            return Vector(X, W[:, 0]) if isinstance(y, Vector) else VectorFamily(X, W.T)
        return LinearOperator(Y, X, apply=adj, adjoint=self._rows_apply, name=f"{self.name}*")

    def matrix(self):
        if not self.rows:
            return super().matrix()
        from .linear import _dense_allowed
        _dense_allowed(self)
        return self._rows_values(np.eye(self.domain.dim))

    def __add__(self, other):
        if self.rows and isinstance(other, Diagonal) and other.domain == self.domain and self.domain == self.codomain \
                and self.domain.is_standard:
            sp = other.d[self.order] + (0.0 if self._shift is None else self._shift)
            out = BlockDiagonal(self.domain, self.matrices, name=f"{self.name} + {other.name}", shared=self.shared,
                                shift=[sp[self.offsets[b]:self.offsets[b + 1]] for b in range(len(self.shared))],
                                order=self.order, claims=[c for c in self._claims if c.statement != "linear"])
            if self._sa and self._sa[0] and other._sa and other._sa[0]:
                out._sa = (True, by_rule(self._sa[1], other._sa[1]), "self-adjoint blocks plus a diagonal")
            if self._pos and other._pos:
                out._pos = ("pd" if other._pos[0] == "pd" else self._pos[0], by_rule(self._pos[1], other._pos[1]),
                            "a positive semidefinite operator plus a positive diagonal")
            return out
        return super().__add__(other)

    @classmethod
    def detect(cls, A, tol=1e-14, name=None):
        """The blocks of a symmetric matrix (or an operator on R^n with one) found from its values: rows i, j are in one
        block when they are joined by entries with |A_ij| > tol sqrt(|A_ii A_jj|). Equal blocks are stored once. The
        structure is a CHECKED claim that records the largest entry treated as zero."""
        from scipy.sparse.csgraph import connected_components
        X = A.domain if isinstance(A, LinearOperator) else Euclidean(np.shape(A)[0])
        M = A.matrix() if isinstance(A, LinearOperator) else np.asarray(A, float)
        sizes, order, ratio = coupled_components(M, tol, connected_components)
        mats, shared, keys, off = [], [], {}, 0
        for m in sizes:
            r = order[off:off + m]
            off += m
            Kb = np.ascontiguousarray(M[np.ix_(r, r)])
            key = (Kb.shape, Kb.tobytes())
            if key not in keys:
                keys[key] = len(mats)
                mats.append(Kb)
            shared.append(keys[key])
        cl = [Claim(f"block diagonal: {len(sizes)} blocks of rows (largest {max(sizes)}), {len(mats)} distinct", CHECKED,
                    f"from the values: entries below {tol:g} sqrt(|A_ii A_jj|) treated as zero (largest such ratio "
                    f"{ratio:.2g})", key="block diagonal")]
        return cls(X, mats, name=name or getattr(A, "name", "A"), shared=shared, order=order, claims=cl)

    def __repr__(self):
        if not self.rows:
            return super().__repr__()
        return (f"{self.name}: {self.domain} → {self.codomain}  [{len(self.shared)} blocks of rows, "
                f"{len(self.matrices)} distinct" + (", with shifts]" if self._shift is not None else "]"))


class ColumnOperator(LinearOperator):
    """x -> (G_k x)_k into the direct sum of the codomains (see the module docstring)."""

    why_linear = "a column of its blocks"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return list(self.blocks)

    def __init__(self, blocks, weights=None, name=None):
        names, ops = (list(blocks), list(blocks.values())) if isinstance(blocks, dict) else \
            (list(range(len(blocks))), list(blocks))
        if not ops or not all(isinstance(G, LinearOperator) for G in ops):
            raise TypeError("a column operator needs one or more linear operators")
        X = ops[0].domain
        for n, G in zip(names, ops):
            require(G.domain, X, f"block {n!r} acts on {G.domain}, not on {X}: a column needs one domain", f"block {n!r}")
        codomains = dict(zip(names, [G.codomain for G in ops]))
        if all(isinstance(D, Euclidean) for D in codomains.values()):
            Y = euclidean_sum(codomains, weights)
            ends = np.cumsum([0] + [G.codomain.dim for G in ops])
            self.ranges = [(int(a), int(b)) for a, b in zip(ends[:-1], ends[1:])]
        else:
            Y = DirectSum(codomains, weights)
            self.ranges = None
        super().__init__(X, Y, name=name or "[" + "; ".join(G.name for G in ops) + "]")
        self.blocks, self.names = ops, tuple(names)
        self.weights = np.ones(len(ops)) if weights is None else np.asarray(weights, float).ravel()

    def index(self, key):
        """The position of block key (its name, or its position when the names are positions)."""
        if key in self.names:
            return self.names.index(key)
        if isinstance(key, (int, np.integer)) and 0 <= key < len(self.blocks):
            return int(key)
        raise KeyError(f"{key!r} is not a block of {self.name} (blocks: {', '.join(map(repr, self.names))})")

    def block(self, d, key):
        """The block key of an element (or family) d of the codomain, as an element of that block's codomain."""
        k = self.index(key)
        if self.ranges is None:
            return self.codomain.part(d, self.names[k])
        (a, b), D = self.ranges[k], self.blocks[k].codomain
        if isinstance(d, VectorFamily):
            return VectorFamily(D, self.codomain._values(d)[:, a:b], name=f"{d.name}[{self.names[k]!r}]")
        return Vector(D, self.codomain._values(d)[a:b], name=f"{d.name}[{self.names[k]!r}]")

    def _apply(self, x):
        parts = [G(x) for G in self.blocks]
        many = isinstance(x, (Family, VectorFamily))
        if self.ranges is None:
            return self.codomain._from_parts(dict(enumerate(parts)), f"{self.name}({x.name})", family=many)
        if many:
            return VectorFamily(self.codomain, np.concatenate([p.values for p in parts], axis=1), name=f"{self.name}(X)")
        return Vector(self.codomain, np.concatenate([p.values for p in parts]), name=f"{self.name}({x.name})")

    def _make_adjoint(self):
        return ColumnAdjoint(self)

    def __repr__(self):
        return f"{self.name}: {self.domain} → {self.codomain}  [a column of {len(self.blocks)} blocks]"


class ColumnAdjoint(LinearOperator):
    """The adjoint y -> sum_k w_k G_k* y_k of a column operator (a row of the blocks' adjoints)."""

    why_linear = "a weighted sum of the blocks' adjoints"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [G.adjoint for G in self.column.blocks]

    def __init__(self, column):
        super().__init__(column.codomain, column.domain, name=f"{column.name}*")
        self.column = column

    def _apply(self, y):
        G, out = self.column, None
        for k, (Gk, w) in enumerate(zip(G.blocks, G.weights)):
            term = Gk.adjoint(G.block(y, k))
            term = term if w == 1.0 else term * float(w)
            out = term if out is None else out + term
        return out


def coupled_components(M, tol, connected_components):
    """(sizes, order, ratio): the connected components of the pattern |M_ij| > tol sqrt(|M_ii M_jj|), the rows in
    component order (ascending within each), and the largest |M_ij| / sqrt(|M_ii M_jj|) treated as zero."""
    import scipy.sparse as sps
    n = M.shape[0]
    d = np.sqrt(np.abs(np.diag(M)))
    d[d == 0] = 1.0
    rows, cols, ratio = [], [], 0.0
    for a in range(0, n, 1024):                    # in strips of rows: no n x n temporary
        R = np.abs(M[a:a + 1024]) / (d[a:a + 1024, None] * d[None, :])
        big = R > tol
        small = R[~big]
        ratio = max(ratio, float(small.max()) if small.size else 0.0)
        i, j = np.nonzero(big)
        rows.append(i + a)
        cols.append(j)
    i, j = np.concatenate(rows), np.concatenate(cols)
    pattern = sps.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n)) + sps.eye(n)
    k, lab = connected_components(pattern, directed=False)
    _, first = np.unique(lab, return_index=True)               # components in the order of their first row
    rank = np.empty(k, int)
    rank[np.argsort(first)] = np.arange(k)
    order = np.argsort(rank[lab], kind="stable")
    return np.bincount(rank[lab], minlength=k), order, ratio
