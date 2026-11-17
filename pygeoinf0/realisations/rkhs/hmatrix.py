"""Hierarchical matrices of atom energies (plan 005, P4, option C): the energies of n measures applied to vectors in
about O(N log N) work and memory (N atoms) instead of O(N^2), for the matrix-free route above settings.dense_max.

    E_atoms ~ (near blocks: exact, dense) + (far blocks: low rank, adaptive cross approximation)

Cluster tree: recursive bisection of the atoms' centres along their longest extent, down to LEAF atoms; every
cluster's box contains its atoms entirely (centres padded by the atoms' radii). Two clusters are far (admissible) when
max(diam) <= ETA * dist of their boxes. There the kernel is smooth (its only singularity is at r = 0), and the block
is approximated by adaptive cross approximation with partial pivoting from exact entries. The target is global,
||E - E~||_F <= settings.compression * ||E||_F: an m x n block stops when the estimate ||u_k|| ||v_k|| of its remainder
is at most compression * F * sqrt(m n) / N, F = sqrt(N) * (rms of the diagonal) <= ||E||_F, so that the squared
allowances of all blocks add up to at most the global budget (sum of m n <= N^2). Blocks whose energies have decayed
(e^{-dist/l}) stop at rank 0 or 1. Every other pair of leaves is near and stored exactly. The matrix is symmetric:
one block per unordered pair of clusters.

The root-sum-square of the far blocks' error estimates, relative to the Frobenius norm of the near field and the
approximations, is reported in the claims. Memory is checked before building (the near field alone may not exceed
settings.memory_gb) and while building (all stored blocks, including far blocks that did not compress and are kept
dense: on 10^4 L3 rays these were 711,000 blocks, so the near field alone underestimated the memory). Periodic domains (tori, circles) are not compressed (images make clusters wrap); there the
exact engine streams.
"""
import math

import numpy as np

from ...core.claims import CHECKED, DERIVED, Claim, Claims
from ...core.errors import Unsupported
from . import pairs
from ...execution import imap_blocks, map_blocks

ETA = 1.0        # admissibility: max(diam) <= ETA * dist
LEAF = 64        # atoms per leaf cluster
KMAX = 256       # rank cap of a far block (above it the block is stored dense)
PARALLEL_BLOCKS = 2000   # products with more blocks than this run in parallel over workers()


class ClusterTree:
    """Binary tree over N points with radii: nodes (start, end, child1, child2) into the permutation perm, and boxes."""

    def __init__(self, M, r, leaf=LEAF):
        N = len(M)
        perm = np.arange(N)
        nodes, lo, hi = [[0, N, -1, -1]], [], []

        def box(a, b):
            idx = perm[a:b]
            lo.append((M[idx] - r[idx, None]).min(axis=0))
            hi.append((M[idx] + r[idx, None]).max(axis=0))
        box(0, N)
        stack = [0]
        while stack:
            k = stack.pop()
            a, b = nodes[k][0], nodes[k][1]
            if b - a <= leaf:
                continue
            idx = perm[a:b]
            c = M[idx]
            ax = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
            perm[a:b] = idx[np.argsort(c[:, ax], kind="stable")]
            m = (a + b) // 2
            for s, e in ((a, m), (m, b)):
                nodes.append([s, e, -1, -1])
                box(s, e)
                stack.append(len(nodes) - 1)
            nodes[k][2], nodes[k][3] = len(nodes) - 2, len(nodes) - 1
        self.perm, self.nodes = perm, np.array(nodes)
        self.lo, self.hi = np.array(lo), np.array(hi)

    def size(self, k):
        return self.nodes[k, 1] - self.nodes[k, 0]

    def leaf(self, k):
        return self.nodes[k, 2] < 0

    def diam(self, k):
        return float(np.linalg.norm(self.hi[k] - self.lo[k]))

    def dist(self, k, m):
        gap = np.maximum(0.0, np.maximum(self.lo[m] - self.hi[k], self.lo[k] - self.hi[m]))
        return float(np.linalg.norm(gap))

    def partition(self, eta=ETA):
        """(near, far): lists of node pairs (t, s), one per unordered pair of clusters."""
        near, far = [], []
        stack = [(0, 0)]
        while stack:
            t, s = stack.pop()
            if t != s and max(self.diam(t), self.diam(s)) <= eta * self.dist(t, s):
                far.append((t, s))
            elif self.leaf(t) and self.leaf(s):
                near.append((t, s))
            elif t == s:
                c1, c2 = self.nodes[t, 2], self.nodes[t, 3]
                stack += [(c1, c1), (c1, c2), (c2, c2)]
            elif self.leaf(t) or (not self.leaf(s) and self.size(s) > self.size(t)):
                stack += [(t, self.nodes[s, 2]), (t, self.nodes[s, 3])]
            else:
                stack += [(self.nodes[t, 2], s), (self.nodes[t, 3], s)]
        return near, far


def _entries(kern, A, I, J, images=(None,), reach=None):
    """(len(I), len(J)) exact energies of atom pairs, without weights (summed over signed reflection images; an image
    farther than reach (the image sum's R, energy.ExactEngine.image_reach) from a pair is skipped, as there)."""
    ia, ja = np.repeat(I, len(J)), np.tile(J, len(I))
    if len(images) == 1 and images[0] is None:
        return pairs.energies(kern, A, ia, A, ja)[0].reshape(len(I), len(J))
    M, r = A.extent()
    out = np.zeros(len(ia))
    for g in images:
        Mg = M[ja] * g.R + g.t
        near = np.linalg.norm(M[ia] - Mg, axis=1) - r[ia] - r[ja] < (kern.reach() if reach is None else reach)
        if np.any(near):
            out[near] += g.sign * pairs.energies(kern, A, ia[near], A, ja[near], g)[0]
    return out.reshape(len(I), len(J))


def _aca(kern, A, I, J, allow, images=(None,), reach=None):
    """Adaptive cross approximation with partial pivoting of the block (I, J), stopped when the remainder estimate is
    at most allow: (U, V, error estimate) with block ~ U V^T, or None when a dense block is cheaper."""
    m, n = len(I), len(J)
    U, V, norm2, used, i, est = [], [], 0.0, np.zeros(m, bool), 0, 0.0
    for _ in range(min(m, n, KMAX)):
        used[i] = True
        row = _entries(kern, A, I[[i]], J, images, reach)[0]
        for u, v in zip(U, V):
            row -= u[i] * v
        j = int(np.argmax(np.abs(row)))
        if abs(row[j]) * math.sqrt(m * n) <= allow:   # the next pivot is negligible: the remainder is (energies are
            est = abs(row[j]) * math.sqrt(m * n)       # positive, so a negligible residual row is not a false zero)
            break
        v = row / row[j]
        u = _entries(kern, A, I, J[[j]], images, reach)[:, 0]
        for uu, vv in zip(U, V):
            u -= vv[j] * uu
        nu, nv = float(u @ u), float(v @ v)
        norm2 += nu * nv + 2 * sum(float(u @ uu) * float(v @ vv) for uu, vv in zip(U, V))
        U.append(u)
        V.append(v)
        est = math.sqrt(nu * nv)
        if est <= allow:
            break
        a = np.abs(u)
        a[used] = -1.0
        i = int(np.argmax(a))
    else:
        return None
    if not U:
        return np.zeros((m, 0)), np.zeros((n, 0)), 0.0
    if len(U) * (m + n) >= m * n:
        return None
    return np.array(U).T, np.array(V).T, est


class HMatrix:
    """The compressed energies of the atoms of A (see the module docstring)."""

    def __init__(self, engine, A):
        from ...settings import settings
        if engine.periods is not None:
            raise Unsupported("H-matrices of periodic energies are not implemented (the exact engine streams there)")
        self.A, kern = A, engine.kernel
        # reflection images (intervals, boxes): |x - g y| >= |x - y| for x, y in the box, so the kernel with its
        # images is as smooth as K on admissible blocks and the same partition applies
        images = engine.shifts(engine.extent(A))
        reach = None if images == [None] else engine.image_reach()
        M, r = A.extent()
        self.tree = T = ClusterTree(M, r)
        near, far = T.partition()
        near_entries = sum(int(T.size(t)) * int(T.size(s)) for t, s in near)
        need = 8 * near_entries / 2 ** 30
        if need > settings.memory_gb:
            raise Unsupported(f"the exact near field of {len(M)} atoms needs {need:.1f} GB (> settings.memory_gb = "
                              f"{settings.memory_gb:g}): too many atoms close together for an H-matrix on this machine")
        N = len(M)
        sample = np.unique(np.linspace(0, N - 1, min(N, 256)).astype(int))
        if images == [None]:
            diag = pairs.energies(kern, A, sample, A, sample)[0]
        else:
            diag = np.array([_entries(kern, A, np.array([i]), np.array([i]), images, reach)[0, 0] for i in sample])
        F = math.sqrt(N * float(np.mean(diag ** 2)))                     # a lower bound for ||E_atoms||_F
        tol = self.compression = settings.compression
        perm, nodes = T.perm, T.nodes
        jobs = [("d", t, s) for t, s in near] + [("f", t, s) for t, s in far]
        chunks = [tuple(range(c, min(c + 64, len(jobs)))) for c in range(0, len(jobs), 64)]

        def work(*idx):
            out = []
            for q in idx:
                kind, t, s = jobs[q]
                I, J = perm[nodes[t, 0]:nodes[t, 1]], perm[nodes[s, 0]:nodes[s, 1]]
                lr = _aca(kern, A, I, J, tol * F * math.sqrt(len(I) * len(J)) / N, images, reach) if kind == "f" \
                    else None
                out.append(("d", _entries(kern, A, I, J, images, reach), 0.0) if lr is None else ("f", lr[:2], lr[2]))
            return out
        self.blocks, err2, norm2, self.stats = [], 0.0, 0.0, dict(near=0, far=0, dense_far=0, entries=0, rank=0)
        budget, done = settings.memory_gb * 2 ** 30 / 8, 0
        results = imap_blocks(lambda *idx: work(*idx), (), chunks)
        for c, res in results:
            done += len(c)
            for q, (kind, data, est) in zip(c, res):
                _, t, s = jobs[q]
                self.blocks.append((nodes[t, 0], nodes[t, 1], nodes[s, 0], nodes[s, 1], kind, data))
                if kind == "d":
                    norm2 += float(np.sum(data ** 2)) * (1 if t == s else 2)
                    self.stats["near" if jobs[q][0] == "d" else "dense_far"] += 1
                    self.stats["entries"] += data.size
                else:
                    err2 += 2 * est ** 2
                    norm2 += 2 * float(np.sum((data[0].T @ data[0]) * (data[1].T @ data[1])))
                    self.stats["far"] += 1
                    self.stats["entries"] += data[0].size + data[1].size
                    self.stats["rank"] = max(self.stats["rank"], data[0].shape[1])
            if self.stats["entries"] > budget:
                results.close()                                  # stops the pool now, not when the traceback goes
                raise Unsupported(f"the compressed energies of {N} atoms exceed settings.memory_gb = "
                                  f"{settings.memory_gb:g} after {done} of {len(jobs)} blocks ({self.stats['dense_far']} "
                                  "far blocks did not compress and are stored dense): too many atoms close together for "
                                  "an H-matrix on this machine")
        self.rel_error = math.sqrt(err2 / max(norm2, 1e-300))
        self.claims = Claims([
            Claim(f"energies compressed: {len(M)} atoms, {self.stats['near']} near blocks exact, {self.stats['far']} far "
                  f"blocks of rank <= {self.stats['rank']}, {8 * self.stats['entries'] / 2 ** 30:.2f} GB", DERIVED,
                  f"cluster tree with leaves of {LEAF} atoms, admissibility max(diam) <= {ETA:g} dist"),
            Claim(f"relative error of the compressed atom energies ~ {self.rel_error:.1e} (Frobenius)", CHECKED,
                  f"adaptive cross approximation to compression = {tol:g} (the setting it was built at): estimates, "
                  "not bounds")])

    def apply_atoms(self, x):
        """E_atoms x for x of shape (N, k), atoms in the original order; blocks in parallel over workers() when there
        are more than PARALLEL_BLOCKS of them."""
        from ...execution import workers
        p, nb = self.tree.perm, len(self.blocks)
        xp = x[p]
        w = workers()
        if w > 1 and nb > PARALLEL_BLOCKS:
            step = -(-nb // (4 * w))
            yp = sum(part for _, part in map_blocks(self._apply_chunk, (xp,), [(i, min(i + step, nb))
                                                                               for i in range(0, nb, step)]))
        else:
            yp = self._apply_chunk(xp, 0, nb)
        y = np.empty_like(yp)
        y[p] = yp
        return y

    def _apply_chunk(self, xp, lo, hi):
        yp = np.zeros_like(xp)
        for a, b, c, d, kind, data in self.blocks[lo:hi]:
            if kind == "d":
                yp[a:b] += data @ xp[c:d]
                if a != c:
                    yp[c:d] += data.T @ xp[a:b]
            else:
                U, V = data
                yp[a:b] += U @ (V.T @ xp[c:d])
                yp[c:d] += V @ (U.T @ xp[a:b])
        return yp

    def dense(self):
        """The (n, n) energies of the measures gathered from the blocks (settings.dense_gram = "compressed")."""
        A, p = self.A, self.tree.perm
        own, w = A.owner[p], A.weight[p]
        E = np.zeros((A.n, A.n))

        def reduce(rows, M):                           # sum the rows of M (one per atom) by their measures
            u, inv = np.unique(own[rows], return_inverse=True)
            out = np.zeros((len(u),) + M.shape[1:])
            np.add.at(out, inv, w[rows].reshape((-1,) + (1,) * (M.ndim - 1)) * M)
            return u, out
        for a, b, c, d, kind, data in self.blocks:
            if kind == "d":
                ut, Bt = reduce(slice(a, b), data)
                us, Bs = reduce(slice(c, d), Bt.T)
                block = Bs.T
            else:
                ut, Ur = reduce(slice(a, b), data[0])
                us, Vr = reduce(slice(c, d), data[1])
                block = Ur @ Vr.T
            E[np.ix_(ut, us)] += block
            if a != c:
                E[np.ix_(us, ut)] += block.T
        return 0.5 * (E + E.T)

    def apply(self, V):
        """E V for the energies E of the measures (weights and owners of the atoms applied on both sides)."""
        A = self.A
        V = np.asarray(V, float)
        two = V.ndim == 2
        V2 = V if two else V[:, None]
        x = A.weight[:, None] * V2[A.owner]
        y = A.weight[:, None] * self.apply_atoms(x)
        out = np.zeros((A.n, V2.shape[1]))
        np.add.at(out, A.owner, y)
        return out if two else out[:, 0]


class HEngine:
    """The exact engine with an H-matrix: for products above settings.dense_max, and for dense energies of a set of
    measures with itself when settings.dense_gram == "compressed" (non-periodic domains; the error estimate is in
    A.hmatrix.claims and in the claims of the Gram operator)."""

    name = "H-matrix (near field exact, far field adaptive cross approximation)"

    def __init__(self, exact):
        self.exact = exact

    def __getattr__(self, item):
        return getattr(self.exact, item)

    def hmatrix(self, A):
        from ...settings import settings
        H = A.hmatrix
        if H is not None and H.compression > settings.compression:     # built looser than now asked (plan 054, G3)
            if A.self_energy is not None and A.self_energy[0] is getattr(H, "_dense", None):
                A.self_energy = None
            A.hmatrix = H = None
        if H is None:
            A.hmatrix = HMatrix(self.exact, A)
        return A.hmatrix

    @staticmethod
    def drop_stale(A):
        """Forget energies of A gathered from its H-matrix when exact ones are asked now, or a tighter compression than
        the H-matrix was built at (plan 054, wrong row G3: they were returned, with the current setting claimed)."""
        from ...settings import settings
        H = A.hmatrix
        if H is not None and A.self_energy is not None and A.self_energy[0] is getattr(H, "_dense", None) \
                and (settings.dense_gram != "compressed" or H.compression > settings.compression):
            A.self_energy = None

    def energy(self, A, B):
        from ...settings import settings
        if A is B:
            self.drop_stale(A)
        if A is B and A.self_energy is None and settings.dense_gram == "compressed" and self.exact.periods is None \
                and A.n > 1:
            H = self.hmatrix(A)
            H._dense = H.dense()
            A.self_energy = (H._dense, np.zeros((A.n, A.n)))
        return self.exact.energy(A, B)

    def apply(self, A, V):
        from ...settings import settings
        if self.exact.periods is not None:
            return self.exact.apply(A, V)
        if A.self_energy is not None or A.n <= settings.dense_max:
            return self.energy(A, A) @ np.asarray(V, float)
        return self.hmatrix(A).apply(V)
