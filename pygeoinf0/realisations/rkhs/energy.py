"""Energies of measures, <k_mu, k_nu> = int int K dmu dnu, for the Sobolev backends (backend.py): the engine that P4 of
plan 005 makes replaceable (fast far fields). This one is exact: every pair of atoms through the table in pairs.py.

    energy(A, B)        the (A.n, B.n) matrix, cached on A when A is B; large sets in blocks of measures (ATOM_BLOCK
                        atoms a side), in parallel over execution.workers()
    apply(A, V)         E V for the energies E of A with itself, V of shape (A.n, k): from the cached E, or E assembled
                        (and cached) when A.n <= settings.dense_max, else streamed block by block without storing E
                        (exact, but every product recomputes every pair: a route for checks, or for sparse near fields)
    self_energies_all   the diagonal ||k_mu_i||^2, from the pairs of atoms within each measure only
"""
import math
import warnings

import numpy as np

from ...core.errors import RealisationWarning
from . import pairs
from ...execution import groups, map_blocks

ATOM_BLOCK = 2000        # atoms per side of a block of the energy assembly (bounds the (atoms x atoms) temporaries)
SURFACE_WARN = 1e-6      # relative error estimate of an energy above which a RealisationWarning is issued


def gather(E_atoms, A, B):
    """Sum atom-level values into measure-level ones: S_A E S_B^T with weights."""
    W = E_atoms * A.weight[:, None] * B.weight[None, :]
    out = np.zeros((A.n, W.shape[1]))
    np.add.at(out, A.owner, W)
    res = np.zeros((A.n, B.n))
    np.add.at(res.T, B.owner, out.T)
    return res


def sign(shift):
    """The sign of an image (green.Image; +1 for translations and the identity)."""
    return getattr(shift, "sign", 1.0)


class ExactEngine:
    """Exact energies of atoms for the kernel K (with lattice images on tori and circles: periods, origin; with signed
    reflection images on intervals and boxes with Neumann/Dirichlet faces: images, a green.Reflections)."""

    name = "exact"

    def __init__(self, kernel, periods=None, origin=None, images=None):
        self.kernel, self.periods, self.origin, self.images = kernel, periods, origin, images

    def shifts(self, extent=0.0):
        """Lattice translations (or signed reflection images) whose images can interact within the kernel's reach
        (atoms lie in one cell; extent is the largest atom diameter)."""
        if self.images is not None:                    # atoms lie in the box: an image box beyond R is beyond R
            return self.images.transforms(self.image_reach())
        if self.periods is None:
            return [None]
        R = self.kernel.reach() + extent
        m = [int(math.ceil(R / p)) + 1 for p in self.periods]
        grids = np.meshgrid(*[np.arange(-k, k + 1) for k in m], indexing="ij")
        S = np.stack([g.ravel() for g in grids], axis=1) * self.periods
        return [s for s in S if np.linalg.norm(s) <= R + float(np.linalg.norm(self.periods))]

    def image_reach(self):
        """R of the reflection images under settings.images (green.Reflections.cut; the claims state its bound)."""
        return self.images.cut(self.kernel)[0]

    def reach(self, shift):
        """The distance beyond which pairs of atoms are skipped for a shift: the image sum's R for reflection images
        (every omitted pair then lies beyond R, as the tail bound assumes), the kernel's reach otherwise."""
        return self.image_reach() if hasattr(shift, "R") else self.kernel.reach()

    @staticmethod
    def extent(*As):
        return max(2 * float(np.max(A.extent()[1], initial=0.0)) for A in As)

    def _atom_energy(self, A, B, shift, E, Err):
        """Add the atom-level energies (and error estimates) for one lattice translation of B (None: none) to E, Err;
        pairs farther apart than the kernel's reach (centre distance minus radii) are zero to rounding and skipped."""
        (Ma, ra), (Mb, rb) = A.extent(), B.extent()
        if hasattr(shift, "R"):
            Mb = Mb * shift.R + shift.t
        elif shift is not None:
            Mb = Mb + shift
        near = np.linalg.norm(Ma[:, None] - Mb[None], axis=-1) - ra[:, None] - rb[None] < self.reach(shift)
        same = A is B and shift is None
        if same:
            near &= np.triu(np.ones(near.shape, bool))                 # symmetric: the upper triangle only
        i, j = np.nonzero(near)
        e, r = pairs.energies(self.kernel, A, i, B, j, shift)
        e = sign(shift) * e
        if same:
            low = i != j
            i, j, e, r = np.r_[i, j[low]], np.r_[j, i[low]], np.r_[e, e[low]], np.r_[r, r[low]]
        np.add.at(E, (i, j), e)
        np.add.at(Err, (i, j), r)

    def measure_energy(self, A, B):
        """(energies, error estimates) of the measures of A and B, from the dense (atoms x atoms) assembly."""
        E, Err = np.zeros((len(A.P), len(B.P))), np.zeros((len(A.P), len(B.P)))
        for s in self.shifts(self.extent(A, B)):
            self._atom_energy(A, B, s, E, Err)
        return gather(E, A, B), np.abs(gather(Err, A, B))          # err: nonzero only for pairs with surfaces

    def _block_energy(self, A, B, gA, gB, I, J):
        a = A.subset(gA[I])
        return self.measure_energy(a, a if (A is B and I == J) else B.subset(gB[J]))

    def _blocks(self, A, B):
        gA = groups(A, ATOM_BLOCK)
        gB = gA if A is B else groups(B, ATOM_BLOCK)
        return gA, gB, [(I, J) for I in range(len(gA)) for J in range(I if A is B else 0, len(gB))]

    def _blocked_energy(self, A, B):
        """The energies in blocks of measures (upper blocks only when A is B), in parallel over workers()."""
        gA, gB, tasks = self._blocks(A, B)
        out, err = np.zeros((A.n, B.n)), np.zeros((A.n, B.n))
        for (I, J), (e, r) in map_blocks(self._block_energy, (A, B, gA, gB), tasks):
            out[np.ix_(gA[I], gB[J])], err[np.ix_(gA[I], gB[J])] = e, r
            if A is B and I != J:
                out[np.ix_(gB[J], gA[I])], err[np.ix_(gB[J], gA[I])] = e.T, r.T
        return out, err

    def energy(self, A, B):
        """<k_mu_i, k_nu_j> for the measures of atom sets A and B (cached on A when A is B)."""
        if A is B and A.self_energy is not None:
            return A.self_energy[0]
        if len(A.P) * len(B.P) > ATOM_BLOCK ** 2 and (A.n > 1 or B.n > 1):
            out, err = self._blocked_energy(A, B)
        else:
            out, err = self.measure_energy(A, B)
        if A is B:
            out = 0.5 * (out + out.T)
            A.self_energy = (out, err)
        if np.any(err):
            r, c = np.nonzero(err)
            dA, dB = (np.sqrt(np.abs(np.diag(out)))[r], np.sqrt(np.abs(np.diag(out)))[c]) if A is B else \
                (np.sqrt(self.self_energies(A, r)[0]), np.sqrt(self.self_energies(B, c)[0]))
            worst = float(np.max(err[r, c] / np.maximum(dA * dB, 1e-300)))
            if worst > SURFACE_WARN:
                warnings.warn(f"energies of interface measures (patches, spheres, circles) carry relative quadrature "
                              f"errors up to {worst:.1e}", RealisationWarning, stacklevel=4)
        return out

    def _block_apply(self, A, gA, V, I, J):
        e, _ = self._block_energy(A, A, gA, gA, I, J)
        return e @ V[gA[J]], (None if I == J else e.T @ V[gA[I]])

    def apply(self, A, V):
        """E V for the energies E of the measures of A with themselves (see the module docstring)."""
        from ...settings import settings
        V = np.asarray(V, float)
        if A.self_energy is not None or A.n <= settings.dense_max:
            return self.energy(A, A) @ V
        gA, _, tasks = self._blocks(A, A)
        out = np.zeros((A.n,) + V.shape[1:])
        for (I, J), (yI, yJ) in map_blocks(self._block_apply, (A, gA, V), tasks):
            out[gA[I]] += yI
            if yJ is not None:
                out[gA[J]] += yJ
        return out

    def self_energies(self, A, idx):
        """(||k_mu_i||^2, error estimates) for the measures idx of the atom set A, each from its own atoms."""
        u, back = np.unique(idx, return_inverse=True)
        E = np.zeros(len(u))
        err = np.zeros(len(u))
        for n, i in enumerate(u):
            S = A.subset([i])
            E[n] = abs(float(self.energy(S, S)[0, 0]))
            err[n] = S.self_energy[1][0, 0]
        return E[back], err[back]

    def self_energies_all(self, A):
        """||k_mu_i||^2 for every measure of A at once: the energies of the pairs of atoms within each measure, in one
        vectorised call (no energy between different measures is computed)."""
        order = np.argsort(A.owner, kind="stable")
        counts = np.bincount(A.owner, minlength=A.n)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        I, J = [], []
        for m in range(A.n):
            a, b = np.triu_indices(counts[m])
            I.append(order[starts[m] + a])
            J.append(order[starts[m] + b])
        I, J = np.concatenate(I), np.concatenate(J)
        e = np.zeros(len(I))
        for s in self.shifts(self.extent(A)):
            e += sign(s) * pairs.energies(self.kernel, A, I, A, J, s)[0]
        e *= A.weight[I] * A.weight[J] * np.where(I == J, 1.0, 2.0)
        return np.abs(np.bincount(A.owner[I], weights=e, minlength=A.n))
