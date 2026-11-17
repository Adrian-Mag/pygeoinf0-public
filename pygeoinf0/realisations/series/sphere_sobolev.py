"""Sobolev spaces on spheres, H^s_l(S_R): spherical harmonics with the diagonal metric R^2 w_n,
w_n = (1 + l^2 n(n+1)/R^2)^s (the spectral definition via the Laplace-Beltrami operator).

Riesz representers of points and great-circle arcs are realised by their moments mu_nm = int Y_nm dmu, divided by
R^2 w_n and truncated at an adaptive degree; a path measure is not band-limited, so the truncation error is algebraic
(O(N^(1 - 2s)) in the energy) and is estimated from the last octave (it is the realisation's error). Every error of
this backend is in the H^s norm.
"""
import math

import numpy as np

from ...core.errors import SupportOutsideDomain, Unsupported
from ...quadrature import gauss
from .spherical import SphericalHarmonicSeries, SphereBackend, degree_energy
from ...spaces.harmonic import assoc, order_rows, rows


class SphereSobolevBackend(SphereBackend):
    """H^s_l(S_R): spherical harmonics with the diagonal metric R^2 w_n, w_n = (1 + l^2 n(n+1)/R^2)^s.

    Riesz representer of u -> int u dmu (points, arcs): coefficients mu_nm / (R^2 w_n), mu_nm = int Y_nm dmu, truncated
    at the degree where the estimated tail of the energy sum_{n > N} |mu_n|^2 / (R^2 w_n) falls below tol^2 (power law
    fitted to the last octave), at most MAX_RIESZ; the estimate is the realisation's error. A path measure is not
    band-limited, so this truncation error is algebraic, O(N^(1 - 2s)) in the energy: exact kernels on the sphere
    (zonal Green's functions) are future work."""

    MAX_RIESZ = 255                 # the largest degree used for representers of measures (cost ~ N^2 per node)
    RIESZ_REL = 1e-6                # target relative H^s error of representers (algebraic convergence: see above)

    def __init__(self, measure, tol, s, length_scale, geometry):
        super().__init__(measure, tol)
        self.s, self.l, self.geometry = float(s), float(length_scale), geometry

    def weights(self, L):
        n = rows(L)[0]
        return (1.0 + self.l ** 2 * n * (n + 1) / self.R ** 2) ** self.s

    def gram(self, A, B):
        L = min(A.L, B.L)
        S = (L + 1) ** 2
        return self.R ** 2 * ((A.C[:S] * self.weights(L)[:, None]).T @ B.C[:S])

    def sqnorms(self, A):
        return self.R ** 2 * np.einsum("i,ij->j", self.weights(A.L), A.C * A.C)

    def fit(self, fun, N=1, breaks=None):
        # fits estimate the L2 size of what they discard (degrees > L); in H^s that is at least w_(L+1)^(1/2) times
        # as large: converted once (a lower-bound estimate), so that every SphericalHarmonicSeries.err of this backend is in the H^s norm
        R, u = super().fit(fun, N, breaks)
        R.err = R.err * math.sqrt((1.0 + self.l ** 2 * (R.L + 1) * (R.L + 2) / self.R ** 2) ** self.s)
        return R, u

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F"):
        R = super().fit_family(fun_idx, N, breaks=breaks, sparse=sparse, name=name)
        R.err = R.err * math.sqrt((1.0 + self.l ** 2 * (R.L + 1) * (R.L + 2) / self.R ** 2) ** self.s)
        return R

    def error(self, R):
        return R.err

    def to_l2(self, R):
        """The same function as an element of L2 of the sphere (errors in H^s bound those in L2: w_n >= 1)."""
        return R

    def _nodes(self, measures, L):
        """Quadrature nodes (theta, phi), weights and owners for the measures (points; arcs: Gauss in arc length,
        enough nodes for trigonometric polynomials of degree L along the arc; signed measures: their terms, weights
        times the coefficients)."""
        from ...measures import DiracMeasure, LebesgueMeasure, terms_of
        from ...domains import Arc
        nodes, wts, owner = [], [], []
        for i, c, mu in ((i, c, m) for i, nu in enumerate(measures) for c, m in terms_of(nu)):
            if isinstance(mu, DiracMeasure):
                x = np.atleast_2d(np.asarray(mu.x0, float))
                if x.shape[1] != 3 or not self.domain.contains(x[0]):
                    raise SupportOutsideDomain(f"{mu} is not a point of {self.domain}")
                nodes.append(x), wts.append(np.full(1, c))
                owner.append(np.full(1, i))
            elif isinstance(mu, LebesgueMeasure) and isinstance(mu.set, Arc):
                A = mu.set
                if A.sphere != self.domain:
                    raise SupportOutsideDomain(f"{A} is not on {self.domain}")
                q = int(math.ceil(0.6 * L * A.angle)) + 16
                t, w = gauss(q)
                nodes.append(np.column_stack(A.to_cartesian(0.5 * A.volume * (t + 1))))
                wts.append(c * 0.5 * A.volume * w), owner.append(np.full(q, i))
            else:
                raise Unsupported(f"Riesz representers of {mu} on the sphere are not implemented (points and arcs are)")
        X = np.vstack(nodes)
        th, ph = self.domain.from_cartesian(X[:, 0], X[:, 1], X[:, 2])
        return th, ph, np.concatenate(wts), np.concatenate(owner)

    def _moments(self, measures, L):
        """mu_nm = int Y_nm dmu, packed ((L+1)^2, n)."""
        th, ph, W, O = self._nodes(measures, L)
        starts = np.concatenate([[0], np.nonzero(np.diff(O))[0] + 1])
        own = O[starts]
        out = np.zeros(((L + 1) ** 2, len(measures)))
        cs, sn = np.cos(th), np.abs(np.sin(th))
        for m, (rc, rs) in enumerate(order_rows(L)):
            Pm = assoc(m, L, cs, sn)
            if m == 0:
                out[np.ix_(rc, own)] = np.add.reduceat(Pm * W, starts, axis=1)
            else:
                out[np.ix_(rc, own)] = math.sqrt(2) * np.add.reduceat(Pm * (W * np.cos(m * ph)), starts, axis=1)
                out[np.ix_(rs, own)] = math.sqrt(2) * np.add.reduceat(Pm * (W * np.sin(m * ph)), starts, axis=1)
        return out

    def _tail(self, e, L):
        """Estimated sum_{n > L} e_n from a power law fitted to the last octave of the degree energies e."""
        lo = max(2, L // 2)
        ls = np.arange(lo, L + 1)
        out = np.empty(e.shape[1])
        for j in range(e.shape[1]):
            v = e[lo:, j]
            ok = v > 0
            if ok.sum() < 4:                                  # (almost) nothing left: the last octave bounds the tail
                out[j] = float(v.sum())
                continue
            p, b = np.polyfit(np.log(ls[ok]), np.log(v[ok]), 1)
            eL = math.exp(b) * L ** p                         # the fitted energy at L (degree energies oscillate)
            out[j] = eL * L / (-p - 1) if p < -1 else np.inf
        return out

    def riesz(self, measures):
        for L in [x for x in (63, 127, 255, 511) if x < self.MAX_RIESZ] + [self.MAX_RIESZ]:
            mu = self._moments(measures, L)
            w = self.weights(L)
            e = degree_energy(mu / np.sqrt(self.R ** 2 * w)[:, None], L)       # |mu_n|^2 / (R^2 w_n) per degree
            tail = self._tail(e, L)
            if np.all(tail <= self.RIESZ_REL ** 2 * e.sum(axis=0)):
                break
        return SphericalHarmonicSeries(L, mu / (self.R ** 2 * w)[:, None], np.sqrt(tail), self.R)
