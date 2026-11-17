"""Bases from the function bank (plan 025 P5): lists of bank functions for pg.Span(basis, space=M) (plan 068 P6).

The span is the finite subspace; pg.Discretisation(pg.Span(basis, space=M)) its approximation role.

    F.bases.hats(D, count, boundary=False)      P1 hats of a uniform grid of `count` nodes per axis (interior nodes, or
                                                all with boundary=True: the natural Neumann space of the form)
    F.bases.bsplines(D, degree, count)          the clamped uniform B-splines of `count` functions per axis
    F.bases.pixels(D, cells)                    indicators of the cells of a uniform grid (L2 only: they jump)
    F.bases.legendre(D, degree)                 orthonormal Legendre products of degree <= degree per axis
    F.bases.fourier(D, K)                       Fourier products 1, cos 2 pi k t, sin 2 pi k t, |k| <= K per axis
    F.bases.spherical_harmonics(S, degree)      the real (package-convention, orthonormal) harmonics of degree <= L
    F.bases.zernike(D, degree)                  orthonormal Zernike polynomials of radial order <= degree on a disk

Each is the bank's own family (functions/families: fem, indicators, polynomials), so every member carries its support (local fits: realisation="local"), regularity (membership:
pixels are refused in H^1) and closed forms. The order of the list is the order of the coefficients (C order over the
axes' indices).
"""
import itertools

import numpy as np

from . import geometry as geo


def _on(D):
    from . import on
    return on(D)


def _axes(D, value):
    """value per axis (an integer for all axes, or a tuple)."""
    d = 1 if geo.kind(D) == "interval" else len(D.sides)
    return tuple(value) if np.ndim(value) else (int(value),) * d


def _index(D, idx):
    return idx[0] if geo.kind(D) == "interval" else tuple(idx)


def hats(D, count, boundary=False):
    """The P1 hats of the uniform grid of `count` nodes per axis (count >= 2): the interior nodes, or all nodes."""
    c = _axes(D, count)
    ranges = [range(0, n) if boundary else range(1, n - 1) for n in c]
    f = _on(D).hat
    cnt = c[0] if geo.kind(D) == "interval" else c
    return [f(count=cnt, index=_index(D, idx)) for idx in itertools.product(*ranges)]


def bsplines(D, degree, count):
    """The clamped uniform B-splines of `degree` with `count` functions per axis."""
    c, p = _axes(D, count), _axes(D, degree)
    f = _on(D).bspline
    one = geo.kind(D) == "interval"
    return [f(degree=p[0] if one else p, count=c[0] if one else c, index=_index(D, idx))
            for idx in itertools.product(*[range(n) for n in c])]


def pixels(D, cells):
    """Indicators of the cells of a uniform grid of `cells` cells per axis on a box (closed cells: their faces are null
    sets in L2; pixels are not in H^s for s >= 1/2, and the bank refuses them there)."""
    c = _axes(D, cells)
    sides = [(float(s.a), float(s.b)) for s in D.sides]
    edges = [np.linspace(a, b, n + 1) for (a, b), n in zip(sides, c)]
    f = _on(D).box_indicator
    return [f(lower=tuple(e[i] for e, i in zip(edges, idx)), upper=tuple(e[i + 1] for e, i in zip(edges, idx)))
            for idx in itertools.product(*[range(n) for n in c])]


def legendre(D, degree):
    """Orthonormal Legendre products of degree <= `degree` on every axis (degree + 1)^d functions."""
    p = _axes(D, degree)
    f = _on(D).legendre
    return [f(index=_index(D, idx), norm="orthonormal") for idx in itertools.product(*[range(n + 1) for n in p])]


def fourier(D, K):
    """Fourier products with |k| <= K per axis: 1, cos(2 pi k t), sin(2 pi k t) (k > 0: cos, k < 0: sin)."""
    k = _axes(D, K)
    f = _on(D).fourier
    return [f(index=_index(D, idx)) for idx in itertools.product(*[range(-n, n + 1) for n in k])]


def spherical_harmonics(S, degree):
    """The real spherical harmonics of degree 0..L in the package convention ((L + 1)^2 functions, orthonormal on the
    unit sphere)."""
    f = _on(S).spherical_harmonic
    return [f(degree=l_, j=j) for l_ in range(int(degree) + 1) for j in range(2 * l_ + 1)]


def zernike(D, degree):
    """Orthonormal Zernike polynomials Z_n^m on a disk, n <= degree, |m| <= n, n - |m| even ((N + 1)(N + 2)/2)."""
    f = _on(D).zernike
    return [f(n=n, m=m, norm="orthonormal") for n in range(int(degree) + 1) for m in range(-n, n + 1, 2)]
