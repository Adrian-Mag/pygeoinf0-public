"""L1 integrals of band-limited functions on the sphere, exact (zonal) or certified (general), and of radial x harmonic
fields on balls and shells (plan 009 P5, G8 (c); EXP-030 DESIGN G8 and NOTES section 8). The support function of a
pointwise box is an L1 integral (convex/pointwise.py), so these brackets are what makes box BGP certified.

Zonal functions p(xi . eta) with p a polynomial of degree n: int_S |p| = 2 pi int_{-1}^{1} |p(x)| dx, exact from the
real roots of p in (-1, 1) (zonal_l1); the signed moments int sign(p) P_s dx of the extremal field likewise
(zonal_sign_moments). DERIVED: a polynomial is integrated exactly between consecutive roots.

General band-limited g of degree <= n (l1_bracket). With phi the latitude and lambda the longitude, every Y_lm is
p_lm(sin phi) cos(m lambda) or sin(m lambda) with p_lm(sin phi) = cos^m(phi) poly_(l-m)(sin phi), so

    p(phi, lambda) = g(x(phi, lambda)) cos(phi),  x = (cos phi cos lambda, cos phi sin lambda, sin phi),

defined by the same formula for every (phi, lambda) of the torus, is a trigonometric polynomial of degree n + 1 in phi
and n in lambda (DERIVED), its values on the torus are values of g cos phi on the sphere (x(pi - phi, lambda) =
x(phi, lambda + pi)), so sup |p| <= ||g||_inf, and int_S |g| = int |p| dphi dlambda over [-pi/2, pi/2] x [0, 2 pi).
Its Fourier coefficients come exactly from 2n + 4 samples per axis (no aliasing); then over a cell of half-width h
(both axes), with S_c >= sup over the cell of |p_phiphi| + 2 |p_philam| + |p_lamlam|:
    int_cell p                exact (Fourier: e^(ikc) 2 sin(kh) / k per axis)
    |p - l| <= R_c = (1/2) h^2 S_c   for the linearisation l of p at the centre (Taylor along the segment, THEOREM)
    no zero in the cell if |p(c)| > (|p_phi| + |p_lambda|) h + R_c; then int_cell |p| = |int_cell p| (exact)
    otherwise max(|int p|, int |l| - R_c area) <= int_cell |p| <= int |l| + R_c area, int |l| in closed form
S_c is local (plan 061; local_curvature): the de la Vallee Poussin formula q(x) = (1/M) sum_j q(x_j) V(x - x_j), exact
for trigonometric polynomials of degree <= n on M > 2n nodes per axis, with |V| bounded by Fejer kernels, bounds the
second derivatives (sampled exactly) on each base cell by a weighted local sum of their samples (THEOREM). A sub-cell
keeps its base cell's S_c. Until plan 061 the remainder used one global sup bound of p for every cell: for a
concentrated q (a cap) no cell away from it was ever decided, the undecided cells filled the torus, the cell budget
stopped the refinement whatever rtol was, and the upper end was 1.4-6x the value.
Undecided cells are split in four, those of smallest gap frozen within half the tolerance, until the bracket is within
rtol: the gap of an undecided cell is <= 2 R_c area = 4 h^4 S_c, so the relative gap is O((n h)^3) where q oscillates
(S_c ~ n^2 times the local amplitude) and no longer grows with the size of q elsewhere. A margin for the rounding of
the cell sums (1e-12 of the coefficients' l1 norm times the torus area) is added. info says why it stopped
(rtol / decided / max_level / max_cells) and converged=False when the bracket is wider than rtol (still valid). Memory
is bounded: one function at a time, cells in chunks.

Radial x harmonic fields f(r, xi) = sum_j f_j(r) Y_j(xi) with f_j piecewise polynomial (radial_l1_bracket): on a
radial piece with local variable t in [-1, 1], f_j = a_j + b_j t + e_j(t), |e_j| <= sum_(k >= 2) |c_jk| (Legendre
coefficients, |P_k| <= 1); h(t) = int_S |sum (a_j + b_j t) Y_j| is convex in t, so its integral lies between the
midpoint and trapezoid values, and the remainder changes int_S |f| by at most sqrt(4 pi) (sum_j e_j^2)^(1/2)
(Cauchy-Schwarz). Pieces are halved until the bracket is within rtol (second order in the radial width).
When the radial factors are known to be affine on given pieces (tabulated kernels read linearly) times a positive
weight with known moments (a taper, an envelope), affine_radial_l1_bracket needs three sphere brackets per piece and no
refinement; harmonic_cell_integrals gives int_cell Y_lj exactly for the cells of a latitude-longitude grid.
"""
import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...spaces.harmonic import values as harmonic_values
from ..series.legendre import restrict
from ...spaces.harmonic import assoc

CHUNK = 1 << 15                 # cells per vectorised step
UPSAMPLE = 1024                 # grid of the sup bound (per torus axis)


# ------------------------------------------------------------------------------------------------ zonal
def _real_roots(c):
    """The real roots of the Legendre series c in (-1, 1), polished by Newton, sorted."""
    c = np.trim_zeros(np.asarray(c, float), "b")
    if len(c) <= 1:
        return np.zeros(0)
    r = npleg.legroots(c)
    r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    dc = npleg.legder(c)
    for _ in range(3):
        d = npleg.legval(r, dc)
        r = r - np.where(d != 0, npleg.legval(r, c) / np.where(d != 0, d, 1.0), 0.0)
    return np.unique(r[(r > -1) & (r < 1)])


def zonal_l1(c):
    """int_S |sum_s c_s P_s(xi . eta)| d xi = 2 pi int_{-1}^{1} |p| dx, exact (module docstring); c: Legendre series."""
    c = np.asarray(c, float)
    pts = np.r_[-1.0, _real_roots(c), 1.0]
    F = npleg.legval(pts, npleg.legint(c))
    return float(2 * math.pi * np.sum(np.abs(np.diff(F))))


def zonal_sign_moments(c, nmax):
    """int_{-1}^{1} sign(p(x)) P_s(x) dx for s = 0..nmax (p the Legendre series c): exact between the roots. The
    extremal field u sign(p) of a symmetric box has the Legendre coefficients (2s + 1)/2 times these."""
    pts = np.r_[-1.0, _real_roots(c), 1.0]
    mid = 0.5 * (pts[:-1] + pts[1:])
    sg = np.sign(npleg.legval(mid, np.asarray(c, float)))
    out = np.zeros(nmax + 1)
    for s in range(nmax + 1):
        e = np.zeros(s + 1)
        e[s] = 1.0
        F = npleg.legval(pts, npleg.legint(e))
        out[s] = float(np.sum(sg * np.diff(F)))
    return out


# ------------------------------------------------------------------------------------------------ torus series
def torus_values(C, L, N):
    """Values of p = g cos(phi) at the N x N torus nodes (2 pi a / N, 2 pi b / N) (phi, lambda) for the columns of C
    ((L + 1)^2, ncol), shape (N, N, ncol): per order m, a_m(phi) = sum_l C_lm p_lm(sin phi) with the signed
    cos^m(phi) (the formula of the module docstring on the whole torus), then the longitude sum. Equals
    harmonics.values at the points x(phi, lambda) times cos(phi)."""
    from ...spaces.harmonic import order_rows
    C = np.asarray(C, float).reshape((L + 1) ** 2, -1)
    ang = 2 * math.pi * np.arange(N) / N
    mu, c = np.sin(ang), np.cos(ang)
    s, sg = np.abs(c), np.where(c < 0, -1.0, 1.0)
    out = np.zeros((N, N, C.shape[1]))
    for m, (rc, rs) in enumerate(order_rows(L)):
        P = assoc(m, L, mu, s).T * (sg ** m)[:, None]                   # (N, l = m..L): p_lm(sin phi), signed cos^m
        if m == 0:
            out += (P @ C[rc])[:, None, :]
        else:
            out += math.sqrt(2) * ((P @ C[rc])[:, None, :] * np.cos(m * ang)[None, :, None]
                                   + (P @ C[rs])[:, None, :] * np.sin(m * ang)[None, :, None])
    return out * c[:, None, None]


def torus_coefficients(C, L):
    """Fourier coefficients F[col, k + L + 1, m + L] (k in [-(L+1), L+1], m in [-L, L]) of p = g cos(phi) for the
    columns of packed harmonic coefficients C ((L + 1)^2, N) in the package's basis (module docstring)."""
    C = np.asarray(C, float).reshape((L + 1) ** 2, -1)
    N = 2 * L + 4
    F = np.fft.fft2(torus_values(C, L, N), axes=(0, 1)) / (N * N)
    kf, km = np.arange(-(L + 1), L + 2), np.arange(-L, L + 1)
    return np.moveaxis(F[np.ix_(kf % N, km % N)], -1, 0)


def sup_bound(F, L, up=UPSAMPLE):
    """G >= sup |p| for one torus series F (module docstring)."""
    G1 = float(np.abs(F).sum())
    delta = 2 * math.pi / up
    shrink = 1.0 - (2 * L + 1) * delta / 2
    if shrink <= 0.5:
        return G1
    kf, km = np.arange(-(L + 1), L + 2), np.arange(-L, L + 1)
    pad = np.zeros((up, up), complex)
    pad[np.ix_(kf % up, km % up)] = F
    vals = np.fft.ifft2(pad) * (up * up)
    return min(G1, float(np.abs(vals.real).max()) / shrink)


def rect_abs_linear(c, a, b, hx, hy):
    """int_{-hx}^{hx} int_{-hy}^{hy} |c + a x + b y| dy dx, exact (arrays c, a, b; scalars or arrays hx, hy): the
    inner integral J(t) = int |t + a x| dx is piecewise quadratic in t, so Simpson's rule between its breakpoints is
    exact."""
    c, a, b = np.broadcast_arrays(*(np.asarray(v, float) for v in (c, a, b)))
    hx, hy = np.broadcast_to(hx, c.shape), np.broadcast_to(hy, c.shape)
    aa = np.abs(a) * hx

    def J(t):
        with np.errstate(divide="ignore", invalid="ignore"):
            inner = (t * t + aa * aa) / np.where(aa > 0, np.abs(a), 1.0)
        return np.where(np.abs(t) >= aa, 2 * hx * np.abs(t), inner)

    with np.errstate(divide="ignore", invalid="ignore"):
        y1, y2 = (-aa - c) / b, (aa - c) / b
    y1, y2 = np.where(b == 0, -hy, y1), np.where(b == 0, -hy, y2)
    y1, y2 = np.clip(np.minimum(y1, y2), -hy, hy), np.clip(np.maximum(y1, y2), -hy, hy)
    out = np.zeros(c.shape)
    for s, e in ((-hy, y1), (y1, y2), (y2, hy)):
        s, e = np.broadcast_to(s, c.shape), np.broadcast_to(e, c.shape)
        m = 0.5 * (s + e)
        out += (e - s) / 6.0 * (J(c + b * s) + 4.0 * J(c + b * m) + J(c + b * e))
    return out


def _fejer_sup(N, s):
    """sup of the Fejer kernel F_N on an interval where sin^2(t / 2) >= s: F_N(t) = sin^2(N t / 2) / (N sin^2(t / 2))
    <= min(N, 1 / (N s)) (F_N <= F_N(0) = N)."""
    with np.errstate(divide="ignore"):
        return np.minimum(float(N), 1.0 / (N * s))


def vp_cell_kernel(n, M):
    """K[d] >= sup |V(t)| over t in [d delta, (d + 1) delta] (delta = 2 pi / M, d = 0..M-1) for the de la Vallee Poussin
    kernel V = (A F_A - B F_B) / (A - B), B = n, A = M - n, whose Fourier coefficients are 1 for |k| <= n and 0 for
    |k| >= A: for every trigonometric polynomial q of degree <= n, q(x) = (1/M) sum_j q(x_j) V(x - x_j) on the nodes
    x_j = x_0 + j delta (no aliasing: A - 1 + n < M), so sup over [x_i, x_i + delta] of |q| <= (1/M) sum_j |q(x_j)|
    K[(i - j) mod M] (THEOREM; |V| <= (A F_A + B F_B) / (A - B), F_N >= 0, and _fejer_sup)."""
    A, B = M - n, n
    if A <= B:
        raise ValueError(f"the de la Vallee Poussin kernel needs M > 2n nodes (M = {M}, n = {n})")
    d = np.arange(M)
    e = np.sin(0.5 * d * (2 * math.pi / M)) ** 2
    s = np.minimum(e, np.roll(e, -1))                                   # min of sin^2(t/2) at the interval's ends
    s[0] = s[M - 1] = 0.0                                               # [0, delta] and [2 pi - delta, 2 pi] hold 0
    return (A * _fejer_sup(A, s) + B * _fejer_sup(B, s)) / (A - B)


def local_curvature(F, L, M):
    """S[i, j] >= sup over the torus cell [-pi/2 + i delta, ... + delta] x [j delta, (j + 1) delta] (delta = 2 pi / M,
    i < M / 2) of |p_phiphi| + 2 |p_philam| + |p_lamlam|, for the torus series F of p (module docstring): the second
    derivatives are trigonometric polynomials of the same degrees (n + 1 in phi, n in lambda), sampled exactly on the
    M x M nodes by their Fourier multipliers, and the tensor de la Vallee Poussin bound (vp_cell_kernel) per axis makes
    S local (THEOREM). A rounding margin (1e-12 of the coefficients' l1 norm, which bounds every sample) is added."""
    kf, km = np.arange(-(L + 1), L + 2), np.arange(-L, L + 1)
    shift = np.exp(-0.5j * math.pi * kf)[:, None]                       # nodes phi_j = -pi/2 + j delta
    Q = np.zeros((M, M))
    margin = 0.0
    for wgt, a, b in ((1.0, 2, 0), (2.0, 1, 1), (1.0, 0, 2)):
        Fd = F * shift * ((1j * kf) ** a)[:, None] * ((1j * km) ** b)[None, :]
        pad = np.zeros((M, M), complex)
        pad[np.ix_(kf % M, km % M)] = Fd
        Q += wgt * np.abs((np.fft.ifft2(pad) * (M * M)).real)
        margin += wgt * 1e-12 * float(np.abs(Fd).sum())
    K1, K2 = vp_cell_kernel(L + 1, M), vp_cell_kernel(L, M)
    i, j = np.arange(M // 2)[:, None], np.arange(M)[None, :]
    C1 = K1[(i - j) % M]                                                # (M/2, M)
    jj = np.arange(M)
    C2 = K2[(jj[:, None] - jj[None, :]) % M]                            # (M, M)
    return (C1 @ (Q + margin) @ C2.T) * ((1.0 + 1e-12) / (M * M))


def _one(F, L, rtol, atol, n0, max_level, max_cells):
    """(lower, upper, info) of int |p| over [-pi/2, pi/2] x [0, 2 pi) for one torus series F: cells of half-width h,
    each with its own remainder R_c = (1/2) h^2 S_c (local_curvature of its base cell); undecided cells are split in
    four, those whose gaps are smallest frozen within half the tolerance, until upper - lower <= rtol lower + atol."""
    M = 2 * n0
    S = local_curvature(F, L, M).ravel()                                # per base cell (i_phi * M + i_lambda)
    G1 = float(np.abs(F).sum())
    kf, km = np.arange(-(L + 1), L + 2), np.arange(-L, L + 1)
    h = math.pi / (2 * n0)
    IP, IL = (v.ravel() for v in np.meshgrid(np.arange(n0), np.arange(2 * n0), indexing="ij"))   # cell indices
    base = IP * M + IL
    sure_lo = sure_hi = 0.0
    frozen = 0.0
    # rounding of the cell sums, every term of which is bounded by G1 (|p|, |I| <= G1 area): far below any tolerance
    fl = 1e-12 * G1 * 2 * math.pi ** 2
    for level in range(max_level + 1):
        area = 4 * h * h
        S1 = 2 * h * np.sinc(np.r_[kf, km] * h / math.pi)
        Sp, Sl = S1[:len(kf)], S1[len(kf):]
        lo_c, hi_c = [], []
        for i in range(0, len(IP), CHUNK):
            up, ip = np.unique(IP[i:i + CHUNK], return_inverse=True)       # latitude rows: shared work
            ul, il = np.unique(IL[i:i + CHUNK], return_inverse=True)
            ep = np.exp(1j * np.outer(-math.pi / 2 + h * (2 * up + 1), kf))
            el = np.exp(1j * np.outer(h * (2 * ul + 1), km))
            A, Ad, AI = ep @ F, (ep * (1j * kf)) @ F, (ep * Sp) @ F
            A, Ad, AI, el = A[ip], Ad[ip], AI[ip], el[il]
            v = np.einsum("cm,cm->c", A, el).real
            dl = np.einsum("cm,cm->c", A, el * (1j * km)).real
            dp = np.einsum("cm,cm->c", Ad, el).real
            I = np.abs(np.einsum("cm,cm->c", AI, el * Sl).real)
            R = 0.5 * h * h * S[base[i:i + CHUNK]]
            sure = np.abs(v) > (np.abs(dp) + np.abs(dl)) * h + R
            Jl = rect_abs_linear(v, dp, dl, h, h)
            lo_c.append(np.where(sure, I, np.maximum(I, Jl - R * area)))
            hi_c.append(np.where(sure, I, Jl + R * area))
        lo_c, hi_c = np.concatenate(lo_c), np.concatenate(hi_c)
        gap = hi_c - lo_c
        lo, hi = sure_lo + lo_c.sum(), sure_hi + hi_c.sum()
        target = rtol * lo + atol
        und = np.flatnonzero(gap > 0)
        stop = "rtol" if hi - lo <= target else "decided" if not len(und) else \
            "max_level" if level == max_level else "max_cells" if 4 * len(und) > max_cells else None
        if stop:
            info = dict(level=level, nodal_cells=int(len(und)), cell_half_width=h, stop=stop,
                        converged=bool(hi - lo <= target), sup=G1)
            return max(lo - fl, 0.0), hi + fl, info
        # freeze the cells of smallest gap within half of what is left of half the tolerance; split the rest
        order = und[np.argsort(gap[und])]
        room = 0.5 * max(0.5 * target - frozen, 0.0)
        k = int(np.searchsorted(np.cumsum(gap[order]), room, side="right"))
        frozen += float(gap[order[:k]].sum())
        keep = np.ones(len(gap), bool)
        keep[order[k:]] = False
        sure_lo += float(lo_c[keep].sum())
        sure_hi += float(hi_c[keep].sum())
        idx = order[k:]
        h /= 2
        IP = np.concatenate([2 * IP[idx] + sp for sp in (0, 0, 1, 1)])
        IL = np.concatenate([2 * IL[idx] + sl for sl in (0, 1, 0, 1)])
        base = np.tile(base[idx], 4)


def l1_bracket(C, L, rtol=1e-8, atol=0.0, n0=None, max_level=14, max_cells=1 << 23):
    """Certified brackets (lower, upper) of int_S |g| for every column of packed harmonic coefficients C ((L + 1)^2,
    N), package basis, unit sphere (module docstring); also a list of per-column info (levels, undecided cells, why it
    stopped, converged). Stops when upper - lower <= rtol lower + atol, or at max_level / max_cells (the bracket is
    still valid, info says converged=False)."""
    C = np.asarray(C, float).reshape((L + 1) ** 2, -1)
    n0 = n0 or 4 * (L + 1)
    lo, hi, info = np.zeros(C.shape[1]), np.zeros(C.shape[1]), []
    for j in range(C.shape[1]):
        if not np.any(C[:, j]):
            info.append(dict(level=0, nodal_cells=0, sup=0.0, stop="zero", converged=True))
            continue
        F = torus_coefficients(C[:, j:j + 1], L)[0]
        lo[j], hi[j], inf = _one(F, L, rtol, atol, n0, max_level, max_cells)
        info.append(inf)
    return lo, hi, info


def harmonic_cell_integrals(L, lat_edges, lon_edges):
    """int_cell Y_lj d xi for every packed harmonic (l <= L) and every cell of the latitude-longitude grid with the
    given edges (radians): shape (n_lat * n_lon, (L + 1)^2), exact (the torus series of Y_lj cos phi, module
    docstring). Cell indicators of a grid are box members whose data images these give exactly (optimisers/dual/nonsmooth.py)."""
    F = torus_coefficients(np.eye((L + 1) ** 2), L)                         # (nlm, Kf, Km)
    kf, km = np.arange(-(L + 1), L + 2), np.arange(-L, L + 1)

    def E(edges, k):
        a, b = np.asarray(edges[:-1], float)[:, None], np.asarray(edges[1:], float)[:, None]
        h, c = 0.5 * (b - a), 0.5 * (a + b)
        return np.exp(1j * k * c) * 2 * h * np.sinc(k * h / math.pi)
    Ep, El = E(lat_edges, kf), E(lon_edges, km)
    out = np.einsum("pk,jkm,qm->pqj", Ep, F, El).real
    return out.reshape(-1, F.shape[0])


# ------------------------------------------------------------------------------------------------ radial x harmonic
def affine_radial_l1_bracket(left, right, m0, m1, L, rtol=1e-4, mapper=map):
    """Certified bracket of sum_i int_(piece i) w(r) int_S |sum_j f_j(r) Y_j| d xi dr when on piece i every f_j is
    affine in the local variable t in [-1, 1], from left[i] (t = -1) to right[i] (t = 1) (packed coefficients
    (pieces, (L + 1)^2)), and w >= 0 has the moments m0[i] = int w dr and m1[i] = int w t dr there (a tabulated kernel
    read linearly, times a taper and an envelope). h(t) = int_S |f| is convex, so (DERIVED; Jensen with the weight w,
    and h below its chord)

        m0 h(t_c) <= int w h dr <= h(-1) (m0 - m1) / 2 + h(1) (m0 + m1) / 2,     t_c = m1 / m0 (the weight's centroid);

    the sphere integrals are certified brackets (l1_bracket at rtol). mapper: map, or a pool's map over columns (the
    sphere brackets of different columns are independent). Returns (lower, upper, per-piece (lower, upper))."""
    left, right = np.atleast_2d(left), np.atleast_2d(right)
    m0, m1 = np.asarray(m0, float), np.asarray(m1, float)
    tc = np.clip(np.where(m0 > 0, m1 / np.where(m0 > 0, m0, 1.0), 0.0), -1.0, 1.0)
    cols = np.concatenate([left, 0.5 * ((1 - tc)[:, None] * left + (1 + tc)[:, None] * right), right])
    out = list(mapper(_bracket_column, [(c, L, rtol) for c in cols]))
    lo, hi = np.array([o[0] for o in out]), np.array([o[1] for o in out])
    n = len(left)
    plo = m0 * lo[n:2 * n]
    phi = 0.5 * (hi[:n] * (m0 - m1) + hi[2 * n:] * (m0 + m1))
    return float(plo.sum()), float(phi.sum()), np.c_[plo, phi]


def _bracket_column(args):
    c, L, rtol = args
    lo, hi, _ = l1_bracket(np.asarray(c, float)[:, None], L, rtol=rtol)
    return float(lo[0]), float(hi[0])


def radial_l1_bracket(breaks, coefs, L, rtol=1e-6, sphere_rtol=None, max_rounds=12, max_pieces=4096):
    """Certified bracket of int_a^b int_S |sum_j f_j(r) Y_j(xi)| d xi dr for one field: f_j piecewise Legendre on
    breaks, coefs a list of (d, (L + 1)^2) arrays per piece (module docstring). A weight (r^2 dr, an envelope) is
    multiplied into f by the caller. Returns (lower, upper, info)."""
    srt = rtol / 4 if sphere_rtol is None else sphere_rtol
    pieces = [(float(breaks[i]), float(breaks[i + 1]), np.asarray(C, float)) for i, C in enumerate(coefs)]
    done_lo = done_hi = 0.0
    for rnd in range(max_rounds + 1):
        cols, rems = [], []
        for r0, r1, C in pieces:
            a0 = C[0]
            a1 = C[1] if C.shape[0] > 1 else np.zeros_like(a0)
            e = np.abs(C[2:]).sum(axis=0) if C.shape[0] > 2 else np.zeros_like(a0)
            cols += [a0 - a1, a0, a0 + a1]
            rems.append(math.sqrt(4 * math.pi) * float(np.sqrt(e @ e)))
        slo, shi, _ = l1_bracket(np.array(cols).T, L, rtol=srt) if cols else (np.zeros(0), np.zeros(0), None)
        width = np.array([r1 - r0 for r0, r1, _ in pieces])
        rems = np.array(rems)
        plo = np.maximum(width * (slo[1::3] - rems), 0.0)
        phi = width * (0.5 * (shi[0::3] + shi[2::3]) + rems)
        lo, hi = done_lo + plo.sum(), done_hi + phi.sum()
        gap = phi - plo
        if hi - lo <= rtol * lo or rnd == max_rounds or 2 * len(pieces) > max_pieces:
            return lo, hi, dict(rounds=rnd, pieces=len(pieces))
        split = gap > 0.25 * rtol * lo / max(len(pieces), 1)
        done_lo += plo[~split].sum()
        done_hi += phi[~split].sum()
        new = []
        for (r0, r1, C), s in zip(pieces, split):
            if s:
                m = 0.5 * (r0 + r1)
                new += [(r0, m, restrict(C, r0, r1, r0, m)), (m, r1, restrict(C, r0, r1, m, r1))]
        pieces = new
