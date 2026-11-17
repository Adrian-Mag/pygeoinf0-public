"""Automatic witness cells for a pointwise box on sphere summands (plan 061 P3; the interval route is
optimisers/dual/nonsmooth.py interval_cells).

Cells: the latitude-longitude grid with n_lat = sub (L + 1) equal latitude bands on [-pi/2, pi/2] and 2 n_lat equal
longitude sectors, L the largest degree of the data representers (and of the direction). The indicator 1_E of a cell is
not band-limited, so it is not an element of the harmonic realisation; but it is an element of L2(S_R), it lies in the
box with the value v_E for every constant l <= v_E <= u (the sphere's box has constant bounds per summand), and every
quantity the cell problem needs is an integral of a band-limited function over E:

    (G 1_E)_i = <r_i, 1_E>_X = w R^2 int_E r_i dxi,     <c, 1_E>_X = w R^2 int_E c dxi

(r_i the representers of G's plain coordinates, w the summand's weight, R its radius), exact from the torus series of
Y_lj cos(phi) (realisations/bounds/sphere_l1.py; THEOREM, as harmonic_cell_integrals), so the cell witnesses of
optimisers/dual/nonsmooth.py are elements of the box with exactly computed data and targets, as on intervals. The images are
applied in factored form (SphereGrid: latitude integrals per harmonic times longitude integrals per order), so neither
the (cells x harmonics) nor the (cells x data) matrix is formed: 7200 cells x 7400 data at degree 14 costs two small
products per application. The resolution is K.sub (default 4): the bracket narrows as the cells shrink, and the dual
side (the certificate) does not depend on the cells.

Refused, with the reason: varying bounds on a sphere (as in the support itself), summands that are neither spheres nor
all spheres (a direct sum that mixes spheres with intervals: give cells=), representers not realised by harmonics.
"""
import math

import numpy as np

from ...core.errors import Unsupported
from .nonsmooth import CellProblem


class SphereGrid:
    """Exact integrals of the package's harmonics (unit sphere) over the cells of a latitude-longitude grid (module
    docstring): Lat[p, row] = int_(band p) p_lm(sin phi) cos(phi) dphi, and per order m the longitude integrals of
    cos(m lambda) and sin(m lambda) (times sqrt 2 for m > 0)."""

    def __init__(self, L, lat_edges, lon_edges):
        from ...spaces.harmonic import assoc, order_rows
        self.L = int(L)
        self.lat, self.lon = np.asarray(lat_edges, float), np.asarray(lon_edges, float)
        if not (self.lat[0] >= -math.pi / 2 - 1e-12 and self.lat[-1] <= math.pi / 2 + 1e-12 and np.all(np.diff(self.lat) > 0)):
            raise ValueError("latitude edges: increasing, within [-pi/2, pi/2]")
        L = self.L
        N = 2 * L + 4
        ang = 2 * math.pi * np.arange(N) / N
        mu, c = np.sin(ang), np.cos(ang)
        s, sg = np.abs(c), np.where(c < 0, -1.0, 1.0)
        kf = np.arange(-(L + 1), L + 2)
        a, b = self.lat[:-1, None], self.lat[1:, None]
        h, mid = 0.5 * (b - a), 0.5 * (a + b)
        Ep = np.exp(1j * kf * mid) * 2 * h * np.sinc(kf * h / math.pi)        # (n_lat, Kf): int_band e^(ik phi)
        self.rows = order_rows(L)
        nlm = (L + 1) ** 2
        self.Lat = np.zeros((len(a), nlm))
        for m, (rc, rs) in enumerate(self.rows):
            P = assoc(m, L, mu, s).T * (sg ** m * c)[:, None]                  # (N, l = m..L): the trig polynomial
            Fp = np.fft.fft(P, axis=0) / N                                     # its Fourier coefficients (exact)
            band = (Ep @ Fp[kf % N]).real * (1.0 if m == 0 else math.sqrt(2))
            self.Lat[:, rc] = band
            if rs is not None:
                self.Lat[:, rs] = band
        m = np.arange(L + 1)
        x0, x1 = self.lon[:-1, None], self.lon[1:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.LonC = np.where(m == 0, x1 - x0, (np.sin(m * x1) - np.sin(m * x0)) / np.where(m == 0, 1, m))
            self.LonS = np.where(m == 0, 0.0, (np.cos(m * x0) - np.cos(m * x1)) / np.where(m == 0, 1, m))
        self.shape = (len(a), len(x0))

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def coefficients(self, V):
        """sum_E V_E int_E Y_row for cell values V (cells,) or (cells, k): shape ((L + 1)^2,) or ((L + 1)^2, k)."""
        V = np.asarray(V, float)
        one = V.ndim == 1
        V = V.reshape(self.shape + (-1,))
        Wc, Ws = np.einsum("pqk,qm->pmk", V, self.LonC), np.einsum("pqk,qm->pmk", V, self.LonS)
        out = np.zeros(((self.L + 1) ** 2, V.shape[2]))
        for m, (rc, rs) in enumerate(self.rows):
            out[rc] = self.Lat[:, rc].T @ Wc[:, m]
            if rs is not None:
                out[rs] = self.Lat[:, rs].T @ Ws[:, m]
        return out[:, 0] if one else out

    def values(self, B):
        """int_E sum_row B_row Y_row for every cell E, B ((L + 1)^2,) or ((L + 1)^2, k): shape (cells,) or (cells, k)."""
        B = np.asarray(B, float)
        one = B.ndim == 1
        B = B.reshape((self.L + 1) ** 2, -1)
        Ac = np.zeros((self.shape[0], self.L + 1, B.shape[1]))
        As = np.zeros_like(Ac)
        for m, (rc, rs) in enumerate(self.rows):
            Ac[:, m] = self.Lat[:, rc] @ B[rc]
            if rs is not None:
                As[:, m] = self.Lat[:, rs] @ B[rs]
        out = (np.einsum("pmk,qm->pqk", Ac, self.LonC) + np.einsum("pmk,qm->pqk", As, self.LonS)).reshape(self.size, -1)
        return out[:, 0] if one else out


def is_sphere_box(box):
    """True when every summand of the box's space is L2 of a sphere with its surface measure or a constant density times
    it (plan 064), realised in spherical harmonics (the sphere cell route applies)."""
    from ...measures import constant_density
    from ...spaces.base import L2
    from ...domains import Sphere
    M = box.space
    spaces = M.summands if box.summed else [M]
    return all(isinstance(H, L2) and isinstance(getattr(H, "domain", None), Sphere)
               and constant_density(getattr(H, "measure", None)) is not None and hasattr(H.backend, "mass")
               for H in spaces)


def _sh_parts(x, box):
    """Per summand: (SphericalHarmonicSeries realisation or None, member indices) of an element or family on the box's space."""
    R = x.realisation
    if not box.summed:
        return [(R, None)]
    return [(None, None) if p is None else p for p in R.parts]


def sphere_representers(box, G, cache, chunk=512):
    """Per summand the harmonic coefficients ((L_k + 1)^2, n) of the representers r_i of G's plain coordinates
    ((G m)_i = <r_i, m>, r_i = G*(W^-1 e_i)), with their degree, radius and largest stated error; cached per G."""
    key = ("sphere_reps", id(G))
    hit = cache.get(key)
    if hit is not None and hit[0] is G:
        return hit[1]
    from ...realisations.registry import require
    D = G.codomain
    n = D.dim
    nk = len(box.space.summands) if box.summed else 1
    blocks = [[] for _ in range(nk)]
    for i in range(0, n, chunk):
        E = np.eye(n)[:, i:i + chunk]
        F = G.adjoint(D.family(np.asarray(D.solve_metric(E), float).T))
        for k, (R, idx) in enumerate(_sh_parts(F, box)):
            blocks[k].append((i, E.shape[1], R, idx))
    out = []
    for k in range(nk):
        Rs = [R for _, _, R, _ in blocks[k] if R is not None]
        if not Rs:
            out.append(None)
            continue
        for R in Rs:
            require(R, "harmonic_coefficients", "sphere witness cells need the representers realised by spherical "
                    "harmonics on every sphere summand; give cells= otherwise")
        L = max(R.L for R in Rs)
        C = np.zeros(((L + 1) ** 2, n))
        err = 0.0
        for i, w, R, idx in blocks[k]:
            if R is None:
                continue
            cols = i + (np.arange(w) if idx is None else np.asarray(idx, int))
            C[:, cols] = R.harmonic_coefficients(L)
            err = max(err, float(np.max(R.err, initial=0.0)))
        out.append((L, C, float(Rs[0].R), err))
    cache[key] = (G, out)
    return out


def sphere_cells(box, G, c, sub=4, cache=None):
    """The CellProblem of a box on sphere summands (module docstring): cells of a latitude-longitude grid with
    sub (L + 1) bands per summand; data images and targets exact (factored, SphereGrid); cp.images(Op) the (cells, dim)
    images Op(1_E) for another operator on the same space."""
    from ...realisations.registry import capability
    cache = {} if cache is None else cache
    if not is_sphere_box(box):
        raise Unsupported("sphere witness cells need every summand to be L2 of a sphere; give cells= for a mix")
    sub = int(sub)
    if sub < 1:
        raise ValueError("sub: cells per (pi / (L + 1)) of latitude, an integer >= 1")
    M = box.space
    weights = list(M.weights) if box.summed else [1.0]
    dens = [H.backend.c for H in (M.summands if box.summed else [M])]       # constant densities (plan 064)
    for l, u in zip(box.lower, box.upper):
        if not (isinstance(l, float) and isinstance(u, float)):
            raise Unsupported("sphere witness cells need finite constant bounds on every sphere summand (varying "
                              "bounds are refused on the sphere, as in its support: split the space into zones)")
    reps = sphere_representers(box, G, cache)
    cparts = _sh_parts(c, box) if hasattr(c, "realisation") else [(None, None)] * len(reps)
    grids, lo, hi, ts, info = [], [], [], [], []
    for k, rep in enumerate(reps):
        Rc = cparts[k][0]
        Lr = rep[0] if rep is not None else 0
        Lc = Rc.L if Rc is not None else 0
        L = max(Lr, Lc, 1)
        nl = sub * (max(Lr, 1) + 1)
        gkey = ("sphere_grid", k, L, nl)
        grid = cache.get(gkey)
        if grid is None:
            grid = cache[gkey] = SphereGrid(L, np.linspace(-math.pi / 2, math.pi / 2, nl + 1),
                                            np.linspace(0.0, 2 * math.pi, 2 * nl + 1))
        rad = rep[2] if rep is not None else float(Rc.R) if Rc is not None else 1.0
        scale = weights[k] * dens[k] * rad ** 2
        grids.append((grid, rep, scale))
        lo.append(np.full(grid.size, box.lower[k]))
        hi.append(np.full(grid.size, box.upper[k]))
        ts.append(scale * grid.values(capability(Rc, "harmonic_coefficients", "sphere witness cells")(L)[:, 0])
                  if Rc is not None else np.zeros(grid.size))
        info.append(f"{grid.shape[0]} x {grid.shape[1]}")
    sizes = [g.size for g, _, _ in grids]
    cuts = np.cumsum([0] + sizes)
    n = G.codomain.dim

    def coef(rep, grid, L):
        """The representers' coefficients at the grid's degree."""
        Lr, C, _, _ = rep
        if Lr == L:
            return C
        out = np.zeros(((L + 1) ** 2, C.shape[1]))
        out[:(Lr + 1) ** 2] = C
        return out
    mats = [None if rep is None else coef(rep, g, g.L) for g, rep, _ in grids]

    def apply(v):                                             # H^T v: the data image of the cell field v
        out = np.zeros(n)
        for (g, rep, s), Cm, a, b in zip(grids, mats, cuts[:-1], cuts[1:]):
            if Cm is not None:
                out += s * (Cm.T @ g.coefficients(v[a:b]))
        return out

    def adjoint(z):                                           # H z: per cell <G* W^-1 z, 1_E>
        return np.concatenate([np.zeros(g.size) if Cm is None else s * g.values(Cm @ z)
                               for (g, rep, s), Cm in zip(grids, mats)])
    err = max([rep[3] for rep in reps if rep is not None], default=0.0)
    note = f"{int(cuts[-1])} sphere cells ({', '.join(info)} latitude x longitude)" + \
        (f"; representers realised with stated error {err:.1e}" if err > 0 else "")
    cp = CellProblem(np.concatenate(lo), np.concatenate(hi), np.concatenate(ts), apply=apply, adjoint=adjoint,
                     note=note)
    cp.key = ("sphere", sub, tuple((g.L, g.shape) for g, _, _ in grids), id(G))

    def images(Op):
        P = sphere_representers(box, Op, cache)
        cols = []
        for (g, _, s), rep in zip(grids, P):
            if rep is None:
                cols.append(np.zeros((g.size, Op.codomain.dim)))
                continue
            Lr, C, _, _ = rep
            if Lr > g.L:
                raise Unsupported("the operator's representers exceed the cells' degree; give cells=")
            Cp = np.zeros(((g.L + 1) ** 2, C.shape[1]))
            Cp[:(Lr + 1) ** 2] = C
            cols.append(s * g.values(Cp))
        return np.vstack(cols)
    cp.images = images
    return cp
