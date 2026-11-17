"""BGP with a pointwise box prior: a certified non-smooth dual and cell witnesses (plan 009 P5, G8 (d), (e); EXP-030
DESIGN G8, NOTES section 8). Decision 4 of plan 009: certified brackets are reported with their gap when the two sides
do not meet.

Problem: U = sup <c, m> over m in B (convex/pointwise.py) with e - G m in N (e = d - offset; N a ball or ellipsoid of D,
or a coordinate box of D). Weak duality (inf:874-895, appC:39-51): for every multiplier z (plain coordinates of D,
<lambda, eta>_D = z . eta),

    U <= phi(z) = z . e + sigma_B(c - G* W^-1 z) + sigma_N(-W^-1 z)                                         (THEOREM)

with sigma_N(-lambda) = -z . centre + rho ||Phi^-T z|| for a ball ||Phi (eta - centre)|| <= rho, and
sum_i max(-hi_i z_i, -lo_i z_i) for a box lo <= eta <= hi. Any z gives an upper bound once sigma_B is replaced by the
upper end of its certified bracket; the bound reported is phi at the final z evaluated that way, so it is certified
whatever the optimiser did.

Cells (the witness side). Fields that are constant on cells E_c (finitely many disjoint sets of positive measure, on
which lo_c >= l and hi_c <= u) with values v_c in [lo_c, hi_c] lie in B exactly, and their data and target values are
the exact sums sum_c v_c G(1_c), sum_c v_c <c, 1_c> (THEOREM, by linearity). A fixed element of B on other summands
(its data image and target value given) may be added. The discretised problem

    max t . v + t_fix  over  v in [lo, hi],  e - g_fix - H^T v in N          (H[c] = G(1_c), t_c = <c, 1_c>)

has the dual phi_h(z) = z . (e - g_fix) + t_fix + sum_c max(hi_c q_c, lo_c q_c) + sigma_N(-W^-1 z), q = t - H z, and
phi_h <= phi (cell fields are a subset of B). phi_h is minimised after smoothing (softplus of width mu, decreasing,
L-BFGS); the smoothed maximiser v_c = lo_c + (hi_c - lo_c) s((hi_c - lo_c) q_c / mu) is moved towards an interior
point v_f of the cell problem (the least violation over [lo, hi], by L-BFGS-B) by the least theta that makes it
feasible (exact for a box, a quadratic for a ball): a witness, whose value t . v + t_fix is a certified lower bound
(CHECKED: the data constraint re-evaluated in floating point).

Statuses (optimisers/dual/reduction.py): EXACT (gap <= TOL max(1, |upper|)), BRACKET (both sides, the gap stated), OUTER (no
witness: the certificate only), EMPTY (phi(z) < 0 at c = 0 with the certified bracket: a Farkas certificate; or an
enclosing set already empty, given by the caller).

From box_feasible.py (merged here by plan 063):
Box BGP on a feasible set (plans 009 P5, 057, 059, 060): the route from K = {m in B : e - G m in N} to its Bound
and its emptiness, for a pointwise box prior B (optimisers/dual/nonsmooth.py: the certified dual, cells and witnesses).

The noise side is a ball, an ellipsoid or a coordinate box (NoiseSide), or slab-type (optimisers/dual/slabs.py: the cells in the
slabs' coordinates, and one slab on L2 of an interval by its exact 1-D dual). Interval cells are kept per edges and
Bounds per direction on K (plan 059): K's data and noise are fixed after construction.
"""


import math

import numpy as np
from numpy.polynomial import legendre as npleg

from ...core.errors import Unsupported
from ...core.precision import TOL
from .bound import CERTIFIED, Bound


class CellProblem:
    """Cell-constant fields of a box (module docstring): bounds lo, hi (ncells,), target integrals t (ncells,), and the
    data images H (ncells, n) as an array or as functions apply(v) = H^T v and adjoint(z) = H z; fixed_data,
    fixed_target: the data image and target of a fixed part (default 0)."""

    def __init__(self, lo, hi, t, H=None, apply=None, adjoint=None, fixed_data=0.0, fixed_target=0.0, note=""):
        self.lo, self.hi, self.t = (np.asarray(v, float).ravel() for v in (lo, hi, t))
        if not (np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi)) and np.all(self.lo <= self.hi)):
            raise Unsupported("cell witnesses need finite bounds lo <= hi on every cell (a bounded box)")
        if H is not None:
            H = np.asarray(H, float)
            apply, adjoint = (lambda v: v @ H), (lambda z: H @ z)
        self.apply, self.adjoint = apply, adjoint
        self.fixed_data, self.fixed_target, self.note = fixed_data, float(fixed_target), note
        self.images = self.key = None                        # interval cells: Op -> images Op(1_c); the cache key

    @property
    def size(self):
        return len(self.t)


class NoiseSide:
    """The noise set in the dual's terms: kind "ball" (whitening Phi, centre, rho) or "box" (lo, hi); built from a set
    by convex/forms.py noise_side."""

    @classmethod
    def ball(cls, phi, centre, rho):
        """The ball ||Phi (eta - centre)|| <= rho with Phi a matrix or the diagonal of one (a vector)."""
        out = cls.__new__(cls)
        out.kind, out.Phi, out.centre, out.rho = "ball", np.asarray(phi, float), np.asarray(centre, float), float(rho)
        return out

    @classmethod
    def box(cls, lo, hi):
        """The coordinate box lo <= eta <= hi."""
        out = cls.__new__(cls)
        out.kind, out.lo, out.hi = "box", lo, hi
        return out

    def _fwd(self, v):
        return self.Phi * v if self.Phi.ndim == 1 else self.Phi @ v

    def _fwdT(self, v):
        return self.Phi * v if self.Phi.ndim == 1 else self.Phi.T @ v

    def sigma(self, z, mu=0.0):
        """sigma_N(-W^-1 z) and its gradient in z (smoothed: mu > 0 bounds the smoothing error, in value units)."""
        if self.kind == "box":
            f, v = _maxlin(-z, self.lo, self.hi, mu)
            return f, -v
        x = z / self.Phi if self.Phi.ndim == 1 else np.linalg.solve(self.Phi.T, z)       # z = Phi^T x
        m = mu / max(self.rho, 1e-300)
        nx = math.sqrt(float(x @ x) + m * m)
        y = self.rho * x / nx if nx > 0 else np.zeros_like(z)
        g = y / self.Phi if self.Phi.ndim == 1 else np.linalg.solve(self.Phi, y)
        return -float(z @ self.centre) + self.rho * nx, -self.centre + g

    def violation(self, r):
        """How far the residual r = e - G m is outside N (0 inside)."""
        if self.kind == "box":
            return float(np.max(np.maximum(r - self.hi, self.lo - r), initial=0.0))
        return max(float(np.linalg.norm(self._fwd(r - self.centre))) - self.rho, 0.0)

    def mix(self, ra, rb):
        """The least theta in [0, 1] with (1 - theta) ra + theta rb in N, given rb in N (None if rb is not)."""
        if self.violation(rb) > 0:
            return None
        if self.violation(ra) == 0:
            return 0.0
        if self.kind == "box":
            th = 0.0
            for bound, s in ((self.hi, 1.0), (self.lo, -1.0)):
                over = s * (ra - bound) > 0
                den = s * (ra - rb)
                if np.any(over):
                    th = max(th, float(np.max(s * (ra - bound)[over] / den[over])))
            return min(th * (1 + 1e-12) + 1e-15, 1.0)
        a, b = self._fwd(ra - self.centre), self._fwd(rb - self.centre)
        d = b - a                                                 # ||a + th d||^2 = rho^2, the smaller root in [0, 1]
        A, B, C = float(d @ d), 2 * float(a @ d), float(a @ a) - self.rho ** 2
        disc = max(B * B - 4 * A * C, 0.0)
        th = (-B - math.sqrt(disc)) / (2 * A) if A > 0 else 1.0
        return min(max(th, 0.0) * (1 + 1e-12) + 1e-15, 1.0)


def _maxlin(q, lo, hi, mu):
    """sum max(hi q, lo q) (mu = 0) or its softplus smoothing lo q + mu log(1 + exp((hi - lo) q / mu)); and the
    gradient (the smoothed maximiser)."""
    if mu <= 0:
        v = np.where(q > 0, hi, lo)
        return float(np.sum(v * q)), v
    s = (hi - lo) * q / mu
    sp = np.where(s > 30, s, np.log1p(np.exp(np.minimum(s, 30))))
    v = lo + (hi - lo) * 0.5 * (1 + np.tanh(0.5 * s))
    return float(np.sum(lo * q) + mu * np.sum(sp)), v


class CellDual:
    """phi_h for a CellProblem, a NoiseSide and the plain residual e (module docstring)."""

    def __init__(self, cells, noise, e):
        self.cells, self.noise = cells, noise
        self.e = np.asarray(e, float) - cells.fixed_data
        self.n = len(self.e)

    def value(self, z, mu=0.0):
        c = self.cells
        q = c.t - c.adjoint(z)
        f, v = _maxlin(q, c.lo, c.hi, mu)
        s, gs = self.noise.sigma(z, mu)
        return float(z @ self.e) + c.fixed_target + f + s, self.e - c.apply(v) + gs, v

    def minimise(self, z0=None, rounds=6, scale=None, maxiter=400):
        """z minimising phi_h by L-BFGS on smoothings of decreasing width (module docstring)."""
        from ..minimise import minimise
        c = self.cells
        z = np.zeros(self.n) if z0 is None else np.asarray(z0, float)
        scale = scale or max(float(np.mean(np.abs(c.t)) * np.mean(c.hi - c.lo)), 1e-300)
        best, fbest = z, self.value(z)[0]
        self.mu = scale
        for k in range(rounds):
            mu = scale * 10.0 ** (-k - 1)
            res = minimise(lambda x: self.value(x, mu)[:2], z, "L-BFGS-B", dict(maxiter=maxiter, ftol=1e-15, gtol=1e-12),
                           jac=True)
            z = res.x
            f = self.value(z)[0]
            if f < fbest:
                best, fbest, self.mu = z, f, mu
        return best, fbest

    def interior(self, maxiter=5000, method="lbfgs"):
        """v_f in [lo, hi] with the least violation of the noise constraint, and that violation: the bound-constrained
        linear least-squares problem min ||S (e - centre - H^T v)|| (S the whitening of a ball, or 1 / half-width of a
        box about its middle), by L-BFGS-B (method="lbfgs", the default) or scipy's lsq_linear (method="lsq":
        trust-region reflective with matrix-free LSMR steps; on the CMB problem's cell sets it was far slower);
        self.interior_info records the iterations and convergence. Any point found is a witness if its violation is 0:
        convergence is not needed for that."""
        from scipy.sparse.linalg import LinearOperator
        from ..minimise import least_squares_box, minimise
        c, N = self.cells, self.noise
        if N.kind == "box":
            mid, half = 0.5 * (N.lo + N.hi), np.maximum(0.5 * (N.hi - N.lo), 1e-300)
            y0, fwd, fwdT = (self.e - mid) / half, (lambda u: u / half), (lambda u: u / half)
        else:
            y0, fwd, fwdT = N._fwd(self.e - N.centre), N._fwd, N._fwdT
        v0 = 0.5 * (c.lo + c.hi)
        if method == "lsq":
            A = LinearOperator((len(y0), c.size), matvec=lambda v: fwd(c.apply(np.ravel(v))),
                               rmatvec=lambda u: c.adjoint(fwdT(np.ravel(u))), dtype=float)
            res = least_squares_box(A, y0, (c.lo, c.hi), maxiter)
            self.interior_info = dict(iterations=int(res.nit), converged=bool(res.status > 0), message=str(res.message))
        else:
            def f(v):
                y = y0 - fwd(c.apply(v))
                return 0.5 * float(y @ y), -c.adjoint(fwdT(y))
            res = minimise(f, v0, "L-BFGS-B", dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-15, gtol=1e-12),
                           jac=True, bounds=list(zip(c.lo, c.hi)))
            self.interior_info = dict(iterations=int(res.nit), converged=bool(res.success), message=str(res.message))
        v = np.clip(res.x, c.lo, c.hi)
        return v, N.violation(self.e - c.apply(v))

    def polish(self, z, mu, max_free=300):
        """A primal candidate from the dual point z: cells where q = t - H z has a clear sign (|q| (hi - lo) >= 20 mu)
        are pinned at the maximising bound; the others (complementary slackness leaves them free) maximise t . v under
        the noise constraint by SLSQP when there are at most max_free of them, else fit the data least-squares."""
        from ..minimise import minimise
        c, N = self.cells, self.noise
        q = c.t - c.adjoint(z)
        v = np.where(q > 0, c.hi, c.lo).astype(float)
        free = np.flatnonzero(np.abs(q) * (c.hi - c.lo) < 20 * mu)
        if not len(free):
            return v
        if len(free) > max_free:
            return _maxlin(q, c.lo, c.hi, mu)[1]
        B = np.zeros((len(free), c.size))
        B[np.arange(len(free)), free] = 1.0
        HF = np.array([c.apply(b) for b in B])                     # (free, n): data images of the free cells
        r0 = self.e - c.apply(v * (np.abs(q) * (c.hi - c.lo) >= 20 * mu))
        if N.kind == "box":
            cons = [dict(type="ineq", fun=lambda u: N.hi - (r0 - u @ HF), jac=lambda u: HF.T),
                    dict(type="ineq", fun=lambda u: (r0 - u @ HF) - N.lo, jac=lambda u: -HF.T)]
        else:
            def g(u):
                y = N._fwd(r0 - u @ HF - N.centre)
                return N.rho ** 2 - float(y @ y)

            def dg(u):                                        # d/du of rho^2 - ||Phi (r0 - HF^T u - centre)||^2
                return 2.0 * HF @ N._fwdT(N._fwd(r0 - u @ HF - N.centre))
            cons = [dict(type="ineq", fun=g, jac=dg)]
        tf = c.t[free]
        res = minimise(lambda u: -float(tf @ u), _maxlin(q, c.lo, c.hi, mu)[1][free], "SLSQP",
                       dict(maxiter=500, ftol=1e-15), jac=lambda u: -tf, bounds=list(zip(c.lo[free], c.hi[free])),
                       constraints=cons)
        v[free] = np.clip(res.x, c.lo[free], c.hi[free])
        return v

    def witness(self, v, vf=None):
        """(lower bound, cell values) of a feasible cell field near v, or (-inf, None)."""
        c, N = self.cells, self.noise
        v = np.clip(v, c.lo, c.hi)
        r = self.e - c.apply(v)
        if N.violation(r) > 0:
            if vf is None:
                return -math.inf, None
            th = N.mix(r, self.e - c.apply(vf))
            if th is None:
                return -math.inf, None
            v = np.clip((1 - th) * v + th * vf, c.lo, c.hi)
            if N.violation(self.e - c.apply(v)) > 0:
                return -math.inf, None
        return float(c.t @ v) + c.fixed_target, v


def certified_phi(z, e, noise, upper_sigma_B):
    """phi(z) with the certified upper end of sigma_B (module docstring): an upper bound for U."""
    return float(z @ e) + upper_sigma_B(z) + noise.sigma(z)[0]


def solve_cells(cells, noise, e, upper_sigma_B, z0=None, interior=None, rounds=10, note=""):
    """The Bound of U (module docstring): z from the cell dual, the certified upper bound phi(z) (and phi(0)), a cell
    witness for the lower bound. upper_sigma_B(z) returns the upper end of sigma_B(c - G* W^-1 z)."""
    dual = CellDual(cells, noise, e)
    z, fh = dual.minimise(z0, rounds=rounds)
    v = dual.polish(z, dual.mu)
    vf, viol = dual.interior() if interior is None else interior
    lower, vw = dual.witness(v, vf if viol == 0 else None)
    q = cells.t - cells.adjoint(z)                            # also the bang-bang field at z (plan 061): kept if better
    bang = dual.witness(np.where(q > 0, cells.hi, cells.lo).astype(float), vf if viol == 0 else None)
    if bang[0] > lower:
        lower, vw = bang
    if lower == -math.inf and viol == 0:
        lower, vw = dual.witness(vf, None)
    upper = certified_phi(z, np.asarray(e, float), noise, upper_sigma_B)
    if np.any(z):
        u0 = certified_phi(np.zeros_like(z), np.asarray(e, float), noise, upper_sigma_B)
        if u0 <= upper:
            z, upper = np.zeros_like(z), u0
    if math.isfinite(lower) and upper - lower <= TOL * max(1.0, abs(upper)):
        status = "EXACT"
    else:
        status = "BRACKET" if math.isfinite(lower) else "OUTER"
    gap = upper - lower
    return Bound(upper if status == "OUTER" else lower, lower, upper, status, z, vw,
                 f"box prior: certified dual (sigma_B upper end) and a {cells.size}-cell witness; gap {gap:.3g}"
                 + (f"; {note}" if note else ""), claim=CERTIFIED)


def emptiness(cells, noise, e, upper_sigma_B0, rounds=6):
    """A Farkas certificate for the empty feasible set: z with phi(z) < 0 at c = 0 (upper_sigma_B0: the certified
    upper end of sigma_B(-G* W^-1 z)), or None. The cell problem's targets must be 0 (direction c = 0)."""
    dual = CellDual(cells, noise, e)
    z, fh = dual.minimise(None, rounds=rounds)
    if fh >= 0:
        return None
    nz = float(np.linalg.norm(z))
    z = z / nz if nz > 0 else z
    val = certified_phi(z, np.asarray(e, float), noise, upper_sigma_B0)
    return (z, val) if val < 0 else None


# ------------------------------------------------------------------------------------------------ feasible sets
def cell_bounds(b, edges, side):
    """Per cell of edges, for a bound b (a float or an element realised piecewise-Legendre on breaks among edges):
    side = +1 a certified lower bound of inf b over the cell (for an upper bound u), side = -1 a certified upper bound
    of sup b (for a lower bound l); and the largest gap to the extremum found (realisations/bounds/extrema.py min_brackets,
    all cells of a piece at once)."""
    from ...realisations.bounds.extrema import min_brackets
    from ...realisations.series.legendre import analysis
    from ...quadrature import gauss
    edges = np.asarray(edges, float)
    m = len(edges) - 1
    if isinstance(b, float):
        return np.full(m, b), 0.0
    R = b.realisation
    mids = 0.5 * (edges[:-1] + edges[1:])
    parent = np.clip(np.searchsorted(R.breaks, mids, side="right") - 1, 0, R.npieces - 1)
    out, gap = np.empty(m), 0.0
    for k in np.unique(parent):                               # the cells of piece k, restricted exactly at once
        idx = np.nonzero(parent == k)[0]
        C = R.coefs[k][:, 0]
        n = len(C)
        t, _ = gauss(n)
        x0, x1 = R.breaks[k], R.breaks[k + 1]
        c0, c1 = edges[idx][:, None], edges[idx + 1][:, None]
        tp = (2.0 * (c0 + 0.5 * (c1 - c0) * (t + 1.0)) - x0 - x1) / (x1 - x0)
        Ck = (npleg.legval(tp, C)) @ analysis(n).T            # (cells, n): each cell's Legendre coefficients
        lo, up = min_brackets(side * Ck)
        out[idx], gap = side * lo, max(gap, float(np.max(up - lo)))
    return out, gap


def _representers(Op, cache, key):
    """The representers r_j of Op's coordinates ((Op m)_j = <r_j, m>: r_j = Op*(W^-1 e_j)) as one realisation."""
    from ...realisations.series.legendre import hstack
    if key not in cache or cache[key][0] is not Op:
        D = Op.codomain
        cache[key] = (Op, hstack([Op.adjoint(D(D.solve_metric(e))).realisation for e in np.eye(D.dim)]))
    return cache[key][1]


def plain_interval(box):
    """Whether the box lives on one L2 of an interval with Lebesgue measure (the exact cell-integral route)."""
    from ...spaces.base import L2
    M = box.space
    return not box.summed and isinstance(M, L2) and type(getattr(M, "measure", None)).__name__ == "LebesgueMeasure"


def interval_cells(box, G, c, sub=4, cache=None):
    """The CellProblem of a box on L2 of intervals (or a direct sum of them): cells between the breakpoints of c, of
    the data representers and of the bounds (their realisations; c's and the representers' only where their pieces
    differ), each piece cut into `sub` equal cells; H[c] = G(1_c) and t_c = <c, 1_c> from the package's exact integrals
    (on one interval with Lebesgue measure as cell integrals of the representers, realisations/piecewise_constant.py). A bound
    is a float or an element of the summand (its realisation is the bound, as for the box's support): on each cell
    lo_c >= sup l and hi_c <= inf u, certified (cell_bounds), so every cell field with values in [lo_c, hi_c] lies in
    the box. cache: a dict owned by one (box, G) pair (a feasible set's): the cells' bounds and data images depend on
    c only through the edges, and are kept by them (plan 059). cp.images(Op): the (cells, dim) images Op(1_c)."""
    from ...spaces.elements import Family
    from ...realisations.piecewise_constant import essential_breaks
    from ...realisations.series.legendre import PiecewiseLegendre, union_breaks
    from ...realisations.registry import require
    cache = {} if cache is None else cache
    M = box.space
    D = G.codomain
    rnd = cache.get("rnd")
    if rnd is None:
        rnd = cache["rnd"] = G.adjoint(D(np.random.default_rng(0).standard_normal(D.dim)))
    spaces = list(zip(M.summands, M.names)) if box.summed else [(M, None)]
    cuts = []
    for k, (H, key) in enumerate(spaces):
        l, u = box.lower[k], box.upper[k]
        if l is None or u is None:
            raise Unsupported("cell witnesses need a bounded box: a finite lower and upper bound on every summand")
        parts = [x if key is None else M.part(x, key) for x in (rnd, c)]
        pls = [p.realisation for p in parts if hasattr(p, "realisation")]
        bls = [b.realisation for b in (l, u) if not isinstance(b, float)]
        for p in pls + bls:
            require(p, "cell_integrals", "generic cell witnesses are built on L2 of intervals; give cells= for other "
                    "spaces")
        nb = union_breaks(*([essential_breaks(p) for p in pls] + [p.breaks for p in bls]))
        cuts.append(np.unique(np.concatenate([np.linspace(a, b, sub + 1) for a, b in zip(nb[:-1], nb[1:])])))
    plain = plain_interval(box)
    ckey = (sub, tuple(e.tobytes() for e in cuts))
    hit = cache.get(ckey)
    if hit is None:
        fams, lo, hi, gap = [], [], [], 0.0
        for k, ((H, key), edges) in enumerate(zip(spaces, cuts)):
            m = len(edges) - 1
            (lk, gl), (uk, gu) = cell_bounds(box.lower[k], edges, -1.0), cell_bounds(box.upper[k], edges, 1.0)
            bad = np.nonzero(lk > uk)[0]
            if len(bad):
                i = bad[0]
                raise Unsupported(f"on the cell [{edges[i]:.6g}, {edges[i + 1]:.6g}]" + ("" if key is None else
                                                                                          f" of {key}")
                                  + " the certified sup of l exceeds the inf of u (the bounds touch or cross within "
                                  "the cell), so no constant there lies in the box: more cells (K.sub), or cells= with "
                                  "a field fixed there")
            if not plain:
                E = np.eye(m)                                 # one identity: its row views share it
                F = Family(H, PiecewiseLegendre(edges, [E[i:i + 1] for i in range(m)], np.zeros(m)), name="1_cells")
                fams.append(F if key is None else M._from_parts({k: F}, "1_cells", family=True))
            lo.append(lk)
            hi.append(uk)
            gap = max(gap, gl, gu)
        Hm = _representers(G, cache, "reps").cell_integrals(cuts[0]) if plain else \
            np.vstack([np.atleast_2d(G(F).values) for F in fams])
        if len(cache) > 256:
            keep = {k: v for k, v in cache.items() if isinstance(k, str)}
            cache.clear()
            cache.update(keep)
        hit = cache[ckey] = (np.concatenate(lo), np.concatenate(hi), fams, gap, Hm)
    lo, hi, fams, gap, Hm = hit
    if plain:
        t = c.realisation.cell_integrals(cuts[0])[:, 0]
    else:
        t = np.concatenate([np.atleast_1d(M.inner(c, F)) for F in fams])
    varying = any(not isinstance(b, float) for b in box.lower + box.upper)
    note = f"{len(t)} interval cells" + (f"; cell bounds certified (chord and curvature bound, Bernstein where loose) "
                                         f"within {gap:.1e} of the bounds' extrema" if varying else "")
    cp = CellProblem(lo, hi, t, H=Hm, note=note)
    cp.key = ckey
    if plain:
        cp.images = lambda Op: _representers(Op, cache, ("reps", id(Op))).cell_integrals(cuts[0])
    else:
        cp.images = lambda Op: np.vstack([np.atleast_2d(Op(F).values) for F in fams])
    return cp


def auto_cells(box, G, c, sub, cache):
    """The automatic cell problem: latitude-longitude cells when every summand is a sphere (optimisers/dual/sphere_cells.py,
    plan 061), else interval cells (optimisers/dual/nonsmooth.py)."""
    from .sphere_cells import is_sphere_box, sphere_cells
    if is_sphere_box(box):
        return sphere_cells(box, G, c, sub, cache)
    return interval_cells(box, G, c, sub, cache)
