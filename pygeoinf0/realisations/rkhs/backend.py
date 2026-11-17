"""Realisations for Sobolev spaces H^s_l(X) defined through their Bessel-potential kernel (restriction spaces on
domains of R^d; periodic images on flat tori and circles).

An element is realised as SobolevRealisation(kx, fx): a kernel part kx (RepresenterCombinations) and a formula part fx (a realisation of the underlying
L2 backend), either of which may be absent:

    RepresenterCombinations      sum_i c_i k_mu_i with k_mu = int K(., y) dmu(y): finite combinations of Riesz representers of measures
            decomposed into atoms (points, segments, patches, spheres: atoms.py); their inner products are the energies
            <k_mu, k_nu> = int int K dmu dnu of the table in pairs.py (exact, or with error estimates for surfaces)
    formula a function given by the user (realised by the L2 backend of the domain for values and integrals)

Inner products: kx with kx (energies), kx with a formula f (int f dmu, the reproducing property, by the quadrature of
each atom's support), formula with formula: exact on tori and circles (the spectral weights on the Fourier realisation);
on intervals for integer orders from the derivatives of the realisation (green.IntervalForm: the form, or the
restriction norm a_s + exterior jet energies); in the form space of order 1 on boxes (box_form.BoxForm, plan 025);
otherwise not available (NormUnavailable: a restriction space needs an extension problem, a fractional spectral space a
fractional power). The kernel is the Matern kernel of the free space, with lattice images (tori, circles), signed
reflection images (spectral Neumann/Dirichlet spaces on intervals and boxes) or a finite-rank boundary correction
(integer orders on intervals): green.py and spaces/constructions.py.
All coordinates are flat: Cartesian on domains of R^d and tori, arc length on paths and circles.
"""

import numpy as np

from ...core.errors import NormUnavailable, Unsupported, unbounded
from ...quadrature import rule
from . import pairs
from .atoms import Atoms, Kind, atoms_from
from ..backends import Backend
from .energy import sign
from ...execution import map_blocks as _map_blocks
from ..registry import energy_engine

FORMULA_NODES = 128      # Gauss nodes per axis of an atom's support for int f dmu (formulas smooth on that scale)
EVAL_POINTS = 2048       # points per block when representers are evaluated (bounds the (points x atoms) array)
EVAL_PAIRS = 1 << 20     # point-atom pairs per block when representers are evaluated


class RepresenterCombinations:
    """N combinations of representers: columns of C (atoms.n, N), or a number c meaning c times the identity (member i
    is c k_mu_i, N = atoms.n), so that a family of n representers stores no n x n matrix; err (N,) estimated
    quadrature error. Products with C go through times (C @ M), rtimes (M @ C) and t_times (C^T @ M)."""

    __slots__ = ("atoms", "C", "_err")

    def __init__(self, atoms, C, err=None):
        self.atoms = atoms
        self.C = float(C) if np.ndim(C) == 0 else np.asarray(C, float)
        self._err = err if callable(err) else np.zeros(self.N) if err is None else np.asarray(err, float)

    @property
    def err(self):
        """The estimated errors, computed on first use when they were given as a callable: creating representers does
        no energy work (plan 022 P1); the self-energies behind the estimate are cached on the atoms."""
        if callable(self._err):
            self._err = np.asarray(self._err(), float)
        return self._err

    @property
    def pending(self):
        return callable(self._err)

    @property
    def identity(self):
        return not isinstance(self.C, np.ndarray)

    @property
    def N(self):
        return self.atoms.n if self.identity else self.C.shape[1]

    @property
    def size(self):
        return 1 if self.identity else self.C.size

    def dense(self):
        return self.C * np.eye(self.atoms.n) if self.identity else self.C

    def times(self, M):
        return self.C * M if self.identity else self.C @ M

    def rtimes(self, M):
        return M * self.C if self.identity else M @ self.C

    def t_times(self, M):
        return self.C * M if self.identity else self.C.T @ M


def _err_of(fn, *ks):
    """fn() now, or fn itself (evaluated on first use) while the errors of any of the RepresenterCombinations ks are pending."""
    return fn if any(k.pending for k in ks) else fn()


class SobolevRealisation:
    """A Sobolev realisation: kernel part kx (RepresenterCombinations or None) plus formula part fx (an L2-backend realisation or None)."""

    __slots__ = ("kx", "fx", "N")

    def __init__(self, kx, fx, N=None):
        self.kx, self.fx = kx, fx
        self.N = N if N is not None else (kx.N if kx is not None else fx.N)

    @property
    def size(self):
        return (self.kx.size if self.kx is not None else 0) + (getattr(self.fx, "size", 0) if self.fx is not None else 0)


class RKHSBackend(Backend):
    """The backend of H^s_l(X): kernel parts via pairs.py, formula parts via the L2 backend `base` of the domain.

    The geometry g (registry.flat_geometry) supplies flat(points), the map from Cartesian points to the flat
    coordinates in which K is the free-space Matern kernel, its inverse, and the periods (None, or the period of each
    flat coordinate, with the cell's origin) that add the lattice images (Poisson summation)."""

    errors_known = False
    last_pairing_error = None        # (N_A, N_B): the two-order estimate of the last formula pairings (EXP-031 O4)

    def __init__(self, base, kernel, domain, g, realisation_name, images=None, correction=None, formula=None,
                 claims=()):
        self.base, self.kernel, self.domain = base, kernel, domain
        self.realisation_name = realisation_name
        self.dim, self.tol = base.dim, base.tol
        self.geometry, self.flat, self.unflat = g, g.flat, g.unflat
        self.periods = None if g.periods is None else np.asarray(g.periods, float)
        self.origin = None if g.periods is None else np.asarray(g.origin, float)
        self.jac_max = getattr(base, "jac_max", 1.0)
        self.formula = formula                       # green.IntervalForm: inner products of formulas (or None)
        self._kernel_claims = list(claims)           # how the kernel is realised, with its accuracy
        self.engine = energy_engine(kernel, self.periods, self.origin, images)    # chosen in registry.py
        if correction is not None:
            from .green import CorrectedEngine
            self.engine = CorrectedEngine(self.engine, *correction)

    @property
    def kernel_claims(self):
        """The claims of the kernel's realisation; for image sums at the current settings.images (green.images_claim)."""
        extra = []
        grid = getattr(self.engine, "grid", None)
        if grid is not None and grid.claim() is not None:
            extra.append(grid.claim())
        if self.engine.images is None:
            return list(self._kernel_claims) + extra
        from .green import images_claim
        return list(self._kernel_claims) + [images_claim(self.engine.images, self.kernel)] + extra

    # ------------------------------------------------------------------ energies (registry.Energies)
    def _shifts(self, extent=0.0):
        return self.engine.shifts(extent)

    def _extent(self, *As):
        return self.engine.extent(*As)

    def energy(self, A, B):
        """<k_mu_i, k_nu_j> = int int K dmu_i dnu_j for the measures of atom sets A and B (cached on A when A is B)."""
        return self.engine.energy(A, B)

    def self_energies(self, A, idx):
        return self.engine.self_energies(A, idx)

    def self_energies_all(self, A):
        return self.engine.self_energies_all(A)

    def _pair_formula(self, A, fx):
        """(A.n, fx.N): int f dmu_i for the formula part: values at points, and the quadrature of every other atom's
        support (its Domain.quadrature with FORMULA_NODES nodes per axis)."""
        X, W, own = [], [], []
        Xl, Wl, ol = [], [], []
        for q in range(len(A.P)):
            if A.kind[q] == Kind.WEIGHTED:                # a weighted piece: its own rule with the weight folded in,
                x, w = A.piece[q].formula_nodes()         # and a lower order for the estimate (O4)
                X.append(x)
                W.append(w * A.weight[q])
                own.append(np.full(len(w), A.owner[q]))
                x, w = A.piece[q].formula_low()
                Xl.append(x)
                Wl.append(w * A.weight[q])
                ol.append(np.full(len(w), A.owner[q]))
                continue
            S = A.support(q)
            x, w = (A.P[q][None], np.ones(1)) if S is None else rule(S, FORMULA_NODES)
            X.append(x)
            W.append(w * A.weight[q])
            own.append(np.full(len(w), A.owner[q]))
        v = self._eval_base(fx, np.vstack(X)) * np.concatenate(W)[:, None]
        out = np.zeros((A.n, fx.N))
        np.add.at(out, np.concatenate(own), v)
        self.last_pairing_error = None
        if Xl:                                             # weighted pieces: the two orders' difference
            hi, lo = np.zeros((A.n, fx.N)), np.zeros((A.n, fx.N))
            np.add.at(lo, np.concatenate(ol), self._eval_base(fx, np.vstack(Xl)) * np.concatenate(Wl)[:, None])
            wq = [k for k in range(len(A.P)) if A.kind[k] == Kind.WEIGHTED]
            Xh = np.vstack([X[k] for k in wq])
            np.add.at(hi, np.concatenate([own[k] for k in wq]), self._eval_base(fx, Xh) * np.concatenate(
                [W[k] for k in wq])[:, None])
            self.last_pairing_error = np.abs(hi - lo)
        return out

    def _eval_base(self, fx, Xflat):
        X = self.unflat(Xflat)
        return self.base.evaluate(fx, *[X[:, k] for k in range(X.shape[1])]).reshape(len(Xflat), fx.N)

    # ------------------------------------------------------------------ required
    def fit(self, fun, N=1, breaks=None):
        return self.from_base(*self.base.fit(fun, N=N, breaks=breaks))

    def from_base(self, fx, u):                       # a fit of a formula by self.base (also core/fits.py)
        self.check_jumps(fx)                          # no jump where elements cannot have one (plan 065 A1)
        if self.formula is not None:
            self.formula.check(fx)                    # the boundary conditions every element satisfies
        return SobolevRealisation(None, fx), u

    def jumps_checked(self, fx):
        """The derivative orders m (m + 1/2 <= s) at whose breaks a formula's realisation fx is checked for jumps
        (sup.check_sobolev): every one when fx is piecewise Legendre in one flat coordinate (intervals, paths); None
        when this realisation has no such check (tensor, local and spectral realisations of 2-D and 3-D domains)."""
        from ..series.legendre import PiecewiseLegendre
        from ..series.sup import sobolev_orders
        return tuple(sobolev_orders(self.kernel.s)) if isinstance(fx, PiecewiseLegendre) else None

    def check_jumps(self, fx):
        orders = self.jumps_checked(fx)
        if orders:
            from ..series.sup import check_sobolev
            check_sobolev(fx, self.kernel.s, self.space_label())

    def space_label(self):
        return f"H^{self.kernel.s:g}({self.domain!r})"

    def fit_family(self, fun_idx, N, breaks=None, sparse=None, name="F"):
        return self.from_base(self.base.fit_family(fun_idx, N, breaks=breaks, sparse=False, name=name), 0)[0]

    def riesz(self, measures, zero=None):
        """Representers of the measures; zero[i] marks the zero functional (a measure on a Dirichlet part): its atoms
        get weight 0, so the representer and its energies are exactly 0."""
        atoms = atoms_from(measures, self.domain, self.flat, self.periods, self.origin, self.unflat)
        if zero is not None:
            atoms.weight[np.isin(atoms.owner, np.flatnonzero(zero))] = 0.0
        own = np.unique(atoms.owner[atoms.kind >= Kind.PATCH])
        if not len(own):
            return SobolevRealisation(RepresenterCombinations(atoms, 1.0, np.zeros(atoms.n)), None)

        def err():                    # surface measures: the error of ||k_mu|| from the quadrature error of its energy
            e = np.zeros(atoms.n)
            E, x = self.self_energies(atoms, own)
            e[own] = x / (2 * np.sqrt(E))
            return e
        return SobolevRealisation(RepresenterCombinations(atoms, 1.0, err), None)                       # on first use only (plan 022 P1)

    def lincomb(self, terms):
        kx = [(a, R.kx) for a, R in terms if R.kx is not None]
        fx = [(a, R.fx) for a, R in terms if R.fx is not None]
        N = terms[0][1].N
        return SobolevRealisation(self._kx_lincomb(kx) if kx else None, self.base.lincomb(fx) if fx else None, N)

    def _kx_lincomb(self, kx):
        atoms = [K.atoms for _, K in kx]
        err = _err_of(lambda: sum(abs(a) * K.err for a, K in kx), *[K for _, K in kx])
        if all(a is atoms[0] for a in atoms):
            same = all(K.identity for _, K in kx)
            return RepresenterCombinations(atoms[0], sum(a * (K.C if same else K.dense()) for a, K in kx), err)
        uniq = []
        for a in atoms:
            if not any(a is u for u in uniq):
                uniq.append(a)
        offs = {id(u): o for u, o in zip(uniq, np.cumsum([0] + [u.n for u in uniq]))}
        allat = Atoms.concat(uniq)
        C = np.zeros((allat.n, kx[0][1].N))
        for a, K in kx:
            C[offs[id(K.atoms)]:offs[id(K.atoms)] + K.atoms.n] += a * K.dense()
        return RepresenterCombinations(allat, C, err)

    def combine(self, R, A):
        A = np.asarray(A, float)
        kx = None if R.kx is None else RepresenterCombinations(R.kx.atoms, R.kx.times(A), _err_of(lambda: np.abs(A).T @ R.kx.err, R.kx))
        fx = None if R.fx is None else self.base.combine(R.fx, A)
        return SobolevRealisation(kx, fx, A.shape[1])

    def take(self, R, idx):
        sel = np.atleast_1d(idx)
        kx = None
        err = None if R.kx is None else _err_of(lambda: R.kx.err[sel], R.kx)
        if R.kx is not None and R.kx.identity and len(np.unique(sel)) == len(sel):
            kx = RepresenterCombinations(R.kx.atoms.subset(sel), R.kx.C, err)      # members of a family of representers
        elif R.kx is not None:
            C = R.kx.dense()[:, sel]
            used = np.flatnonzero(np.any(C != 0, axis=1))
            if len(used) < R.kx.atoms.n // 2:               # a few members of a large family: only the measures they use
                kx = RepresenterCombinations(R.kx.atoms.subset(used), C[used], err)
            else:
                kx = RepresenterCombinations(R.kx.atoms, C, err)
        fx = None if R.fx is None else self.base.take(R.fx, sel)
        return SobolevRealisation(kx, fx, len(sel))

    def hstack(self, Rs):
        if all(R.fx is None for R in Rs):
            parts = [R.kx for R in Rs]
            err = _err_of(lambda: np.concatenate([p.err for p in parts]), *parts)
            if all(p.atoms is parts[0].atoms for p in parts):
                return SobolevRealisation(RepresenterCombinations(parts[0].atoms, np.hstack([p.dense() for p in parts]), err), None)
            if all(p.identity and p.C == parts[0].C for p in parts) and \
                    len({id(p.atoms) for p in parts}) == len(parts):       # families of representers: still identity
                return SobolevRealisation(RepresenterCombinations(Atoms.concat([p.atoms for p in parts]), parts[0].C, err), None)
            allat = Atoms.concat([p.atoms for p in parts])     # block-diagonal coefficients over the union of atoms
            C = np.zeros((allat.n, sum(p.N for p in parts)))
            r = c = 0
            for p in parts:
                C[r:r + p.atoms.n, c:c + p.N] = p.dense()
                r, c = r + p.atoms.n, c + p.N
            return SobolevRealisation(RepresenterCombinations(allat, C, err), None)
        if all(R.kx is None for R in Rs):
            return SobolevRealisation(None, self.base.hstack([R.fx for R in Rs]))
        raise Unsupported("stacking representers and formulas in one family")

    def gram(self, A, B):
        if A.kx is None and B.kx is None:                   # formulas only: as the form gives it (sparse for bases)
            return self.formula_gram(A.fx, B.fx)
        out = np.zeros((A.N, B.N))
        if A.kx is not None and B.kx is not None:
            if A.kx.atoms is B.kx.atoms and not (A.kx.identity and B.kx.identity):
                if B.kx.identity:                               # (C_A^T E)^T = E C_A, E symmetric
                    out += B.kx.rtimes(self.engine.apply(A.kx.atoms, A.kx.C).T)
                else:                                           # C_A^T (E C_B): E applied, not necessarily formed
                    out += A.kx.t_times(self.engine.apply(A.kx.atoms, B.kx.C))
            else:
                out += B.kx.rtimes(A.kx.t_times(self.energy(A.kx.atoms, B.kx.atoms)))
        if A.kx is not None and B.fx is not None:
            out += A.kx.t_times(self._pair_formula(A.kx.atoms, B.fx))
            if self.last_pairing_error is not None:              # |C|^T |error|: the estimate of the combinations
                e = self.last_pairing_error
                self.last_pairing_error = abs(A.kx.C) * e if A.kx.identity else np.abs(A.kx.C).T @ e
        if A.fx is not None and B.kx is not None:
            out += B.kx.t_times(self._pair_formula(B.kx.atoms, A.fx)).T
        if A.fx is not None and B.fx is not None:
            out += (lambda G: G.toarray() if hasattr(G, "toarray") else G)(self.formula_gram(A.fx, B.fx))
        return out

    def formula_gram(self, F, G):
        if self.periods is not None:                      # tori and circles: the spectral definition, exact on
            return self._spectral_gram(F, G)              # Fourier realisations
        if self.formula is not None:                      # intervals, integer orders: the form from derivatives
            return self.formula.gram(F, G)
        raise NormUnavailable(
            "the inner product of two functions given by formulas is not computed in this space: in a restriction space "
            "it is the minimal-extension (quotient) norm and needs an extension problem; in a fractional spectral space a "
            "fractional power of the operator. Exact: inner products with Riesz representers (int f dmu), and everything "
            "in their span. Compare functions in L2 with pg.Inclusion(H, L2); the classic H^1 (FEM) inner product int uv "
            "+ l^2 grad u.grad v is the form space's, pg.SobolevForm(X, 1, length_scale=l).space() (intervals, boxes).")

    def _spectral_gram(self, F, G):
        """sum (1 + l^2 lambda_k)^s f_k g_k for Fourier realisations, lambda_k = sum_j (2 pi k_j / period_j)^2
        (periods in flat coordinates): the weight multiplies G's coefficients, the L2 backend does the rest."""
        from ..series.tensor import TensorSeries, FourierBasis, common_bases, to_bases
        bases = common_bases(F, G)
        if not all(isinstance(b, FourierBasis) for b in bases):
            raise NormUnavailable("exact spectral norms need Fourier realisations (realisation='tensor')")
        lam = 0.0
        for k, (b, p) in enumerate(zip(bases, self.periods)):
            freq = (np.arange(b.n) + 1) // 2                 # 1, cos 1, sin 1, cos 2, ...
            shape = [1] * len(bases)
            shape[k] = b.n
            lam = lam + ((2 * np.pi * freq / p) ** 2).reshape(shape)
        W = (1.0 + self.kernel.l ** 2 * lam) ** self.kernel.s
        CG = to_bases(G, bases)
        return self.base.gram(F, TensorSeries(bases, CG * W[..., None], G.err))

    def sqnorms(self, A):
        if A.fx is not None:
            if A.kx is None:
                return np.asarray(self.formula_gram(A.fx, A.fx).diagonal(), float).copy()   # dense or sparse
            return np.diag(self.gram(A, A)).copy()
        if A.kx.identity:                                   # c^2 <k_mu_i, k_mu_i>: the diagonal energies only
            E = A.kx.atoms.self_energy
            d = np.diag(E[0]).copy() if E is not None else self.self_energies_all(A.kx.atoms)
            return A.kx.C ** 2 * d
        return np.einsum("ij,ij->j", A.kx.C, self.engine.apply(A.kx.atoms, A.kx.C))

    def evaluate(self, R, *X):
        shape = np.broadcast(*X).shape
        pts = np.stack([np.broadcast_to(np.asarray(a, float), shape).ravel() for a in X], axis=1)
        out = np.zeros((len(pts), R.N))
        if R.fx is not None:
            out += self.base.evaluate(R.fx, *[pts[:, k] for k in range(pts.shape[1])]).reshape(len(pts), R.N)
        if R.kx is not None:
            inside = np.atleast_1d(self.domain.contains(pts))
            F = self.flat(pts[inside])
            step = max(1, min(EVAL_POINTS, EVAL_PAIRS // max(len(R.kx.atoms.P), 1)))
            tasks = [(s, min(s + step, len(F))) for s in range(0, len(F), step)]
            vals = np.zeros((len(F), R.N))
            for (a, b), v in _map_blocks(self._block_values, (F, R.kx), tasks):
                vals[a:b] = v
            out[inside] += vals
            out[~inside] = np.nan
        return out.reshape(shape + (R.N,))

    def _block_values(self, F, kx, a, b):
        return kx.rtimes(self.kernel_values(F[a:b], kx.atoms))

    def kernel_values(self, F, A):
        """(len(F), A.n): k_mu_i at flat points F (the POINT rows of the table in pairs.py, with lattice images)."""
        V = np.zeros((len(F), len(A.P)))
        for shift in self._shifts(self._extent(A)):
            V += sign(shift) * pairs.values(self.kernel, F, A, shift)
        out = np.zeros((A.n, len(F)))
        np.add.at(out, A.owner, (V * A.weight[None]).T)
        corr = getattr(self.engine, "correction_values", None)
        return out.T if corr is None else out.T + corr(F, A)

    def diagonal_max(self):
        """(sup_x K(x, x), exact?): k(0) for the stationary kernels (restriction, tori, circles); with a boundary
        correction or images the maximum of K(x, x) on a grid (4097 points on an interval, ends included)."""
        if getattr(self.engine, "correction_values", None) is None and self.engine.images is None:
            return self.kernel.k0, True
        from ...measures import DiracMeasure
        box = self.domain.bounding_box
        m = {1: 4097, 2: 65}.get(len(box), 17)
        grids = np.meshgrid(*[np.linspace(lo, hi, m) for lo, hi in box], indexing="ij")
        X = np.stack([g.ravel() for g in grids], axis=1)
        X = X[np.atleast_1d(self.domain.contains(X))]
        A = atoms_from([DiracMeasure(x if len(x) > 1 else x[0]) for x in X], self.domain, self.flat)
        return float(np.max(self.self_energies_all(A))), False

    def breaks(self, R):
        """Where the function may have kinks, on domains with one Cartesian coordinate (intervals): the points and
        segment ends of its atoms, with the breaks of a formula part; None elsewhere."""
        if self.domain.cartesian_dim != 1:
            return None
        parts = [] if R.fx is None else [np.asarray(self.base.breaks(R.fx), float)]
        if R.kx is not None:
            A = R.kx.atoms
            ends = [np.atleast_2d(e) for p in A.piece if p is not None for e in getattr(p, "ends", ())]
            flat = np.concatenate([A.P, (A.P + A.E1)[A.kind == Kind.SEGMENT]] + ends)
            parts.append(np.asarray(self.unflat(flat), float)[:, 0])
        return np.unique(np.concatenate(parts)) if parts else None

    def to_l2(self, R):
        return R.fx if R.kx is None else None

    def errors_pending(self, R):
        """Whether the estimated errors of R are still to be computed (plan 022 P1)."""
        return R.kx is not None and R.kx.pending

    def exact_zeros(self, R):
        """Members that are exactly 0 by construction, read off the atoms without energies: a family of representers
        whose measure has no atom of nonzero weight (a measure on a Dirichlet part, riesz(zero=)). None when it cannot
        be read off (formulas, combinations, negative weights that might cancel): the caller then uses norms."""
        if R.fx is not None or R.kx is None or not R.kx.identity:
            return None
        A = R.kx.atoms
        if np.any(A.weight < 0):
            return None
        if R.kx.C == 0:
            return np.ones(A.n, bool)
        live = np.zeros(A.n, bool)
        live[np.unique(A.owner[A.weight != 0])] = True
        return ~live

    def error(self, R):
        e = np.zeros(R.N) if R.kx is None else R.kx.err.copy()       # exact energies: rounding only
        if R.fx is not None:
            e = e + np.nan                                   # the H^s error of a formula's realisation is unknown
        return e

    def describe(self, R):
        d = dict(kind="Sobolev", size=R.size)
        if R.kx is not None:
            d["representers"] = f"{R.kx.atoms.n} measures, {len(R.kx.atoms.P)} atoms"
        if R.fx is not None:
            d["formula"] = self.base.summary(R.fx)
        return d

    def summary(self, R):
        parts = []
        if R.kx is not None:
            parts.append(f"combination of {R.kx.atoms.n} Riesz representers")
        if R.fx is not None:
            parts.append(f"formula ({self.base.summary(R.fx)})")
        return " + ".join(parts)

    # ------------------------------------------------------------------ derived/optional
    def colsum(self, R):
        return self.combine(R, np.ones((R.N, 1)))

    def constant(self, values):
        return SobolevRealisation(None, self.base.constant(values))

    def coordinate(self, k):
        return SobolevRealisation(None, self.base.coordinate(k))

    def probe(self):
        return self.base.probe()

    def integral(self, R, over=None):
        """int f dmu over a set or measure `over` inside the domain (a ray, a patch, a sphere, a circle, a point, a
        subinterval): the pairing <f, k_mu> with its representer (the reproducing property), exact like every energy,
        and bounded only when the trace theorem says so. Over the whole domain: exact for formulas; for representers,
        exact (an energy with the domain's own measure) when that measure is an atom (intervals and paths), else a
        48-point-per-axis tensor Gauss rule on the domain, accurate to about 1e-6 relative (representers have kinks
        across their supports)."""
        from ...measures import LebesgueMeasure, as_measure, measure_dim
        if over is not None and over != self.domain:
            mu = as_measure(over)
            k = measure_dim(mu)
            from ...spaces.dual import trace_excess        # the one trace rule
            if not trace_excess(self.kernel.s, self.kernel.d, k) > 0:
                raise unbounded(mu, k, f"H^{self.kernel.s:g}({self.domain!r})",
                                f"it needs s > (d - k)/2 = {(self.kernel.d - k) / 2:g} (trace theorem)")
            if R.kx is None and k == self.kernel.d:           # a formula over a part of full dimension: exact
                return self.base.integral(R.fx, over)
            return self.gram(R, self.riesz([mu]))[:, 0]
        if R.kx is None:
            return self.base.integral(R.fx, None)
        if R.fx is None:
            try:
                whole = atoms_from([LebesgueMeasure(self.domain)], self.domain, self.flat, self.periods, self.origin,
                                   self.unflat)
                return R.kx.t_times(self.energy(R.kx.atoms, whole))[:, 0]
            except Unsupported:
                pass
        X, w = rule(self.domain, 48)
        return w @ self.evaluate(R, *[X[:, k] for k in range(X.shape[1])]).reshape(len(w), R.N)
