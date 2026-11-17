"""The spectral energy engine (plan 005 P4, the author's decision 5): products E V with the energies
E_ij = int int K dmu_i dmu_j of segment measures (rays, polylines with densities) for the matrix-free route above
settings.dense_max, when settings.engine = "spectral", or "auto" (the default) for segments in 3-D with finufft
installed (registry.Energies routes the products here).

From EXP-025 (session pygeoinf0-c8, "spec+C"; package pin 92fa5a8, module commit 229a59b, whose
proposed/spectral_engine.py this module started from):

    E V  =  E_T V + C V  (+ the omitted rest)

    E_T  the energies of K_T = F^-1[Khat T], T(|xi|) an erfc taper to Xi = l Xi / l (spectral_terms.taper), on the
         lattice xi_k = 2 pi k_j / P_j:  (E_T)_ij = (P_1...P_d)^-1 sum_k Khat T(xi_k) muhat_i(xi_k) conj muhat_j(xi_k),
         muhat(xi) = int e^{-i xi.y} dmu(y) from Gauss nodes on the segments; applied as one NUFFT type 1 (nodes to
         modes), a diagonal multiplication and one NUFFT type 2. On a domain of R^d, P_j = extent_j + the distance
         beyond which K < reach_tol of its peak (Poisson: E_T are the energies of K_T periodised, whose images lie beyond
         that reach); on a flat torus P_j are its periods, and the lattice sum is the definition (no aliasing).
    C    exact corrections E_K - E_{K_T} on every pair of segments closer than r_corr = 0.25 l: the segment itself, its
         neighbours along the polyline, close segments of other measures (and, on tori, of their lattice images);
         a sparse n x n matrix, computed once per atom set and resolution (spectral_terms.py).

Why those pairs. The omitted kernel K_H = F^-1[Khat (1 - T)] of a segment with itself, or with a nearly collinear
segment, decays like the self tail (l Xi)^(d-1-2s) (a segment's transform does not decay in the d - 1 directions
normal to it; EXP-025 §1); for two segments at an angle their transforms decay except on the (d-2)-dimensional
subspace normal to both, so the term decays like (l Xi)^(d-2-2s) / sin(angle). The collinear pairs are exactly the
close ones (adjacent pieces of a ray, rays leaving one source), so after the corrections products converge like
(l Xi)^-p with p = 2s + 2 - d (rate(): 3 on L3, s = 2, d = 3, as measured there; 3 on L2, s = 3/2, d = 2).

Claims (claims()), in the sense of EXP-025 §5:
  DERIVED  Khat (1 - T) >= 0, so E - E_T is the energy matrix of a positive-definite kernel: v^T E_T v <= v^T E v, and
           |(E - E_T)_ij| <= sqrt((E - E_T)_ii (E - E_T)_jj); the corrected pairs carry the exact energies of K - K_T
           (K_T tabulated to ~2e-8); on a domain of R^d the aliasing is that of K_T beyond reach_tol (not estimated
           below, because both resolutions share the periods); on a torus there is none.
  CHECKED  the relative error of products: the two-resolution estimate ||E_hi v - E_lo v|| / ((r^p - 1) ||E_hi v||)
           with r = Xi_hi / Xi_lo = sqrt 2 on probe vectors (a constant and a Gaussian one): an estimate, not a bound
           (EXP-025: 1.3-2.4 times the actual error on L3 and the fan; tests/test_spectral.py checks it on L2 and L3).

Choice of parameters (settings.accuracy, plan 005 decision 3: predicted-data errors at most accuracy noise std, in rms
over the data as the CG rule of solvers.recommend reads it):
  target    relative product error tol = accuracy * NOISE_REF; predicted data err by about tol * rms(d) in rms, i.e. by
            at most accuracy noise std when the noise is at least NOISE_REF = 5 % of the rms data (as on the ladder).
            The largest error over the data is 2-3 times the rms (L3, n = 60: 0.0146 sigma where the rms meets 0.01)
  l Xi      a priori from the measured L3 law, max data error 0.004 sigma at l Xi = 10 falling like (l Xi)^-p
            (0.0005-0.0009 sigma at 20): l Xi = 10 (0.004 / accuracy)^(1/p), at least 6. Beyond L3 the law is only a
            first guess: the estimate is checked after building, and l Xi raised by sqrt 2 until it meets tol (refused
            above 64)
  eps       NUFFT tolerance tol / 50; reach_tol (the period) tol / 500. At the default accuracy these are 1e-5 and 1e-6,
            the values EXP-025 measured to change no digit of the products (against 1e-7 and 1e-10, at 0.57 of the
            modes); tied to tol so that rounding of the transforms and aliasing, which the estimate does not see, stay
            50 times below it
  r_corr    0.25 l (EXP-025 §4: the shared endpoints of the fan need it; 1 l gains only a factor 2 at l Xi = 10)
  pairs     at fixed l and domain the pairs closer than r_corr grow like n^2: rays crossing within r_corr, as the near
            field of the H-matrix (EXP-025 at 1e5 rays of 8 segments: ~1.25e8 pairs, the search of 2.7e9 sample pairs
            exhausting europa; 1.5e6 at 1e4). The search streams its blocks and refuses (Unsupported) once the pairs
            need more than settings.memory_gb at ST.PAIR_BYTES each. settings.collinear = c (None by default):
            correct pairs of different measures only when they meet (distance 0: shared endpoints and crossings;
            without them the estimate fell below the actual error on the fan at n = 60, 1.69e-4 against 1.44e-4) or
            are at sin(angle) < c, the ones whose omitted term converges slowly by the argument above; the others
            converge at the rate p. Scale validation (europa, 2026-10-26, c = 0.3 against None, L3 and fan at 3e3 and
            1e4, L2 at 2e3): every run within the target, the estimate 1.5-2.1 times the actual error, measured rates
            3.0-3.8; L3 at 1e4: 2.05e5 pairs instead of 1.49e6, error 2.6e-4 instead of 2.1e-4; L3 at 1e5 rays of 27
            segments: 2.3e7 pairs, 808 s, 15.7 GB (results/benchmarks/validate_spectral_2026-10-26)

Scope (unsupported()): segment atoms and weighted pieces (EXP-031 W4: far field only, weighted_spectral.py) in flat
coordinates of dimension 2 or 3, on domains of R^d and flat tori. Points, patches, spheres, circles and 1-D
coordinates are not handled; the registry falls back to the H-matrix or exact energies and says so. Reflection images
(boxes with Neumann/Dirichlet faces) are served on a torus of signed copies by spectral_images.py (plan 006).

finufft (Apache-2.0) is the optional dependency (pip install 'pygeoinf0[fast]'), imported only in _finufft(). It runs
in the parent process with settings.workers threads: forked workers (the correction blocks) never transform, since
finufft bundles libgomp and OpenMP in a child of a process that used it can hang. Without finufft the transforms are
direct dense sums (exact to rounding, O(nodes x modes): tests and tiny problems; larger ones are refused with the
install hint), and the claims say so.
"""
import math
import time

import numpy as np

from ...core.claims import CHECKED, DERIVED, Claim, Claims
from ...core.errors import Unsupported
from . import spectral_terms as ST
from .atoms import Kind
from ...execution import in_worker
from .spectral_terms import far_reach
from .weighted_spectral import observed_q, weighted_corrections

NOISE_REF = 0.05         # noise std / rms data at which a relative product error tol is accuracy noise std
LAW = (10.0, 0.004)      # L3 (EXP-025 §3): predicted-data error 0.004 noise std at l Xi = 10
LXI_MIN, LXI_MAX = 6.0, 64.0
RCORR = 0.25             # r_corr / l
DENSE_WORK = 2e8         # nodes x modes of the direct transforms (without finufft) above which the engine refuses
MEM_BYTES = 4e8          # memory of the transforms of one batch of columns
SQ2 = math.sqrt(2.0)
INSTALL = "pip install finufft (or pip install 'pygeoinf0[fast]')"


def rate(kernel):
    """p = 2s + 2 - d: the decay (l Xi)^-p of corrected products (module docstring)."""
    return 2 * kernel.s + 2 - kernel.d


def target(accuracy):
    return accuracy * NOISE_REF


def initial_lxi(accuracy, kernel):
    return max(LXI_MIN, LAW[0] * (LAW[1] / accuracy) ** (1.0 / rate(kernel)))


def unsupported(exact, A):
    """Why the spectral engine does not compute the energies of the atom set A (None when it does)."""
    k = exact.kernel
    if exact.images is not None:
        return "the kernel has reflection images (Neumann or Dirichlet ends or faces)"
    if k.d not in (2, 3):
        return f"{k.d}-dimensional flat coordinates (intervals, paths, circles)"
    other = sorted({Kind(int(x)).name.lower() for x in np.unique(A.kind)} - {"segment", "weighted"})
    if other:
        return f"atoms other than segments and weighted pieces ({', '.join(other)})"
    if exact.periods is not None and float(np.max(np.linalg.norm(A.E1, axis=1))) + RCORR * k.l >= min(exact.periods):
        return "segments as long as the period"
    return None


def unsupported_auto(exact):
    """Why settings.engine = "auto" does not choose the spectral engine for a kernel it applies to (None when it
    does): in 2-D the corrections of crossing chords made setups slower than the H-matrix (L2, 2e3 rays: 265 s);
    without finufft only tiny problems run; and a settings.accuracy for which the a-priori law asks for l Xi above
    LXI_MAX would be refused by fit()."""
    from ...settings import settings
    if exact.kernel.d != 3:
        return f"{exact.kernel.d}-D: the H-matrix is faster there"
    if _finufft() is None:
        return f"finufft is not installed ({INSTALL})"
    lxi = initial_lxi(settings.accuracy, exact.kernel)
    if lxi > LXI_MAX:
        return (f"settings.accuracy = {settings.accuracy:g} asks for l Xi = {lxi:.3g} > {LXI_MAX:g} of the spectral "
                "engine")
    return None


# ------------------------------------------------------------------------------------------ the transforms (adapter)

def _finufft():
    """The optional dependency, imported here only: None when it is not installed."""
    try:
        import finufft
    except ImportError:
        return None
    return finufft


class _Transforms:
    """f_k = sum_q c_q e^{+i k.x_q} (type 1) and g_q = sum_k f_k e^{-i k.x_q} (type 2) on M_1 x ... x M_d modes in FFT
    order for nodes x in [-pi, pi)^d: finufft with nthreads threads in this process, or direct dense sums."""

    def __init__(self, X, M, eps, nthreads):
        self._parent_only()
        self.X, self.M, self.eps, self.nthreads = X, tuple(int(m) for m in M), eps, max(1, int(nthreads))
        self.lib, self.plans = _finufft(), {}
        if self.lib is None:
            modes = int(np.prod(self.M))
            if len(X) * modes > DENSE_WORK:
                raise Unsupported(f"the spectral engine needs finufft for {len(X)} nodes x {modes} modes: {INSTALL}")
            self.F = [np.exp(1j * X[:, j:j + 1] * np.fft.fftfreq(m, 1.0 / m)[None]) for j, m in enumerate(self.M)]
            ax = "abc"[:len(self.M)]
            self.e1 = "kq," + ",".join(f"q{a}" for a in ax) + "->k" + ax
            self.e2 = "k" + ax + "," + ",".join(f"q{a}" for a in ax) + "->kq"

    @property
    def route(self):
        return f"finufft, {self.nthreads} thread(s)" if self.lib is not None else "direct sums (finufft not installed)"

    @staticmethod
    def _parent_only():
        if in_worker():
            raise Unsupported("spectral transforms run in the parent process only, never in a forked worker (finufft "
                              "bundles libgomp, and OpenMP in a child of a process that used it can hang)")

    def type1(self, c):                                              # (k, N) complex -> (k, modes)
        if self.lib is None:
            return np.einsum(self.e1, c, *self.F, optimize=True).reshape(len(c), -1)
        f = self._plan(1, len(c)).execute(np.ascontiguousarray(c if len(c) > 1 else c[0]))
        return f.reshape(len(c), -1)

    def type2(self, f):                                              # (k, modes) -> (k, N)
        k = len(f)
        if self.lib is None:
            return np.einsum(self.e2, f.reshape((k,) + self.M), *[F.conj() for F in self.F], optimize=True)
        g = self._plan(2, k).execute(np.ascontiguousarray(f.reshape((k,) + self.M if k > 1 else self.M)))
        return g.reshape(k, -1)

    def _plan(self, kind, k):
        self._parent_only()
        if (kind, k) not in self.plans:
            if len(self.plans) >= 4:
                self.plans = {key: p for key, p in self.plans.items() if key[1] == 1}
            p = self.lib.Plan(kind, self.M, n_trans=k, eps=self.eps, isign=1 if kind == 1 else -1, modeord=1,
                              upsampfac=1.25 if self.eps >= 1e-8 else 2.0, nthreads=self.nthreads)
            p.setpts(*[np.ascontiguousarray(self.X[:, j]) for j in range(self.X.shape[1])])
            self.plans[(kind, k)] = p
        return self.plans[(kind, k)]


# ------------------------------------------------------------------------------------------ the plan of an atom set

class _Level:
    """One resolution: l Xi, Xi, the modes M_j, the far field (nodes, transforms, lattice weights) and corrections C."""

    def __init__(self, lxi, Xi, M):
        self.lxi, self.Xi, self.M = lxi, Xi, M
        self.far = self.C = None


class Plan:
    """What the engine keeps for one atom set A, stored as A.spectral: the lattice (periods, centre, the shifts of the
    corrections on tori), the corrected pairs and their exact energies (shared by every resolution), the resolution in
    use with its estimate, and timings and counts (stats). key: kernel, periods, (r_corr, collinear), the caller's
    l Xi, eps and reach_tol, and the target; a plan with another key is rebuilt (reusing the pairs when only resolution
    parameters changed)."""

    def __init__(self, engine, A, key, tol, reuse=None):
        from ...settings import settings
        k = self.kernel = engine.kernel
        self.A, self.key, self.tol, self.accuracy = A, key, tol, settings.accuracy
        self.eps = engine.eps if engine.eps is not None else max(1e-14, tol / 50)
        self.reach_tol = engine.reach_tol if engine.reach_tol is not None else tol / 500
        self.rc, self.smax = engine.rcorr * k.l, key[2][1]                # key[2] = (r_corr / l, collinear)
        self.nprobes = engine.probes                                         # Gaussian probes besides the constant
        from .weighted_spectral import split               # segments (corrected) and weighted pieces (EXP-031 W4)
        self.sidx, self.widx, self.p, box = split(A, k, rate(k))
        self.sowner, self.sweight = A.owner[self.sidx], A.weight[self.sidx]
        self.P, self.Qe = A.P[self.sidx], A.P[self.sidx] + A.E1[self.sidx]
        L = np.linalg.norm(A.E1[self.sidx], axis=1)
        self.stats = dict(measures=A.n, segments=len(L), weighted=len(self.widx))
        if engine.periods is not None:
            self.periods = np.asarray(engine.periods, float)
            self.centre = np.asarray(engine.origin, float) + 0.5 * self.periods
            S = np.stack(np.meshgrid(*[np.arange(-1, 2)] * k.d, indexing="ij"), -1).reshape(-1, k.d) * self.periods
            self.shifts = S[np.argsort(np.linalg.norm(S, axis=1), kind="stable")]       # the zero shift first
        else:
            lo, hi = np.vstack([self.P, self.Qe, box]).min(0), np.vstack([self.P, self.Qe, box]).max(0)
            self.periods = hi - lo + far_reach(k, self.reach_tol)
            self.centre = 0.5 * (lo + hi)
            self.shifts = np.zeros((1, k.d))
        self.periodic = engine.periods is not None
        self.rmax = 2 * float(L.max(initial=0.0)) + self.rc
        same = reuse is not None and reuse.key[:3] == key[:3] and reuse.pairs is not None
        self.pairs = reuse.pairs if same else None
        if same:
            self.stats.update({x: reuse.stats[x] for x in ("pairs", "cross", "t_pairs", "t_exact")})
        self.levels, self.used, self.est, self.fixed = {}, None, None, False

    # ------------------------------------------------------------------ resolutions
    def level(self, lxi):
        """The resolution l Xi, built if needed: far field first (a refusal comes before the corrections), then C."""
        L = self.levels.get(lxi)
        if L is None:
            Xi = lxi / self.kernel.l
            L = _Level(lxi, Xi, tuple(2 * np.ceil(Xi * self.periods / (2 * np.pi)).astype(int)))
            self._check_memory(L)
            self.levels[lxi] = L
        if L.far is None:
            L.far = self._far(L)
        if L.C is None:
            L.C = self._corrections(L)
        return L

    def _check_memory(self, L):
        from ...settings import settings
        up = 1.25 if self.eps >= 1e-8 else 2.0                      # finufft's upsampling (one fine grid per plan)
        grid = 16 * (2 * float(np.prod([math.ceil(up * m) for m in L.M])) + float(np.prod(L.M))) / 2 ** 30
        if grid > settings.memory_gb:
            raise Unsupported(f"the spectral grid at l Xi = {L.lxi:.3g} ({' x '.join(map(str, L.M))} modes) needs "
                              f"{grid:.1f} GB > settings.memory_gb = {settings.memory_gb:g}")

    def _far(self, L):
        from ...settings import settings
        t0 = time.perf_counter()
        A, k = self.A, self.kernel
        X, W, own = ST.segment_nodes(self.P, self.Qe - self.P, self.sweight, self.sowner, 1.05 * L.Xi)
        if len(self.widx):
            from .weighted_spectral import weighted_nodes
            Xw, Ww, ow = weighted_nodes(A, self.widx, 1.05 * L.Xi)
            X, W, own = np.vstack([X, Xw]), np.r_[W, Ww], np.r_[own, ow]
        X = 2 * np.pi * (X - self.centre) / self.periods
        if self.periodic:
            X = np.mod(X + np.pi, 2 * np.pi) - np.pi                   # the lattice sum is periodic: exact
        xi = [2 * np.pi * np.fft.fftfreq(m, 1.0 / m) / p for m, p in zip(L.M, self.periods)]
        xi2 = sum(np.meshgrid(*[x * x for x in xi], indexing="ij", sparse=True))
        weights = (ST.khat(k, L.Xi, xi2) / float(np.prod(self.periods))).ravel()
        far = dict(tr=_Transforms(X, L.M, self.eps, settings.workers), W=W, own=own, weights=weights)
        self.stats[f"t_far(lXi={L.lxi:.3g})"] = time.perf_counter() - t0
        return far

    def _pairs(self):
        if self.pairs is None:
            t0 = time.perf_counter()
            from ...settings import settings
            a, b, k = (ST.close_pairs(self.P, self.Qe, self.rc, self.shifts, self.sowner, self.smax, settings.memory_gb)
                       if len(self.sidx) else (np.zeros(0, int),) * 3)
            t1 = time.perf_counter()
            eK = ST.exact_energies(self.kernel, self.P, self.Qe, a, b, k, self.shifts) if len(a) else np.zeros(0)
            self.pairs = (a, b, k, eK)
            self.stats.update(pairs=len(a), cross=int(np.sum(self.sowner[a] != self.sowner[b])), t_pairs=t1 - t0,
                              t_exact=time.perf_counter() - t1)
        return self.pairs

    def _corrections(self, L):
        import scipy.sparse as sps
        a, b, k, eK = self._pairs()
        t0 = time.perf_counter()
        e = np.zeros(0) if not len(a) else (eK - ST.taper_energies(ST.TaperedTable(self.kernel, L.Xi, self.rmax), self.P,
                                                                      self.Qe, a, b, k, self.shifts, 1.05 * L.Xi)) \
            * self.sweight[a] * self.sweight[b]
        oa, ob, off = self.sowner[a], self.sowner[b], a != b
        n = self.A.n
        C = sps.coo_matrix((np.r_[e, e[off]], (np.r_[oa, ob[off]], np.r_[ob, oa[off]])), shape=(n, n)).tocsr()
        C = C + weighted_corrections(self, L) if len(self.widx) else C           # EXP-031 O2: weighted near pairs
        self.stats[f"t_taper(lXi={L.lxi:.3g})"] = time.perf_counter() - t0
        return C

    def _drop(self, L):
        self.levels.pop(L.lxi, None)

    # ------------------------------------------------------------------ products and estimates
    def product(self, L, V):
        V = np.asarray(V, float)
        one = V.ndim == 1
        V2 = V[:, None] if one else V
        far = L.far
        tr, W, own = far["tr"], far["W"], far["own"]
        per = 16.0 * (2 * len(W) + 2 * float(np.prod([math.ceil(1.25 * m) for m in L.M])))
        step = int(max(1, min(V2.shape[1], MEM_BYTES // per)))
        out = np.asarray(L.C @ V2, float)
        for c0 in range(0, V2.shape[1], step):
            c = (W[:, None] * V2[own, c0:c0 + step]).T.astype(complex)
            g = tr.type2(tr.type1(c) * far["weights"]).real
            for j in range(g.shape[0]):
                out[:, c0 + j] += np.bincount(own, weights=W * g[j], minlength=V2.shape[0])
        return out[:, 0] if one else out

    def apply(self, V):
        return self.product(self.used, V)

    def _estimate(self, hi, lo, V):
        Eh, El = self.product(hi, V), self.product(lo, V)
        q = (hi.Xi / lo.Xi) ** self.p - 1 if not len(self.widx) else observed_q(self, hi, lo, V, Eh, El)
        est = float(np.max(np.linalg.norm(Eh - El, axis=0) / (q * np.linalg.norm(Eh, axis=0))))
        return est if not len(self.widx) else max(est, 10 * self.eps)    # the floor of the transforms' tolerance

    def probes(self):
        return np.c_[np.ones(self.A.n), np.random.default_rng(0).standard_normal((self.A.n, self.nprobes))]

    def fit(self, lxi=None):
        """The resolution in use: the caller's l Xi, or from the a-priori law up by sqrt 2 until the two-resolution
        estimate meets the target (refused above LXI_MAX). Only the resolution in use is kept."""
        t0 = time.perf_counter()
        self.fixed = lxi is not None
        lxi = float(lxi) if self.fixed else initial_lxi(self.accuracy, self.kernel)
        if lxi > LXI_MAX and not self.fixed:
            raise Unsupported(f"the spectral engine does not reach the relative error {self.tol:.1e} (settings.accuracy "
                              f"= {self.accuracy:g} x {NOISE_REF:g}) within l Xi <= {LXI_MAX:g}: the L3 law asks for "
                              f"l Xi = {lxi:.3g}; use settings.engine = 'hmatrix' or 'exact', or a larger "
                              "settings.accuracy")
        V = self.probes()
        lo = self.level(lxi / SQ2)
        while True:
            hi = self.level(lxi)
            est = self._estimate(hi, lo, V)
            if self.fixed or est <= self.tol:
                break
            if lxi * SQ2 > LXI_MAX:
                raise Unsupported(f"the spectral engine does not reach the relative error {self.tol:.1e} (settings."
                                  f"accuracy = {self.accuracy:g} x {NOISE_REF:g}) by l Xi = {LXI_MAX:g}: estimate "
                                  f"{est:.1e} at l Xi = {lxi:.3g}; use settings.engine = 'hmatrix' or 'exact', or a "
                                  "larger settings.accuracy")
            self._drop(lo)
            lo, lxi = hi, lxi * SQ2
        self._drop(lo)
        self.used, self.est = hi, est
        self.stats.update(lxi=hi.lxi, modes=hi.M, nodes=len(hi.far["W"]), est=est, t_fit=time.perf_counter() - t0)
        return self

    def estimate(self, V=None):
        """The two-resolution estimate of the relative error of products E V (columns of V; default the probes)."""
        lo = self.level(self.used.lxi / SQ2)
        try:
            return self._estimate(self.used, lo, self.probes() if V is None else np.asarray(V, float).reshape(
                self.A.n, -1))
        finally:
            self._drop(lo)

    def estimate_data(self, c, sigma):
        """Two-resolution estimates (rms, max) over the data of |((E_a - E) c)_i| / sigma_i: the error of predicted
        data E c in noise standard deviations (plan 005 decision 3), for measure coefficients c (e.g. those of a
        posterior mean)."""
        lo = self.level(self.used.lxi / SQ2)
        try:
            D = (self.product(self.used, c) - self.product(lo, c)) / np.asarray(sigma, float)
        finally:
            self._drop(lo)
        q = (self.used.Xi / lo.Xi) ** self.p - 1
        return float(np.sqrt(np.mean(D ** 2)) / q), float(np.max(np.abs(D)) / q)

    def claims(self):
        L, st, p = self.used, self.stats, self.p
        per = " x ".join(f"{x:.3g}" for x in self.periods)
        alias = ("on the torus the lattice sum is the definition: no aliasing" if self.periodic else
                 f"aliasing: images of K_T beyond the reach of K at {self.reach_tol:.0e} of its peak (periods = extent "
                 "+ reach; not in the estimate, whose two resolutions share the periods)")
        goal = (f"l Xi fixed by the caller, target {self.tol:.1e} not enforced" if self.fixed else
                f"target {self.tol:.1e} = settings.accuracy {self.accuracy:g} x {NOISE_REF:g}")
        return Claims([
            Claim(f"products E V by tapered spectral sums (l Xi = {L.lxi:.3g}, {' x '.join(map(str, L.M))} modes on "
                  f"periods {per}; {L.far['tr'].route}, eps {self.eps:.0e}"
                  + (f"; {st['weighted']} weighted pieces by their node rules, without near corrections"
                     if st["weighted"] else "") + f") plus exact corrections on {st['pairs']} "
                  f"segment pairs closer than {self.rc / self.kernel.l:g} l ({st['cross']} between different measures"
                  + ("" if self.smax is None else f", only those meeting or at sin(angle) < {self.smax:g}") + ")",
                  DERIVED, "Khat (1 - T) >= 0, so E - E_T is the energy matrix of a positive-definite kernel: v^T E_T "
                  "v <= v^T E v and |(E - E_T)_ij| <= sqrt((E - E_T)_ii (E - E_T)_jj); the corrected pairs carry exact "
                  f"energies of K - K_T (kernels.py; K_T tabulated to ~2e-8); {alias}"),
            Claim(f"relative error of products ~ {self.est:.1e} ({goal})", CHECKED,
                  f"two resolutions (l Xi and l Xi / sqrt 2) and the rate p = {p:g} (2s + 2 - d for corrected segments; "
                  "2 beta, or 2s + 1 for pieces of full dimension, without corrections: weighted_spectral.py): "
                  "an estimate, not a bound. Predicted data err by about this times rms(d) in rms: at most "
                  f"settings.accuracy noise std when the noise std is at least {100 * NOISE_REF:g} % of the rms data "
                  "(the largest error over the data is 2-3 times the rms)")])


# ------------------------------------------------------------------------------------------ the engine

class SpectralEngine:
    """The energies of measures with the interface of energy.ExactEngine: matrices, diagonals and lattice translations
    from the exact engine; products E V from the plan of each atom set (module docstring). lxi, eps, reach_tol: fixed
    values instead of those derived from settings.accuracy (tests and studies); rcorr: r_corr / l (default RCORR);
    collinear: correct pairs of different measures only when they meet or sin(angle) < collinear (default:
    settings.collinear when a plan is made); probes: the number of Gaussian probe vectors of the estimate besides the
    constant (1; image sums take more, spectral_images.PROBES)."""

    name = "spectral (tapered NUFFT far field, exact corrections)"

    def __init__(self, exact, lxi=None, eps=None, reach_tol=None, rcorr=None, collinear=None, probes=1):
        self.exact, self.kernel = exact, exact.kernel
        self.periods, self.origin, self.images = exact.periods, exact.origin, exact.images
        self.lxi, self.eps, self.reach_tol = lxi, eps, reach_tol
        self.rcorr = float(RCORR if rcorr is None else rcorr)
        self.collinear = None if collinear is None else float(collinear)
        self.probes = int(probes)

    def shifts(self, extent=0.0):
        return self.exact.shifts(extent)

    def extent(self, *As):
        return self.exact.extent(*As)

    def energy(self, A, B):
        return self.exact.energy(A, B)

    def self_energies(self, A, idx):
        return self.exact.self_energies(A, idx)

    def self_energies_all(self, A):
        return self.exact.self_energies_all(A)

    def plan(self, A):
        """The plan of A (A.spectral), built and fitted to the current target if it is missing or has another key."""
        from ...settings import settings
        why = unsupported(self.exact, A)
        if why is not None:
            raise Unsupported(f"the spectral engine does not apply: {why}")
        tol, k = target(settings.accuracy), self.kernel
        key = ((k.s, k.d, k.l), None if self.periods is None else tuple(np.asarray(self.periods, float)),
               (self.rcorr, settings.collinear if self.collinear is None else self.collinear), self.lxi, self.eps,
               self.reach_tol, tol)
        if A.spectral is None or A.spectral.key != key:
            A.spectral = Plan(self, A, key, tol, reuse=A.spectral).fit(self.lxi)
        return A.spectral

    def apply(self, A, V):
        """E V (columns of V), by the plan of A."""
        return self.plan(A).apply(V)

    def estimate(self, A, V=None):
        return self.plan(A).estimate(V)

    def estimate_data(self, A, c, sigma):
        return self.plan(A).estimate_data(c, sigma)

    def claims(self, A):
        return self.plan(A).claims()

    def describe(self, A):
        """How products are computed, with the estimate (the claims of Gram operators)."""
        P = self.plan(A)
        L = P.used
        sel = "" if P.smax is None else f" (between measures only meeting or at sin(angle) < {P.smax:g})"
        return (f"spectral energies (tapered NUFFT at l Xi = {L.lxi:.3g} with exact corrections on {P.stats['pairs']} "
                f"segment pairs{sel}; {L.far['tr'].route}; estimated relative error {P.est:.1e}, target {P.tol:.1e})")
