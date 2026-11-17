"""Settings that change how results are computed, never which mathematical object a result is (plan 005, decisions 3 and
4); some decide whether a result is given at all: resolution_policy = "refuse" refuses where "warn" warns, and the
verification level decides how sharply a stated error is checked before an inverse is refused.

    pg.settings.accuracy          numerical error allowed in large-scale routes, as a fraction of the statistical
                                  uncertainty it sits in (default 0.01: an iterative solve for the data space stops when
                                  the rms of its residual is 1 % of a noise standard deviation). Exact routes ignore it.
    pg.settings.dense_max         the largest data dimension n for which data-space operators are dense matrices
                                  (G G* assembled, factorised, eigendecomposed); above it they stay operators (products
                                  only) and solvers are iterative. Dense n x n float64 matrices take 8 n^2 bytes.
    pg.settings.workers           processes for blocks of work (default: the environment variable PYGEOINF0_WORKERS, 1)
    pg.settings.compression       relative tolerance of compressed (far-field) blocks of energies above dense_max
                                  (default 1e-8; realisations/rkhs/hmatrix.py)
    pg.settings.memory_gb         memory allowed for what an engine stores (default 40): the exact near field and the
                                  blocks of a compressed energy matrix, the grids of the spectral engine
    pg.settings.dense_gram        "exact" (default): dense Gram matrices (n <= dense_max) from exact energies;
                                  "compressed": gathered from the H-matrix (error ~ compression, stated in the claims;
                                  faster when the measures have many atoms)
    pg.settings.weighted_grid     "auto" (default): the Gram and cross energies of weighted measures in 2-D (disks and
                                  axis-aligned boxes with a smooth weight) by the grid-spectral route when the weight's own
                                  indicator settles (realisations/rkhs/weighted_grid.py), the exact routes otherwise;
                                  "off": always the exact routes (the route used is in the engine's report)
    pg.settings.engine            the engine of products with energies above dense_max (the matrix-free route), chosen
                                  in realisations/registry.py: "auto" (default: "spectral" for segment measures in 3-D
                                  when finufft is installed, else "hmatrix"; the author's decision of 2026-10-27 after
                                  the scale validation, where 2-D setups were slower than the H-matrix), "hmatrix"
                                  (realisations/rkhs/hmatrix.py), "spectral" (segment measures: tapered NUFFT far field plus
                                  exact corrections, error estimated against settings.accuracy; realisations/rkhs/spectral.py,
                                  fast with the optional finufft; with reflection images on boxes the doubled torus of
                                  signed copies, realisations/rkhs/spectral_images.py, a prototype; other measures fall back
                                  to the H-matrix, and the claims say so) or "exact" (streamed: every product recomputes every pair)
    pg.settings.collinear         which pairs of segments of different measures the spectral engine corrects exactly:
                                  None (default) every pair closer than r_corr = 0.25 l; a number c in (0, 1] only
                                  those that meet (distance 0: shared endpoints, crossings) or are at sin(angle) < c.
                                  The pairs closer than r_corr grow like n^2 at fixed l; c = 0.3 kept the target at
                                  1e4 rays with 7x fewer pairs on L3 and made 1e5 rays fit in 16 GB (plan 005)
    pg.settings.images            where sums over reflection images (spectral spaces with Neumann/Dirichlet ends or
                                  faces, realisations/rkhs/green.py) stop: "exact" (default) where the omitted images are
                                  provably below 1e-16 of k(0) |mu| |nu| in every energy (rounding); "accuracy" where they
                                  are below settings.accuracy x 0.05 of it (fewer images: 5-40 x faster on boxes, EXP-026
                                  §14). The bound is stated in the space's claims. Set it before building Gram matrices:
                                  energies cached on an atom set are not recomputed when it changes
    pg.settings.verification      how hard an inverse checks the error it inherits from its operator's inputs (realised
                                  kernels, a truncated covariance, compressed energies) where their first-order bound
                                  leaves no correct digit (plan 054; solvers.inherited, operators/refinement.py):
                                  0 "quick": nothing is formed again; the first-order statement is given as it is and,
                                  where it leaves no digit, the inverse is given with a PrecisionWarning saying it was
                                  not verified (no refusal on it); 1 (default): one refinement (kernels at tol/100,
                                  energies at compression/100, the covariance with 4 x its modes), refused where the
                                  a-posteriori statement still leaves no digit; d >= 2: d nested refinements, each about
                                  100 x tighter, the first-order statement applied only at the deepest (closer to the
                                  true error, at the cost of the deeper realisations). Read when an inverse is first
                                  used: A.inv keeps its factorisation and its verdict
    pg.settings.resolution_policy what a fit from samples of a bare callable does (one with neither breaks=, support= nor
                                  resolution=, and pointwise re-fits such as np.exp(-x**2) of an element; core/fits.py,
                                  plan 054): "warn" (default during the warning period): it is fitted, its claims state
                                  the resolution its samples achieved, and a ResolutionWarning names the choices;
                                  "refuse": ResolutionUnstated (a Refused), the later default

    with pg.settings.using(accuracy=0.1, dense_max=2000):     # a scoped change (restored on exit)
        ...

A setting is global by design: the same user code runs at every size and on every machine, and only the settings (or
their defaults) differ. Results computed under different settings agree to within the stated accuracy (exact routes:
to rounding); the route actually used is recorded in the claims of the operators and solvers that used it.
"""
import contextlib
import numbers
import os


class Settings:
    def __init__(self):
        self.accuracy = 0.01
        self.dense_max = 10000
        self.compression = 1e-8
        self.memory_gb = 40.0
        self.dense_gram = "exact"
        self.engine = "auto"
        self.weighted_grid = "auto"
        self.collinear = None
        self.images = "exact"
        self.verification = 1
        self.resolution_policy = "warn"
        self._workers = None

    @property
    def weighted_grid(self):
        return self._weighted_grid

    @weighted_grid.setter
    def weighted_grid(self, name):
        if name not in ("auto", "off"):
            raise ValueError(f"settings.weighted_grid is 'auto' or 'off', not {name!r}")
        self._weighted_grid = name

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, name):
        if name not in ENGINES:
            raise ValueError(f"settings.engine is one of {ENGINES}, not {name!r}")
        self._engine = name

    @property
    def collinear(self):
        return self._collinear

    @collinear.setter
    def collinear(self, c):
        if c is not None and not 0 < float(c) <= 1:
            raise ValueError(f"settings.collinear is None or a number in (0, 1] (a sine), not {c!r}")
        self._collinear = None if c is None else float(c)

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, mode):
        if mode not in IMAGES:
            raise ValueError(f"settings.images is one of {IMAGES}, not {mode!r}")
        self._images = mode

    @property
    def verification(self):
        return self._verification

    @verification.setter
    def verification(self, d):
        if isinstance(d, bool) or not isinstance(d, numbers.Integral) or d < 0:
            raise ValueError(f"settings.verification is an integer >= 0 (0 quick, 1 one refinement, d nested), not {d!r}")
        self._verification = int(d)

    @property
    def resolution_policy(self):
        return self._resolution_policy

    @resolution_policy.setter
    def resolution_policy(self, p):
        if p not in ("warn", "refuse"):
            raise ValueError(f"settings.resolution_policy is 'warn' or 'refuse', not {p!r}")
        self._resolution_policy = p

    @property
    def workers(self):
        return self._workers if self._workers is not None else max(1, int(os.environ.get("PYGEOINF0_WORKERS", "1")))

    @workers.setter
    def workers(self, n):
        self._workers = None if n is None else max(1, int(n))

    @contextlib.contextmanager
    def using(self, **kw):
        old = {}
        for k, v in kw.items():
            if k not in KEYS:
                raise AttributeError(f"no setting {k!r} ({', '.join(KEYS)})")
            old[k] = self._workers if k == "workers" else getattr(self, k)
            setattr(self, k, v)
        try:
            yield self
        finally:
            for k, v in old.items():
                if k == "workers":
                    self._workers = v
                else:
                    setattr(self, k, v)

    def __repr__(self):
        return (f"settings(accuracy={self.accuracy:g}, dense_max={self.dense_max}, workers={self.workers}, "
                f"compression={self.compression:g}, memory_gb={self.memory_gb:g}, dense_gram={self.dense_gram!r}, "
                f"engine={self.engine!r}, collinear={self.collinear!r}, images={self.images!r}, "
                f"verification={self.verification}, resolution_policy={self.resolution_policy!r})")


KEYS = ("accuracy", "dense_max", "workers", "compression", "memory_gb", "dense_gram", "engine", "collinear", "images", "weighted_grid",
        "verification", "resolution_policy")
ENGINES = ("auto", "hmatrix", "spectral", "exact")
IMAGES = ("exact", "accuracy")


settings = Settings()
