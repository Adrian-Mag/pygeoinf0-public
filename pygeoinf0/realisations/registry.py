"""The one place where a function space's realisation is chosen.

choose(measure, tol, realisation=None, sobolev=None, sup=False) -> backend   (backend.realisation_name: resolved name)

    domain                     default realisation                    alternatives (realisation=...)
    Interval                   "legendre" (piecewise, adaptive)        -
    Sphere                     "harmonics" (spherical harmonics)       "tensor" (Legendre in theta x Fourier in phi)
    Ball, Shell                "tensor" (Legendre in r, theta;         "legendre"; "radial_harmonics" (piecewise
                               Fourier in phi)                         Legendre in r x spherical harmonics, plan 009
                                                                       G1: radial_harmonic.py)
    Box                        "local" (tensor Legendre; functions     "tensor", "legendre"
                               with a stated support box fitted on
                               that box only, sums kept as patches,
                               translates shifted: local.py, plan 025;
                               other functions exactly as "tensor")
    other Domain               "tensor" (Legendre; Fourier on          "legendre" (Legendre on every axis, periodic
                               periodic axes)                          ones included)

"radial_harmonics" is an explicit option, not the default on balls and shells (plan 009 P2): the tensor realisation's
results (its tests, the examples, plan 007's Moon) stay as they were, and its product families (rays and other
localised kernels, separated.py) have no counterpart there. Choose it for problems whose data are harmonic or zonal
(separable functionals, band projections, rotation-invariant priors); a layered ball takes it per layer
(pg.PiecewiseSpace.lebesgue(B, realisation="radial_harmonics") for a LayeredBall B).

With sup=True the backend of C(X) for a closed interval X (sup.SupIntervalBackend: continuity checked, errors and
norms in the sup norm). With sobolev=C, a construction of spaces/constructions.py, the backend of that Sobolev space:

    construction                                   realisation
    RestrictionSpace(D, s, l)                      RKHSBackend with the Matern kernel, in the flat coordinates of D
                                                   (on intervals with integer s: formula norms, green.IntervalForm)
    Spectral(A, s), A = 1 - l^2 Delta on a torus   RKHSBackend with the lattice sum of the Matern kernel (Poisson)
      or a circle
    Spectral(A, s) on a sphere                     SphereSobolevBackend (spherical harmonics)
    Spectral(A, s) on an interval, s/k integer     RKHSBackend, Matern kernel + finite-rank boundary correction
      (forms a.space(): s = k)                     (green.EndBasis), formula norms from derivatives
    Spectral(A, s) on an interval, other s         RKHSBackend, signed reflection images (green.Reflections;
                                                   Neumann/Dirichlet ends)
    Spectral(A, s) on a box (rectangle), any s     RKHSBackend, signed reflection images (Neumann/Dirichlet faces)

Spectra of Sturm-Liouville operators (plan 009 G5; spectrum(op)): the per-degree radial Laplacian of a ball or shell in
closed form (bessel.py), any other in Galerkin brackets (sl_galerkin.py).

No other module decides on a realisation from the type of a domain.

Capabilities (plan 069 P3, review C3): what a realisation can do is asked by name, never read from its class. A
realisation (an element's R, a backend, a Gram matrix) declares CAPABILITIES (names of the protocol below, inherited by
subclasses); a backend also has the optional operations it implements (Backend.supports) and the derived ones.

    provides(obj, name) -> bool          capabilities(obj) -> frozenset
    require(obj, name, purpose) -> obj   capability(obj, name, purpose) -> the bound method
    a missing one refuses (core.errors.CapabilityMissing, an Unsupported) with its name, the purpose, what obj has and
    which realisations have it; a name outside CAPABILITIES is a ValueError naming the protocol.

Outside realisations/ nothing tests the type of a realisation (tests/test_structure.py, a list that may only shrink).

The energies of the measures of an RKHSBackend are realised here too (energy_engine, Energies): exact matrices and
diagonals, and products E V above settings.dense_max by the engine settings.engine names, where it applies:

    settings.engine   products above dense_max                     falls back (stated in the claims) for
    "auto" (default)  "spectral" for segments in 3-D with finufft   everything else: as "hmatrix"
    "hmatrix"         hmatrix.HEngine                               periodic domains: exact energies, streamed
    "spectral"        spectral.SpectralEngine (tapered NUFFT plus   atoms other than segments, 1-D coordinates:
                      exact corrections; segments in 2-D and 3-D,   the H-matrix (exact energies, streamed, on
                      domains of R^d and flat tori); with           periodic domains)
                      reflection images (boxes with Neumann/
                      Dirichlet faces) spectral_images.
                      ImagesSpectralEngine: signed copies on the
                      doubled (quadrupled on mixed axes) torus,
                      a prototype ("auto" keeps the H-matrix there)
    "exact"           energy.ExactEngine, streamed block by block   -
"""

import collections
import math

import numpy as np

from ..core.errors import Unsupported

Capability = collections.namedtuple("Capability", "what method backend")
Capability.__doc__ = """One capability of the protocol: what it computes or states; method: obj has a method of that
name; backend: None, or "optional" (a Backend operation, provided where Backend.supports says so) or "derived" (every
Backend has it)."""
_E, _B = "an element's realisation R", "a backend B"
CAPABILITIES = {
    "box_support": Capability(f"{_E}: R.box_support(H, l, u, rtol) -> (lower, upper), the certified ends of "
                              "int (u q+ - l q-) dmu in the summand H (equal where exact; box_integrals.py)", True, None),
    "box_member": Capability(f"{_E}: R.box_member(H, l, u) -> the Truth of l <= q <= u a.e., exact", True, None),
    "box_rule": Capability(f"{_E}: R.box_rule(H, Rs, l, u, weight, oversample, least) -> the fixed search quadrature "
                           "of oCSOLA with a box on one summand (w, lo, hi, rho2, exact, kind, values(R, cols, N, chunk))",
                           True, None),
    "cell_integrals": Capability(f"{_E} piecewise on an interval (R.breaks): R.cell_integrals(edges) -> (cells, N), the "
                                 "exact integrals over [edges_i, edges_i+1]", True, None),
    "harmonic_coefficients": Capability(f"{_E} in spherical harmonics (R.L, R.R, R.err): R.harmonic_coefficients(L) -> "
                                        "((L + 1)^2, N), the coefficients padded to degree L", True, None),
    "radial_harmonic_coefficients": Capability(f"{_E} or {_B} in radial x harmonic factors (R.L, R.breaks, R.coefs; "
                                               "B.a, B.b, B.domain)", False, None),
    "piecewise_legendre": Capability(f"{_E} as a piecewise Legendre series on an interval (R.breaks, R.coefs): the "
                                     "functions of the Legendre series apply to it", False, None),
    "symbolic_measures": Capability(f"{_E} of a direct-sum summand kept symbolic as its measures (R.measures)",
                                    False, None),
    "kronecker_gram": Capability(f"{_E} as a separated product family: its Gram matrices are Kronecker products",
                                 False, None),
    "kronecker_product": Capability("a Gram matrix kept as a Kronecker product of factors", False, None),
    "cell_values": Capability(f"{_B} of cell-constant functions on a partition (B.cells: volumes, n)", False, None),
    "breaks_per_axis": Capability(f"{_B} taking breaks per axis (a tuple of break arrays), on an interval too",
                                  False, None),
    "fit_family": Capability(f"{_B}: B.fit_family(fun_idx, N, ...) -> R, N functions realised at once", True,
                             "derived"),
    "legendre_decay_generators": Capability(f"{_B}: B.legendre_decay_generators(decay, K) -> (R, lam), the orthonormal "
                                            "Legendre functions and decay^(2k)", True, "optional"),
}


def capabilities(obj):
    """The capabilities obj provides (module docstring)."""
    out = set(getattr(type(obj), "CAPABILITIES", ()))
    from .backends import Backend
    if isinstance(obj, Backend):
        out |= {n for n, c in CAPABILITIES.items() if c.backend == "derived" or
                (c.backend == "optional" and obj.supports(n))}
    return frozenset(out)


def _known(name):
    if name not in CAPABILITIES:
        raise ValueError(f"{name!r} is not a capability; the protocol has {', '.join(sorted(CAPABILITIES))}")


def provides(obj, name):
    _known(name)
    return name in capabilities(obj)


def _holders(name):
    """The realisation classes declaring name (for the refusal; every module of this package is imported)."""
    import importlib
    import pkgutil
    import sys
    pkg = sys.modules[__package__]
    out = set()
    for m in pkgutil.walk_packages(pkg.__path__, __package__ + "."):
        mod = importlib.import_module(m.name)
        out |= {v.__name__ for v in vars(mod).values()
                if isinstance(v, type) and v.__module__ == m.name and name in getattr(v, "CAPABILITIES", ())}
    return sorted(out) + (["backends that implement it"] if CAPABILITIES[name].backend else [])


def require(obj, name, purpose=None):
    """obj, when it provides name; else CapabilityMissing naming name, the purpose, what obj has and who has name."""
    if provides(obj, name):
        return obj
    from ..core.errors import CapabilityMissing
    have = capabilities(obj)
    holders = _holders(name)
    raise CapabilityMissing(
        (f"{purpose}: " if purpose else "") + f"the realisation {type(obj).__name__} has no capability {name!r} "
        f"({CAPABILITIES[name].what}); it has " + (", ".join(sorted(have)) or "none") +
        (f"; {name!r} is provided by " + ", ".join(holders) if holders else ""), name, have)


def capability(obj, name, purpose=None):
    """The method of obj that implements the capability name (require's refusal when obj has none)."""
    require(obj, name, purpose)
    if not CAPABILITIES[name].method:
        raise ValueError(f"{name!r} is a stated capability without a method; ask provides or require")
    return getattr(obj, name)


REALISATIONS = {"Interval": ("legendre",), "Sphere": ("harmonics", "tensor"),
                "Ball": ("tensor", "legendre", "radial_harmonics"), "Domain": ("tensor", "legendre", "local")}

Geometry = collections.namedtuple("Geometry", "dim flat unflat periods origin definition")
Geometry.__doc__ = """How a domain carries Sobolev spaces: the dimension of its flat coordinates, the maps between
Cartesian and flat coordinates, the periods and cell origin of periodic ones (else None), and the definition:
'restriction' (of H^s(R^dim) to the domain), 'periodic' (spectral = lattice sum of images) or 'spherical' (spectral
in spherical harmonics)."""


def choose(measure, tol, realisation=None, sobolev=None, sup=False):
    from .piecewise_constant import Cells
    if isinstance(realisation, Cells):                 # an explicit partition (plan 020 P2): L2 only
        if sobolev is not None or sup:
            raise Unsupported("cell-constant functions realise L2 only: they are not in Sobolev spaces of order >= 1/2 "
                              "nor continuous")
        from ..measures import LebesgueMeasure
        from ..domains import Box
        if not isinstance(measure, LebesgueMeasure) or not isinstance(measure.set, Box):
            raise Unsupported("the cell realisation is for Lebesgue measure on a box")
        from .piecewise_constant import CellsBackend
        return CellsBackend(measure, tol, realisation)
    D = measure.set
    if sup:
        from ..sets import Interval
        if not isinstance(D, Interval) or realisation not in (None, "legendre"):
            raise Unsupported(f"spaces of continuous functions are realised on intervals (piecewise Legendre); not on {D!r}")
        from .series.sup import SupIntervalBackend
        return SupIntervalBackend(measure, tol)
    if sobolev is None:
        return _choose(measure, tol, realisation)
    C = sobolev
    if realisation == "radial_harmonics":
        raise Unsupported("Sobolev spaces are not realised in radial x harmonic factors yet (L2 only, plan 009 P2)")
    s, l = C.s, C.length_scale
    g = flat_geometry(D)
    if g.definition == "spherical":
        if realisation not in (None, "harmonics"):
            raise Unsupported("Sobolev spaces on a sphere are realised in spherical harmonics")
        from .series.sphere_sobolev import SphereSobolevBackend
        return SphereSobolevBackend(measure, tol, s, l, g)
    from .rkhs.kernels import Matern
    from .rkhs.backend import RKHSBackend
    from ..sets import Interval
    if C.kind == "restriction" or g.definition == "periodic":
        B = _choose(measure, tol, realisation)
        formula = None
        if isinstance(D, Interval) and C.kind == "restriction":
            from .rkhs.green import interval_form
            formula = interval_form(C, D.a, D.b)
        return RKHSBackend(B, Matern(s, g.dim, l), D, g, B.realisation_name, formula=formula)
    if isinstance(D, Interval):
        from .rkhs.green import interval_realisation
        r = interval_realisation(C, D)
        B = _choose(measure, tol, realisation)
        return RKHSBackend(B, Matern(s, 1, l), D, g, B.realisation_name, images=r.get("images"),
                           correction=r.get("correction"), formula=r.get("formula"), claims=r.get("claims", ()))
    from ..domains import Box
    if isinstance(D, Box):
        from .rkhs.green import box_realisation
        B = _choose(measure, tol, realisation)
        r = box_realisation(C, D)
        formula = None
        if C.kind == "spectral" and C.form.k == 1 and C.s == 1 and not C.form.robin:   # the form space (plan 025 P4)
            from .rkhs.box_form import BoxForm
            formula = BoxForm(B, l, D.boundary_parts, C.form.dirichlet)
        return RKHSBackend(B, Matern(s, g.dim, l), D, g, B.realisation_name, images=r["images"], claims=r["claims"],
                           formula=formula)
    raise Unsupported(f"{C.label()} on {D!r} has no realisation")


def _choose(measure, tol, realisation):
    from ..sets import Interval
    from ..domains import Ball, Shell, Sphere
    D = measure.set
    kind = "Interval" if isinstance(D, Interval) else "Sphere" if isinstance(D, Sphere) else \
        "Ball" if isinstance(D, (Ball, Shell)) else "Domain"
    options = REALISATIONS[kind]
    if realisation is None and kind == "Domain":
        from ..domains import Box
        if isinstance(D, Box):
            realisation = "local"                          # the default on boxes (the author, 2026-10-31; plan 025)
    name = options[0] if realisation is None else realisation
    if name not in options:
        raise ValueError(f"realisation {name!r} is not available on {D!r}; choose one of {options}")
    if kind == "Interval":
        from .backends import IntervalBackend
        return IntervalBackend(measure, tol)
    if name == "harmonics":
        from .series.spherical import SphereBackend
        return SphereBackend(measure, tol)
    if name == "radial_harmonics":
        from .series.radial_harmonic import RadialHarmonicsBackend
        return RadialHarmonicsBackend(measure, tol)
    if name == "local":
        from ..domains import Box
        if not isinstance(D, Box):
            raise Unsupported(f"the support-local realisation is for boxes, not {D!r}")
        from .series.local import LocalBackend
        return LocalBackend(measure, tol)
    from .series.tensor import TensorBackend
    return TensorBackend(measure, tol, periodic_basis="legendre" if name == "legendre" else "fourier")


def flat_geometry(D):
    """The Geometry of a domain: the coordinates in which the kernel of its Sobolev spaces is the free-space Matern
    kernel (Cartesian on domains of R^d and tori, arc length on paths and circles), or the spectral sphere."""
    from ..sets import Interval
    from ..domains import Arc, Ball, Box, Circle, Curve, Cylinder, Disk, Flat, Polyline, Shell, Sphere, Torus
    if isinstance(D, Interval):
        return Geometry(1, lambda X: X[:, :1], lambda F: F, None, None, "restriction")
    if isinstance(D, (Box, Disk, Cylinder, Ball, Shell)):
        return Geometry(D.dim, lambda X: X, lambda F: F, None, None, "restriction")
    if isinstance(D, Torus):
        return Geometry(D.dim, lambda X: X, lambda F: F, [s.length for s in D.sides], [s.a for s in D.sides],
                        "periodic")
    if isinstance(D, Sphere):
        return Geometry(2, None, None, None, None, "spherical")
    if isinstance(D, Circle):
        R = D.radius

        def flat(X):
            return R * D.from_cartesian(X[:, 0], X[:, 1])[0][:, None]

        def unflat(F):
            return np.column_stack(D.to_cartesian(F[:, 0] / R))
        return Geometry(1, flat, unflat, [2 * math.pi * R], [0.0], "periodic")
    if isinstance(D, Flat) and D.dim == 2:                # a flat patch: orthonormal coordinates in its plane
        o, (e1, e2) = np.array(D.origin), np.array(D.edges)
        u1 = e1 / np.linalg.norm(e1)
        u2 = e2 - (e2 @ u1) * u1
        frame = np.stack([u1, u2 / np.linalg.norm(u2)])

        def flat(X):
            return (X - o) @ frame.T

        def unflat(F):
            return o + F @ frame
        return Geometry(2, flat, unflat, None, None, "restriction")
    if isinstance(D, Curve) and D.closed:                  # a closed curve: periodic in arc length, like a circle
        def flat(X):
            return D.from_cartesian(*[X[:, k] for k in range(X.shape[1])])[0][:, None]

        def unflat(F):
            return np.column_stack(D.to_cartesian(np.mod(F[:, 0], D.length)))
        return Geometry(1, flat, unflat, [D.length], [0.0], "periodic")
    if isinstance(D, (Flat, Polyline, Arc, Curve)) and D.dim == 1:
        def flat(X):
            return D.from_cartesian(*[X[:, k] for k in range(X.shape[1])])[0][:, None]

        def unflat(F):
            return np.column_stack(D.to_cartesian(F[:, 0]))
        return Geometry(1, flat, unflat, None, None, "restriction")
    raise Unsupported(f"Sobolev spaces on {D!r} are not implemented (curved surfaces other than spheres)")


def energy_engine(kernel, periods=None, origin=None, images=None):
    """The energies of an RKHSBackend's measures: Energies around the exact engine of the kernel (with lattice
    periods and origin on tori and circles, reflection images on intervals and boxes)."""
    from .rkhs.energy import ExactEngine
    return Energies(ExactEngine(kernel, periods, origin, images))


class Energies:
    """Energies of measures (the engine interface of energy.ExactEngine): matrices and diagonals exact (or gathered
    from the H-matrix under settings.dense_gram = "compressed", hmatrix.HEngine), products E V from the energies when
    they are stored or n <= settings.dense_max, else from the engine route(A) chooses (module docstring)."""

    def __init__(self, exact):
        from .rkhs.hmatrix import HEngine
        self.exact, self.kernel = exact, exact.kernel
        self.periods, self.origin, self.images = exact.periods, exact.origin, exact.images
        self.hengine = HEngine(exact)
        self.name = self.hengine.name
        self._spectral = None
        from .rkhs.weighted_grid import GridEngine
        self.grid = GridEngine(exact)           # weighted measures in 2-D (plan 022 P3)

    def shifts(self, extent=0.0):
        return self.exact.shifts(extent)

    def extent(self, *As):
        return self.exact.extent(*As)

    def energy(self, A, B):
        if A is B:
            self.hengine.drop_stale(A)
        if A is B and A.self_energy is not None:
            return A.self_energy[0]
        E = self.grid.energy(A, B)                      # smooth weights on the grid route, the rest exact (P3)
        if E is None:
            return self.hengine.energy(A, B)
        if A is B:
            E = 0.5 * (E + E.T)
            d = np.sqrt(np.abs(np.diag(E)))
            A.self_energy = (E, self.grid.bound() * np.outer(d, d))       # stated, not zero
        return E

    def self_energies(self, A, idx):
        out = self.grid.self_energies(A, idx)
        return out if out is not None else self.exact.self_energies(A, idx)

    def self_energies_all(self, A):
        return self.exact.self_energies_all(A)

    def hmatrix(self, A):
        return self.hengine.hmatrix(A)

    def apply(self, A, V):
        from ..settings import settings
        self.hengine.drop_stale(A)
        if A.self_energy is not None or A.n <= settings.dense_max:
            return self.energy(A, A) @ np.asarray(V, float)
        return self.route(A)[1].apply(A, V)

    def route(self, A):
        """(name, engine, reason) of the products with the energies of the atom set A above settings.dense_max."""
        from ..settings import settings
        if settings.engine == "exact":
            return "exact", self.exact, "settings.engine = 'exact': exact energies streamed block by block"
        fallback = (("exact", self.exact, "exact energies streamed block by block (no H-matrix on periodic domains)")
                    if self.periods is not None else ("hmatrix", self.hengine, "the H-matrix"))
        if settings.engine == "spectral" and self.images is not None:
            from .rkhs.energy import ExactEngine
            from .rkhs.spectral import SpectralEngine
            from .rkhs.spectral_images import PROBES, ImagesSpectralEngine, copies, unsupported
            if self._spectral is None:
                _, P, origin = copies(self.images)
                self._spectral = ImagesSpectralEngine(self.exact, SpectralEngine(ExactEngine(self.kernel, P, origin),
                                                                                 probes=PROBES))
            why = unsupported(self._spectral, A)
            if why is None:
                return "spectral", self._spectral, ("settings.engine = 'spectral' (reflection images as signed copies "
                                                    "on a torus, a prototype)")
            return "hmatrix", self.hengine, f"settings.engine = 'spectral' does not apply ({why}): the H-matrix"
        if settings.engine in ("spectral", "auto"):
            from .rkhs.spectral import SpectralEngine, unsupported, unsupported_auto
            why = unsupported(self.exact, A)
            if why is None and settings.engine == "auto":
                why = unsupported_auto(self.exact)
            if why is None:
                if self._spectral is None:
                    self._spectral = SpectralEngine(self.exact)
                return "spectral", self._spectral, f"settings.engine = {settings.engine!r}"
            if settings.engine == "auto":
                return fallback[0], fallback[1], f"settings.engine = 'auto' ({why}): {fallback[2]}"
            return fallback[0], fallback[1], f"settings.engine = 'spectral' does not apply ({why}): {fallback[2]}"
        return fallback

    def describe(self, A):
        """How products with the energies of A are computed, for the claims of Gram operators; None for the H-matrix
        chosen by "hmatrix" or "auto" (whose claim states settings.compression and the H-matrix's estimate)."""
        from ..settings import settings
        if settings.engine == "hmatrix":
            return None
        name, engine, why = self.route(A)
        if name == "hmatrix" and settings.engine == "auto":
            return None
        if name == "spectral":
            return engine.describe(A)
        if name == "exact":
            return why
        H = A.hmatrix
        return (f"{engine.name} energies (settings.compression = "
                f"{settings.compression if H is None else H.compression:g}"
                + (f" when the H-matrix was built; estimated relative error {H.rel_error:.1e}" if H is not None else "")
                + f"), because {why}")


def spectrum(op):
    """The realisation of the spectrum of a Sturm-Liouville operator (operators/unbounded.py): closed forms in
    spherical Bessel functions for the per-degree radial Laplacian (bessel.RadialBessel), Galerkin upper bounds with
    comparison lower bounds otherwise (sl_galerkin.GalerkinSpectrum)."""
    if op.radial is not None:
        from .spectra.bessel import RadialBessel
        return RadialBessel(op)
    from .spectra.sl_galerkin import GalerkinSpectrum
    return GalerkinSpectrum(op)


def laplacian_geometry(D):
    """What the Laplacian of the domain D is built from (operators/unbounded.Laplacian; plan 068 P3): "interval", "box",
    "ball", "shell", "circle", "sphere", "torus", or the name of a domain whose Laplacian spectrum is not built ("disk",
    "cylinder", ...), or None."""
    from ..sets import Interval
    from ..domains import Ball, Box, Circle, Cylinder, Disk, Shell, Sphere, Torus
    if isinstance(D, Interval):
        return "interval"
    for cls, name in ((Shell, "shell"), (Ball, "ball"), (Box, "box"), (Torus, "torus"), (Sphere, "sphere"),
                      (Circle, "circle"), (Disk, "disk"), (Cylinder, "cylinder")):
        if isinstance(D, cls):
            return name
    return None


def laplacian_spectrum(lap):
    """The realisation of the spectrum of pg.Laplacian(D, bc) (realisations/spectra/laplace_spectra.py): closed forms on an
    interval, a circle and a sphere; sums over the axes of a box or a flat torus; the radial spectra of the degrees
    (bessel.RadialBessel) merged on a ball or a shell."""
    from .spectra import laplace_spectra as LS
    kind, D = lap.geometry, lap.region
    if kind == "interval":
        return LS.IntervalModes(D.a, D.b, lap.bc["left"], lap.bc["right"])
    if kind == "box":
        return LS.ProductSpectrum([LS.IntervalModes(s.a, s.b, lap.bc[f"{n}-"], lap.bc[f"{n}+"])
                                   for n, s in zip(D.internal_names, D.sides)], what=f"{D!r}")
    if kind in ("ball", "shell"):
        return LS.DegreeSpectrum(lap)
    if kind == "circle":
        return LS.CircleSpectrum(D.radius)
    if kind == "sphere":
        return LS.SphereSpectrum(D.radius)
    if kind == "torus":
        return LS.ProductSpectrum([LS.CircleSpectrum(s.length / (2 * math.pi)) for s in D.sides], what=f"{D!r}")
    raise Unsupported(f"the spectrum of the Laplacian of {D!r} is not realised")


def choose_sum(space):
    """The realisation of a direct sum (spaces/direct_sum.py): the tuple of its summands' realisations, so nothing is
    chosen here; each summand keeps the realisation chosen for it above."""
    from .series.direct_sum import SumBackend
    return SumBackend(space)
