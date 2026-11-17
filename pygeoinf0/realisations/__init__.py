"""Numerical realisations of elements, grouped by the mathematics they compute (plan 069 P4, review C4).

The numerical column below the spaces it serves (AGENTS.md: explicit, optional objects; a space is the same whatever the
realisation of its elements, plan 069 P1):

series/              expansions of functions in a basis: piecewise Legendre series on an interval (legendre, sparse,
                     sup), tensor products on boxes and their support-local patches and product families (tensor,
                     local, separated, cuts), spherical harmonics (spherical, sphere_sobolev, harmonic_degrees),
                     radial x harmonic factors on balls and shells (radial_harmonic, radial_fit, radial_direct), and
                     the tuples of a direct sum (direct_sum)
rkhs/                the pairing oracle <k_mu, k_nu> = int int K dmu dnu of a reproducing (Bessel-potential) kernel:
                     kernels, measures as atoms, the energy engines (exact, hierarchical, spectral, weighted pieces)
                     and the Sobolev backend built on them (rkhs/__init__ states the formula)
spectra/             spectra of self-adjoint operators that are realisations: closed-form Laplacian spectra, radial
                     Bessel spectra, Sturm-Liouville Galerkin brackets, and the Bessel-potential covariances built
                     on them
bounds/              certified bounds of realised functions: extrema of Legendre series, L1 integrals on spheres,
                     balls and shells, and the integrals of a pointwise box per realisation
piecewise_constant.py  cell averages on a partition of a box into boxes (Cells: a realisation option for L2; plan 020
                     P2; renamed from cells.py, decision Q6)
backends.py          the backend contract (class Backend) and the interval backend
registry.py          choose(): the one place where a space's realisation is chosen; the capability protocol
quadrature.py        explicit quadrature rules as realisation options (Trapezoid); the rules themselves are
                     pygeoinf0.quadrature

The continuum's spherical harmonics (Y_lj, pg.HarmonicConvention, Funk-Hecke, P_l) are spaces/harmonic.py; the worker
pools are the runtime's pygeoinf0.execution, beside pygeoinf0.settings.
"""
from .piecewise_constant import Cells  # noqa: F401
from .registry import choose  # noqa: F401
from .quadrature import Trapezoid  # noqa: F401
