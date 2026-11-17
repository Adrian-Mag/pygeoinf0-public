"""Realisations as expansions of functions in a basis (plan 069 P4): an element of L2(D) (or of a space built on it)
is held as the coefficients of a truncated series f = sum_k c_k phi_k with an error estimate, and inner products,
norms and integrals are exact on the realised series.

legendre.py          piecewise Legendre series on an interval (PiecewiseLegendre; adaptive, with error estimates), and
                     the density fit of a measure on an interval (fit_density)
sparse.py            localised families on an interval (SparsePiecewiseLegendre: each piece keeps its own members)
sup.py               the piecewise Legendre realisation of C([a, b]) in the sup norm
tensor.py            tensor products of axis bases on the reference box of a domain (TensorSeries)
local.py             support-local patches of tensor series on boxes (PS)
separated.py         product families, kernel expansions and lazy combinations (ProductFamily, KernelExpansion,
                     LazyCombination)
cuts.py              tensor realisations cut to a sub-box, and pieces joined into one
spherical.py         spherical-harmonic series on a sphere (SphericalHarmonicSeries)
sphere_sobolev.py    Sobolev spaces on spheres: spherical harmonics with the spectral metric
harmonic_degrees.py  degree-wise operations on harmonic realisations (band limits, degree scalings)
radial_harmonic.py   radial x spherical-harmonic series on balls and shells (RadialHarmonicSeries, and the separable
                     functions SeparableRadialHarmonics); radial_fit.py its adaptive fit, radial_direct.py separable
                     functionals against it
radial_integrals.py  radial integrals of piecewise Legendre functions in L2(r^2 dr) (Gauss rules on the factors'
                     breakpoints; plan 070 P7: from probability/degrees.py)
direct_sum.py        the tuple of the summands' realisations of an element of a direct sum (DirectSumRealisation)
symbolic_harmonics.py  the harmonic coefficients t_lj(r) of a density kept symbolic on a ball or shell, and the box
                     holding a measure's support (plan 070 P1: from operators/harmonic_targets.py)

The harmonics Y_lj themselves are the continuum's: spaces/harmonic.py.
"""
