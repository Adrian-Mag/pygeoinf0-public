"""The pairing oracle of a reproducing-kernel space (plan 069 P4, review C4): for measures mu, nu and the reproducing
kernel K of a Sobolev space H^s_l(X) (the Bessel-potential kernel of H^s_l(R^d), restricted to a domain or with its
periodic or reflected images, kernels.py), this package computes

    <k_mu, k_nu>_H = int int K(x, y) dmu(x) dnu(y),      k_mu = int K(., y) dmu(y)  (the Riesz representer of mu),

and the values k_mu(x) = int K(x, y) dmu(y), each with an error estimate. The Riesz representers of measures (point
values, line, surface and volume integrals, weighted ones) are the elements these pairings are the inner products of;
backend.py realises Sobolev spaces on them (a kernel part sum_i c_i k_mu_i plus a formula part).

kernels.py           the Matern (Bessel-potential) kernels and their integrals against points, segments and polygons
atoms.py             measures decomposed into atoms (points, segments, patches, spheres, weighted pieces)
pairs.py             the one table of atom-atom energies and atom-point values, indexed by the kinds of the two atoms
surfaces.py          potentials and energies of patches, spheres and circles
weighted*.py         weighted pieces w dsigma (segments, curves, spheres, volumes, solids, contacts; grid and spectral
                     node rules), segment_partition.py their adaptive panels
energy.py            the exact engine (every pair of atoms); hmatrix.py the hierarchical engine (far fields of low
                     rank); spectral.py, spectral_terms.py, spectral_images.py the spectral engine
green.py, box_form.py  the kernels and form inner products of the Sobolev constructions on intervals and boxes
                     (Green functions with boundary corrections)
backend.py           RKHSBackend: Sobolev realisations through the kernel (SobolevRealisation, RepresenterCombinations)

The engine is chosen in realisations/registry.py (energy_engine).
"""
