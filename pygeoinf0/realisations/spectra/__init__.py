"""Spectra that are realisations (plan 069 P4; the review's C4 correction: the Galerkin and Bessel spectra legitimately
are realisations): eigenvalues and eigenfunctions of self-adjoint operators, closed-form or bracketed, and the
covariances of functional calculus built on them.

laplace_spectra.py       the spectra of -Delta on intervals, boxes, tori, circles and spheres (closed forms and the
                         unions of separation of variables), chosen in registry.laplacian_spectrum
bessel.py                the per-degree radial Laplacian on balls and shells: spherical Bessel spectra (RadialBessel)
sl_galerkin.py           eigenvalue brackets of regular Sturm-Liouville operators with interfaces (GalerkinSpectrum)
laplacian_covariance.py  Bessel-potential covariances c (1 + kappa^-2 L)^(-q/2) of the Laplacian, degree by degree
"""
