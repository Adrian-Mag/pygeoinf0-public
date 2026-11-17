"""Measures (plan 067 P5; architecture review 2026-11-13, layer L2 and B2): one root, Measure(set, sigma_algebra), with
claims (finite, probability, signed), mass, integration and restriction.

    Measure, SignedMeasure          measures/base.py: the root; c_1 mu_1 + ... + c_m mu_m (signed is a claim)
    LebesgueMeasure, DensityMeasure measures/lebesgue.py: Hausdorff measure on a Region, w dsigma, w_j ds on a polyline
      PieceDensity
    CountingMeasure, DiracMeasure   measures/points.py: weighted points, the unit point mass of a domain
    as_measure(s), measure_of, ...  measures/convert.py: reading measures from sets and points; mass, variation
    Weight, as_weight               measures/weights.py: the densities w of weighted measures
    Density, density_against        measures/densities.py: Radon-Nikodym derivatives with their local orders
    restrict                        measures/restrict.py: mu|_D

Probability measures on spaces (GaussianMeasure, Dirac, DiscreteMeasure; pygeoinf0.probability) are Measures too: they
share this root, not the classes of the base measures of a domain. A cylindrical Gaussian is not a Measure on its space
(pygeoinf0.probability.CylindricalGaussian, decision Q11).
"""
from .base import Measure, SignedMeasure, terms_of
from .lebesgue import (DensityMeasure, LebesgueMeasure, PieceDensity, carries_hausdorff, check_space_measure,
                       constant_density, space_factor)
from .points import CountingMeasure, DiracMeasure
from .convert import as_measure, as_measures, measure_dim, measure_of, total_mass, total_variation

__all__ = ["Measure", "SignedMeasure", "LebesgueMeasure", "DensityMeasure", "PieceDensity", "CountingMeasure",
           "DiracMeasure", "as_measure", "as_measures", "measure_of", "measure_dim", "total_mass", "total_variation",
           "terms_of", "carries_hausdorff", "check_space_measure", "constant_density", "space_factor"]
