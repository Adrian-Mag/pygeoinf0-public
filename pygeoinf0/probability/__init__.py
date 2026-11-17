"""Probability: Gaussian measures on Hilbert spaces (gaussian.py) and covariance operators with trace bookkeeping
(covariance.py), weighted chi-square laws (chisquare.py) and hardening (hardening.py). Probability measures share the
Measure root (pygeoinf0.measures): ProbabilityMeasure and DiscreteMeasure (measures.py); GaussianMeasure, Dirac and the
CylindricalGaussian, which is not a measure on its space (gaussian.py; plan 067 P5). Named priors, chosen by the caller
(priors.py; plan 069 P6, decision Q2): legendre_decay, random_features, cylindrical; no space picks one."""
from .chisquare import WeightedChiSquare
from .covariance import Covariance
from .gaussian import GaussianMeasure
from .gaussian import CylindricalGaussian, Dirac, is_gaussian
from .measures import DiscreteMeasure, ProbabilityMeasure
from .priors import cylindrical, legendre_decay, random_features
from . import circular              # circular(X, covariance, mean): complex coordinates (named; plan 069 P6)

__all__ = ["GaussianMeasure", "Covariance", "WeightedChiSquare"]
__all__ += ["ProbabilityMeasure", "DiscreteMeasure", "Dirac", "CylindricalGaussian"]       # plan 067 P5
__all__ += ["is_gaussian"]                                                           # plan 067 P4
__all__ += ["legendre_decay", "random_features", "cylindrical"]                      # plan 069 P6
