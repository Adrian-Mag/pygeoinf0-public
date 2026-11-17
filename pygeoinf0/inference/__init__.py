"""Inference problems with the seven elements, methods as recipes over the cells, and calibration statements (plan 008);
data-space decompositions shared by the methods (plan 009 G3), whose engine is internal (solvers/dataspace/, plan 070
P3): pg.decompose is its public route (inference/dataspace.py)."""
from .calibration import Bayesian, Calibration, Deterministic, Frequentist, Hardened, Unstated
from .dataspace import decompose
from .methods import METHODS, bgp, csola, least_squares, minimum_norm, ocsola, pli, sola
from .problem import Problem
from .sola_slabs import slab_family

__all__ = ["Problem", "slab_family", "METHODS", "bgp", "csola", "least_squares", "minimum_norm", "ocsola", "pli", "sola",
           "Calibration", "Deterministic", "Frequentist", "Bayesian", "Hardened", "Unstated",
           "decompose"]
