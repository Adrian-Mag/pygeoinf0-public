"""pygeoinf0: continuum-first inference. Spaces, measures and operators are mathematical objects; their numerics are
explicit, replaceable realisations.

    import numpy as np, pygeoinf0 as pg
    I  = pg.Interval("[0, 1]")
    mu = pg.LebesgueMeasure(I)
    M  = pg.L2(mu)                         # or pg.L2("[0, 1]")
    f  = M(np.sin);  g = np.cos(3 * M.x);  (f + g).plot()
    D  = pg.Euclidean(20)
    G  = pg.Functionals([np.exp(-((M.x - c) / 0.05) ** 2) for c in np.linspace(0.05, 0.95, 20)], into=D)
    m_dagger = G.adjoint((G @ G.adjoint).inv(G(f)))

    X  = pg.Continuous("[0, 1]")           # C([0, 1]): functions with values; the sup norm; functionals are measures
    X(np.sin).norm(), X.evaluation(0.3)(X(np.sin)), pg.Integrals(X, [0.2, "[0.1, 0.5]"])(X(np.cos))

    H  = pg.Sobolev(pg.Disk(1.0), 1.5, length_scale=0.1)          # H^{3/2}_l: rays are bounded functionals
    G  = pg.Functionals(H.riesz([pg.Segment(p, q) for p, q in pairs]), into=pg.Euclidean(n, covariance=sigma**2))
    m  = G.adjoint((G @ G.adjoint + alpha * pg.Identity(G.codomain)).inv(d))

    The data space and the noise (Q1 as amended, plan 071 P4), two routes, one model:
    D  = pg.Euclidean(n);  R = pg.Additive(G, pg.GaussianMeasure(D, C))   # a plain D, the noise in the relation
    D  = pg.Euclidean(n, covariance=C);  R = pg.Additive(G, D.standard_gaussian())   # the whitened norm: N(0, C)
    pg.Problem(mdr=R, datum=d).run("sola", gamma=0.1)   # the same numbers by both; a point relation into a plain D
                                                        # states no noise (pg.NoiseUnstated where one is needed)

Self-contained: NumPy, SciPy (and matplotlib for plots).
"""
# The namespace by mathematical role (plan 070 P6; review section 2 "Namespace", E4): each group opens with its roots and
# ends with its subpackage. The periphery (optimisers, port, plotting, catalogue) is reached through its subpackage only:
# the modules are pg names, their contents are not.
from .settings import settings
from .core import errors
from .core.claims import Claim, Claims, assert_claim
from .core.errors import (AveragingWarning, ExternalError, GenerationError, HypothesisUnmet, NoiseUnstated,
                          NotACovariance, NotATiling, NotContinuous, PointEvaluationUndefined, PriorUnstated,
                          RealisationError, RealisationMismatch, RealisationWarning, Refused, SpaceMismatch)
from .core.logic import Truth
from . import sets, domains
from .sets import FiniteSet, Interval, R, RealLine, Region, Set, Subset
from .domains import (Arc, Ball, Box, Circle, Curve, Cylinder, Disk, Domain, LayeredBall, Partition, Patch, Polyline,
                      Segment, Shell, Simplex, Simplicial, Sphere, Surface, Torus, clip)
from . import measures
from .measures import (CountingMeasure, DensityMeasure, DiracMeasure, LebesgueMeasure, Measure, PieceDensity,
                       SignedMeasure)
from . import spaces
from .spaces import (BanachSpace, Complexified, Continuous, ContinuousSpace, DataSpace, DirectSum, DualSpace, Euclidean,
                     Family, Function, Functional, HarmonicConvention, HilbertSpace, L2, Lp, LpSpace, NormedSpace,
                     PiecewiseSpace, Sobolev, SobolevForm, Vector, VectorFamily, VectorSpace, gradient)
from .core.tabulated import Tabulated
from . import operators
from .operators import (BlockDiagonal, Composition, Diagonal, FiniteRank, Functionals, Identity, Inclusion, Integrals,
                        Inverse, KroneckerMatrix, LinearOperator, Matrix, SparseMatrix, Sum)
from .spaces import ComplexMatrix
from .operators.band import BandProjection
from .operators.galerkin import Discretisation
from .operators.multiplication import Multiplication
from .operators.pieces import Restrict
from .operators.unbounded import Laplacian
from . import realisations, solvers
from .solvers import CG, LU, BlockDirect, Cholesky, Direct, Eigh, KroneckerEigh
from . import convex
from .convex import (AffineSubset, Complement, ConvexSet, CoordinateBox, Ellipsoid, EmptySet, NormBall, Polytope, Slab,
                     Span, Whole)
from .convex.pointwise import PointwiseBox
from . import probability
from .probability import (Covariance, CylindricalGaussian, Dirac, DiscreteMeasure, GaussianMeasure, ProbabilityMeasure,
                          WeightedChiSquare)
from . import states
from .states import NoPrior, Ordering, State, harden, state_kind
from .convex.feasible import FeasibleSet
from . import relations
from .core.relation import Relation
from .relations import Additive, Exact, LinearRelation, Map, MarkovKernel, SetValued, stack
from . import cells
from .cells import push, update
from . import properties
from .properties import CapAverages, RotatedFamily, harmonic_target
from . import inference
from .inference import (Bayesian, Deterministic, Frequentist, Problem, bgp, csola, decompose, least_squares,
                        minimum_norm, ocsola, pli, sola)
from . import meta
from . import catalogue, optimisers, plotting, port  # the periphery: pg.port.External, pg.plotting.slices, ...

__version__ = "0.3.0"

GROUPS = (
    ("sets and regions", ("Set", "Region", "Domain", "Interval", "FiniteSet", "R", "RealLine", "Box", "Disk",
                          "Cylinder", "Ball", "Shell", "LayeredBall", "Circle", "Sphere", "Torus", "Segment", "Patch",
                          "Polyline", "Arc", "Curve", "Surface", "Simplex", "Simplicial", "Partition", "clip", "sets",
                          "domains")),
    ("measures", ("Measure", "SignedMeasure", "LebesgueMeasure", "DensityMeasure", "PieceDensity", "CountingMeasure",
                  "DiracMeasure", "measures")),
    ("spaces and duals", ("VectorSpace", "NormedSpace", "BanachSpace", "HilbertSpace", "DualSpace", "Functional",
                          "Euclidean", "DataSpace", "L2", "Lp", "LpSpace", "Continuous", "ContinuousSpace", "Sobolev",
                          "SobolevForm", "DirectSum", "PiecewiseSpace", "Complexified", "HarmonicConvention",
                          "Function", "Family", "Vector", "VectorFamily", "Tabulated", "gradient", "spaces")),
    ("operators", ("LinearOperator", "Identity", "Matrix", "Diagonal", "FiniteRank", "SparseMatrix", "KroneckerMatrix",
                   "ComplexMatrix", "Composition", "Sum", "BlockDiagonal", "Inverse", "Functionals", "Integrals",
                   "Inclusion", "Multiplication", "BandProjection", "Laplacian", "Restrict", "Discretisation",
                   "operators")),
    ("subsets", ("Subset", "ConvexSet", "Whole", "Span", "Complement", "AffineSubset", "NormBall", "Ellipsoid", "Slab",
                 "CoordinateBox", "Polytope", "PointwiseBox", "EmptySet", "convex")),
    ("probability", ("ProbabilityMeasure", "Covariance", "GaussianMeasure", "CylindricalGaussian", "Dirac",
                     "DiscreteMeasure", "WeightedChiSquare", "probability")),
    ("states", ("State", "NoPrior", "Ordering", "FeasibleSet", "state_kind", "harden", "states")),
    ("relations", ("Relation", "MarkovKernel", "Map", "SetValued", "LinearRelation", "Exact", "Additive", "stack",
                   "relations")),
    ("table", ("push", "update", "cells")),
    ("inference and methods", ("Problem", "Deterministic", "Frequentist", "Bayesian", "decompose", "minimum_norm",
                               "least_squares", "sola", "csola", "ocsola", "bgp", "pli", "inference")),
    ("properties", ("CapAverages", "RotatedFamily", "harmonic_target", "properties")),
    ("meta", ("meta",)),
    ("claims and refusals", ("Truth", "Claim", "Claims", "assert_claim", "Refused", "HypothesisUnmet", "SpaceMismatch",
                             "RealisationMismatch", "RealisationError", "RealisationWarning", "AveragingWarning",
                             "PointEvaluationUndefined", "NotContinuous", "NotATiling", "PriorUnstated",
                             "NoiseUnstated", "NotACovariance",
                             "ExternalError", "GenerationError", "errors")),
    ("numerics", ("settings", "Cholesky", "Eigh", "LU", "CG", "Direct", "KroneckerEigh", "BlockDirect", "solvers",
                  "realisations")),
    ("periphery", ("optimisers", "port", "plotting", "catalogue")),   # subpackages only: pg.port.External, ...
)
__all__ = [n for _, names in GROUPS for n in names]
