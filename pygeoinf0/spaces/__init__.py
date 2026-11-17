"""Spaces (layer L3 of the architecture review, plan 068 P1): vector -> normed -> Banach -> Hilbert, each level with its
contract, the concrete spaces, their elements and the dual.

    base.py           VectorSpace(Set) (field, zero, linear combinations), NormedSpace (norm, dual(), functional()),
                      BanachSpace, HilbertSpace (inner, riesz); L2, Lp; Euclidean(n, field="R" | "C") (both
                      CoordinateSpaces; C^n is not an instance of Euclidean, so real-coordinate routes refuse it)
    dual.py           DualSpace X' of a normed space, its one element type Functional (from_representer,
                      from_measure, from_operator), the Frechet gradient, and each space's boundedness rule with
                      the one trace rule (plan 068 P2: absorbs LinearForm, MeasureForm, Covector, boundedness.py)
    elements.py       Function and Vector (points), Family and VectorFamily (batches)
    identity.py       space identity without realisation (plan 069 P1, decision Q4): X == Y by the measure and the norm,
                      X.realised(r), and require(): mixing realisations is refused with RealisationMismatch
    metrics.py        the metric of a weighted R^n or C^n; labels.py: labelled coordinates of R^n
    sobolev.py        Sobolev spaces H^s; constructions.py: their constructions (SobolevForm, RestrictionSpace, Spectral:
                      different Hilbert spaces with equivalent norms; a form's operator, FormOperator, is in
                      operators/unbounded.py)
    continuous.py     C([a, b]) with the sup norm (a Banach space: no inner, no riesz)
    data.py           DataSpace: the data space, a plain R^n with labelled coordinates; D.whitened(C) its whitened
                      coordinates (plan 068 P5, Q1 and Q9)
    direct_sum.py     DirectSum, and euclidean_sum: a direct sum of Euclideans is a Euclidean with block labels;
                      complex.py: the complex structure of C^n, Complexified, ComplexMatrix
    piecewise.py      PiecewiseSpace(partition, fields, glued | broken): fields on the pieces of a partition (plan 069
                      P7: it replaced Layered and OnPieces); gluing.py: extension by zero, traces on the seams and the
                      interface conditions of the glued space
    representers.py, membership.py: the Riesz representers of measures on L2, and the regularity rules of
                      membership
    harmonic.py       the real spherical harmonics Y_lj of the sphere (the packed layout, the associated Legendre
                      functions, values at points), HarmonicConvention (another basis by its phase and scale, exactly),
                      P_l and Funk-Hecke coefficients of zonal weights (plan 069 P4: moved from realisations/)

pg.spaces is this package (the catalogue of spaces on domains is pg.catalogue: pg.catalogue.how, .table, .probe).
"""
from .base import BanachSpace, CoordinateSpace, Euclidean, HilbertSpace, L2, Lp, LpSpace, NormedSpace, VectorSpace
from .dual import DualSpace, Functional, gradient
from .elements import Family, Function, Vector, VectorFamily
from .sobolev import Sobolev, SobolevSpace
from .constructions import SobolevForm
from .continuous import Continuous, ContinuousSpace
from .direct_sum import DirectSum, euclidean_sum
from .data import DataSpace
from .piecewise import PiecewiseSpace
from .complex import Complexified, ComplexMatrix
from .harmonic import HarmonicConvention

__all__ = ["VectorSpace", "NormedSpace", "BanachSpace", "HilbertSpace", "DualSpace", "Functional", "gradient", "L2", "Lp", "LpSpace", "Euclidean",
           "CoordinateSpace",
           "Function", "Vector", "Family", "VectorFamily", "Sobolev", "SobolevSpace", "SobolevForm", "Continuous",
           "ContinuousSpace", "DirectSum", "euclidean_sum", "DataSpace", "PiecewiseSpace", "Complexified", "ComplexMatrix",
           "HarmonicConvention"]
