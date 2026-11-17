"""Convex analysis on a space (plan 008; plan 067 P3 and the author's amendment 1): the convex sets with a support
function, the computable subfamily of the subsets of a space (pygeoinf0.sets: Subset, PredicateSubset), the convex set
algebra, linear and closed subspaces, and the feasible sets of set-valued inversions with their dual certificates.
Every support is an Enclosure with certified ends (convex/sets.py ConvexSet)."""
from .algebra import CartesianProduct, Image, Intersection, Minkowski, Scale, Translate, Union, image, minkowski
from .sets import AffineSubset, ConvexSet, CoordinateBox, Ellipsoid, EmptySet, NormBall, Polytope, Slab, singleton
from .subspaces import ClosedSubspace, Complement, LinearSubspace, Projector, Span, Whole, Zero

__all__ = ["CartesianProduct", "Image", "Intersection", "Minkowski", "Scale", "Translate", "Union", "image",
           "minkowski", "AffineSubset", "ConvexSet", "CoordinateBox", "Ellipsoid", "EmptySet", "NormBall", "Polytope",
           "Slab", "singleton", "LinearSubspace", "ClosedSubspace", "Whole", "Zero", "Span",
           "Complement", "Projector"]
