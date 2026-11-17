"""The root of the hierarchy: sets (plan 067 P2; architecture review 2026-11-13, layer L1; decision Q3).

    Set, EmptySet                  sets/base.py: member(x) -> Truth, `x in S`, &, |, one empty set per ambient
    Region, RealSpace, RealLine, R sets/regions.py: subsets of R^d (dim, volume, contains(points) on (..., d) arrays)
    Interval                       sets/intervals.py
    FiniteSet                      sets/finite.py
    intersections, restrict        S & T exactly (and S cap T up to null sets for L2 pairings); mu|_D for measures
    Subset, PredicateSubset        sets/subsets.py: subsets of a space (states of kind set); the convex ones with a
                                   support function are pygeoinf0.convex.ConvexSet (plan 067 P3)

Domains (domains/) are Regions with a chart; vector spaces (spaces/base.py) are Sets.
"""
from .base import EmptySet, Intersection, Set, Union
from .regions import R, RealLine, RealSpace, Region, RegionIntersection, RegionUnion
from .finite import FiniteSet
from .intervals import Interval
from .subsets import PredicateSubset, Subset, SubsetIntersection, SubsetUnion

__all__ = ["Set", "EmptySet", "Intersection", "Union", "Region", "RealSpace", "RealLine", "R", "RegionIntersection",
           "RegionUnion", "FiniteSet", "Interval", "Subset", "PredicateSubset", "SubsetIntersection", "SubsetUnion"]
