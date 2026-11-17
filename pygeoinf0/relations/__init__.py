"""Relations between spaces (plan 008; core/relation.py has the kinds and the base class): maps, set-valued
relations, linear relations, Markov kernels, their constructors from a forward operator and noise, and their algebra;
selection of some of the data (noise families fitted by marginal likelihood are meta rules: pg.meta.NoiseFamily,
pg.meta.fit_noise, plan 070 P2). The subspaces that are the dom, ran, ker and mul of linear relations are convex sets:
pygeoinf0.convex.subspaces (plan 067 P3)."""
from .additive import Additive, AdditiveSet, Exact
from ..operators.claim_rules import status
from .algebra import as_kind, compose, converse, join, stack
from .kernels import AdditiveMarkovKernel, MarkovKernel, PosteriorKernel
from .linear import ClosedOperator, LinearRelation, OperatorDomain
from .maps import Map, SetValued
from .selection import CoordinateProjection, select

__all__ = ["Additive", "AdditiveSet", "Exact", "as_kind", "compose", "converse", "join", "stack",
           "status", "AdditiveMarkovKernel", "MarkovKernel", "PosteriorKernel", "LinearRelation", "ClosedOperator", "OperatorDomain", "Map", "SetValued",
           # plan 009 G9: selection of data rows (named by plan 068 P7)
           "CoordinateProjection", "select"]
