"""Operators: general linear operators (linear.py), stacks of functionals given by kernels (functionals.py) and
block operators (blocks.py: block-diagonal operators between direct sums, the column operator x -> (G_k x)_k) and the
whitened coordinates of a noise covariance (whitening.py: D.whitened(C))."""
from .blocks import BlockDiagonal, ColumnOperator
from .functionals import Functionals, FunctionalsAdjoint, Integrals
from .inclusion import Inclusion
from .inverses import Inverse, MinimumNorm, PseudoInverse
from .linear import Composition, Diagonal, FiniteRank, Identity, LinearOperator, Matrix, Sum
from .sparse import KroneckerMatrix, SparseMatrix

__all__ = ["LinearOperator", "Composition", "Sum", "Identity", "Matrix", "Diagonal", "FiniteRank", "SparseMatrix", "KroneckerMatrix", "Inverse", "PseudoInverse", "MinimumNorm", "Functionals",
           "FunctionalsAdjoint", "Integrals", "Inclusion", "BlockDiagonal", "ColumnOperator"]
