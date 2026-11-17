"""The claim rules of operators (plan 065 A3; out of relations/algebra.py in plan 070 P7, so that the operators and
the sets below relations read them without importing relations/):

    status(obj, key)    (holds, status) of the property key of obj, holds a Truth: its claims on record, else a rule
                        below, else (Truth.UNKNOWN, unknown). pg.relations.status is this function
    all_hold(results)   (Truth.TRUE, the weakest status) when every premise holds with a status on record, else unknown
    the rules           "linear": compositions, sums and inverses of linear operators, and a class that states why it
                        is linear (given its parts'); "bounded": linear maps between finite-dimensional spaces,
                        compositions, sums, columns of bounded blocks, the adjoint of a bounded map, the identity
"""
from ..core.claims import ORDER, THEOREM, UNKNOWN, Claims
from ..core.logic import Truth
from ..spaces.base import Euclidean
from .blocks import ColumnOperator
from .inverses import Inverse
from .linear import Composition, Identity, LinearOperator, Sum, class_linearity


def all_hold(results):
    """(Truth.TRUE, the weakest status) when every premise holds with a status on record, else (Truth.UNKNOWN,
    unknown): a failed or missing premise leaves the conclusion unknown, not false."""
    held = all(h is Truth.TRUE for h, _ in results)
    st = max((s for _, s in results), key=ORDER.index, default=UNKNOWN)
    return (Truth.TRUE, st) if held and st != UNKNOWN else (Truth.UNKNOWN, UNKNOWN)


def _linear(op):
    if not isinstance(op, LinearOperator):
        return Truth.UNKNOWN, UNKNOWN
    if isinstance(op, Composition):
        return all_hold([status(op.A, "linear"), status(op.B, "linear")])
    if isinstance(op, Sum):
        return all_hold([status(A, "linear") for _, A in op.terms])
    if isinstance(op, Inverse):
        return all_hold([status(op.A, "linear")])      # the inverse of a linear bijection is linear
    if class_linearity(op) is not None:                # the class that defines the action states why (plan 065 A3),
        parts = [P for P in getattr(op, "linear_parts", lambda: ())() if isinstance(P, LinearOperator)]
        return all_hold([(Truth.TRUE, THEOREM)] + [status(P, "linear") for P in parts])   # given its parts' linearity
    return Truth.UNKNOWN, UNKNOWN                               # a callable or a subclass without a claim on record


def _bounded(op):
    if not isinstance(op, LinearOperator):
        return Truth.UNKNOWN, UNKNOWN
    if isinstance(op.domain, Euclidean) and isinstance(op.codomain, Euclidean):
        return all_hold([status(op, "linear")])        # linear maps between finite-dimensional spaces (linear first)
    if isinstance(op, Composition):
        return all_hold([status(op.A, "bounded"), status(op.B, "bounded")])
    if isinstance(op, Sum):
        return all_hold([status(A, "bounded") for _, A in op.terms])
    if isinstance(op, ColumnOperator):                 # a column of bounded blocks is bounded
        return all_hold([status(G, "bounded") for G in op.blocks])
    from ..operators.functionals import FunctionalsAdjoint
    G = getattr(op, "G", None)                         # FunctionalsAdjoint: the adjoint of a bounded map
    if isinstance(op, FunctionalsAdjoint) and G is not None:
        return status(G, "bounded")
    if isinstance(op, Identity):
        return Truth.TRUE, THEOREM
    return Truth.UNKNOWN, UNKNOWN


_RULES = {"linear": _linear, "bounded": _bounded}


def status(obj, key):
    """(holds, status) of the property key of obj, holds a Truth: its claims, else a rule of the package, else
    (Truth.UNKNOWN, unknown)."""
    cl = getattr(obj, "_claims", None)
    if not isinstance(cl, list):
        cl = getattr(obj, "claims", ())
    held, st = Claims(cl).status(key)
    if st != UNKNOWN:
        return held, st
    rule = _RULES.get(key)
    return rule(obj) if rule else (Truth.UNKNOWN, UNKNOWN)
