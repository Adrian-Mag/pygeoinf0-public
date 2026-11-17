"""Space identity without realisation (plan 069 P1; the architecture review's C1; decision Q4).

A function space is the same space whatever realises its elements. X == Y compares what defines the space: the measure
up to null sets and the norm construction (p for L^p, the construction of a Sobolev space, the summands of a direct sum),
never the backend. The realisation belongs to the elements: an element is made in a space object whose backend realises
it (X.realisation names that realisation), and X.realised(r) is the same space (X.realised(r) == X) with its elements
realised by r.

Elements realised differently are elements of one space (X.member(y) is TRUE), but their realisations are coefficients
of different bases, so nothing combines them numerically without a conversion. Every operation that reads a
realisation of an element made elsewhere (arithmetic, inner products and norms, operator application and composition,
X.element(y)) asks require(), which raises RealisationMismatch naming both realisations, never a SpaceMismatch and never
silently. No conversion is made here (a refit with its error stated is an open decision of plan 069 P1).

    realisation_of(X)      the realisation of X's elements: a name (cells(n, digest) for a cell partition), the tuple
                           of the summands' for a direct sum, the primal's for a dual, None for coordinate spaces
    same_realisation(X, Y) whether elements of X and of Y can be combined numerically as they are
    identical(X, Y)        X == Y and realised alike: the identity of a realised space object (the space equality before
                           plan 069), for code that takes a numerical shortcut when two space objects agree
    require(S, X, message, what)
                           SpaceMismatch(message) when S != X; RealisationMismatch when S == X realised differently
    require_realised(S, X, what)
                           only the second half: RealisationMismatch when S == X realised differently (after a check
                           of equality that keeps its own exception)
    mismatch(S, X, what)   that RealisationMismatch
"""
from ..core.errors import RealisationMismatch, SpaceMismatch


def realisation_of(X):
    """The realisation of the elements of the space X (module docstring)."""
    r = getattr(X, "realisation", None)
    if isinstance(r, str):
        return r
    summands = getattr(X, "summands", None)
    if summands is not None:
        return tuple(realisation_of(H) for H in summands)
    primal = getattr(X, "primal", None)
    if primal is not None:
        return realisation_of(primal)
    return None


def same_realisation(X, Y):
    return X is Y or realisation_of(X) == realisation_of(Y)


def identical(X, Y):
    """X == Y and realised alike: the old identity of a realised space object."""
    return X is Y or (X == Y and same_realisation(X, Y))


def _name(r):
    if isinstance(r, tuple):
        return "(" + ", ".join(_name(q) for q in r) + ")"
    return "unrealised" if r is None else str(r)


def mismatch(S, X, what=None):
    """The refusal to combine an element of S with one of X, where S == X is realised differently."""
    a, b = _name(realisation_of(S)), _name(realisation_of(X))
    head = f"{what}: " if what else ""
    return RealisationMismatch(
        f"{head}one space {X}, realised two ways: {a} and {b}. A realisation is coefficients in its own basis, and an "
        "element realised one way is not combined with one realised the other without a conversion",
        needs=(f"make both in one realisation: X.realised(r) is the space X with its elements realised by r ({a} or {b})",
               "a conversion of one element (a refit, its error stated): not provided (plan 069 P1, an open decision)"))


def require_realised(S, X, what=None):
    """RealisationMismatch when the spaces S and X are equal but realised differently; nothing otherwise."""
    if S is not X and S == X and not same_realisation(S, X):
        raise mismatch(S, X, what)


def require(S, X, message, what=None):
    """Refuse an element of the space S where X is expected: SpaceMismatch(message) when S != X, RealisationMismatch
    (about what, e.g. the element's name) when S == X but realised differently; nothing when they are identical."""
    if S is X:
        return
    if S != X:
        raise SpaceMismatch(message)
    if not same_realisation(S, X):
        raise mismatch(S, X, what)
