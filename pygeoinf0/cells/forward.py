"""Forward cells F4-F15 (EXP-028 DESIGN.md 2.2): a state on X pushed through a relation R: X -> Y. The thesis's
composition table (el:575, el:584) for the set and measure rows, with the point rows for an element given as the
asserted point; the ordering row (an extension, [M]) is the inf-projection of a cost. The prior "none" is the whole
space, a set (plan 067 P4: NoPrior(X) is Whole(X)): its push is the set row's (F7-F9), and the former rows F1-F3 of
"none" are gone.

Cells choose by the kinds of the state and the relation, then by structure (the operator's class, a claim on record),
never by class names (review D8)."""
import math

from ..core.claims import DERIVED, THEOREM, Claim
from ..core.logic import Truth
from ..core.kinds import MEASURE, ORDERING, POINT, SET
from ..operators.linear import LinearOperator
from .registry import FORWARD, cell, refusal, unsupported


def _linear_op(state, R):
    """A LinearOperator: its row is the linear one, and the cell's hypothesis "linear" is then read from its claims (a
    subclass that states nothing is refused with HypothesisUnmet, not computed by the linear rule; plan 065 A3)."""
    return isinstance(R, LinearOperator)


def _linear_map(state, R):
    """A map that is not a LinearOperator but whose linearity is on record (e.g. pg.Map(..., linear="why"))."""
    from ..operators.claim_rules import status
    return not isinstance(R, LinearOperator) and status(R, "linear")[0] is Truth.TRUE


_AS_OPERATOR = ("its linearity is on record, but this cell needs the operator's adjoint: give it as "
                "pg.LinearOperator(X, Y, f, adjoint=...)")


def _additive_set(state, R):
    from ..relations.additive import AdditiveSet
    return isinstance(R, AdditiveSet) and isinstance(R.G, LinearOperator)


def _linear_relation(state, R):
    from ..relations.linear import LinearRelation
    return isinstance(R, LinearRelation) and R.G is not None


def _gaussian_kernel(state, R):
    from ..relations.kernels import AdditiveMarkovKernel
    return isinstance(R, AdditiveMarkovKernel)


def _gaussian(mu):
    from ..probability import is_gaussian
    return is_gaussian(mu)


def _whole(S):
    """S is the whole space (the prior "none")."""
    from ..convex.subspaces import Whole
    return isinstance(S, Whole)


# ------------------------------------------------------------------------------------------------ point
@cell(FORWARD, "F4", POINT, POINT, "point: f(x)")
def _f4(x, R):
    return R(x)


@cell(FORWARD, "F5", POINT, SET, "set: R(x)")
def _f5(x, R):
    return R(x)


@cell(FORWARD, "F6", POINT, MEASURE, "measure: L(x, .)", exists_as="noise + G(x)")
def _f6(x, R):
    return R(x)


# ------------------------------------------------------------------------------------------------ set
@cell(FORWARD, "F7", SET, POINT, "set: T(S); support sigma_S(T* q) (S = X: ran T)", requires=[("relation", "linear")],
      when=_linear_op, exists_as="BGP's property step [LAND EXP-014]")
def _f7(S, T):
    from ..convex.algebra import image
    return image(T, S)


unsupported(FORWARD, "F7", SET, POINT, "the image of a set under a map given by a rule: " + _AS_OPERATOR,
            when=_linear_map)
unsupported(FORWARD, "F7", SET, POINT, "the image of a set under a map not claimed linear is not computed")


@cell(FORWARD, "F8", SET, SET, "set: G(S) + N (Minkowski sum; S = X: ran G + N)", requires=[("G", "linear")], when=_additive_set)
def _f8(S, R):
    from ..convex.algebra import image, minkowski
    return minkowski(image(R.G, S), R.noise)


@cell(FORWARD, "F8", SET, SET, "set: G(S) + mul R (S = X: ran R)", when=_linear_relation)
def _f8_linear(S, R):
    from ..convex.algebra import image, minkowski
    if _whole(S):
        return R.ran                                    # ran R = G(X) + mul R, the subspace (exact)
    return minkowski(image(R.G, S), R.mul)


unsupported(FORWARD, "F8", SET, SET, "the union over a set of the values of a general relation is not computed")
refusal(FORWARD, "F9", SET, MEASURE, "a set (the whole space included: the prior 'none') pushed through a kernel is a "
        "family of measures {L(x, .) : x in S}, not a measure: the set gives no mixing weights (el:575)",
        needs=["a prior measure on the model space", "a measure on the set (softening, a choice)"])


# ------------------------------------------------------------------------------------------------ measure
@cell(FORWARD, "F10", MEASURE, POINT, "measure: T#mu (Gaussian: N(T m, T C T*))", requires=[("relation", "linear")],
      when=lambda mu, T: _gaussian(mu) and _linear_op(mu, T), exists_as="G(mu), mu.pushforward(A), mu.std(u)")
def _f10(mu, T):
    from ..probability import CylindricalGaussian
    out = mu.pushforward(T)
    if isinstance(mu, CylindricalGaussian):             # its own kind check (decision Q11, plan 067 P5)
        out.claims = type(out.claims)(list(out.claims) + [Claim(
            "the prior is cylindrical: a Gaussian measure on the finite-dimensional codomain, statements about "
            "bounded functionals of the model only (EXP-028 DESIGN 1.9)", THEOREM, "inv:659-669, inf:1290")])
    return out


unsupported(FORWARD, "F10", MEASURE, POINT, "push-forwards of measures that are not Gaussian are not implemented",
            when=lambda mu, T: not _gaussian(mu))
unsupported(FORWARD, "F10", MEASURE, POINT, "the push-forward of a measure under a map given by a rule: "
            + _AS_OPERATOR, when=_linear_map)
unsupported(FORWARD, "F10", MEASURE, POINT, "push-forwards of measures under maps not claimed linear (samples only) "
            "are not implemented")
refusal(FORWARD, "F11", MEASURE, SET, "a measure pushed through a set-valued relation is a random set: it gives no "
        "probabilities to points inside a realised set (el:575, el:584); that kind is not implemented")


@cell(FORWARD, "F12", MEASURE, MEASURE, "measure: the mixture (predictive measure) G#mu * noise",
      requires=[("G", "linear"), ("relation", "independent noise")], when=_gaussian_kernel,
      exists_as="G(mu) + noise")
def _f12(mu, R):
    G, noise = R.effective
    return G(mu) + noise


unsupported(FORWARD, "F12", MEASURE, MEASURE, "mixtures through general kernels are not implemented")


# ------------------------------------------------------------------------------------------------ ordering
def _norm_ordering(J, R):
    return getattr(J, "is_norm", False) and J.feasible is None and isinstance(R, LinearOperator)


@cell(FORWARD, "F13", ORDERING, POINT, "ordering: inf{J(x) : T x = p}", requires=[("relation", "linear")],
      when=_norm_ordering)
def _f13(J, T):
    from ..solvers import Eigh
    from ..solvers.direct import rounding_claim
    from ..states.base import Ordering
    P = T.codomain
    p0 = T(J.centre) if J.centre is not None else P.zero
    A = (T @ T.adjoint).inverse(Eigh("rounding"))                             # (T T*)^+

    def cost(p):
        r = p - p0
        z = A(r)
        res = r - (T @ T.adjoint)(z)                                # r in ran T?
        if P.norm(res) > 1e-10 * max(P.norm(r), 1e-300):
            return math.inf
        return 0.5 * float(P.inner(z, r))
    return Ordering(P, cost, name=f"inf {{{J.name} : {T.name} x = p}}", argmin=lambda: p0,
                    claims=[Claim("the inf-projection of the norm cost: (1/2) <(T T*)^+ (p - T c), p - T c>", DERIVED,
                                  "EXP-028 DESIGN 2.5 (b)"), rounding_claim(A, "(T T*)^+"),
                            Claim("an extension: the thesis's forward table has no ordering row", THEOREM, "[M]")])


unsupported(FORWARD, "F13", ORDERING, POINT, "inf-projections of other costs or of updated orderings are not "
            "implemented")
unsupported(FORWARD, "F14", ORDERING, SET, "the inf-projection through a set-valued relation is not implemented yet")
refusal(FORWARD, "F15", ORDERING, MEASURE, "an ordering supplies no probabilities and a kernel no ordering: combining "
        "them (e.g. by an expected cost) is a further choice")

