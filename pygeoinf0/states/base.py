"""States of knowledge on a space (plan 008 decision 4; EXP-028 DESIGN.md 1.9; el:458-485; plan 067 P4: the State
protocol, decision Q12 and the author's amendment 3). A state is what the prior is before the datum and what the update
gives after it. Its kind is one of three (core/kinds.py), declared by its class:

    set        a subset of the space (pygeoinf0.sets.Subset; the convex ones with a support function in
               pygeoinf0.convex): an admissible set before the datum, a feasible set after it. The prior "none" is the
               whole space: NoPrior(X) is Whole(X) (el:460, el:592, inv:435-439)
    measure    a probability measure (pygeoinf0.probability: ProbabilityMeasure, GaussianMeasure, Dirac,
               DiscreteMeasure) or a CylindricalGaussian, which speaks about bounded functionals only (decision Q11);
               a base measure of a domain (pg.DiracMeasure, pg.LebesgueMeasure) is not a state
    ordering   Ordering(X, cost, feasible): a preorder represented by a cost J (lower is preferred) together with the set
               it is restricted to, because +inf alone cannot tell "excluded" from "infinitely costly" (inv:318).
               Declared, not developed (amendment 3): the class and its cells stay as they are

A point is not a kind of state: it is the singleton set {x} or the Dirac measure delta_x (states/conversions.py). The
table (pygeoinf0.cells) still accepts an element x where a state is expected, as the asserted point (el:481); a
selected model is a summary, not a state: it keeps no grading (el:474, el:533).

State is the protocol: a state has a space, a kind its class declares (one of core.kinds.STATE_KINDS) and claims.
isinstance(obj, State) and state_kind(obj) read the declared kind, never an instance attribute or other attributes.
"""
import math

from ..core.claims import THEOREM, Claim, Claims
from ..core.errors import Refused, Unsupported
from ..core.kinds import ORDERING, POINT, STATE_KINDS, declared_kind, is_state
from ..core.relation import Relation
from ..core.logic import Truth


class _Protocol(type):
    def __instancecheck__(cls, obj):
        return is_state(obj)


class State(metaclass=_Protocol):
    """The protocol of a state of knowledge (module docstring): obj.space, the kind obj's class declares (SET, MEASURE
    or ORDERING: core.kinds.STATE_KINDS) and obj.claims. isinstance(obj, State) checks it; no class inherits from it,
    since the states live in the layers below (sets, convex, probability)."""

    kinds = STATE_KINDS


def NoPrior(space):
    """The prior "none": no assertion beyond the model space itself (el:460), which is the whole space as a set:
    Whole(space) (the author's amendment 3), so the set cells give its answer by the one route of a set prior."""
    from ..convex.subspaces import Whole
    return Whole(space)


class Ordering:
    """A preorder on a space represented by a cost J (lower is preferred), restricted to a feasible set (module
    docstring). Ordering.norm(X, centre) is J = (1/2) ||x - c||^2; the cells give updated orderings whose summary()
    is the minimiser (a summary, no guarantee)."""

    kind = ORDERING

    def __init__(self, space, cost=None, feasible=None, name="J", claims=(), centre=None, argmin=None):
        self.space, self._cost, self.name, self.centre = space, cost, name, centre
        self.feasible = feasible
        self._argmin = argmin
        self._claims = list(claims)

    @classmethod
    def norm(cls, space, centre=None, name=None):
        """J(x) = (1/2) ||x - c||^2: the ordering behind minimum-norm models (inv:51-54)."""
        c = centre

        def cost(x):
            return 0.5 * space.norm(x if c is None else x - c) ** 2
        out = cls(space, cost, name=name or ("½‖x‖²" if c is None else "½‖x - c‖²"), centre=c,
                  claims=[Claim("a cost representing an ordering (its numerical values are a further choice)", THEOREM,
                                "el:466-476")])
        out.is_norm = True
        return out

    is_norm = False

    @property
    def claims(self):
        return Claims(self._claims)

    def cost(self, x):
        """J(x), +inf outside the feasible set."""
        if self.feasible is not None and self.feasible.member(x) is Truth.FALSE:
            return math.inf
        if self._cost is None:
            raise Unsupported(f"{self.name} has no numerical cost (a preorder without a representation)")
        return float(self._cost(x))

    def compare(self, a, b):
        """Truth of "a is at least as preferred as b"."""
        return Truth.of(self.cost(a) <= self.cost(b))

    def summary(self):
        """The minimiser of the cost on the feasible set: a summary of the ordering, not a state (el:474)."""
        if self._argmin is None:
            raise Unsupported(f"no minimiser is computed for {self.name}; it summarises the ordering only (el:474)")
        return self._argmin()

    def __repr__(self):
        where = "" if self.feasible is None else f" on {self.feasible.name}"
        return f"Ordering {self.name}{where} on {self.space!r}"


_A_POINT = ("a point is not a kind of state: state it as the singleton set {x} (pg.states.as_set(x)) or the Dirac "
            "measure delta_x (pg.states.as_measure(x), pg.Dirac(x)); push(x, R) and update(x, R, d) accept an element "
            "as the asserted point")


def is_element(obj):
    """Whether obj is an element of a space (its class declares the kind point; not a relation, not a family)."""
    return declared_kind(obj) is POINT and hasattr(obj, "space") and not isinstance(obj, Relation)


def state_kind(obj):
    """The kind of a state: set, measure or ordering, as its class declares it (module docstring)."""
    if obj is None:
        raise Refused("the prior is not stated", needs=["pg.NoPrior(X) for the prior 'none' (the whole space)",
                                                        "a set, a measure, an ordering or an asserted point"])
    if is_state(obj):
        return declared_kind(obj)
    if is_element(obj):
        raise TypeError(f"{getattr(obj, 'name', 'the element')} is an element of {obj.space!r}: " + _A_POINT)
    from ..measures import Measure
    if isinstance(obj, Measure):
        raise TypeError(f"{obj!r} is a measure on the set {obj.set!r}, a base measure of a domain (it defines L2 and "
                        "functionals), not a state of knowledge on a model space: a point prior is pg.Dirac(x) for an "
                        "element x of the space (or the singleton set of x), a probability prior a ProbabilityMeasure "
                        "(pg.GaussianMeasure, pg.DiscreteMeasure)")
    if hasattr(obj, "space") and hasattr(obj, "__len__") and hasattr(obj, "_memo"):
        raise TypeError(f"a family of {len(obj)} elements is not a state: push each member (e.g. G(F) applies G to "
                        "all of them), and a set of models is stated as a set (e.g. pg.Span(F) or a convex set)")
    raise TypeError(f"{type(obj).__name__} is not a state: a state's class declares its kind (set, measure or "
                    "ordering; pygeoinf0.states)")
