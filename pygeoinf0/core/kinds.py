"""The kinds of states and relations (plan 067 P4; decision Q12 and the author's amendment 3 of the architecture review,
2026-11-13). The kinds are closed: there is no registry.

    SET       a subset of a space
    MEASURE   a probability measure on a space (or a cylindrical Gaussian, which speaks about bounded functionals only)
    ORDERING  a preorder on a space (declared, not developed: states.Ordering and its cells stay as they are)
    POINT     an element of a space

A state of knowledge on a space has the kind set, measure or ordering (STATE_KINDS). "None" is not a kind: it is the
whole space, a set (pg.NoPrior(X) is pg.Whole(X)); and a point is not a kind of state: it is the singleton set {x} or
the Dirac measure delta_x (pygeoinf0.states.as_set, as_measure).

A relation R: X -> Y has the kind of its values R(x) (RELATION_KINDS): point (a map: R(x) is an element of Y), set
(R(x) is a subset of Y) or measure (R(x) is a probability measure on Y, a kernel). Elements of a space declare the kind
point too: it is the kind of a map's values, which push(x, R) and update(x, R, d) accept as the asserted point.

An object's kind is declared by its class (the class attribute `kind`), never read from an instance attribute or
inferred from other attributes: declared_kind(obj) reads it, and is_state(obj) says whether obj is a state (a class
that declares a state kind, with a space, and not a relation).
"""


class Kind:
    """A kind of state or relation: its name and what its values are."""

    __slots__ = ("name", "value")

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name


POINT = Kind("point", "an element of a space")
SET = Kind("set", "a subset of a space")
MEASURE = Kind("measure", "a probability measure on a space")
ORDERING = Kind("ordering", "a preorder on a space, represented by a cost (declared, not developed)")

STATE_KINDS = (SET, MEASURE, ORDERING)
RELATION_KINDS = (POINT, SET, MEASURE)


def declared_kind(obj):
    """The kind obj's class declares (its class attribute `kind`), or None. An instance attribute is not read."""
    k = getattr(type(obj), "kind", None)
    return k if isinstance(k, Kind) else None


def is_state(obj):
    """Whether obj is a state of knowledge: its class declares a state kind, it has a space and it is not a relation."""
    from .relation import Relation
    return declared_kind(obj) in STATE_KINDS and hasattr(obj, "space") and not isinstance(obj, Relation)


def stated_kind(obj):
    """The kind of the state obj (SET, MEASURE or ORDERING), or None when obj is not a state."""
    return declared_kind(obj) if is_state(obj) else None
