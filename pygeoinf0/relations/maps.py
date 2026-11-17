"""Relations given by a rule, without a forward operator G (the SYNTHESIS addendum: the MDR is a relation, not a G):

    Map(X, Y, f, dom=None)                    point kind: x -> f(x), possibly partial (dom a subset of X), nonlinear
    Composite(S, R)                           S after R for point relations (what S @ R gives): its derivative by the
                                              chain rule DS(R(x)) DR(x) where both parts have one (a LinearOperator
                                              is its own derivative; plan 020 P4)
    SetValued(X, Y, values=...)               set kind: x -> a subset of Y
    SetValued(X, Y, member=lambda x, y: ...)  set kind given by its graph: (x, y) in R (a bool or a Truth)

Claims are what the constructor knows: a Map is single-valued by construction; measurability, continuity or linearity
of a callable are asserted with the reason you give (measurable="why"), never inferred from passing tests.
"""
import numpy as np

from ..core.claims import ASSERTED, THEOREM, Claim, Claims
from ..core.errors import Refused
from ..core.kinds import POINT, SET, is_state
from ..core.relation import Relation
from ..core.logic import Truth, decide


def _equal(Y, a, b):
    """Truth of a == b in Y, to rounding."""
    return decide(Y.norm(a - b), max(Y.norm(a), Y.norm(b), 1e-300))


class Map(Relation):
    """x -> f(x): a point relation from a rule (module docstring). dom: a subset of X (with member) for a partial map."""

    kind = POINT

    def __init__(self, domain, codomain, f, dom=None, name="f", measurable=None, continuous=None, linear=None):
        self.domain, self.codomain, self.f, self._dom, self.name = domain, codomain, f, dom, name
        self._claims = [Claim("single-valued", THEOREM, "a map", key="single-valued"),
                        Claim("total" if dom is None else "partial", THEOREM, "dom f", key="total", holds=dom is None)]
        for key, why in (("measurable", measurable), ("continuous", continuous), ("linear", linear)):
            if why:
                self._claims.append(Claim(key, ASSERTED, why, key=key))

    @property
    def claims(self):
        return Claims(self._claims)

    def __call__(self, x):
        if is_state(x):
            from ..cells import push
            return push(x, self)
        if self._dom is not None and self._dom.member(x) is Truth.FALSE:
            raise Refused(f"{getattr(x, 'name', 'x')} is not in dom {self.name}: the partial map is not defined there")
        return self.f(x)

    def member(self, x, y):
        """Truth of (x, y) in the graph."""
        if self._dom is not None and self._dom.member(x) is Truth.FALSE:
            return Truth.FALSE
        if not hasattr(self.codomain, "norm"):                      # a set without a norm: exact comparison
            return Truth.of(np.array_equal(np.asarray(self.f(x)), np.asarray(y)))
        return _equal(self.codomain, self.f(x), y)


def _derivative(F, x, whole):
    from ..operators.linear import LinearOperator
    if isinstance(F, LinearOperator):
        return F
    if hasattr(F, "derivative"):
        return F.derivative(x)
    raise Refused(f"{getattr(F, 'name', 'a part')} has no derivative, so {whole} has none (the chain rule needs both "
                  "parts)", needs=(f"a derivative of {getattr(F, 'name', 'that part')}",))


class Composite(Map):
    """S after R, x -> S(R(x)), with the chain rule (module docstring); dom is R's (S refuses for itself)."""

    def __init__(self, S, R, name=None):
        name = name or f"{S.name}∘{R.name}"
        super().__init__(R.domain, S.codomain, lambda x: S(R(x)), dom=getattr(R, "_dom", None), name=name)
        self.parts = (S, R)

    def derivative(self, theta):
        S, R = self.parts
        D = _derivative(S, R(theta), self.name) @ _derivative(R, theta, self.name)
        D.name = f"D{self.name}"
        return D


class SetValued(Relation):
    """x -> R(x), a subset of the codomain (module docstring)."""

    kind = SET

    def __init__(self, domain, codomain, values=None, member=None, name="R", claims=()):
        if (values is None) == (member is None):
            raise ValueError("give values (x -> a subset) or member ((x, y) -> bool), not both")
        self.domain, self.codomain, self.name = domain, codomain, name
        self._values, self._member = values, member
        self._claims = list(claims)
        self._converse_of = None

    @property
    def claims(self):
        return Claims(self._claims)

    def __call__(self, x):
        if is_state(x):
            from ..cells import push
            return push(x, self)
        if self._values is not None:
            return self._values(x)
        from ..sets.subsets import PredicateSubset
        return PredicateSubset(self.codomain, lambda y: self._member(x, y), name=f"{self.name}({getattr(x, 'name', 'x')})")

    def member(self, x, y):
        """Truth of (x, y) in the graph: y in R(x)."""
        if self._member is not None:
            return Truth.of(self._member(x, y))
        return self._values(x).member(y)

    def converse(self, prior=None):
        """R^-1 = {(y, x) : y in R(x)}: always defined for a set-valued relation (inv:116-122)."""
        if prior is not None:
            raise Refused("a set-valued relation has a converse without a prior; a prior set is intersected by "
                          "pg.update(prior, R, d)")
        if self._converse_of is not None:
            return self._converse_of
        out = SetValued(self.codomain, self.domain, member=lambda y, x: self.member(x, y), name=f"{self.name}⁻¹",
                        claims=[Claim("the converse of a set-valued relation", THEOREM, "inv:116-122")])
        out._converse_of = self
        return out
