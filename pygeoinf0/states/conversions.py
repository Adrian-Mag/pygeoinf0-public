"""Conversions between the kinds of state (plan 067 P4; review B3). Each takes its stated choice as an argument; a
conversion that needs a choice not given is refused, naming it.

    as_set(state, form=, mass=)   a set: a set is itself; an element x is the singleton {x} (no choice); a measure
                                  needs the form and the mass of a hardened set (harden); an ordering's level sets are
                                  not developed (amendment 3)
    as_measure(state)             a measure: a measure is itself; an element x is delta_x (no choice); a set gives no
                                  probabilities (softening is a choice, refused)
    harden(mu, form, mass, ...)   {m : Q(m - c) <= t} with mu-mass >= mass (probability/hardening.py computes it: the
                                  law of the form and its quantile; mu.harden is the same call)

The conversions of relations live with the relations, which sit above the states: as_kind (relations/algebra.py) and
a kernel's acceptance(alpha) (relations/kernels.py), the set-valued relation of calibrated acceptance regions.
"""
from ..core.errors import Refused
from ..core.kinds import MEASURE, ORDERING, SET
from .base import is_element, state_kind


def harden(mu, form, mass, kind="norm", law=None, method="auto", name=None):
    """The set {m : Q(m - c) <= t} with P_mu(S) >= mass, c the mean of mu (probability/hardening.py): the choice is
    the form Q (kind "norm": ||B (m - c)||^2 with form B; "quadratic": <A (m - c), m - c>) and the mass."""
    from ..probability.hardening import harden as _harden
    if state_kind(mu) is not MEASURE:
        raise TypeError(f"harden takes a measure; {mu!r} is a state of kind {state_kind(mu)}")
    return _harden(mu, form, mass, kind=kind, law=law, method=method, name=name)


def as_set(state, form=None, mass=None, **options):
    """The state as a set (module docstring); form and mass are the choice of a hardened set of a measure."""
    if is_element(state):
        from ..convex.sets import singleton
        return singleton(state)
    k = state_kind(state)
    if k is SET:
        if form is not None or mass is not None:
            raise Refused("a set is already a set: form and mass choose the hardened set of a measure")
        return state
    if k is MEASURE:
        if form is None or mass is None:
            raise Refused(f"a measure gives a set only with a stated choice: harden(mu, form, mass) (a sublevel set of "
                          f"the form Q with mu-mass >= mass); {state!r} has none here",
                          needs=["as_set(mu, form=B, mass=1 - beta)", "pg.states.harden(mu, form, mass)"])
        return harden(state, form, mass, **options)
    assert k is ORDERING
    raise Refused("the level sets {J <= c} of an ordering are not developed (the author's amendment 3: the ordering "
                  "kind is declared, not developed)")


def as_measure(state):
    """The state as a measure (module docstring)."""
    if is_element(state):
        from ..probability import Dirac
        return Dirac(state)
    k = state_kind(state)
    if k is MEASURE:
        return state
    if k is SET:
        raise Refused("a set gives no probabilities on its elements: a measure on it is a choice (softening) that adds "
                      "probabilities the set does not contain (el:496-515)", needs=["a stated prior measure"])
    raise Refused("an ordering supplies no probabilities: a measure from it (e.g. exp(-J)) is a further choice")
