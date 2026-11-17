"""Claims: what is known about an object, with its status and the reason.

Statuses, strongest first:
  theorem   a standard mathematical fact (e.g. L2 is complete: Riesz-Fischer)
  derived   follows from how the object was built (G G* is self-adjoint and positive semidefinite)
  checked   verified numerically on this realisation (a passing check, not a proof)
  asserted  stated by the user, not verified
  unknown   nobody knows

A claim may carry a key (a short property name such as "linear", "bounded", "closed graph", "independent noise"): the
cells of pygeoinf0.cells require claims by key (plan 008). Claims written before keys existed are found by their
statement ("bounded, norm <= 1" answers the key "bounded"). holds=False records that the property fails.
"""
from dataclasses import dataclass

from .logic import Truth

THEOREM, DERIVED, CHECKED, ASSERTED, UNKNOWN = "theorem", "derived", "checked", "asserted", "unknown"
ORDER = (THEOREM, DERIVED, CHECKED, ASSERTED, UNKNOWN)
ON_RECORD = (THEOREM, DERIVED, CHECKED, ASSERTED)      # statuses that satisfy a required hypothesis (strict policy)


def weakest(*statuses):
    """The weakest of the statuses (UNKNOWN for none)."""
    return max(statuses, key=ORDER.index, default=UNKNOWN)


def by_rule(*premises):
    """The status of a conclusion drawn by a named rule from premises with these statuses: the weakest of DERIVED and
    the premises. A rule adds no evidence: checked premises give a checked conclusion, asserted ones an asserted
    conclusion (docs/research/findings/40-claims-rule-table.md section 1: CHECKED never promotes to DERIVED)."""
    return weakest(DERIVED, *premises)


def status_of(record):
    """The status of an operator's (value, status, reason) record (_sa, _pos) as a premise: UNKNOWN when nothing is
    recorded or the record says the property fails (plan 065 A5: a missing record is not promoted to ASSERTED; a
    construction that relies on the property names whose statement it relies on, probability.gaussian.premise)."""
    return record[1] if record and record[0] else UNKNOWN


def derived(value, reason, *premises):
    """The (value, status, reason) record of a conclusion drawn by a rule from premises with these statuses, or None
    when a premise is unknown: then nothing is recorded (plan 065 A5)."""
    st = by_rule(*premises)
    return None if st == UNKNOWN else (value, st, reason)


def _holds(h):
    """A claim's holds as a Truth: None (nothing said) is UNKNOWN, never FALSE."""
    return Truth.UNKNOWN if h is None else Truth.of(h)


@dataclass(frozen=True)
class Claim:
    statement: str
    status: str
    reason: str
    key: str = None
    holds: bool = True

    def __repr__(self):
        return f"[{self.status}] {self.statement}: {self.reason}"


class Claims(tuple):
    """A tuple of Claim objects, printed one per line."""

    def __repr__(self):
        return "\n".join(repr(c) for c in self) if self else "(no claims)"

    def weakest(self):
        return max((c.status for c in self), key=ORDER.index, default=UNKNOWN)

    def find(self, key):
        """The strongest claim about the property key (by key, or by its statement for claims without one), or None."""
        found = [c for c in self if c.key == key or (c.key is None and c.statement.split(",")[0].strip() == key)]
        return min(found, key=lambda c: ORDER.index(c.status)) if found else None

    def status(self, key):
        """(holds, status) of the property key, holds a Truth (core/logic.py): (Truth.UNKNOWN, "unknown") when nothing
        is on record, Truth.FALSE for a claim that records the property fails (plan 067 P1: was (None, "unknown") and a
        bool)."""
        c = self.find(key)
        return (Truth.UNKNOWN, UNKNOWN) if c is None else (_holds(c.holds), c.status)


def sampled_claim(name, note=""):
    """The claim of an element realised from samples of its callable (plan 054 P3, tests/test_error_account_p3.py)."""
    return Claim(f"{name}'s realisation is fitted from samples of its callable{note}: a feature narrower than the samples' "
                 f"spacing and lying between them is not seen, without a warning, and its error is not in {name}.error",
                 THEOREM, "finitely many samples do not determine an element of a function space; measured (plan 054 P3): "
                 "a Gaussian bump of width 1e-4 on [0, 1] lost entirely with an error estimate of 1e-15; breaks= at the "
                 "feature's support (not its centre) resolve it", key="sampled")


def assert_claim(obj, key, reason, statement=None):
    """State a property of an object (a relation, an operator, a set) with your reason: a claim with status asserted.
    It satisfies the hypothesis key of a cell (strict policy) and is printed with every result that used it."""
    target = getattr(obj, "_claims", None)
    if not isinstance(target, list):
        raise TypeError(f"{type(obj).__name__} does not keep claims that can be added")
    target.append(Claim(statement or key, ASSERTED, reason, key=key))
    return obj
