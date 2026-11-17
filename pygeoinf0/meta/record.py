"""Meta records (plan 009 G11): what every result of pg.meta states about itself.

The meta layer is not one of the seven elements of an inference problem (thesis el:59): a trade-off curve, a knee, a
predictive check or a sensitivity lane is computed from a problem's elements, and choosing among results is a decision
about the problem, not part of it. A choice made from the datum is recorded as data-dependent (el:524-526). Every
record therefore carries two claims besides those of its computation:

    key "meta"                 the record is meta (el:59), DERIVED: it is computed from the elements, not one of them
    key "used the datum"       holds = whether the datum entered the record, DERIVED from the inputs actually used
    key "data-dependent rule"  only for choices (a knee, operating points, a lane's variant, hyperparameters) made from
                               the datum: the choice rules of pg.meta (meta/choice.py) write it; no other package does

and the claims of the decomposition it was built on (each record costs no more than one data-space decomposition,
pg.decompose; plan 009 G3).
"""
import numpy as np

from ..core.claims import ASSERTED, DERIVED, Claim, Claims


def meta_claims(kind, used_datum, how, choice_from_datum=False, choice=None):
    """The claims every meta record carries (module docstring)."""
    out = [Claim(f"{kind}: a meta-layer record, not one of the seven elements of the problem", DERIVED,
                 "computed from the elements of the problem (thesis el:59)", key="meta"),
           Claim("used the datum" if used_datum else "did not use the datum", DERIVED, how, key="used the datum",
                 holds=bool(used_datum))]
    if choice_from_datum:
        out.append(Claim(f"a data-dependent choice ({choice or kind}): a statement computed after it is conditional on "
                         "that choice", ASSERTED, "el:524-526: a choice made from the datum is recorded as such",
                         key="data-dependent rule"))
    return out


class Record:
    """A meta record: named values (attributes), the claims (meta_claims first, then the computation's) and
    used_datum. Subclasses set the values; repr shows scalars and the shapes of arrays."""

    kind = "meta record"
    meta = True                     # every record of pg.meta is meta (el:59)

    def _record(self, used_datum, how, claims=(), choice_from_datum=False, choice=None):
        self.used_datum = bool(used_datum)
        self.claims = Claims(meta_claims(self.kind, used_datum, how, choice_from_datum, choice) + list(claims))

    def _show(self):
        return []

    def __repr__(self):
        def fmt(v):
            if isinstance(v, dict):
                return "{" + ", ".join(map(str, v)) + "}"
            a = np.asarray(v)
            if a.dtype.kind not in "biuf":             # strings (a lane's element, name, baseline), objects
                return repr(v) if isinstance(v, str) else str(v)
            return f"{a.item():.6g}" if a.size == 1 else f"array{a.shape}"
        vals = ", ".join(f"{k}={fmt(getattr(self, k))}" for k in self._show() if getattr(self, k, None) is not None)
        return f"{type(self).__name__}({vals}; {'used' if self.used_datum else 'no'} datum)"


class Choice(Record):
    """The result of a choice rule of pg.meta (plan 070 P2): values {parameter: value} of its family (a
    HyperparameterFamily), each also an attribute (a float carrying the record's claims, so that a method given it
    states the choice), the rule's name and the claims: meta_claims (with "data-dependent rule" when the datum entered
    the choice), the family as a named record, then the computation's. options() gives the method's options at the
    chosen values (problem.run(family.method, **choice.options()) where the method takes them: damping of
    least_squares and of sola; pli takes none and refuses them, naming how to state them: inference/methods.py)."""

    kind = "data-dependent choice"

    def _choose(self, rule, family, values, used_datum, how, claims=()):
        from ..solvers.choice import Alpha
        self.rule, self.family = rule, family
        self._record(used_datum, how, [family.claim(values)] + list(claims), choice_from_datum=used_datum,
                     choice=f"{', '.join(values)} by {rule}")
        self.values = {k: Alpha(float(v), self.claims) for k, v in values.items()}
        for k, v in self.values.items():
            setattr(self, k, v)

    def options(self):
        """The method's options at the chosen values: the family's fixed options and the values."""
        return {**self.family.options, **self.values}

    def _show(self):
        return list(self.values)


def inherited(claims):
    """The claims of another record without its own meta and datum claims (a record states its own)."""
    return [c for c in claims if c.key not in ("meta", "used the datum")]


def column(a, j):
    """Column j of an array of shape (n_gamma, p) or (p,) (closed-form results), as (n_gamma,) or a scalar."""
    a = np.asarray(a, float)
    return a[..., j] if a.ndim else a
