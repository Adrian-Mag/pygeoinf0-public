"""The state-by-relation table (plan 008 P3; EXP-028 DESIGN.md section 2): two registries of cells.

    push(state, R)          forward: a state on X through a relation R: X -> Y (cells F4-F15)
    update(prior, R, d)     backward: a prior on X, a model-data relation R and the observed datum d (cells B4-B14)
    table()                 the cells, their hypotheses, outputs and status (documentation)

A cell is chosen by the kind of the state (set, measure or ordering; an element given as the asserted point has the
rows of the kind point: plan 067 P4) and of the relation, then by a structural test (the most specific first:
"linear Gaussian" before the general case). Its hypotheses are claim keys of the objects it uses (the prior, the
relation, its operator G, its noise). policy="strict" (the default) raises HypothesisUnmet when one is not on record;
policy="tagged" computes and marks the result incomplete (result.missing). Cells that are mathematically undefined
refuse with the reason (Refused); cells not implemented yet say so (Unsupported). Every result that can carry them
gets the claims of the cell (what was assumed and built) and result.cell, the cell's name.
"""
from ..core.claims import THEOREM, UNKNOWN, Claim, Claims
from ..core.errors import HypothesisUnmet, Refused, Unsupported
from ..core.logic import Truth
from ..core.kinds import POINT
from ..states.base import is_element, state_kind

FORWARD, BACKWARD = [], []


class Cell:
    """One cell of the table: (state kind, relation kind), when it applies, what it needs, what it gives."""

    def __init__(self, table, name, state, relation, output, impl=None, requires=(), when=None, refuse=None,
                 needs=(), exists_as="", note=""):
        self.name, self.state, self.relation, self.output = name, state, relation, output
        self.impl, self.requires, self.when, self.refuse, self.needs = impl, tuple(requires), when, refuse, tuple(needs)
        self.exists_as, self.note = exists_as, note
        table.append(self)

    def applies(self, state, R):
        return self.when is None or bool(self.when(state, R))

    def __repr__(self):
        return f"{self.name}: {self.state} × {self.relation} -> {self.output}"


def cell(table, name, state, relation, output, requires=(), when=None, exists_as="", note=""):
    """Register the decorated function as a cell."""
    def wrap(fn):
        Cell(table, name, state, relation, output, impl=fn, requires=requires, when=when, exists_as=exists_as,
             note=note)
        return fn
    return wrap


def refusal(table, name, state, relation, reason, needs=(), when=None):
    """Register a cell that is undefined without a further stated choice."""
    Cell(table, name, state, relation, "refused", refuse=reason, needs=needs, when=when)


def _role(role, state, R):
    return {"state": state, "prior": state, "relation": R, "G": getattr(R, "G", R),
            "noise": getattr(R, "noise", None)}[role]


def hypotheses(c, state, R):
    """The claims a cell uses, with their statuses, and the missing ones."""
    from ..operators.claim_rules import status
    used, missing = [], []
    for role, key in c.requires:
        obj = _role(role, state, R)
        held, st = status(obj, key)
        name = getattr(obj, "name", role)
        if held is Truth.TRUE and st != UNKNOWN:
            used.append(Claim(f"{role} {name}: {key}", st, f"hypothesis of cell {c.name}", key=key))
        else:
            missing.append((role, name, key))
    return used, missing


def row_kind(state):
    """The row of the table: the kind of the state, or point for an element given as the asserted point (a point is
    not a kind of state, amendment 3: these rows give the answer of its singleton or Dirac)."""
    return POINT if is_element(state) else state_kind(state)


def _dispatch(table, state, R, args, policy):
    sk = row_kind(state)
    rk = R.kind
    for c in table:
        if c.state is not sk or c.relation is not rk or not c.applies(state, R):
            continue
        if c.refuse is not None:
            raise Refused(f"{c.name}: {c.refuse}", needs=c.needs)
        used, missing = hypotheses(c, state, R)
        if missing and policy == "strict":
            raise HypothesisUnmet(f"{c.name} needs hypotheses that are not on record: "
                                  + ", ".join(f"{k} of the {r} {n}" for r, n, k in missing),
                                  needs=[f"pg.assert_claim({r}, {k!r}, 'your reason')" for r, n, k in missing])
        out = c.impl(state, R, *args)
        return annotate(out, c, used, missing)
    raise Refused(f"no cell for a {sk} state and a {rk} relation ({type(R).__name__})")


def annotate(out, c, used, missing=()):
    """Attach the cell's name and hypotheses to a result that can carry them."""
    extra = [Claim(f"computed by cell {c.name} ({c.state} × {c.relation} -> {c.output})", THEOREM,
                   "pygeoinf0.cells")] + list(used)
    extra += [Claim(f"{k} of the {r} {n}", UNKNOWN, "not on record (tagged policy)", key=k) for r, n, k in missing]
    if isinstance(getattr(out, "_claims", None), list):
        out._claims.extend(extra)
    elif isinstance(getattr(out, "claims", None), Claims):
        try:
            out.claims = Claims(list(out.claims) + extra)
        except AttributeError:
            pass
    for attr, value in (("cell", c.name), ("missing", tuple(missing)), ("complete", not missing)):
        try:
            setattr(out, attr, value)
        except AttributeError:
            break
    return out


def push(state, R, policy="strict"):
    """Push a state on X forward through a relation R: X -> Y (cells F4-F15; module docstring)."""
    if getattr(R, "domain", None) != getattr(state, "space", R.domain):
        from ..core.errors import SpaceMismatch
        raise SpaceMismatch(f"{R.name} acts on {R.domain!r}; the state lives on {state.space!r}")
    return _dispatch(FORWARD, state, R, (), policy)


def update(prior, R, d, rule=None, policy="strict"):
    """Update a prior on X with the datum d through the model-data relation R: X -> D (cells B4-B14; module docstring).
    rule: for orderings, "hard" (the default for set and point relations) or ("soft", gamma) with a stated gamma (a
    number, or the gamma of a meta rule such as pg.meta.discrepancy, whose record the update carries)."""
    if getattr(prior, "space", R.domain) != R.domain:
        from ..core.errors import SpaceMismatch
        raise SpaceMismatch(f"the prior lives on {prior.space!r}; {R.name} acts on {R.domain!r}")
    return _dispatch(BACKWARD, prior, R, (d, rule), policy)


def table():
    """The registered cells as text: name, kinds, output, hypotheses, status."""
    from . import backward, forward                                      # noqa: F401  (registration)
    rows = ["| cell | state | relation | output | hypotheses | status |", "|---|---|---|---|---|---|"]
    for c in FORWARD + BACKWARD:
        status = "refused" if c.refuse else ("unsupported (later)" if c.output == "unsupported" else "implemented")
        hyp = ", ".join(f"{k} ({r})" for r, k in c.requires) or "-"
        rows.append(f"| {c.name} | {c.state} | {c.relation} | {c.output} | {hyp} | {status}"
                    + (f"; today: {c.exists_as}" if c.exists_as else "") + " |")
    return "\n".join(rows)


def unsupported(table_, name, state, relation, reason, when=None):
    """Register a meaningful cell that is not implemented yet."""
    def impl(*_):
        raise Unsupported(f"{name}: {reason}")
    Cell(table_, name, state, relation, "unsupported", impl=impl, when=when)
