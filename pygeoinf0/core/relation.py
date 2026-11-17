"""Relations between spaces (plan 008; experiments/EXP-028-relations/DESIGN.md section 1). What R(x) is decides the
kind of a relation (core/kinds.py; plan 067 P4: the kinds are closed, decision Q12):

    point     R(x) is an element of the codomain (a map, possibly partial): LinearOperator, pg.Map
    set       R(x) is a subset of the codomain, possibly empty: pg.SetValued, pg.LinearRelation, pg.Additive(G, set)
    measure   R(x) is a probability measure on the codomain (a Markov kernel): pg.MarkovKernel, pg.Additive(G, gaussian)

States on a space have the kinds set, measure and ordering (pygeoinf0.states; the author's amendment 3): the prior
"none" is the whole space, a set, and a point is a singleton set or a Dirac measure.

Relation is the common base. domain and codomain are the ambient spaces (as for a LinearOperator); dom is the restricted
domain, a subset of the domain where R(x) is not empty. R(x) gives the value; R @ S composes (R after S); R.converse()
reverses the graph (set kind; a kernel needs a prior); R.as_kind(kind) embeds points into sets or measures, and gives
the operator part of a single-valued total linear relation;
R.select(rows) keeps some of the data of a relation into R^n (plan 009 G9: the marginal, exact deletion). The
implementations live in pygeoinf0.relations, imported when first used (core imports nothing above it at import time).
"""


class Relation:
    """A relation between two spaces (module docstring). Subclasses set kind, domain, codomain and name and define
    __call__ (the value R(x)) and claims."""

    kind = None

    @property
    def dom(self):
        """The restricted domain: the subset of the domain where R(x) is not empty."""
        from ..relations.algebra import structure
        return structure(self, "dom")

    @property
    def ran(self):
        from ..relations.algebra import structure
        return structure(self, "ran")

    @property
    def ker(self):
        from ..relations.algebra import structure
        return structure(self, "ker")

    @property
    def mul(self):
        """The multivalued part R(0) of a linear relation ({0} for an operator)."""
        from ..relations.algebra import structure
        return structure(self, "mul")

    def converse(self, prior=None):
        """The converse relation {(y, x) : y in R(x)} (set kind). A kernel has none without a prior measure on its
        domain: converse(prior=mu) is the posterior kernel d -> mu^d."""
        from ..relations.algebra import converse
        return converse(self, prior)

    def as_kind(self, kind):
        """The embedding into another kind: points as singletons (set) or Dirac measures (measure); a single-valued
        total linear relation as its operator part (point); other conversions need a stated choice and are refused
        with the reason."""
        from ..relations.algebra import as_kind
        return as_kind(self, kind)

    def select(self, rows):
        """The relation of some of the data only (a relation into R^n): rows are indices, a mask, a slice, a predicate
        on the data space's labels or a dict of label conditions. A Markov kernel gives its marginal on those rows (exact
        deletion), a set-valued relation the projection of its noise set (relations/selection.py)."""
        from ..relations.selection import select
        return select(self, rows)

    def __matmul__(self, other):
        if isinstance(other, Relation):
            from ..relations.algebra import compose
            return compose(self, other)
        return self(other)

    def __rmatmul__(self, other):
        if isinstance(other, Relation):
            from ..relations.algebra import compose
            return compose(other, self)
        return NotImplemented

    def __repr__(self):
        return f"{getattr(self, 'name', 'R')}: {self.domain} ⇉ {self.codomain}  [{self.kind}]"
