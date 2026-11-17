"""The algebra of relations (EXP-028 DESIGN.md 1.3-1.8): composition, converse, embeddings between kinds, stacking of
several observations, and the claim rules the cells use.

    compose(S, R)          S after R. Within a kind: maps compose, set relations by union over the intermediate
                           set, Gaussian additive kernels by Chapman-Kolmogorov; points into sets or measures by the
                           embeddings. Set then kernel (a family of measures) and kernel then set (a random set) leave
                           the three kinds: refused with the reason (el:575)
    converse(R, prior)     set kind always; a map's fibres; a kernel needs a prior
    as_kind(R, kind)       points as singletons (set) or Dirac measures (measure); a single-valued total linear
                           relation as its operator part (point, review D8); everything else needs a choice
    stack(R1, R2, ..., rule=...)
                           several observations of one model, into the direct sum of their data spaces (a Euclidean
                           with block labels; the column operator x -> (G_k x)_k, operators/blocks.py): rule "all" (one
                           model explains every block), "any", ("at least", k) (the voting rule, app-obs), "independent"
                           (kernels: the product, conditional independence asserted). There is no default: what a collection means is a choice (inv:163)
    status(obj, key)       (holds, status) of a property: the claims on record, else the operator rules
                           (operators/claim_rules.py since plan 070 P7; pg.relations.status is that function)
"""
import numpy as np

from ..core.claims import ASSERTED, THEOREM, Claim, Claims
from ..core.errors import Refused, SpaceMismatch, Unsupported
from ..core.kinds import MEASURE, POINT, SET
from ..core.relation import Relation
from ..core.logic import Truth
from ..spaces.base import Euclidean
from ..operators.blocks import ColumnOperator
from ..operators.linear import LinearOperator
from ..spaces.identity import require
from ..operators.claim_rules import status


# ------------------------------------------------------------------------------------------------ structure
def structure(R, which):
    """dom, ran, ker or mul of a relation (Relation.dom etc.)."""
    from .linear import LinearRelation
    from ..convex.subspaces import Whole, Zero
    if isinstance(R, LinearOperator):
        if which == "dom":
            return Whole(R.domain)
        if which == "mul":
            return Zero(R.codomain)
        return getattr(LinearRelation(R, name=R.name), which)
    if which == "dom":
        d = getattr(R, "_dom", None)
        if d is not None:
            return d
        if status(R, "total")[0] is Truth.TRUE or R.kind is MEASURE:
            return Whole(R.domain)
        raise Unsupported(f"dom {R.name} is not known (a set-valued relation given by its graph)")
    raise Unsupported(f"{which} is defined here for linear relations; {R.name} is a {R.kind} relation")


# ------------------------------------------------------------------------------------------------ composition
def compose(S, R):
    """S after R (module docstring)."""
    from ..convex.algebra import image, minkowski
    from .additive import AdditiveSet
    from .kernels import AdditiveMarkovKernel, MarkovKernel
    from .linear import LinearRelation
    from .maps import Composite, SetValued
    from ..convex.subspaces import Whole, image as sub_image, span_sum
    require(R.codomain, S.domain, f"cannot compose: {R.name} maps into {R.codomain!r}, {S.name} acts on {S.domain!r}",
            f"{S.name} @ {R.name}")
    LO, ks, kr = LinearOperator, S.kind, R.kind
    name = f"{S.name}∘{R.name}"
    if isinstance(S, LO) and isinstance(R, LO):
        return S @ R
    if isinstance(R, LinearRelation) and R.G is not None and (isinstance(S, LO) or isinstance(S, LinearRelation)):
        G = S if isinstance(S, LO) else S.G
        if isinstance(S, LinearRelation) and not isinstance(S.dom, Whole):
            raise Unsupported("composition after a partial linear relation is not implemented")
        mul = sub_image(G, R.mul) if isinstance(S, LO) else span_sum(sub_image(G, R.mul), S.mul)
        return LinearRelation(G @ R.G, mul=mul, dom=R.dom, name=name)
    if isinstance(S, LinearRelation) and isinstance(R, LO) and S.G is not None:
        from ..convex.subspaces import preimage
        return LinearRelation(S.G @ R, mul=S.mul, dom=preimage(R, S.dom), name=name)
    Y = getattr(R.codomain, "points", None)                 # relations through a finite set: by enumeration
    if Y is not None and (ks, kr) in ((SET, SET), (MEASURE, MEASURE), (POINT, MEASURE)):
        return _finite_compose(S, R, list(Y), name)
    if ks is POINT and kr is POINT:
        return Composite(S, R, name=name)
    if ks is SET and kr is POINT:
        if isinstance(S, AdditiveSet) and isinstance(R, LO):
            return AdditiveSet(S.G @ R, S.noise, name=name)
        return SetValued(R.domain, S.codomain, values=lambda x: S(R(x)), name=name)
    if ks is POINT and kr is SET:
        if isinstance(R, AdditiveSet) and isinstance(S, LO):
            return AdditiveSet(S @ R.G, image(S, R.noise), name=name)
        return SetValued(R.domain, S.codomain, values=lambda x: S(R(x)), name=name)
    if ks is SET and kr is SET:
        if isinstance(R, AdditiveSet) and isinstance(S, AdditiveSet):
            return AdditiveSet(S.G @ R.G, minkowski(image(S.G, R.noise), S.noise), name=name)
        raise Unsupported(f"the composition of {S.name} and {R.name}: union over the intermediate set, implemented "
                          "for additive relations")
    if ks is MEASURE and kr is POINT:
        if isinstance(S, AdditiveMarkovKernel) and S.q is None and isinstance(R, LO):
            return AdditiveMarkovKernel(S.G @ R, S.noise, name=name)
        return MarkovKernel(R.domain, S.codomain, lambda x: S(R(x)), name=name)
    if ks is POINT and kr is MEASURE:
        if isinstance(S, LO) and isinstance(R, AdditiveMarkovKernel) and R.q is None:
            return AdditiveMarkovKernel(S @ R.G, S(R.noise), name=name)
        return MarkovKernel(R.domain, S.codomain, lambda x: S(R(x)), name=name)
    if ks is MEASURE and kr is MEASURE:
        if isinstance(S, AdditiveMarkovKernel) and isinstance(R, AdditiveMarkovKernel) and S.q is None and R.q is None:
            return AdditiveMarkovKernel(S.G @ R.G, S.G(R.noise) + S.noise, name=name)  # Chapman-Kolmogorov, Gaussian
        raise Unsupported("Chapman-Kolmogorov composition of general kernels is not implemented")
    if ks is MEASURE and kr is SET:
        raise Refused(f"{S.name} after {R.name}: a set pushed through a kernel gives a family of measures, not a "
                      "measure (no mixing weights, el:575); that kind is not implemented")
    if ks is SET and kr is MEASURE:
        raise Refused(f"{S.name} after {R.name}: a kernel followed by a set-valued relation gives a random set (el:575, "
                      "el:584); that kind is not implemented")
    raise Refused(f"no composition rule for a {ks} relation after a {kr} relation")


def _finite_compose(S, R, Y, name):
    """Composition through a finite set: set relations by existence of an intermediate point, kernels (laws
    DiscreteMeasure, plan 067 P5) by Chapman-Kolmogorov sums, a map after a kernel by the push-forward of the law."""
    from ..probability import DiscreteMeasure
    from .kernels import MarkovKernel
    from .maps import SetValued
    if S.kind is SET:
        def member(x, z):
            out = Truth.FALSE
            for y in Y:
                out = out | (Truth.of(R.member(x, y)) & Truth.of(S.member(y, z)))
            return out
        return SetValued(R.domain, S.codomain, member=member, name=name)

    def law(x):
        out = {}
        for y, p in R(x).items():
            if S.kind is POINT:
                out[S(y)] = out.get(S(y), 0.0) + p
            else:
                for z, q in S(y).items():
                    out[z] = out.get(z, 0.0) + p * q
        Z = S.codomain
        return DiscreteMeasure(out, on=Z if getattr(Z, "points", None) is not None else None)
    return MarkovKernel(R.domain, S.codomain, law, name=name)


# ------------------------------------------------------------------------------------------------ converse, embeddings
def converse(R, prior=None):
    """The converse relation (module docstring)."""
    from .linear import LinearRelation
    from .maps import SetValued
    if R.kind is MEASURE:
        return R.converse(prior)
    if prior is not None:
        raise Refused(f"the converse of a {R.kind} relation needs no prior; pg.update(prior, {R.name}, d) intersects "
                      "a prior set with it")
    if isinstance(R, LinearOperator):
        try:
            return LinearRelation(R, name=R.name).converse()
        except Unsupported:
            pass
    if R.kind is POINT:
        return SetValued(R.codomain, R.domain, member=lambda y, x: R.member(x, y) if hasattr(R, "member") else
                         _same(R.codomain, R(x), y), name=f"{R.name}⁻¹",
                         claims=[Claim("the fibres of a map", THEOREM, "inv:104-114")])
    return R.converse()


def _same(Y, a, b):
    from .maps import _equal
    return _equal(Y, a, b)


def as_kind(R, kind):
    """The embedding of R into another kind (DESIGN 1.3)."""
    from ..convex.sets import singleton
    from ..probability import Dirac, DiscreteMeasure
    from .kernels import AdditiveMarkovKernel, MarkovKernel
    from .linear import LinearRelation
    from .maps import SetValued
    if kind is R.kind:
        return R
    if kind is POINT:
        return _as_point(R)
    if R.kind is POINT and kind is SET:
        if isinstance(R, LinearOperator):
            return LinearRelation(R, name=R.name)
        if hasattr(R.codomain, "norm"):
            return SetValued(R.domain, R.codomain, values=lambda x: singleton(R(x)), name=R.name,
                             claims=[Claim("singletons of a map", THEOREM, "the embedding of points into sets")])
        return SetValued(R.domain, R.codomain, member=lambda x, y: R.member(x, y), name=R.name,
                         claims=[Claim("singletons of a map", THEOREM, "the embedding of points into sets")])
    if R.kind is POINT and kind is MEASURE:
        Y = R.codomain
        if getattr(Y, "points", None) is not None:                 # a finite set: the discrete law delta_f(x)
            return MarkovKernel(R.domain, Y, lambda x: DiscreteMeasure({R(x): 1.0}, on=Y, name="δ"), name=f"δ∘{R.name}")
        dirac = Dirac(space=Y)
        if isinstance(R, LinearOperator):
            return AdditiveMarkovKernel(R, dirac, name=f"δ∘{R.name}")
        held, st = status(R, "measurable")
        cl = [Claim("measurable", st, "the map's claim", key="measurable")] if held is Truth.TRUE else []
        return MarkovKernel(R.domain, Y, lambda x: Dirac(R(x), space=Y),
                      name=f"δ∘{R.name}", claims=cl)
    if R.kind is SET and kind is MEASURE:
        raise Refused("a set-valued relation has no canonical measure-valued counterpart: choosing a measure on each "
                      "set adds probabilities the sets do not contain (el:496-515, el:624-628)",
                      needs=["a stated measure on each R(x) (softening, a choice)"])
    if R.kind is MEASURE and kind is SET:
        raise Refused("a kernel has no canonical set-valued counterpart (the support loses the probabilities, el:515)",
                      needs=["L.acceptance(alpha) (calibrated acceptance regions, inv:203-246)"])
    raise Refused(f"no embedding of a {R.kind} relation into the {kind} kind")


def _as_point(R):
    """A relation as a map: a linear relation that is single-valued and total (claims on record) is its operator part
    (review D8); otherwise a choice is needed and the conversion is refused with the reason."""
    if R.kind is MEASURE:
        raise Refused(f"{R.name} is a kernel: a map from it chooses a point of each law (a mean, a mode), a stated "
                      "choice the kernel does not contain", needs=["a stated point of each law (e.g. its mean)"])
    G = getattr(R, "G", None)
    if R.kind is SET and isinstance(G, LinearOperator):
        single, total = status(R, "single-valued")[0], status(R, "total")[0]
        if single is Truth.TRUE and total is Truth.TRUE:
            return G
        why = [w for w, t in (("single-valued (mul R = {0})", single), ("total (dom R = X)", total))
               if t is not Truth.TRUE]
        raise Refused(f"{R.name} is a map only where it is single-valued and total; it is not on record as "
                      + " and ".join(why) + ": a selection (e.g. the minimum-norm one) is a choice",
                      needs=["R.G, the operator part, if that selection is meant"])
    raise Refused(f"a {R.kind} relation is a map only where it is single-valued and total with an operator part on "
                  f"record; {R.name} has none here", needs=["a map stated by the caller"])


# ------------------------------------------------------------------------------------------------ stacking
def _embed_span(mul, D, block):
    from ..convex.subspaces import Span, Zero
    if isinstance(mul, Zero):
        return Zero(D)
    if not isinstance(mul, Span):
        raise Unsupported("stacking linear relations whose mul is not a span")
    a, b = block
    V = np.zeros((len(mul.vectors), D.dim))
    V[:, a:b] = mul.vectors.values
    return Span(D.family(V), name=mul.name)


def stack(*relations, rule=None):
    """Several observations of one model as one relation into the stacked data space (module docstring)."""
    from ..convex.algebra import CartesianProduct
    from .additive import AdditiveSet
    from .kernels import AdditiveMarkovKernel
    from .linear import LinearRelation
    from ..convex.subspaces import Zero, span_sum
    if rule is None:
        raise Refused("stacking several observations needs a stated rule: what the collection means is a choice "
                      "(inv:163, app-obs)", needs=["rule='all' (one model explains every block)", "rule='any'",
                                                   "rule=('at least', k)", "rule='independent' (kernels: a product)"])
    rels = list(relations)
    if len(rels) == 1:
        return rels[0]                                       # one observation: the relation itself
    for R in rels:
        require(R.domain, rels[0].domain, "stacked relations need one model space", R.name)
    if not all(isinstance(R.codomain, Euclidean) for R in rels):
        raise Unsupported("stacking observations is implemented for Euclidean data spaces (the column operator into a "
                          "pg.DirectSum of other spaces is operators.blocks.ColumnOperator)")
    if rule == "independent":
        if not all(isinstance(R, AdditiveMarkovKernel) and R.q is None for R in rels):
            raise Refused("rule 'independent' stacks kernels (a product of the noise laws); for sets use 'all'")
        G = ColumnOperator([R.G for R in rels])
        return AdditiveMarkovKernel(G, _product_noise([R.noise for R in rels], G.codomain), name=f"⊗ {G.name}")
    if any(isinstance(R, AdditiveMarkovKernel) for R in rels):
        raise Refused(f"rule {rule!r} is for set-valued relations; kernels are stacked with rule='independent' (the "
                      "product, conditional independence asserted) or as one kernel with a joint noise law")
    sets = [R if isinstance(R, (AdditiveSet, LinearRelation)) else AdditiveSet(R, Zero(R.codomain))
            if isinstance(R, LinearOperator) else None for R in rels]
    if any(S is None for S in sets):
        raise Unsupported("stacking is implemented for operators, additive set relations and linear relations")
    G = ColumnOperator([S.G for S in sets])
    if rule == "all":
        if all(isinstance(R, LinearOperator) for R in rels):
            return G
        if all(isinstance(S, LinearRelation) or isinstance(S.noise, Zero) for S in sets):
            mul = Zero(G.codomain)
            for S, blk in zip(sets, G.ranges):
                mul = span_sum(mul, _embed_span(S.mul if isinstance(S, LinearRelation) else Zero(S.codomain),
                                                G.codomain, blk))
            return LinearRelation(G, mul=mul, name=G.name)
        noises = [S.noise if isinstance(S, AdditiveSet) else S.mul for S in sets]
        return AdditiveSet(G, CartesianProduct(noises, G.codomain, blocks=G.ranges), name=f"{G.name} + noise")
    k = 1 if rule == "any" else rule[1] if isinstance(rule, tuple) and rule[0] == "at least" else None
    if k is None or not 1 <= k <= len(rels):
        raise ValueError(f"unknown rule {rule!r}: 'all', 'any', ('at least', k) with 1 <= k <= {len(rels)}, "
                         "'independent'")
    return Voting(sets, G, k)


class Voting(Relation):
    """The relation of at least k of n observations explained (app-obs): kept as its members; pg.update gives the
    union over the subsets of k members of their 'all' feasible sets."""

    kind = SET

    def __init__(self, members, G, k):
        self.members, self.G, self.k = members, G, k
        self.domain, self.codomain, self.name = G.domain, G.codomain, f"at least {k} of {len(members)}"
        self._claims = [Claim(f"rule: at least {k} of {len(members)} blocks explained", ASSERTED,
                              "a choice of meaning (inv:163)", key="rule")]

    @property
    def claims(self):
        return Claims(self._claims)

    def __call__(self, x):
        raise Unsupported(f"the values of {self.name} are unions of products; use pg.update(prior, R, d)")


def _product_noise(noises, D):
    from ..operators.linear import Identity, Matrix
    from ..probability import GaussianMeasure
    if all(isinstance(m.covariance, Identity) and m.zero_mean for m in noises):
        return D.standard_gaussian()
    from scipy.linalg import block_diag
    S = block_diag(*[m.space.solve_metric(m.covariance.matrix().T).T for m in noises])   # coordinate covariances
    mean = None if all(m.zero_mean for m in noises) else D(np.concatenate([m.mean.values for m in noises]))
    C = Matrix(S @ D.metric_matrix(), D, D, name="C")
    return GaussianMeasure(D, C, mean, name="⊗ noise", _claims=[Claim("independent blocks", ASSERTED,
                                                                      "conditional independence of the observations",
                                                                      key="independent noise")])


def join(R, *data):
    """The stacked datum of a stacked relation: data of the blocks concatenated."""
    return R.codomain(np.concatenate([d.values for d in data]))

