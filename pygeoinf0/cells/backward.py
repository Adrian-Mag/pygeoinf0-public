"""Backward cells B4-B14 (EXP-028 DESIGN.md 2.3): a prior on X, a model-data relation R: X -> D and the observed
datum d give an updated state (inv:252-585). The existing methods are cells here: minimum norm (B5 with the whole
space, B11), damped least squares with a stated gamma (B13, B14; choosing gamma from the datum is a meta rule,
pg.meta.discrepancy, plan 070 P2), GaussianMeasure.condition (B10); BGP's feasible set (B6). The prior "none" is the
whole space, a set (plan 067 P4: NoPrior(X) is Whole(X)), so its update is the set row's (B5-B7) by one route; the
former rows B1-B3 of "none" are gone.

Cells choose by the kinds of the state and the relation, then by structure (the relation's class, a claim on record),
never by class names (review D8)."""
import itertools

from ..core.claims import ASSERTED, DERIVED, THEOREM, Claim, derived
from ..core.errors import Refused
from ..core.kinds import MEASURE, ORDERING, POINT, SET
from ..core.logic import Truth, decide
from ..spaces.base import Euclidean
from ..operators.linear import Identity, LinearOperator
from .registry import BACKWARD, cell, refusal, unsupported


def _op_into_rn(state, R):
    return isinstance(R, LinearOperator) and isinstance(R.codomain, Euclidean)


def _additive_set(state, R):
    from ..relations.additive import AdditiveSet
    return isinstance(R, AdditiveSet)


def _linear_relation(state, R):
    from ..relations.linear import LinearRelation
    return isinstance(R, LinearRelation)


def _voting(state, R):
    from ..relations.algebra import Voting
    return isinstance(R, Voting)


def _additive_kernel(state, R):
    from ..relations.kernels import AdditiveMarkovKernel
    return isinstance(R, AdditiveMarkovKernel)


def _whole(S):
    """S is the whole space (the prior "none", NoPrior(X) = Whole(X)): S ∩ A is A."""
    from ..convex.subspaces import Whole
    return isinstance(S, Whole)


def _gaussian(prior):
    """A Gaussian law: a GaussianMeasure or a CylindricalGaussian (by class, plan 067 P5)."""
    from ..probability import is_gaussian
    return is_gaussian(prior)


def _cylindrical(mu):
    """The cylindrical Gaussian's own kind check (decision Q11): a law that speaks about bounded functionals only."""
    from ..probability import CylindricalGaussian
    return isinstance(mu, CylindricalGaussian)


def _fibre(G, d, name):
    """The fibre G^-1(d) = m-dagger + ker G (B1), or the empty set with the minimum-norm summary of the projected d."""
    from ..convex.sets import AffineSubset, EmptySet
    from ..relations.linear import LinearRelation
    R = G if isinstance(G, LinearRelation) else LinearRelation(G, name=G.name)
    Rinv = R.converse()
    m = Rinv.G(d)
    if Rinv.dom.member(d) is Truth.FALSE:
        out = EmptySet(R.domain, why="d is not attainable: no model fits it exactly (inv:64-88)")
        out.summary = lambda: m
        out._claims.append(Claim("summary(): the minimum-norm least-squares model, d projected on the range first",
                                 THEOREM, "a stated rule (inv:79), not a model that fits d"))
        return out
    out = AffineSubset(m, Rinv.mul, name=name, least_norm=True)
    out._claims += [Claim("the fibre m-dagger + ker of the relation (exact data)", THEOREM, "inv:42-50, DESIGN L2"),
                    Claim("summary(): the element of least norm, the operator part of the converse (G^+ d)", THEOREM,
                          "DESIGN Corollary; it uses the inner product of the model space (inv:914-921)")]
    return out


def _pred(R, d):
    from ..sets.subsets import PredicateSubset
    return PredicateSubset(R.domain, lambda m: R.member(m, d), name=f"{R.name}⁻¹(d)")


# ------------------------------------------------------------------------------------------------ asserted point
@cell(BACKWARD, "B4", POINT, POINT, "set: {x0} if f(x0) = d, else empty (a consistency test)")
def _b4(x0, R, d, rule):
    D = R.codomain
    y = R(x0)
    return _point_result(x0, decide(D.norm(y - d), max(D.norm(d), D.norm(y), 1e-300)))


@cell(BACKWARD, "B4", POINT, SET, "set: {x0} if d in R(x0), else empty")
def _b4_set(x0, R, d, rule):
    return _point_result(x0, R(x0).member(d))


@cell(BACKWARD, "B4", POINT, MEASURE, "measure: the Dirac prior is its own posterior (Gaussian likelihood > 0)",
      when=_additive_kernel)
def _b4_kernel(x0, R, d, rule):
    from ..probability import Dirac
    return Dirac(x0, _claims=[Claim("an asserted point: the Dirac prior is its own posterior", THEOREM, "el:481")])


def _point_result(x0, t):
    from ..convex.sets import EmptySet, singleton
    if t is Truth.FALSE:
        return EmptySet(x0.space, why="the asserted model does not explain the datum")
    out = singleton(x0)
    if t is Truth.UNKNOWN:
        out._claims.append(Claim("the datum is explained only to rounding", ASSERTED, "undecided membership"))
    return out


# ------------------------------------------------------------------------------------------------ set prior
@cell(BACKWARD, "B5", SET, POINT, "set: S ∩ G^-1(d) (exact data); S = X (the prior none): the fibre m-dagger + ker G",
      requires=[("relation", "linear")], when=_op_into_rn,
      exists_as="minimum norm G.adjoint((G @ G.adjoint).inv(d)) (the summary of the fibre)")
def _b5(S, G, d, rule):
    if _whole(S):
        return _fibre(G, d, f"{G.name}⁻¹(d)")
    from ..convex.feasible import FeasibleSet
    from ..convex.subspaces import Zero
    return FeasibleSet(S, G, Zero(G.codomain), d)


@cell(BACKWARD, "B5", SET, POINT, "set: S ∩ {m : f(m) = d} (membership)")
def _b5_map(S, R, d, rule):
    return _meet(S, _pred(R, d))


@cell(BACKWARD, "B6", SET, SET, "set: K(d) = S ∩ G^-1(d - N), the BGP feasible set", requires=[("G", "linear")],
      when=_additive_set, exists_as="BGP [LAND EXP-001, EXP-014, EXP-019]")
def _b6(S, R, d, rule):
    from ..convex.feasible import FeasibleSet
    return FeasibleSet(S, R.G, R.noise, d, name=f"{R.name}⁻¹(d)" if _whole(S) else "K(d)")


@cell(BACKWARD, "B6", SET, SET, "set: S ∩ {m : d - G m in mul R}; S = X: the fibre of the linear relation (nuisances "
      "projected out)", when=_linear_relation)
def _b6_linear(S, R, d, rule):
    if _whole(S):
        return _fibre(R, d, f"{R.name}⁻¹(d)")
    from ..convex.feasible import FeasibleSet
    return FeasibleSet(S, R.G, R.mul, d)


@cell(BACKWARD, "B6", SET, SET, "set: the union over k-subsets of the blocks' feasible sets", when=_voting)
def _b6_voting(S, R, d, rule):
    return _voting_update(S, R, d)


@cell(BACKWARD, "B6", SET, SET, "set: S ∩ R^-1(d) (membership)")
def _b6_general(S, R, d, rule):
    return _meet(S, _pred(R, d))


def _meet(S, A):
    """S ∩ A for a subset A of the model space: A itself when S is the whole space."""
    if _whole(S):
        return A
    from ..convex.algebra import Intersection
    return Intersection([S, A])


refusal(BACKWARD, "B7", SET, MEASURE, "a set prior (the whole space included: the prior 'none') gives no probabilities "
        "on models, and an observation kernel has no converse without a prior measure: it gives probabilities of data "
        "given a model, not of models given data (inv:177-183, el:579, el:594)",
        needs=["L.acceptance(alpha): confidence-set inversion with the prior set (B6)",
               "a prior measure (posterior: B10)", "L.likelihood(d): an ordering by likelihood"])


def _voting_update(prior, R, d):
    from ..convex.algebra import Union
    from ..convex.feasible import FeasibleSet
    from ..relations.algebra import join, stack
    from ..convex.subspaces import Zero
    sets = []
    for idx in itertools.combinations(range(len(R.members)), R.k):
        S = stack(*[R.members[i] for i in idx], rule="all")
        dI = join(S, *[R.G.block(d, i) for i in idx])
        G = S if isinstance(S, LinearOperator) else S.G
        N = S.noise if hasattr(S, "noise") else S.mul if hasattr(S, "mul") else Zero(G.codomain)
        sets.append(FeasibleSet(prior, G, N, dI, name=f"K{tuple(i + 1 for i in idx)}"))
    out = Union(sets, name=f"at least {R.k} of {len(R.members)} explained")
    out._claims.append(Claim(f"rule: at least {R.k} of {len(R.members)} observations explained", ASSERTED,
                             "a choice of meaning (inv:163, app-obs)"))
    return out


# ------------------------------------------------------------------------------------------------ measure prior
@cell(BACKWARD, "B8", MEASURE, POINT, "measure: mu conditioned on G = d (exact data; affine version)",
      requires=[("relation", "linear")], when=lambda mu, R: _gaussian(mu) and _op_into_rn(mu, R))
def _b8(mu, G, d, rule):
    from ..probability import CylindricalGaussian, GaussianMeasure
    from ..solvers import Eigh
    C = mu.covariance
    S = G(mu).covariance
    Sp = S.inverse(Eigh("rounding"))
    r0 = d if mu.zero_mean else d - G(mu.mean)
    D = G.codomain
    if D.norm(r0 - S(Sp(r0))) > 1e-8 * max(D.norm(r0), 1e-300):
        raise Refused("the exact datum lies outside the support of G#mu: prior and datum contradict each other "
                      "(d - G m0 is not in the range of G C G*)")
    K = C @ G.adjoint
    mean = K(Sp(r0)) if mu.zero_mean else mu.mean + K(Sp(r0))
    cov = C - K @ Sp @ G @ C
    from ..probability.gaussian import premise
    (sa, n1), (pos, n2) = premise(mu, "sa"), premise(mu, "psd")
    cov._sa = derived(True, f"C - C G* (G C G*)^+ G C{n1}", sa)
    cov._pos = derived("psd", f"a Schur complement{n2}", pos)

    def sampler(N, rng):
        X0 = mu._centred(N, rng)
        return X0 - K(Sp(G(X0)))
    from ..solvers.direct import rounding_claim
    law = CylindricalGaussian if _cylindrical(mu) else GaussianMeasure
    return law(mu.space, cov, mean, name=f"{mu.name} | {G.name} = d", _sampler=sampler,
                           _claims=[Claim("conditioned on exact data: a disintegration over the fibres of G, the "
                                          "affine version (the event has probability 0)", DERIVED, "el:577, inv:605-618"),
                                    rounding_claim(Sp, "(G C G*)^+")])


unsupported(BACKWARD, "B9", MEASURE, SET, "mu conditioned on K = R^-1(d) needs mu(K) > 0 and is a truncated Gaussian "
            "here (EXP-028 DESIGN 2.5 (g)): not implemented yet")


@cell(BACKWARD, "B10", MEASURE, MEASURE, "measure: the posterior (Kalman formulas; Bayes' rule for Gaussians)",
      requires=[("G", "linear"), ("relation", "independent noise")],
      when=lambda mu, R: _gaussian(mu) and _additive_kernel(mu, R), exists_as="prior.condition(G, d, noise)")
def _b10(mu, R, d, rule):
    G, noise = R.effective
    post = mu._condition(G, R.datum(d), noise)
    post.relation = R
    extra = [Claim(f"model-data relation: {R.name}, d = G x + e, e ~ {noise.name} independent of x", ASSERTED,
                   "pg.Additive(G, noise) (built by condition when called through it)"),
             Claim("version: the affine one (Gaussian conditioning formulas at every datum)", THEOREM, "inv:556")]
    if R.q is not None:
        extra.append(Claim("free nuisances: conditioned on the quotient datum q(d)", THEOREM, "EXP-028 DESIGN 1.7"))
    if _cylindrical(mu):
        extra.append(Claim("the prior is cylindrical: the posterior speaks about bounded functionals of the model",
                           THEOREM, "inv:659-669, inf:1290 (EXP-028 DESIGN 1.9)"))
    post.claims = type(post.claims)(list(post.claims) + extra)
    return post


unsupported(BACKWARD, "B10", MEASURE, MEASURE, "posteriors through non-Gaussian kernels are not implemented")


# ------------------------------------------------------------------------------------------------ ordering prior
CHOICES = ["a number gamma, fixed before the data", "the gamma of a meta rule: pg.meta.discrepancy(problem).gamma "
           "(its record states that it was chosen from the datum)"]


def _gamma(G, y, gamma):
    """A stated gamma (review D3: the cells take a stated alpha), and its claims: a gamma chosen by a meta rule carries
    its record (claims), which the update passes on."""
    if gamma is None or isinstance(gamma, str):
        raise Refused(f"the soft update takes a stated gamma, not {gamma!r}: a rule that chooses gamma from the datum "
                      "is meta (pg.meta.discrepancy, plan 070 P2)", needs=CHOICES)
    record = list(getattr(gamma, "claims", None) or ())
    if any(c.key == "used the datum" for c in record):          # a meta record (pg.meta: a choice rule's result)
        return float(gamma), record
    return float(gamma), [Claim(f"gamma = {float(gamma):g}, fixed before the data (stated)", ASSERTED, "inv:289-300")]


def _soft(J, G, d, gamma, misfit_name):
    """Phi + gamma J with Phi = (1/2) ||G m - d||^2 in D's metric, and its minimiser (damped least squares)."""
    from ..states.base import Ordering
    D = G.codomain
    c = J.centre
    y = d if c is None else d - G(c)
    g, claims = _gamma(G, y, gamma)
    A = G @ G.adjoint

    def argmin():
        m = G.adjoint((A + g * Identity(D)).inv(y))
        return m if c is None else c + m

    def cost(m):
        return 0.5 * D.norm(G(m) - d) ** 2 + g * J.cost(m)
    return Ordering(J.space, cost, name=f"{misfit_name} + {g:.4g} {J.name}", argmin=argmin, claims=claims + [
        Claim("the updated ordering Phi + gamma J; its minimiser is a summary (inv:326-347)", THEOREM, "inv:296-300")])


@cell(BACKWARD, "B11", ORDERING, POINT, "ordering: J on the fibre (hard); B14: Phi + gamma J (soft)",
      requires=[("relation", "linear")], when=lambda J, R: getattr(J, "is_norm", False) and _op_into_rn(J, R),
      exists_as="minimum norm (hard), damped least squares (soft)")
def _b11(J, G, d, rule):
    from ..states.base import Ordering
    if isinstance(rule, tuple) and rule[0] == "soft":
        return _soft(J, G, d, rule[1], "½‖G m - d‖²")
    c = J.centre
    K = _fibre(G, d, f"{G.name}⁻¹(d)")
    shift = (lambda: K.summary()) if c is None else (lambda: c + _fibre(G, d - G(c), "").summary())
    return Ordering(J.space, J._cost, feasible=K, name=J.name, argmin=shift,
                    claims=[Claim("hard update: the prior ordering restricted to the fibre", THEOREM, "inv:302-324")])


@cell(BACKWARD, "B12", ORDERING, SET, "ordering: J on the feasible set (hard)",
      when=lambda J, R: getattr(J, "is_norm", False) and _additive_set(J, R))
def _b12(J, R, d, rule):
    from ..convex.feasible import FeasibleSet
    from ..convex.sets import NormBall
    from ..convex.subspaces import Whole
    from ..states.base import Ordering
    K = FeasibleSet(Whole(R.domain), R.G, R.noise, d)
    argmin = None
    N = R.noise
    if isinstance(N, NormBall) and N.centre is None:          # the Occam model: least norm with misfit <= rho
        G, D, c = R.G, R.codomain, J.centre
        y = d if c is None else d - G(c)

        def argmin():
            if D.norm(y) <= N.radius:
                return c if c is not None else J.space.zero
            from ..solvers import damping_for_misfit
            a = float(damping_for_misfit(G @ G.adjoint, y, target=N.radius ** 2))   # the multiplier of the constraint
            m = G.adjoint((G @ G.adjoint + a * Identity(D)).inv(y))
            return m if c is None else c + m
    return Ordering(J.space, J._cost, feasible=K, name=J.name, argmin=argmin,
                    claims=[Claim("hard update: the prior ordering restricted to the feasible set", THEOREM,
                                  "inv:302-318")])


@cell(BACKWARD, "B13", ORDERING, MEASURE, "ordering: -log l(d | m) + gamma J (soft; a stated gamma)",
      requires=[("G", "linear")], when=lambda J, R: getattr(J, "is_norm", False) and _additive_kernel(J, R),
      exists_as="damped least squares (a stated gamma; pg.meta.discrepancy chooses one from the datum)")
def _b13(J, R, d, rule):
    if not (isinstance(rule, tuple) and rule[0] == "soft"):
        raise Refused("updating an ordering with a kernel needs the weight of the misfit: rule=('soft', gamma) with "
                      "a stated gamma (the weight is part of the update, inv:289-300)", needs=CHOICES)
    G, datum = R.standardised()          # N(b, C) on a plain D read as D_C (plan 071 P4, Q1 as amended: both routes)
    out = _soft(J, G, datum(d), rule[1], "-log l(d | m)")
    noise = R.effective[1]
    if not isinstance(noise.covariance, Identity) or not noise.zero_mean:
        out._claims.append(Claim(f"-log l(d | m) = (1/2)||d - b - G m||^2 in the inverse-covariance norm of {noise.name}"
                                 " (D's vectors with the noise's inner product: R.standardised())", THEOREM,
                                 "the Gaussian log-likelihood up to a constant"))
    return out


unsupported(BACKWARD, "B8", MEASURE, POINT, "conditioning non-Gaussian measures or on nonlinear maps is not implemented")
unsupported(BACKWARD, "B11", ORDERING, POINT, "updates of orderings other than the norm cost are not implemented")
unsupported(BACKWARD, "B12", ORDERING, SET, "updates of orderings other than the norm cost are not implemented")
unsupported(BACKWARD, "B13", ORDERING, MEASURE, "updates of orderings other than the norm cost are not implemented")
