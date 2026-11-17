"""Every exception and warning pygeoinf0 raises on purpose, in one place.

SpaceMismatch            an element of one space was used where another space was expected
RealisationMismatch      elements of one space realised differently, mixed without a conversion (a Refused: plan 069
                         P1, decision Q4; the space is the same, so this is never a SpaceMismatch): .needs the choices
SupportOutsideDomain     a measure (e.g. a ray) whose support leaves the domain of the space
UnboundedFunctional      a linear functional that is not bounded on the space (e.g. a point or a path in L2):
                         it has no Riesz representer there; the message names the space where it is bounded
PointEvaluationUndefined the special case of point evaluation
Unsupported              a mathematically meaningful operation that this realisation does not implement
NotDefined               an operation the mathematics does not define (a Refused, not an Unsupported: there is no such
                         object to build, e.g. a Riesz map of L^p for p != 2, an inner product on C(K))
NotAnElement             a function given as an element of a space it is not in, seen on its realisation (a Refused): a
                         jump at a break in H^s for s >= 1/2, a jump of the m-th derivative for s >= m + 1/2
NormUnavailable          the norm (or an inner product) of this element in this space cannot be computed exactly
                         from what is known about it (e.g. a formula in a restriction space H^s(Omega))
RealisationError         the adaptive fit could not resolve a function
NotContinuous            a function given as an element of C(K) whose realisation jumps (it is not continuous)
ViolatesBoundaryCondition a function given as an element of a Sobolev space whose elements satisfy a boundary
                         condition (u = 0 on a Dirichlet part; d_n u = 0 in a spectral Neumann space of order >= 3/2)
                         and which does not
ZeroFunctional           a datum of the zero functional (u -> int u dmu for a measure on a Dirichlet part, where every
                         element vanishes): it carries no information, and any value other than 0 contradicts the space
RealisationWarning       the fit converged only partially
TruthComparedWithBool    a Truth (core/logic.py) compared with a bool by == or != (plan 071 P1): the comparison was
                         silently False; test `t is Truth.TRUE` (or `is not Truth.FALSE`), or a Claim's .holds (a bool)
Refused                  an operation that is mathematically undefined without a further stated choice (a cell of the
                         state-by-relation table: the converse of a kernel needs a prior, a set pushed through a kernel
                         has no mixing weights); .reason says why, .needs what would make it defined
NotATiling               pieces given as a partition (pg.Partition) leave a gap in the domain, overlap, or leave it
                         (a Refused and a ValueError: .needs says what would make them tile)
NoiseUnstated            a method or choice rule that needs the noise law, given a point relation into a plain data space
                         (a Refused: plan 071 P4, Q1 as amended): .needs the two routes (the noise in the relation, or
                         a covariance-weighted data space)
NotACovariance           a noise covariance that is not one: not symmetric beyond rounding, a variance that is not
                         positive, or a matrix that is not positive definite (a Refused: plan 071 P4, review item 6)
HypothesisUnmet          a cell or method whose hypothesis is not on record (claim unknown) under the strict policy
ExternalError            external code wrapped through the port (pygeoinf0.port) failed at a point of its declared
                         domain of definition: the message names the code, the map and the declared dom
PrecisionWarning         a result whose digits are at stake: the condition and the digits it costs (solvers/)
ResolutionWarning        a fit from samples of a callable (or a pointwise re-fit) with neither breaks=, support= nor
                         resolution=: its stated resolution, the choices, and that it will become a refusal (core/fits.py)
ResolutionUnstated       that refusal, now under pg.settings.resolution_policy = "refuse" (a Refused: .needs the choices)
GenerationError          building a problem from a profile (pg.port.generate) failed: the message names the element and
                         its counterpart reference

unbounded(mu, k, space, why) builds the refusal of a functional (one wording for every space).
"""


class PrecisionWarning(UserWarning):
    """A result whose digits are at stake: its condition and what it costs, or directions with no correct digit (plan 054)."""


class ResolutionWarning(UserWarning):
    """A fit from samples of a callable with neither breaks=, support= nor resolution= (plan 054, narrow bumps, option C):
    a feature narrower than its stated resolution may be lost without another warning. It will become a refusal
    (ResolutionUnstated); pg.settings.resolution_policy = "refuse" gives that now."""


class SpaceMismatch(TypeError):
    """An element of one space was used where another space was expected."""


class MembershipUndecided(TypeError):
    """`x in S` where S.member(x) is Truth.UNKNOWN (plan 067 P2, decision Q3): `in` needs a decided answer; S.member(x)
    gives the three-valued one."""


class TruthComparedWithBool(TypeError):
    """A Truth compared with a bool by == or != (plan 071 P1, the author 2026-11-15: refuse, not compare). Truth.TRUE ==
    True was False, so (holds, status) == (True, "checked") was silently False: test `t is Truth.TRUE` (or `t is not
    Truth.FALSE`, which counts UNKNOWN as possible), or read a Claim's .holds (a bool)."""


class ExternalError(RuntimeError):
    """External code wrapped through the port (pygeoinf0.port) failed at a point of the map's declared domain of
    definition; the original exception is its __cause__."""


class GenerationError(RuntimeError):
    """pg.port.generate could not build an element of a profile; the message names the element and its counterpart, and the
    original exception is its __cause__."""


class Refused(ValueError):
    """An operation that is mathematically undefined without a further stated choice; .reason says why and .needs lists
    what would make it defined (plan 008: the refusals of the state-by-relation table)."""

    def __init__(self, reason, needs=()):
        self.reason, self.needs = reason, tuple(needs)
        super().__init__(reason + (f" (needs: {'; '.join(self.needs)})" if self.needs else ""))


class RealisationMismatch(Refused):
    """Elements of one space whose realisations differ, combined without a conversion (plan 069 P1, decision Q4): the
    space is the same whatever realises its elements, so this is not a SpaceMismatch. The message names both
    realisations; .needs the choices (spaces/identity.py)."""


class ResolutionUnstated(Refused):
    """A fit from samples of a callable with neither breaks=, support= nor resolution=, refused under
    pg.settings.resolution_policy = "refuse" (the later default; ResolutionWarning until then)."""


class PriorUnstated(Refused):
    """A prior (a Gaussian law) asked of a space without naming it (plan 069 P6, decision Q2): no space chooses one, so
    X.gaussian() and a function space's X.random() refuse; .needs names the constructors (probability/priors.py)."""


class NoiseUnstated(Refused):
    """A noise law asked of a point relation G into a plain data space (plan 071 P4, Mac review item 4, Q1 as amended):
    nothing states it, and D's dot product is not a noise model. .needs names the two routes: the noise in the
    relation, pg.Additive(G, N(0, C)), or the whitened norm, D = Euclidean(n, covariance=C), where unit noise in D's
    metric is the meaning."""


class NotACovariance(Refused):
    """A noise covariance that is not one (plan 071 P4, Mac review item 6): not symmetric beyond the rounding tolerance
    (spaces.metrics.symmetric_within_rounding), a variance that is not positive and finite (the message names the
    coordinate), or a matrix that is not positive definite. Never symmetrised or clipped silently."""


class UnboundedFunctional(TypeError, Refused):
    """A linear functional that is not bounded on the space (it has no Riesz representer there). A Refused too: the
    one refusal path of the boundedness decision (spaces/dual.py, EXP-031)."""

    def __init__(self, reason, needs=()):
        Refused.__init__(self, reason, needs)


class PointEvaluationUndefined(UnboundedFunctional):
    """Point evaluation is not a bounded functional on the space (e.g. L2, or H^s with s <= d/2)."""


class SupportOutsideDomain(ValueError):
    """A measure whose support leaves the domain of the space (a ray that leaves the domain must be clipped)."""


class Unsupported(NotImplementedError):
    """A meaningful operation that this realisation does not implement."""


class CapabilityMissing(Unsupported):
    """A capability of the realisation protocol (realisations/registry.py, plan 069 P3) that this realisation does not
    provide: .capability its name, .provided the capabilities it has; the message also names who has it."""

    def __init__(self, reason, capability, provided=()):
        super().__init__(reason)
        self.capability, self.provided = capability, tuple(sorted(provided))


class NotDefined(Refused):
    """An operation the mathematics does not define (plan 065 A2): no such object exists, so nothing could implement it
    (against Unsupported: meaningful, not built). .needs names what is defined instead."""


class NotAnElement(Refused):
    """A function given as an element of a space it is not in, seen on its realisation (plan 065 A1: a fit that jumps at
    a break in H^s, s >= 1/2); .needs names the choices."""


class NormUnavailable(Unsupported):
    """A norm or inner product that cannot be computed exactly from what is known about the element."""


class RealisationError(ArithmeticError):
    """The adaptive fit could not resolve a function (typically: not square integrable, or singular)."""


class NotContinuous(ValueError):
    """A function given as an element of a space of continuous functions jumps (checked on its realisation)."""


class ViolatesBoundaryCondition(ValueError):
    """A function given as an element of a space with a boundary condition on its elements violates it."""


class ZeroFunctional(ValueError):
    """A datum of a functional that is zero on the space (a measure on a Dirichlet part): refused by Functionals."""


class AveragingWarning(UserWarning):
    """f.normalised() on a function that is not a localized average: it changes sign, or its integral is nearly zero."""


class RealisationWarning(UserWarning):
    """The adaptive fit converged only partially."""


class NotATiling(Refused):
    """Pieces that do not tile their domain up to a null set (a gap, an overlap, a piece outside it): pg.Partition."""


class HypothesisUnmet(Refused):
    """A hypothesis of a cell or method is not on record (its claim is unknown, or refuted) under the strict policy;
    .needs names the claims and how to state them (pg.assert_claim(obj, key, reason))."""


class ProbeTimeout(TimeoutError):
    """An attempt of pg.catalogue.probe exceeded its time limit."""


def unbounded(mu, k, space, why):
    """The refusal of u -> int u dmu for a measure mu of dimension k on space: PointEvaluationUndefined for points,
    UnboundedFunctional otherwise; why says what would make it bounded."""
    what = "point evaluation" if k == 0 else f"integration over a {k}-dimensional set"
    return (PointEvaluationUndefined if k == 0 else UnboundedFunctional)(f"{what} ({mu}) is not bounded on {space}: {why}")
