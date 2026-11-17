"""Constructors of model-data (and model-property) relations from a forward operator and a noise description. They
are conveniences: each builds an explicit relation whose claims say what it is (el:243-246, inv:870-904).

    Exact(G)                       R(m) = {G m}: G itself, the point relation (exact data)
    Additive(G, S)                 R(m) = G m + S for a set S of the codomain (bounded noise, a confidence region):
                                   AdditiveSet, with convex / closed graph derived from S and G (DESIGN L9)
    Additive(G, E)                 E a subspace (span of per-event indicators): the linear relation m -> G m + E
    Additive(G, mu)                mu a GaussianMeasure: the kernel m -> G m + mu (AdditiveMarkovKernel)
    Additive(G, mu, nuisance=E)    Gaussian noise and free nuisances in E: the kernel on the quotient D / E
    Additive(G, S, nuisance=E)     the set E + S (both free nuisances and bounded noise)
    Additive(G, noise, offset=b)   m -> G m + b + noise with a known offset b in the data space (a correction such as
                                   a crustal one, plan 009 G9): the noise law or set translated by b; the offset is
                                   asserted, and its own uncertainty is not represented (a claim says so)
R.select(rows) gives any of them on some of the data only (relations/selection.py).

The data space and the noise (plan 071 P4, the author's decision Q1 as amended 2026-11-15): either route works, and
they are one model. A: a plain D (Euclidean(n), DataSpace(n)) with the noise stated here, Additive(G, N(b, C)). B: the
whitened norm, D = Euclidean(n, covariance=C), with D.standard_gaussian() (coordinates N(0, C)). A point relation G
(Exact(G), exact data) states no noise: where a method or choice rule needs a noise law, point_noise(G, what) reads unit
noise in D's metric on a covariance-weighted D (route B: the covariance on D is the noise statement) and refuses on a
plain D (NoiseUnstated, naming both routes): D's dot product is not a noise model.
"""
import numpy as np

from ..core.claims import ASSERTED, DERIVED, THEOREM, Claim
from ..core.errors import NoiseUnstated, SpaceMismatch
from ..core.kinds import MEASURE, SET, stated_kind
from ..core.logic import Truth
from ..operators.claim_rules import status
from .maps import SetValued
from ..convex.subspaces import LinearSubspace, span_sum
from ..probability import is_gaussian
from ..spaces.identity import require


def Exact(G):
    """The point relation of exact data, R(m) = {G m}: the operator (or Map) itself (a LinearOperator is a relation of
    point kind); the cells record "exact data" when they use it."""
    return G


NOISE_ROUTES = ("the noise in the relation: pg.Additive(G, pg.GaussianMeasure(D, C)) on the plain data space",
                "the whitened norm: D = pg.Euclidean(n, covariance=C), where unit noise in D's metric is the meaning")


def point_noise(G, what):
    """(the noise law, its claim) that a point relation G: M -> D implies for what (a method or a choice rule that needs
    a noise law; module docstring): on a covariance-weighted D (its metric is not the dot product) the standard
    Gaussian of D, unit noise in D's metric, with an ASSERTED claim that says so; on a plain D, NoiseUnstated naming the
    two routes (plan 071 P4, Mac review item 4: no silent unit noise)."""
    D = G.codomain
    if getattr(D, "is_standard", True):
        raise NoiseUnstated(f"{what} needs the noise law, and the point relation {G.name} into the plain data space {D} "
                            "states none: D's dot product is not a noise model (Q1 as amended)", needs=NOISE_ROUTES)
    return D.standard_gaussian(), Claim(
        "the noise: the standard Gaussian of the data space (a point relation into a covariance-weighted D: unit noise "
        "in D's metric, the whitened norm)", ASSERTED, "Euclidean(n, covariance=C) states the noise (plan 071 P4, Q1 as "
        "amended)", key="noise")


UNSTATED_NOISE_ROUTES = ("the noise as an argument: a law on D (D.standard_gaussian() states unit noise; "
                         "pg.GaussianMeasure(D, C)), a noise set (a NormBall or Ellipsoid of D) or a covariance C",
                         "the noise in the relation: pg.Additive(G, noise)",
                         "the whitened norm: D = pg.Euclidean(n, covariance=C), where unit noise in D's metric is the meaning")


def require_noise(noise, D, what):
    """noise itself, or a refusal when it is None on a plain data space D (plan 071 P8, the author's decision on
    open decision 2 of 071 P4): nothing states the noise, and D's dot product is not a noise model. On a
    covariance-weighted D (its metric is not the dot product) None keeps its meaning, unit noise in D's metric."""
    if noise is None and getattr(D, "is_standard", True):
        raise NoiseUnstated(f"{what} needs the noise, and none is stated for the plain data space {D}: D's dot product "
                            "is not a noise model (plan 071 P8)", needs=UNSTATED_NOISE_ROUTES)
    return noise


class AdditiveSet(SetValued):
    """m -> G m + S, S a subset of G's codomain (bounded additive noise; model-independent)."""

    def __init__(self, G, noise, name=None, claims=()):
        require(noise.space, G.codomain, f"the noise set must be a subset of {G.codomain!r}", "the noise set")
        from ..convex.algebra import minkowski
        self.G, self.noise = G, noise
        cl = [Claim("additive (fibre-constant: depends on m through G m)", THEOREM, "the construction", key="additive"),
              Claim("total", THEOREM, "R(m) is a translate of a nonempty set", key="total",
                    holds=noise.is_empty() is not Truth.TRUE)]
        lin = status(G, "linear")[0] is Truth.TRUE
        if lin and noise.claims.status("convex")[0] is Truth.TRUE:
            cl.append(Claim("convex graph", DERIVED, "G linear and S convex: the graph is Phi^-1(S), Phi(m, y) = y - G m "
                            "(DESIGN L9)", key="convex graph"))
        if status(G, "bounded")[0] is Truth.TRUE and noise.claims.status("closed")[0] is Truth.TRUE:
            cl.append(Claim("closed graph", DERIVED, "G bounded and S closed (L9)", key="closed graph"))
        super().__init__(G.domain, G.codomain, values=lambda m: minkowski(noise, G(m)),
                         name=name or f"{G.name} + {noise.name}", claims=cl + list(claims))

    def member(self, m, y):
        return self.noise.member(y - self.G(m))


def Additive(G, noise=None, nuisance=None, offset=None):
    """A relation m -> G m + noise (module docstring): the kind follows from what the noise is."""
    from ..convex.algebra import minkowski
    from .kernels import AdditiveMarkovKernel
    from .linear import LinearRelation
    if offset is not None:
        return _offset(G, noise, nuisance, offset)
    if noise is None and nuisance is None:
        return Exact(G)
    if stated_kind(noise) is MEASURE:
        if not is_gaussian(noise):
            raise TypeError(f"additive noise given as a measure is a Gaussian measure on {G.codomain!r} (an additive "
                            f"kernel), not a {type(noise).__name__}: a general kernel is pg.MarkovKernel(X, Y, law)")
        return AdditiveMarkovKernel(G, noise, nuisance=nuisance)
    if noise is None or isinstance(noise, LinearSubspace):
        mul = noise if nuisance is None else (nuisance if noise is None else span_sum(noise, nuisance))
        return LinearRelation(G, mul=mul, name=f"{G.name} + {mul.name}")
    if stated_kind(noise) is SET:
        return AdditiveSet(G, noise if nuisance is None else minkowski(nuisance, noise))
    raise TypeError(f"the noise is a subset, a subspace or a GaussianMeasure of {G.codomain!r}, not a "
                    f"{type(noise).__name__}")


def _offset(G, noise, nuisance, offset):
    """m -> G m + b + noise: the noise translated by the known offset b (a mean for a Gaussian, a translate for a
    set; exact data become the single point G m + b)."""
    from ..convex.algebra import minkowski
    from ..core.claims import ASSERTED
    from ..convex.subspaces import Zero
    D = G.codomain
    b = offset if hasattr(offset, "space") else D(np.asarray(offset, float), name="b")
    require(b.space, D, f"the offset is an element of {b.space}, not of the data space {D}", "the offset")
    claim = Claim(f"known offset {b.name}: d = G m + {b.name} + noise; the offset's uncertainty is not represented",
                  ASSERTED, "a stated correction (Additive(..., offset=))", key="offset")
    if stated_kind(noise) is MEASURE:
        out = Additive(G, noise + b, nuisance=nuisance)
        out._claims.append(claim)
    else:
        if noise is None:
            base = Zero(D) if nuisance is None else nuisance
        elif nuisance is None:
            base = noise
        else:
            base = span_sum(noise, nuisance) if isinstance(noise, LinearSubspace) else minkowski(nuisance, noise)
        out = AdditiveSet(G, minkowski(base, b), name=f"{G.name} + {b.name} + {getattr(base, 'name', 'N')}",
                          claims=[claim])
    out.offset = b
    return out
