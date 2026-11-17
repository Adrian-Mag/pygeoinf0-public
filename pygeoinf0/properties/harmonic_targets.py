"""Targets kept symbolic, as property families of harmonic coefficients up to a degree (plan 044; plan 070 P1: from
operators/harmonic_targets.py).

A target t cut to the pieces of a ball (a measure restricted to shells, measures/restrict.py) has no realisation, but against
separable functionals k (x) Y_lj only its coefficients t_lj(r) enter (realisations/series/symbolic_harmonics.py computes them,
by Gauss quadrature over the support's natural box). They are the radial factors of a separable family tau, so T = Phi tau is
a factored family (properties/families.py) that the data-space decomposition takes.

    pg.harmonic_target(T, degree)        T a functional, or a pg.Functionals of several: rows of t_lj up to degree
    pg.RotatedFamily(T0, centres, degree=L)    T0 zonal and symbolic: its m = 0 coefficients and the addition theorem

T(m) itself, for a radial x harmonic m, pairs these coefficients with m's (realised_moment; spaces/representers.moments), so the plain
call has neither a degree to choose nor a rule that misses m's kinks.

The expansion is exact on fields of degree <= L, so its pairings with data of degree <= L are the target's, up to the quadrature
(stated). A prior reaching degrees above L sees only the truncation: the target's own norm is exact without a prior, and the
truncated part is stated, but with a prior give L at least the prior's band.
"""
import numpy as np

from ..core.claims import CHECKED, DERIVED, Claim
from ..core.errors import Refused
from ..realisations.series import legendre as LG
from ..realisations.series.symbolic_harmonics import coefficients, members


def expansion(space, t, L):
    """({summand: PiecewiseLegendre of (L + 1)^2 kernels against r^2 dr}, estimate) of an element of a direct sum whose parts are symbolic:
    each part's coefficients times its weight and density (a separable family divides by both again, plan 066)."""
    from ..spaces.representers import DensityRepresenter
    out, err = {}, 0.0
    pl = space._pl_of(t)
    for k, name in enumerate(space.names):
        if pl.parts[k] is None:
            continue
        part = space.part(t, name)
        if not isinstance(part, DensityRepresenter):
            raise Refused(f"the part {name!r} of the target is not a density kept symbolic", needs=("a measure on the pieces",))
        p, e = coefficients(part, L)
        B = space.summands[k].backend           # kernels against r^2 dr: the representer times w c rho (plan 066)
        w = space.weights[k] * B.c
        out[name] = LG.PiecewiseLegendre(p.breaks, [C * w for C in p.coefs], np.zeros(p.N))
        if getattr(B, "rho", None) is not None:
            out[name] = LG.multiply(out[name], B.rho)
        err = max(err, e)
    return out, err


def harmonic_target(T, degree, name="T"):
    """T = Phi tau: the rows of a functional kept symbolic as a factored family, its coefficients up to degree (module
    docstring). Exact on fields of degree <= degree, up to the stated quadrature."""
    from .families import FactoredFamily
    from ..operators.separable import separable
    rows = members(T)
    if not rows:
        raise TypeError("give a functional (pg.Functional, pg.Functionals) kept symbolic on a direct sum")
    M = rows[0].space
    L = int(degree)
    lab = np.array([(l, j) for l in range(L + 1) for j in range(2 * l + 1)], int)
    radial = {k: [None] * (len(rows) * len(lab)) for k in M.names}
    err = 0.0
    for i, t in enumerate(rows):
        parts, e = expansion(M, t, L)
        err = max(err, e)
        for key, pl in parts.items():
            for b in range(len(lab)):
                radial[key][i * len(lab) + b] = LG.PiecewiseLegendre(pl.breaks, [C[:, [b]] for C in pl.coefs], np.zeros(1))
    tau = separable(M, {k: v for k, v in radial.items() if any(x is not None for x in v)}, np.tile(lab, (len(rows), 1)), name="tau")
    Phi = np.zeros((len(rows), len(rows) * len(lab)))
    for i in range(len(rows)):
        Phi[i, i * len(lab):(i + 1) * len(lab)] = 1.0
    norms = np.array([float(M.norm(t)) ** 2 for t in rows])
    claim = Claim(f"expanded in harmonics up to degree {L}: exact on fields of degree <= {L}; the target norms are exact, a prior "
                  f"above that degree sees only the truncation", DERIVED, "orthonormality of the harmonics")
    return FactoredFamily(tau, Phi, name=name, norms2=(norms, claim),
                          claims=[claim, Claim(f"coefficients by Gauss quadrature over the support's box, estimated {err:.1e} relative "
                                               "against twice the nodes", CHECKED, "realisations/series/symbolic_harmonics.py", key="quadrature")])
