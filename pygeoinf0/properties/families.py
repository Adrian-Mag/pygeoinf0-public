"""Property families indexed by centres (and radii) whose pairings factor (plan 009 G4; EXP-030 DESIGN G4; plan 070
P1: from operators/families.py).

    T = pg.CapAverages(M, radius, centres, profile="raised-cosine", band=None, summand=None)
    T = pg.RotatedFamily(functional, centres)

Let T_0 be a functional invariant under the rotations about the pole (zonal). Its rotations T_xi = T_0 o U_(g_xi)^-1 are
well defined for xi on the sphere, and by the addition theorem a zonal representer sum_l c_l (x) Y_l0 rotates to
sum_l c_l (x) sqrt(4 pi / (2l + 1)) sum_j Y_lj(xi) Y_lj, so

    T_xi = sum_b phi_b(xi) tau_b,    phi_(l,j)(xi) = sqrt(4 pi / (2l + 1)) Y_lj(xi),   tau_(l,j) = <c_l (x) Y_lj, .>

(DERIVED). For an average of a field on a sphere with a zonal weight w (Funk-Hecke, THEOREM),
T_xi(h) = int w(xi . eta) h(eta) d eta = sum_l lambda_l sum_j Y_lj(xi) h_lj, lambda_l = 2 pi int w(x) P_l(x) dx: then
phi_(l,j)(xi) = lambda_l Y_lj(xi) and tau_(l,j)(h) = h_lj (the coefficient in the unit-sphere harmonics). Either way the
family is a LinearOperator M -> R^p whose factors (Phi (p x B), tau: a separable family, operators/separable.py) are
exposed: T = Phi tau. A method that is linear in the target works on the B blocks and synthesises (Phi times per-block
numbers): estimates, variances sum_b phi_b^2 v_b when the blocks are uncorrelated, Phi V Phi^T in general, never a
p x p matrix (solvers/dataspace/factored.py). With band=B (the target T o P_B) the family is exact; without, it is the degree
truncation at L (degree=, default the largest degree with lambda_l above 1e-16 relative, at most 255), exact on fields of
degree <= L, and the target norms ||T_xi||^2 = int w^2 come from the weight itself (the truncated tail is stated).

Centres: (p, 3) directions, or (p, 2) geographic (latitude, longitude) in degrees. radius: the angular radius psi0 of
the cap (radians), or an array of them (properties ordered radius by radius). profile: "raised-cosine"
(cos^2(pi psi / (2 psi0)) on psi <= psi0: the thesis's caps), "spherical-cap", "gaussian" (exp(-psi^2 / (2 psi0^2))) or a
callable of psi; normalise: divide by int w0 (an average).
"""
import math

import numpy as np

from ..core.claims import CHECKED, DERIVED, THEOREM, Claim
from ..spaces.elements import Function, Vector, VectorFamily
from ..core.errors import Refused
from ..spaces.base import Euclidean
from ..realisations.series import harmonic_degrees as H
from ..spaces import harmonic as HA
from ..operators.linear import LinearOperator
from ..solvers.dataspace.quadratic import rowquad


def directions(centres):
    """Unit vectors (p, 3) from (p, 3) directions or (p, 2) (latitude, longitude) in degrees."""
    C = np.atleast_2d(np.asarray(centres, float))
    if C.shape[1] == 2:
        lat, lon = np.radians(C[:, 0]), np.radians(C[:, 1])
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    if C.shape[1] != 3:
        raise ValueError("centres: (p, 3) directions or (p, 2) (latitude, longitude) in degrees")
    n = np.linalg.norm(C, axis=1)
    if np.any(n == 0):
        raise ValueError("a centre is the zero vector")
    return C / n[:, None]


class FactoredFamily(LinearOperator):
    """T = Phi tau: M -> R^p with exposed factors (module docstring)."""

    why_linear = "the fixed matrix Phi after tau"

    def linear_parts(self):
        """The operators whose linearity this one inherits."""
        return [self.tau]

    def __init__(self, tau, Phi, name="T", claims=(), norms2=None):
        Phi = np.array(Phi, float)
        Phi.flags.writeable = False
        super().__init__(tau.domain, Euclidean(Phi.shape[0]), name=name)
        self.tau, self.Phi = tau, Phi
        self._norms2 = norms2
        self._claims += [Claim("linear", THEOREM, "a matrix times bounded functionals", key="linear"),
                         Claim("bounded", DERIVED, "finitely many bounded functionals", key="bounded"),
                         Claim(f"factored: T = Phi tau ({Phi.shape[0]} properties, {Phi.shape[1]} blocks)", DERIVED,
                               "rotations of a zonal functional (addition theorem, Funk-Hecke)")] + list(claims)

    @property
    def factors(self):
        """(Phi (p x B), tau: the separable functionals of the blocks)."""
        return self.Phi, self.tau

    def target_norms2(self, S=None):
        """(||T_xi||_S^2 per property, claim): from the weight itself where known (S the model metric), else from the
        factors (exact for a band-limited family)."""
        if S is None and self._norms2 is not None:
            return self._norms2
        tt = self.tau.structured_gram(S)
        if tt is None:
            return None
        V = tt.matrix() if self.tau.codomain.dim <= 4000 else None
        if V is None:
            return None
        return rowquad(self.Phi, V), None

    def _apply(self, x):
        v = self.tau(x)
        if isinstance(x, Function):
            return Vector(self.codomain, self.Phi @ v.values, name=f"{self.name}({x.name})")
        return VectorFamily(self.codomain, v.values @ self.Phi.T, name=f"{self.name}({x.name})")

    def _make_adjoint(self):
        tau, Phi = self.tau, self.Phi

        def adj(y):
            if isinstance(y, Vector):
                return tau.adjoint(tau.codomain(Phi.T @ y.values))
            return tau.adjoint(tau.codomain.family(y.values @ Phi))
        return LinearOperator(self.codomain, self.domain, apply=adj, adjoint=self._apply, name=f"{self.name}*")


class CapAverages(FactoredFamily):
    """Averages of a field on a sphere over caps around many centres, as a factored family (module docstring)."""

    def __init__(self, space, radius, centres, profile="raised-cosine", band=None, normalise=True, summand=None,
                 degree=None, name="T"):
        from ..operators.separable import separable
        names = getattr(space, "names", None)
        if names is not None:
            spheres = [k for k, Hk in zip(names, space.summands) if H.kind(getattr(Hk, "backend", None)) == "sphere"]
            if summand is None:
                if len(spheres) != 1:
                    raise ValueError(f"name the summand that carries the field (summand=; sphere summands: {spheres})")
                summand = spheres[0]
            Hs, w = space.summand(summand), space.weights[space.index(summand)]
        else:
            Hs, w = space, 1.0
        if H.require(Hs, "cap averages") != "sphere":
            raise ValueError("cap averages are averages of a field on a sphere (a sphere summand)")
        psi0 = np.atleast_1d(np.asarray(radius, float))
        if np.any(psi0 <= 0) or np.any(psi0 > math.pi):
            raise ValueError("radius: angular radii in (0, pi] (radians)")
        X = directions(centres)
        self.space, self.radius, self.centres, self.profile = space, psi0, X, profile
        self.band, self.normalise, self.summand = band, normalise, summand
        if band is not None:
            degs = sorted({int(l) for l in band})
        else:
            Lmax = int(degree) if degree is not None else 255
            degs = list(range(Lmax + 1))
        L = max(degs)
        lam = np.empty((len(psi0), L + 1))
        norms = np.empty(len(psi0))
        for i, p0 in enumerate(psi0):
            lam[i], norms[i] = HA.funk_hecke(profile, p0, L, normalise)
        if band is None and degree is None:
            rel = np.abs(lam) / np.abs(lam[:, :1])
            big = np.flatnonzero((rel > 1e-16).any(axis=0))
            L = int(big[-1]) if len(big) else 0
            degs, lam = list(range(L + 1)), lam[:, :L + 1]
        self.lam, self.degrees = lam, tuple(degs)
        lab = np.array([(l, j) for l in degs for j in range(2 * l + 1)], int)
        R = Hs.backend.R                         # the average integrates against R^2 dOmega (separable: plan 066)
        R2 = Hs.backend.mass                     # c R^2: the space's measure, for the representers' norms (plan 064)
        tau = separable(space, {summand: [1.0 / R ** 2] * len(lab)} if names is not None else [1.0 / R ** 2] * len(lab),
                        lab, name="tau")
        Y = HA.values(L, X)[:, HA.index(lab[:, 0], lab[:, 1])]
        Phi = np.concatenate([lam[i][lab[:, 0]][None, :] * Y for i in range(len(psi0))], axis=0)
        mass = np.array([float(np.sum(lam[i][degs] ** 2 * (2 * np.asarray(degs) + 1))) / (4 * math.pi)
                         for i in range(len(psi0))])
        exact = norms / (R2 * w)
        trunc = mass / (R2 * w)
        cl = [Claim("T_xi(h) = sum_l lambda_l sum_j Y_lj(xi) h_lj (Funk-Hecke), lambda_l = 2 pi int w P_l", THEOREM,
                    "Funk-Hecke; lambda_l by Gauss-Legendre in psi over the support (2L + 64 nodes, analytic integrand)")]
        if band is not None:
            cl.append(Claim(f"the target is T o P_B, B = {list(degs)} (band-limited: the factors are exact)", DERIVED,
                            "band= given"))
            n2 = np.repeat(trunc, len(X))
        else:
            tail = float(np.max(1.0 - mass / np.maximum(norms, 1e-300)))
            cl.append(Claim(f"factors truncated at degree {L}: exact on fields of degree <= {L}; the target norms "
                            f"||T_xi||^2 = int w^2 are exact, the truncated part of them is {tail:.2g} relative", DERIVED,
                            "Parseval: ||w||^2 = sum_l lambda_l^2 (2l + 1) / (4 pi)"))
            n2 = np.repeat(exact, len(X))
        super().__init__(tau, Phi, name=name, claims=cl, norms2=(n2, cl[-1]))

    def at(self, centres):
        """The same caps around other centres."""
        return CapAverages(self.space, self.radius, centres, self.profile, self.band, self.normalise, self.summand,
                           None if self.band is not None else max(self.degrees), name=self.name)


class RotatedFamily(FactoredFamily):
    """The rotations T_xi of a zonal functional T_0 to many centres (module docstring). functional: a pg.Functional,
    a one-row pg.Functionals, or an element (the representer)."""

    def __init__(self, functional, centres, tol=1e-12, name="T", degree=None):
        from ..realisations.series.symbolic_harmonics import is_symbolic
        from .harmonic_targets import expansion
        from ..operators.separable import separable
        if is_symbolic(functional):
            if degree is None:
                raise Refused("a functional kept symbolic has no degree of its own: give degree= (at least the data's and the "
                              "prior's)", needs=("degree",))
            t0 = getattr(functional, "representer", None) or functional.kernels[0]
            self._init_symbolic(functional, t0, centres, int(degree), tol, name, expansion, separable)
            return
        t0 = getattr(functional, "representer", None) or (functional.kernels[0] if hasattr(functional, "kernels")
                                                     else functional)
        if not isinstance(t0, Function):
            raise TypeError("give a functional (pg.Functional, a one-row pg.Functionals) or its representer")
        space = t0.space
        names = getattr(space, "names", None)
        R = t0.realisation
        parts = list(R.parts) if names is not None else [(R, np.array([0]))]
        spaces = list(space.summands) if names is not None else [space]
        if names is not None:
            from ..realisations.series.direct_sum import realised
            parts = [None if p is None else (realised(H, p[0]), p[1]) for H, p in zip(spaces, parts)]
        weights = list(space.weights) if names is not None else [1.0]
        L = max(H.degree(p[0]) for p in parts if p is not None)
        lab = np.array([(l, j) for l in range(L + 1) for j in range(2 * l + 1)], int)
        radial, off2, tot2 = {}, 0.0, 0.0
        for k, p in enumerate(parts):
            if p is None:
                continue
            Hk, Rk = spaces[k], p[0]
            zonal, o2, a2 = _zonal(Hk, Rk, weights[k])
            off2, tot2 = off2 + o2, tot2 + a2
            items = [zonal[l] for l in lab[:, 0]]
            radial[names[k] if names is not None else 0] = items
        ratio = math.sqrt(off2 / max(tot2, 1e-300))
        if ratio > tol:
            raise ValueError(f"the functional is not zonal: its representer has {ratio:.2g} of its norm outside the m = 0 "
                             "harmonics (rotate it so its axis is the pole)")
        tau = separable(space, radial if names is not None else radial[0], lab, name="tau")
        X = directions(centres)
        self.centres, self.functional = X, functional
        Y = HA.values(L, X)[:, HA.index(lab[:, 0], lab[:, 1])]
        Phi = Y * np.sqrt(4 * math.pi / (2 * lab[:, 0] + 1))[None, :]
        cl = [Claim(f"T_0 is zonal (invariant under rotations about the pole): off-zonal part {ratio:.2g} of its norm",
                    CHECKED, "from the coefficients of its representer"),
              Claim("T_xi = sum_lj sqrt(4 pi/(2l + 1)) Y_lj(xi) <c_l (x) Y_lj, .>", DERIVED, "the addition theorem")]
        super().__init__(tau, Phi, name=name, claims=cl)

    def _init_symbolic(self, functional, t0, centres, L, tol, name, expansion, separable):
        """As __init__, from the coefficients of a target kept symbolic (properties/harmonic_targets.py): its m = 0 rows."""
        from ..realisations.series import legendre as LG
        from ..spaces.harmonic import rows
        M = t0.space
        parts, err = expansion(M, t0, L)
        lab = np.array([(l, j) for l in range(L + 1) for j in range(2 * l + 1)], int)
        keep = rows(L)[1] != 0
        off2, tot2 = 0.0, 0.0
        radial = {}
        for key, pl in parts.items():
            e = LG.sqnorms(pl)
            off2, tot2 = off2 + float(e[keep].sum()), tot2 + float(e.sum())
            zonal = {l: LG.PiecewiseLegendre(pl.breaks, [C[:, [l * l]] for C in pl.coefs], [0.0]) for l in range(L + 1)}
            radial[key] = [zonal[l] for l in lab[:, 0]]
        ratio = math.sqrt(off2 / max(tot2, 1e-300))
        if ratio > max(tol, 10 * err):
            raise ValueError(f"the functional is not zonal: {ratio:.2g} of its coefficients lie outside the m = 0 harmonics (rotate "
                             "it so its axis is the pole)")
        tau = separable(M, radial, lab, name="tau")
        X = directions(centres)
        self.centres, self.functional, self.degree = X, functional, L
        Y = HA.values(L, X)[:, HA.index(lab[:, 0], lab[:, 1])]
        Phi = Y * np.sqrt(4 * math.pi / (2 * lab[:, 0] + 1))[None, :]
        n2 = float(M.norm(t0)) ** 2
        cl = [Claim(f"T_0 kept symbolic, expanded up to degree {L}: exact on fields of degree <= {L}; the target norms are exact, a "
                    "prior above that degree sees only the truncation", DERIVED, "orthonormality of the harmonics"),
              Claim(f"T_0 is zonal: off-zonal part {ratio:.2g} of its coefficients", CHECKED, "from its coefficients"),
              Claim(f"coefficients by Gauss quadrature, estimated {err:.1e} relative against twice the nodes", CHECKED,
                    "realisations/series/symbolic_harmonics.py", key="quadrature"),
              Claim("T_xi = sum_lj sqrt(4 pi/(2l + 1)) Y_lj(xi) <c_l (x) Y_lj, .>", DERIVED, "the addition theorem")]
        FactoredFamily.__init__(self, tau, Phi, name=name, claims=cl, norms2=(np.full(len(X), n2), cl[0]))

    def at(self, centres):
        if getattr(self, "degree", None) is not None:
            return RotatedFamily(self.functional, centres, name=self.name, degree=self.degree)
        return RotatedFamily(self.functional, centres, name=self.name)


def _zonal(Hk, Rk, w):
    """{l: zonal factor of degree l (a number on a sphere, a radial legendre.PiecewiseLegendre on a ball)}, the squared norm of the
    non-zonal part (summed directly, no cancellation) and the total; factors scaled by w and by the density c (rho)
    (separable() divides the kernels by the summand weight and the density again: plan 066)."""
    from ..realisations.series import legendre as LG
    from ..spaces.harmonic import rows
    B = Hk.backend
    kind = H.require(Hk, "rotated families")
    L = H.degree(Rk)
    tot = float(B.sqnorms(Rk)[0])
    keep = rows(L)[1] != 0                                      # the non-zonal rows (m != 0)
    if kind == "sphere":
        c = H.coefficient_columns(B, Rk, [l * l for l in range(L + 1)])[:, 0]
        return {l: float(c[l]) * w * B.c for l in range(L + 1)}, float(B.mass * np.sum(Rk.C[:(L + 1) ** 2][keep] ** 2)), \
            tot
    from ..realisations.series.radial_harmonic import RadialHarmonicSeries, dense
    Rd = dense(Rk)
    out = {l: LG.PiecewiseLegendre(Rd.breaks, [C[:, l * l, :1] * (w * B.c) for C in Rd.coefs], [0.0]) for l in range(L + 1)}
    if B.rho is not None:                                       # the kernel is the representer times rho (plan 066)
        out = {l: LG.multiply(p, B.rho) for l, p in out.items()}
    off = RadialHarmonicSeries(Rd.L, Rd.breaks, [C * keep[None, :, None] for C in Rd.coefs], Rd.err)
    return out, float(B.sqnorms(off)[0]), tot

