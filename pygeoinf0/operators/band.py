"""Functions of the angular Laplacian on spaces of fields on spheres, balls, shells and their direct sums, and band
projections (plan 009 G1; EXP-030 DESIGN G1, NOTES.md section 3).

    P_B = pg.BandProjection(M, degrees=range(2, 15, 2))    the orthogonal projection onto the harmonics of degree in B
    A = AngularFunction(M, lambda l: (1 + l * (l + 1)) ** -1.5)   f(-Delta_S): degree l multiplied by f(l)

On L2 of a sphere, ball or shell (and on a direct sum of them) the angular Laplacian -Delta_S acts on the degree-l
harmonics as l(l + 1), so f(-Delta_S) multiplies the degree-l part by f(l) (functional calculus; defined on the continuum
space, without a realisation). It commutes with every rotation and with every radial multiplication, and is self-adjoint
for real f (THEOREM). P_B = 1_B(-Delta_S) is an orthogonal projection: idempotent, self-adjoint, norm 1 (unless B is
empty). Applied in the harmonic realisations (spherical harmonics on a sphere, radial x harmonic factors on a ball or
shell: realisations/series/harmonic_degrees.py) it is exact: the coefficients of the other degrees are set to zero. On a direct sum
it acts on the listed summands (default: all) and leaves the others unchanged.
"""
import numpy as np

from ..core.claims import CHECKED, THEOREM, Claim
from ..spaces.elements import Family, Function
from ..realisations.series import harmonic_degrees as H
from .linear import LinearOperator
from ..spaces.identity import identical, require_realised


class AngularFunction(LinearOperator):
    """f(-Delta_S) on a harmonic space or a direct sum of them (module docstring); f maps arrays of degrees l to
    real numbers."""

    why_linear = "each spherical-harmonic coefficient multiplied by the fixed number f(l)"

    def __init__(self, space, f, summands=None, name="f(-Δ_S)"):
        super().__init__(space, space, name=name)
        self.f = f
        names = getattr(space, "names", None)
        if names is None:
            H.require(space, name)
            self.parts = {0: f}
        else:
            keys = list(names) if summands is None else list(summands)
            self.parts = {}
            for key in keys:
                k = space.index(key)
                H.require(space.summands[k], f"{name} on the summand {names[k]!r}")
                self.parts[k] = f
        self._sa = (True, THEOREM, "a real function of the self-adjoint angular Laplacian (functional calculus)")
        self._claims += [Claim("linear", THEOREM, "a function of the angular Laplacian", key="linear"),
                         Claim("commutes with rotations", THEOREM, "a function of the angular Laplacian, which commutes "
                               "with every rotation; Schur's lemma gives the block form (EXP-030 NOTES.md section 3)",
                               key="rotation invariant")]
        self._degree_claims()

    def _degree_claims(self):
        v = np.asarray(self.f(np.arange(4096)), float)
        self._claims.append(Claim(f"bounded, norm <= {np.max(np.abs(v)):.4g} on degrees l < 4096", CHECKED,
                                  "sup |f(l)| over the sampled degrees", key="bounded"))
        if np.all(v >= 0):
            self._pos = ("psd", CHECKED, "f(l) >= 0 on the sampled degrees")

    def degree_values(self, L):
        return np.asarray(self.f(np.arange(L + 1)), float)

    def _transform(self, backend, R):
        return H.scale_degrees(backend, R, self.degree_values(H.degree(R)))

    def _apply(self, x):
        if not isinstance(x, (Function, Family)) or x.space != self.domain:
            raise TypeError(f"{self.name} acts on elements of {self.domain}")
        require_realised(x.space, self.domain, x.name)
        sp = self.domain
        R = x.realisation
        if getattr(sp, "names", None) is None:
            out = self._transform(sp.backend, R)
        else:
            from ..realisations.series.direct_sum import DirectSumRealisation, realised
            parts = list(R.parts)
            for k, p in enumerate(parts):
                if p is not None and k in self.parts:
                    parts[k] = (self._transform(sp.summands[k].backend, realised(sp.summands[k], p[0])), p[1])
            out = DirectSumRealisation(parts, R.N)
        if isinstance(x, Function):
            return Function(sp, pl=out, name=f"{self.name}{x.name}")
        return Family(sp, out, name=x.name)

    def _make_adjoint(self):
        return self


class BandProjection(AngularFunction):
    """P_B = 1_B(-Delta_S): the orthogonal projection onto the harmonics of the degrees in B (module docstring)."""

    def __init__(self, space, degrees, summands=None, name="P_B"):
        B = sorted({int(l) for l in degrees})
        if any(l < 0 for l in B):
            raise ValueError("degrees are non-negative integers")
        self.degrees = tuple(B)
        keep = np.zeros(max(B, default=-1) + 1, bool)
        keep[B] = True
        self.keep = keep

        def ind(l):
            l = np.asarray(l, int)
            return ((l < len(keep)) & keep[np.minimum(l, max(len(keep) - 1, 0))]).astype(float) if len(keep) else \
                np.zeros(np.shape(l))
        super().__init__(space, ind, summands, name=name)

    def _degree_claims(self):
        self._claims.append(Claim("an orthogonal projection: idempotent, self-adjoint, norm "
                                  f"{1 if self.degrees else 0}", THEOREM, "an indicator function of the self-adjoint "
                                  "angular Laplacian", key="bounded"))
        self._pos = ("psd", THEOREM, "an orthogonal projection")

    def _transform(self, backend, R):
        return H.band(backend, R, self.keep)

    def _compose(self, other):
        if isinstance(other, BandProjection) and identical(other.domain, self.domain) \
                and other.parts.keys() == self.parts.keys():
            both = sorted(set(self.degrees) & set(other.degrees))
            names = getattr(self.domain, "names", None)
            return BandProjection(self.domain, both, None if names is None else [names[k] for k in self.parts],
                                  name=f"{self.name}{other.name}")
        return super()._compose(other)

    def __repr__(self):
        return f"{self.name}: projection onto degrees {list(self.degrees)} of {self.domain}"
