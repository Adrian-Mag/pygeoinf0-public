"""Degree-wise operations on harmonic realisations, whatever the domain (plan 009 G1, G2, G4; harmonic_ops.py, merged
into realisations/harmonics.py by plan 063; plan 069 P4 moved that module's continuum half, the harmonics Y_lj, their
conventions and Funk-Hecke, to spaces/harmonic.py): spherical harmonics on a sphere (spherical.SphericalHarmonicSeries) and radial x harmonic
factors on a ball or shell (radial_harmonic.RadialHarmonicSeries, SeparableRadialHarmonics). A backend declares which by its attribute `harmonic` ("sphere",
"radial"); operators/band.py, operators/separable.py and properties/families.py go through here, never through the type
of a domain.
"""
import numpy as np

from ...core.errors import Unsupported


def kind(backend):
    """"sphere", "radial" or None (no harmonic structure in this realisation)."""
    return getattr(backend, "harmonic", None)


def require(space, what):
    k = kind(getattr(space, "backend", None))
    if k is None:
        raise Unsupported(f"{what} acts on spaces realised in spherical harmonics (L2 of a sphere) or in radial x harmonic "
                          f"factors (pg.L2(ball or shell, realisation='radial_harmonics')); {space} is realised "
                          f"{getattr(space, 'realisation', 'otherwise')}")
    return k


def _degrees(L):
    from ...spaces.harmonic import rows
    return rows(L)[0]


def band(backend, R, keep):
    """R with the harmonics of the degrees not in `keep` (a boolean array over l = 0, 1, ...) removed."""
    if kind(backend) == "radial":
        from .radial_harmonic import band as rb
        return rb(R, keep)
    from .spherical import SphericalHarmonicSeries
    deg = _degrees(R.L)
    mask = (deg < len(keep)) & np.asarray(keep, bool)[np.minimum(deg, len(keep) - 1)]
    return SphericalHarmonicSeries(R.L, R.C * mask[:, None], R.err, R.R)


def scale_degrees(backend, R, f):
    """R with its degree-l part multiplied by f[l] (len(f) > R.L)."""
    if kind(backend) == "radial":
        from .radial_harmonic import scale_degrees as rs
        return rs(R, f)
    from .spherical import SphericalHarmonicSeries
    fl = np.asarray(f, float)[_degrees(R.L)]
    return SphericalHarmonicSeries(R.L, R.C * fl[:, None], R.err * float(np.max(np.abs(fl))), R.R)


def degree(R):
    return R.L


def sphere_separable(backend, values, lm, L):
    """The spherical-harmonic family c_i Y_lm[i] on a sphere backend (dense coefficients, (L + 1)^2 x N)."""
    from .spherical import SphericalHarmonicSeries
    lm = np.asarray(lm, int)
    C = np.zeros(((L + 1) ** 2, len(lm)))
    C[lm, np.arange(len(lm))] = np.asarray(values, float)
    return SphericalHarmonicSeries(L, C, np.zeros(len(lm)), backend.R)


def coefficient_columns(backend, R, lm):
    """For a sphere realisation, the coefficients of the rows lm of every member: (len(lm), N)."""
    if kind(backend) != "sphere":
        raise Unsupported("coefficients as numbers exist for spherical-harmonic realisations")
    lm = np.asarray(lm, int)
    out = np.zeros((len(lm), R.N))
    ok = lm < R.C.shape[0]
    out[ok] = R.C[lm[ok]]
    return out
