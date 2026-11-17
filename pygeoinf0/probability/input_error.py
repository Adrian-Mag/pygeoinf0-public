"""What a Gaussian measure's moments inherit from the stated errors of the representers they are built from (plan 054,
the general rule: sensitivity x stated input error).

A realised representer k~ of a functional carries its stated error ||k~ - k|| <= e (Family.errors(), Function.error;
estimates where a realisation truncates, as the sphere's harmonic degree cap does). Two places use such representers:

  - the forward operator G of a posterior, G x = (<k_i, x>)_i: ||G~ - G|| <= ||e|| sqrt(||W||) =: e_G from H to the data
    space D (metric W), since |<k~_i - k_i, x>| <= e_i ||x||;
  - the element u of a variance Var <x, u> = <C u, u>, a point's or a set's representer.

First order in e_G (S = G C G* + C_e, z = S^-1 r, m = C G* z, K = C G*, y_u = S^-1 G C u):

    mean        dm = C dG* z - K S^-1 (dG m + G C dG* z):     ||dm|| <= e_G (||C|| ||z|| + ||K S^-1|| (||m|| + ||K|| ||z||))
    variance    d<C_post u, u> = -2 <dG C u, y_u> + <dS y_u, y_u> = -2 <dG C_post u, y_u>, y_u = S^-1 G C u with the
                PRIOR's C (<dS y, y> = 2 <dG C G* y, y> and C u - C G* y_u = C_post u):
                <= 2 e_G ||C_post u|| ||y_u|| + e_G^2 ||C|| ||y_u||^2 (the second: dG C dG*; the second-order terms
                amplified by S^-1 are not included: first order, as the rule's E1)
    the element 2 ||C_post u|| e_u + ||C|| e_u^2 (C_post <= C: the prior's norm bounds the posterior's)

with ||C_post u|| <= sqrt(||C|| <C_post u, u>) (positive semidefinite, C_post <= C), so no norm is formed in the space
for a variance. y_u is the PRIOR's: S^-1 G C_post u = S^-1 C_e y_u is smaller by about the noise over the signal, and a
bound from it understated the variances of exact elements by up to 2e5 (plan 054 varerr, measured). For
C = c I (a cylindrical prior) the operator norms are formed on the data space (||K||^2 = c ||G C G*||), else in
the space; both by power iteration (SAFETY times: operators/error.power_estimate). A bound where the
stated errors are bounds, an estimate where they are (Family.errors() is one). Judged by the general rule: a
PrecisionWarning where the bound exceeds TOL of the value, a refusal with the choices named where it reaches the value (no
correct digit). Nothing is computed where every stated error is 0 (exact representers: kernels in closed form) or unknown
(NaN: a formula in H^s)."""
import math
import warnings

import numpy as np

from ..spaces.elements import Family
from ..core.errors import PrecisionWarning, Refused
from ..operators.error import power_estimate, scaled_identity

CHOICES = ["the representers realised more accurately: a smaller tol of the space, or a space whose realisation reaches "
           "the needed degree (the sphere's harmonic cap truncates the representers of a weak Sobolev order)",
           "a smoother prior (a larger Sobolev order s): its representers converge faster",
           "a narrower prior or a larger noise covariance: a mathematical choice"]


def operator_error(G):
    """(e_G, how) with ||G~ - G|| <= e_G from G's domain to its codomain, from its kernels' stated errors, or None (no
    stated error beyond rounding, or errors unknown)."""
    from ..operators.functionals import Functionals, symbolic_representers
    if not isinstance(G, Functionals) or symbolic_representers(G.kernels):
        return None
    err = np.asarray(G._kerr, float)
    if not err.size or not np.all(np.isfinite(err)) or err.max(initial=0.0) <= 0:
        return None
    D = G.codomain
    w = 1.0 if D.is_standard else (float(np.max(D.weights)) if D.weights is not None else
                                   float(np.linalg.eigvalsh(D.apply_metric(np.eye(D.dim))).max()))
    e = float(np.linalg.norm(err)) * math.sqrt(w)
    return e, (f"the kernels of {G.name} realised with errors up to {err.max():.2g} (estimates, Family.errors()): "
               f"||dG|| <= {e:.3g}")


def element_errors(u):
    """The stated errors ||u~ - u|| of an element or a family's members (an array), or None where none is stated."""
    if isinstance(u, Family):
        e = u.errors() if hasattr(u, "errors") else None
    else:
        e = getattr(u, "error", None)
    if e is None:
        return None
    e = np.atleast_1d(np.asarray(e, float))
    return e if np.all(np.isfinite(e)) and e.max(initial=0.0) > 0 else None


def cov_norm(mu, probe):
    """A bound (or a power-iteration estimate) of ||C|| for mu's covariance: exact for c I; a posterior's by its prior's
    (C_post <= C); else from probe (kept on mu: it costs applications of C and norms in the space)."""
    got = getattr(mu, "_cnorm", None)
    if got is not None:
        return got
    from .posterior_covariance import PosteriorCovariance
    C = mu.covariance
    C = C.C if isinstance(C, PosteriorCovariance) else C
    c = scaled_identity(C)
    mu._cnorm = abs(c) if c is not None else power_estimate(C, probe, mu.space)
    return mu._cnorm


class InputError:
    """A posterior's inherited input error: what its mean and variances carry from e_G (module docstring)."""

    def __init__(self, space, C, K, G, GC, Sinv, z, mean, size, eG, how):
        from .posterior_covariance import _norm_bound
        self.C, self.G, self.Sinv, self.eG, self.how = C, G, Sinv, eG, how
        D = G.codomain
        c = scaled_identity(C)
        if c is not None:                   # C = c I: ||K||^2 = c ||G C G*||, ||K S^-1||^2 = c ||S^-1 G C G* S^-1||, on D
            self.Cn = abs(c)
            self.Kn = (abs(c) * power_estimate(GC, z, D)) ** 0.5
            self.KSn = (abs(c) * power_estimate(lambda w: Sinv(GC(Sinv(w))), z, D)) ** 0.5
        else:                               # in the space (applications of C and norms of its elements)
            self.Cn = power_estimate(C, mean if size > 0 else K(z), space)
            self.Kn = _norm_bound(space, K, lambda w: G(C(w)), D, z)
            self.KSn = _norm_bound(space, lambda w: K(Sinv(w)), lambda x: Sinv(G(C(x))), D, z)
        zn = float(D.norm(z))
        self.mean = eG * (self.Cn * zn + self.KSn * (size + self.Kn * zn))

    def variance(self, u, cn, yn=None):
        """The bound on |d Var <x, u>| from e_G for the members of u (an element or a family) with ||C_post u|| <= cn; yn:
        bounds of ||S^-1 G C u||_D, C the PRIOR's, where the caller has them (probability/scale.py: from its own solves),
        else formed here (one application of the prior's C, of G and of S^-1)."""
        if yn is None:
            Y = self.Sinv(self.G(self.C(u)))
            yn = np.atleast_1d(np.asarray(self.G.codomain.norm(Y), float))
        yn = np.atleast_1d(np.asarray(yn, float))
        return 2.0 * self.eG * cn * yn + self.eG ** 2 * self.Cn * yn ** 2


def variance_errors(mu, u, Cu, var, yn=None, refuse=True):
    """The stated errors of the variances var = <C u, u> (an array) from u's own stated errors and, for a posterior, from
    its forward operator's (module docstring); None where nothing is stated. ||C u|| is bounded by sqrt(||C|| var)
    (C positive semidefinite: ||C u||^2 <= ||C|| <C u, u>), with no norm formed in the space. Judged by the general
    rule. yn: bounds of ||S^-1 G C u||_D given by the caller (a posterior at scale, probability/scale.py, whose low-rank
    route gives upper bounds of them: there a warning, not a refusal, refuse=False)."""
    eu = element_errors(u)
    inp = getattr(mu, "_input", None)
    if eu is None and inp is None:
        return None
    v = np.maximum(np.atleast_1d(np.asarray(var, float)), 0.0)
    Cn = inp.Cn if inp is not None else cov_norm(mu, Cu if not isinstance(Cu, Family) else Cu[0])
    cn = np.sqrt(Cn * v)
    err, how = np.zeros_like(v), []
    if eu is not None:
        err = err + 2.0 * cn * eu + Cn * eu ** 2
        how.append(f"the element's representers (errors up to {eu.max():.2g}, estimates)")
    if inp is not None:
        err = err + inp.variance(u, cn, yn)
        how.append(inp.how)
    judge("variance", mu.name, v, err, "; ".join(how), refuse)
    return err


def judge(what, name, value, err, how, refuse=True):
    """The general rule on stated input errors: a PrecisionWarning where err > TOL |value| somewhere, a refusal where err
    reaches |value| (no correct digit). refuse=False: a warning there too (the posterior mean as an element: its error in
    the space's norm says how far the realised element is from the one meant, while what is computed from it, a value at
    a point or a variance, is judged where it is computed)."""
    from ..core.precision import assess, fewer
    bad, none, worst = assess(np.atleast_1d(value), np.atleast_1d(err))
    if not np.any(bad):
        return
    msg = (f"{name}: {fewer(int(bad.sum()), bad.size, what + '(s)')} under the stated input "
           f"error (relative up to {worst:.2g}: {how}; probability/input_error.py)")
    if refuse and np.any(none):
        raise Refused(msg + "; some hold no correct digit", needs=CHOICES)
    warnings.warn(msg + ". " + "; ".join(CHOICES), PrecisionWarning, stacklevel=4)

