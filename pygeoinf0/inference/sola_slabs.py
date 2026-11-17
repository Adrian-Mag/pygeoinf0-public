"""SOLA's slabs (moved out of convex/ by plan 070 P4: the oCSOLA and SOLA pieces go to their methods):

    slab_family(D, X, sigma, mass=1.0, centre=None)   SOLA's noise slab per property, {e : |x_j . (e - centre)| <= mass
                                                      sigma_j} (x_j the columns of X, SOLA's estimator rows): noise
                                                      sets for BGP over the region of one fixed estimator (pg.inference.
                                                      slab_family; was pg.convex.slab_family)
    ocsola_ends(J, s, w, r, shift_t, cerr=None)       oCSOLA's [lower, upper] over a norm-ball prior and one slab, by the
                                                      one-dimensional dual (optimisers/dual/reduction.py slab_dual) in
                                                      the coordinates of the 2 x 2 Gram J, with the general rule of the
                                                      reduce route (optimisers/dual/exactness_reduce.py)

oCSOLA's endpoints are numbers, not Bounds: an endpoint is moved outwards by WIDEN x E where E exceeds TOL max(1, |end|),
and a PrecisionWarning names the choices (inference/methods.py ocsola).
"""
import warnings

import numpy as np

from ..core.errors import PrecisionWarning
from ..core.precision import TOL
from ..optimisers.dual import exactness, reduction
from ..optimisers.dual import exactness_reduce as xr


def slab_family(D, X, sigma, mass=1.0, centre=None):
    """[Slab_j]: SOLA's noise slab of property j, {e : |x_j . (e - centre)| <= mass sigma_j} (x_j the columns of X, a
    plain-coordinate row: the slab's normal is W^-1 x_j, so <normal, e>_D = x_j . e)."""
    from ..convex.sets import Slab
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != D.dim:
        X = X.T if X.shape[1] == D.dim else None
    if X is None:
        raise ValueError(f"X needs {D.dim} rows (the data) and one column per property")
    sig = np.broadcast_to(np.asarray(sigma, float), (X.shape[1],))
    return [Slab(D, D(D.solve_metric(X[:, j])), mass * float(sig[j]), centre, name=f"slab {j}") for j in range(X.shape[1])]


def ocsola_ends(J, s, w, r, shift_t, cerr=None):
    """oCSOLA's [lower, upper] from the 2 x 2 Gram J of (t, r~), the estimate s (shifted), half-width w and prior
    radius r, with the rule: each end moved outwards by WIDEN x E where E exceeds TOL max(1, |end|), with a
    PrecisionWarning."""
    info = {}
    Yt, B = reduction.reduce(J, 1, info)
    red = xr.with_errors(info, cerr).get("reduction")
    out, moved = [], []
    for sign in (1.0, -1.0):
        val, tau = reduction.slab_dual(sign * Yt[:, 0], B[0], s, w, r)
        value = val + sign * shift_t
        if red is None:                                         # a reduction that does not state its backward error
            out.append(value)
            continue
        z = np.array([sign, -tau])
        E1 = xr.first_order(red, z, None, r, exactness.C * reduction.EPS * abs(tau) * (abs(s) + w))
        def again(Y2, B2, _, sign=sign):
            return reduction.slab_dual(sign * Y2[:, 0], B2[0], s, w, r)[0] + sign * shift_t
        E = xr.judge(value, E1, again, lambda: xr.perturbed(red), not red[4].all())
        if E > TOL * max(1.0, abs(value)):
            value += exactness.WIDEN * E
            moved.append((("upper", "lower")[sign < 0], E))
        out.append(value)
    if moved:
        ends = "; ".join(f"the {k} end moves by up to {E:.2g} under the stated input errors ({xr.what(red)}, and the "
                         "rounding of the estimate)" for k, E in moved)
        warnings.warn(f"oCSOLA: {ends}, more than TOL max(1, |end|); moved outwards by {exactness.WIDEN:g} x that "
                      "(inference/sola_slabs.py, the rule of optimisers/dual/exactness_reduce.py). " + exactness.CHOICES,
                      PrecisionWarning, stacklevel=3)
    return -out[1], out[0]
