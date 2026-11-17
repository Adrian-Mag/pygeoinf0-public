"""The general rule (plan 054; optimisers/dual/exactness.py) in the reduce route: bounds solved in the coordinates of the joint
Gram matrix J of [directions, data representers] (optimisers/dual/reduction.reduce): FeasibleSet.bound and FeasibleImage.bound where
no data-space decomposition applies (ball_ball, ball_exact, none_ball, ball_slab), and oCSOLA's endpoints. A bound is
EXACT only if (its sensitivity to the stated input errors) x (their stated size) <= core.precision.TOL max(1, |value|); otherwise
EXACT becomes BRACKET and the bracket is widened by WIDEN x E, with the note naming the choices. oCSOLA's endpoints are
numbers, not Bounds: an endpoint is moved outwards by WIDEN x E and a PrecisionWarning names the choices.

The input error (stated). J enters through its correlation matrix Cr = J / (d d^T), d_i = sqrt(J_ii), and the
reduction is exact for Cr + dC with
    ||dC||_2 <= eps_J = C eps q lam_max(Cr)          (q = the size of J)
its backward error: it drops eigenvalues <= eps q lam_max, and the rounding of J's entries and of the eigendecomposition
are of that size. The datum and the noise radius: their rounding, componentwise (so that a datum's units do not
enter), C eps (n sum_i |e~_i| |l~_i| + rho ||l~||) (e~, l~ the whitened datum and multiplier; a slab: C eps |tau| (|s| +
w)). Nothing here reads what produced J or what structure it has.

The sensitivity, first order. With representers R (J = R^T R), the bound's saddle point has a model m* = R w and dual
weights z on J's columns (z = (the direction's weights, -the data multiplier)), and a change dJ moves the value by
w^T dJ z / 2 (the envelope theorem; with a prior ball m* = r R z / v and this is r z^T dJ z / (2 v), the term of
exactness.first_order; with no prior the constraint is R z = 0 and its multiplier is the model). So, without the 1/2
(as in exactness.py),
    E1 = eps_J ||Lam^(-1/2) y|| ||d z|| + (the datum's term),
y the witness in the reduction's coordinates (y = Lam^(1/2) U^T (d w), so Lam^(-1/2) y = U^T (d w), the least ||d w||
over the w with R w = m*) and U, Lam the kept eigenpairs of Cr. d z is not projected on U: dC acts on all of it, and
with no prior d z lies in the null space of Cr (J z = 0). Without a witness, and with a prior ball, the dual form
eps_J r ||d z||^2 / v, v^2 = z^T J z; inf where v^2 <= eps_J ||d z||^2 (v has no correct digit).

The columns' own stated errors (plan 054, truncation round; optimisers/dual/exactness.py, "The representers' own stated
errors"): where the caller states them (reduction.Reduction.cerr: the directions', then the data representers'; FeasibleSet,
FeasibleImage, oCSOLA), eps_J gains 2 ||e / d|| sqrt(lam_max) + ||e / d||^2 (stated_part), so the first-order estimate,
the re-solves' perturbations and the lifted eigenvalues are of the stated size. With no prior a target built from the
data functionals is exempt (its bound does not depend on the kernels; exactness.built).

Re-solves. First order fails near degeneracy (measured in the spectral route up to 160 x low), so where E1 > GRAY x TOL
max(1, |value|), or E1 is not available, the bound is solved again on RESOLVES matrices J' = d (Cr + Z) d^T, Z random
symmetric with ||Z||_2 = eps_J (seed fixed: reproducible), each reduced again. A dropped direction is seen neither by
first order nor by those (the cut is at the size of the perturbation, so it stays dropped), while the stated error
admits its eigenvalue anywhere in [0, eps_J]. So where the reduction dropped directions the re-solves always run, with
one more, the lifted problem: every dropped eigenvalue at eps_J, all directions kept, except those stated zero, q less
a bound on rank J (feasible._stated_zeros): at most dim M (structure), and at most the data's rank plus the number of
targets, plus none where the targets are declared or built estimable (core/estimable.py: in the data's span); the
data's rank is n less their floor directions where they are declared exactly dependent (K.dependent). This lifts a
target in the data's span to rounding too: with no prior its lifted bound is +inf, so a target there that is neither
declared nor built estimable is not EXACT (E = inf), as in the spectral route (with exact data or no prior the data's
own floor directions are refused before, reduction.floor_refusal). Measured (the fix round's near-dependent problems): lifting only the data's
dependencies left 2 of 77 widened brackets missing the exact value (by 26 and 3 TOL); lifting all left none. E =
max(E1, the largest change), inf where a re-solve is refused (the rounding floor), empty or unbounded: the stated error
then admits a different answer. The constants C, GRAY, RESOLVES, WIDEN are those of optimisers/dual/exactness.py (measured
choices)."""
import math

import numpy as np

from ...core.errors import Refused
from ...core.precision import TOL
from . import exactness, reduction


def eps_J(red):
    """The stated backward error of the reduction red = info["reduction"] of reduction.reduce, in correlation units, with
    the columns' stated realisation errors where red states them (stated_part)."""
    J, d, lam, U, keep, n_dirs = red
    return exactness.C * reduction.EPS * len(d) * max(float(lam.max(initial=0.0)), 1e-300) + stated_part(red)


def stated_part(red):
    """The columns' own stated errors in correlation units: with ||r~_j - r_j|| <= e_j and R^ = R / d (||R^||^2 =
    lam_max), ||dC||_2 <= 2 ||e / d|| sqrt(lam_max) + ||e / d||^2 (optimisers/dual/exactness.py, "The representers' own stated
    errors"); 0 where none is stated, inf where a stated error is infinite or a column of norm 0 carries one."""
    cerr = getattr(red, "cerr", None)
    if cerr is None:
        return 0.0
    J, d, lam, U, keep, n_dirs = red
    cerr = np.asarray(cerr, float)
    if not np.any(cerr):
        return 0.0
    dd = np.sqrt(np.clip(np.diag(J), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        et = np.where(cerr > 0, cerr / dd, 0.0)
    n = float(np.linalg.norm(et))
    if not math.isfinite(n):
        return math.inf
    return 2.0 * n * math.sqrt(max(float(lam.max(initial=0.0)), 0.0)) + n * n


def with_errors(info, cerr):
    """Attach the columns' stated errors (an array over J's columns, or None) to info["reduction"]."""
    red = info.get("reduction")
    if red is not None and cerr is not None and np.any(np.asarray(cerr, float)):
        if not isinstance(red, reduction.Reduction):
            red = info["reduction"] = reduction.Reduction(red)
        red.cerr = np.asarray(cerr, float)
    return info


def first_order(red, z, y=None, r=math.inf, extra=0.0):
    """E1 (module docstring) for dual weights z on J's columns, the witness y (reduction coordinates) and the prior
    radius r; extra: the datum's term. inf where neither form applies."""
    J, d, lam, U, keep, n_dirs = red
    lam = lam[keep]
    if z is None or not np.all(np.isfinite(z)):
        return math.inf
    e = eps_J(red)
    dz = d * np.asarray(z, float)                  # not projected on U: dC acts on all of it (with no prior J z = 0)
    nu2 = float(dz @ dz)
    if y is not None and len(y) == len(lam) and np.all(np.isfinite(y)):
        lead = e * float(np.linalg.norm(np.asarray(y, float) / np.sqrt(lam))) * math.sqrt(nu2)
    elif math.isfinite(r):
        v2 = float(z @ J @ z)                      # r sqrt(e nu2) where v has no correct digit (exactness.py)
        lead = r * (e * nu2 / math.sqrt(v2) if v2 > e * nu2 else math.sqrt(e * nu2))
    else:
        return math.inf
    return lead + extra


def perturbed(red, structural=0):
    """The perturbed reductions (Yd, B, info) of the re-solves (module docstring): RESOLVES of J' = d (Cr + Z) d^T, and,
    where the reduction dropped directions, the lifted one: the dropped eigenvalues of Cr at eps_J, the other end of the
    stated error (dropped, they are taken as 0), all directions kept. structural: how many of the dropped ones are
    stated zero (by structure, or declared: feasible._stated_zeros), left at 0 (the smallest)."""
    J, d, lam, U, keep, n_dirs = red
    rng = np.random.default_rng(exactness.SEED)
    q, e, dd = len(d), eps_J(red), np.outer(d, d)
    Cr = J / dd
    out = []
    for _ in range(exactness.RESOLVES):
        Z = rng.standard_normal((q, q))
        Z = Z + Z.T
        nz = np.linalg.norm(Z, 2)
        info = {}
        Yd, B = reduction.reduce((Cr + Z * (e / (nz if nz > 0 else 1.0))) * dd, n_dirs, info)
        out.append((Yd, B, info))
    lifted = _lifted(red, Cr, e, structural)
    if lifted is not None:
        out.append(lifted)
    return out


def _lifted(red, Cr, e, structural):
    """The lifted reduction (module docstring), or None where nothing was dropped beyond the `structural` stated zero:
    every other dropped eigenvalue of Cr given the value eps_J, all those directions kept."""
    J, d, lam, U, keep, n_dirs = red
    lift = np.flatnonzero(~keep)[max(int(structural), 0):]        # eigh: ascending, so the stated zeros go first
    if not len(lift):
        return None
    lam2 = np.where(keep, lam, 0.0)
    lam2[lift] = e
    live = lam2 > 0
    Y = np.sqrt(lam2[live])[:, None] * U[:, live].T * d
    return Y[:, :n_dirs], Y[:, n_dirs:].T, {"dropped": int((~live).sum()), "size": len(d)}


def judge(value, E1, resolve, cases, dropped=False):
    """E for a bound of the given value with first-order estimate E1; resolve(Yd, B, info) -> the value of the same
    bound in a perturbed reduction, cases: perturbed(red) (or a callable giving it, called only when needed). dropped:
    the reduction dropped directions, so the re-solves run whatever E1 (first order does not see a dropped one)."""
    scale = TOL * max(1.0, abs(value))
    if not math.isfinite(value) or (E1 <= exactness.GRAY * scale and not dropped):
        return E1
    worst = 0.0
    for Yd, B, info in (cases() if callable(cases) else cases):
        try:
            v = resolve(Yd, B, info)
        except Refused:
            return math.inf
        worst = max(worst, abs(v - value)) if math.isfinite(v) else math.inf
    return max(E1 if math.isfinite(E1) else 0.0, worst)


WHAT = "the reduction's backward error, eps q lam_max of the joint Gram's correlation matrix"


def columns(info, dirs, G, r, how):
    """Attach the joint Gram's columns' stated errors to info["reduction"]: the directions' (dirs: their map T, an
    array of their errors, or None), then G's kernels'; none for a target built from the data functionals with no prior
    (exactness.built). Returns how (for the caller's next line)."""
    from ...core.estimable import BUILT
    from ...operators.linear import LinearOperator
    red = info.get("reduction")
    if red is None:
        return how
    if not math.isfinite(r) and how == BUILT:
        if isinstance(red, reduction.Reduction):
            red.cerr = None
        return how
    de = exactness.kernel_errors(dirs) if isinstance(dirs, LinearOperator) else dirs
    nd, ge = red[5], exactness.kernel_errors(G)
    de = np.zeros(nd) if de is None else np.broadcast_to(np.asarray(de, float), (nd,))
    if ge is None and not np.any(de):
        if isinstance(red, reduction.Reduction):
            red.cerr = None
        return how
    with_errors(info, np.r_[de, np.zeros(len(red[1]) - nd) if ge is None else ge])
    return how


def reduce_stated(J, n_dirs, info, dirs, G, r, how):
    """reduction.reduce(J, n_dirs, info) with the columns' stated errors attached (columns)."""
    out = reduction.reduce(J, n_dirs, info)
    columns(info, dirs, G, r, how)
    return out


def what(red=None):
    """The input errors a note names: the reduction's backward error, and the columns' stated errors where red has
    them."""
    sp = 0.0 if red is None else stated_part(red)
    return WHAT + (f"; the representers' stated realisation errors ({sp:.2g} in correlation units, estimates)"
                   if sp else "")


def apply(value, lower, upper, status, note, E, dropped=False, red=None):
    """(lower, upper, status, note) after the rule for a bound whose E is given; dropped: the reduction dropped
    directions (the note then names the declaration); red: the reduction (its stated errors named in the note)."""
    if not E > TOL * max(1.0, abs(value)):
        return lower, upper, status, note
    W = exactness.WIDEN * E
    note = (note + exactness.note(E, value, what(red), "optimisers/dual/exactness_reduce.py")).lstrip("; ")
    if dropped:
        note += ("; or, where the representers are exactly dependent at the rounding floor: a target in the data's "
                 "span built from the data functionals or declared (estimable=True; core/estimable.py), duplicate "
                 "data declared (K.dependent = True)")
    return lower - W, upper + W, ("BRACKET" if status == "EXACT" else status), note
