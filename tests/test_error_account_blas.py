"""Plan 054, verdicts that must not depend on the linear-algebra library (the Mac's integration: Accelerate gave PRIMAL
where europa's OpenBLAS gave BRACKET, and EXACT or BRACKET by library). Another library rounds the last bit of a
product differently; here that is emulated by a componentwise relative change of the kernel matrix and the datum by up
to eps / 2 (_problem(..., rounding=k)), and reproduced on europa with OPENBLAS_CORETYPE=Sandybridge or Prescott. Each
test is checked against an exact answer and has a planted-defect control.

    optimisers/dual/reduction.residual_misfit    the misfit ||d - B y||^2 formed from the residual ||dp - s y||^2 + f0. The expansion
                                   ||d||^2 - 2 bb.y + s2.y^2 rounds at eps ||d||^2: at ||d|| / rho = 6e4 (noise 1e-3)
                                   the witness it located was off the noise ball by up to 9e-7 relative, against the
                                   1e-7 at which SpectralBall._finish judges it, so the lower end was finite or -inf
                                   (BRACKET or PRIMAL) by library
    optimisers/dual/exactness.first_order   where v has no correct digit (e2 >= v^2: an estimable target, a_perp^2 a few ulps
                                   either side of 0) the change of r v is at most r sqrt(e2), not inf: E was then the
                                   three re-solves' alone, 1e-14 or 6e-7 by where their rounding fell (EXACT or BRACKET)
References: the exact ends of P2's Gaussian-kernel problems (tests/test_error_account_rule.py, mpmath at 60 digits)."""
import math
import sys
import warnings

import numpy as np

from pygeoinf0.core.precision import TOL
from pygeoinf0.optimisers.dual import exactness, reduction, spectral

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_error_account_rule import EXACT, _problem  # noqa: E402

ROUNDINGS = range(10)


def _witness_misfits(case, monkeypatch):
    """(statuses of both ends, |m - rho| / rho of every witness judged) over ROUNDINGS."""
    seen, orig = [], spectral.SpectralBall._finish

    def finish(self, value, y, x, a_pos, a_perp, const):
        if y is not None and math.isfinite(self.r):
            m = math.sqrt(max(float(np.sum((self.dp - self.s * y[:len(self.s)]) ** 2)) + self.f0, 0.0))
            seen.append(abs(m - self.rho) / self.rho)
        return orig(self, value, y, x, a_pos, a_perp, const)
    monkeypatch.setattr(spectral.SpectralBall, "_finish", finish)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = [tuple(b.status for b in _problem(*case, rounding=k).bounds(0)) for k in ROUNDINGS]
    return st, np.array(seen)


def test_the_witness_sits_on_the_noise_ball(monkeypatch):
    """(0.15, 20, block), ||d|| / rho = 6e4: every witness within 1e-10 of the noise ball (judged TRUE), the same
    statuses under every rounding and no PRIMAL; the ends hold the exact ones. Control: the misfit by the expansion
    (the former form) puts witnesses beyond 1e-7 and gives PRIMAL under some rounding."""
    case = (0.15, 20, "block")
    st, mis = _witness_misfits(case, monkeypatch)
    assert mis.max() <= 1e-10 and len(set(st)) == 1 and "PRIMAL" not in st[0]
    for k in ROUNDINGS[:3]:
        for b, ref in zip(_problem(*case, rounding=k).bounds(0), EXACT[case]):
            assert b.lower <= ref <= b.upper
    monkeypatch.setattr(reduction, "residual_misfit",
                        lambda bb, s2, dd, res=None: lambda y: dd - 2.0 * (bb @ y) + s2 @ (y * y))
    st0, mis0 = _witness_misfits(case, monkeypatch)
    assert mis0.max() > 1e-7 and any("PRIMAL" in s for s in st0)


def test_an_estimable_targets_status_does_not_depend_on_rounding(monkeypatch):
    """(0.1, 10, estimable): BRACKET at both ends under every rounding, with E >= the actual error (up to 6e-7) and
    the exact ends held; E's first-order part alone (no re-solves) is finite and covers it. Control: E1 = inf where v
    has no correct digit (the former form) leaves E to the re-solves, and without them E = 0: EXACT ends that miss."""
    case = (0.1, 10, "estimable")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bs = [_problem(*case, rounding=k).bounds(0) for k in ROUNDINGS]
    for pair in bs:
        for b, ref in zip(pair, EXACT[case]):
            assert b.status == "BRACKET" and b.lower <= ref <= b.upper and b.error >= abs(b.value - ref)
    gray = exactness.GRAY
    monkeypatch.setattr(exactness, "GRAY", math.inf)                        # first order alone
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b1 = [_problem(*case, rounding=k).bounds(0) for k in ROUNDINGS]
    assert all(b.status == "BRACKET" and b.error >= abs(b.value - ref)
               for pair in b1 for b, ref in zip(pair, EXACT[case]))
    monkeypatch.setattr(exactness, "GRAY", gray)
    orig = exactness.first_order

    def former(sp, X, v2, cc, terr=None):
        E = np.atleast_1d(orig(sp, X, v2, cc, terr))
        return np.where(np.atleast_1d(np.asarray(v2, float)) <= _e2(sp, X, cc), math.inf, E)
    monkeypatch.setattr(exactness, "first_order", former)
    monkeypatch.setattr(exactness, "_perturbed", lambda sp: [])              # no re-solves
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b0 = [_problem(*case, rounding=k).bounds(0) for k in ROUNDINGS]
    assert any(b.status == "EXACT" and abs(b.value - ref) > TOL * max(1.0, abs(ref))
               for pair in b0 for b, ref in zip(pair, EXACT[case]))


def _e2(sp, X, cc):
    starts, nb, L, _ = exactness.blocks(sp)
    X = np.asarray(X, float).reshape(len(sp.delta), -1)
    nx2 = np.add.reduceat(X * X, starts, axis=0)
    w = exactness.C * exactness.EPS * nb
    cc = np.maximum(np.atleast_1d(np.asarray(cc, float)), 0.0)
    return (w * L) @ nx2 + 2.0 * np.sqrt(cc) * ((w * np.sqrt(L)) @ np.sqrt(nx2)) + exactness.C * exactness.EPS * cc

