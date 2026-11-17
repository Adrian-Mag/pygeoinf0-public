"""Plan 054: BGP's and oCSOLA's exactness rule counts the representers' own stated errors (optimisers/dual/exactness.py, "The
representers' own stated errors"; optimisers/dual/exactness_reduce.py), and so do posterior variances at scale
(probability/scale.py through probability/input_error.py). On the unit sphere the representers of point evaluations in
H^s are realised to the harmonic degree cap and state errors (about 1e-2 at s = 2, 1e-4 at s = 3; Family.errors());
before, BGP bounds built from them said EXACT while they were off by 1e-7 to 1e-4, beyond TOL. Reference: the exact
zonal kernel summed to degree 60000 (as in test_error_account_reperr.py)."""
import math
import warnings

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.precision import TOL
from pygeoinf0.optimisers.dual import exactness, exactness_reduce
from pygeoinf0.convex.feasible import FeasibleSet
from pygeoinf0.core.errors import PrecisionWarning

B, SIGMA, A = 0.09, 0.05, 3.0           # length_scale 0.3: (1 + b l(l+1))^-s with b = 0.3^2; prior radius A


def _zonal(x, s, L=60000):
    """sum_l (2l+1)/(4 pi) (1 + b l(l+1))^-s P_l(x), with the tail at x = 1 added (its asymptote l^(1-2s))."""
    x = np.asarray(x, float)
    p0, p1 = np.ones_like(x), x.copy()
    tot = p0 / (4 * np.pi) + 3.0 / (4 * np.pi) * (1 + 2 * B) ** -s * p1
    for deg in range(1, L):
        p0, p1 = p1, ((2 * deg + 1) * x * p1 - deg * p0) / (deg + 1)
        n = deg + 1
        tot = tot + (2 * n + 1) / (4 * np.pi) * (1 + B * n * (n + 1)) ** -s * p1
    tot[np.isclose(x, 1.0)] += 2 / (4 * np.pi) * B ** -s * L ** (2 - 2 * s) / (2 * s - 2)
    return tot


def _points(k, seed):
    v = np.random.default_rng(seed).standard_normal((k, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


@pytest.fixture(scope="module", params=[2, 3])
def case(request):
    s = request.param
    obs, test = _points(10, 5), _points(4, 6)
    H = pg.Sobolev(pg.Sphere(1.0), s, length_scale=0.3)
    G, T = pg.Functionals(H.riesz(obs)), pg.Functionals(H.riesz(test))
    D = G.codomain
    d = D(0.5 * obs[:, 2] + 0.2 + SIGMA * np.random.default_rng(2).standard_normal(len(obs)))
    def z(X, Y):
        return _zonal(np.clip(X @ Y.T, -1, 1), s)
    rho = 1.2 * SIGMA * math.sqrt(len(obs))
    exact = pg.decompose(None, D.standard_gaussian(), gram=z(obs, obs), space=D).ellipsoid_bound(
        a=A, r=rho, datum=d, pairings=(z(obs, test), z(test, test)))
    return dict(s=s, H=H, G=G, T=T, D=D, d=d, rho=rho, exact=exact, test=test)


def _ends(pairs):
    return [b for lo_hi in pairs for b in lo_hi]


def test_truncated_kernels_are_not_exact_and_the_bracket_holds(case):
    """Every end through the decomposition is BRACKET with E stated, and its widened bracket holds the exact value."""
    got = pg.decompose(case["G"], case["D"].standard_gaussian()).ellipsoid_bound(case["T"], A, case["rho"], case["d"])
    for b, e in zip(_ends(got), _ends(case["exact"])):
        assert e.status == "EXACT"
        assert b.status == "BRACKET" and b.error is not None
        assert b.lower <= e.value <= b.upper
        assert abs(b.value - e.value) <= b.error                     # E bounds the true error
        assert "stated realisation" in b.note


def test_planted_control_the_former_rule_says_exact(case, monkeypatch):
    """The former rule (rounding only): ends EXACT while off by more than TOL (the defect)."""
    monkeypatch.setattr(exactness, "stated", lambda dec: None)
    monkeypatch.setattr(exactness, "coefficients", lambda sp, terr: None)
    got = pg.decompose(case["G"], case["D"].standard_gaussian()).ellipsoid_bound(case["T"], A, case["rho"], case["d"])
    false = [abs(b.value - e.value) > TOL * max(1.0, abs(e.value))
             for b, e in zip(_ends(got), _ends(case["exact"])) if b.status == "EXACT"]
    assert false and all(false)


def test_many_directions_at_once_state_the_same(case):
    """The vectorised route (FeasibleImage.intervals through spectral_many) carries the targets' errors too."""
    K = FeasibleSet(pg.NormBall(case["H"], A), case["G"], pg.NormBall(case["D"], case["rho"]), case["d"])
    lo, hi, sl, su, _, br = K.image(case["T"]).intervals(brackets=True)
    ex = case["exact"]
    assert all(st == "BRACKET" for st in list(sl) + list(su))
    for k, (elo, ehi) in enumerate(ex):
        assert br["lower_bracket"][k, 0] <= elo.value <= br["lower_bracket"][k, 1]
        assert br["upper_bracket"][k, 0] <= ehi.value <= br["upper_bracket"][k, 1]


def test_the_reduce_route_states_the_columns_errors(case, monkeypatch):
    """FeasibleSet.bound without a decomposition (the joint Gram, optimisers/dual/exactness_reduce.py): BRACKET, the bracket
    holds the exact value; the planted control (no stated errors on the reduction) says EXACT, off beyond TOL."""
    t = case["H"].riesz(case["test"][:1])[0]
    ehi = case["exact"][0][1].value
    K = FeasibleSet(pg.NormBall(case["H"], A), case["G"], pg.NormBall(case["D"], case["rho"]), case["d"], decomposition=False)
    b = K.bound(t)
    assert b.status == "BRACKET" and b.lower <= ehi <= b.upper
    monkeypatch.setattr(exactness_reduce, "stated_part", lambda red: 0.0)
    b0 = FeasibleSet(pg.NormBall(case["H"], A), case["G"], pg.NormBall(case["D"], case["rho"]), case["d"],
                     decomposition=False).bound(t)
    assert b0.status == "EXACT" and abs(b0.value - ehi) > TOL * max(1.0, abs(ehi))


def test_exact_representers_change_nothing():
    """Kernels in closed form (no stated error): the rule is the rounding one, EXACT as before."""
    rng = np.random.default_rng(1)
    X = pg.Euclidean(6)
    Gm = rng.standard_normal((4, 6))
    G = pg.Matrix(Gm, X, pg.Euclidean(4))
    d = G.codomain(rng.standard_normal(4))
    K = FeasibleSet(pg.NormBall(X, 2.0), G, pg.NormBall(G.codomain, 1.0), d)
    b = K.bound(X(rng.standard_normal(6)))
    assert b.status == "EXACT" and exactness.stated(K._spectral().dec) is None


def test_ocsola_moves_its_ends_by_the_stated_errors(case):
    """oCSOLA with truncated kernels: a PrecisionWarning names the representers' stated errors, the ends move outwards."""
    H, G, D = case["H"], case["G"], case["D"]
    T = pg.Functionals(H.riesz(case["test"][:1]))
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        lo, hi = pg.inference.methods.ocsola(pg.NormBall(H, A), pg.Additive(G, D.standard_gaussian()), T, case["d"], gamma=1.0)
    msgs = [str(w.message) for w in W if issubclass(w.category, PrecisionWarning) and "oCSOLA" in str(w.message)]
    assert msgs and any("stated realisation" in m for m in msgs) and lo < hi


def test_variances_at_scale_state_the_representers_errors(case):
    """Above settings.dense_max (probability/scale.py) the posterior variances state their input error as below it."""
    H, G, test = case["H"], case["G"], case["test"]
    u = H.riesz(test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        small = pg.probability.cylindrical(H).condition(G, case["d"], G.codomain.standard_gaussian())
        v0, e0 = small.variance(u), small.variance_error
        with pg.settings.using(dense_max=4):
            big = pg.probability.cylindrical(H).condition(G, case["d"], G.codomain.standard_gaussian())
            v1, e1 = big.variance(u), big.variance_error
    assert getattr(big, "_scale", None) is not None and e0 is not None and e1 is not None
    assert np.all(np.asarray(e1) > 0) and np.allclose(v1, v0, rtol=1e-6, atol=1e-12)
