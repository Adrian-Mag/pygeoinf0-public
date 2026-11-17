"""Problem.run and the calibration element (plan 070 P5, review D9; thesis el:535-567): every result carries the
calibration claims of its problem, each method declares the calibrations it supports and the options it takes, a
calibration or an option it does not declare is refused naming the declared ones, and the numbers are those of the
method called directly (only claims are added). Also the map from the hyperparameter families of pg.meta to the
methods' options (plan 070 P2, open decision 6)."""
import inspect

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.inference import methods as imethods


def problem(calibration=None, **fields):
    rng = np.random.default_rng(4)
    m, n = 6, 4
    M, D = pg.Euclidean(m), pg.Euclidean(n)
    A = rng.normal(size=(n, m)) + 2.0 * np.eye(n, m)
    G = pg.Matrix(A, M, D, name="G")
    T = pg.Matrix(rng.normal(size=(1, m)) + 0.5, M, pg.Euclidean(1), name="T")
    d = D(A @ (3.0 * rng.normal(size=m) + 1.0) + rng.normal(size=n))
    K = pg.Additive(G, D.standard_gaussian())
    base = dict(M=M, D=D, mdr=K, mpr=T, datum=d, prior=pg.GaussianMeasure(M, pg.Identity(M)))
    return pg.Problem(**{**base, "calibration": calibration, **fields}), G, K


def _numbers(x):
    """The numbers of a result: arrays of its values (bit comparison)."""
    if isinstance(x, tuple):
        return [np.asarray(v, float) for v in x]
    for k in ("estimate", "values"):
        if hasattr(x, k):
            v = getattr(x, k)
            return [np.asarray(getattr(v, "values", v), float)]
    if hasattr(x, "covariance") and hasattr(x, "mean"):
        return [np.asarray(x.mean.values), np.asarray(x.covariance.matrix())]
    if hasattr(x, "summary"):
        return [np.asarray(x.summary().values)]
    if hasattr(x, "bounds"):
        return [np.asarray(x.interval(0))]
    return [np.asarray(x.support(x.space([1.0]))) if hasattr(x, "support") else np.asarray(x)]


def _cases(pb, G, K):
    ball = pg.NormBall(pb.M, 100.0)
    return {"minimum_norm": (dict(mdr=G), {}),
            "least_squares": (dict(mdr=G), dict(gamma=0.5)),
            "sola": ({}, dict(gamma=0.5)),
            "csola": (dict(prior=ball, mdr=K.acceptance(0.05)), dict(gamma=0.5)),
            "ocsola": (dict(prior=ball), dict(gamma=0.5, alpha=0.05)),
            "bgp": (dict(prior=ball, mdr=K.acceptance(0.05)), {}),
            "pli": ({}, {})}


def test_the_declared_calibrations_of_each_method():
    """The declaration (methods.py, the conservative map recorded in plan 070 P5): a guarantee is supported where the
    method's result is the object it is about (a set or an interval of the data for the deterministic, frequentist and
    hardened statements; a posterior measure for the Bayesian one); a point or an ordering supports none."""
    from pygeoinf0.inference import Bayesian, Deterministic, Frequentist, Hardened, Unstated
    want = {"minimum_norm": (Unstated,), "least_squares": (Unstated,), "sola": (Unstated,),
            "csola": (Unstated, Deterministic), "ocsola": (Unstated, Frequentist, Hardened),
            "bgp": (Unstated, Deterministic, Frequentist, Hardened), "pli": (Unstated, Bayesian)}
    assert {n: m.calibrations for n, m in imethods.METHODS.items()} == want


def test_results_carry_the_calibration_claims_and_the_same_numbers():
    pb0, G, K = problem()
    for name, (fields, options) in _cases(pb0, G, K).items():
        pb = pb0.with_(**fields)
        out = pb.run(name, **options)
        assert out.claims.find("calibration") is not None, name
        st = out.claims.find("calibration")
        assert st.statement == "calibration: unstated" and "no calibration element" in st.reason, (name, st)
        f = {k: v for k, v in pb._f.items() if v is not None and k not in ("M", "D", "P", "calibration")}
        direct = imethods.METHODS[name].fn(**f, **options)
        for a, b in zip(_numbers(out), _numbers(direct)):
            assert np.array_equal(a, b), (name, a, b)                     # bit-identical: only claims added
    pb = pb0.with_(prior=pg.NormBall(pb0.M, 100.0), mpr=pb0.mpr)
    ends = pb.run("ocsola", gamma=0.5)
    assert isinstance(ends, tuple) and ends == pg.inference.ocsola(pb.prior, K, pb.mpr, pb.datum, gamma=0.5)


def test_a_stated_calibration_is_carried_with_its_hypotheses():
    pb, G, K = problem(pg.Bayesian())
    post = pb.run("pli")
    assert "a prior measure on the model space" in repr(post.claims)
    assert post.claims.find("calibration").statement == "calibration: Bayesian"
    h = pb.with_(calibration=pg.inference.Hardened(0.05, 0.1), prior=pg.NormBall(pb.M, 100.0),
                 mdr=K.acceptance(0.05)).run("bgp")
    assert h.claims.find("coverage").status == "theorem" and h.claims.find("prior mass").status == "unknown"


def test_an_unsupported_calibration_is_refused_naming_the_supported():
    pb, G, K = problem(pg.inference.Deterministic())
    with pytest.raises(pg.Refused, match="pli supports the calibrations Unstated, Bayesian") as e:
        pb.run("pli")
    assert "Deterministic" in str(e.value)
    with pytest.raises(pg.Refused, match="Unstated"):
        pb.with_(calibration=pg.Bayesian(), mdr=G).run("least_squares", gamma=0.5)


def test_each_method_declares_the_options_it_takes():
    for name, m in imethods.METHODS.items():
        sig = inspect.signature(m.fn).parameters
        taken = {k for k, p in sig.items() if p.default is not inspect.Parameter.empty and k != "cache"}
        assert set(m.options) == taken, (name, m.options, taken)


def test_hyperparameter_families_map_to_the_options_of_their_method():
    """The map (plan 070 P2 open decision 6, settled in P5): damping of least_squares -> gamma, centre; damping of
    sola -> gamma (also csola's and ocsola's gamma, SOLA's X); the scales (a, nu), the damping of pli and a noise family
    -> no option of pli: refused naming how the choice is stated (the scaled prior and noise, or the fitted noise law
    passed explicitly), not silently dropped."""
    pb, G, K = problem()
    ch = pg.meta.discrepancy(pb)                       # the kernel's noise (a point G into a plain D: NoiseUnstated)
    assert pb.with_(mdr=G).run("least_squares", **ch.options()).summary() is not None
    k = pg.meta.knee(pb, "sola", pg.meta.damping("sola", lo=1e-3, hi=1e3), rule=pg.meta.Knee(samples=51))
    s = pb.run("sola", **k.options())
    assert np.array_equal(s.estimate.values, pg.inference.sola(K, pb.mpr, pb.datum, gamma=float(k.gamma)).estimate.values)
    ev = pg.meta.maximum_evidence(pb, "pli", pg.meta.scales())
    with pytest.raises(pg.Refused, match=r"prior N\(m0, a\^2 S\)") as e:
        pb.run("pli", **ev.options())
    assert "hyperparameter family 'scales'" in str(e.value)
    kp = pg.meta.knee(pb, "pli", pg.meta.damping("pli", lo=1e-3, hi=1e3), rule=pg.meta.Knee(samples=51))
    with pytest.raises(pg.Refused, match="pli takes no option gamma"):
        pb.run("pli", **kp.options())
    fit = pg.meta.fit_noise(pb, "pli", pg.meta.NoiseFamily(pb.D))
    with pytest.raises(pg.Refused, match=r"fit.measure\(\)"):
        pb.run("pli", **fit.options())
    with pytest.raises(pg.Refused, match="its options: gamma"):
        pb.run("sola", gama=0.5)                                          # a mistyped option is not dropped


def test_minimum_norm_on_a_function_space_carries_the_claims():
    H = pg.Sobolev("[0, 1]", 1, length_scale=0.1)
    D = pg.Euclidean(5)
    G = pg.Functionals(H.riesz([pg.Interval(a, a + 0.1) for a in np.linspace(0.05, 0.8, 5)]), into=D)
    d = D(np.linspace(0.1, 0.5, 5))
    u = pg.Problem(M=H, D=D, mdr=G, datum=d).run("minimum_norm")
    v = pg.inference.minimum_norm(G, d)
    assert isinstance(u, type(v)) and u.distance(v) == 0.0
    assert u.claims.find("calibration") is not None
    assert {c.statement for c in v.claims} <= {c.statement for c in u.claims}
