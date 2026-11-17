"""The meta layer (plan 009 G11; EXP-030 DESIGN.md G11), outside any application: the closed-form Pareto curve against
repeated dense solves; the knee rule on curves with analytic knees and the thesis's grid rule; the predictive
statistics' laws by Monte Carlo from the prior predictive; the coverage formula against Monte Carlo; the variance split
and leakage against dense references; lanes; one use on the ladder; the README section runs."""
import math
import pathlib
import re

import numpy as np
import pytest
from scipy.special import ndtri

import pygeoinf0 as pg

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def problem(seed=0, n=8, m=14, p=2):
    """M = R^m, G = A, T = B, dense noise covariance Sig, a prior covariance C1 and its mean m0, a datum."""
    rng = np.random.default_rng(seed)
    M, D = pg.Euclidean(m), pg.Euclidean(n)
    A, B = rng.normal(size=(n, m)), rng.normal(size=(p, m))
    Q = rng.normal(size=(n, n))
    Sig = Q @ Q.T / n + np.diag(rng.uniform(0.5, 2.0, n))
    R = rng.normal(size=(m, m))
    C1 = R @ R.T / m + 0.1 * np.eye(m)
    noise = pg.GaussianMeasure(D, pg.Matrix(Sig, D, D, name="C"))
    m0 = M(0.3 * rng.normal(size=m))
    prior = pg.GaussianMeasure(M, pg.Matrix(C1, M, M, name="C1"), mean=m0)
    G, T = pg.Matrix(A, M, D, name="G"), pg.Matrix(B, M, pg.Euclidean(p), name="T")
    return dict(M=M, D=D, A=A, B=B, Sig=Sig, C1=C1, noise=noise, prior=prior, m0=m0, G=G, T=T,
                d=D(rng.normal(size=n)))


def dense_sola(P, g):
    A, B, Sig = P["A"], P["B"], P["Sig"]
    X = B @ A.T @ np.linalg.inv(A @ A.T + g * Sig)
    R = B - X @ A
    return X, np.sqrt(np.diag(X @ Sig @ X.T)), np.sqrt(np.diag(R @ R.T) / np.diag(B @ B.T))


def keys(rec):
    return {c.key: c for c in rec.claims if c.key}


def test_pareto_curve_against_repeated_dense_solves():
    P = problem(1)
    dec = pg.decompose(P["G"], P["noise"])
    tc = pg.meta.TradeOff(dec, P["T"])
    gam = np.logspace(-3, 3, 25)
    r = tc.at(gam)
    rng = np.random.default_rng(2)
    for i, g in enumerate(gam):
        X, sig, trm = dense_sola(P, g)
        assert np.allclose(r.sigma[i], sig, rtol=1e-11) and np.allclose(r.trm[i], trm, rtol=1e-10, atol=1e-13)
        J = lambda Y: np.diag((P["B"] - Y @ P["A"]) @ (P["B"] - Y @ P["A"]).T) + g * np.diag(Y @ P["Sig"] @ Y.T)  # noqa
        for _ in range(5):                       # X_gamma minimises mismatch + gamma variance: a Pareto point
            assert np.all(J(X + 1e-3 * rng.normal(size=X.shape)) >= J(X))
    assert np.all(np.diff(r.sigma, axis=0) < 0) and np.all(np.diff(r.trm, axis=0) > 0)
    k = keys(tc)
    assert k["meta"].holds and k["used the datum"].holds is False and "el:59" in k["meta"].reason
    assert "decomposition" in repr(r.claims)


def analytic(L0=-2.0, L1=4.0, y=lambda t: (1 - t) ** 3):
    """A curve with trm = t and log10 sigma = y(t), t linear in log10 gamma over [10^L0, 10^L1]."""
    def curve(g):
        t = (np.log10(g) - L0) / (L1 - L0)
        return dict(trm=t, sigma=10.0 ** y(t))
    return curve, 10.0 ** L0, 10.0 ** L1, (lambda t: 10.0 ** (L0 + t * (L1 - L0)))


def test_knee_on_curves_with_analytic_knees():
    # rescaled: x = t, y = (1 - t)^3 below the chord x + y = 1; distance (1 - t - (1 - t)^3) / sqrt 2, max at
    # 1 - t = 1 / sqrt 3
    curve, lo, hi, gam = analytic()
    t_star = 1 - 1 / math.sqrt(3)
    k = pg.meta.Knee().find(curve, lo, hi)
    assert k.gamma == pytest.approx(gam(t_star), rel=1e-7)
    assert k.distance == pytest.approx((1 - t_star - (1 - t_star) ** 3) / math.sqrt(2), rel=1e-12)
    assert keys(k)["used the datum"].holds is False and "knee rule" in keys(k)
    flipped = pg.meta.Knee(side="concave").find(curve, lo, hi)    # nothing on that side: r3 takes the largest |d|
    assert flipped.gamma == pytest.approx(k.gamma, rel=1e-7) and "fallback" in repr(flipped.claims)
    # both sides populated (no rescaling): y = 1 - t + 0.3 sin(2 pi t), distance -0.3 sin(2 pi t) / sqrt 2
    curve, lo, hi, gam = analytic(y=lambda t: 1 - t + 0.3 * np.sin(2 * np.pi * t))
    assert pg.meta.Knee(rescale=None).find(curve, lo, hi).gamma == pytest.approx(gam(0.75), rel=1e-7)
    assert pg.meta.Knee(rescale=None, side="concave").find(curve, lo, hi).gamma == pytest.approx(gam(0.25), rel=1e-7)
    with pytest.raises(ValueError):
        pg.meta.Knee(axes=("trm", "estimate"))


def r3(trm, sig):
    """The thesis's rule r3 as EXP-030 p4a_sola.py writes it (the reference for the grid realisation)."""
    x, y = np.asarray(trm, float), np.log10(sig)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    ch = p1 - p0
    s = (ch[0] * (y - p0[1]) - ch[1] * (x - p0[0])) / math.hypot(*ch)
    q = 0.5 * (p0 + p1) - 1.0
    side = np.sign(ch[0] * (q[1] - p0[1]) - ch[1] * (q[0] - p0[0]))
    conv = np.where(np.sign(s) == side, np.abs(s), -np.inf)
    return int(np.argmax(conv)) if np.isfinite(conv).any() and conv.max() > 0 else int(np.argmax(np.abs(s)))


def test_grid_rule_is_the_thesis_procedure_and_the_continuous_knee_maximises():
    P = problem(3, p=1)
    dec = pg.decompose(P["G"], P["noise"])
    tc = pg.meta.TradeOff(dec, P["T"], datum=P["d"])
    lo, hi = 1e-4, 1e6
    v = lambda g: tc.values(g)                                                         # noqa: E731
    coarse = np.geomspace(lo, hi, 53)
    c = v(coarse)
    kc = coarse[r3(c["trm"], c["sigma"])]
    grid = np.sort(np.r_[coarse, [x for x in np.geomspace(kc / 10, kc * 10, 30) if x not in set(coarse)]])
    f = v(grid)
    knee = grid[r3(f["trm"], f["sigma"])]
    op = pg.meta.OperatingPoints(tc, lo, hi, knee=pg.meta.Knee(refine=(30, 10.0)))
    snap = lambda x: float(grid[np.argmin(np.abs(np.log10(grid) - np.log10(x)))])      # noqa: E731
    assert op.gamma == dict(resolution=snap(knee / 10), knee=knee, variance=snap(knee * 10))
    assert keys(op)["used the datum"].holds and "data-dependent rule" not in keys(op)
    # the continuous knee: no gamma of a dense independent scan (dense solves) is further from the chord
    ends = [dense_sola(P, g) for g in (lo, hi)]
    x0, x1 = ends[0][2][0], ends[1][2][0]
    y0, y1 = np.log10(ends[1][1][0]), np.log10(ends[0][1][0])

    def dist(g):
        _, s, t = dense_sola(P, g)
        return (1 - (t[0] - x0) / (x1 - x0) - (np.log10(s[0]) - y0) / (y1 - y0)) / math.sqrt(2)
    k = tc.knee(lo, hi)
    scan = max(dist(g) for g in np.geomspace(lo, hi, 300))
    assert k.distance >= scan - 1e-12 and k.distance == pytest.approx(dist(k.gamma), rel=1e-9)
    oc = tc.operating_points(lo, hi)
    assert oc.gamma["resolution"] == pytest.approx(k.gamma / 10, rel=1e-14)
    assert oc.sigma["resolution"] > oc.sigma["knee"] > oc.sigma["variance"]
    X = dense_sola(P, oc.gamma["knee"])[0]
    assert oc.estimate["knee"] == pytest.approx((X @ P["d"].values)[0], rel=1e-10)


def test_predictive_statistics_follow_their_laws_by_monte_carlo():
    P = problem(4, n=10)
    a, nu = 0.7, 1.3
    dec = pg.decompose(P["G"], P["noise"], middle=P["prior"])
    rng = np.random.default_rng(5)
    N = 3000
    Lc, Ls = np.linalg.cholesky(P["C1"]), np.linalg.cholesky(P["Sig"])
    m = P["m0"].values[:, None] + a * Lc @ rng.normal(size=(14, N))
    d = P["A"] @ m + nu * Ls @ rng.normal(size=(10, N))
    stats = np.array([[c.chi2, c.residual_chi2] + [x for b in c.bins for x in (b.chi2, b.residual_chi2)] for c in
                      (pg.meta.PredictiveCheck(dec, P["D"](d[:, j]), a, nu, bins=2, prior_mean=P["m0"]) for j in range(N))])
    pc = pg.meta.PredictiveCheck(dec, P["D"](d[:, 0]), a, nu, bins=2, prior_mean=P["m0"])
    laws = [pc.law, pc.residual_law] + [law for b in pc.bins for law in (b.law, b.residual_law)]
    for j, law in enumerate(laws):
        for p in (0.1, 0.5, 0.9):
            q = float(law.quantile(p))
            assert abs(np.mean(stats[:, j] <= q) - p) <= 4.5 * math.sqrt(p * (1 - p) / N), (j, p)
    assert sum(b.dof for b in pc.bins) == 10 and sum(b.chi2 for b in pc.bins) == pytest.approx(pc.chi2)
    assert pc.bins[0].snr[1] == pc.bins[1].snr[0] and 0 <= float(pc.bins[1].residual_p_value) <= 1
    from scipy.stats import chi2
    assert float(pc.p_value) == pytest.approx(chi2.sf(pc.chi2, 10), rel=1e-10)
    assert pc.residual_p_value.lower <= float(pc.residual_p_value) <= pc.residual_p_value.upper
    assert keys(pc)["used the datum"].holds and "fixed before the data" in repr(pc.claims)


def test_coverage_formula_against_monte_carlo():
    P = problem(6)
    A, B, Sig, C1 = P["A"], P["B"], P["Sig"], P["C1"]
    a, nu = 0.8, 1.5
    rng = np.random.default_rng(7)
    truth = P["M"](3.0 * rng.normal(size=14))                    # far from the prior mean: a real bias
    dec = pg.decompose(P["G"], P["noise"], middle=P["prior"])
    cv = pg.meta.Coverage(dec, P["T"], truth, a=a, nu=nu, level=0.9, prior_mean=P["m0"])
    S = a * a * A @ C1 @ A.T + nu * nu * Sig                     # the dense posterior-mean map and moments
    K = a * a * B @ C1 @ A.T @ np.linalg.inv(S)
    post = np.sqrt(np.diag(a * a * B @ C1 @ B.T - K @ (a * a * A @ C1 @ B.T)))
    alt = np.sqrt(np.diag(nu * nu * K @ Sig @ K.T))
    m0, mt = P["m0"].values, truth.values
    eps = B @ m0 + K @ (A @ mt - A @ m0) - B @ mt
    assert np.allclose(cv.sigma_post, post, rtol=1e-10) and np.allclose(cv.sigma_alt, alt, rtol=1e-10)
    assert np.allclose(cv.bias, eps, rtol=1e-9, atol=1e-12) and cv.z == pytest.approx(ndtri(0.95))
    N = 40000
    e = nu * np.linalg.cholesky(Sig) @ rng.normal(size=(8, N))
    est = (B @ m0)[:, None] + K @ (A @ (mt - m0))[:, None] + K @ e
    freq, se = cv.frequency(est.T)
    assert np.all(np.abs(freq - cv.probability) <= 4.5 * se) and np.all(np.abs(cv.bias) > 0.1 * cv.sigma_alt)
    g = 0.3                                                      # SOLA's intervals: sigma_post = sigma_alt
    sv = pg.meta.Coverage(pg.decompose(P["G"], P["noise"]), P["T"], truth, gamma=g, z=2.0)
    X, sig, _ = dense_sola(P, g)
    est = (X @ (A @ mt))[:, None] + X @ (e / nu)
    freq, se = sv.frequency(est.T)
    assert np.allclose(sv.sigma_post, sig, rtol=1e-11) and np.all(np.abs(freq - sv.probability) <= 4.5 * se)
    assert keys(sv)["used the datum"].holds is False
    # a generating law other than the MDR's (the thesis's fitted alternative, appQ): sigma_alt^2 = K C_gen K^T
    Cg = np.diag(rng.uniform(0.5, 3.0, 8)) @ Sig @ np.diag(rng.uniform(0.5, 3.0, 8))
    Cg = 0.5 * (Cg + Cg.T) + 0.1 * np.eye(8)
    cg = pg.meta.Coverage(dec, P["T"], truth, a=a, nu=nu, level=0.9, prior_mean=P["m0"], generating=Cg)
    assert np.allclose(cg.sigma_alt, np.sqrt(np.diag(K @ Cg @ K.T)), rtol=1e-9)
    assert np.allclose(cg.sigma_post, post, rtol=1e-10) and np.allclose(cg.bias, eps, rtol=1e-9, atol=1e-12)
    e2 = np.linalg.cholesky(Cg) @ rng.normal(size=(8, N))
    freq, se = cg.frequency(((B @ m0)[:, None] + K @ (A @ (mt - m0))[:, None] + K @ e2).T)
    assert np.all(np.abs(freq - cg.probability) <= 4.5 * se)


def test_variance_split_against_the_dense_posterior():
    P = problem(8)
    A, B, Sig, C1 = P["A"], P["B"], P["Sig"], P["C1"]
    a, nu = 1.7, 0.6
    dec = pg.decompose(P["G"], P["noise"], middle=P["prior"])
    vs = pg.meta.VarianceSplit(dec, P["T"], a, nu)
    S = a * a * A @ C1 @ A.T + nu * nu * Sig
    K = a * a * B @ C1 @ A.T @ np.linalg.inv(S)
    var = np.diag(a * a * B @ C1 @ B.T - K @ (a * a * A @ C1 @ B.T))
    assert np.allclose(vs.total, var, rtol=1e-10) and np.allclose(vs.noise, np.diag(nu * nu * K @ Sig @ K.T), rtol=1e-10)
    assert np.allclose(vs.total, dec.posterior(P["T"], P["d"], a, nu).std ** 2, rtol=1e-10)
    assert np.allclose(vs.noise_share + vs.mismatch_share, 1.0)


def test_leakage_by_summand_against_dense_kernels():
    rng = np.random.default_rng(3)
    X = pg.DirectSum({"a": pg.Euclidean(5), "b": pg.Euclidean(4)}, weights=[1.0, 2.5])
    A = rng.normal(size=(6, 9))
    G = pg.Functionals(X.riesz([{"a": r[:5], "b": r[5:]} for r in A]))
    T = pg.Functionals(X.riesz([{"a": rng.normal(size=5), "b": np.zeros(4)}]))     # a target on summand "a"
    dec = pg.decompose(G, G.codomain.standard_gaussian())
    lk = pg.meta.Leakage(dec, T, 0.05)
    assert lk.target == ["a"]
    w = np.r_[np.ones(5), np.full(4, 2.5)]
    Ag = A / w
    K = Ag @ np.diag(w) @ Ag.T
    c = dec.targets(T).cross[:, 0]                               # <g_i, t>
    x = np.linalg.solve(K + 0.05 * np.eye(6), c)                 # the weights of the delivered kernel
    rcm2 = x @ Ag[:, 5:] @ np.diag(w[5:]) @ Ag[:, 5:].T @ x
    tc = pg.meta.TradeOff(dec, T).at(0.05)
    tt = float(np.ravel(tc.target)[0])
    assert lk.rcm[0] == pytest.approx(math.sqrt(rcm2 / tt), rel=1e-10)
    assert lk.rrm[0] ** 2 + lk.rcm[0] ** 2 == pytest.approx(tc.trm[0] ** 2, rel=1e-11)
    assert sum(v[0] for v in lk.share.values()) == pytest.approx(1.0)
    m = X({"a": rng.normal(size=5), "b": rng.normal(size=4)})
    parts = lk.apply(m)
    full = dec.sola_operator(T, 0.05)(G(m)).values
    assert np.allclose(parts["a"] + parts["b"], full, rtol=1e-12) and abs(parts["b"][0]) > 0


def test_lane_records_the_element_and_the_datum():
    lane = pg.meta.Lane("noise", "noise scale", {"x1": [1.0, 2.0], "x2": [1.5, 2.0]}, baseline="x1")
    assert np.allclose(lane.differences["x2"], [0.5, 0.0]) and np.allclose(lane.relative["x2"], [0.5, 0.0])
    assert "model-data relation" in lane.changes and keys(lane)["used the datum"].holds is False
    run = pg.meta.Lane.run("realisation", "points", lambda n: 1.0 / n, {"10": 10, "20": 20}, used_datum=True,
                           chosen_from_datum=True)
    assert run.baseline == "10" and "data-dependent rule" in keys(run) and "not an element" in run.changes
    with pytest.raises(ValueError):
        pg.meta.Lane("likelihood", "x", {"a": 1.0})


def test_record_repr_shows_string_values():
    # Record.__repr__ formatted every shown value with :g; a lane shows its element, name and baseline (strings)
    lane = pg.meta.Lane("prior", "amplitude a", {"a=1": [1.0, 2.0], "a=2": [1.5, 2.5]}, baseline="a=1")
    s = repr(lane)
    assert s.startswith("Lane(") and "element='prior'" in s and "baseline='a=1'" in s and "no datum" in s


def test_tradeoff_and_predictive_check_on_the_ladder():
    import ladder
    P = ladder.build("L1", 40)
    G, alpha, post = ladder.solve(P)
    dec = pg.decompose(G)                                        # noise: the data space's own; S: the model metric
    T = pg.Functionals(P.H.riesz([pg.LebesgueMeasure(pg.Interval(0.4, 0.6))]))
    tc = pg.meta.TradeOff(dec, T, datum=P.d)
    op = tc.operating_points(1e-6, 1e4)
    assert 1e-6 < op.gamma["knee"] < 1e4 and op.trm["resolution"] < op.trm["knee"] < op.trm["variance"]
    one = pg.inference.sola(pg.Additive(G, P.D.standard_gaussian()), T, P.d, gamma=op.gamma["knee"])
    assert one.estimate.values[0] == pytest.approx(op.estimate["knee"], rel=1e-10)
    # PLI with the ladder's prior (covariance 1/alpha in the model metric): the discrepancy residual chi^2 is n
    pc = pg.meta.PredictiveCheck(dec, P.d, a=1 / math.sqrt(float(alpha)), nu=1.0, bins=3,
                                 hyperparameters_from_datum=True)
    assert pc.residual_chi2 == pytest.approx(P.n, rel=1e-8) and pc.residual_law.mean == pytest.approx(P.n - pc.trace_hat)
    assert 0 <= float(pc.residual_p_value) <= 1 and "data-dependent rule" in keys(pc)
    vs = pg.meta.VarianceSplit(dec, T, a=1 / math.sqrt(float(alpha)), nu=1.0)
    assert vs.total[0] == pytest.approx(float(post.std([pg.LebesgueMeasure(pg.Interval(0.4, 0.6))])[0]) ** 2, rel=1e-8)


@pytest.mark.skipif(not README.exists(), reason="README.md not staged")
def test_meta_snippet_runs():
    block = re.search(r"```python\n(import numpy as np, pygeoinf0 as pg\nrng = .*?)```", README.read_text(), re.S).group(1)
    ns = {}
    exec(compile(block, "README.md", "exec"), ns)
    assert ns["op"].gamma["resolution"] < ns["op"].gamma["knee"] < ns["op"].gamma["variance"]
    assert 0 < float(ns["pc"].p_value) < 1 and ns["vs"].noise_share[0] < 1
