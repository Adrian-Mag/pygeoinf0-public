"""The Sobolev snippet of README.md runs as written, and the prior scale it states is the kernel's value at 0."""
import math
import pathlib
import re

import numpy as np
import pytest

import pygeoinf0 as pg

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


@pytest.mark.skipif(not README.exists(), reason="README.md not staged")
def test_sobolev_snippet_runs():
    text = README.read_text()
    block = re.search(r"```python\n(import numpy as np, pygeoinf0 as pg\nD = pg.Disk.*?)```", text, re.S).group(1)
    block = block.replace("m.plot()", "")                       # plotting is tested elsewhere (evaluation-bound: 10 s)
    ns = {}
    exec(compile(block, "README.md", "exec"), ns)
    m, post = ns["m"], ns["post"]
    assert post.mean(0.2, 0.3) == pytest.approx(m(0.2, 0.3), rel=1e-10)
    assert (ns["G"](m) - ns["d"]).norm() ** 2 == pytest.approx(ns["n"], rel=1e-6)


@pytest.mark.parametrize("D, s, l", [(pg.Disk(1.0), 1.5, 0.1), (pg.Ball(1.0), 2.0, 0.3), (pg.Interval("[0, 1]"), 1.0, 0.2)])
def test_prior_point_variance_is_k0(D, s, l):
    H = pg.Sobolev(D, s, length_scale=l)
    d = D.cartesian_dim if hasattr(D, "cartesian_dim") else 1
    x0 = 0.5 if d == 1 else np.zeros(d) + 0.1
    k0 = math.gamma(s - d / 2) / (math.gamma(s) * (4 * math.pi) ** (d / 2) * l ** d)
    assert pg.probability.cylindrical(H, alpha=2.0).variance(x0) == pytest.approx(k0 / 2, rel=1e-12)


@pytest.mark.skipif(not README.exists(), reason="README.md not staged")
def test_which_sobolev_space_snippet_runs():
    text = README.read_text()
    block = re.search(r"```python\n(import numpy as np, pygeoinf0 as pg\nI, l = .*?)```", text, re.S).group(1)
    ns = {}
    exec(compile(block, "README.md", "exec"), ns)
    sp = ns["spaces"]
    assert pg.probability.cylindrical(sp["Dirichlet"]).std(0.0) == 0.0 == pg.probability.cylindrical(sp["spectral Dirichlet s=1.5"]).std(0.0)
    assert pg.probability.cylindrical(sp["intrinsic"]).std(0.0) == pytest.approx(math.sqrt(2 / math.tanh(1 / 0.1) / 0.2), rel=1e-12)
    m0 = {}                                               # the minimum-norm models differ at the boundary
    for name in ("restriction", "intrinsic", "spectral Neumann s=2", "intrinsic order 2"):
        H = sp[name]
        G = pg.Functionals(H.riesz([pg.DiracMeasure(0.05), pg.Interval(0.4, 0.6)]))
        m0[name] = G.adjoint((G @ G.adjoint).inv(G.codomain([1.0, 0.1])))(0.0)
    assert len({round(v, 3) for v in m0.values()}) == 4
    assert pg.probability.cylindrical(sp["Robin"]).std(1.0) < pg.probability.cylindrical(sp["intrinsic"]).std(1.0)       # Robin penalises u(1)
    assert "reflection images" in repr(ns["Hb"].claims)


@pytest.mark.skipif(not README.exists(), reason="README.md not staged")
def test_layered_snippet_runs():
    text = README.read_text()
    block = re.search(r"```python\n(import numpy as np, pygeoinf0 as pg\nP = pg.LayeredBall.*?)```", text, re.S).group(1)
    ns = {}
    exec(compile(block, "README.md", "exec"), ns)
    G, d, m = ns["G"], ns["d"], ns["m"]
    assert (G(m) - d).norm() ** 2 == pytest.approx(len(ns["data"]), rel=1e-6)          # discrepancy: chi^2 = n
    assert ns["values"][1] == pytest.approx(m["v"]["crust"](np.array([0.0, 0.0, 0.97])), rel=1e-12)
    prior = pg.probability.cylindrical(ns["X"], alpha=ns["alpha"]).std({"v": np.array([[0, 0, 0.985], [0, 0, 0.5]])})
    assert np.all(ns["std"] <= prior)


@pytest.mark.skipif(not README.exists(), reason="README.md not staged")
def test_relations_snippet_runs():
    text = README.read_text()
    block = re.search(r"```python\n(import numpy as np, pygeoinf0 as pg\nM = pg.Sobolev.*?)```", text, re.S).group(1)
    ns = {}
    exec(compile(block, "README.md", "exec"), ns)
    G, d, M = ns["G"], ns["d"], ns["M"]
    assert ns["K0"].summary().distance(G.adjoint((G @ G.adjoint).inv(d))) == 0.0
    assert ns["lo"].value < ns["hi"].value and ns["hi"].status == "EXACT"
    ref = pg.probability.cylindrical(M, alpha=1.0).condition(G, d)
    assert ns["post"].mean(0.3) == ref.mean(0.3)
    assert ns["R"](ns["m_free"]).member(d) is pg.Truth.TRUE
    assert "| B10 |" in ns["table"]


@pytest.mark.skipif(not README.exists(), reason="README.md not staged")
def test_bring_your_own_implementation_snippet_runs():
    text = README.read_text()
    block = re.search(r"```python\n(import numpy as np, pygeoinf0 as pg\nI, x = .*?)```", text, re.S).group(1)
    ns = {}
    exec(compile(block, "README.md", "exec"), ns)
    assert ns["fit"].converged and min(pg.port.taylor_test(ns["G"], ns["m0"]).rates) > 1.9
