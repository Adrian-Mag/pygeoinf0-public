"""Problems generated from profiles (plan 020 P5, EXP-035 GENERATOR.md): the restricted resolver, profiles with core and
plugin counterparts and experiment values, provenance claims and coverage, the run, and the refusals."""
import sys

import numpy as np
import pytest

import pygeoinf0 as pg


def tikhonov_profile():
    """A linear problem with metrics on both sides, generated: spaces, the map, the data and the method."""
    return {
        "profile": "tikhonov-test", "graph": {"id": "test-graph"},
        "objects": {
            "model": {"ref": "core:Euclidean", "args": [3], "kwargs": {"weights": [1.0, 2.0, 0.5]}, "element": "m"},
            "data_space": {"ref": "core:Euclidean", "args": [4], "kwargs": {"covariance": [0.1, 0.2, 0.3, 0.4]}},
            "G": {"ref": "core:port.External", "args": [{"$id": "model"}, {"$id": "data_space"}, {"$value": "forward"}],
                  "kwargs": {"jacobian": {"$value": "jacobian"}, "name": "A"}, "element": "R_obs"},
            "d": {"ref": "core:Euclidean.element", "args": [{"$id": "data_space"}, {"$value": "observed"}]},
        },
        "missing": ["noise_model"],
        "run": {"ref": "core:optimisers.gauss_newton", "args": [{"$id": "G"}, {"$id": "d"}, {"$value": "start"}],
                "kwargs": {"alpha": 0.5, "tol": 1e-12}},
    }


def test_generated_problem_equals_the_hand_built_one():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 3))
    d = rng.standard_normal(4)
    values = {"forward": lambda x: A @ x, "jacobian": lambda x: A, "observed": d}
    M0 = pg.Euclidean(3, weights=[1.0, 2.0, 0.5])
    values["start"] = M0.zero
    g = pg.port.generate(tikhonov_profile(), values)
    w, c, alpha = np.array([1.0, 2.0, 0.5]), np.array([0.1, 0.2, 0.3, 0.4]), 0.5
    exact = np.linalg.solve(A.T @ (A / c[:, None]) + alpha * np.diag(w), A.T @ (d / c))
    assert np.allclose(g.result.theta.values, exact, atol=1e-9)
    assert g["G"].claims.status("generated") == (pg.Truth.TRUE, "asserted")
    assert "graph element R_obs" in g["G"].claims.find("generated").statement
    assert [b[1] for b in g.built] == ["m", "data_space", "R_obs", "d"]
    assert g.coverage.status("coverage") == (pg.Truth.FALSE, "asserted")                 # noise_model has no counterpart


def test_complex_data_and_a_taylor_check_on_the_generated_map():
    w_freq = np.logspace(-1, 1, 5)

    def debye(x):
        a, tau = np.exp(x)
        return a / (1 + 1j * w_freq * tau)

    def jac(x):
        d = debye(x)
        tau = np.exp(x[1])
        return np.c_[d, -d * (1j * w_freq * tau) / (1 + 1j * w_freq * tau)]

    profile = {"profile": "debye", "objects": {
        "M": {"ref": "core:Euclidean", "args": [2]},
        "D": {"ref": "core:Euclidean", "args": [5], "kwargs": {"covariance": 0.01, "field": "C"}},
        "G": {"ref": "core:port.External", "args": [{"$id": "M"}, {"$id": "D"}, {"$value": "debye"}],
              "kwargs": {"jacobian": {"$value": "jac"}}}}}
    g = pg.port.generate(profile, {"debye": debye, "jac": jac})
    res = pg.port.taylor_test(g["G"], g["M"](np.array([0.1, -0.3])), directions=3)
    assert all(r > 1.9 for r in res.rates) and res.adjoint_gap < 1e-12


def test_plugin_counterparts_pin_their_version(monkeypatch):
    monkeypatch.setattr(pg.port.plugins, "_REGISTRY", {})
    sys.modules.pop("plugin_demo", None)
    import plugin_demo
    pg.port.plugins.register(plugin_demo.PLUGIN)
    square = pg.port.resolve("plugin:demo/square@0.1")
    assert square is plugin_demo.SQUARE
    with pytest.raises(pg.Refused, match="installed is version 0.1, the profile pins 0.2"):
        pg.port.resolve("plugin:demo/square@0.2")
    with pytest.raises(pg.Refused, match="declares no 'cube'"):
        pg.port.resolve("plugin:demo/cube@0.1")
    with pytest.raises(pg.Refused, match="plugin:name/object@version"):
        pg.port.resolve("plugin:demo")


def test_the_resolver_reaches_only_public_objects():
    assert pg.port.resolve("core:port.Coordinates.nodal") == pg.port.Coordinates.nodal
    for bad, why in (("core:port.plugins.importlib", "not in the public names"), ("core:np", "not a public"),
                     ("core:Map.__init__", "private"), ("os:system", "unknown counterpart kind"),
                     ("core:port", "names a module"), ("Euclidean", "a counterpart reference is")):
        with pytest.raises(pg.Refused, match=why):
            pg.port.resolve(bad)


def test_generation_failures_name_the_element():
    p = tikhonov_profile()
    with pytest.raises(pg.GenerationError, match="supplies no value 'forward'"):
        pg.port.generate(p, {})
    p["objects"]["G"]["args"][0] = {"$id": "nowhere"}
    with pytest.raises(pg.GenerationError, match="refers to 'nowhere'"):
        pg.port.generate(p, {"forward": None})
    bad = {"profile": "bad", "objects": {"D": {"ref": "core:Euclidean", "args": [-1], "element": "data"}}}
    with pytest.raises(pg.GenerationError, match="bad/D: building graph element 'data' with core:Euclidean failed"):
        pg.port.generate(bad)


def test_mineralnetwork_element_shapes():
    """MineralNetwork's counterpart schema (ASC-041 P1): elements as ids, {id, instance, proposed} or lists; missing
    entries as {element, reason}; extra fields (kind, note, claims, values) ignored by the generator."""
    from pygeoinf0.port.generate import element_label
    assert element_label({"id": "dv", "instance": "P_mantle"}) == "dv[P_mantle]"
    assert element_label([{"id": "R3v"}, {"id": "station_terms", "proposed": True}]) == "R3v + station_terms (proposed)"
    assert element_label({"element": {"id": "robust_loss", "proposed": True}, "reason": "Huber"}) == \
        "robust_loss (proposed) (Huber)"
    profile = {"profile": "mn-shapes", "claims": [{"key": "x"}], "values": [{"name": "n"}],
               "objects": {"D": {"kind": "space", "ref": "core:Euclidean", "args": [{"$value": "n"}],
                                 "element": {"id": "observed_times"}, "note": "ignored"}},
               "missing": [{"element": {"id": "noise_model", "proposed": True}, "reason": "per-wave classes"}]}
    g = pg.port.generate(profile, {"n": 3})
    assert g["D"].dim == 3
    assert "noise_model (proposed) (per-wave classes)" in g.coverage[0].reason


def test_coverage_report_is_attached_as_checked_claims():
    profile = {"profile": "cov", "objects": {"D": {"ref": "core:Euclidean", "args": [2], "element": {"id": "observed_times"}}}}
    report = {"profile": "cov", "graph": {"id": "ray-tomography-graph"}, "export": {"sha256": "ab" * 32, "stale": False},
              "entries": [{"object": "D", "elements": [{"id": "observed_times", "status": "exists"}]}],
              "summary": {"exists": 1, "proposed": 1, "unknown id": 0, "errors": 0, "proposed_ids": ["noise_model"]}}
    g = pg.port.generate(profile, coverage=report)
    c = g.graph_claims.find("graph coverage")
    assert c.holds and c.status == "checked" and "1 exist, 1 proposed, 0 unknown ids; proposed: noise_model" in c.statement
    with pytest.raises(pg.GenerationError, match="of profile 'other'"):
        pg.port.generate(profile, coverage={**report, "profile": "other"})
    stale = pg.port.generate(profile, coverage={**report, "export": {"sha256": "x", "stale": True}})
    assert not stale.graph_claims.find("graph coverage").holds
