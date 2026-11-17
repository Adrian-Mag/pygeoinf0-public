"""pg.port.fields (plan 056 P1): an element's pieces, fields and members for plotting and export plugins. Exact answers are the
formulas the elements were built from; a seam must give each piece its own limit."""
import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import NotDefined, Unsupported

R1 = 0.55
IN = lambda x, y, z: 1 + 0.2 * z          # noqa: E731
OUT = lambda x, y, z: -1 + 0.2 * z        # noqa: E731


def _layered():
    P = pg.LayeredBall([R1, 1.0], names=["core", "mantle"])
    V = pg.PiecewiseSpace.lebesgue(P)
    return P, V, V({"core": IN, "mantle": OUT}, name="m1")


def _sphere_points(r, n=50, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return r * d / np.linalg.norm(d, axis=1, keepdims=True)


def test_layered_pieces_keep_the_jump_on_the_seam():
    P, V, x = _layered()
    F = pg.port.fields(x)
    assert [p.name for p in F.pieces] == ["core", "mantle"] and F.fields == ("value",) and F.members == ["m1"]
    assert F.piece("core").chart.reference[0] == (0.0, R1) and F.piece("mantle").chart.reference[0] == (R1, 1.0)
    S = _sphere_points(R1)
    np.testing.assert_allclose(F.values("core", "value", 0, S), IN(*S.T), atol=1e-12)
    np.testing.assert_allclose(F.values("mantle", "value", 0, S), OUT(*S.T), atol=1e-12)
    # the same through the space's own one-sided evaluation
    np.testing.assert_allclose(F.values("mantle", "value", 0, S), V.evaluate(x, S, side="above"), atol=1e-12)
    # control: the other side's value differs by the jump, so mixing sides would fail the checks above
    assert np.all(np.abs(V.evaluate(x, S, side="below") - F.values("mantle", "value", 0, S)) > 1.9)


def test_members_of_a_list_and_a_family():
    P, V, x = _layered()
    F = pg.port.fields([x, 2 * x])
    assert len(F.members) == 2
    S = _sphere_points(0.3)
    np.testing.assert_allclose(F.values("core", "value", 1, S), 2 * IN(*S.T), atol=1e-12)
    G = pg.port.fields(V.family([x, 2 * x], name="F"))
    assert G.members == ["F[0]", "F[1]"]
    np.testing.assert_allclose(G.values("mantle", "value", "F[1]", _sphere_points(0.8)), 2 * OUT(*_sphere_points(0.8).T),
                               atol=1e-12)


def test_on_pieces_fields_per_leaf_and_absent_ones():
    part = pg.Partition(pg.Ball(1.0), [pg.Ball(R1), pg.Shell(R1, 1.0)], names=["in", "out"])
    M = pg.PiecewiseSpace(part, {"vp": pg.L2, "vs": pg.L2}, where={"vs": ["out"]})
    m = M.split({"vp": lambda x, y, z: 1.0 + x, "vs": lambda x, y, z: 2.0 + y})
    F = pg.port.fields(m)
    assert F.fields == ("vp", "vs") and F.piece("in").fields == ("vp",) and F.piece("out").fields == ("vp", "vs")
    S = _sphere_points(0.8)
    np.testing.assert_allclose(F.values("out", "vs", 0, S), 2.0 + S[:, 1], atol=1e-12)
    with pytest.raises(KeyError):
        F.values("in", "vs", 0, _sphere_points(0.3))


def test_direct_sum_lists_non_function_summands_as_absent():
    B = pg.L2(pg.Box((0, 1), (0, 2)))
    D = pg.DirectSum({"a": B, "e": pg.Euclidean(2)})
    F = pg.port.fields(D({"a": B.element(lambda x, y: x * y)}))
    assert [p.name for p in F.pieces] == ["a"] and F.absent[0][0] == "e"
    np.testing.assert_allclose(F.values("a", "value", 0, [[0.5, 1.5]]), [0.75], atol=1e-12)


def test_accuracy_states_what_it_knows():
    B = pg.L2(pg.Box((0, 1), (0, 2)))
    u = B.element(lambda x, y: np.sin(x) * y, name="u")
    a = pg.port.fields(u).accuracy("whole", "value", 0)
    assert a["kind"] in ("formula", "realisation")
    if a["kind"] == "realisation":
        assert a["error"] == pytest.approx(u.error) and "bounds no single value" in a["text"]


def test_a_chart_of_every_gridded_domain_maps_into_it():
    for D in (pg.Ball(1.0), pg.Shell(0.5, 1.0), pg.Box((0, 1), (0, 2), (0, 3)), pg.Sphere(2.0)):
        c = pg.port.Chart(D)
        t = [np.linspace(a, b, 5) for a, b in c.reference]
        X = c.to_cartesian(*np.meshgrid(*t, indexing="ij")).reshape(-1, c.cartesian_dim)
        assert np.all(D.contains(X) if c.dim == c.cartesian_dim else np.isclose(np.linalg.norm(X, axis=1), 2.0))


def test_refusals_name_the_reason():
    from pygeoinf0.domains.pieces import BallPiece
    with pytest.raises(Unsupported, match="no chart"):
        pg.port.Chart(BallPiece(((0.5, 0, 0), 0.3), ((0, 0, 0), 0.0, 0.6)))
    with pytest.raises(NotDefined, match="not a space of functions"):           # no grid exists (plan 068 P1)
        pg.port.fields(pg.Euclidean(3).zero)


def test_a_glued_sobolev_field_is_one_element_seen_on_every_leaf():
    part = pg.Partition(pg.Ball(1.0), [pg.Ball(R1), pg.Shell(R1, 1.0)], names=["in", "out"])
    M = pg.PiecewiseSpace(part, {"s": lambda D: pg.Sobolev(D, 2, length_scale=0.3), "vp": pg.L2})
    F = pg.port.fields(M.split({"s": lambda x, y, z: 1 + x * y, "vp": lambda x, y, z: z}))
    assert F.piece("in").fields == ("s", "vp") and F.piece("out").fields == ("s", "vp")
    S = _sphere_points(R1)
    np.testing.assert_allclose(F.values("in", "s", 0, S), F.values("out", "s", 0, S), atol=1e-12)   # glued: no jump
    np.testing.assert_allclose(F.values("out", "s", 0, S), 1 + S[:, 0] * S[:, 1], atol=1e-10)
