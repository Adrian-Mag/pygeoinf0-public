"""Interactive slices (plan 046): the values on each slice against the formula, the sliders driving them, the refusals."""
import numpy as np
import pytest

import pygeoinf0 as pg

pytest.importorskip("plotly")
pytest.importorskip("ipywidgets")
pytest.importorskip("anywidget")

F = lambda x, y, z: np.exp(z) * x + y ** 2 + 0.5
BALL = pg.Ball(1.0)


def field():
    return pg.L2(BALL, realisation="radial_harmonics").element(F)


def test_the_plane_shows_the_field_and_blanks_what_is_outside():
    s = pg.plotting.slices(field(), backend="widget")
    t = s.figure.data[0]
    X, Y, Z, v = (np.asarray(a, float) for a in (t.x, t.y, t.z, t.surfacecolor))
    inside = X ** 2 + Y ** 2 + Z ** 2 < 0.99
    assert inside.any() and np.isfinite(v[inside]).all()
    assert v[inside] == pytest.approx(F(X[inside], Y[inside], Z[inside]), abs=1e-10)
    assert np.isnan(v[X ** 2 + Y ** 2 + Z ** 2 > 1.01]).all()


def test_the_sliders_move_the_plane():
    s = pg.plotting.slices(field(), backend="widget")
    s.view(polar=60.0, azimuth=120.0, offset=0.3)
    t = s.figure.data[0]
    X, Y, Z, v = (np.asarray(a, float) for a in (t.x, t.y, t.z, t.surfacecolor))
    p, a = np.radians(60.0), np.radians(120.0)
    n = np.array([np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)])
    assert n[0] * X + n[1] * Y + n[2] * Z == pytest.approx(np.full(X.shape, 0.3), abs=1e-12)
    keep = np.isfinite(v)
    assert keep.any() and v[keep] == pytest.approx(F(X[keep], Y[keep], Z[keep]), abs=1e-10)
    flat = np.asarray(s.figure.data[1].z, float)
    assert np.nan_to_num(flat) == pytest.approx(np.nan_to_num(v))


def test_the_sphere_moves_with_r_and_the_map_is_the_sphere():
    s = pg.plotting.slices(field(), "sphere", backend="widget")
    s.view(r=0.7)
    t = s.figure.data[0]
    X, Y, Z, v = (np.asarray(a, float) for a in (t.x, t.y, t.z, t.surfacecolor))
    assert np.sqrt(X ** 2 + Y ** 2 + Z ** 2) == pytest.approx(np.full(X.shape, 0.7), abs=1e-12)
    assert v == pytest.approx(F(X, Y, Z), abs=1e-10)
    h = s.figure.data[1]
    assert np.asarray(h.z, float) == pytest.approx(v) and len(h.x) == 2 * len(h.y) - 1
    assert "r = 0.7" in s.figure.layout.title.text


def test_a_shell_blanks_nothing_on_its_spheres_and_keeps_its_range():
    X = pg.L2(pg.Shell(0.4, 0.8), realisation="radial_harmonics")
    s = pg.plotting.slices(X.element(F), "sphere", backend="widget")
    assert s.sliders["r"].min == 0.4 and s.sliders["r"].max == 0.8
    s.view(r=0.6)
    assert np.isfinite(np.asarray(s.figure.data[0].surfacecolor, float)).all()
    p = pg.plotting.slices(X.element(F), backend="widget")
    v, r = np.asarray(p.figure.data[0].surfacecolor, float), np.sqrt(sum(np.asarray(a, float) ** 2 for a in (p.figure.data[0].x, p.figure.data[0].y, p.figure.data[0].z)))
    assert np.isnan(v[r < 0.39]).all() and np.isfinite(v[(r > 0.41) & (r < 0.79)]).all()


def test_the_colour_scale_is_fixed_and_symmetric_and_can_be_set():
    s = pg.plotting.slices(field(), backend="widget")
    before = (s.figure.data[0].cmin, s.figure.data[0].cmax)
    assert before[0] == -before[1] and before[1] > 0
    s.view(polar=90.0, offset=0.5)
    assert (s.figure.data[0].cmin, s.figure.data[0].cmax) == before
    t = pg.plotting.slices(field(), clim=(-2.0, 3.0), backend="widget")
    assert (t.figure.data[1].zmin, t.figure.data[1].zmax) == (-2.0, 3.0)


def test_boundaries_are_circles_on_the_plane():
    s = pg.plotting.slices(field(), boundaries=[0.5], backend="widget", clim="whole")
    s.view(polar=0.0, offset=0.3)
    c = s.figure.data[2]
    r = np.sqrt(np.asarray(c.x, float) ** 2 + np.asarray(c.y, float) ** 2)
    assert r == pytest.approx(np.full(r.shape, 0.4), abs=1e-12) and np.asarray(c.z, float) == pytest.approx(np.full(r.shape, 0.3))
    s.view(offset=0.6)                                          # the plane misses the sphere
    assert len(s.figure.data[2].x) == 0


def test_a_family_has_a_selector_with_its_own_scale():
    X = pg.L2(BALL, realisation="radial_harmonics")
    fam = X.family([X.element(F), X.element(lambda x, y, z: 10.0 * F(x, y, z))])
    s = pg.plotting.slices(fam, backend="widget")
    top = s.figure.data[0].cmax
    s.view(member=1)
    assert s.figure.data[0].cmax == pytest.approx(10.0 * top, rel=1e-9)
    assert "[1]" in s.figure.layout.title.text


def test_a_glued_field_on_pieces():
    P = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"]).partition
    M = pg.PiecewiseSpace(P, {"a": pg.L2(BALL, realisation="radial_harmonics")})
    s = pg.plotting.slices(M.glue(M.split({"a": F}), "a"), boundaries=[0.3, 0.55], backend="widget")
    t = s.figure.data[0]
    v = np.asarray(t.surfacecolor, float)
    keep = np.isfinite(v)
    assert v[keep] == pytest.approx(F(*(np.asarray(a, float)[keep] for a in (t.x, t.y, t.z))), abs=1e-9)


def test_a_layered_field_brings_its_boundaries():
    planet = pg.LayeredBall([0.5, 1.0], names=["in", "out"])
    V = pg.PiecewiseSpace.lebesgue(planet, realisation="radial_harmonics")
    x = V({k: F for k in planet.names})
    s = pg.plotting.slices(x, backend="widget")
    assert s.boundaries == [0.5, 1.0]
    t = s.figure.data[0]
    v = np.asarray(t.surfacecolor, float)
    keep = np.isfinite(v)
    assert keep.any() and v[keep] == pytest.approx(F(*(np.asarray(a, float)[keep] for a in (t.x, t.y, t.z))), abs=1e-9)


def test_a_box_has_the_plane_and_refuses_the_sphere():
    from pygeoinf0.core.errors import Unsupported
    X = pg.L2(pg.Box([(0, 1), (0, 2), (0, 1)]))
    s = pg.plotting.slices(X.element(F), backend="widget")
    v = np.asarray(s.figure.data[0].surfacecolor, float)
    assert np.isfinite(v).any() and np.isnan(v).any()
    with pytest.raises(Unsupported, match="balls and shells"):
        pg.plotting.slices(X.element(F), "sphere", backend="widget")
    with pytest.raises(Unsupported, match="3-D"):
        pg.plotting.slices(pg.L2(pg.Interval(0, 1)).element(np.sin), backend="widget")


def test_a_view_is_cached_and_a_page_can_be_saved(tmp_path, monkeypatch):
    s = pg.plotting.slices(field(), backend="widget")
    s.view(offset=0.2)
    s.view(offset=0.4)
    n = len(s._cache)
    s.view(offset=0.2)
    assert len(s._cache) == n
    path = tmp_path / "view.html"
    s.write_html(str(path))
    assert path.stat().st_size > 1000


def test_without_plotly_it_says_what_is_needed(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_plotly(name, *a, **k):
        if name.startswith("plotly"):
            raise ImportError("No module named plotly")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_plotly)
    with pytest.raises(ImportError, match="interactive"):
        pg.plotting.slices(field(), backend="widget")


def test_the_entry_point_stays_a_function_after_use():
    import pygeoinf0.plotting as P
    P.slices(field(), backend="widget")
    assert callable(P.slices) and callable(pg.plotting.slices)
