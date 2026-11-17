"""Slices in a window and short static maps (plan 048), on Agg: values against formulas, sliders moved by mouse events through the canvas.
That a real window opens is not something this file can show: plan 048's record says how it was checked."""
import warnings

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402

import pygeoinf0 as pg  # noqa: E402
import pygeoinf0.functions as FN  # noqa: E402
from pygeoinf0.plotting.slice_views import SliceField  # noqa: E402
from pygeoinf0.plotting.slice_views import OPEN, prepare_backend  # noqa: E402

F = lambda x, y, z: np.exp(z) * x + y ** 2 - 0.3
BALL = pg.Ball(1.0)
X = pg.L2(BALL, realisation="radial_harmonics")


@pytest.fixture(autouse=True)
def tidy():
    yield
    plt.close("all")
    OPEN.clear()


def window(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pg.plotting.slices(*a, **k)


def click(w, key, frac):
    s = w.sliders[key]
    bb = s.ax.get_window_extent()
    x, y = bb.x0 + frac * bb.width, 0.5 * (bb.y0 + bb.y1)
    for name in ("button_press_event", "button_release_event"):
        w.figure.canvas.callbacks.process(name, MouseEvent(name, w.figure.canvas, x, y, button=1))


def image(w):
    return np.ma.filled(np.ma.masked_invalid(w.image.get_array()).astype(float), np.nan)


def test_the_window_is_the_default_and_warns_where_it_cannot_draw():
    with pytest.warns(UserWarning, match="cannot draw a window"):
        w = pg.plotting.slices(X.element(F))
    assert type(w).__name__ == "SliceWindow" and not prepare_backend()


def test_the_sphere_map_is_the_field_and_a_mouse_click_moves_r():
    w = window(X.element(F), "sphere")
    click(w, "r", 0.7)
    r = w.sliders["r"].val
    assert r == pytest.approx(0.7, abs=0.006)
    lat, lon = np.radians(np.linspace(-90, 90, w.n)), np.radians(np.linspace(-180, 180, 2 * w.n - 1))
    LO, LA = np.meshgrid(lon, lat)
    want = F(r * np.cos(LA) * np.cos(LO), r * np.cos(LA) * np.sin(LO), r * np.sin(LA))
    assert image(w) == pytest.approx(want, abs=1e-10)
    assert f"r = {r:.5g}" in w.ax.get_title()


def test_the_plane_is_the_field_and_blanks_outside():
    w = window(X.element(F), "plane", normal=(1.0, 0.0, 0.0))
    from pygeoinf0.plotting.slice_views import basis
    w.view(polar=60.0, azimuth=120.0, offset=0.3)
    p, a = np.radians(60.0), np.radians(120.0)
    n, e1, e2 = basis((np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)))
    g = np.linspace(-1.0, 1.0, w.n)
    U, V = np.meshgrid(g, g)
    P = [0.3 * n[k] + U * e1[k] + V * e2[k] for k in range(3)]
    v, inside = image(w), P[0] ** 2 + P[1] ** 2 + P[2] ** 2 < 0.99
    assert v[inside] == pytest.approx(F(*(q[inside] for q in P)), abs=1e-10)
    assert np.isnan(v[P[0] ** 2 + P[1] ** 2 + P[2] ** 2 > 1.01]).all()


def test_a_ball_fills_the_view_and_the_offset_stays_inside_it():
    w = window(X.element(F))
    assert w.extent == 1.0 and (w.sliders["offset"].valmin, w.sliders["offset"].valmax) == (-1.0, 1.0)
    assert w.ax.get_xlim() == (-1.0, 1.0)
    b = window(pg.L2(pg.Box([(0, 1), (0, 2), (0, 1)])).element(F))
    assert b.extent == pytest.approx(np.sqrt(1.5))


def test_boundaries_are_circles_and_vanish_where_the_plane_misses():
    w = window(X.element(F), boundaries=[0.5])
    w.view(offset=0.3)
    x, y = w.rings[0].get_data()
    assert np.hypot(x, y) == pytest.approx(np.full(len(x), 0.4), abs=1e-12)
    w.view(offset=0.6)
    assert len(w.rings[0].get_data()[0]) == 0


def test_a_layered_field_brings_its_boundaries_and_its_values():
    planet = pg.LayeredBall([0.5, 1.0], names=["in", "out"])
    V = pg.PiecewiseSpace.lebesgue(planet, realisation="radial_harmonics")
    w = window(V({k: F for k in planet.names}), "sphere")
    assert w.boundaries == [0.5, 1.0]
    assert np.isfinite(image(w)).all()


@pytest.mark.parametrize("clim, want", [("whole", 1.0), ("robust", None)])
def test_one_strong_layer_does_not_pale_the_rest(clim, want):
    """Most of the ball at 1, a thin skin at 100: robust keeps the scale of the bulk, whole takes the skin."""
    f = X.element(lambda x, y, z: 1.0 + 99.0 * (x * x + y * y + z * z > 0.998 ** 2), breaks=((0.998,), None, None))
    F2 = SliceField(f, clim)
    low, high = F2.limits()
    assert low == -high
    if clim == "whole":
        assert high > 50.0
    else:
        assert high < 5.0


def test_clim_view_moves_and_numbers_are_kept():
    f = X.element(lambda x, y, z: z)
    w = window(f, clim="view", normal=(0.0, 0.0, 1.0))
    w.view(offset=0.2)
    assert w.image.get_clim()[1] == pytest.approx(0.2, rel=1e-9)
    w.view(offset=0.5)
    assert w.image.get_clim()[1] == pytest.approx(0.5, rel=1e-9)
    t = window(f, clim=(-2.0, 3.0))
    t.view(offset=0.5)
    assert t.image.get_clim() == (-2.0, 3.0)
    with pytest.raises(ValueError, match="clim"):
        window(f, clim="nonsense")


def test_a_list_of_bank_functions_stays_formulas_and_a_narrow_one_gets_a_scale():
    bumps = [FN.on(BALL).bump(centre=(0.0, 0.0, 0.5), radius=rho).normalised() for rho in (0.2, 0.02)]
    els = [b.element(X) for b in bumps]
    assert all(e._pl is None for e in els)
    w = window(els, normal=(1.0, 0.0, 0.0))
    assert all(e._pl is None for e in els)                       # nothing was fitted
    first = w.image.get_clim()[1]
    assert first > 0
    w.view(member=1)
    assert w.image.get_clim()[1] > 100 * first                  # the narrow bump is much taller, and the sample found it
    assert "[1]" in w.ax.get_title() or els[1].name in w.ax.get_title()


def test_a_family_too():
    fam = X.family([X.element(F), X.element(lambda x, y, z: 10.0 * F(x, y, z))])
    w = window(fam, "sphere")
    top = w.image.get_clim()[1]
    w.view(member=1)
    assert w.image.get_clim()[1] == pytest.approx(10.0 * top, rel=1e-6)


def test_it_stays_alive_unassigned_until_closed_and_saves(tmp_path):
    window(X.element(F))
    w = list(OPEN)[-1]
    assert w in OPEN
    path = tmp_path / "view.png"
    assert w.save(str(path)) == str(path) and path.stat().st_size > 1000
    from matplotlib.backend_bases import CloseEvent             # Agg sends no close event; a real window's was seen (plan 048)
    w.figure.canvas.callbacks.process("close_event", CloseEvent("close_event", w.figure.canvas))
    assert w not in OPEN


def test_a_glued_field_on_pieces():
    P = pg.LayeredBall([0.3, 0.55, 1.0], names=["core", "mid", "top"]).partition
    M = pg.PiecewiseSpace(P, {"a": X})
    w = window(M.glue(M.split({"a": X.element(F)}), "a"), "sphere", boundaries=[0.3, 0.55])
    assert np.isfinite(image(w)).all()


def test_refusals():
    from pygeoinf0.core.errors import Unsupported
    Y = pg.L2(pg.Box([(0, 1), (0, 2), (0, 1)]))
    with pytest.raises(Unsupported, match="balls and shells"):
        window(Y.element(F), "sphere")
    with pytest.raises(Unsupported, match="3-D"):
        window(pg.L2(pg.Interval(0, 1)).element(np.sin))
    with pytest.raises(Unsupported, match="sphere_map"):
        window(pg.L2(pg.Sphere(1.0)).element(lambda x, y, z: x))
    with pytest.raises(ValueError, match="backend"):
        window(X.element(F), backend="nonsense")


def test_sphere_map_is_the_field_in_a_given_axis():
    _, ax = plt.subplots()
    assert pg.plotting.sphere_map(X.element(F), 0.7, ax=ax, colorbar=False) is ax
    v = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)
    lat, lon = np.radians(np.linspace(-90, 90, v.shape[0])), np.radians(np.linspace(-180, 180, v.shape[1]))
    LO, LA = np.meshgrid(lon, lat)
    assert v == pytest.approx(F(0.7 * np.cos(LA) * np.cos(LO), 0.7 * np.cos(LA) * np.sin(LO), 0.7 * np.sin(LA)), abs=1e-10)
    assert "r = 0.7" in ax.get_title()


def test_sphere_map_on_a_surface_needs_no_radius():
    S = pg.L2(pg.Sphere(1.0))
    ax = pg.plotting.sphere_map(S.element(lambda x, y, z: z))
    v = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)
    assert v[-1].mean() == pytest.approx(1.0, abs=1e-8) and v[0].mean() == pytest.approx(-1.0, abs=1e-8)
    with pytest.raises(ValueError, match="give r"):
        pg.plotting.sphere_map(X.element(F))
    with pytest.raises(ValueError, match="outside"):
        pg.plotting.sphere_map(X.element(F), 2.0)


def test_section_with_boundaries_and_a_grid_for_several():
    ax = pg.plotting.section(X.element(F), boundaries=[0.5])
    assert len(ax.lines) == 1 and ax.images[0].get_extent()[1] == 1.0
    els = [X.element(F), X.element(lambda x, y, z: z), X.element(lambda x, y, z: x)]
    axs = pg.plotting.section(els, ncols=2)
    assert axs.shape == (2, 2) and not axs[1, 1].get_visible() and len(axs[1, 0].images) == 1
    assert pg.plotting.sphere_map(els, 0.5).shape == (1, 3)
    assert callable(pg.plotting.slices)
