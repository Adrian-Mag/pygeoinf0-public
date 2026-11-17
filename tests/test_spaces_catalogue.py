"""pg.catalogue: the shipped snapshot is current, its recipes run, and the table, plot and names behave.

The staleness guard (test_snapshot_is_current) probes every combination again (about 15 s, dominated by the 3-D spectral
spaces) and compares it with src/pygeoinf0/spaces_snapshot.json. If it fails, the package changed what it can build;
read the differences it prints, and if they are intended run

    python -c "import pygeoinf0 as pg; pg.catalogue.write_snapshot()"

and commit the file."""
import json
import warnings

import pytest

import pygeoinf0 as pg
from pygeoinf0 import catalogue

SNAP = catalogue.snapshot()


def cell(domain, space):
    return SNAP["cells"][domain][space]


# ------------------------------------------------------------------------------------------------- staleness
def _guard(speed):
    fresh = catalogue.stable(pg.catalogue.probe(speed=speed))
    diff = catalogue.differences(fresh=fresh, shipped=SNAP)
    assert not diff, "pg.spaces snapshot is stale (pg.catalogue.write_snapshot() regenerates it):\n" + "\n".join(diff)
    return fresh


def test_snapshot_is_current_fast_part():
    """Every pair that takes well under a second (about 5 s together with the extras)."""
    fresh = _guard("fast")
    assert sum(len(r) for r in fresh["cells"].values()) == len(SNAP["domains"]) * len(SNAP["spaces"]) - len(
        [p for p in catalogue.SLOW if len(p) == 2])


def test_snapshot_is_current_slow_part():
    """The pairs that take seconds each (3-D spectral spaces, a surface functional): about 25 s. With the fast part this
    is the full guard; `pytest -k "not slow_part"` skips it."""
    _guard("slow")


def test_snapshot_has_no_errors_and_covers_the_registry():
    assert [d["key"] for d in SNAP["domains"]] == [d[0] for d in catalogue.DOMAINS]
    assert [s["key"] for s in SNAP["spaces"]] == [s[0] for s in catalogue.SPACES]
    for dk, row in SNAP["cells"].items():
        assert list(row) == [s[0] for s in catalogue.SPACES]
        for sk, c in row.items():
            assert c["status"] in ("ok", "constructed", "refused", "n/a"), (dk, sk, c)
            assert (c["reason"] is not None) == (c["status"] in ("refused", "n/a")), (dk, sk, c)
    assert all("seconds" not in json.dumps(v) for v in SNAP["cells"].values())


def test_snapshot_is_shipped_as_package_data():
    assert catalogue.SNAPSHOT.exists() and catalogue.SNAPSHOT.parent.name == "pygeoinf0"


# ------------------------------------------------------------------------------------------------- facts the table states
@pytest.mark.parametrize("domain, space, status", [
    ("Interval", "SpecRobin", "ok"), ("Interval", "C", "ok"), ("Rectangle", "C", "refused"),
    ("Rectangle", "SpecRobin", "refused"), ("Circle", "SpecN", "n/a"), ("Interval", "SpecMixed", "ok"),
    ("Segment", "SpecMixed", "n/a"), ("Box", "SpecMixed", "ok"), ("Shell", "SpecMixed", "refused"),
    ("Layered", "L2", "constructed"), ("Layered", "Sobolev", "ok"), ("Layered", "Lp", "refused"),
])
def test_known_cells(domain, space, status):
    assert cell(domain, space)["status"] == status


def test_closed_manifolds_use_the_spectral_sobolev_space():
    for dk in ("Circle", "Sphere", "Torus2", "Torus3"):
        assert cell(dk, "Sobolev")["definition"] in ("periodic", "spherical")
    for dk in ("Interval", "Rectangle", "Box", "Disk"):
        assert cell(dk, "Sobolev")["definition"] == "restriction"


def test_robin_refuses_fractional_orders_on_intervals():
    fr = {(e["domain"], e["space"], e["order"]): e for e in SNAP["extras"]["fractional"]}
    assert fr["Interval", "SpecRobin", 1]["status"] == fr["Interval", "SpecRobin", 2]["status"] == "ok"
    for s in (1.5, 2.5):
        assert fr["Interval", "SpecRobin", s]["status"] == "refused"
        assert "integer orders" in fr["Interval", "SpecRobin", s]["reason"]
    for sp in ("SpecN", "SpecD", "SpecMixed"):
        assert fr["Interval", sp, 1.5]["status"] == fr["Interval", sp, 2.5]["status"] == "ok"
    assert fr["Box", "SpecMixed", 2.5]["status"] == "ok"


def test_formula_norms_are_computed_in_1d_and_refused_above():
    fn = {e["domain"]: e for e in SNAP["extras"]["formula_norms"]}
    assert fn["Interval"]["status"] == "ok"
    for d in ("Rectangle", "Disk", "Box"):
        assert fn[d]["status"] == "refused" and fn[d]["error"] == "NormUnavailable"
    assert "formula" in pg.catalogue.how("Disk", "Sobolev", show=False).note


def test_solves_beyond_the_norm():
    assert SNAP["extras"]["solves"] and all(e["ok"] for e in SNAP["extras"]["solves"])


# ------------------------------------------------------------------------------------------------- how
def test_how_names_are_forgiving():
    a = pg.catalogue.how("Box", "Spec Mixed", show=False)
    for d, s in [("box", "spec mixed"), ("BOX", "SPECMIXED"), ("  box ", "spectral mixed"), ("Bo", "mixed")]:
        b = pg.catalogue.how(d, s, show=False)
        assert (b.domain, b.space) == (a.domain, a.space) == ("Box", "Spec Mixed")
    assert pg.catalogue.how("torus 3d", "neumann", show=False).domain == "Torus 3D"
    assert pg.catalogue.how("great circle", "h1", show=False).space == "Intr H1"
    with pytest.raises(KeyError, match="ambiguous"):
        pg.catalogue.how("torus", "L2", show=False)
    with pytest.raises(KeyError, match="did you mean"):
        pg.catalogue.how("Cylindr", "L3", show=False)


def test_how_prints_code_status_and_note(capsys):
    r = pg.catalogue.how("Box", "Spec Mixed")
    out = capsys.readouterr().out
    assert r.status == "ok" and out.strip() == r.text.strip()
    assert 'dirichlet=(\'x-\',)' in r.code and "pg.Box(" in r.code and "length_scale=0.2" in r.code
    assert "# note:" in out and "s > d/2" in out


def test_how_builds_the_domain_before_the_space():
    for d, s in [("Box", "Spec Mixed"), ("Rectangle", "h1"), ("Interval", "L2"), ("Curve", "Sobolev")]:
        code = pg.catalogue.how(d, s, show=False).code
        assert code.index("# 1. the domain") < code.index("D = ") < code.index("# 2. the space on D") < code.index("H = ")


def test_how_gives_a_form_its_own_line_and_the_same_space():
    warnings.simplefilter("ignore")
    r = pg.catalogue.how("Rectangle", "Spec Mixed", show=False)
    assert "form = pg.SobolevForm(D, 1" in r.code and "H = pg.Sobolev(form.operator(), 2)" in r.code
    assert "H = form.space()" in pg.catalogue.how("Rectangle", "h1", show=False).code
    assert isinstance(r.run()["H"], type(eval(cell("Rectangle", "SpecMixed")["space_code"],
                                              {"pg": pg, "D": pg.Box("[0, 1]", "[0, 1]")})))


def test_how_of_a_row_and_a_column():
    row = pg.catalogue.how("box", show=False)
    assert (row.domain, row.space, row.status) == ("Box", None, None)
    assert all(s["label"] in row.text for s in SNAP["spaces"]) and "D = pg.Box(" in row.text
    assert "D = " in row.code and "import pygeoinf0" in row.code and row.run()["D"].dim == 3
    col = pg.catalogue.how(space="neumann", show=False)
    assert (col.domain, col.space) == (None, "Spec N")
    assert all(d["label"] in col.text for d in SNAP["domains"])
    assert "pg.catalogue.how(" in pg.catalogue.how(show=False)
    with pytest.raises(KeyError, match="ambiguous"):
        pg.catalogue.how("torus", show=False)


def test_how_of_a_refusal_says_what_works_on_the_domain():
    text = pg.catalogue.how("Rectangle", "C", show=False).text
    assert "works on this domain: L2, Lp, Sobolev(D), Intr H1" in text and "C," not in text.split("works")[1]


def test_the_new_row_and_column():
    assert cell("Curve", "L2")["status"] == cell("Curve", "Sobolev")["status"] == "ok"
    assert all(cell(d["key"], "ComplexL2")["status"] == "ok" for d in SNAP["domains"] if d["key"] != "Layered")
    assert cell("Layered", "ComplexL2")["status"] == "refused"
    assert isinstance(pg.catalogue.how("box", "complex", show=False).run()["H"], pg.Complexified)
    # the caveat's claim: Complexified also applies to a Sobolev space
    H = pg.Complexified(pg.Sobolev(pg.Interval("[0, 1]"), 1, length_scale=0.2))
    assert H.norm(H({"re": lambda x: 1.0 + x, "im": lambda x: x})) > 0


def test_how_names_the_realisation_as_numerics():
    note = pg.catalogue.how("Box", "L2", show=False).note
    assert "Realisation here: 'local'" in note and "not another space" in note and "pg.realisations.Cells" in note
    assert set(cell("Box", "L2")) >= {"realisation"} and pg.L2(pg.Box("[0, 1]", "[0, 1]")).realisation == "local"


def test_how_code_runs_for_a_sample():
    warnings.simplefilter("ignore")
    for d, s in [("Interval", "L2"), ("Interval", "C"), ("Interval", "SpecRobin"), ("Rectangle", "SpecMixed"),
                 ("Disk", "Sobolev"), ("Circle", "IntrH2"), ("Sphere", "IntrH1"), ("Torus2", "Sobolev"),
                 ("Layered", "Sobolev"), ("Arc", "Sobolev")]:
        r = pg.catalogue.how(d, s, show=False)
        assert r.status == "ok", (d, s, r.status)
        ns = r.run()
        assert ns["H"] is not None


def test_how_returns_the_package_reason_for_a_refusal():
    r = pg.catalogue.how("Rectangle", "Spec Robin", show=False)
    assert r.status == "refused" and r.reason
    with pytest.raises(Exception) as e:                 # the code raises exactly that refusal
        r.run()
    assert str(e.value).split("\n")[0] == r.reason
    assert r.reason in r.text
    assert isinstance(e.value, NotImplementedError)


def test_how_not_applicable_and_constructed():
    r = pg.catalogue.how("Circle", "Spec Neumann", show=False)
    assert r.status == "n/a" and r.code == "" and "no boundary" in r.reason
    r = pg.catalogue.how("Layered", "L2", show=False)
    assert r.status == "constructed" and "Tried:" in r.note
    r.run()                                              # constructs


def test_unbounded_functionals_are_recorded_for_constructed_cells():
    c = cell("Torus3", "IntrH1")
    assert c["status"] in ("ok", "constructed")
    assert c["tried"][0] == {"functional": "point evaluation", "result": "unbounded"}


# ------------------------------------------------------------------------------------------------- table, plot, probe
def test_table_text_and_markdown():
    t = pg.catalogue.table()
    assert "Great-circle arc" in t and "Spec Mixed" in t and "Legend:" in t
    first, last = [r for r in t.splitlines() if "Interval" in r][0], [r for r in t.splitlines() if "Layered" in r][0]
    assert first.split()[0] == "1D" and last.split()[0] == "3D"
    m = pg.catalogue.table("markdown")
    assert m.startswith("| dim | domain |") and m.count("\n|") >= len(SNAP["domains"])
    with pytest.raises(ValueError):
        pg.catalogue.table("html")
    assert repr(pg.catalogue.spaces) == t


def test_plot_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = pg.catalogue.plot()
    assert len(ax.patches) == len(SNAP["domains"]) * len(SNAP["spaces"])
    assert [t.get_text() for t in ax.get_xticklabels()] == [s["label"] for s in SNAP["spaces"]]
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    fig, ax2 = plt.subplots()
    assert pg.catalogue.plot(ax=ax2) is ax2
    plt.close(fig)


def test_probe_subset_and_limits():
    r = pg.catalogue.probe(domains=["interval", "circle"], spaces=["L2", "neumann"])
    assert list(r["cells"]) == ["Interval", "Circle"] and "extras" not in r
    assert r["cells"]["Circle"]["SpecN"]["status"] == "n/a"
    assert catalogue.stable(r)["cells"]["Interval"]["L2"] == SNAP["cells"]["Interval"]["L2"]
    with pytest.raises(KeyError):
        pg.catalogue.probe(domains=["nowhere"])


def test_time_limit_turns_a_hang_into_an_error():
    import time
    with pytest.raises(catalogue.ProbeTimeout):
        with catalogue._limits(0.2, None):
            time.sleep(2)
    with catalogue._limits(None, None):                 # no limit asked: nothing happens
        pass
