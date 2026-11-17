"""The capability protocol of realisations (plan 069 P3, review C3; realisations/registry.py): what a realisation can do
is asked by name (provides, require, capability), never read from its class; a missing capability refuses with its
name, the capabilities the realisation has and the realisations that have it; every declared capability is one of the
protocol's and is implemented; the routes of convex/, inference/ and probability/ ask the capability."""
import inspect

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import CapabilityMissing, Unsupported
from pygeoinf0.realisations import registry


def _sphere_element():
    M = pg.L2(pg.Sphere(1.0))
    return M, M(lambda x, y, z: 1.0 + z, resolution=0.5)


def test_a_missing_capability_refuses_naming_it_and_the_existing_ones():
    M, f = _sphere_element()
    R = f.realisation
    have = registry.capabilities(R)
    assert "box_support" in have and "cell_integrals" not in have
    with pytest.raises(CapabilityMissing) as e:
        registry.capability(R, "cell_integrals", "interval cells of a test")
    err = e.value
    assert isinstance(err, Unsupported)
    assert err.capability == "cell_integrals" and set(err.provided) == set(have)
    msg = str(err)
    assert "cell_integrals" in msg and "interval cells of a test" in msg
    assert all(name in msg for name in have)                           # what the realisation has
    assert "PiecewiseLegendre" in msg                                                  # who has the one asked for
    with pytest.raises(CapabilityMissing, match="cell_integrals"):
        registry.require(R, "cell_integrals")
    assert not registry.provides(R, "cell_integrals") and registry.provides(R, "box_support")


def test_an_unknown_capability_name_is_refused_with_the_protocol():
    _, f = _sphere_element()
    with pytest.raises(ValueError, match="box_support"):
        registry.provides(f.realisation, "l1_brackets_of_everything")
    assert registry.capabilities(None) == frozenset() and registry.capabilities(np.zeros(3)) == frozenset()


def _realisation_classes():
    import importlib
    import pkgutil
    import pygeoinf0.realisations as pkg
    for m in pkgutil.walk_packages(pkg.__path__, "pygeoinf0.realisations."):     # the subpackages too (plan 069 P4)
        mod = importlib.import_module(m.name)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__:
                yield cls


def test_every_declared_capability_is_in_the_protocol_and_implemented():
    seen = set()
    for cls in _realisation_classes():
        for name in getattr(cls, "CAPABILITIES", ()):
            assert name in registry.CAPABILITIES, f"{cls.__name__} declares {name!r}, which is not in the protocol"
            if registry.CAPABILITIES[name].method:
                assert callable(getattr(cls, name, None)), f"{cls.__name__} declares {name!r} without the method"
            seen.add(name)
    backends = {n for n, c in registry.CAPABILITIES.items() if c.backend}
    assert set(registry.CAPABILITIES) == seen | backends, "capabilities in the protocol that nothing provides"


def test_backend_operations_are_capabilities():
    """The optional operations of a backend (Backend.supports) are capabilities: legendre_decay_generators on an
    interval's 'legendre', not on a sphere's harmonics."""
    I, (_, S) = pg.L2(pg.Interval(0, 1)), _sphere_element()
    assert registry.provides(I.backend, "legendre_decay_generators")
    assert not registry.provides(S.space.backend, "legendre_decay_generators")
    assert registry.provides(S.space.backend, "fit_family") and registry.provides(I.backend, "fit_family")
    with pytest.raises(CapabilityMissing, match="legendre_decay_generators"):
        registry.capability(S.space.backend, "legendre_decay_generators")


def test_a_box_support_without_the_capability_refuses_naming_it():
    M = pg.L2(pg.Box((0, 1), (0, 1)))
    q = M(lambda x, y: x - y, resolution=0.5)
    with pytest.raises(CapabilityMissing, match="box_support"):
        pg.PointwiseBox(M, lower=-1.0, upper=1.0).support(q)


def test_an_ocsola_rule_on_a_ball_refuses_naming_box_rule():
    from pygeoinf0.inference.ocsola_box import Rule
    from pygeoinf0.realisations.series.radial_harmonic import RadialHarmonicSeries
    from pygeoinf0.spaces.elements import Function
    M = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    C = np.zeros(4)
    C[0] = 1.0
    f = Function(M, pl=RadialHarmonicSeries(1, [0.0, 1.0], [np.einsum("d,j->dj", np.array([1.0]), C)[:, :, None]], 0.0))
    with pytest.raises(CapabilityMissing, match="box_rule"):
        Rule(pg.PointwiseBox(M, envelope=1.0), [f])


def test_the_sphere_cell_route_asks_for_harmonic_coefficients():
    """Interval cells and sphere cells ask their capabilities; a sphere element's realisation has the harmonic
    coefficients, an interval element's does not (so it is refused by name, not by class)."""
    I = pg.L2(pg.Interval(0, 1))
    q = I(lambda x: x, resolution=0.5)
    assert registry.provides(q.realisation, "cell_integrals")
    assert not registry.provides(q.realisation, "harmonic_coefficients")
    _, f = _sphere_element()
    L = f.realisation.L
    assert np.array_equal(registry.capability(f.realisation, "harmonic_coefficients")(L),
                          np.asarray(f.realisation.C, float))
