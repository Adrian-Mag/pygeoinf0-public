"""Plan 068 P7 (review E1 and the Kernel part of E2): "restrict" is kept for u|_D, mu|_D and R|_S, and the other
meanings have their own names (the structure law is tests/test_structure.py::test_restrict_means_a_restriction):

    R.select(rows)          the relation of some of the data (a kernel's marginal on those rows), was R.restrict(rows)
    CoordinateProjection    R_J: D -> D_J, y -> (y_i)_(i in J) (relations/selection.py), was relations.Restriction
    RestrictionSpace        R^s_l(D) = H^s_l(R^d)|_D with the quotient norm, was spaces.constructions.Restriction
    C.compressed(P)         P C P for an orthogonal projection P, was C.restricted(P)
    D.subpiece(lo, hi)      the part of a domain whose natural coordinates lie in [lo, hi], was D.restricted(lo, hi)
    X.restrictions(P)       u -> (u|_(Omega_k))_k over a partition P, was P.restrict(X)
    MarkovKernel            x -> law(x), was Kernel (and AdditiveMarkovKernel, was AdditiveKernel)
"""
import math

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0 import relations
from pygeoinf0.spaces import constructions


def test_select_is_the_relation_of_some_data():
    rng = np.random.default_rng(0)
    M, D = pg.Euclidean(5), pg.Euclidean(4, covariance=[1.0, 2.0, 3.0, 4.0])
    G = pg.Matrix(rng.normal(size=(4, 5)), M, D, name="G")
    L = pg.Additive(G, D.standard_gaussian())
    LJ = L.select([0, 2])
    assert isinstance(LJ, relations.AdditiveMarkovKernel) and LJ.codomain.dim == 2 and np.allclose(LJ.codomain.weights, [1, 1 / 3])
    assert LJ.claims.find("selection").status == "derived"
    assert not hasattr(L, "restrict")
    S = relations.CoordinateProjection(D, np.array([0, 2]))
    y = D(np.arange(4.0))
    assert np.array_equal(S(y).values, [0.0, 2.0])


def test_a_markov_kernel_is_a_relation_of_measure_kind():
    assert pg.MarkovKernel is relations.MarkovKernel and issubclass(relations.AdditiveMarkovKernel, pg.MarkovKernel)
    M = pg.Euclidean(2)
    K = pg.Additive(pg.Identity(M), M.standard_gaussian())
    from pygeoinf0.core.kinds import MEASURE
    assert isinstance(K, pg.MarkovKernel) and K.kind is MEASURE


def test_the_restriction_space_keeps_its_construction():
    H = pg.Sobolev(pg.Interval("[0, 1]"), 1, length_scale=0.1)
    assert isinstance(H.construction, constructions.RestrictionSpace) and H.construction.kind == "restriction"


def test_a_compression_is_p_c_p():
    X = pg.L2(pg.Ball(1.0), realisation="radial_harmonics")
    vol = 4 * math.pi / 3
    phi = X.family([lambda x, y, z: np.ones_like(x) / math.sqrt(vol),
                    lambda x, y, z: z / math.sqrt(4 * math.pi / 15)])
    C = pg.Covariance.spectral([3.0, 2.0], phi)
    R = C.compressed(pg.BandProjection(X, [0]))
    assert float(R.trace()) == pytest.approx(3.0, rel=1e-10) and R.claims.find("compressed") is not None
    assert not hasattr(C, "restricted")


def test_a_subpiece_of_a_domain():
    F = pg.Segment((0.0, 0.0), (2.0, 0.0))
    S = F.subpiece([0.5], [1.5])
    assert S.volume == pytest.approx(1.0) and not hasattr(F, "restricted")


def test_restrictions_to_a_partition_are_the_spaces():
    P = pg.Partition(pg.Interval("[0, 1]"), {"l": "[0, 0.4]", "r": "[0.4, 1]"})
    X = pg.L2(pg.Interval("[0, 1]"))
    R = X.restrictions(P)
    assert R is X.restrictions(P) and R.glue is P.glue(X) and not hasattr(P, "restrict")
    u = X(lambda x: 1 + x ** 2)
    assert (R.glue(R(u)) - u).norm() < 1e-12 * u.norm()
    assert R.claims.find("isometry").status == "theorem"
