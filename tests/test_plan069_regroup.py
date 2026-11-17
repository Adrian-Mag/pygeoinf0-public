"""Plan 069 P4 (review C4, A11, B9, E2; decision Q6): realisations/ grouped by the mathematics, a clean break (Q5).

    series/    expansions in a basis (piecewise Legendre, tensor, harmonic, radial-harmonic, direct sums)
    rkhs/      the pairing oracle <k_mu, k_nu> = int int K dmu dnu (kernels, atoms, energy engines)
    spectra/   spectra that are realisations (Laplacian closed forms, Bessel, Sturm-Liouville Galerkin)
    bounds/    certified bounds of realised functions (extrema, L1 integrals, pointwise boxes)

cells.py is piecewise_constant.py and a Cells realisation can be built on a domains.Partition; the continuum's
spherical harmonics are spaces/harmonic.py (pg.HarmonicConvention); the worker pools are pygeoinf0.execution."""
import ast
import importlib
import os

import numpy as np
import pytest

import pygeoinf0 as pg

NEW = {
    "realisations.series.legendre": ["PiecewiseLegendre", "fit_density"],
    "realisations.series.sparse": ["SparsePiecewiseLegendre"],
    "realisations.series.sup": ["SupIntervalBackend"],
    "realisations.series.tensor": ["TensorSeries", "TensorBackend"],
    "realisations.series.local": ["PS", "LocalBackend"],
    "realisations.series.separated": ["ProductFamily", "KernelExpansion", "LazyCombination", "KronGram"],
    "realisations.series.cuts": ["restricted", "glued"],
    "realisations.series.spherical": ["SphericalHarmonicSeries", "SphereBackend"],
    "realisations.series.sphere_sobolev": ["SphereSobolevBackend"],
    "realisations.series.harmonic_degrees": ["band", "scale_degrees", "coefficient_columns"],
    "realisations.series.radial_harmonic": ["RadialHarmonicSeries", "SeparableRadialHarmonics",
                                            "RadialHarmonicsBackend"],
    "realisations.series.radial_fit": ["fit_rh"],
    "realisations.series.radial_direct": ["gram_sr_rh"],
    "realisations.series.direct_sum": ["DirectSumRealisation", "Symbolic", "SumBackend"],
    "realisations.rkhs.backend": ["SobolevRealisation", "RepresenterCombinations", "RKHSBackend"],
    "realisations.rkhs.kernels": ["Matern"],
    "realisations.rkhs.atoms": ["Atoms", "atoms_inside"],
    "realisations.rkhs.pairs": ["energies"],
    "realisations.rkhs.energy": ["ExactEngine"],
    "realisations.rkhs.hmatrix": ["HEngine"],
    "realisations.rkhs.spectral": ["SpectralEngine"],
    "realisations.rkhs.green": ["CorrectedEngine"],
    "realisations.rkhs.box_form": ["BoxForm"],
    "realisations.rkhs.weighted": ["SegmentPiece"],
    "realisations.spectra.bessel": ["RadialBessel"],
    "realisations.spectra.sl_galerkin": ["GalerkinSpectrum"],
    "realisations.spectra.laplace_spectra": ["SphereSpectrum"],
    "realisations.spectra.laplacian_covariance": ["bessel_potential"],
    "realisations.bounds.extrema": ["min_brackets"],
    "realisations.bounds.sphere_l1": ["zonal_l1"],
    "realisations.bounds.box_integrals": ["IntervalRule", "SphereRule"],
    "realisations.piecewise_constant": ["Cells", "CellValues", "CellsBackend"],
    "spaces.harmonic": ["HarmonicConvention", "PACKAGE", "THESIS", "labels", "values", "index", "funk_hecke",
                        "legendre_P", "assoc", "rows", "order_rows"],
    "execution": ["workers", "map_blocks", "Pool"],
}
GONE = ["legendre", "sparse", "sup", "tensor", "local", "separated", "cuts", "spherical", "sphere_sobolev", "harmonics",
        "radial_harmonic", "radial_fit", "radial_direct", "direct_sum", "kernels", "atoms", "pairs", "energy", "hmatrix",
        "spectral", "spectral_terms", "spectral_images", "surfaces", "segment_partition", "weighted",
        "weighted_contacts", "weighted_grid", "weighted_solids", "weighted_spectral", "weighted_volume", "green",
        "box_form", "bessel", "sl_galerkin", "laplace_spectra", "laplacian_covariance", "extrema", "sphere_l1",
        "box_integrals", "cells", "execution"]
ACRONYMS = {"realisations.series.legendre": "PL", "realisations.series.sparse": "SPL",
            "realisations.series.tensor": "TR", "realisations.series.spherical": "SH",
            "realisations.series.radial_harmonic": ("RH", "SR"), "realisations.rkhs.backend": ("SR", "KX"),
            "realisations.series.direct_sum": "DS", "realisations.series.separated": ("SF", "EX", "LC")}


@pytest.mark.parametrize("path, names", sorted(NEW.items()))
def test_new_homes(path, names):
    mod = importlib.import_module(f"pygeoinf0.{path}")
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"pygeoinf0.{path} lacks {missing}"


@pytest.mark.parametrize("old", GONE)
def test_old_paths_are_gone(old):
    """A clean break (Q5): no alias modules, no forwarders."""
    with pytest.raises(ImportError):
        importlib.import_module(f"pygeoinf0.realisations.{old}")


@pytest.mark.parametrize("path, old", sorted(ACRONYMS.items()))
def test_public_acronym_classes_have_mathematical_names(path, old):
    mod = importlib.import_module(f"pygeoinf0.{path}")
    olds = (old,) if isinstance(old, str) else old
    assert not [o for o in olds if hasattr(mod, o)], f"{path} still has {olds}"


def test_the_harmonic_convention_is_the_continuums():
    from pygeoinf0.spaces import harmonic
    assert pg.HarmonicConvention is harmonic.HarmonicConvention is pg.spaces.HarmonicConvention
    assert "HarmonicConvention" in pg.__all__
    assert not hasattr(pg.realisations, "HarmonicConvention") and not hasattr(pg.realisations, "THESIS")
    # the continuum module imports nothing of the numerical column (only the quadrature rules)
    src = open(harmonic.__file__).read()
    assert "realisations" not in "".join(ast.unparse(n) for n in ast.walk(ast.parse(src))
                                         if isinstance(n, (ast.Import, ast.ImportFrom)))
    # Y_00 = 1/sqrt(4 pi), Y_10 = sqrt(3/(4 pi)) cos(theta) (row l^2 + j = 1): the package's orthonormal harmonics
    v = harmonic.values(1, np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(v[0], [1 / np.sqrt(4 * np.pi), np.sqrt(3 / (4 * np.pi)), 0.0, 0.0], atol=1e-15)


def test_the_pairing_oracle_states_its_formula():
    from pygeoinf0.realisations import rkhs
    assert "int int K(x, y) dmu(x) dnu(y)" in rkhs.__doc__ and "<k_mu, k_nu>" in rkhs.__doc__


def test_execution_is_runtime_beside_settings():
    from pygeoinf0 import execution
    tree = ast.parse(open(execution.__file__).read())
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level}
    assert imported <= {"settings"}, imported
    settings = importlib.import_module("pygeoinf0.settings")
    assert execution.__name__ == "pygeoinf0.execution"
    assert os.path.dirname(execution.__file__) == os.path.dirname(settings.__file__)


# ------------------------------------------------------------------ piecewise constant on a Partition (A11, B9)
def _strips():
    """A Partition of [0, 1] x [0, 2] into five boxes that is not a tensor mesh (two strips cut differently)."""
    return pg.Partition(pg.Box((0, 1), (0, 2)), {
        "a": pg.Box((0, 0.4), (0, 0.5)), "b": pg.Box((0, 0.4), (0.5, 2)),
        "c": pg.Box((0.4, 1), (0, 1.2)), "d": pg.Box((0.4, 1), (1.2, 1.7)), "e": pg.Box((0.4, 1), (1.7, 2))})


def test_cells_on_a_partition_are_its_pieces_with_its_tiling():
    P = _strips()
    C = pg.realisations.Cells(P)
    assert C.partition is P and C.n == len(P) and C.names == P.order
    want = [(np.array([s.a for s in P[n].sides]), np.array([s.b for s in P[n].sides])) for n in P.order]
    assert all(np.array_equal(C.lo[i], a) and np.array_equal(C.hi[i], b) for i, (a, b) in enumerate(want))
    assert abs(C.volumes.sum() - P.domain.volume) <= 1e-15 * P.domain.volume
    tiling = C.claims.find("tiling")
    assert tiling.status == P.claims.find("tiling").status and "Partition" in tiling.reason


def test_cells_on_a_refined_partition_take_its_leaves():
    P = pg.Partition(pg.Box((0, 1), (0, 1)), {"l": pg.Box((0, 0.5), (0, 1)), "r": pg.Box((0.5, 1), (0, 1))})
    Q = P.refine("r", [pg.Box((0.5, 1), (0, 0.3)), pg.Box((0.5, 1), (0.3, 1))], names=["r0", "r1"])
    C = pg.realisations.Cells(Q)
    assert C.names == ("l", "r0", "r1") and np.allclose(C.volumes, [0.5, 0.15, 0.35])


def test_cells_refuse_a_partition_not_of_boxes():
    P = pg.Partition("[0, 1]", ["[0, 0.4]", "[0.4, 1]"])
    with pytest.raises(pg.errors.Unsupported, match="boxes"):
        pg.realisations.Cells(P)


def test_l2_on_a_partition_realises_the_cell_averages():
    """pg.L2(P.domain, realisation=Cells(P)): an element is realised by its averages on the pieces (the orthogonal
    projection onto the functions constant on each piece), exact here for a bilinear function."""
    P = _strips()
    M = pg.L2(P.domain, realisation=pg.realisations.Cells(P))
    u = M(lambda x, y: 1 + 2 * x - y + 3 * x * y)
    for i, n in enumerate(P.order):
        (a, b), (c, d) = [(s.a, s.b) for s in P[n].sides]
        mx, my = (a + b) / 2, (c + d) / 2
        assert abs(u.realisation.V[i, 0] - (1 + 2 * mx - my + 3 * mx * my)) <= 1e-13
