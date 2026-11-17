"""A.inv is the inverse, or a refusal that names the choices (plan 054). Against exact answers: Hilbert matrices solved in
rationals, and spectra set by hand. A.pseudo_inverse(rcond) is the only way to a threshold."""
from fractions import Fraction as F

import numpy as np
import pytest

import pygeoinf0 as pg
from pygeoinf0.core.errors import Refused

EPS = np.finfo(float).eps


def hilbert_exact(n, b):
    A = [[F(1, i + j + 1) for j in range(n)] + [F(x)] for i, x in enumerate(b)]
    for i in range(n):
        A[i] = [v / A[i][i] for v in A[i]]
        for k in range(n):
            if k != i and A[k][i] != 0:
                A[k] = [u - A[k][i] * v for u, v in zip(A[k], A[i])]
    return np.array([float(r[n]) for r in A])


def op(M):
    A = pg.Matrix(M)
    A._sa = (True, "derived", "symmetric by construction")
    return A


def spectrum(lam, seed=1):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(len(lam), len(lam))))
    M = (Q * np.asarray(lam)) @ Q.T
    return 0.5 * (M + M.T), Q


@pytest.mark.parametrize("n, digits", [(6, 8), (8, 5), (10, 2)])
def test_ill_conditioned_is_solved_to_the_digits_it_has(n, digits):
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    b = [F(k + 1, 7) for k in range(n)]
    A = op(H)
    x = A.inv(A.domain(np.array([float(v) for v in b]))).values
    exact = hilbert_exact(n, b)
    assert np.linalg.norm(x - exact) <= 10.0 ** -digits * np.linalg.norm(exact)


def test_a_direction_at_the_floor_is_refused_and_the_choices_named():
    M, _ = spectrum([1.0] * 4 + [1e-17, 1e-18])
    A = op(M)
    with pytest.raises(Refused) as e:
        A.inv(A.domain(np.ones(6)))
    assert "2 of 6" in e.value.reason
    assert any("pseudo_inverse" in s for s in e.value.needs) and any("shift" in s for s in e.value.needs)


def test_the_claims_carry_the_numbers():
    M, _ = spectrum(np.logspace(0, -12, 8))
    A = op(M)
    text = " ".join(str(c) for c in A.inv.claims)
    assert "1e+12" in text and "12 of 16" in text and "nothing dropped" in text
    B = op(spectrum([1.0] * 4 + [1e-18, 1e-18])[0])
    assert "no inverse here" in " ".join(str(c) for c in B.inv.claims)


def test_nothing_is_dropped_just_above_the_floor():
    lam = np.logspace(0, -14, 10)
    M, Q = spectrum(lam)
    A = op(M)
    y = Q[:, -1]
    x = A.inv(A.domain(y)).values
    assert 1e13 < np.linalg.norm(x) < 1e15                       # the cut at 1e-12 gave 0 in this direction


@pytest.mark.parametrize("gamma", [1e-13, 1e-10])
def test_a_small_shift_is_not_cut(gamma):
    lam = np.logspace(0, -14, 12)
    M, Q = spectrum(lam)
    A = op(M)
    S = A + gamma * pg.Identity(A.domain)
    S._sa = A._sa
    x = S.inv(A.domain(Q[:, -1])).values
    assert np.linalg.norm(x) == pytest.approx(1.0 / (lam[-1] + gamma), rel=1e-2)


def test_the_pseudo_inverse_is_asked_for_with_its_threshold():
    lam = [1.0, 1e-3, 1e-14]
    M, Q = spectrum(lam)
    A = op(M)
    P = A.pseudo_inverse(1e-12)
    y = Q @ np.ones(3)
    assert P(A.domain(y)).values == pytest.approx(Q[:, 0] + 1e3 * Q[:, 1], rel=1e-9, abs=1e-9)
    assert "1 direction(s) below" in " ".join(str(c) for c in P.claims)
    with pytest.raises(TypeError):
        pg.Eigh()
    with pytest.raises(ValueError):
        A.pseudo_inverse(None)


def blocks(mats, shift=None):
    n = sum(len(m) for m in mats)
    return pg.BlockDiagonal(pg.Euclidean(n), mats, name="K", shift=shift)


def test_blocks_refuse_relative_to_their_own_block_and_keep_a_small_block():
    good, _ = spectrum([1e-20, 2e-20, 3e-20])
    bad, _ = spectrum([1.0, 1.0, 1e-18])
    A = blocks([good, 1e6 * spectrum([1.0, 2.0, 3.0])[0]])
    A.inv(A.domain(np.ones(6)))                                  # scale apart, each block is fine
    B = blocks([bad, good])
    with pytest.raises(Refused) as e:
        B.inv(B.domain(np.ones(6)))
    assert "1 of 6" in e.value.reason and "1 of 2 blocks" in e.value.reason


def test_blocks_solve_ill_conditioned_with_nothing_dropped():
    lam = np.logspace(0, -14, 6)
    M, Q = spectrum(lam)
    A = blocks([M])
    x = A.inv(A.domain(Q[:, -1])).values
    assert np.linalg.norm(x) == pytest.approx(1 / lam[-1], rel=0.5)
    assert "nothing dropped" in A.inv.report()


def test_blocks_pseudo_inverse_states_its_count():
    M, Q = spectrum([1.0, 1.0, 1e-14])
    A = blocks([M, M])
    P = A.pseudo_inverse(1e-12)
    assert "2 directions dropped, in 2 blocks" in P.report()


def test_a_shift_on_blocks_that_leaves_a_direction_at_the_floor_is_refused():
    M, _ = spectrum([1.0, 1.0, 0.0])
    A = blocks([M], shift=[np.full(3, 1e-18)])
    with pytest.raises(Refused):
        A.inv(A.domain(np.ones(3)))


@pytest.mark.parametrize("metric", [np.array([1.0, 4.0, 9.0, 16.0])])
def test_a_weighted_gram_refuses_too_and_has_a_pseudo_inverse(metric):
    M, _ = spectrum([1.0, 1.0, 1.0, 1e-18])
    D = pg.Euclidean(4, weights=metric)
    from pygeoinf0.operators.plain_gram import WeightedGram
    A = WeightedGram(D, blocks([M / np.sqrt(np.outer(metric, metric))]))
    with pytest.raises(Refused):
        A.inv(D(np.ones(4)))
    assert A.pseudo_inverse(1e-12)(D(np.ones(4))).values.size == 4
    assert "rcond 1e-12" in " ".join(str(c) for c in A.pseudo_inverse(1e-12).claims)


def test_a_scaled_identity_keeps_the_block_route_on_a_weighted_gram():
    from pygeoinf0.operators.plain_gram import WeightedGram
    D = pg.Euclidean(6, weights=np.ones(6))
    big, small = spectrum([1e12, 1e12, 1e12])[0], spectrum([1e-8, 2e-8, 3e-8])[0]
    A = WeightedGram(D, blocks([big, small]))
    S = A + 1e-9 * pg.Identity(D)                                # as one matrix, the small block is at the floor
    y = np.ones(6)
    exact = np.r_[np.linalg.solve(big + 1e-9 * np.eye(3), y[:3]), np.linalg.solve(small + 1e-9 * np.eye(3), y[3:])]
    assert S.inv(D(y)).values == pytest.approx(exact, rel=1e-9)


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-9])
def test_cg_above_dense_max_does_not_return_zero_on_small_data(scale):
    M, _ = spectrum(np.linspace(1, 2, 50))
    A = op(M)
    y = scale * np.random.default_rng(0).normal(size=50)
    with pg.settings.using(dense_max=10):
        x = A.inv(A.domain(y)).values
    assert np.linalg.norm(x - np.linalg.solve(M, y)) <= 0.02 * np.linalg.norm(np.linalg.solve(M, y))


def test_cg_that_does_not_reach_its_tolerance_is_refused():
    M, _ = spectrum(np.logspace(0, -8, 30))
    A = op(M)
    with pytest.raises(Refused):
        A.inverse(pg.CG(rtol=1e-14, maxiter=3))(A.domain(np.ones(30)))


def test_a_singular_matrix_is_refused_by_inv_and_warned_by_an_explicit_lu():
    A = pg.Matrix(np.array([[1.0, 2.0], [3.0, 6.0 + 1e-17]]))
    with pytest.raises(Refused):
        A.inv(A.domain(np.ones(2)))
    with pytest.warns(pg.errors.PrecisionWarning, match="no correct digit"):
        A.inverse(pg.LU()).factorisation()


@pytest.mark.parametrize("solver, needs", [(pg.Cholesky(), "digits lost"), (pg.Direct(), "digits lost"), (pg.LU(), "digits lost")])
def test_an_explicit_solver_warns_when_digits_are_at_stake(solver, needs):
    M, _ = spectrum(np.logspace(0, -12, 8))
    A = op(M)
    with pytest.warns(pg.errors.PrecisionWarning, match=needs):
        A.inverse(solver).factorisation()
    assert "12 of 16" in " ".join(str(c) for c in A.inverse(solver).claims)


def test_well_conditioned_gives_no_warning():
    import warnings
    A = op(spectrum(np.linspace(1, 2, 8))[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", pg.errors.PrecisionWarning)
        A.inv(A.domain(np.ones(8)))


def test_refuse_false_returns_the_numbers_with_a_warning_and_names_the_count():
    M, Q = spectrum([1.0, 1.0, 1e-18])
    A = op(M)
    with pytest.warns(pg.errors.PrecisionWarning, match="1 directions hold no correct digit"):
        P = A.inverse(pg.Direct(refuse=False))
        P(A.domain(np.ones(3)))
    assert "no correct digit" in P.report()
    B = blocks([M, spectrum([1.0, 2.0, 3.0])[0]])
    with pytest.raises(Refused) as e:
        B.inv(B.domain(np.ones(6)))
    assert any("BlockDirect(refuse=False)" in s for s in e.value.needs)
    with pytest.warns(pg.errors.PrecisionWarning):
        assert "no correct digit" in B.inverse(pg.BlockDirect(refuse=False)).report()


def test_block_direct_can_be_given_explicitly_on_a_weighted_gram():
    from pygeoinf0.operators.plain_gram import WeightedGram
    D = pg.Euclidean(3, weights=np.ones(3))
    A = WeightedGram(D, blocks([spectrum([1.0, 2.0, 3.0])[0]]))
    y = D(np.ones(3))
    assert A.inverse(pg.BlockDirect())(y).values == pytest.approx(A.inv(y).values)


def test_every_solver_says_what_it_is_for():
    import inspect
    from pygeoinf0 import solvers
    table = solvers.usage()
    public = {n for n, c in vars(solvers).items() if inspect.isclass(c) and issubclass(c, solvers.Solver) and c is not solvers.Solver}
    assert set(table) == public
    for row in table.values():
        assert set(row) == {"computes", "when", "error", "cost", "simplification"} and all(row.values())
    assert table["Eigh"]["simplification"] != "none" and table["CG"]["simplification"] != "none"
