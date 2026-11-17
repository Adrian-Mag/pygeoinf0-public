"""The scale ladder (plan 005) at its smallest rung: every problem is solved by the same user code, and the solution
meets the discrepancy principle it was asked for."""
import inspect

import numpy as np
import pytest

import ladder


@pytest.mark.parametrize("name", ["L1", "L2", "L3", "L4"])
def test_same_user_code_solves_every_problem(name):
    P = ladder.build(name, 40)
    G, alpha, post = ladder.solve(P)
    r = G(post.mean) - P.d
    assert P.D.norm(r) ** 2 == pytest.approx(P.n, rel=1e-6)          # chi^2 = n: the discrepancy principle
    assert alpha > 0 and np.all(np.isfinite(post.std(P.measures[:3])))


def test_the_user_code_has_no_size_dependence():
    src = inspect.getsource(ladder.solve)
    assert "P.n" not in src and "len(" not in src
