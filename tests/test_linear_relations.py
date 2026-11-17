"""Linear relations (plan 008 P1; EXP-028 DESIGN.md L1-L8 and the Corollary), checked against independent routes:
graphs and orthogonal complements by SVD in weighted metrics, least squares with explicit nuisance columns, and the
package's own minimum-norm code."""
import numpy as np
import pytest
from scipy.linalg import block_diag, null_space

import pygeoinf0 as pg
from pygeoinf0.core.claims import THEOREM
from pygeoinf0.relations import compose
from pygeoinf0.convex.subspaces import basis_matrix

rng = np.random.default_rng(3)
X = pg.Euclidean(4, weights=rng.uniform(1, 3, 4))
_A = rng.normal(size=(3, 3))
Y = pg.Euclidean(3, metric=_A @ _A.T + 3 * np.eye(3))
Z = pg.Euclidean(2, weights=[2.0, 0.5])
G = pg.Matrix(rng.normal(size=(3, 4)), X, Y, name="G")
S = pg.Matrix(rng.normal(size=(2, 3)), Y, Z, name="S")
MUL = pg.Span([Y(rng.normal(size=3))], name="E")
DOM = pg.Span([X(rng.normal(size=4)), X(rng.normal(size=4))], name="V")
R = pg.LinearRelation(G, mul=MUL, dom=DOM)              # partial and multivalued
R2 = pg.LinearRelation(G, mul=MUL)                      # total, multivalued


def graph(Rel):
    """Columns: a basis of the graph {(x, y)} in standard coordinates."""
    Bx, Bm = basis_matrix(Rel.dom), basis_matrix(Rel.mul)
    Gm = Rel.G.matrix()
    return np.r_[np.c_[Bx, np.zeros((Bx.shape[0], Bm.shape[1]))], np.c_[Gm @ Bx, Bm]]


def same(A, B, tol=1e-9):
    r = lambda M: np.linalg.matrix_rank(M, tol) if M.shape[1] else 0
    return r(np.c_[A, B]) == r(A) == r(B)


def perp(A, M):
    return null_space(A.T @ M)


def W(sp):
    return sp.metric_matrix()


def test_adjoint_is_the_orthogonal_complement_of_the_rotated_graph():          # L3
    Gam = graph(R)
    nx = X.dim
    VG = np.r_[Gam[nx:], -Gam[:nx]]                                           # V(x, y) = (y, -x)
    assert same(graph(R.adjoint), perp(VG, block_diag(W(Y), W(X))))
    G0 = pg.LinearRelation(G)                                                  # the adjoint of an operator's graph
    VG0 = np.r_[graph(G0)[nx:], -graph(G0)[:nx]]
    assert same(np.r_[np.eye(3), G.adjoint.matrix()], perp(VG0, block_diag(W(Y), W(X))))


def test_mul_and_ker_of_the_adjoint():                                          # L4
    Rs = R.adjoint
    assert same(basis_matrix(Rs.mul), perp(basis_matrix(R.dom), W(X)))
    Gs = graph(Rs)                                                             # (k, h) columns, k in Y, h in X
    c = null_space(Gs[3:])                                                     # (k, 0) in R*: ker R*
    ran = graph(R)[4:]
    assert same(Gs[:3] @ c, perp(ran, W(Y)))
    assert R2.adjoint.is_single_valued is pg.Truth.TRUE                        # dom R2 = X is dense: R2* single-valued
    assert R.adjoint.is_single_valued is pg.Truth.FALSE                        # dom R is not


def test_double_adjoint_is_the_closure():                                       # L5
    assert same(graph(R.adjoint.adjoint), graph(R))
    assert R.closure() is R


def test_adjoint_of_a_composition_and_of_the_converse():                       # L6
    lhs, rhs = compose(S, R).adjoint, compose(R.adjoint, S.adjoint)
    assert same(graph(lhs), graph(rhs))
    a = graph(R2.converse().adjoint)                                           # (R^-1)*: X -> Y
    b = graph(R2.adjoint)                                                      # R*: Y -> X; its converse swaps blocks
    assert same(a, np.r_[b[3:], b[:3]])


def test_operator_part_is_the_least_norm_value():                               # L7
    x = X(rng.normal(size=4))
    Lt = np.linalg.cholesky(W(Y)).T                                            # ||y||_Y = ||Lt y||
    E = MUL.vectors.values.T
    c = np.linalg.lstsq(Lt @ E, -(Lt @ G.A @ x.values), rcond=None)[0]
    assert np.allclose(R2.operator_part()(x).values, G.A @ x.values + E @ c, atol=1e-12)
    Xg = R2.domain_space()                                                     # the graph norm
    assert Xg.norm(Xg(x.values)) ** 2 == pytest.approx(X.norm(x) ** 2 + Y.norm(R2.operator_part()(x)) ** 2, rel=1e-12)


def test_quotient_norm_is_the_distance_to_mul():                                 # L8
    q = R2.quotient_map()
    y = Y(rng.normal(size=3))
    Lt = np.linalg.cholesky(W(Y)).T
    E = MUL.vectors.values.T
    c = np.linalg.lstsq(Lt @ E, Lt @ y.values, rcond=None)[0]
    assert np.linalg.norm(q(y).values) == pytest.approx(np.linalg.norm(Lt @ (y.values - E @ c)), rel=1e-12)
    assert np.allclose(q(Y.standard_gaussian()).covariance.matrix(), np.eye(2), atol=1e-12)


@pytest.mark.parametrize("make", [lambda: pg.Sobolev("[0, 1]", 1, length_scale=0.1),
                                  lambda: pg.SobolevForm("[0, 1]", 1, length_scale=0.1).space()],
                         ids=["restriction", "intrinsic"])
@pytest.mark.parametrize("weighted", [False, True])
def test_minimum_norm_is_the_operator_part_of_the_converse(make, weighted):     # Corollary
    H = make()
    n = 8
    D = pg.Euclidean(n, covariance=np.linspace(0.5, 2.0, n) ** 2) if weighted else pg.Euclidean(n)
    Gf = pg.Functionals(H.riesz([pg.Interval(a, a + 0.1) for a in np.linspace(0, 0.85, n)]), into=D)
    d = D(np.linspace(-1, 1, n))
    m_ref = Gf.adjoint((Gf @ Gf.adjoint).inv(d))
    m = Gf.converse().operator_part()(d)
    assert m.distance(m_ref) <= 1e-14 * m_ref.norm()
    K = Gf.converse()
    k = K.mul.project(H.riesz(pg.DiracMeasure(0.5)))                         # an element of ker G
    assert np.abs(Gf(k).values).max() <= 1e-10 * Gf(H.riesz(pg.DiracMeasure(0.5))).norm()
    assert K.mul.member(k) is pg.Truth.TRUE and K.mul.member(H.riesz(pg.DiracMeasure(0.5))) is pg.Truth.FALSE
    assert K.dom.dim == n                                                    # ran G = R^n (independent kernels)


def test_origin_times_minimum_norm_against_contrasts():
    nm, N = 30, 9
    Xe, D = pg.Euclidean(nm), pg.Euclidean(N, weights=rng.uniform(0.5, 2, N))
    Ge = pg.Matrix(rng.normal(size=(N, nm)), Xe, D)
    Em = np.eye(3)[[0, 0, 0, 1, 1, 1, 1, 2, 2]]
    Rel = pg.Additive(Ge, pg.Span([D(Em[:, j]) for j in range(3)], name="origin times"))
    assert isinstance(Rel, pg.LinearRelation) and Rel.mul.dim == 3
    d = D(rng.normal(size=N))
    m = Rel.converse().operator_part()(d)
    Q = null_space(Em.T).T                                                     # any Q with ker Q = span 1_e
    m_ref = np.linalg.pinv(Q @ Ge.A) @ (Q @ d.values)
    assert np.allclose(m.values, m_ref, rtol=0, atol=1e-12 * np.abs(m_ref).max())
    assert Rel(m).member(d) is pg.Truth.TRUE


def test_form_operator_is_a_closed_densely_defined_self_adjoint_relation():
    A = pg.SobolevForm("[0, 1]", 1, length_scale=0.1).operator()
    Ar = A                                                       # the operator is its relation (plan 068 P3)
    assert Ar.claims.status("closed graph") == (pg.Truth.TRUE, THEOREM)
    assert Ar.claims.status("densely defined") == (pg.Truth.TRUE, THEOREM)
    assert Ar.is_single_valued is pg.Truth.TRUE and Ar.adjoint is Ar
    assert Ar.domain_space() == A.domain_space()
    assert Ar.dom.member(A.domain_space().zero) is pg.Truth.TRUE
    assert Ar.dom.member(Ar.domain.zero) is pg.Truth.TRUE                    # 0 is in D(A) (plan 068 P3)
    assert Ar.dom.member(Ar.domain(lambda t: t, resolution=0.5)) is pg.Truth.UNKNOWN    # an element of L2: undecided
    assert Ar.domain == pg.L2("[0, 1]") and Ar.dom.perp().dim == 0           # dense: its complement is {0}
