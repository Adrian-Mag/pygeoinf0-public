"""Conforming finite-subspace SOLA; see unconstrained_limits.md for the proof."""
import time

import numpy as np
import scipy.linalg as la

import sola_demo as S


def projected_gram(M, E, u):
    r"""Return $E M^{-1} E^T$, $E M^{-1}u$, $u^T M^{-1}u$.

    M is the Hilbert Gram matrix of independent trial functions, E their data
    evaluations and u their target evaluations. This is an orthogonal projection,
    not a Euclidean coefficient pairing. A dependent basis is rejected.
    """
    M, E, u = np.asarray(M, float), np.asarray(E, float), np.asarray(u, float)
    if (M.ndim != 2 or M.shape[0] != M.shape[1] or E.ndim != 2
            or E.shape[1] != len(M) or u.shape != (len(M),)):
        raise ValueError('incompatible trial Gram/data/target shapes')
    factor = la.cho_factor(M, lower=True, check_finite=True)
    Z = la.cho_solve(factor, np.column_stack([E.T, u]))
    K = E@Z[:, :-1]
    return (K+K.T)/2, E@Z[:, -1], float(u@Z[:, -1])


def projection_study(levels=(3, 5, 9, 17, 33, 65), nodes=24, target_nodes=32):
    r"""Nested trial spaces $V_h=\operatorname{span}\{k_z:z\in Z_h\}\subset H^2$.

    Point representers are trial functions, not extra observations. Keep the
    continuum objective's tau fixed; independently record quadrature sensitivity.
    """
    ref = S.sobolev_gram(nodes, target_nodes)
    H, F = ref['H'], ref['F']
    tau = S.LAMBDA*ref['tt']
    xref = S.sola_weights(ref['K'], ref['b'], tau)
    d0, d = S.P.data()
    truth = S.target_truth()
    rows, weights = [], []
    for n in levels:
        start = time.perf_counter()
        z = np.linspace(0, 1, n)
        X, Y = np.meshgrid(z, z)
        points = np.column_stack([X.ravel(), Y.ravel()])
        with S.quadrature_nodes(nodes):
            Phi = H.riesz(points)
            M = np.asarray(H.inner(Phi, Phi))
            EF = np.asarray(H.inner(F, Phi))
        K, b, tt = projected_gram(M, EF[:-1], EF[-1])
        x = S.sola_weights(K, b, tau)
        # Exact Hilbert projection residual identities, up to recorded quadrature.
        data_residual2 = float(np.trace(ref['K']-K))
        target_residual2 = float(ref['tt']-tt)
        residual_tol = 2e-9*max(float(np.trace(ref['K'])), ref['tt'])
        if min(data_residual2, target_residual2) < -residual_tol:
            raise ArithmeticError('projection residual incompatible with reference quadrature')
        row = dict(n=n, dimension=n*n, h=1/(n-1),
                   gram_error=float(la.norm(K-ref['K'])/la.norm(ref['K'])),
                   cross_error=float(la.norm(b-ref['b'])/la.norm(ref['b'])),
                   x_error=float(la.norm(x-xref)/la.norm(xref)),
                   data_projection_residual2=data_residual2,
                   target_projection_residual2=target_residual2,
                   # Compute delivered representer mismatch using the full Gram.
                   delivered_mismatch=float(np.sqrt(max(ref['tt']-2*x@ref['b']+x@ref['K']@x,0)/ref['tt'])),
                   seconds=time.perf_counter()-start,
                   **S.estimator_diagnostics(x,d0,d,truth,S.P.SIGMA))
        rows.append(row)
        weights.append(x)
        print(f"H2 trial {n:2d}x{n:2d}: K error={row['gram_error']:.3e}, "
              f"X error={row['x_error']:.3e}, estimate={row['estimate']:.9f}", flush=True)
        del Phi, M, EF
    reference = dict(nodes=nodes,target_nodes=target_nodes,tau=tau,tt=ref['tt'],
                     **S.estimator_diagnostics(xref,d0,d,truth,S.P.SIGMA))
    return rows, np.asarray(weights), reference, xref
