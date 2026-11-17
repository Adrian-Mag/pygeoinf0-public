"""Diagonals of quadratic forms without the matrix (plan 070 P3: rowquad from probability/lazy_covariance.py, where plan
070 P1 had put it, down to the engine that uses it: the closed forms of a factored family, the lazy covariances they
return and the property families all read it from here)."""
import numpy as np


def rowquad(Phi, V):
    """diag(Phi V Phi^T) without the p x p matrix: (Phi V) * Phi summed by rows (V diagonal: Phi^2 diag(V)). The diagonal
    of the lazy covariance (probability/lazy_covariance.py), of the closed forms of a factored family (factored.py) and
    of the property families (properties/families.py)."""
    V = np.asarray(V, float)
    if not np.any(V - np.diag(np.diag(V))):
        return (Phi * Phi) @ np.diag(V)
    out = np.empty(Phi.shape[0])
    for a in range(0, Phi.shape[0], 8192):
        P = Phi[a:a + 8192]
        out[a:a + 8192] = np.einsum("pb,pb->p", P @ V, P)
    return out
