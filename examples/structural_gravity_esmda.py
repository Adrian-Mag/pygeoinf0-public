"""Gravity of a geological interface, inverted for the interface's heights (plan 020 P4; three plugins).

The parameters are the depths of five surface points of one interface in a GemPy model (pygeoinf0-gempy); the
geology gives densities on 400 prisms (2400 kg/m^3 above, 2900 below: cell averages of the lithology); their gravity
is g_z at 49 points 10 m above the surface (pygeoinf0-harmonica). The map from depths to data, a composition of
pygeoinf0 maps across two plugins, has no derivative (the lithology is piecewise constant in the depths), so the
posterior is approached with ES-MDA (pygeoinf0-sampling): 100 members, 4 updates, 400 GemPy runs.

What it gives (europa, 2026-11-05, plan 054 P4): one forward 0.5 s, ES-MDA 214-331 s; prior -400 +- 100 m; the five
depths recovered to 7-31 m posterior sd, the truth within 0.3-1.3 sd of the ensemble mean. The deepest point (-520 m,
under the weakest anomaly) is the least resolved. ES-MDA's spread is a Gaussian approximation here (the claim says so).
The ensemble is sensitive to rounding in the data: the seeded noise draw changed by 1.6e-16 relative when the sampling
jitter was removed (plan 054 P3), and the deepest point moved from -571 +- 29 m (1.8 sd; 2026-10-31) to -552 +- 31 m.

    python examples/structural_gravity_esmda.py     (needs plugins/pygeoinf0-{gempy,harmonica,sampling} on the path)
"""
import time
import warnings

import numpy as np

import pygeoinf0 as pg
import pygeoinf0_gempy as pge
import pygeoinf0_harmonica as ph
import pygeoinf0_sampling as ps

warnings.filterwarnings("ignore")


def geology(z):
    import gempy as gp
    m = gp.create_geomodel(project_name="interface", extent=[0, 1000, 0, 1000, -1000, 0], resolution=[4, 4, 4],
                           structural_frame=gp.data.StructuralFrame.initialize_default_structure())
    gp.add_surface_points(m, x=np.r_[150.0, 500, 850, 300, 700], y=np.r_[500.0, 500, 500, 200, 800], z=z,
                          elements_names=["surface1"] * 5)
    gp.add_orientations(m, x=[500], y=[500], z=[-400], elements_names=["surface1"], pole_vector=[[0, 0, 1]])
    return m


def main():
    e = n = np.linspace(0.0, 1000.0, 11)
    u = np.linspace(-1000.0, 0.0, 5)
    M, C, P = ph.prism_space(e, n, u)
    truth = np.r_[-300.0, -450, -380, -520, -330]
    S = pge.structural_map(geology(truth), C, (2400.0, 2900.0), points=range(5), samples=2)
    g = np.linspace(50, 950, 7)
    X = np.column_stack([*(a.ravel() for a in np.meshgrid(g, g)), np.full(49, 10.0)])
    F = ph.gravity_operator(C, P, X) @ S                                  # depths -> g_z (mGal)
    R, D = S.domain, F.codomain

    t = time.time()
    d_clean = F(R(truth))
    print(f"one forward (GemPy + gravity): {time.time() - t:.2f} s; anomaly range {np.ptp(d_clean.values):.2f} mGal")
    noise = pg.GaussianMeasure(D, 0.05 ** 2 * pg.Identity(D))
    d = d_clean + noise.sample(seed=4)
    prior = pg.GaussianMeasure(R, pg.Matrix(100.0 ** 2 * np.eye(5), R, R), mean=R(np.full(5, -400.0)))
    problem = ps.SamplingProblem(F, d, noise, prior)
    t = time.time()
    s = ps.esmda(problem, ensemble=100, iterations=4, seed=5)
    print(f"ES-MDA: {s.info['evaluations']} forwards in {time.time() - t:.0f} s")
    print(s.claims)
    sd = np.sqrt(np.diag(s.cov))
    for i in range(5):
        print(f"  point {i}: true {truth[i]:6.0f} m   prior -400 +- 100   ES-MDA {s.mean[i]:6.0f} +- {sd[i]:4.0f}"
              f"   ({(s.mean[i] - truth[i]) / sd[i]:+.1f} sd)")


if __name__ == "__main__":
    main()
