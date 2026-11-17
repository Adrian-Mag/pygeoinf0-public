"""The meta layer (plan 009 G11; EXP-030 DESIGN.md G11): trade-off curves, knee rules, operating points, predictive
checks, coverage at a fixed truth, leakage by summand, the PLI variance split and sensitivity lanes. Each is built on
one data-space decomposition (pg.decompose) and is a record that states it is meta (thesis el:59: not one of the seven
elements) and whether it used the datum (el:524-526).

    tc = pg.meta.TradeOff(dec, T)                       closed-form (sigma, TRM) curve for every gamma
    k  = pg.meta.Knee(axes=("trm", "log sigma"), side="convex", refine="continuous").find(tc, lo, hi)
    op = pg.meta.OperatingPoints(tc, lo, hi, knee=k, factor=10.0)
    pc = pg.meta.PredictiveCheck(dec_pli, d, a, nu, bins=4)
    cv = pg.meta.Coverage(dec_pli, T, truth, a=a, nu=nu, level=0.95)
    lk = pg.meta.Leakage(dec, T, gamma)                 per summand of a pg.DirectSum model space
    vs = pg.meta.VarianceSplit(dec_pli, T, a, nu)
    ln = pg.meta.Lane("prior", "length scale", {...}, baseline=...)

The data-dependent choice rules (plan 070 P2, review D3; meta/choice.py) take (problem, method, hyperparameter
family) and return a Choice that states its family and whether it used the datum; the families (meta/families.py)
are named records carried in claims (review B14):

    ch = pg.meta.discrepancy(pb, "least_squares", pg.meta.damping("least_squares"))   gamma with chi^2 = n
    ev = pg.meta.maximum_evidence(pb, "pli", pg.meta.scales())                        (a, nu) by the evidence
    fit = pg.meta.fit_noise(pb, "pli", pg.meta.NoiseFamily(D, groups="s"))            a noise law by the evidence
    k = pg.meta.knee(pb, "sola", pg.meta.damping("sola", lo=1e-4, hi=1e4))            the knee of the trade-off curve
    pb.run("least_squares", **ch.options())                                           a method at the chosen values
"""
from .checks import Coverage, PredictiveCheck
from .checks import ELEMENTS, Lane
from .record import Choice, Record, meta_claims
from .families import HyperparameterFamily, NoiseFamily, damping, scales
from .choice import RULES, EvidenceChoice, NoiseFit, discrepancy, fit_noise, knee, maximum_evidence
from .checks import Leakage, VarianceSplit
from .tradeoff import Knee, KneeResult, OperatingPoints, TradeOff

__all__ = ["TradeOff", "Knee", "KneeResult", "OperatingPoints", "PredictiveCheck", "Coverage", "Leakage",
           "VarianceSplit", "Lane", "ELEMENTS", "Record", "meta_claims",
           # plan 070 P2: the choice rules and their hyperparameter families
           "Choice", "EvidenceChoice", "NoiseFit", "HyperparameterFamily", "NoiseFamily", "damping", "scales", "RULES",
           "discrepancy", "fit_noise", "knee", "maximum_evidence"]
