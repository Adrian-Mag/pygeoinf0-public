"""The precision rule (AGENTS.md: a result is computed to a stated precision, or refused with the choices named), in one
place (plan 065 P5; architecture review 2026-11-13, D2).

    TOL = 1e-9          the general rule: a value whose stated error exceeds TOL times its size holds fewer than DIGITS
                        correct digits, and a PrecisionWarning says so; where the error reaches the value itself (no
                        correct digit) the result is refused, unless the caller asked for the numbers
    DIGITS = 9          the same threshold in words
    DECIDE_TOL = 1e-10  core.logic.decide: TRUE when a residual is at most DECIDE_TOL times its scale,
    UNDECIDED = 1e-7    FALSE above UNDECIDED times it, UNKNOWN between (the rounding level)

    assess(value, err)  (lost, none, worst): where err > TOL |value|, where err >= |value|, the largest relative error
    fewer(n, total, what)   the words of a warning: "n of total what hold fewer than 9 correct digits"

The judges keep their own mathematics (what the error of each result is: probability/input_error.py,
probability/posterior_covariance.py, solvers/dataspace/cancellation.py, optimisers/dual/exactness.py, optimisers/dual/exactness_reduce.py) and
take the threshold and its words from here.
"""
import numpy as np

TOL = 1e-9
DIGITS = 9
DECIDE_TOL = 1e-10
UNDECIDED = 1e-7


def assess(value, err):
    """(lost, none, worst) for values and their stated errors (arrays of one shape, or numbers): lost where err > TOL
    |value| (fewer than DIGITS correct digits), none where err >= |value| (no correct digit), worst the largest relative
    error among the lost ones (0.0 when none is lost; inf for a lost zero value)."""
    value, err = np.abs(np.asarray(value, float)), np.asarray(err, float)
    lost = err > TOL * value
    none = err >= value
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(value > 0, err / np.where(value > 0, value, 1.0), np.inf)
    worst = float(np.max(np.where(lost, rel, 0.0), initial=0.0))
    return lost, none, worst


def fewer(n, total, what):
    """"n of total what hold fewer than DIGITS correct digits" (the words of the rule's warning)."""
    return f"{n} of {total} {what} hold fewer than {DIGITS} correct digits"
