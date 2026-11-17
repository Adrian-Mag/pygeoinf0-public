"""Probability measures under the Measure root (plan 067 P5; architecture review 2026-11-13, L5b and B2).

    ProbabilityMeasure(Measure)     a non-negative measure of mass 1 on a set or a space (mu.space is mu.set): the
                                    states of kind measure; GaussianMeasure and Dirac are ones (probability/gaussian.py)
    DiscreteMeasure(atoms, on=Y)    sum_y p_y delta_y on a finite set Y: the laws of kernels between finite sets
                                    (relations/), replacing the dict laws {point: probability}

A probability measure states its mass (1) and its integrals; it need not evaluate mu(A) for sets A (a Gaussian measure on
a Hilbert space evaluates none). Combinations c * mu and mu - nu of probability measures are not formed here (a mixture is
a further choice; the signed combinations of pygeoinf0.measures are for the base measures of a domain).
"""
import numpy as np

from ..core.claims import CHECKED, THEOREM, Claim, Claims
from ..core.errors import Unsupported
from ..core.kinds import MEASURE
from ..core.logic import Truth
from ..measures import Measure
from ..sets import FiniteSet

ROUNDING = 16 * np.finfo(float).eps


class ProbabilityMeasure(Measure):
    """A non-negative measure of mass 1 on a set or a space (module docstring).

    Subclasses set self.set (and self.space, the same object) and their claims, an instance attribute."""

    kind = MEASURE                        # a state of kind measure (plan 067 P4: declared by the class)
    sigma_algebra = "Borel subsets"
    claims = None                         # set by each measure when it is built (cells extend it)
    _probability_status = THEOREM

    def mass(self):
        return 1.0, 0.0

    def _properties(self):
        why = "a probability measure"
        return {"finite": (Truth.TRUE, THEOREM, why), "probability": (Truth.TRUE, self._probability_status, why),
                "signed": (Truth.FALSE, THEOREM, why)}

    def __call__(self, A):
        raise Unsupported(f"{self!r}: the probability of a set is not evaluated for this measure (mass() and "
                          "integrate(f) are)")

    # no linear combinations of probability measures here (module docstring); + is each measure's own operation
    def __mul__(self, c):
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self):
        raise TypeError(f"-{self!r}: negative multiples of a probability measure are not formed")

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __sub__(self, other):
        return NotImplemented


class DiscreteMeasure(ProbabilityMeasure):
    """A probability measure sum_y p_y delta_y on a finite set: the laws of kernels between finite sets.

    atoms: a dict {point: probability} with p_y >= 0 summing to 1 to rounding; on= the finite set it lives on (by default the set of the atoms' points, which must lie in it).

        mu(A)            the probability of a set A (a FiniteSet or a Region of the points)
        mu.at(y)         mu({y}) (0.0 for a point of the set that is not an atom)
        mu.items()       the atoms (point, probability)
        mu.integrate(f)  (sum_y p_y f(y), 0.0), f a function of one point"""

    sigma_algebra = "all subsets"
    _probability_status = CHECKED                   # the masses sum to 1 to rounding

    def __init__(self, atoms, on=None, name="P"):
        atoms = dict(atoms)
        p = np.array(list(atoms.values()), float)
        if not atoms or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("a discrete probability measure needs finite non-negative masses p_y >= 0")
        total = float(p.sum())
        if abs(total - 1.0) > ROUNDING * len(p):
            raise ValueError(f"the masses of a discrete probability measure sum to its total mass 1, not {total!r}")
        Y = on if on is not None else FiniteSet(list(atoms))
        for y in atoms:
            if Y.member(np.atleast_1d(np.asarray(y, float))) is not Truth.TRUE:
                raise ValueError(f"{y!r} is not a point of {Y!r}")
        self.atoms, self.set, self.space, self.name = atoms, Y, Y, name
        self.claims = Claims([Claim("a discrete probability measure", CHECKED,
                                    f"{len(p)} non-negative masses summing to 1 to rounding ({total!r})")]
                             + self.property_claims())

    def at(self, y):
        """mu({y})."""
        return float(self.atoms.get(y, 0.0))

    def items(self):
        return self.atoms.items()

    def __call__(self, A):
        return float(sum(p for y, p in self.atoms.items() if np.atleast_1d(np.asarray(y, float)) in A))

    def integrate(self, f):
        return float(sum(p * f(y) for y, p in self.atoms.items())), 0.0

    def __repr__(self):
        shown = ", ".join(f"{y!r}: {p:.4g}" for y, p in list(self.atoms.items())[:4])
        return f"DiscreteMeasure({{{shown}{', …' if len(self.atoms) > 4 else ''}}})"
