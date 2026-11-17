"""A small plugin for the tests of pg.port.plugins (tests/test_plugins.py): a squaring map on R^3 with its Jacobian, the
identity coordinates of a weighted R^3, and one claim stated too strongly (it must be capped)."""
import numpy as np

import pygeoinf0 as pg
from pygeoinf0.core.claims import THEOREM, Claim

M = pg.Euclidean(3, weights=np.array([1.0, 2.0, 3.0]))
SQUARE = pg.port.External(M, pg.Euclidean(3), lambda x: x ** 2, jacobian=lambda x: np.diag(2 * x), name="square",
                     code="demo square")
COORDS = pg.port.Coordinates.identity(M)

PLUGIN = pg.port.plugins.Plugin(
    name="demo", version="0.1", port=">=1,<2", objects={"square": SQUARE, "coords": COORDS},
    claims=(Claim("square is smooth", THEOREM, "a polynomial", key="smooth"),),
    requires=("numpy",), licence="MIT", source="tests/plugin_demo.py",
    examples=lambda: [(SQUARE, M(np.array([1.0, -0.5, 2.0])))])
