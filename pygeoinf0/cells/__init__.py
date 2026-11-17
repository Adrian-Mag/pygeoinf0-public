"""The state-by-relation table (plan 008 P3): push(state, R) forward, update(prior, R, d) backward, table() as
documentation. The cells live in forward.py and backward.py; registry.py dispatches and checks the hypotheses."""
from . import backward, forward  # noqa: F401  (registration)
from .registry import BACKWARD, FORWARD, push, table, update

__all__ = ["push", "update", "table", "FORWARD", "BACKWARD"]
