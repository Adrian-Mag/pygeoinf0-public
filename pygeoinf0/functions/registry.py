"""The registry of the bank: family -> builder, domain kinds, summary, references, examples (plan 021; DESIGN.md 3, 5).

    @register(name, kinds, summary, references=(), examples=None)
    def builder(D, **params) -> BankFunction

examples(D) returns parameter dicts for the contract test and the catalogue (default: [{}], the builder's defaults).
A name is registered once; the kinds say on which domains the builder works (geometry.kind).
"""
from dataclasses import dataclass, field

from . import geometry as geo

KINDS = ("interval", "box", "disk", "ball", "shell", "sphere", "circle", "cylinder", "torus", "curve")
_ENTRIES = {}


@dataclass(frozen=True)
class Entry:
    name: str
    kinds: tuple
    summary: str
    builder: object
    references: tuple = ()
    examples: object = None
    module: str = field(default="")

    def example_params(self, D):
        return list(self.examples(D)) if self.examples else [{}]


def register(name, kinds, summary, references=(), examples=None):
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown domain kinds {sorted(unknown)} (known: {KINDS})")

    def deco(builder):
        if name in _ENTRIES:
            raise ValueError(f"a family named {name!r} is registered already (in {_ENTRIES[name].module})")

        def checked(D, **params):
            k = geo.kind(D)
            if k not in kinds:
                from ..core.errors import Refused
                raise Refused(f"{name} is not defined on a {k} (it is on {', '.join(kinds)})")
            return builder(D, **params)
        checked.__name__, checked.__doc__ = builder.__name__, builder.__doc__
        _ENTRIES[name] = Entry(name, tuple(kinds), summary, checked, tuple(references), examples, builder.__module__)
        return checked
    return deco


def entries():
    return dict(_ENTRIES)


def on_kind(k):
    return {n: e for n, e in _ENTRIES.items() if k in e.kinds}
