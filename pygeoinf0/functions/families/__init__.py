"""The families of the bank, one module per family group (plan 021; EXP-037 DESIGN.md 5, 6). Every module here is
imported when pygeoinf0.functions is, and registers its families with functions.registry.register."""
import importlib
import pkgutil

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    importlib.import_module(f"{__name__}.{_m}")
