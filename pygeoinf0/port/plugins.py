"""Plugins: the rules by which external implementations dock onto the port (plan 019 P3; experiments/EXP-034-external-
implementations/DESIGN.md section 4, the author's answers of 2026-10-30).

A plugin is an ordinary Python distribution beside pygeoinf0 (in this workspace: plugins/<name>, its own Git
repository). It declares a Plugin object and makes it known in either of two ways; pygeoinf0 itself never imports one:

    explicit       the plugin's package calls pg.port.plugins.register(PLUGIN) when imported, or the user calls
                   pg.port.plugins.load("name"), which imports the entry point of that name
    listing        an entry point in the group "pygeoinf0.plugins" (name = plugin name, value = "module:PLUGIN") lets
                   pg.port.plugins.available() list installed plugins from their metadata, without importing any of them

    Plugin(name, version, port=">=1,<2", objects={...}, claims=(...), requires=(...), licence=, source=, examples=)
    pg.port.plugins.register(plugin)   check the port range, stamp provenance on the objects, cap the claims; refuse
                                  core names and a different plugin under a registered name
    pg.port.plugins.load(name)         import the entry point and register it (a registered plugin is returned as it is)
    pg.port.plugins.available()        [(name, value, distribution, version)] from metadata only
    pg.port.plugins.loaded()           the registered plugins
    pg.port.plugins.check(plugin)      the contract kit: Taylor rates and adjoint identities of the plugin's examples, E P = I
                                  of its coordinates, claims within the allowed statuses (checked claims)

What the core guarantees: the port (pygeoinf0.port) changes compatibly within a major PORT version; a plugin's objects
are ordinary pygeoinf0 objects, so every method and cell takes them; their claims carry the plugin's name, version and
source. What it requires: the claims a plugin states itself are asserted or checked (theorem and derived are the core's,
from its own constructions; stronger statuses are downgraded to asserted, saying so); no core name is replaced.
"""
import importlib
import importlib.metadata
from dataclasses import dataclass, field

import numpy as np

from ..core.claims import ASSERTED, CHECKED, ON_RECORD, Claim, Claims
from ..core.errors import Refused

PORT = 1
GROUP = "pygeoinf0.plugins"
ALLOWED = (CHECKED, ASSERTED)
_REGISTRY = {}


@dataclass
class Plugin:
    name: str
    version: str
    port: str = ">=1,<2"
    objects: dict = field(default_factory=dict)     # public objects: name -> object (maps, coordinates, constructors)
    claims: tuple = ()                              # Claim objects the plugin states about its objects
    requires: tuple = ()                            # names of the external distributions it uses
    licence: str = ""
    source: str = ""                                # where it comes from: URL and commit
    examples: object = None                         # () -> [(map, theta)]: what check() runs
    versions: dict = field(default_factory=dict)    # filled by register: the versions of `requires` found

    @property
    def label(self):
        return f"{self.name} {self.version}"

    def __repr__(self):
        return f"Plugin({self.label}, port {self.port}, provides {sorted(self.objects)})"


def _supports(spec, port=PORT):
    """Whether the port version is in a spec such as ">=1,<2" (integer bounds)."""
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
           "==": lambda a, b: a == b}
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        op = next((o for o in (">=", "<=", "==", ">", "<") if part.startswith(o)), None)
        if op is None:
            raise ValueError(f"cannot read the port range {spec!r} (use e.g. '>=1,<2')")
        if not ops[op](port, int(part[len(op):])):
            return False
    return True


def _core_names():
    import pygeoinf0
    return set(pygeoinf0.__all__)


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def register(plugin):
    """Make a plugin known (module docstring); returns it."""
    if not isinstance(plugin, Plugin):
        raise TypeError("register takes a pg.port.plugins.Plugin")
    if not _supports(plugin.port):
        raise Refused(f"plugin {plugin.label} supports port {plugin.port}; this pygeoinf0 has port {PORT}",
                      needs=(f"a version of {plugin.name} for port {PORT}",))
    old = _REGISTRY.get(plugin.name)
    if old is not None:
        if old is plugin:
            return plugin
        raise Refused(f"a different plugin is registered as {plugin.name!r} ({old.label}, {old.source or 'no source'})")
    clash = sorted(set(plugin.objects) & _core_names())
    if clash:
        raise Refused(f"plugin {plugin.label} would replace core names {clash}", needs=("other names",))
    plugin.versions = {d: _version(d) for d in plugin.requires}
    capped = []
    for c in plugin.claims:
        if c.status in ALLOWED or c.status not in ON_RECORD:
            capped.append(c)
        else:
            capped.append(Claim(c.statement, ASSERTED, f"{c.reason} [stated by plugin {plugin.label} as {c.status}; "
                                f"a plugin's own claims are asserted or checked]", key=c.key, holds=c.holds))
    plugin.claims = tuple(capped)
    where = f"plugin {plugin.label}" + (f" ({plugin.source})" if plugin.source else "")
    for name, obj in plugin.objects.items():
        target = getattr(obj, "_claims", None)
        if isinstance(target, list):
            target.append(Claim(f"provided by {where}", ASSERTED, f"pg.port.plugins.register as {name!r}",
                                key="provenance"))
        try:
            obj.provenance = where
        except (AttributeError, TypeError):
            pass
    _REGISTRY[plugin.name] = plugin
    return plugin


def _entry_points():
    return importlib.metadata.entry_points(group=GROUP)


def available():
    """[(name, value, distribution, version)] of the installed plugins, from metadata only (nothing is imported)."""
    out = []
    for ep in _entry_points():
        dist = getattr(ep, "dist", None)
        out.append((ep.name, ep.value, getattr(dist, "name", None), getattr(dist, "version", None)))
    return sorted(out)


def load(name):
    """The registered plugin of that name, importing its entry point first if needed."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    eps = [ep for ep in _entry_points() if ep.name == name]
    if not eps:
        listed = [a[0] for a in available()]
        raise Refused(f"no plugin named {name!r} is installed" + (f" (installed: {listed})" if listed else ""),
                      needs=("pip install the plugin, or import its package (it registers itself)",))
    obj = eps[0].load()
    plugin = obj if isinstance(obj, Plugin) else getattr(obj, "PLUGIN", None)
    if not isinstance(plugin, Plugin):
        raise Refused(f"the entry point {eps[0].value!r} of {name!r} is not a pg.port.plugins.Plugin")
    return register(plugin)


def loaded():
    return dict(_REGISTRY)


def check(plugin):
    """The contract kit (module docstring): Claims, each checked, with what failed named."""
    from .coordinates import Coordinates
    from .taylor import taylor_test
    out = []
    bad = [c for c in plugin.claims if c.status not in ALLOWED and c.status in ON_RECORD]
    out.append(Claim("the plugin's own claims are asserted or checked", CHECKED,
                     f"{len(plugin.claims)} claims" + (f"; too strong: {[c.statement for c in bad]}" if bad else ""),
                     key="claims", holds=not bad))
    for name, obj in plugin.objects.items():
        if isinstance(obj, Coordinates):
            x = np.random.default_rng(0).standard_normal(obj.dim)
            gap = float(np.max(np.abs(obj.numbers(obj.element(x)) - x)) / max(np.max(np.abs(x)), 1e-300))
            out.append(Claim(f"{name}: E P = I", CHECKED, f"relative gap {gap:.1e}", key="E P = I", holds=gap < 1e-8))
    for k, (G, theta) in enumerate(plugin.examples() if plugin.examples else ()):
        if not hasattr(G, "derivative"):
            continue
        res = taylor_test(G, theta, directions=3)
        for c in res.claims:
            out.append(Claim(f"example {k} ({getattr(G, 'name', 'G')}): {c.statement}", c.status, c.reason, key=c.key,
                             holds=c.holds))
    return Claims(out)
