"""Running the probe: one attempt per (domain, space) pair under a time and memory limit, the extra checks, and the
snapshot that records the result (see the package docstring)."""
import contextlib
import gc
import json
import math
import signal
import threading
import time
import warnings
from pathlib import Path

import numpy as np

from ..spaces.constructions import boundary_parts
from ..settings import settings
from ..core.errors import ProbeTimeout, Refused, UnboundedFunctional, Unsupported
from .registry import (BOUNDARY_SPACES, CONSTANT, DOMAIN_ALIASES, DOMAINS, FORMAT, FORMULAS, FRACTIONAL, FUNCTIONAL_LIMIT,
                       LENGTH_SCALE, SLOW, SNAPSHOT, SOLVE_TEMPLATE, SOLVE_TOLERANCE, SOLVES, SPACES, _resolve,
                       _space_code)


@contextlib.contextmanager
def _limits(seconds, memory_mb):
    """A wall-clock limit (POSIX main thread) and an address-space limit (where ``resource`` exists), both optional."""
    restore = []
    if memory_mb:
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            cap = int(memory_mb) * 2 ** 20
            if hard == resource.RLIM_INFINITY or cap < hard:
                resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
                restore.append(lambda: resource.setrlimit(resource.RLIMIT_AS, (soft, hard)))
        except (ImportError, ValueError, OSError):
            pass
    timed = bool(seconds) and hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if timed:
        def handler(*_):
            raise ProbeTimeout(f"no result within {seconds} s")
        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if timed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        for undo in restore:
            undo()


def _first_line(e):
    return (str(e).split("\n")[0]).strip() or type(e).__name__


def _classify(e):
    """('refused', reason) for the package's own refusals, ('error', reason) for anything else."""
    if isinstance(e, ProbeTimeout):
        return "error", f"time-out: {e}"
    if isinstance(e, (Unsupported, Refused, ValueError, TypeError, NotImplementedError)):
        return "refused", _first_line(e)
    return "error", f"{type(e).__name__}: {_first_line(e)}"


def _namespace(**kw):
    import pygeoinf0 as pg
    return {"pg": pg, "np": np, **kw}


# ------------------------------------------------------------------------------------------------- one attempt
def _definition(H):
    try:
        return getattr(H.backend.geometry, "definition", None)
    except Exception:                                                                     # noqa: BLE001
        return None


def _realisation(H):
    r = getattr(H, "realisation", None)
    return r if isinstance(r, str) else None


def _smoke(H, space, domain, functionals, ns, limit, memory_mb):
    """('ok'|'constructed', functional label or None, tried list). The tried list is the honest record."""
    tried = []
    if space in CONSTANT and domain != "Layered":
        code = CONSTANT[space]
        with _limits(limit, memory_mb):
            v = float(eval(code, dict(ns, H=H)))
        tried.append({"functional": "constant", "result": "bounded" if math.isfinite(v) else "not finite"})
        return ("ok" if math.isfinite(v) else "constructed"), ("constant" if math.isfinite(v) else None), tried
    for label, code in functionals:
        try:
            with _limits(FUNCTIONAL_LIMIT, memory_mb):
                v = float(H.norm(H.riesz(eval(code, dict(ns, H=H)))))
        except UnboundedFunctional:
            tried.append({"functional": label, "result": "unbounded"})
            continue
        except Exception as e:                                                            # noqa: BLE001
            kind, why = _classify(e)
            tried.append({"functional": label, "result": ("refused: " if kind == "refused" else "error: ") + why})
            continue
        if math.isfinite(v):
            tried.append({"functional": label, "result": "bounded"})
            return "ok", label, tried
        tried.append({"functional": label, "result": "not finite"})
    return "constructed", None, tried


def _attempt(domain, space, D, dim, parts, functionals, order, limit, memory_mb):
    """One cell: construct the space and smoke-test it. Returns the cell record (with 'seconds')."""
    dcode = next(d[3] for d in DOMAINS if d[0] == domain)
    na = None
    if space in BOUNDARY_SPACES and not parts:
        na = ("a closed manifold has no boundary: there is no boundary condition to impose "
              "(its spectral space is the Sobolev column)")
    elif space == "SpecMixed" and domain != "Layered" and len(parts) < 2:
        na = f"a mixed condition needs two boundary parts; this domain has {len(parts)} ({', '.join(parts)})"
    if na:
        return dict(status="n/a", reason=na, definition=None, functional=None, tried=[], domain_code=dcode,
                    space_code=None, seconds=0.0)
    code = _space_code(space, domain, dim, parts, order)
    if code is None:                       # a layered ball offers pg.PiecewiseSpace only: show what the plain constructor says
        code = _space_code(space, "plain", dim, parts, order)
    ns = _namespace(D=D)
    t0 = time.perf_counter()
    status, reason, definition, functional, tried, realised = "error", None, None, None, [], None
    try:
        with _limits(limit, memory_mb), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = eval(code, ns)
        definition, realised = _definition(H), _realisation(H)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status, functional, tried = _smoke(H, space, domain, functionals, ns, limit, memory_mb)
    except Exception as e:                                                                # noqa: BLE001
        status, reason = _classify(e)
    gc.collect()
    return dict(status=status, reason=reason, definition=definition, realisation=realised, functional=functional,
                tried=tried, domain_code=dcode, space_code=code, seconds=round(time.perf_counter() - t0, 2))


def _build_domain(code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = eval(code, _namespace())
    try:
        parts = tuple(boundary_parts(D))
    except Exception:                                                                     # noqa: BLE001
        parts = ()
    return D, parts


def _formula_norms(limit, memory_mb):
    out = []
    for domain, formula in FORMULAS:
        key, _, dim, dcode, _f = next(d for d in DOMAINS if d[0] == domain)
        D, parts = _build_domain(dcode)
        scode = _space_code("Sobolev", domain, dim, parts)
        rec = dict(domain=domain, space="Sobolev", space_code=scode, formula=formula)
        try:
            with _limits(limit, memory_mb), warnings.catch_warnings(), settings.using(resolution_policy="warn"):
                warnings.simplefilter("ignore")                  # FORMULAS: fixed polynomials of degree <= 3, pygeoinf0's
                H = eval(scode, _namespace(D=D))                 # own (no feature narrower than the domain)
                v = float(H(eval(formula)).norm())
            rec.update(status="ok" if math.isfinite(v) else "error", reason=None)
        except Exception as e:                                                            # noqa: BLE001
            rec.update(status=_classify(e)[0], reason=_classify(e)[1], error=type(e).__name__)
        out.append(rec)
    return out


def _fractional(limit, memory_mb, wanted=lambda pair: True):
    out, cache = [], {}
    for domain, space, order in FRACTIONAL:
        if not wanted((domain, space, order)):
            continue
        key, _, dim, dcode, functionals = next(d for d in DOMAINS if d[0] == domain)
        if domain not in cache:
            cache[domain] = _build_domain(dcode)
        D, parts = cache[domain]
        cell = _attempt(domain, space, D, dim, parts, functionals, order, limit, memory_mb)
        out.append(dict(domain=domain, space=space, order=order, **cell))
    return out


def _solves(limit, memory_mb):
    out = []
    for name, domain, space, order, rays in SOLVES:
        key, _, dim, dcode, _f = next(d for d in DOMAINS if d[0] == domain)
        D, parts = _build_domain(dcode)
        scode = _space_code(space, domain, dim, parts, order)
        snippet = SOLVE_TEMPLATE.format(rays=rays)
        rec = dict(name=name, domain=domain, space=space, order=order, domain_code=dcode, space_code=scode,
                   code=snippet, tolerance=SOLVE_TOLERANCE)
        ns = _namespace(D=D)
        t0 = time.perf_counter()
        try:
            with _limits(limit, memory_mb), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ns["H"] = eval(scode, ns)
                exec(snippet, ns)
            r = float(ns["residual"])
            rec.update(ok=bool(math.isfinite(r) and r < SOLVE_TOLERANCE), reason=None)
        except Exception as e:                                                            # noqa: BLE001
            rec.update(ok=False, reason=_classify(e)[1])
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        out.append(rec)
        gc.collect()
    return out


# ------------------------------------------------------------------------------------------------- probe
def _resolve_many(items, registry, what):
    if items is None:
        return [r[0] for r in registry]
    if isinstance(items, str):
        items = [items]
    return [_resolve(i, registry, what, DOMAIN_ALIASES if what == "domain" else None) for i in items]


def probe(domains=None, spaces=None, extras=None, limit=90, memory_mb=None, progress=False, speed=None):
    """Try the combinations now and return the result structure (the snapshot has the same shape).

    domains, spaces: names (as ``how`` accepts them) to restrict the attempt; None means all. extras: also run the
    fractional-order and solve checks (default: only when nothing is restricted). limit: seconds per attempt (POSIX main
    thread only); memory_mb: optional address-space cap per attempt. speed="fast" leaves out the pairs in ``SLOW``
    (seconds each: spectral spaces in 3-D, the surface functional of the intrinsic H1 on a box or 3-D torus) and
    speed="slow" tries only those; the two together are the full probe. Each cell has the keys status, reason,
    definition, functional, tried, domain_code, space_code, seconds."""
    dkeys = _resolve_many(domains, DOMAINS, "domain")
    skeys = _resolve_many(spaces, SPACES, "space")
    if speed not in (None, "fast", "slow"):
        raise ValueError("speed is None, 'fast' or 'slow'")
    wanted = lambda pair: speed is None or (speed == "slow") == (pair in SLOW)             # noqa: E731
    cells = {}
    for key, label, dim, dcode, functionals in DOMAINS:
        if key not in dkeys:
            continue
        D, parts = _build_domain(dcode)
        cells[key] = {}
        for sk in skeys:
            if not wanted((key, sk)):
                continue
            cells[key][sk] = _attempt(key, sk, D, dim, parts, functionals, None, limit, memory_mb)
            if progress:
                print(f"{label:18s} {sk:10s} {cells[key][sk]['status']}", flush=True)
        del D
        gc.collect()
    if extras is None:
        extras = domains is None and spaces is None
    out = dict(format=FORMAT, length_scale=LENGTH_SCALE,
               domains=[dict(key=d[0], label=d[1], dim=d[2]) for d in DOMAINS if d[0] in dkeys],
               spaces=[dict(key=s[0], label=s[1]) for s in SPACES if s[0] in skeys], cells=cells)
    if extras:
        quick = speed != "slow"                      # the solves and formula norms are cheap: part of the fast tier
        out["extras"] = dict(fractional=_fractional(limit, memory_mb, wanted),
                             solves=_solves(limit, memory_mb) if quick else [],
                             formula_norms=_formula_norms(limit, memory_mb) if quick else [])
    return out


def stable(result):
    """The result without timings, i.e. what the snapshot records and what a staleness test compares."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "seconds"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return json.loads(json.dumps(clean(result)))


def write_snapshot(result=None, path=None):
    """Probe (or take ``result``) and store it as the shipped snapshot; returns the path."""
    path = Path(path or SNAPSHOT)
    result = stable(probe() if result is None else result)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    _cache.clear()
    return path


_cache = {}


def snapshot():
    """The shipped snapshot, as a dictionary."""
    if "snapshot" not in _cache:
        _cache["snapshot"] = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return _cache["snapshot"]


def differences(fresh=None, shipped=None):
    """A list of strings naming what differs between a fresh probe and the snapshot (empty: the snapshot is current)."""
    fresh = stable(probe() if fresh is None else fresh)
    shipped = snapshot() if shipped is None else shipped
    out = []
    for key in ("format", "length_scale"):
        if fresh.get(key) != shipped.get(key):
            out.append(f"{key} differs")

    def compare(name, a, b):
        if a == b:
            return
        if a is None or b is None:
            out.append(f"{name}: {'missing from the snapshot' if b is None else 'no longer probed'}")
        elif a.get("status") != b.get("status") or a.get("reason") != b.get("reason"):
            out.append(f"{name}: snapshot {b.get('status')} ({b.get('reason')}), now {a.get('status')} ({a.get('reason')})")
        else:
            out.append(f"{name}: recipe or details differ: snapshot {json.dumps(b)} now {json.dumps(a)}")
    for dk, row in fresh["cells"].items():
        for sk, c in row.items():
            compare(f"{dk} x {sk}", c, shipped["cells"].get(dk, {}).get(sk))
    for part, ident in (("fractional", lambda e: (e["domain"], e["space"], e["order"])), ("solves", lambda e: e["name"]),
                       ("formula_norms", lambda e: (e["domain"], e["space"]))):
        old = {ident(e): e for e in shipped.get("extras", {}).get(part, [])}
        for e in fresh.get("extras", {}).get(part, []):
            compare(f"{part} {ident(e)}", e, old.get(ident(e)))
    return out

