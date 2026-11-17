"""Reading the snapshot: ``how`` (the commands for one pair, a row or a column), ``table`` and ``plot`` (see the package
docstring)."""
import re

from .probing import snapshot
from .registry import (CAVEATS, COLOUR, CONSTANT, DOMAIN_ALIASES, DOMAINS, GLYPH, MEANING, SPACES, STATUSES, SYMBOL,
                       _resolve)


# ------------------------------------------------------------------------------------------------- how
class Recipe:
    """The answer of ``pg.catalogue.how``: ``code`` (runnable), ``status``, ``reason`` (the package's own message for a
    refusal), ``note``, ``domain``, ``space``; ``str(recipe)`` is the text printed. For a whole row or column the
    domain or the space is None, and so is the status."""

    def __init__(self, domain, space, cell, text, code, note):
        self.domain, self.space = domain, space
        self.status, self.reason = cell["status"], cell["reason"]
        self.cell, self.text, self.code, self.note = cell, text, code, note

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Recipe({self.domain} x {self.space}: {self.status})"

    def run(self):
        """Execute the code; returns its namespace (a refusal raises the package's exception)."""
        ns = {}
        exec(self.code, ns)
        return ns


def _note(sk, cell, snap=None, dk=None):
    parts = [CAVEATS[sk]]
    if cell["status"] == "constructed":
        parts.append("Tried: " + "; ".join(f"{t['functional']} ({t['result']})" for t in cell["tried"]) + ".")
    elif cell["status"] == "ok" and cell["functional"] not in (None, "constant"):
        parts.append(f"Smoke test: the Riesz representer of the {cell['functional']} has a finite norm.")
    if cell.get("definition") and sk == "Sobolev":
        parts.append(f"Realised as: {cell['definition']}.")
    if cell.get("realisation"):
        parts.append(f"Realisation here: {cell['realisation']!r}. That is the numerics, not another space; "
                     "realisation= on the constructor asks for another, and one not available is refused with the list.")
    if sk == "L2" and dk == "Box":
        parts.append("Cell averages on a partition: realisation=pg.realisations.Cells(...).")
    if sk == "Sobolev" and cell["status"] == "ok" and cell.get("definition") == "restriction":
        fn = (snap or {}).get("extras", {}).get("formula_norms", [])
        got = [e["domain"] for e in fn if e["status"] == "ok"]
        no = [e["domain"] for e in fn if e["status"] != "ok"]
        if got and no:
            parts.append(f"The norm of an element given by a formula was computed on {', '.join(got)} and refused "
                         f"on {', '.join(no)} (checked: pg.catalogue.snapshot()['extras']['formula_norms']).")
    return " ".join(parts)


def _space_steps(code):
    """The lines that build H from D: a form with its boundary conditions gets a line of its own."""
    m = re.fullmatch(r"pg\.Sobolev\((pg\.SobolevForm\(.*\))\.operator\(\), (.+)\)", code)
    if m:
        return [f"form = {m.group(1)}", f"H = pg.Sobolev(form.operator(), {m.group(2)})"]
    m = re.fullmatch(r"(pg\.SobolevForm\(.*\))\.space\(\)", code)
    if m:
        return [f"form = {m.group(1)}", "H = form.space()"]
    return [f"H = {code}"]


def _imports(lines):
    return ["import pygeoinf0 as pg"] + (["import numpy as np"] if any("np." in s for s in lines) else [])


def _cell_line(cell):
    return f"H = {cell['space_code']}" if cell["space_code"] else "(no recipe)"


def _row(dk, snap):
    """Everything on one domain: the domain's command, then one line per column."""
    label = next(d[1] for d in DOMAINS if d[0] == dk)
    row = snap["cells"][dk]
    make = f"D = {next(iter(row.values()))['domain_code']}"
    code = "\n".join(_imports([make]) + [make]) + "\n"
    lines = [f"# {label}: every space of the table on it", "# 1. the domain, first:"] + _imports([make]) + [make,
             "# 2. then a space on D (the commands for one: pg.catalogue.how(domain, space)):"]
    names = {s[0]: s[1] for s in SPACES}
    for sk, cell in row.items():
        made = _cell_line(cell) if cell["status"] != "n/a" else ""
        lines.append(f"#   {names[sk]:<11}{cell['status']:<12}{made}".rstrip())
    return Recipe(label, None, {"status": None, "reason": None}, "\n".join(lines), code, "")


def _column(sk, snap):
    """One space on every domain: its status and the domain's command."""
    label = next(s[1] for s in SPACES if s[0] == sk)
    lines = [f"# {label}: on every domain of the table",
             "# 1. the domain, first (its status for this space, its command):"]
    make = None
    for d in snap["domains"]:
        cell = snap["cells"][d["key"]][sk]
        if make is None and cell["space_code"]:
            make = _cell_line(cell)
        lines.append(f"#   {d['label']:<18}{cell['status']:<12}D = {cell['domain_code']}")
    lines += [f"# 2. then the space on D: {make}", "# The exact commands for one: pg.catalogue.how(domain, space)."]
    return Recipe(None, label, {"status": None, "reason": None}, "\n".join(lines), "", CAVEATS[sk])


USAGE = ('pg.catalogue.how("Box", "L2") gives the commands for one entry of the table, in order: the domain, then the '
         'space on it.\npg.catalogue.how("Box") lists every space on a domain; pg.catalogue.how(space="L2") every domain for '
         'a space.')


def how(domain=None, space=None, show=True, result=None):
    """The exact commands that make the pair (domain, space), in order, its status, and for a refusal the package's
    reason and what does work on that domain.

        pg.catalogue.how("Box", "Spec Mixed")      # one entry of the table
        pg.catalogue.how("Box")                    # a row: every space on that domain
        pg.catalogue.how(space="Spec Mixed")       # a column: every domain for that space

    A domain comes first and a space is built on it: the commands are numbered in that order. Names are forgiving
    ("box", "spec mixed", "neumann", "h1", "torus 3d"). Returns a ``Recipe``; ``show=False`` does not print. The text comes from
    the snapshot (or ``result``); nothing is computed."""
    snap = snapshot() if result is None else result
    if domain is None and space is None:
        if show:
            print(USAGE)
        return USAGE
    if domain is not None:
        dk = _resolve(domain, [d for d in DOMAINS if d[0] in snap["cells"]], "domain", DOMAIN_ALIASES)
    if domain is None or space is None:
        if domain is None:
            sk = _resolve(space, [s for s in SPACES if s[0] in next(iter(snap["cells"].values()))], "space")
        rec = _row(dk, snap) if space is None else _column(sk, snap)
        if show:
            print(rec.text)
        return rec
    sk = _resolve(space, [s for s in SPACES if s[0] in snap["cells"][dk]], "space")
    cell = snap["cells"][dk][sk]
    dlabel = next(d[1] for d in DOMAINS if d[0] == dk)
    slabel = next(s[1] for s in SPACES if s[0] == sk)
    functionals = dict(next(d[4] for d in DOMAINS if d[0] == dk))
    lines = [f"# {dlabel} x {slabel}: {cell['status']}"]
    code = ""
    if cell["status"] == "n/a":
        lines.append(f"# not applicable: {cell['reason']}")
    else:
        body = ["# 1. the domain", f"D = {cell['domain_code']}", "# 2. the space on D",
                *_space_steps(cell["space_code"])]
        if cell["status"] == "ok":
            body.append("# 3. try it")
            if cell["functional"] == "constant":
                body.append(f"{CONSTANT[sk]}   # a constant's norm")
            else:
                body += [f"F = H.riesz({functionals[cell['functional']]})   # representer of: {cell['functional']}",
                         "H.norm(F)"]
        use = _imports(body) + body
        if cell["status"] == "refused":
            lines += ["# refused by the package; the commands below raise:", "# " + cell["reason"]]
            names = {s[0]: s[1] for s in SPACES}
            works = [names[k] for k, c in snap["cells"][dk].items() if c["status"] in ("ok", "constructed")]
            lines.append(f"# works on this domain: {', '.join(works)} (pg.catalogue.how({dlabel!r}) lists them)")
        if cell["status"] == "error":
            lines.append("# unexpected failure: " + str(cell["reason"]))
        lines += use
        code = "\n".join(use) + "\n"
    note = _note(sk, cell, snap, dk)
    lines.append("# note: " + note)
    rec = Recipe(dlabel, slabel, cell, "\n".join(lines), code, note)
    if show:
        print(rec.text)
    return rec


# ------------------------------------------------------------------------------------------------- table
def _grid(result):
    doms = result["domains"]
    spcs = result["spaces"]
    return doms, spcs, result["cells"]


def table(fmt="text", result=None):
    """The matrix as text (``fmt="text"``, aligned columns) or Markdown (``fmt="markdown"``), with a legend. Read from
    the snapshot, or from ``result`` (a ``probe()`` result)."""
    snap = snapshot() if result is None else result
    doms, spcs, cells = _grid(snap)
    head = ["dim", "domain"] + [s["label"] for s in spcs]
    rows = [[f"{d['dim']}D", d["label"]] + [SYMBOL[cells[d["key"]][s["key"]]["status"]] for s in spcs] for d in doms]
    seen = {c["status"] for r in cells.values() for c in r.values()}
    keys = [k for k in STATUSES if k != "error" or "error" in seen]
    legend = [f"{SYMBOL[k]:>3}  {MEANING[k]}" for k in keys]
    tail = ("Computed by pg.catalogue.probe() (this is its snapshot). pg.catalogue.how(domain, space) gives the commands for "
            "one entry, in order (the domain, then the space); pg.catalogue.how(domain) a whole row.")
    if fmt == "markdown":
        out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out + ["", "Legend:", "", *[f"- `{SYMBOL[k]}`: {MEANING[k]}" for k in keys], "", tail])
    if fmt != "text":
        raise ValueError("fmt is 'text' or 'markdown'")
    width = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    fmt_row = lambda r: "  ".join(c.ljust(w) if i < 2 else c.center(w) for i, (c, w) in enumerate(zip(r, width))).rstrip()  # noqa
    out = [fmt_row(head), "  ".join("-" * w for w in width)]
    last = None
    for r in rows:
        if last is not None and r[0] != last:
            out.append("")
        out.append(fmt_row(r))
        last = r[0]
    return "\n".join(out + ["", "Legend:", *legend, "", tail])


# ------------------------------------------------------------------------------------------------- plot
def plot(ax=None, result=None, annotate=True):
    """The matrix as a coloured grid (matplotlib, imported here). Colour and glyph both carry the status, so it reads in
    greyscale, in light and in dark notebooks. Returns the Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch as LegendPatch, Rectangle

    snap = snapshot() if result is None else result
    doms, spcs, cells = _grid(snap)
    own = ax is None
    ink = "#222222"
    if own:
        fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(spcs), 1.4 + 0.42 * len(doms)), facecolor="white")
        ax.set_facecolor("white")
    else:
        ink = plt.rcParams["text.color"]           # the caller's axes: follow the caller's theme (light or dark)
    nrow = len(doms)
    for i, d in enumerate(doms):
        for j, s in enumerate(spcs):
            st = cells[d["key"]][s["key"]]["status"]
            ax.add_patch(Rectangle((j, nrow - 1 - i), 1, 1, facecolor=COLOUR[st], edgecolor="white", linewidth=1.5))
            if annotate:
                dark = st in ("refused", "error")
                ax.text(j + 0.5, nrow - 1 - i + 0.5, GLYPH[st], ha="center", va="center", fontsize=11,
                        color="white" if dark else "#1b1b1b", fontweight="bold")
    ax.set_xlim(0, len(spcs))
    ax.set_ylim(0, nrow)
    ax.set_xticks([j + 0.5 for j in range(len(spcs))])
    ax.set_xticklabels([s["label"] for s in spcs], rotation=45, ha="left", color=ink)
    ax.xaxis.tick_top()
    ax.set_yticks([nrow - 1 - i + 0.5 for i in range(nrow)])
    ax.set_yticklabels([d["label"] for d in doms], color=ink)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # group by dimension: a heavier line between groups and the dimension at the right edge
    start = 0
    for i in range(1, nrow + 1):
        if i == nrow or doms[i]["dim"] != doms[start]["dim"]:
            if i < nrow:
                ax.plot([0, len(spcs)], [nrow - i, nrow - i], color=ink, linewidth=2.5)
            ax.text(len(spcs) + 0.12, nrow - (start + i) / 2, f"{doms[start]['dim']}D", rotation=-90, va="center",
                    ha="left", fontsize=10, fontweight="bold", color=ink)
            start = i
    present = [k for k in STATUSES if any(cells[d["key"]][s["key"]]["status"] == k for d in doms for s in spcs)]
    ax.legend(handles=[LegendPatch(facecolor=COLOUR[k], edgecolor="none", label=f"{GLYPH[k]}  {MEANING[k]}")
                       for k in present], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False,
              fontsize=8.5, labelcolor=ink, title_fontsize=8.5,
              title="pg.catalogue.how(domain, space): the commands for one square")
    ax.set_aspect("auto")
    return ax

