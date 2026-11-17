"""Every ```python block of README.md runs (plan 026 P1): the README and the package cannot drift apart.

Each block is executed as written, in a fresh namespace, with a non-interactive plotting backend. A block that
demonstrates a refusal catches it itself (try / except), so it shows the reader what happens and still runs.
"""
import warnings
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def blocks():
    out, heading, inside, buf = [], "top", False, []
    for line in README.read_text().splitlines():
        if not inside and line.startswith("## "):
            heading = line[3:].split("(")[0].strip()
        elif not inside and line.strip() == "```python":
            inside, buf = True, []
        elif inside and line.strip() == "```":
            inside = False
            out.append(pytest.param("\n".join(buf), id=f"{len(out)}-{heading[:40]}"))
        elif inside:
            buf.append(line)
    return out


def test_the_readme_has_examples():
    assert len(blocks()) >= 10


@pytest.mark.parametrize("code", blocks())
def test_readme_example_runs(code):
    import matplotlib
    matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})  # noqa: S102  (running it is the test)
