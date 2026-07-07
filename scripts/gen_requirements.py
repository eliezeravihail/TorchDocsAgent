"""Regenerate requirements.txt from pyproject.toml [project.dependencies].

pyproject is the single source of dependency truth. HF Spaces installs from
requirements.txt and needs an EXPLICIT list — a bare "." (pip install .) makes
the Space build fail with BUILD_ERROR — so requirements.txt is a generated
mirror, not a hand-maintained second list. Regenerate after editing pyproject:

    python scripts/gen_requirements.py

CI (test_requirements_mirror) fails if the committed file is stale, so the two
can never drift and the deploy can't silently lose a package.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = [
    "# GENERATED from pyproject.toml [project.dependencies] — do not edit by hand.",
    "# pyproject is the single source of truth; regenerate with:",
    "#   python scripts/gen_requirements.py",
    "# HF Spaces installs from this file and needs the explicit list (a bare '.'",
    "# fails the Space build with BUILD_ERROR).",
]


def render() -> str:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    return "\n".join(HEADER + list(deps)) + "\n"


if __name__ == "__main__":
    (ROOT / "requirements.txt").write_text(render())
    print("requirements.txt regenerated from pyproject.toml")
