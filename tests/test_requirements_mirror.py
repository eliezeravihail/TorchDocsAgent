"""requirements.txt must mirror pyproject runtime deps exactly.

HF Spaces installs from requirements.txt while CI installs from pyproject, so
a package present in one but not the other passes CI yet breaks the live Space
at runtime (this is exactly how the gemini fallback lost google-genai and
crashed with "cannot import name 'genai' from 'google'"). Fail loudly here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_requirements() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


def _read_pyproject_deps() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return set(data["project"]["dependencies"])


def test_requirements_mirror_pyproject_dependencies():
    reqs = _read_requirements()
    deps = _read_pyproject_deps()
    missing = deps - reqs
    extra = reqs - deps
    assert not missing, f"requirements.txt is missing pyproject deps: {sorted(missing)}"
    assert not extra, f"requirements.txt has deps not in pyproject: {sorted(extra)}"
