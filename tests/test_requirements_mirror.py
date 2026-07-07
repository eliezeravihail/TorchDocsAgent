"""requirements.txt must delegate to pyproject, not duplicate the dep list.

HF Spaces installs from requirements.txt while everything else installs from
pyproject. Keeping a second hand-maintained list here is what once silently
dropped google-genai and broke the live Space. The fix is a single source of
truth: requirements.txt contains only "." (install this project), so pip pulls
the deps straight from pyproject. This test fails if a package list creeps back.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _requirement_lines() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


def test_requirements_delegates_to_pyproject():
    lines = _requirement_lines()
    extra = lines - {".", "-e ."}
    assert lines, "requirements.txt must install the project (expected '.')"
    assert not extra, (
        "requirements.txt must only install this project ('.') so pyproject.toml "
        f"stays the single source of dependency truth; found hardcoded deps: {sorted(extra)}"
    )
