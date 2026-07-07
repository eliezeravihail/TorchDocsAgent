"""requirements.txt must equal the list generated from pyproject.

pyproject [project.dependencies] is the single source of truth; requirements.txt
is a generated mirror (HF Spaces installs it and needs the explicit list — a
bare "." fails the Space build with BUILD_ERROR). This fails CI if someone edits
pyproject without regenerating, so the two can't drift and the deploy can't
silently lose a package (which is exactly how the Space lost google-genai).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def test_requirements_txt_is_generated_from_pyproject():
    from gen_requirements import render

    current = (ROOT / "requirements.txt").read_text()
    assert current == render(), (
        "requirements.txt is out of sync with pyproject.toml dependencies — "
        "run `python scripts/gen_requirements.py` and commit the result."
    )
