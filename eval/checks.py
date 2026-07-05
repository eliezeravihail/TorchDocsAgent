"""Static checks on every Answer — no code execution, ever.

Three checks per the plan:
  parses   — every code block in answer_md is valid Python (ast.parse)
  imports  — every import in those blocks is torch-family or stdlib
  symbols  — every symbol in symbols_used actually appears in the answer
"""

from __future__ import annotations

import ast
import re
import sys
import textwrap

from agent.schemas import Answer

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

ALLOWED_IMPORT_ROOTS = frozenset(sys.stdlib_module_names) | {
    "torch",
    "torchvision",
    "torchaudio",
}


def extract_code_blocks(answer_md: str) -> list[str]:
    # models often indent whole blocks (e.g. inside markdown lists) — dedent first
    return [textwrap.dedent(block).strip() for block in CODE_BLOCK_RE.findall(answer_md)]


def check_code_parses(answer: Answer) -> str | None:
    for i, block in enumerate(extract_code_blocks(answer.answer_md)):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            return f"code block {i}: {exc.msg} (line {exc.lineno})"
    return None


def check_imports_allowed(answer: Answer) -> str | None:
    for block in extract_code_blocks(answer.answer_md):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # already reported by check_code_parses
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            for root in roots:
                if root not in ALLOWED_IMPORT_ROOTS:
                    return f"disallowed import: {root}"
    return None


def _symbol_present(symbol: str, answer_md: str) -> bool:
    """True if the symbol appears in any conventional spelling.

    Answers legitimately write `nn.Linear` for torch.nn.Linear, `F.relu`
    for torch.nn.functional.relu, or `.add_(...)` for torch.Tensor.add_ —
    this check verifies bookkeeping consistency, not exact spelling.
    """
    parts = symbol.split(".")
    candidates = {symbol}
    if symbol.startswith("torch."):
        candidates.add(symbol.removeprefix("torch."))
    if len(parts) >= 2:
        candidates.add(".".join(parts[-2:]))
    candidates.add(parts[-1])
    return any(c in answer_md for c in candidates)


def check_symbols_present(answer: Answer) -> str | None:
    missing = [s for s in answer.symbols_used if not _symbol_present(s, answer.answer_md)]
    if missing:
        return f"symbols listed but not in answer: {', '.join(missing)}"
    return None


CHECKS = [
    ("parses", check_code_parses),
    ("imports", check_imports_allowed),
    ("symbols", check_symbols_present),
]


def run_checks(answer: Answer) -> dict[str, str | None]:
    """Run all checks; value is None on pass, else the failure reason."""
    return {name: fn(answer) for name, fn in CHECKS}


def format_table(rows: list[tuple[str, dict[str, str | None]]]) -> str:
    """rows: [(question_id, check_results)] → aligned pass/fail table."""
    names = [name for name, _ in CHECKS]
    header = f"{'id':<8}" + "".join(f"{n:<10}" for n in names)
    lines = [header, "-" * len(header)]
    for qid, results in rows:
        cells = "".join(f"{'ok' if results[n] is None else 'FAIL':<10}" for n in names)
        lines.append(f"{qid:<8}{cells}")
        for n in names:
            if results[n] is not None:
                lines.append(f"         {n}: {results[n]}")
    return "\n".join(lines)
