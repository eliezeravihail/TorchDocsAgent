# TorchDocsAgent

AI-powered chat agent for PyTorch's public API (Python and C++/libtorch) — ask questions in natural language and get explanations with real citations into the actual source and docs. This is a personal project and is not official PyTorch team.

The agent never writes or generates code. It explains, references, and quotes existing PyTorch source/docs — every snippet a user sees is a verbatim, cited excerpt of real code or documentation, never something synthesized.

## Goals

- Answer natural-language questions about PyTorch's public API (Python and C++/libtorch), concepts, and usage patterns.
- Ground every answer in the official PyTorch source and documentation, with citations, to reduce hallucination.
- Quote existing code/docs verbatim when illustrating usage — never generate new code.
- Support searching and browsing PyTorch docs conversationally instead of manually digging through pages.
- Stay easy to run locally with minimal setup.

See [PLAN.md](PLAN.md) for the current roadmap and TODO list.
