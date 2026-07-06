"""Hugging Face Spaces entrypoint — delegates to app.main."""

from app.main import _warm_up, build_ui

_warm_up()
demo = build_ui()

if __name__ == "__main__":
    demo.launch()
