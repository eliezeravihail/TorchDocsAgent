"""Hugging Face Spaces entrypoint — delegates to app.main."""

from app.main import _warm_up, build_ui, serve

_warm_up()
demo = build_ui()

if __name__ == "__main__":
    serve(demo)
