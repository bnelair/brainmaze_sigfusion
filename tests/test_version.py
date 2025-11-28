import importlib


def test_version_importable():
    """Basic smoke test: package imports and exposes __version__."""
    mod = importlib.import_module("brainmaze_sigcoreg")
    assert hasattr(mod, "__version__"), "package must expose __version__"
    assert isinstance(mod.__version__, str) and len(mod.__version__) > 0
    print("package version:", mod.__version__)

