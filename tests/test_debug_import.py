"""Tests that help debug import resolution during CI and local runs."""

import sys


def test_debug_imports():
    """Print sys.path and ensure package is importable."""
    for p in sys.path:
        print("  ", p)
    try:
        import importlib

        pkg = importlib.import_module("snowflake_ml_template")
        print("top package file:", getattr(pkg, "__file__", None))
        print("top package path:", getattr(pkg, "__path__", None))
        print(
            "available submodules:",
            [m.name for m in __import__("pkgutil").iter_modules(pkg.__path__)],
        )
        m = importlib.import_module("snowflake_ml_template.snowpark")
        print("import OK, module file:", getattr(m, "__file__", None))
    except Exception as e:
        print("import failed:", repr(e))
        raise
