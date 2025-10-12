"""Pytest configuration for the Snowflake ML Template project."""

import sys
from pathlib import Path
from typing import List, Union


# Ensure 'src' (project source) is on sys.path so tests can import package modules.
# Use parent directory of this file (repo root) to find `src` reliably.
def _add_src_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists():
        import os

        src_str = str(src_dir)
        # Remove any entries that point to a directory which contains a competing
        # 'snowflake_ml_template' package (for example the repo root or its parent)
        cleaned: List[str] = []
        for p in sys.path:
            if not p:
                continue
            try:
                candidate = os.path.abspath(os.path.join(p, "snowflake_ml_template"))
            except OSError:
                # if path operations fail, skip this entry
                continue
            if os.path.isdir(candidate) and os.path.abspath(
                candidate
            ) != os.path.abspath(src_str):
                # skip this entry to avoid shadowing
                continue
            cleaned.append(p)

        # Ensure src is not duplicated and put it at index 0
        cleaned = [p for p in cleaned if p != src_str]
        cleaned.insert(0, src_str)
        sys.path[:] = cleaned


_add_src_to_syspath()


def pytest_ignore_collect(collection_path: Union[str, Path], config) -> bool:
    """Ignore test collection under `src/snowflake_ml_template/`.

    This avoids requiring heavy workflow dependencies (Flyte) when running unit tests.
    """
    path = Path(str(collection_path))
    # If the path is inside the package and contains workflows, skip collection.
    if "snowflake_ml_template" in path.parts and "workflows" in path.parts:
        return True
    return False


# Additionally, explicitly ignore any test files under workflows to be safe for glob collectors.
collect_ignore_glob = [
    "src/snowflake_ml_template/workflows/*_test.py",
]
