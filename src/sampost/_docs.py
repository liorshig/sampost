"""Documentation build helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    """Build the HTML documentation with the project's uv environment."""
    root = Path(__file__).parents[2]
    subprocess.run(
        ["sphinx-build", "-W", "-b", "html", str(root / "docs"), str(root / "docs" / "_build" / "html")],
        check=True,
    )
