"""Compatibility package for running from repo root.

The project sources live under `src/`. This package makes `import core...`
work without requiring users to manually set PYTHONPATH.

It extends this package's module search path to include `src/core`.
"""

from __future__ import annotations

import os
from pkgutil import extend_path

# Allow namespace-style extension of this package.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_src_core = os.path.join(_repo_root, "src", "core")

if os.path.isdir(_src_core) and _src_core not in __path__:
    __path__.append(_src_core)  # type: ignore[attr-defined]
