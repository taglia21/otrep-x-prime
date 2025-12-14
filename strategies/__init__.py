"""Compatibility package for running from repo root.

Extends this package's module search path to include `src/strategies`.
"""

from __future__ import annotations

import os
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_src_pkg = os.path.join(_repo_root, "src", "strategies")

if os.path.isdir(_src_pkg) and _src_pkg not in __path__:
    __path__.append(_src_pkg)  # type: ignore[attr-defined]
