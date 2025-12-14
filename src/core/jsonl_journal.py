from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JsonlJournal:
    path: Path
    rotate_bytes: int = 10_000_000

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        self._rotate_if_needed()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
            f.write("\n")

    def _rotate_if_needed(self) -> None:
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.rotate_bytes:
            return

        rotated = self.path.with_suffix(self.path.suffix + ".1")
        if rotated.exists():
            rotated.unlink()
        self.path.rename(rotated)
