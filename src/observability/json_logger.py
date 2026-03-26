from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class JsonStageLogger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, level: str, event: str, **payload: Any) -> None:
        row = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **payload,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
