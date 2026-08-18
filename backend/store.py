"""JSON-file-backed persistence for SLAM runs.

Same rationale as the rest of this portfolio's rebuilds: a single-user local
tool doesn't need a database. Each run gets a folder under data/runs/<id>/
holding its results payload; a flat index.json lists them all for history.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()


@dataclass
class RunMeta:
    id: str
    dataset_kind: str  # tum | kitti | demo
    dataset_label: str
    max_frames: int
    created_at: str
    status: str = "running"  # running -> ready | error
    source: str = "live"  # live | demo
    current_frame: int = 0
    total_frames: int = 0
    state: str = "INITIALIZING"
    landmarks: int = 0
    keyframes: int = 0
    log_tail: Optional[list] = None
    duration_seconds: float = 0.0
    processed_frames: int = 0
    final_landmarks: int = 0
    final_keyframes: int = 0
    has_ground_truth: bool = False
    ATE_rmse: Optional[float] = None
    ATE_mean: Optional[float] = None
    RPE_rmse: Optional[float] = None
    RPE_mean: Optional[float] = None
    scale: Optional[float] = None
    error: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class RunStore:
    def __init__(self, runs_dir: Path):
        self.dir = runs_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir.parent / "index.json"

    def _read_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write_index(self, index: dict[str, dict]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2))

    def list_runs(self) -> list[dict]:
        index = self._read_index()
        return sorted(index.values(), key=lambda r: r.get("created_at", ""), reverse=True)

    def create(self, meta: RunMeta) -> None:
        with _LOCK:
            (self.dir / meta.id).mkdir(parents=True, exist_ok=True)
            index = self._read_index()
            index[meta.id] = asdict(meta)
            self._write_index(index)

    def update(self, run_id: str, **fields) -> None:
        with _LOCK:
            index = self._read_index()
            if run_id in index:
                index[run_id].update(fields)
                self._write_index(index)

    def get_meta(self, run_id: str) -> Optional[dict]:
        return self._read_index().get(run_id)

    def save_results(self, run_id: str, results: dict) -> None:
        (self.dir / run_id / "results.json").write_text(json.dumps(results))

    def load_results(self, run_id: str) -> Optional[dict]:
        path = self.dir / run_id / "results.json"
        return json.loads(path.read_text()) if path.exists() else None
