"""Environment/paths configuration for the SLAM Console web app."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    data_dir: Path
    runs_dir: Path
    tum_dir: Path
    kitti_dir: Path

    @property
    def tum_available(self) -> bool:
        return (self.tum_dir / "rgb.txt").exists()

    @property
    def kitti_sequences(self) -> list[str]:
        seq_dir = self.kitti_dir / "sequences"
        if not seq_dir.exists():
            return []
        return sorted(p.name for p in seq_dir.iterdir() if p.is_dir() and (p / "calib.txt").exists())


def _load() -> AppConfig:
    data_dir = BASE_DIR / "data" / "runs"
    return AppConfig(
        host=os.getenv("HOST", "127.0.0.1"),
        port=_int_env("PORT", 5100),
        data_dir=BASE_DIR / "data",
        runs_dir=data_dir,
        tum_dir=BASE_DIR / "data" / "tum" / "rgbd_dataset_freiburg1_desk",
        kitti_dir=BASE_DIR / "data" / "kitti" / "dataset",
    )


config = _load()
config.runs_dir.mkdir(parents=True, exist_ok=True)
