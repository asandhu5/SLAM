"""Seed the local history with one real, short SLAM run so the dashboard has
something to explore immediately, without waiting through a full run first.

This actually runs the real pipeline (on a short slice of whichever dataset
is available) rather than faking results -- it's just done once, up front,
the same way the other rebuilt projects in this portfolio seed one real
example instead of a synthetic placeholder.

Safe to re-run: skips seeding if a demo run already exists.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.runner import run_slam  # noqa: E402
from backend.store import RunMeta, RunStore, new_id, now_iso  # noqa: E402

logger = logging.getLogger(__name__)

DEMO_FRAMES = 80


def seed_if_needed(store: RunStore) -> int:
    if any(r.get("source") == "demo" for r in store.list_runs()):
        return 0

    if config.tum_available:
        dataset_kind, dataset_path, label = "tum", str(config.tum_dir), "TUM fr1/desk"
    elif config.kitti_sequences:
        seq = config.kitti_sequences[0]
        dataset_kind = "kitti"
        dataset_path = str(config.kitti_dir / "sequences" / seq)
        label = f"KITTI seq{seq}"
    else:
        logger.info("No datasets available to seed a demo run from.")
        return 0

    run_id = new_id()
    meta = RunMeta(
        id=run_id,
        dataset_kind=dataset_kind,
        dataset_label=label,
        max_frames=DEMO_FRAMES,
        created_at=now_iso(),
        status="running",
        source="demo",
    )
    store.create(meta)
    logger.info("Seeding demo run: %s frames of %s (this runs the real pipeline once)...", DEMO_FRAMES, label)
    run_slam(store, run_id, dataset_kind, dataset_path, DEMO_FRAMES)
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _store = RunStore(config.runs_dir)
    count = seed_if_needed(_store)
    print(f"Seeded {count} demo run(s) into {_store.dir}")
