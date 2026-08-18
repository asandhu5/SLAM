"""SLAM Console -- a web dashboard for running and inspecting this project's
monocular visual SLAM pipeline against real datasets.

Run with:
    python app.py

See README.md for dataset setup.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, redirect, render_template, request, url_for

from backend.config import config
from backend.runner import run_slam
from backend.store import RunMeta, RunStore, new_id, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("slam_console")

app = Flask(__name__)
store = RunStore(config.runs_dir)

# The SLAM pipeline is CPU-bound and, more importantly, run_slam() redirects
# the *process-wide* sys.stdout to capture the pipeline's print()s into a
# live log tail for the browser. That's only safe with one run in flight at
# a time -- a second concurrent run would interleave into the same
# redirected stream and could restore the wrong original stdout when it
# finished. A single worker turns "two runs at once" into "the second one
# waits," which is a perfectly reasonable behavior for a local CPU-bound
# tool anyway.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slam-worker")

MAX_FRAMES_CAP = 2000


@app.route("/")
def index():
    has_demo = any(r.get("source") == "demo" for r in store.list_runs())
    return render_template(
        "setup.html",
        config=config,
        has_demo=has_demo,
        active_page="setup",
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True, silent=True) or {}
    dataset_kind = payload.get("dataset_kind")
    sequence = (payload.get("sequence") or "").strip()

    if dataset_kind == "tum":
        if not config.tum_available:
            return jsonify({"error": "TUM dataset not found under data/tum/. See README.md."}), 400
        dataset_path = str(config.tum_dir)
        label = "TUM fr1/desk"
    elif dataset_kind == "kitti":
        if sequence not in config.kitti_sequences:
            return jsonify({"error": "Unknown or unavailable KITTI sequence."}), 400
        dataset_path = str(config.kitti_dir / "sequences" / sequence)
        label = f"KITTI seq{sequence}"
    else:
        return jsonify({"error": "Choose a dataset."}), 400

    try:
        max_frames = int(payload.get("max_frames") or 300)
    except (TypeError, ValueError):
        max_frames = 300
    max_frames = max(10, min(max_frames, MAX_FRAMES_CAP))

    run_id = new_id()
    meta = RunMeta(
        id=run_id,
        dataset_kind=dataset_kind,
        dataset_label=label,
        max_frames=max_frames,
        created_at=now_iso(),
        status="running",
        source="live",
    )
    store.create(meta)
    executor.submit(run_slam, store, run_id, dataset_kind, dataset_path, max_frames)

    return jsonify({"run_id": run_id})


@app.route("/run/<run_id>")
def run_page(run_id):
    meta = store.get_meta(run_id)
    if not meta:
        return redirect(url_for("index"))
    if meta["status"] == "ready":
        return redirect(url_for("results", run_id=run_id))
    return render_template("run.html", run=meta, active_page="setup")


@app.route("/api/status/<run_id>")
def api_status(run_id):
    meta = store.get_meta(run_id)
    if not meta:
        return jsonify({"error": "Unknown run"}), 404
    return jsonify(meta)


@app.route("/results/<run_id>")
def results(run_id):
    meta = store.get_meta(run_id)
    if not meta:
        return redirect(url_for("index"))
    if meta["status"] != "ready":
        return redirect(url_for("run_page", run_id=run_id))
    run_results = store.load_results(run_id) or {}
    return render_template("results.html", run=meta, results=run_results, active_page="results")


@app.route("/history")
def history():
    return render_template("history.html", runs=store.list_runs(), active_page="history")


@app.route("/demo")
def demo():
    demo_runs = [r for r in store.list_runs() if r.get("source") == "demo"]
    if not demo_runs:
        return redirect(url_for("index"))
    return redirect(url_for("results", run_id=demo_runs[0]["id"]))


def _ensure_demo_seeded() -> None:
    from scripts.seed_demo_data import seed_if_needed
    try:
        seeded = seed_if_needed(store)
        if seeded:
            logger.info("Seeded %d demo run(s)", seeded)
    except Exception:
        logger.exception("Demo data seeding skipped due to an error")


if __name__ == "__main__":
    _ensure_demo_seeded()

    if not config.tum_available and not config.kitti_sequences:
        logger.warning("No datasets found under data/. See README.md for setup.")

    logger.info("SLAM Console running at http://%s:%s", config.host, config.port)
    app.run(host=config.host, port=config.port, debug=False, threaded=True)
