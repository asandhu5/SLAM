"""Flask route tests. The real SLAM pipeline is mocked out here -- it's
exercised for real by scripts/seed_demo_data.py and the manual end-to-end
runs described in README.md, not by the automated suite, which stays fast
and network/dataset-independent.
"""
import dataclasses
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
from backend.store import RunMeta, RunStore, now_iso  # noqa: E402


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    monkeypatch.setattr(app_module, "store", store)
    return store


@pytest.fixture
def ready_run(temp_store):
    meta = RunMeta(
        id="testrun1",
        dataset_kind="tum",
        dataset_label="TUM fr1/desk (fixture)",
        max_frames=80,
        created_at=now_iso(),
        status="ready",
        source="demo",
        processed_frames=80,
        final_landmarks=1234,
        final_keyframes=42,
        duration_seconds=12.3,
        has_ground_truth=True,
        ATE_rmse=0.118,
        ATE_mean=0.1,
        RPE_rmse=0.012,
        RPE_mean=0.011,
        scale=0.07,
    )
    temp_store.create(meta)
    temp_store.save_results(
        "testrun1",
        {
            "trajectory": [[0, 0, 0], [1, 0, 0.5]],
            "trajectory_raw": [[0, 0, 0], [12, 0, 6]],
            "ground_truth": [[0, 0, 0], [1.1, 0, 0.4]],
            "map_points": [[0.5, 0.1, 1.0], [0.2, -0.1, 2.0]],
            "landmark_history": [10, 50, 1234],
            "keyframe_history": [1, 5, 42],
            "state_timeline": ["INITIALIZING", "TRACKING", "TRACKING"],
        },
    )
    return "testrun1"


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"SLAM Console" in resp.data


def test_history_page_loads_empty(client, temp_store):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"No runs yet" in resp.data


def test_unknown_run_redirects_to_index(client, temp_store):
    assert client.get("/results/does-not-exist").status_code == 302
    assert client.get("/run/does-not-exist").status_code == 302


def test_results_page_renders_ready_run(client, ready_run):
    resp = client.get(f"/results/{ready_run}")
    assert resp.status_code == 200
    assert b"chart-trajectory" in resp.data
    assert b"0.118" in resp.data  # ATE rmse rendered


def test_results_redirects_to_run_page_when_not_ready(client, temp_store):
    temp_store.create(
        RunMeta(id="stillgoing", dataset_kind="tum", dataset_label="x", max_frames=10, created_at=now_iso(), status="running")
    )
    resp = client.get("/results/stillgoing")
    assert resp.status_code == 302
    assert "/run/stillgoing" in resp.headers["Location"]


def test_history_lists_run(client, ready_run):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"TUM fr1/desk (fixture)" in resp.data


def test_demo_route_redirects_to_demo_run(client, ready_run):
    resp = client.get("/demo")
    assert resp.status_code == 302
    assert f"/results/{ready_run}" in resp.headers["Location"]


def test_api_status_returns_meta(client, ready_run):
    resp = client.get(f"/api/status/{ready_run}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ready"
    assert data["final_landmarks"] == 1234


def test_api_run_rejects_missing_dataset(client, temp_store):
    resp = client.post("/api/run", json={})
    assert resp.status_code == 400


def test_api_run_rejects_unavailable_tum(client, temp_store, monkeypatch):
    fake_config = dataclasses.replace(app_module.config, tum_dir=Path("/nonexistent"))
    monkeypatch.setattr(app_module, "config", fake_config)
    resp = client.post("/api/run", json={"dataset_kind": "tum"})
    assert resp.status_code == 400
    assert "TUM" in resp.get_json()["error"]


def test_api_run_starts_a_run_with_mocked_pipeline(client, temp_store, monkeypatch):
    captured = {}

    def fake_run_slam(store, run_id, dataset_kind, dataset_path, max_frames):
        captured["args"] = (dataset_kind, dataset_path, max_frames)
        store.update(run_id, status="ready", processed_frames=max_frames)

    monkeypatch.setattr(app_module, "run_slam", fake_run_slam)
    fake_config = dataclasses.replace(app_module.config, tum_dir=Path(__file__).resolve().parent)
    # tum_available checks for rgb.txt under tum_dir; point it somewhere real
    # enough that the availability check passes for this test's purposes.
    (Path(__file__).resolve().parent / "rgb.txt").touch()
    monkeypatch.setattr(app_module, "config", fake_config)

    try:
        resp = client.post("/api/run", json={"dataset_kind": "tum", "max_frames": 50})
        assert resp.status_code == 200
        run_id = resp.get_json()["run_id"]
        assert temp_store.get_meta(run_id) is not None
    finally:
        (Path(__file__).resolve().parent / "rgb.txt").unlink(missing_ok=True)
