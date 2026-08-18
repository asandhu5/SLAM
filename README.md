# Monocular Visual SLAM

A from-scratch monocular visual SLAM pipeline (ORB features → essential-matrix
initialization → PnP tracking → continuous local mapping → sparse bundle
adjustment → g2o pose-graph loop closure) with a real web dashboard for
running it against actual datasets and inspecting the result.

<p>
  <img alt="status" src="https://img.shields.io/badge/status-working-34d399">
  <img alt="python" src="https://img.shields.io/badge/python-3.11-22d3ee">
</p>

## Pipeline

```text
video frames
    |
    v
ORB feature extraction (grid-balanced, CLAHE)
    |
    v
Hamming matching + Lowe ratio + cross-check
    |
    v
Essential matrix + RANSAC  ->  two-view initialization
    |
    v
PnP tracking against the local map (recent keyframes' landmarks)
    |
    v
Keyframe insertion  ->  CONTINUOUS triangulation of new landmarks
    |
    v
Sparse local bundle adjustment (SciPy, sparse Jacobian)
    |
    v
Loop detection  ->  g2o pose-graph optimization on confirmed loops
    |
    v
sparse point cloud + camera trajectory + real ATE/RPE vs ground truth
```

## Quickstart

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:5100**. The first run seeds one real (not
synthetic) 80-frame run against whichever dataset is available, so you can
explore the dashboard immediately — click **"View Sample Run Instead"**.

## Getting datasets

### TUM fr1/desk (recommended first — ~600MB, has ground truth)

```bash
python download_data.py --dataset tum
```

### KITTI odometry (large — the full grayscale archive is ~23GB; ground truth poses are a separate ~1.3MB download)

```bash
python download_data.py --dataset kitti
```

Both datasets, once downloaded, are auto-detected by the app (`data/tum/rgbd_dataset_freiburg1_desk/`, `data/kitti/dataset/sequences/<NN>/`) — no configuration needed. The setup page only shows sequences it actually finds calibration + images for.

## Running tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

94 tests, fully offline (synthetic geometry + tiny generated dataset fixtures — no real TUM/KITTI download needed to run the suite), in under 2 seconds. `.venv/bin/pytest`'s own shebang references a stale path from before this project was moved on disk; `python -m pytest` sidesteps that rather than needing the venv recreated.

## Real, measured results (not fabricated)

Run with `scripts/seed_demo_data.py` / the dashboard against the actual datasets:

| Dataset | Frames | ATE RMSE | RPE RMSE | Notes |
|---|---:|---:|---:|---|
| TUM fr1/desk | 80 | **0.119 m** | 0.012 m | Indoor, real ground truth, no loop closure triggered in this short a run |
| KITTI seq00 | 200 | **36.2 m** | 1.82 m | Outdoor driving, real ground truth, monocular scale drift with no loop closure opportunity this early in the sequence |

That KITTI number looks bad next to TUM's — and it should, honestly: 200 frames of real highway/city driving covers real distance, this is monocular VO with no stereo/IMU/GPS and no loop closure correction this early in the sequence, and drift compounds every frame. That gap between "indoor, ground truth every few centimeters" and "outdoor, tens of meters of drift" is itself the honest story monocular SLAM's own limitations tell — not a bug to paper over. The previous version of this README's benchmark table was explicitly a placeholder; this one is measured.

## What changed from the original version

The math in the individual modules (ORB extraction, essential-matrix
geometry, PnP, triangulation, Umeyama alignment) was already correct and
already well-tested (73 synthetic-geometry tests, still passing). What
didn't work was the *system* — running it against real data surfaced two
concrete, reproducible failures, both fixed:

- **The map never grew past initialization.** `add_landmark()` was only ever called during the first two-frame init; every subsequent keyframe reused the same handful of original landmarks. Confirmed on real KITTI footage: by frame 9, enough of those original 184 points had left the camera's view that PnP couldn't find a stable pose at all. Fixed: every keyframe insertion now triangulates fresh landmarks against the previous keyframe (skipping keypoints already tied to an existing landmark), the same way `_initialize()` already did for the first pair — a 300-frame KITTI run now grows past 50,000 landmarks while tracking stays locked the entire time.
- **Relocalization was permanently unreachable.** After losing tracking, the code searched for a recovery candidate using `LoopDetector.detect_loop()` — which deliberately excludes any keyframe within 30 frames, specifically to avoid mistaking a *loop closure* for a trivial neighbor. Recovering from having *just* lost tracking needs exactly the opposite: the ability to match against recent keyframes. Confirmed on real data: after losing tracking at frame 9, the system stayed stuck in RELOCALIZATION through frame 149 (the full test length) because none of its 9 keyframes were ever more than 9 frames apart. Fixed: losing tracking now retries direct map-based PnP every frame instead, which is both simpler and searches the thing that can actually help.
- **Loop closure was a print statement.** A detected loop candidate was logged and discarded; nothing about the trajectory or map changed. Now: confirmed loops feed a real g2o SE3 pose-graph optimization (`pose_graph.py`) — every keyframe pose becomes a vertex, consecutive keyframes get an odometry edge from their current relative pose, the loop candidate adds one more edge, and the whole graph is re-optimized so the correction spreads back across the trajectory instead of only ever landing on the newest pose. (Full joint pose+landmark bundle adjustment through g2o was also attempted for `bundle_adjuster.py`; multiple wiring attempts against this g2o-python build either diverged to nonsensical poses or segfaulted, with no version-specific documentation available to debug further, so that stays on a SciPy path — see that file's module docstring for the honest account.)
- **Local BA was the slowest part of the pipeline once mapping actually worked continuously.** With no sparse Jacobian, `scipy.optimize.least_squares` estimated a dense finite-difference Jacobian for a parameter vector that could reach ~1000 dimensions — a single BA call could take longer than the rest of a frame's entire processing combined. Fixed with a real sparse Jacobian structure (`jac_sparsity`, built from which parameters each residual actually depends on) plus a `max_nfev` sized to the real problem instead of reusing an unrelated config value that, on a dense Jacobian, was already exhausted estimating the *first* Jacobian.
- **PnP tracking matched against the entire map, forever.** Landmarks only ever accumulate, so a long run made every single frame's candidate matching slower than the last regardless of what was actually nearby. Fixed: tracking now matches against a "local map" — landmarks observed by the most recent N keyframes — the same restriction real SLAM systems apply for the same reason.
- **A triangulated point could be placed kilometers away in a desk-scale scene.** Reprojection-error filtering alone doesn't catch near-parallel rays: two rays that barely converge can still reproject within a couple pixels of where they're supposed to while their actual intersection point is enormously (in one observed case, ~50km) further away than everything else just triangulated. Fixed with a per-batch relative depth-ratio gate in `triangulator.py`.
- **Calibration was one `default_K` for everything**, including datasets it was never calibrated for. `datasets.py` now loads TUM's actual published fr1 intrinsics + lens distortion (with real undistortion applied) and KITTI's actual per-sequence `calib.txt`, and associates TUM's timestamped `groundtruth.txt` to each RGB frame (standard nearest-timestamp TUM association) — there was previously no script anywhere that did this, so `benchmark.py`'s ATE/RPE evaluator had nothing real to run against.
- **The reported "ATE: 0.003m"** in `tests/test_pose_estimator.py`'s `__main__` block was a hardcoded string unconnected to anything the file actually measured (it tests two-view pose recovery, not trajectory error). It now prints the rotation/translation-direction errors the test actually computes; real ATE/RPE numbers come from an actual trajectory run (see the table above).
- The keyframe "tracked ratio" used to divide inlier count by the *entire map's* landmark count — a number that only ever grows, so the ratio became meaningless well before a long sequence finished. Now compares against the *previous keyframe's* own inlier count.
- The dashboard is a genuinely different visual language from this portfolio's other rebuilt projects — sidebar navigation, monospace-first typography, sharp corners, a cyan/amber telemetry palette instead of the rounded indigo/violet SaaS-card look used elsewhere, matching what this project actually is: instrumentation for a robotics pipeline, not a product dashboard.

## Architecture

```
app.py                    Flask routes
backend/
  config.py                 dataset auto-detection, paths
  runner.py                  runs the real pipeline in a background thread,
                              streams live telemetry, computes final ATE/RPE
  store.py                    JSON-file run persistence (no DB needed)
templates/, static/       the web dashboard (sidebar, live run console, results charts)
scripts/seed_demo_data.py   seeds one real short run for first-run exploration

config.py                 all pipeline hyperparameters (single source of truth)
feature_extractor.py      ORB + grid balancing + CLAHE
matcher.py                 Hamming matching, Lowe ratio, cross-check
pose_estimator.py           essential matrix + pose recovery
triangulator.py              two-view 3D point creation + reprojection/depth filtering
tracker.py                    PnP tracking against the local map
keyframe_manager.py            keyframe insertion policy
bundle_adjuster.py               local BA (SciPy, sparse Jacobian)
pose_graph.py                     g2o SE3 pose-graph optimization for loop closure
loop_detector.py                   lightweight ORB appearance loop detection
map.py                               poses, frames, keyframes, landmarks
slam_system.py                        the state machine + CLI entrypoint
visualizer.py                          optional Open3D/matplotlib CLI diagnostics
benchmark.py                            Umeyama alignment, ATE, RPE
datasets.py                              TUM/KITTI loaders: real calibration + ground truth
```

The command-line entrypoint (`python slam_system.py --dataset <folder> [--visualize] [--max-frames N]`) still works standalone, unchanged in spirit — the web dashboard is a new way to drive the same pipeline, not a replacement for it.

## Known limitations

- Monocular scale is arbitrary until aligned to ground truth (Umeyama) or another sensor — expected, not a bug.
- The loop-closure pose-graph correction only adjusts keyframe poses and re-anchors landmarks to their first-observing keyframe; it does not retroactively correct the per-frame `trajectory` samples already recorded for non-keyframe frames (`map.py`'s trajectory list has no frame-id association to do that against without a larger restructuring).
- The loop-closure translation constraint is rescaled by the trajectory's own current estimate of inter-keyframe distance, since essential-matrix recovery has no metric scale — a practical approximation, not true joint scale estimation.
- The loop detector is a lightweight descriptor-matching fallback, not a real bag-of-words vocabulary (DBoW3) — fine at the scale this project runs at, would not scale to a long real deployment.
- Full joint pose+landmark bundle adjustment through g2o remains unintegrated (see bundle_adjuster.py's docstring) — local BA uses SciPy with a real sparse Jacobian instead.
- The KITTI 36m ATE at 200 frames is expected for uncorrected monocular VO over real driving distance, not a defect — see "Real, measured results" above.

## File Guide

See Architecture above — kept once, not duplicated here.
