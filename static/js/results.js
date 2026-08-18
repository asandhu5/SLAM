(() => {
  const results = window.__RESULTS__ || {};
  const root = getComputedStyle(document.documentElement);
  const c = (name) => root.getPropertyValue(name).trim();
  const palette = {
    cyan: c("--cyan") || "#22d3ee",
    amber: c("--amber") || "#f5a524",
    green: c("--green") || "#4ade80",
    red: c("--red") || "#f8555f",
    textDim: c("--text-dim") || "#7d95a0",
    border: c("--border") || "#1f2b33",
  };

  Chart.defaults.color = palette.textDim;
  Chart.defaults.font.family = "JetBrains Mono, monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = palette.border;

  const trajectory = results.trajectory || [];
  const trajectoryRaw = results.trajectory_raw || trajectory;
  const groundTruth = results.ground_truth;
  const mapPoints = results.map_points || [];

  // ---- trajectory: top-down X (world) vs Z (forward) ----------------------
  const trajCtx = document.getElementById("chart-trajectory");
  if (trajCtx) {
    const datasets = [
      {
        label: "Estimated",
        data: trajectory.map((p) => ({ x: p[0], y: p[2] })),
        borderColor: palette.cyan,
        backgroundColor: palette.cyan,
        pointRadius: 0,
        borderWidth: 2,
        showLine: true,
        tension: 0.1,
      },
    ];
    if (groundTruth) {
      datasets.push({
        label: "Ground truth",
        data: groundTruth.map((p) => ({ x: p[0], y: p[2] })),
        borderColor: palette.amber,
        backgroundColor: palette.amber,
        pointRadius: 0,
        borderWidth: 2,
        borderDash: [5, 4],
        showLine: true,
        tension: 0.1,
      });
    }
    new Chart(trajCtx, {
      type: "scatter",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        aspectRatio: 1,
        scales: {
          x: { title: { display: true, text: "x [m]" }, grid: { color: palette.border } },
          y: { title: { display: true, text: "z [m]" }, grid: { color: palette.border } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // ---- sparse map points, top-down -----------------------------------------
  const mapCtx = document.getElementById("chart-map");
  if (mapCtx) {
    new Chart(mapCtx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Landmarks",
            data: mapPoints.map((p) => ({ x: p[0], y: p[2] })),
            backgroundColor: "rgba(34, 211, 238, 0.35)",
            pointRadius: 1.5,
          },
          {
            label: "Trajectory",
            data: trajectoryRaw.map((p) => ({ x: p[0], y: p[2] })),
            borderColor: palette.amber,
            backgroundColor: palette.amber,
            pointRadius: 0,
            borderWidth: 1.5,
            showLine: true,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        aspectRatio: 1,
        scales: {
          x: { title: { display: true, text: "x [m]" }, grid: { color: palette.border } },
          y: { title: { display: true, text: "z [m]" }, grid: { color: palette.border } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // ---- growth line charts -----------------------------------------------------
  function lineChart(canvasId, values, color, yLabel) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    new Chart(ctx, {
      type: "line",
      data: {
        labels: values.map((_, i) => i),
        datasets: [
          {
            data: values,
            borderColor: color,
            backgroundColor: "transparent",
            pointRadius: 0,
            borderWidth: 2,
            tension: 0.15,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "frame" }, grid: { display: false } },
          y: { title: { display: true, text: yLabel }, grid: { color: palette.border }, beginAtZero: true },
        },
      },
    });
  }
  lineChart("chart-growth", results.landmark_history || [], palette.cyan, "landmarks");
  lineChart("chart-keyframes", results.keyframe_history || [], palette.amber, "keyframes");

  // ---- state timeline -----------------------------------------------------------
  const timelineEl = document.getElementById("state-timeline");
  const stateTimeline = results.state_timeline || [];
  if (timelineEl && stateTimeline.length) {
    const stateClass = (s) => {
      const lower = (s || "").toLowerCase();
      if (lower === "tracking") return "tracking";
      if (lower === "initializing") return "initializing";
      return "lost";
    };
    // Merge consecutive identical states into single segments so the DOM
    // stays small even for a long run instead of one div per frame.
    const segments = [];
    for (const state of stateTimeline) {
      const cls = stateClass(state);
      if (segments.length && segments[segments.length - 1].cls === cls) {
        segments[segments.length - 1].count += 1;
      } else {
        segments.push({ cls, count: 1 });
      }
    }
    timelineEl.innerHTML = segments
      .map((seg) => `<div class="seg ${seg.cls}" style="width:${(seg.count / stateTimeline.length) * 100}%"></div>`)
      .join("");
  }
})();
