(() => {
  const run = window.__RUN__;
  if (run.status === "error") return;

  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");
  const progressPct = document.getElementById("progress-pct");
  const stateBadge = document.getElementById("state-badge");
  const consoleEl = document.getElementById("console");
  const rLandmarks = document.getElementById("r-landmarks");
  const rKeyframes = document.getElementById("r-keyframes");
  const rFrame = document.getElementById("r-frame");
  const rFrameTotal = document.getElementById("r-frame-total");
  const rState = document.getElementById("r-state");

  let renderedLogCount = 0;

  function applyState(stateName) {
    const cls = "st-" + stateName.toLowerCase();
    stateBadge.className = "led-badge " + cls;
    stateBadge.innerHTML = '<span class="led led-pulse"></span> ' + stateName;
    if (rState) rState.textContent = stateName;
  }

  function renderLog(lines) {
    if (!lines || lines.length === renderedLogCount) return;
    consoleEl.innerHTML = lines
      .map((line, i) => `<div class="log-line"><span class="t">[${String(i).padStart(3, "0")}]</span>${escapeHtml(line)}</div>`)
      .join("");
    consoleEl.scrollTop = consoleEl.scrollHeight;
    renderedLogCount = lines.length;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  async function poll() {
    try {
      const resp = await fetch(run.statusUrl);
      const data = await resp.json();

      if (data.status === "ready") {
        window.location.href = run.resultsUrl;
        return;
      }
      if (data.status === "error") {
        window.location.reload();
        return;
      }

      const total = data.total_frames || 0;
      const pct = total ? Math.min(100, Math.round((data.current_frame / total) * 100)) : 1;
      progressFill.style.width = `${Math.max(pct, 1)}%`;
      progressPct.textContent = `${pct}%`;
      progressLabel.textContent = `Frame ${data.current_frame} / ${total || "?"}`;
      if (rFrame) rFrame.textContent = data.current_frame;
      if (rFrameTotal) rFrameTotal.textContent = `of ${total || "?"}`;
      if (rLandmarks) rLandmarks.textContent = data.landmarks;
      if (rKeyframes) rKeyframes.textContent = data.keyframes;
      applyState(data.state || "INITIALIZING");
      renderLog(data.log_tail);
    } catch (err) {
      console.warn("Status check failed, retrying...", err);
    } finally {
      setTimeout(poll, 1200);
    }
  }

  poll();
})();
