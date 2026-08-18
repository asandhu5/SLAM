(() => {
  const form = document.getElementById("setup-form");
  const datasetGrid = document.getElementById("dataset-grid");
  const errorBox = document.getElementById("form-error");
  const startBtn = document.getElementById("start-btn");
  const maxFramesInput = document.getElementById("max_frames");
  const maxFramesValue = document.getElementById("max_frames_value");

  maxFramesInput.addEventListener("input", () => {
    maxFramesValue.textContent = maxFramesInput.value;
  });

  datasetGrid.addEventListener("change", () => {
    const checked = datasetGrid.querySelector("input[name=dataset_kind]:checked");
    datasetGrid.querySelectorAll(".option-card").forEach((card) => {
      const cardInput = card.querySelector("input");
      card.classList.toggle("selected", cardInput === checked);
    });
  });

  function showError(message) {
    errorBox.textContent = message;
    errorBox.style.display = "block";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.style.display = "none";

    const checked = datasetGrid.querySelector("input[name=dataset_kind]:checked");
    if (!checked) {
      showError("Choose a dataset.");
      return;
    }

    const payload = {
      dataset_kind: checked.value,
      sequence: checked.dataset.sequence || "",
      max_frames: maxFramesInput.value,
    };

    startBtn.disabled = true;
    startBtn.textContent = "STARTING…";

    try {
      const resp = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "Could not start the run.");
      }
      window.location.href = `/run/${data.run_id}`;
    } catch (err) {
      showError(err.message);
      startBtn.disabled = false;
      startBtn.textContent = "▶ Start SLAM Run";
    }
  });
})();
