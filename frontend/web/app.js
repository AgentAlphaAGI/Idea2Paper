const statusBadge = document.getElementById("statusBadge");
const runBtn = document.getElementById("runBtn");
const downloadBtn = document.getElementById("downloadBtn");
const ideaInput = document.getElementById("ideaInput");
const apiKeyInput = document.getElementById("apiKey");
const apiUrlInput = document.getElementById("apiUrl");
const modelInput = document.getElementById("model");
const toggleNovelty = document.getElementById("toggleNovelty");
const toggleVerification = document.getElementById("toggleVerification");
const stageName = document.getElementById("stageName");
const stageDetail = document.getElementById("stageDetail");
const progressBar = document.getElementById("progressBar");
const substageEvent = document.getElementById("substageEvent");
const idleInfo = document.getElementById("idleInfo");
const activityInfo = document.getElementById("activityInfo");
const finalStoryEl = document.getElementById("finalStory");
const summaryEl = document.getElementById("summary");

let uiRunId = null;
let pollTimer = null;
let lastResultFetched = false;

function setBadge(status) {
  statusBadge.className = "badge";
  if (status === "running" || status === "starting") {
    statusBadge.classList.add("badge-running");
    statusBadge.textContent = "Running";
  } else if (status === "done") {
    statusBadge.classList.add("badge-done");
    statusBadge.textContent = "Done";
  } else if (status === "failed") {
    statusBadge.classList.add("badge-failed");
    statusBadge.textContent = "Failed";
  } else {
    statusBadge.classList.add("badge-idle");
    statusBadge.textContent = "Idle";
  }
}

function setStage(stage) {
  stageName.textContent = stage?.name || "Idle";
  stageDetail.textContent = stage?.detail || "Waiting for a run…";
  const progress = Math.round((stage?.progress || 0) * 100);
  progressBar.style.width = `${progress}%`;

  substageEvent.textContent = stage?.substage_event_type || "—";

  if (stage?.idle_seconds == null) {
    idleInfo.textContent = "—";
  } else {
    const s = Math.floor(stage.idle_seconds);
    const m = Math.floor(s / 60);
    const r = s % 60;
    idleInfo.textContent = `${m}m ${r}s`;
  }

  const act = stage?.activity || {};
  const pills = [];
  if (act.llm_active) pills.push('<span class="pill">LLM active</span>');
  if (act.embedding_active) pills.push('<span class="pill">Embedding active</span>');
  activityInfo.innerHTML = pills.length ? pills.join(" ") : '<span class="pill gray">No recent activity</span>';
}

function safeText(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

function renderSummary(summary) {
  if (!summary) {
    summaryEl.textContent = "—";
    return;
  }
  const lines = [
    `Success: ${safeText(summary.success)}`,
    `Avg Score: ${safeText(summary.avg_score)}`,
    `Verification: collision=${safeText(summary.verification?.collision_detected)} max_similarity=${safeText(summary.verification?.max_similarity)}`,
    `Novelty: risk=${safeText(summary.novelty?.risk_level)} max_similarity=${safeText(summary.novelty?.max_similarity)}`,
  ];
  summaryEl.textContent = lines.join("\n");
}

function renderFinalStory(story) {
  if (!story) {
    finalStoryEl.textContent = "No result yet.";
    return;
  }
  const sections = [];
  const mapping = [
    ["title", "Title"],
    ["abstract", "Abstract"],
    ["problem_framing", "Problem Framing"],
    ["gap_pattern", "Gap Pattern"],
    ["solution", "Solution"],
    ["method", "Method"],
    ["claims", "Claims"],
  ];
  const lowerKeys = {};
  Object.keys(story).forEach((k) => {
    lowerKeys[k.toLowerCase()] = k;
  });
  for (const [key, label] of mapping) {
    const k = lowerKeys[key];
    if (!k) continue;
    const value = story[k];
    if (Array.isArray(value)) {
      sections.push(`${label}:\n- ${value.join("\n- ")}`);
    } else {
      sections.push(`${label}:\n${value}`);
    }
  }
  if (sections.length === 0) {
    finalStoryEl.textContent = JSON.stringify(story, null, 2);
  } else {
    finalStoryEl.textContent = sections.join("\n\n");
  }
}

async function startRun() {
  const idea = ideaInput.value.trim();
  if (!idea) {
    alert("Please enter a research idea.");
    return;
  }
  runBtn.disabled = true;
  downloadBtn.disabled = true;
  lastResultFetched = false;

  const payload = {
    idea,
    llm: {
      api_key: apiKeyInput.value || "",
      api_url: apiUrlInput.value || "",
      model: modelInput.value || "",
    },
    toggles: {
      novelty: !!toggleNovelty.checked,
      verification: !!toggleVerification.checked,
    },
  };

  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "Failed to start run");
    }
    uiRunId = data.ui_run_id;
    setBadge("running");
    setStage({ name: "Starting", progress: 0.05, detail: "Launching pipeline…" });
    poll();
  } catch (err) {
    alert(`Failed to start: ${err.message}`);
    runBtn.disabled = false;
  }
}

async function poll() {
  if (!uiRunId) return;
  clearTimeout(pollTimer);
  try {
    const res = await fetch(`/api/runs/${uiRunId}`);
    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "Run not found");
    }
    setBadge(data.status);
    setStage(data.stage);
    if (data.run_id) {
      downloadBtn.disabled = false;
    }
    if ((data.status === "done" || data.status === "failed") && !lastResultFetched) {
      await fetchResult();
    }
    if (data.status === "running" || data.status === "starting") {
      pollTimer = setTimeout(poll, 1500);
    } else {
      runBtn.disabled = false;
    }
  } catch (err) {
    stageDetail.textContent = `Error: ${err.message}`;
    pollTimer = setTimeout(poll, 2000);
  }
}

async function fetchResult() {
  if (!uiRunId) return;
  try {
    const res = await fetch(`/api/runs/${uiRunId}/result`);
    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "No result");
    }
    renderFinalStory(data.final_story);
    renderSummary(data.summary);
    lastResultFetched = true;
  } catch (err) {
    finalStoryEl.textContent = `Result not available yet: ${err.message}`;
  }
}

function downloadLogs() {
  if (!uiRunId) return;
  window.location.href = `/api/runs/${uiRunId}/logs.zip`;
}

runBtn.addEventListener("click", startRun);
downloadBtn.addEventListener("click", downloadLogs);

// Initialize
setBadge("idle");
setStage({ name: "Idle", progress: 0, detail: "Waiting for a run…" });
