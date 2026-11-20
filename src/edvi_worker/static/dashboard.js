"use strict";
const $ = id => document.getElementById(id);
let token = "", timer, refreshing = false;
function message(text, error = false) { $("message").textContent = text; $("message").classList.toggle("error", error); }
async function api(path, body) {
  const response = await fetch(path, {method: body ? "POST" : "GET", headers: {Authorization: `Bearer ${token}`, "Content-Type": "application/json"}, ...(body ? {body: JSON.stringify(body)} : {})});
  const result = await response.json();
  if (!response.ok) { if (response.status === 401) disconnect(); throw new Error(result.error?.message || "Coordinator request failed"); }
  return result;
}
function element(tag, text, className) { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; }
function disconnect() { token = ""; clearInterval(timer); $("content").hidden = true; $("login").hidden = false; $("disconnect").hidden = true; $("workers").replaceChildren(); $("jobs").replaceChildren(); }
async function refresh() {
  if (refreshing || !token) return;
  refreshing = true;
  try {
    const [workers, jobs] = await Promise.all([api("/v1/workers"), api("/v1/jobs")]);
    if (!token) return;
    $("workers").replaceChildren(); $("models").replaceChildren();
    $("worker-count").textContent = `${workers.workers.filter(w => w.status === "idle").length} online`;
    const models = new Set();
    for (const worker of workers.workers) {
      const card = element("div", "", "card"), caps = worker.capabilities;
      card.append(element("strong", caps.hostname + " "), element("span", worker.status === "idle" ? "online" : "offline", `badge ${worker.status}`));
      card.append(element("div", `${caps.memory_gb} GB · ${caps.logical_cpus} CPUs · ${caps.accelerator}`, "muted"));
      card.append(element("div", `Last seen: ${new Date(worker.last_seen).toLocaleString()}`, "muted"));
      card.append(element("div", worker.worker_id, "muted"));
      card.append(element("div", `Models: ${(caps.mlx_models || []).join(", ") || "none"}`, "muted"));
      (caps.mlx_models || []).forEach(model => models.add(model));
      $("workers").append(card);
    }
    models.forEach(model => { const option = document.createElement("option"); option.value = model; $("models").append(option); });
    if (!workers.workers.length) $("workers").append(element("p", "No workers yet. Start a worker to connect it."));
    $("jobs").replaceChildren();
    for (const job of [...jobs.jobs].reverse()) {
      const card = element("div", "", "card");
      card.append(element("span", job.status, `badge ${job.status}`), element("div", job.request.prompt));
      card.append(element("div", `${job.request.model} · ${job.job_id}`, "muted"));
      if (job.worker_id) card.append(element("div", `Worker: ${job.worker_id}`, "muted"));
      if (job.result) card.append(element("div", job.result.text || job.result.error, "output"));
      $("jobs").append(card);
    }
    if (!jobs.jobs.length) $("jobs").append(element("p", "Your first prompt will appear here."));
  } catch (error) { message(error.message, true); } finally { refreshing = false; }
}
$("login-form").addEventListener("submit", async event => {
  event.preventDefault();
  token = $("token").value;
  try { await api("/v1/workers"); $("token").value = ""; $("login").hidden = true; $("content").hidden = false; $("disconnect").hidden = false; message("Connected."); await refresh(); clearInterval(timer); timer = setInterval(refresh, 3000); }
  catch (error) { token = ""; message(error.message, true); }
});
$("disconnect").addEventListener("click", () => { disconnect(); message("Disconnected."); });
$("job-form").addEventListener("submit", async event => {
  event.preventDefault(); $("submit-job").disabled = true;
  try { const job = await api("/v1/jobs", {model: $("model").value, prompt: $("prompt").value, max_tokens: Number($("max-tokens").value), temperature: Number($("temperature").value), timeout_seconds: Number($("timeout").value)}); message(`Job queued: ${job.job_id}`); await refresh(); }
  catch (error) { message(error.message, true); } finally { $("submit-job").disabled = false; }
});
