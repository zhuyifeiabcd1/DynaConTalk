// Thin JSON client for the studio backend.
async function request(method, url, { body, form, signal } = {}) {
  const init = { method, headers: {}, signal };
  if (form) init.body = form;
  else if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  const res = await fetch(url, init);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { data = { detail: text }; }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || res.statusText || `HTTP ${res.status}`;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  config: () => request("GET", "/api/config"),
  system: () => request("GET", "/api/system"),
  assets: (kind) => request("GET", `/api/assets/${kind}`),
  patchAsset: (kind, id, patch) => request("PATCH", `/api/assets/${kind}/${encodeURIComponent(id)}`, { body: patch }),
  trajectoryPath: (id) => request("GET", `/api/assets/trajectories/${encodeURIComponent(id)}/path`),
  trajPreview: (base, script) => request("POST", "/api/traj/preview", { body: { base, script } }),
  jobs: () => request("GET", "/api/jobs"),
  jobsVersion: () => request("GET", "/api/jobs/version"),
  job: (id) => request("GET", `/api/jobs/${id}`),
  createJob: (form) => request("POST", "/api/jobs", { form }),
  editJob: (id, body) => request("POST", `/api/jobs/${id}/edit`, { body }),
  patchJob: (id, patch) => request("PATCH", `/api/jobs/${id}`, { body: patch }),
  cancelJob: (id) => request("POST", `/api/jobs/${id}/cancel`),
  deleteJob: (id) => request("DELETE", `/api/jobs/${id}?purge=1`),
  words: (id) => request("GET", `/api/jobs/${id}/words`),
  transcribe: (id, body = {}) => request("POST", `/api/jobs/${id}/transcribe`, { body }),
  logs: (id, offset = 0) => request("GET", `/api/jobs/${id}/logs?offset=${offset}`),
  agentConfig: () => request("GET", "/api/agent/config"),
  saveAgentConfig: (patch) => request("PUT", "/api/agent/config", { body: patch }),
  agentSuggest: (id, body) => request("POST", `/api/jobs/${id}/agent`, { body }),
  events: (id) => new EventSource(`/api/jobs/${id}/events`),
};
