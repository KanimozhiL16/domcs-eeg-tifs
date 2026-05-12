const output = document.getElementById("output");
const evidenceLog = document.getElementById("evidence-log");
const systemStatus = document.getElementById("system-status");
const databaseDetails = document.getElementById("database-details");
const enrollSubject = document.getElementById("enroll-subject");
const impostorSubject = document.getElementById("impostor-subject");
const subjectDetails = document.getElementById("subject-details");

let subjects = [];
let enrolledUserId = null;

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function setBusy(button, busy) {
  button.disabled = busy;
  button.dataset.originalText = button.dataset.originalText || button.textContent;
  button.textContent = busy ? "Running..." : button.dataset.originalText;
}

function renderOutput(data) {
  output.textContent = JSON.stringify(data, null, 2);
  output.className = "";
  if (data.decision === "accept" || data.verified === true) {
    output.classList.add("accept");
  } else if (data.decision === "reject" || data.verified === false || data.error) {
    output.classList.add("reject");
  }
}

function appendEvidence(title, data) {
  const row = document.createElement("div");
  row.className = "evidence-row";
  const decision = data.decision ? `Decision: ${data.decision.toUpperCase()}` : data.status || "Recorded";
  const score = typeof data.score === "number" ? ` | Score: ${data.score.toFixed(4)}` : "";
  const subject = data.evidence?.subject_id || data.probe_subject_id || "";
  const windows = data.evidence?.num_windows || data.num_windows || data.num_probe_windows || "";
  row.innerHTML = `
    <strong>${title}</strong>
    <span>${decision}${score}</span>
    <small>Subject ${subject || "-"} | windows ${windows || "-"} | user ${data.user_id || data.claimed_user_id || "-"}</small>
  `;
  if (evidenceLog.textContent === "Awaiting action...") {
    evidenceLog.textContent = "";
  }
  evidenceLog.prepend(row);
}

function renderDatabase(meta) {
  const loaded = meta.exists && !meta.error;
  systemStatus.textContent = loaded
    ? `Loaded ${meta.num_subjects} subjects | device=${meta.device} | threshold=${meta.threshold}`
    : `Database not ready: ${meta.error || "file missing"}`;
  systemStatus.className = loaded ? "status ok" : "status bad";

  databaseDetails.innerHTML = `
    <dt>NPZ path</dt><dd>${meta.path || "-"}</dd>
    <dt>Subjects</dt><dd>${meta.num_subjects || 0}</dd>
    <dt>Windows</dt><dd>${meta.num_windows || "-"}</dd>
    <dt>Shape</dt><dd>${meta.shape ? meta.shape.join(" x ") : "-"}</dd>
    <dt>Sampling rate</dt><dd>${meta.fs || "-"} Hz</dd>
    <dt>Channels</dt><dd>${meta.channel_count || "-"}</dd>
    <dt>Enrollment runs</dt><dd>${(meta.enrollment_runs || []).join(", ")}</dd>
    <dt>Verification runs</dt><dd>${(meta.verification_runs || []).join(", ")}</dd>
    <dt>Checkpoint</dt><dd>${meta.checkpoint_exists ? "loaded" : "missing"}</dd>
  `;
}

function fillSubjectSelects(rows) {
  subjects = rows || [];
  const options = subjects.map((subject) => {
    const label = `Subject ${subject.subject_id} - enroll ${subject.enrollment_windows}, verify ${subject.verification_windows}`;
    return `<option value="${subject.subject_id}" ${subject.ready ? "" : "disabled"}>${label}</option>`;
  }).join("");
  enrollSubject.innerHTML = options;
  impostorSubject.innerHTML = options;
  if (subjects.length > 1) {
    impostorSubject.selectedIndex = 1;
  }
  renderSubjectDetails();
}

function renderSubjectDetails() {
  const selected = subjects.find((subject) => subject.subject_id === enrollSubject.value);
  if (!selected) {
    subjectDetails.textContent = "No subject selected.";
    return;
  }
  subjectDetails.innerHTML = `
    <div><strong>Subject ${selected.subject_id}</strong></div>
    <div>Total windows: ${selected.total_windows}</div>
    <div>Enrollment windows R01-R02: ${selected.enrollment_windows}</div>
    <div>Verification windows R03-R14: ${selected.verification_windows}</div>
    <div>Runs found: ${selected.runs.join(", ")}</div>
  `;
}

async function loadDatabase() {
  try {
    const meta = await fetchJson("/api/demo/database");
    renderDatabase(meta);
    fillSubjectSelects(meta.subjects || []);
    renderOutput({ database: "ready", subjects: meta.num_subjects, path: meta.path });
  } catch (error) {
    renderOutput({ error: error.message });
    systemStatus.textContent = error.message;
    systemStatus.className = "status bad";
  }
}

async function postDemo(url, payload, button, title) {
  setBusy(button, true);
  try {
    const data = await fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderOutput(data);
    appendEvidence(title, data);
    return data;
  } catch (error) {
    const data = { error: error.message };
    renderOutput(data);
    appendEvidence(`${title} failed`, data);
    return data;
  } finally {
    setBusy(button, false);
  }
}

document.getElementById("reload-database").addEventListener("click", loadDatabase);
enrollSubject.addEventListener("change", () => {
  enrolledUserId = null;
  renderSubjectDetails();
});

document.getElementById("enroll-button").addEventListener("click", async (event) => {
  const subjectId = enrollSubject.value;
  enrolledUserId = `subject_${subjectId}`;
  await postDemo(
    "/api/demo/enroll",
    { subject_id: subjectId, user_id: enrolledUserId },
    event.currentTarget,
    `Enrolled subject ${subjectId}`
  );
});

document.getElementById("genuine-button").addEventListener("click", async (event) => {
  const subjectId = enrollSubject.value;
  const userId = enrolledUserId || `subject_${subjectId}`;
  await postDemo(
    "/api/demo/verify",
    { claimed_user_id: userId, probe_subject_id: subjectId },
    event.currentTarget,
    `Genuine verification for subject ${subjectId}`
  );
});

document.getElementById("impostor-button").addEventListener("click", async (event) => {
  const subjectId = enrollSubject.value;
  const probeId = impostorSubject.value;
  const userId = enrolledUserId || `subject_${subjectId}`;
  await postDemo(
    "/api/demo/verify",
    { claimed_user_id: userId, probe_subject_id: probeId },
    event.currentTarget,
    `Impostor verification: subject ${probeId} against ${subjectId}`
  );
});

document.getElementById("reset-demo").addEventListener("click", async (event) => {
  await postDemo("/api/demo/reset", {}, event.currentTarget, "Reset enrollment store");
  enrolledUserId = null;
});

loadDatabase();
