const output = document.getElementById("output");
const usersList = document.getElementById("users-list");
const healthStatus = document.getElementById("health-status");

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function renderOutput(data) {
  output.textContent = JSON.stringify(data, null, 2);
  if (data.decision === "accept" || data.verified === true) {
    output.className = "accept";
  } else if (data.decision === "reject" || data.verified === false || data.error) {
    output.className = "reject";
  } else {
    output.className = "";
  }
}

async function submitForm(formId, endpoint, onSuccess) {
  const form = document.getElementById(formId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const data = await fetchJson(endpoint, {
        method: "POST",
        body: new FormData(form),
      });
      renderOutput(data);
      if (onSuccess) {
        await onSuccess(data);
      }
    } catch (error) {
      renderOutput({ error: error.message });
    }
  });
}

async function loadUsers() {
  try {
    const data = await fetchJson("/api/users");
    if (!data.users.length) {
      usersList.textContent = "No users enrolled.";
      return;
    }
    usersList.innerHTML = data.users.map((user) => `
      <div class="user-chip">
        <strong>${user.user_id}</strong><br>
        windows: ${user.num_windows} | dim: ${user.embedding_dim}
      </div>
    `).join("");
  } catch (error) {
    usersList.textContent = error.message;
  }
}

async function loadHealth() {
  try {
    const data = await fetchJson("/api/health");
    healthStatus.textContent = `device=${data.device} | threshold=${data.threshold} | min_windows=${data.min_windows}`;
  } catch (error) {
    healthStatus.textContent = error.message;
  }
}

document.getElementById("refresh-users").addEventListener("click", loadUsers);

submitForm("enroll-form", "/api/enroll", loadUsers);
submitForm("verify-form", "/api/verify");
submitForm("identify-form", "/api/identify");
submitForm("threshold-form", "/api/settings/threshold", loadHealth);
submitForm("inspect-master-form", "/api/master/inspect");
submitForm("master-enroll-form", "/api/master/enroll", loadUsers);
submitForm("master-verify-form", "/api/master/verify");
submitForm("edf-enroll-form", "/api/edf/enroll", loadUsers);
submitForm("edf-verify-form", "/api/edf/verify");

loadUsers();
loadHealth();
