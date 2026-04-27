const protocolData = [
  { name: "Random Split", eer: 1.12, auc: 0.9991, note: "Convenient but optimistic baseline" },
  { name: "Same Task", eer: 2.28, auc: 0.9969, note: "Task leakage still present" },
  { name: "B2T (Ours)", eer: 3.75, auc: 0.9928, note: "Realistic baseline-to-task verification" }
];

const galleryItems = [
  {
    key: "roc",
    label: "ROC / DET",
    title: "ROC and DET Under Realistic Verification",
    image: "../figures/FIG_05_ROC_DET_60ep.png",
    description: "This view anchors the biometric story: the separation remains strong even when enrollment and testing occur under different cognitive-task conditions."
  },
  {
    key: "scores",
    label: "Scores",
    title: "Score Distributions Reveal Strong Identity Separation",
    image: "../figures/FIG_06_score_distribution_60ep.png",
    description: "The genuine and impostor distributions remain decisively apart, supporting the high AUC and low EER observed across seeds."
  },
  {
    key: "drift",
    label: "Drift",
    title: "Session Drift Makes the Evaluation Honest",
    image: "../figures/FIG_13_session_drift_60ep.png",
    description: "This figure communicates why cross-session and cross-task realism matters. Performance is reported under the harder, deployment-relevant condition."
  },
  {
    key: "embedding",
    label: "Embeddings",
    title: "Embeddings Stay Structured Across Subjects",
    image: "../figures/FIG_16_tSNE_embeddings_60ep.png",
    description: "Latent identity structure remains separable, showing that the model does not simply memorize a single task-specific pattern."
  },
  {
    key: "interpretability",
    label: "Interpretability",
    title: "Interpretability Makes the Model Defensible",
    image: "../figures/FIG_INTERP_SUMMARY_all_methods.png",
    description: "Channel- and time-level explanations strengthen publication quality and help convert the work into a deployable, inspectable system."
  }
];

const seedResults = [
  ["1", "3.73", "0.9926", "87.03"],
  ["2", "3.98", "0.9925", "86.77"],
  ["3", "3.89", "0.9918", "86.38"],
  ["4", "3.45", "0.9938", "87.15"],
  ["5", "3.70", "0.9932", "87.42"],
  ["Mean", "3.75", "0.9928", "86.95"],
  ["Std", "0.20", "0.0008", "0.40"]
];

const focusContent = {
  science: `
    <strong>Scientific impact:</strong> The interface foregrounds the paper's strongest claim:
    evaluation protocol realism matters. Instead of advertising a flattering random split, it
    shows the more difficult baseline-to-task condition and ties the headline metrics to that protocol.
  `,
  deployment: `
    <strong>Deployment readiness:</strong> The model is light enough for real-time use, the metrics
    are biometric rather than classroom-style classification accuracy, and the interpretability panels
    support auditability for security-sensitive use cases.
  `,
  demo: `
    <strong>Demo narrative:</strong> This can sit next to a live EEG acquisition or prerecorded subject
    stream. A visitor sees enrollment, cross-task verification, figure-backed evidence, and clear product
    directions without needing to read the entire manuscript first.
  `
};

function renderProtocols() {
  const host = document.getElementById("protocol-bars");
  const maxEer = Math.max(...protocolData.map((item) => item.eer));

  protocolData.forEach((item) => {
    const row = document.createElement("div");
    row.className = "protocol-row";
    row.innerHTML = `
      <div class="protocol-meta">
        <strong>${item.name}</strong>
        <span>EER ${item.eer.toFixed(2)}% | AUC ${item.auc.toFixed(4)}</span>
      </div>
      <div class="protocol-track">
        <div class="protocol-fill" style="width:${(item.eer / maxEer) * 100}%"></div>
      </div>
      <span class="small-text">${item.note}</span>
    `;
    host.appendChild(row);
  });
}

function renderGallery() {
  const tabs = document.getElementById("gallery-tabs");
  const image = document.getElementById("gallery-image");
  const title = document.getElementById("gallery-title");
  const description = document.getElementById("gallery-description");

  function selectItem(itemKey) {
    const item = galleryItems.find((entry) => entry.key === itemKey) || galleryItems[0];
    image.src = item.image;
    image.alt = item.title;
    title.textContent = item.title;
    description.textContent = item.description;
    [...tabs.children].forEach((button) => {
      button.classList.toggle("active", button.dataset.key === item.key);
    });
  }

  galleryItems.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = `tab-button${index === 0 ? " active" : ""}`;
    button.dataset.key = item.key;
    button.textContent = item.label;
    button.addEventListener("click", () => selectItem(item.key));
    tabs.appendChild(button);
  });

  selectItem(galleryItems[0].key);
}

function renderTable() {
  const host = document.getElementById("results-body");
  seedResults.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = row.map((value) => `<td>${value}</td>`).join("");
    host.appendChild(tr);
  });
}

function renderFocus() {
  const copy = document.getElementById("focus-copy");
  const buttons = [...document.querySelectorAll(".focus-button")];

  function setFocus(key) {
    copy.innerHTML = focusContent[key];
    buttons.forEach((button) => {
      button.classList.toggle("active", button.dataset.focus === key);
    });
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => setFocus(button.dataset.focus));
  });

  setFocus("science");
}

renderProtocols();
renderGallery();
renderTable();
renderFocus();
