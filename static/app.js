const grid = document.getElementById("grid");
const nextBtn = document.getElementById("nextBtn");
const resetBtn = document.getElementById("resetBtn");
const algoSelect = document.getElementById("algoSelect");
const datasetSelect = document.getElementById("datasetSelect");
const statusText = document.getElementById("statusText");

function clip(text, maxLen = 110) {
  if (!text) return "";
  return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function modelLabel(modelName) {
  if (modelName === "poprec") return "PopRec";
  if (modelName === "bprmf") return "BPR-MF";
  if (modelName === "gru4rec") return "GRU4Rec";
  if (modelName === "bert4rec") return "BERT4Rec";
  if (modelName === "lightgcn") return "LightGCN";
  if (modelName === "multvae") return "Mult-VAE";
  if (modelName === "sasrec") return "SASRec";
  return "Random";
}

function datasetLabel(datasetKey) {
  if (datasetKey === "ml1m") return "MovieLens-1M";
  if (datasetKey === "amazon_all_beauty") return "Amazon All Beauty";
  if (datasetKey === "amazon_magazine_subscriptions") return "Amazon Magazine Subscriptions";
  return String(datasetKey || "Unknown Dataset");
}

function applyAvailableModels(models, selectedModel) {
  const allowed = new Set(models || []);
  Array.from(algoSelect.options).forEach((opt) => {
    opt.disabled = !allowed.has(opt.value);
  });

  if (allowed.size === 0) {
    return;
  }

  if (allowed.has(selectedModel)) {
    algoSelect.value = selectedModel;
    return;
  }

  if (!allowed.has(algoSelect.value)) {
    const first = Array.from(algoSelect.options).find((opt) => !opt.disabled);
    if (first) algoSelect.value = first.value;
  }
}

function applyAvailableDatasets(datasets, selectedDataset) {
  const enabled = new Set((datasets || []).filter((x) => x.enabled).map((x) => x.key));

  Array.from(datasetSelect.options).forEach((opt) => {
    opt.disabled = !enabled.has(opt.value);
  });

  if (enabled.has(selectedDataset)) {
    datasetSelect.value = selectedDataset;
    return;
  }

  if (!enabled.has(datasetSelect.value)) {
    const first = Array.from(datasetSelect.options).find((opt) => !opt.disabled);
    if (first) datasetSelect.value = first.value;
  }
}

function renderCards(cards) {
  if (!cards || cards.length === 0) {
    grid.innerHTML = `<div class="empty">没有更多推荐了，你可以点击“随机重置”重新开始。</div>`;
    return;
  }

  grid.innerHTML = cards
    .map(
      (card) => `
      <article class="card" data-movie-id="${escapeHtml(card.id)}">
        <img class="poster" src="${escapeHtml(card.image)}" alt="${escapeHtml(card.title)}">
        <div class="content">
          <h3 class="title">${escapeHtml(card.title)}</h3>
          <p class="desc">${escapeHtml(clip(card.description))}</p>
        </div>
      </article>`
    )
    .join("");

  document.querySelectorAll(".card").forEach((el) => {
    el.addEventListener("click", async () => {
      const movieId = el.dataset.movieId;
      await selectMovie(movieId);
    });
  });
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

async function initPage() {
  const modelName = algoSelect.value;
  const datasetKey = datasetSelect.value;
  statusText.textContent = `初始化随机推荐（${datasetLabel(datasetKey)} / ${modelLabel(modelName)}）...`;

  const data = await fetchJSON(
    `/api/init?dataset_key=${encodeURIComponent(datasetKey)}&model_name=${encodeURIComponent(modelName)}`
  );

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);
  renderCards(data.cards);

  nextBtn.disabled = true;
  if ((data.available_models || []).length === 0) {
    statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)} 当前无可用模型，显示随机样本。`;
  } else {
    statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)}：初始随机 4 个样本（当前模型：${modelLabel(data.model_name)}）`;
  }
}

async function selectMovie(movieId) {
  const modelName = algoSelect.value;
  const datasetKey = datasetSelect.value;

  statusText.textContent = `已点击 ${movieId}，正在请求 ${datasetLabel(datasetKey)} / ${modelLabel(modelName)} 推荐...`;

  const data = await fetchJSON("/api/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ movie_id: movieId, model_name: modelName, dataset_key: datasetKey }),
  });

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);
  renderCards(data.cards);

  nextBtn.disabled = !data.has_next;
  statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)} / ${modelLabel(data.model_name)} 推荐第 ${data.page} 页（每页 4 个）`;
}

async function nextPage() {
  const datasetKey = datasetSelect.value;
  statusText.textContent = "加载下一页...";

  const data = await fetchJSON("/api/next", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_key: datasetKey }),
  });

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);
  renderCards(data.cards);

  nextBtn.disabled = !data.has_next;
  statusText.textContent = data.cards.length
    ? `${data.dataset_label || datasetLabel(data.dataset_key)} / ${modelLabel(data.model_name)} 推荐第 ${data.page} 页（每页 4 个）`
    : (data.message || "没有更多推荐了");
}

nextBtn.addEventListener("click", nextPage);
resetBtn.addEventListener("click", initPage);
algoSelect.addEventListener("change", initPage);
datasetSelect.addEventListener("change", initPage);

initPage().catch((err) => {
  statusText.textContent = `初始化失败: ${err.message}`;
});
