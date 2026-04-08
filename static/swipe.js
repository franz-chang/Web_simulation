const datasetSelect = document.getElementById("datasetSelect");
const algoSelect = document.getElementById("algoSelect");
const resetBtn = document.getElementById("resetBtn");
const upBtn = document.getElementById("upBtn");
const downBtn = document.getElementById("downBtn");
const statusText = document.getElementById("statusText");
const swipeCard = document.getElementById("swipeCard");
const swipePoster = document.getElementById("swipePoster");
const swipeTitle = document.getElementById("swipeTitle");
const swipeDesc = document.getElementById("swipeDesc");
const datasetBadge = document.getElementById("datasetBadge");
const modelBadge = document.getElementById("modelBadge");
const indexBadge = document.getElementById("indexBadge");

const feedState = {
  history: [],
  cursor: -1,
  queue: [],
  loading: false,
};

function clip(text, maxLen = 160) {
  if (!text) return "";
  return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
}

function modelLabel(modelName) {
  if (modelName === "poprec") return "PopRec";
  if (modelName === "bprmf") return "BPR-MF";
  if (modelName === "gru4rec") return "GRU4Rec";
  if (modelName === "bert4rec") return "BERT4Rec";
  if (modelName === "lightgcn") return "LightGCN";
  if (modelName === "multvae") return "Mult-VAE";
  if (modelName === "sasrec") return "SASRec";
  if (!modelName) return "Random";
  return String(modelName).toUpperCase();
}

function datasetLabel(datasetKey) {
  if (datasetKey === "amazon_mi") return "Amazon Musical Instruments";
  if (datasetKey === "ml1m") return "MovieLens-1M";
  return String(datasetKey || "Unknown Dataset");
}

function syncMeta() {
  if (datasetBadge) {
    datasetBadge.textContent = datasetLabel(datasetSelect.value);
  }
  if (modelBadge) {
    modelBadge.textContent = modelLabel(algoSelect.value);
  }
  if (indexBadge) {
    indexBadge.textContent = feedState.cursor >= 0 ? `第 ${feedState.cursor + 1} 条` : "第 0 条";
  }
}

function currentCard() {
  if (feedState.cursor < 0 || feedState.cursor >= feedState.history.length) {
    return null;
  }
  return feedState.history[feedState.cursor];
}

function dedupeCards(cards) {
  const seen = new Set();
  const out = [];
  for (const card of cards || []) {
    const itemId = String(card?.id || "").trim();
    if (!itemId || seen.has(itemId)) continue;
    seen.add(itemId);
    out.push(card);
  }
  return out;
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

function syncButtons() {
  upBtn.disabled = feedState.loading || feedState.cursor <= 0;
  downBtn.disabled = feedState.loading || feedState.cursor < 0;
  resetBtn.disabled = feedState.loading;
  syncMeta();
}

function renderCard(card, withAnimation = false) {
  if (!card) {
    swipePoster.src = "";
    swipePoster.alt = "暂无推荐内容";
    swipeTitle.textContent = "暂无推荐";
    swipeDesc.textContent = "请点击“随机重置按钮”重新加载。";
    return;
  }

  swipePoster.src = String(card.image || "");
  swipePoster.alt = String(card.title || "推荐内容");
  swipeTitle.textContent = String(card.title || "未知标题");
  swipeDesc.textContent = clip(String(card.description || "暂无简介"));

  if (withAnimation) {
    swipeCard.classList.remove("card-enter");
    void swipeCard.offsetWidth;
    swipeCard.classList.add("card-enter");
  }
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

async function initFeed() {
  if (feedState.loading) return;
  feedState.loading = true;
  syncButtons();

  const modelName = algoSelect.value;
  const datasetKey = datasetSelect.value;
  statusText.textContent = `初始化中：${datasetLabel(datasetKey)} / ${modelLabel(modelName)} ...`;

  try {
    const data = await fetchJSON(
      `/api/init?dataset_key=${encodeURIComponent(datasetKey)}&model_name=${encodeURIComponent(modelName)}`
    );

    applyAvailableDatasets(data.available_datasets, data.dataset_key);
    applyAvailableModels(data.available_models, data.model_name);

    const cards = dedupeCards(data.cards);
    feedState.history = [];
    feedState.queue = [];
    feedState.cursor = -1;

    if (cards.length > 0) {
      feedState.history.push(cards[0]);
      feedState.cursor = 0;
      feedState.queue = cards.slice(1);
      renderCard(cards[0], true);
      statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)} / ${modelLabel(data.model_name)} 已加载第一条内容`;
    } else {
      renderCard(null);
      statusText.textContent = "初始化后没有可展示内容，请重置重试。";
    }
  } catch (err) {
    renderCard(null);
    statusText.textContent = `初始化失败: ${err.message}`;
  } finally {
    feedState.loading = false;
    syncButtons();
  }
}

function moveUp() {
  if (feedState.loading) return;
  if (feedState.cursor <= 0) {
    statusText.textContent = "已经到最上方了。";
    return;
  }
  feedState.cursor -= 1;
  renderCard(currentCard(), true);
  statusText.textContent = `回看第 ${feedState.cursor + 1} 条`;
  syncButtons();
}

async function requestMoreByCurrentCard() {
  const card = currentCard();
  if (!card) return null;

  const modelName = algoSelect.value;
  const datasetKey = datasetSelect.value;

  const data = await fetchJSON("/api/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      movie_id: String(card.id || ""),
      model_name: modelName,
      dataset_key: datasetKey,
    }),
  });

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);

  const candidates = dedupeCards(data.cards);
  if (candidates.length === 0) {
    return { data, nextCard: null };
  }

  let pickIdx = candidates.findIndex((x) => String(x.id || "") !== String(card.id || ""));
  if (pickIdx < 0) pickIdx = 0;

  const nextCard = candidates[pickIdx];
  const queue = candidates.filter((_, idx) => idx !== pickIdx);
  return { data, nextCard, queue };
}

async function moveDown() {
  if (feedState.loading) return;

  if (feedState.cursor >= 0 && feedState.cursor < feedState.history.length - 1) {
    feedState.cursor += 1;
    renderCard(currentCard(), true);
    statusText.textContent = `继续浏览第 ${feedState.cursor + 1} 条`;
    syncButtons();
    return;
  }

  if (feedState.queue.length > 0) {
    const nextCard = feedState.queue.shift();
    feedState.history.push(nextCard);
    feedState.cursor = feedState.history.length - 1;
    renderCard(nextCard, true);
    statusText.textContent = `下滑到第 ${feedState.cursor + 1} 条`;
    syncButtons();
    return;
  }

  const current = currentCard();
  if (!current) {
    statusText.textContent = "当前没有内容，请先点击随机重置。";
    return;
  }

  feedState.loading = true;
  syncButtons();
  statusText.textContent = "根据当前内容加载下一条...";

  try {
    const result = await requestMoreByCurrentCard();
    const data = result?.data;
    if (!result || !result.nextCard) {
      statusText.textContent = (data && data.message) || "没有更多推荐了，请重置后继续。";
      return;
    }

    feedState.history.push(result.nextCard);
    feedState.cursor = feedState.history.length - 1;
    feedState.queue = result.queue || [];
    renderCard(result.nextCard, true);

    const showDataset = (data && (data.dataset_label || datasetLabel(data.dataset_key))) || datasetLabel(datasetSelect.value);
    const showModel = (data && modelLabel(data.model_name)) || modelLabel(algoSelect.value);
    statusText.textContent = `${showDataset} / ${showModel} 下滑到第 ${feedState.cursor + 1} 条`;
  } catch (err) {
    statusText.textContent = `下滑失败: ${err.message}`;
  } finally {
    feedState.loading = false;
    syncButtons();
  }
}

upBtn.addEventListener("click", moveUp);
downBtn.addEventListener("click", moveDown);
resetBtn.addEventListener("click", initFeed);
datasetSelect.addEventListener("change", initFeed);
algoSelect.addEventListener("change", initFeed);

window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp") {
    event.preventDefault();
    moveUp();
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    moveDown();
  }
});

initFeed();
