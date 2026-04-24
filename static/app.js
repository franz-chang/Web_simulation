const grid = document.getElementById("grid");
const nextBtn = document.getElementById("nextBtn");
const resetBtn = document.getElementById("resetBtn");
const algoSelect = document.getElementById("algoSelect");
const datasetSelect = document.getElementById("datasetSelect");
const historyCommentSelect = document.getElementById("historyCommentSelect");
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

function historyCommentsEnabled() {
  if (!historyCommentSelect) return true;
  return historyCommentSelect.value !== "hide";
}

function formatStars(card) {
  const percent = Number(card?.rating_percent || 0);
  const label = escapeHtml(card?.rating_label || "0.0");
  return `
    <span class="metric-group rating-group">
      <span class="metric-label">Rating:</span>
      <span class="stars" style="--rating-percent:${Math.max(0, Math.min(100, percent))}%;" aria-hidden="true"></span>
      <span class="metric-value">${label}</span>
    </span>
  `;
}

function formatHeat(card) {
  const heatLabel = escapeHtml(card?.heat_label || "0°C");
  const base = Number(card?.heat_base || 0);
  const interaction = Number(card?.heat_interaction || 0);
  return `
    <span class="metric-group heat-group" title="Heat = historical baseline + session increment (impression +2°C, click +10°C)">
      <span class="metric-label">Heat:</span>
      <span class="metric-value">${heatLabel}</span>
      <span class="metric-subtle">(${Math.max(0, base)}+${Math.max(0, interaction)})</span>
    </span>
  `;
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

function formatCommentRows(movieId, comments) {
  const rows = Array.isArray(comments) ? comments : [];
  if (rows.length === 0) {
    return `<div class="comment-empty">No comments yet.</div>`;
  }

  return rows
    .map((comment) => {
      const commentId = escapeHtml(comment?.id || "");
      const text = escapeHtml(comment?.text || "");
      const likeCount = Number(comment?.like_count || 0);
      return `
        <div class="comment-row">
          <span class="comment-text">${text}</span>
          <button class="comment-like-btn" type="button" data-comment-id="${commentId}" data-movie-id="${escapeHtml(movieId)}">
            👍 ${Math.max(0, likeCount)}
          </button>
        </div>
      `;
    })
    .join("");
}

function renderCards(cards) {
  if (!cards || cards.length === 0) {
    grid.innerHTML = `<div class="empty">No more recommendations are available. Click "Random Reset" to start again.</div>`;
    return;
  }

  grid.innerHTML = cards
    .map((card) => {
      const movieId = escapeHtml(card.id);
      return `
        <article class="card" data-movie-id="${movieId}">
          <div class="card-main" data-movie-id="${movieId}">
            <img class="poster" src="${escapeHtml(card.image)}" alt="${escapeHtml(card.title)}">
            <div class="content">
              <h3 class="title">${escapeHtml(card.title)}</h3>
              <div class="card-metrics">
                ${formatStars(card)}
                ${formatHeat(card)}
              </div>
              <p class="desc">${escapeHtml(clip(card.description))}</p>
            </div>
          </div>
          <section class="comment-block">
            <div class="comment-list" data-movie-id="${movieId}">
              ${formatCommentRows(card.id, card.comments)}
            </div>
            <div class="comment-compose">
              <input
                class="comment-input"
                type="text"
                maxlength="500"
                data-movie-id="${movieId}"
                placeholder="Write a comment..."
              >
              <button class="comment-submit-btn" type="button" data-movie-id="${movieId}">Post</button>
            </div>
          </section>
        </article>
      `;
    })
    .join("");
}

function updateCommentList(movieId, comments) {
  const targetId = String(movieId || "");
  document.querySelectorAll(".comment-list").forEach((el) => {
    if (el.dataset.movieId === targetId) {
      el.innerHTML = formatCommentRows(targetId, comments);
    }
  });
}

function findCommentInput(movieId) {
  const targetId = String(movieId || "");
  const inputs = Array.from(document.querySelectorAll(".comment-input"));
  return inputs.find((el) => String(el.dataset.movieId || "") === targetId) || null;
}

function visibleMovieIds() {
  return Array.from(document.querySelectorAll(".card[data-movie-id]"))
    .map((el) => String(el.dataset.movieId || "").trim())
    .filter(Boolean);
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
  statusText.textContent = `Initializing random recommendations (${datasetLabel(datasetKey)} / ${modelLabel(modelName)})...`;

  const data = await fetchJSON(
    `/api/init?dataset_key=${encodeURIComponent(datasetKey)}&model_name=${encodeURIComponent(modelName)}&include_history_comments=${historyCommentsEnabled() ? "1" : "0"}`
  );

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);
  renderCards(data.cards);

  nextBtn.disabled = true;
  if ((data.available_models || []).length === 0) {
    statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)} currently has no available model, so random samples are shown.`;
  } else {
    statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)}: loaded 4 initial random samples (current model: ${modelLabel(data.model_name)}).`;
  }
}

async function selectMovie(movieId) {
  const modelName = algoSelect.value;
  const datasetKey = datasetSelect.value;

  statusText.textContent = `Clicked ${movieId}. Requesting recommendations from ${datasetLabel(datasetKey)} / ${modelLabel(modelName)}...`;

  const data = await fetchJSON("/api/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      movie_id: movieId,
      model_name: modelName,
      dataset_key: datasetKey,
      include_history_comments: historyCommentsEnabled(),
    }),
  });

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);
  renderCards(data.cards);

  nextBtn.disabled = !data.has_next;
  statusText.textContent = `${data.dataset_label || datasetLabel(data.dataset_key)} / ${modelLabel(data.model_name)} page ${data.page} recommendations (4 items per page).`;
}

async function nextPage() {
  const datasetKey = datasetSelect.value;
  statusText.textContent = "Loading the next page...";

  const data = await fetchJSON("/api/next", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_key: datasetKey,
      include_history_comments: historyCommentsEnabled(),
    }),
  });

  applyAvailableDatasets(data.available_datasets, data.dataset_key);
  applyAvailableModels(data.available_models, data.model_name);
  renderCards(data.cards);

  nextBtn.disabled = !data.has_next;
  statusText.textContent = data.cards.length
    ? `${data.dataset_label || datasetLabel(data.dataset_key)} / ${modelLabel(data.model_name)} page ${data.page} recommendations (4 items per page).`
    : (data.message || "No more recommendations are available.");
}

async function submitComment(movieId, inputEl) {
  const text = String(inputEl?.value || "").trim();
  if (!text) {
    statusText.textContent = "Please enter a comment before posting.";
    return;
  }
  if (!inputEl) return;

  inputEl.disabled = true;
  statusText.textContent = "Saving comment...";
  try {
    const data = await fetchJSON("/api/comment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_key: datasetSelect.value,
        item_id: movieId,
        text,
        include_history_comments: historyCommentsEnabled(),
      }),
    });
    updateCommentList(movieId, data.comments || []);
    inputEl.value = "";
    statusText.textContent = "Comment saved.";
  } catch (err) {
    statusText.textContent = `Failed to save comment: ${err.message}`;
  } finally {
    inputEl.disabled = false;
    inputEl.focus();
  }
}

async function likeComment(commentId, movieId, btnEl) {
  if (btnEl) btnEl.disabled = true;
  try {
    const data = await fetchJSON("/api/comment/like", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        comment_id: commentId,
        include_history_comments: historyCommentsEnabled(),
      }),
    });
    updateCommentList(movieId, data.comments || []);
    statusText.textContent = "Like recorded for current simulation.";
  } catch (err) {
    statusText.textContent = `Failed to like comment: ${err.message}`;
  } finally {
    if (btnEl) btnEl.disabled = false;
  }
}

async function refreshVisibleComments() {
  const movieIds = visibleMovieIds();
  if (movieIds.length === 0) return;

  try {
    const data = await fetchJSON(
      `/api/comments?dataset_key=${encodeURIComponent(datasetSelect.value)}&include_history_comments=${historyCommentsEnabled() ? "1" : "0"}&item_ids=${encodeURIComponent(movieIds.join(","))}`
    );
    const commentsByItem = data?.comments_by_item || {};
    movieIds.forEach((movieId) => {
      updateCommentList(movieId, commentsByItem[movieId] || []);
    });
    statusText.textContent = historyCommentsEnabled()
      ? "Historical comments are now visible."
      : "Historical comments are now hidden.";
  } catch (err) {
    statusText.textContent = `Failed to refresh comments: ${err.message}`;
  }
}

nextBtn.addEventListener("click", nextPage);
resetBtn.addEventListener("click", initPage);
algoSelect.addEventListener("change", initPage);
datasetSelect.addEventListener("change", initPage);

if (historyCommentSelect) {
  historyCommentSelect.addEventListener("change", refreshVisibleComments);
}

grid.addEventListener("click", (event) => {
  const submitBtn = event.target.closest(".comment-submit-btn");
  if (submitBtn) {
    event.preventDefault();
    event.stopPropagation();
    const movieId = String(submitBtn.dataset.movieId || "").trim();
    const inputEl = findCommentInput(movieId);
    submitComment(movieId, inputEl);
    return;
  }

  const likeBtn = event.target.closest(".comment-like-btn");
  if (likeBtn) {
    event.preventDefault();
    event.stopPropagation();
    const commentId = String(likeBtn.dataset.commentId || "").trim();
    const movieId = String(likeBtn.dataset.movieId || "").trim();
    likeComment(commentId, movieId, likeBtn);
    return;
  }

  const cardMain = event.target.closest(".card-main");
  if (cardMain) {
    const movieId = String(cardMain.dataset.movieId || "").trim();
    if (movieId) selectMovie(movieId);
  }
});

grid.addEventListener("keydown", (event) => {
  const inputEl = event.target.closest(".comment-input");
  if (!inputEl || event.key !== "Enter") return;
  event.preventDefault();
  const movieId = String(inputEl.dataset.movieId || "").trim();
  submitComment(movieId, inputEl);
});

window.addEventListener("pagehide", () => {
  if (navigator.sendBeacon) {
    navigator.sendBeacon("/api/session/end", new Blob([], { type: "application/json" }));
  }
});

initPage().catch((err) => {
  statusText.textContent = `Initialization failed: ${err.message}`;
});
