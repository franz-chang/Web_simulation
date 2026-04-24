import html
import json
import math
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file, session

from recommender import MovieRecommender


BASE_DIR = Path(__file__).resolve().parent
PAGE_SIZE = 4
TOPK = 200


def first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def first_env_or_default(env_names: list[str], default: str) -> str:
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return default


def int_env_or_default(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


DEFAULT_ML1M_DIR = first_existing_path(
    [
        BASE_DIR.parent / "WebSim_Dataset" / "MM-ML-1M-main",
        BASE_DIR / "Dataset" / "MM-ML-1M-main",
        Path("/Users/chongzhang/ml1m_sasrec_web/Dataset/MM-ML-1M-main"),
    ]
)
DEFAULT_AMAZON_MM_2018_ROOT = first_existing_path(
    [
        BASE_DIR.parent / "WebSim_Dataset" / "Amazon_MM_2018",
        BASE_DIR / "Dataset" / "Amazon_MM_2018",
    ]
)
DEFAULT_AMAZON_ALL_BEAUTY_DIR = first_existing_path(
    [
        DEFAULT_AMAZON_MM_2018_ROOT / "All_Beauty",
        BASE_DIR.parent / "WebSim_Dataset" / "All_Beauty",
        BASE_DIR / "Dataset" / "All_Beauty",
    ]
)
DEFAULT_AMAZON_MAGAZINE_SUBSCRIPTIONS_DIR = first_existing_path(
    [
        DEFAULT_AMAZON_MM_2018_ROOT / "Magazine_Subscriptions",
        BASE_DIR.parent / "WebSim_Dataset" / "Magazine_Subscriptions",
        BASE_DIR / "Dataset" / "Magazine_Subscriptions",
    ]
)


DATASET_CONFIGS = {
    "ml1m": {
        "label": "MovieLens-1M",
        "dataset_dir": os.getenv("ML1M_DATASET_DIR", str(DEFAULT_ML1M_DIR)),
        "models": {
            "sasrec": first_env_or_default(
                ["SASREC_ML1M_MODEL_PATH", "SASREC_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "sasrec_ml1m.pt"),
            ),
            "lightgcn": first_env_or_default(
                ["LIGHTGCN_ML1M_MODEL_PATH", "LIGHTGCN_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "lightgcn_ml1m.pt"),
            ),
            "multvae": first_env_or_default(
                ["MULTVAE_ML1M_MODEL_PATH", "MULTVAE_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "multvae_ml1m.pt"),
            ),
            "poprec": first_env_or_default(
                ["POPREC_ML1M_MODEL_PATH", "POPREC_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "poprec_ml1m.pt"),
            ),
            "bprmf": first_env_or_default(
                ["BPRMF_ML1M_MODEL_PATH", "BPRMF_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "bprmf_ml1m.pt"),
            ),
            "gru4rec": first_env_or_default(
                ["GRU4REC_ML1M_MODEL_PATH", "GRU4REC_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "gru4rec_ml1m.pt"),
            ),
            "bert4rec": first_env_or_default(
                ["BERT4REC_ML1M_MODEL_PATH", "BERT4REC_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "bert4rec_ml1m.pt"),
            ),
        },
    },
    "amazon_all_beauty": {
        "label": "Amazon All Beauty",
        "dataset_dir": os.getenv("AMAZON_ALL_BEAUTY_DATASET_DIR", str(DEFAULT_AMAZON_ALL_BEAUTY_DIR)),
        "models": {
            "sasrec": first_env_or_default(
                ["SASREC_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "sasrec_amazon_all_beauty.pt"),
            ),
            "lightgcn": first_env_or_default(
                ["LIGHTGCN_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "lightgcn_amazon_all_beauty.pt"),
            ),
            "multvae": first_env_or_default(
                ["MULTVAE_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "multvae_amazon_all_beauty.pt"),
            ),
            "poprec": first_env_or_default(
                ["POPREC_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "poprec_amazon_all_beauty.pt"),
            ),
            "bprmf": first_env_or_default(
                ["BPRMF_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "bprmf_amazon_all_beauty.pt"),
            ),
            "gru4rec": first_env_or_default(
                ["GRU4REC_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "gru4rec_amazon_all_beauty.pt"),
            ),
            "bert4rec": first_env_or_default(
                ["BERT4REC_AMAZON_ALL_BEAUTY_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "bert4rec_amazon_all_beauty.pt"),
            ),
        },
    },
    "amazon_magazine_subscriptions": {
        "label": "Amazon Magazine Subscriptions",
        "dataset_dir": os.getenv(
            "AMAZON_MAGAZINE_SUBSCRIPTIONS_DATASET_DIR",
            str(DEFAULT_AMAZON_MAGAZINE_SUBSCRIPTIONS_DIR),
        ),
        "models": {
            "sasrec": first_env_or_default(
                ["SASREC_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "sasrec_amazon_magazine_subscriptions.pt"),
            ),
            "lightgcn": first_env_or_default(
                ["LIGHTGCN_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "lightgcn_amazon_magazine_subscriptions.pt"),
            ),
            "multvae": first_env_or_default(
                ["MULTVAE_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "multvae_amazon_magazine_subscriptions.pt"),
            ),
            "poprec": first_env_or_default(
                ["POPREC_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "poprec_amazon_magazine_subscriptions.pt"),
            ),
            "bprmf": first_env_or_default(
                ["BPRMF_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "bprmf_amazon_magazine_subscriptions.pt"),
            ),
            "gru4rec": first_env_or_default(
                ["GRU4REC_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "gru4rec_amazon_magazine_subscriptions.pt"),
            ),
            "bert4rec": first_env_or_default(
                ["BERT4REC_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "bert4rec_amazon_magazine_subscriptions.pt"),
            ),
        },
    },
}

DEFAULT_DATASET_KEY = os.getenv("DEFAULT_DATASET_KEY", "ml1m")
DEFAULT_INIT_RANDOM_SEED = int_env_or_default("WEBSIM_INIT_RANDOM_SEED", 42)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

recommenders: dict[str, MovieRecommender] = {}
simulation_stats_store: dict[str, dict[str, dict[str, int]]] = {}
simulation_comment_likes_store: dict[str, dict[str, int]] = {}

COMMENTS_STORE_PATH = BASE_DIR / "artifacts" / "item_comments.json"
comment_records: list[dict[str, str]] = []
comment_records_by_id: dict[str, dict[str, str]] = {}


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "show"}:
        return True
    if text in {"0", "false", "no", "n", "off", "hide"}:
        return False
    return default


def normalize_comment_text(value) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:500]


def _load_comment_records() -> list[dict[str, str]]:
    if not COMMENTS_STORE_PATH.exists():
        return []

    try:
        raw = json.loads(COMMENTS_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, dict):
        data = raw.get("comments", [])
    else:
        data = raw
    if not isinstance(data, list):
        return []

    records: list[dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        comment_id = str(row.get("id", "")).strip()
        raw_dataset_key = str(row.get("dataset_key", "")).strip().lower()
        if raw_dataset_key in DATASET_CONFIGS:
            dataset_key = raw_dataset_key
        else:
            dataset_key = DEFAULT_DATASET_KEY if DEFAULT_DATASET_KEY in DATASET_CONFIGS else "ml1m"
        item_id = str(row.get("item_id", "")).strip()
        text = normalize_comment_text(row.get("text", ""))
        created_at = str(row.get("created_at", "")).strip() or datetime.now(timezone.utc).isoformat()
        simulation_id = str(row.get("simulation_id", "")).strip()
        if not comment_id or not item_id or not text:
            continue
        records.append(
            {
                "id": comment_id,
                "dataset_key": dataset_key,
                "item_id": item_id,
                "text": text,
                "created_at": created_at,
                "simulation_id": simulation_id,
            }
        )
    return records


def _save_comment_records() -> None:
    COMMENTS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "comments": comment_records}
    COMMENTS_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _index_comment_records(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["id"]): row for row in records if row.get("id")}


def _fresh_comment_likes() -> dict[str, int]:
    return {}


comment_records = _load_comment_records()
comment_records_by_id = _index_comment_records(comment_records)


def normalize_dataset_key(dataset_key: str | None) -> str:
    if not dataset_key:
        return DEFAULT_DATASET_KEY if DEFAULT_DATASET_KEY in DATASET_CONFIGS else "ml1m"
    key = str(dataset_key).strip().lower()
    if key in DATASET_CONFIGS:
        return key
    return DEFAULT_DATASET_KEY if DEFAULT_DATASET_KEY in DATASET_CONFIGS else "ml1m"


def available_datasets() -> list[dict[str, str | bool]]:
    items = []
    for key, cfg in DATASET_CONFIGS.items():
        dataset_dir = Path(cfg["dataset_dir"])
        items.append(
            {
                "key": key,
                "label": cfg["label"],
                "enabled": dataset_dir.exists(),
            }
        )
    return items


def get_recommender(dataset_key: str) -> MovieRecommender:
    key = normalize_dataset_key(dataset_key)
    if key in recommenders:
        return recommenders[key]

    cfg = DATASET_CONFIGS[key]
    dataset_dir = Path(cfg["dataset_dir"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    models = cfg["models"]
    recommender = MovieRecommender(
        dataset_key=key,
        dataset_dir=str(dataset_dir),
        random_seed=DEFAULT_INIT_RANDOM_SEED,
        sasrec_artifact_path=models.get("sasrec"),
        lightgcn_artifact_path=models.get("lightgcn"),
        multvae_artifact_path=models.get("multvae"),
        poprec_artifact_path=models.get("poprec"),
        bprmf_artifact_path=models.get("bprmf"),
        gru4rec_artifact_path=models.get("gru4rec"),
        bert4rec_artifact_path=models.get("bert4rec"),
    )
    recommenders[key] = recommender
    return recommender


def unique_keep_order(item_ids: list[str]) -> list[str]:
    seen = set()
    unique_ids = []
    for item_id in item_ids:
        item_id = str(item_id)
        if item_id in seen:
            continue
        seen.add(item_id)
        unique_ids.append(item_id)
    return unique_ids


def _fresh_simulation_stats() -> dict[str, dict[str, int]]:
    return {"impressions": {}, "clicks": {}}


def _pop_simulation_stats(simulation_id: str | None) -> None:
    if simulation_id:
        simulation_stats_store.pop(str(simulation_id), None)


def _pop_comment_likes(simulation_id: str | None) -> None:
    if simulation_id:
        simulation_comment_likes_store.pop(str(simulation_id), None)


def reset_simulation_stats() -> str:
    old_simulation_id = session.get("simulation_id")
    _pop_simulation_stats(old_simulation_id)
    _pop_comment_likes(old_simulation_id)
    simulation_id = secrets.token_hex(12)
    session["simulation_id"] = simulation_id
    simulation_stats_store[simulation_id] = _fresh_simulation_stats()
    simulation_comment_likes_store[simulation_id] = _fresh_comment_likes()
    return simulation_id


def get_simulation_stats(create: bool = False) -> dict[str, dict[str, int]]:
    simulation_id = str(session.get("simulation_id", "") or "")
    if not simulation_id and create:
        simulation_id = reset_simulation_stats()
    if not simulation_id:
        return _fresh_simulation_stats()
    return simulation_stats_store.setdefault(simulation_id, _fresh_simulation_stats())


def get_comment_likes(create: bool = False) -> dict[str, int]:
    simulation_id = str(session.get("simulation_id", "") or "")
    if not simulation_id and create:
        simulation_id = reset_simulation_stats()
    if not simulation_id:
        return _fresh_comment_likes()
    return simulation_comment_likes_store.setdefault(simulation_id, _fresh_comment_likes())


def clear_simulation_state() -> None:
    old_simulation_id = session.get("simulation_id")
    _pop_simulation_stats(old_simulation_id)
    _pop_comment_likes(old_simulation_id)
    session.pop("simulation_id", None)
    session["history"] = []
    session["rec_ids"] = []
    session["page_idx"] = 0
    session.pop("model_name", None)
    session.pop("dataset_key", None)


def _serialize_comment_record(
    row: dict[str, str], *, like_counts: dict[str, int], simulation_id: str
) -> dict[str, str | int | bool]:
    comment_id = str(row.get("id", ""))
    return {
        "id": comment_id,
        "dataset_key": str(row.get("dataset_key", "")),
        "item_id": str(row.get("item_id", "")),
        "text": str(row.get("text", "")),
        "created_at": str(row.get("created_at", "")),
        "like_count": int(like_counts.get(comment_id, 0)),
        "is_current_simulation": bool(simulation_id and str(row.get("simulation_id", "")) == simulation_id),
    }


def list_comments_for_items(
    *,
    dataset_key: str,
    item_ids: list[str],
    include_history_comments: bool,
) -> dict[str, list[dict[str, str | int | bool]]]:
    clean_dataset_key = normalize_dataset_key(dataset_key)
    clean_item_ids = unique_keep_order([str(x).strip() for x in item_ids if str(x).strip()])
    out: dict[str, list[dict[str, str | int | bool]]] = {item_id: [] for item_id in clean_item_ids}
    if not clean_item_ids:
        return out

    simulation_id = str(session.get("simulation_id", "") or "")
    like_counts = get_comment_likes(create=False)
    target_item_ids = set(clean_item_ids)

    for row in comment_records:
        row_dataset_key = str(row.get("dataset_key", ""))
        row_item_id = str(row.get("item_id", ""))
        if row_dataset_key != clean_dataset_key or row_item_id not in target_item_ids:
            continue
        if not include_history_comments and str(row.get("simulation_id", "")) != simulation_id:
            continue
        out.setdefault(row_item_id, []).append(
            _serialize_comment_record(row, like_counts=like_counts, simulation_id=simulation_id)
        )

    for item_id in clean_item_ids:
        out[item_id].sort(key=lambda x: str(x.get("created_at", "")))
    return out


def attach_comments_to_cards(
    cards: list[dict], *, dataset_key: str, include_history_comments: bool
) -> list[dict]:
    item_ids = [str(card.get("id", "")).strip() for card in cards]
    comments_by_item = list_comments_for_items(
        dataset_key=dataset_key,
        item_ids=item_ids,
        include_history_comments=include_history_comments,
    )
    out = [dict(card) for card in cards]
    for card in out:
        item_id = str(card.get("id", "")).strip()
        card["comments"] = comments_by_item.get(item_id, [])
    return out


def record_card_impressions(cards: list[dict]) -> None:
    stats = get_simulation_stats(create=True)
    impressions = stats["impressions"]
    for item_id in unique_keep_order([str(card.get("id", "")).strip() for card in cards]):
        if not item_id:
            continue
        impressions[item_id] = int(impressions.get(item_id, 0)) + 1


def record_item_click(item_id: str) -> None:
    clean_id = str(item_id or "").strip()
    if not clean_id:
        return
    stats = get_simulation_stats(create=True)
    clicks = stats["clicks"]
    clicks[clean_id] = int(clicks.get(clean_id, 0)) + 1


def enrich_cards(cards: list[dict], *, track_impressions: bool) -> list[dict]:
    cards = [dict(card) for card in cards]
    if track_impressions and cards:
        record_card_impressions(cards)

    stats = get_simulation_stats(create=track_impressions)
    impressions = stats.get("impressions", {})
    clicks = stats.get("clicks", {})
    out: list[dict] = []
    for card in cards:
        item_id = str(card.get("id", "")).strip()
        rating_value = float(card.get("rating_value", 0.0) or 0.0)
        rating_value = max(0.0, min(5.0, rating_value))
        rating_count = int(card.get("rating_count", 0) or 0)
        impression_count = int(impressions.get(item_id, 0))
        click_count = int(clicks.get(item_id, 0))

        # Heat combines:
        # 1) global popularity baseline from rating_count (log-scaled),
        # 2) session interaction increments from impressions/clicks.
        if rating_count > 0:
            base_heat = int(round(min(90.0, math.log10(rating_count + 1.0) * 30.0)))
        else:
            base_heat = 0
        interaction_heat = min(60, impression_count * 2 + click_count * 10)
        heat_celsius = max(0, min(150, base_heat + interaction_heat))

        card["rating_value"] = rating_value
        card["rating_label"] = f"{rating_value:.1f}"
        card["rating_percent"] = max(0.0, min(100.0, rating_value / 5.0 * 100.0))
        card["rating_count"] = rating_count
        card["heat_celsius"] = heat_celsius
        card["heat_label"] = f"{heat_celsius}\u00b0C"
        card["heat_base"] = base_heat
        card["heat_interaction"] = interaction_heat
        card["heat_clicks"] = click_count
        card["heat_impressions"] = impression_count
        out.append(card)
    return out


def paginate_cards(recommender: MovieRecommender, item_ids: list[str], page_idx: int) -> list[dict]:
    item_ids = unique_keep_order(item_ids)
    start = page_idx * PAGE_SIZE
    end = start + PAGE_SIZE
    page_ids = item_ids[start:end]
    return [recommender.movie_card(mid) for mid in page_ids]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/swipe")
def swipe():
    return render_template("swipe.html")


@app.route("/health")
def health():
    status = {}
    for key in DATASET_CONFIGS:
        try:
            rec = get_recommender(key)
            status[key] = {
                "dataset_dir": DATASET_CONFIGS[key]["dataset_dir"],
                "available_models": rec.available_models(),
                "default_model": rec.default_model(),
            }
        except Exception as err:
            status[key] = {
                "dataset_dir": DATASET_CONFIGS[key]["dataset_dir"],
                "error": str(err),
                "available_models": [],
            }

    return jsonify(
        {
            "status": "ok",
            "service": "multidataset-multimodel-web",
            "datasets": status,
            "default_dataset": normalize_dataset_key(DEFAULT_DATASET_KEY),
        }
    )


@app.route("/api/init", methods=["GET"])
def api_init():
    dataset_key = normalize_dataset_key(request.args.get("dataset_key"))
    include_history_comments = parse_bool(request.args.get("include_history_comments"), True)
    try:
        recommender = get_recommender(dataset_key)
    except Exception as err:
        return jsonify({"error": str(err), "available_datasets": available_datasets()}), 400

    model_name = recommender.normalize_model_name(request.args.get("model_name"))
    reset_simulation_stats()
    cards = [
        recommender.movie_card(mid)
        for mid in recommender.random_movie_ids(PAGE_SIZE, avoid_last_same=True)
    ]
    cards = enrich_cards(cards, track_impressions=True)
    cards = attach_comments_to_cards(
        cards,
        dataset_key=dataset_key,
        include_history_comments=include_history_comments,
    )

    session["history"] = []
    session["rec_ids"] = []
    session["page_idx"] = 0
    session["model_name"] = model_name
    session["dataset_key"] = dataset_key

    return jsonify(
        {
            "cards": cards,
            "mode": "random",
            "dataset_key": dataset_key,
            "dataset_label": DATASET_CONFIGS[dataset_key]["label"],
            "available_datasets": available_datasets(),
            "model_name": model_name,
            "available_models": recommender.available_models(),
            "page": 1,
            "has_next": False,
            "history": [],
            "include_history_comments": include_history_comments,
        }
    )


@app.route("/api/select", methods=["POST"])
def api_select():
    payload = request.get_json(silent=True) or {}
    include_history_comments = parse_bool(payload.get("include_history_comments"), True)

    dataset_key = normalize_dataset_key(payload.get("dataset_key") or session.get("dataset_key"))
    try:
        recommender = get_recommender(dataset_key)
    except Exception as err:
        return jsonify({"error": str(err), "available_datasets": available_datasets()}), 400

    model_name = recommender.normalize_model_name(payload.get("model_name") or session.get("model_name"))
    movie_id = str(payload.get("movie_id", "")).strip()

    prev_dataset_key = session.get("dataset_key")
    if prev_dataset_key != dataset_key:
        history: list[str] = []
        reset_simulation_stats()
    else:
        history = [str(x) for x in session.get("history", [])]

    if movie_id:
        record_item_click(movie_id)
        history.append(movie_id)
        history = history[-20:]

    rec_ids = recommender.recommend_ids(history, model_name=model_name, topk=TOPK)
    rec_ids = unique_keep_order(rec_ids)
    cards = enrich_cards(paginate_cards(recommender, rec_ids, 0), track_impressions=True)
    cards = attach_comments_to_cards(
        cards,
        dataset_key=dataset_key,
        include_history_comments=include_history_comments,
    )

    session["history"] = history
    session["rec_ids"] = rec_ids
    session["page_idx"] = 0
    session["model_name"] = model_name
    session["dataset_key"] = dataset_key

    return jsonify(
        {
            "cards": cards,
            "mode": model_name,
            "dataset_key": dataset_key,
            "dataset_label": DATASET_CONFIGS[dataset_key]["label"],
            "available_datasets": available_datasets(),
            "model_name": model_name,
            "available_models": recommender.available_models(),
            "page": 1,
            "has_next": len(rec_ids) > PAGE_SIZE,
            "history": history,
            "include_history_comments": include_history_comments,
        }
    )


@app.route("/api/next", methods=["POST"])
def api_next():
    payload = request.get_json(silent=True) or {}
    include_history_comments = parse_bool(payload.get("include_history_comments"), True)
    dataset_key = normalize_dataset_key(payload.get("dataset_key") or session.get("dataset_key"))

    try:
        recommender = get_recommender(dataset_key)
    except Exception as err:
        return jsonify({"error": str(err), "available_datasets": available_datasets()}), 400

    model_name = recommender.normalize_model_name(session.get("model_name"))
    if session.get("dataset_key") != dataset_key:
        rec_ids = []
    else:
        rec_ids = unique_keep_order([str(x) for x in session.get("rec_ids", [])])

    if not rec_ids:
        return jsonify(
            {
                "cards": [],
                "mode": model_name,
                "dataset_key": dataset_key,
                "dataset_label": DATASET_CONFIGS[dataset_key]["label"],
                "available_datasets": available_datasets(),
                "model_name": model_name,
                "available_models": recommender.available_models(),
                "page": 1,
                "has_next": False,
                "history": [],
                "include_history_comments": include_history_comments,
            }
        )

    page_idx = int(session.get("page_idx", 0)) + 1
    cards = enrich_cards(paginate_cards(recommender, rec_ids, page_idx), track_impressions=True)
    cards = attach_comments_to_cards(
        cards,
        dataset_key=dataset_key,
        include_history_comments=include_history_comments,
    )
    if not cards:
        return jsonify(
            {
                "cards": [],
                "mode": model_name,
                "dataset_key": dataset_key,
                "dataset_label": DATASET_CONFIGS[dataset_key]["label"],
                "available_datasets": available_datasets(),
                "model_name": model_name,
                "available_models": recommender.available_models(),
                "page": page_idx + 1,
                "has_next": False,
                "history": session.get("history", []),
                "message": "No more recommendations are available.",
                "include_history_comments": include_history_comments,
            }
        )

    session["page_idx"] = page_idx
    start = (page_idx + 1) * PAGE_SIZE
    return jsonify(
        {
            "cards": cards,
            "mode": model_name,
            "dataset_key": dataset_key,
            "dataset_label": DATASET_CONFIGS[dataset_key]["label"],
            "available_datasets": available_datasets(),
            "model_name": model_name,
            "available_models": recommender.available_models(),
            "page": page_idx + 1,
            "has_next": start < len(rec_ids),
            "history": session.get("history", []),
            "include_history_comments": include_history_comments,
        }
    )


@app.route("/api/comments", methods=["GET"])
def api_comments():
    dataset_key = normalize_dataset_key(request.args.get("dataset_key") or session.get("dataset_key"))
    include_history_comments = parse_bool(request.args.get("include_history_comments"), True)
    item_ids_raw = str(request.args.get("item_ids", "") or "")
    item_ids = unique_keep_order([x.strip() for x in item_ids_raw.split(",") if str(x).strip()])
    comments_by_item = list_comments_for_items(
        dataset_key=dataset_key,
        item_ids=item_ids,
        include_history_comments=include_history_comments,
    )
    return jsonify(
        {
            "dataset_key": dataset_key,
            "item_ids": item_ids,
            "include_history_comments": include_history_comments,
            "comments_by_item": comments_by_item,
        }
    )


@app.route("/api/comment", methods=["POST"])
def api_comment():
    payload = request.get_json(silent=True) or {}
    dataset_key = normalize_dataset_key(payload.get("dataset_key") or session.get("dataset_key"))
    include_history_comments = parse_bool(payload.get("include_history_comments"), True)
    item_id = str(payload.get("item_id", "")).strip()
    comment_text = normalize_comment_text(payload.get("text", ""))

    if not item_id:
        return jsonify({"error": "item_id is required"}), 400
    if not comment_text:
        return jsonify({"error": "comment text is required"}), 400

    try:
        recommender = get_recommender(dataset_key)
    except Exception as err:
        return jsonify({"error": str(err), "available_datasets": available_datasets()}), 400

    simulation_id = str(session.get("simulation_id", "") or "")
    if not simulation_id:
        simulation_id = reset_simulation_stats()

    record = {
        "id": secrets.token_hex(8),
        "dataset_key": dataset_key,
        "item_id": item_id,
        "text": comment_text,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "simulation_id": simulation_id,
    }
    comment_records.append(record)
    comment_records_by_id[record["id"]] = record
    _save_comment_records()

    comments = list_comments_for_items(
        dataset_key=dataset_key,
        item_ids=[item_id],
        include_history_comments=include_history_comments,
    ).get(item_id, [])
    return jsonify(
        {
            "ok": True,
            "dataset_key": dataset_key,
            "item_id": item_id,
            "include_history_comments": include_history_comments,
            "comments": comments,
        }
    )


@app.route("/api/comment/like", methods=["POST"])
def api_comment_like():
    payload = request.get_json(silent=True) or {}
    include_history_comments = parse_bool(payload.get("include_history_comments"), True)
    comment_id = str(payload.get("comment_id", "")).strip()
    if not comment_id:
        return jsonify({"error": "comment_id is required"}), 400

    row = comment_records_by_id.get(comment_id)
    if row is None:
        return jsonify({"error": "Comment not found"}), 404

    like_counts = get_comment_likes(create=True)
    like_counts[comment_id] = int(like_counts.get(comment_id, 0)) + 1

    dataset_key = normalize_dataset_key(row.get("dataset_key"))
    item_id = str(row.get("item_id", "")).strip()
    comments = list_comments_for_items(
        dataset_key=dataset_key,
        item_ids=[item_id],
        include_history_comments=include_history_comments,
    ).get(item_id, [])
    return jsonify(
        {
            "ok": True,
            "dataset_key": dataset_key,
            "item_id": item_id,
            "include_history_comments": include_history_comments,
            "comments": comments,
        }
    )


@app.route("/api/session/end", methods=["POST"])
def api_session_end():
    clear_simulation_state()
    return jsonify({"ok": True})


@app.route("/poster/<dataset_key>/<path:item_id>")
def poster(dataset_key: str, item_id: str):
    try:
        recommender = get_recommender(dataset_key)
    except Exception:
        recommender = None

    if recommender is not None:
        poster_path = recommender.poster_path(item_id)
        if poster_path is not None:
            return send_file(poster_path)
        title = recommender.movie_card(item_id)["title"]
    else:
        title = f"Item {item_id}"

    safe_title = html.escape(title.replace("&", "and"))
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
<stop offset='0%' stop-color='#1f2937'/><stop offset='100%' stop-color='#374151'/></linearGradient></defs>
<rect width='300' height='450' fill='url(#g)'/>
<text x='150' y='220' fill='#f3f4f6' text-anchor='middle' font-size='18' font-family='Arial'>{safe_title[:24]}</text>
<text x='150' y='250' fill='#d1d5db' text-anchor='middle' font-size='14' font-family='Arial'>No Poster</text>
</svg>"""
    return Response(svg, mimetype="image/svg+xml")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "19001"))
    app.run(host="127.0.0.1", port=port, debug=False)
