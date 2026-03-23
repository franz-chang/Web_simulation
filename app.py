import html
import os
import secrets
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


DEFAULT_ML1M_DIR = first_existing_path(
    [
        BASE_DIR.parent / "WebSim_Dataset" / "MM-ML-1M-main",
        BASE_DIR / "Dataset" / "MM-ML-1M-main",
        Path("/Users/chongzhang/ml1m_sasrec_web/Dataset/MM-ML-1M-main"),
    ]
)
DEFAULT_AMAZON_MI_DIR = first_existing_path(
    [
        BASE_DIR.parent / "WebSim_Dataset" / "amazon_v2" / "Musical_Instruments",
        BASE_DIR / "Dataset" / "amazon_v2" / "Musical_Instruments",
        Path("/Users/chongzhang/ml1m_sasrec_web/Dataset/amazon_v2/Musical_Instruments"),
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
        },
    },
    "amazon_mi": {
        "label": "Amazon Musical Instruments",
        "dataset_dir": os.getenv("AMAZON_MI_DATASET_DIR", str(DEFAULT_AMAZON_MI_DIR)),
        "models": {
            "sasrec": first_env_or_default(
                ["SASREC_AMAZON_MI_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "sasrec_amazon_mi.pt"),
            ),
            "lightgcn": first_env_or_default(
                ["LIGHTGCN_AMAZON_MI_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "lightgcn_amazon_mi.pt"),
            ),
            "multvae": first_env_or_default(
                ["MULTVAE_AMAZON_MI_MODEL_PATH"],
                str(BASE_DIR / "artifacts" / "multvae_amazon_mi.pt"),
            ),
        },
    },
}

DEFAULT_DATASET_KEY = os.getenv("DEFAULT_DATASET_KEY", "ml1m")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

recommenders: dict[str, MovieRecommender] = {}


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
        sasrec_artifact_path=models.get("sasrec"),
        lightgcn_artifact_path=models.get("lightgcn"),
        multvae_artifact_path=models.get("multvae"),
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


def paginate_cards(recommender: MovieRecommender, item_ids: list[str], page_idx: int) -> list[dict]:
    item_ids = unique_keep_order(item_ids)
    start = page_idx * PAGE_SIZE
    end = start + PAGE_SIZE
    page_ids = item_ids[start:end]
    return [recommender.movie_card(mid) for mid in page_ids]


@app.route("/")
def index():
    return render_template("index.html")


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
    try:
        recommender = get_recommender(dataset_key)
    except Exception as err:
        return jsonify({"error": str(err), "available_datasets": available_datasets()}), 400

    model_name = recommender.normalize_model_name(request.args.get("model_name"))
    cards = [recommender.movie_card(mid) for mid in recommender.random_movie_ids(PAGE_SIZE)]

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
        }
    )


@app.route("/api/select", methods=["POST"])
def api_select():
    payload = request.get_json(silent=True) or {}

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
    else:
        history = [str(x) for x in session.get("history", [])]

    if movie_id:
        history.append(movie_id)
        history = history[-20:]

    rec_ids = recommender.recommend_ids(history, model_name=model_name, topk=TOPK)
    rec_ids = unique_keep_order(rec_ids)
    cards = paginate_cards(recommender, rec_ids, 0)

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
        }
    )


@app.route("/api/next", methods=["POST"])
def api_next():
    payload = request.get_json(silent=True) or {}
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
            }
        )

    page_idx = int(session.get("page_idx", 0)) + 1
    cards = paginate_cards(recommender, rec_ids, page_idx)
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
                "message": "没有更多推荐了。",
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
        }
    )


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
