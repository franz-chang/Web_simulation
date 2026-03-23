import ast
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_json_or_python_dict(line: str) -> dict[str, Any]:
    text = line.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def _load_movielens_sequences(ratings_path: Path, min_seq_len: int) -> dict[int, list[int]]:
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    ratings = ratings.sort_values(["user_id", "timestamp"])
    user_sequences_raw = ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
    return {int(uid): [int(mid) for mid in seq] for uid, seq in user_sequences_raw.items() if len(seq) >= min_seq_len}


def _find_amazon_review_file(dataset_dir: Path) -> Path:
    raw_dir = dataset_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Cannot find raw directory under: {dataset_dir}")

    preferred_names = [f"{dataset_dir.name}.json.gz", "Musical_Instruments.json.gz"]
    for name in preferred_names:
        preferred = raw_dir / name
        if preferred.exists():
            return preferred

    candidates = sorted(
        p for p in raw_dir.glob("*.json.gz") if not p.name.startswith("meta_")
    )
    if not candidates:
        raise FileNotFoundError(f"No review json.gz found under: {raw_dir}")
    return candidates[0]


def _load_amazon_sequences(review_path: Path, min_seq_len: int) -> dict[str, list[str]]:
    user_events: dict[str, list[tuple[int, int, str]]] = {}
    with gzip.open(review_path, "rt", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            row = parse_json_or_python_dict(line)
            user_id = str(row.get("reviewerID", "")).strip()
            item_id = str(row.get("asin", "")).strip()
            if not user_id or not item_id:
                continue

            ts_raw = row.get("unixReviewTime", None)
            try:
                timestamp = int(ts_raw)
            except Exception:
                timestamp = line_idx
            user_events.setdefault(user_id, []).append((timestamp, line_idx, item_id))

    user_sequences: dict[str, list[str]] = {}
    for user_id, events in user_events.items():
        events.sort(key=lambda x: (x[0], x[1]))
        sequence = [item_id for _, _, item_id in events]
        if len(sequence) >= min_seq_len:
            user_sequences[user_id] = sequence
    return user_sequences


def load_user_sequences(dataset_source: str | Path, min_seq_len: int = 5) -> dict[Any, list[Any]]:
    source = Path(dataset_source)

    if source.is_file():
        if source.name == "ratings.dat":
            return _load_movielens_sequences(source, min_seq_len=min_seq_len)
        raise FileNotFoundError(f"Unsupported source file: {source}")

    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")

    ratings_path = source / "ratings.dat"
    if ratings_path.exists():
        return _load_movielens_sequences(ratings_path, min_seq_len=min_seq_len)

    review_path = _find_amazon_review_file(source)
    return _load_amazon_sequences(review_path, min_seq_len=min_seq_len)
