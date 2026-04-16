import ast
import gzip
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import torch

from bert4rec import BERT4Rec
from gru4rec import GRU4Rec
from multvae import MultiVAE
from sasrec import SASRec


def normalize_item_id(item_id: Any) -> str:
    return str(item_id).strip()


def candidate_item_keys(item_id: str) -> list[Any]:
    candidates: list[Any] = [item_id]
    if item_id.isdigit():
        try:
            candidates.append(int(item_id))
        except Exception:
            pass
    return candidates


class MovieCatalog:
    def __init__(
        self,
        dataset_key: str,
        dataset_dir: str,
        item_whitelist: set[str] | None = None,
        random_seed: int = 42,
    ):
        self.dataset_key = dataset_key
        self.dataset_dir = Path(dataset_dir)
        self.poster_dir = self.dataset_dir / "posters"
        self.image_dir = self.dataset_dir / "images"
        self.item_whitelist = item_whitelist
        self._rng = random.Random(int(random_seed))
        self._last_random_ids: tuple[str, ...] | None = None
        self.movies = self._load_movies()
        self.movie_ids = list(self.movies.keys())

    @staticmethod
    def _parse_json_or_python_dict(line: str) -> dict[str, Any]:
        text = line.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return ast.literal_eval(text)

    @staticmethod
    def _safe_text(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text.lower() == "nan" else text

    @staticmethod
    def _pick_description(row: dict[str, Any]) -> str:
        description = row.get("description", [])
        if isinstance(description, list):
            for part in description:
                text = str(part).strip()
                if text:
                    return text
        elif isinstance(description, str) and description.strip():
            return description.strip()

        category = row.get("category", [])
        if isinstance(category, list) and category:
            parts = [str(x).strip() for x in category if str(x).strip()]
            if parts:
                return " / ".join(parts[:3])

        brand = str(row.get("brand", "")).strip()
        if brand:
            return brand
        return "No description"

    def _load_movielens(self) -> dict[str, dict[str, str]]:
        details_path = self.dataset_dir / "movies_details_clean.csv"
        base_path = self.dataset_dir / "movies.dat"
        movies: dict[str, dict[str, str]] = {}

        if details_path.exists():
            details = pd.read_csv(details_path)
            for _, row in details.iterrows():
                raw_id = row.get("Id")
                if pd.isna(raw_id):
                    continue
                movie_id = normalize_item_id(int(raw_id) if str(raw_id).isdigit() else raw_id)
                if self.item_whitelist is not None and movie_id not in self.item_whitelist:
                    continue

                title = self._safe_text(row.get("Title", "")) or f"Movie {movie_id}"
                overview = self._safe_text(row.get("Overview", ""))
                genres = self._safe_text(row.get("Genres", ""))
                poster = self._safe_text(row.get("Poster Path", ""))
                description = overview or genres or "No description"
                movies[movie_id] = {
                    "title": title,
                    "description": description,
                    "poster": poster,
                }

        if base_path.exists():
            with base_path.open("r", encoding="latin-1") as fin:
                for line in fin:
                    parts = line.strip().split("::")
                    if len(parts) < 3:
                        continue
                    movie_id = normalize_item_id(parts[0])
                    if self.item_whitelist is not None and movie_id not in self.item_whitelist:
                        continue
                    if movie_id not in movies:
                        movies[movie_id] = {
                            "title": parts[1],
                            "description": parts[2].replace("|", ", "),
                            "poster": "",
                        }
        return movies

    def _find_amazon_meta_path(self) -> Path | None:
        raw_dir = self.dataset_dir / "raw"
        name_variants: list[str] = []
        for name in (
            self.dataset_dir.name,
            self.dataset_dir.name.replace("-", "_"),
            self.dataset_dir.name.replace(" ", "_"),
        ):
            if name and name not in name_variants:
                name_variants.append(name)
        preferred_names = [f"meta_{name}.json.gz" for name in name_variants]
        for name in preferred_names:
            preferred = raw_dir / name
            if preferred.exists():
                return preferred
        matches = sorted(raw_dir.glob("meta_*.json.gz")) if raw_dir.exists() else []
        return matches[0] if matches else None

    def _load_amazon(self) -> dict[str, dict[str, str]]:
        movies: dict[str, dict[str, str]] = {}
        meta_path = self._find_amazon_meta_path()

        if meta_path is not None:
            with gzip.open(meta_path, "rt", encoding="utf-8") as fin:
                for line in fin:
                    row = self._parse_json_or_python_dict(line)
                    item_id = normalize_item_id(row.get("asin", ""))
                    if not item_id:
                        continue
                    if self.item_whitelist is not None and item_id not in self.item_whitelist:
                        continue

                    title = self._safe_text(row.get("title", "")) or f"Item {item_id}"
                    description = self._pick_description(row)
                    poster_name = f"{item_id}.jpg" if (self.image_dir / f"{item_id}.jpg").exists() else ""
                    movies[item_id] = {
                        "title": title,
                        "description": description,
                        "poster": poster_name,
                    }

        if self.item_whitelist:
            for item_id in self.item_whitelist:
                if item_id in movies:
                    continue
                poster_name = f"{item_id}.jpg" if (self.image_dir / f"{item_id}.jpg").exists() else ""
                movies[item_id] = {
                    "title": f"Item {item_id}",
                    "description": "No description",
                    "poster": poster_name,
                }
        return movies

    def _load_movies(self) -> dict[str, dict[str, str]]:
        has_ml1m = (self.dataset_dir / "movies.dat").exists() or (self.dataset_dir / "movies_details_clean.csv").exists()
        if has_ml1m:
            return self._load_movielens()
        return self._load_amazon()

    def random_movie_ids(self, n: int, avoid_last_same: bool = False) -> list[str]:
        if len(self.movie_ids) <= n:
            ids = self.movie_ids.copy()
            if avoid_last_same:
                self._last_random_ids = tuple(ids)
            return ids

        ids = self._rng.sample(self.movie_ids, n)
        if avoid_last_same and self._last_random_ids is not None and tuple(ids) == self._last_random_ids:
            for _ in range(5):
                candidate = self._rng.sample(self.movie_ids, n)
                if tuple(candidate) != self._last_random_ids:
                    ids = candidate
                    break
        if avoid_last_same:
            self._last_random_ids = tuple(ids)
        return ids

    def has_movie(self, movie_id: str) -> bool:
        return movie_id in self.movies

    def movie_card(self, movie_id: str) -> dict[str, str]:
        info = self.movies.get(movie_id, {})
        quoted = quote(movie_id, safe="")
        return {
            "id": movie_id,
            "title": info.get("title", f"Item {movie_id}"),
            "description": info.get("description", "No description"),
            "image": f"/poster/{self.dataset_key}/{quoted}",
        }

    def poster_path(self, movie_id: str) -> Path | None:
        info = self.movies.get(movie_id, {})
        poster_name = self._safe_text(info.get("poster", ""))
        poster_candidates: list[str] = []

        # Prefer explicit poster name from metadata first.
        if poster_name:
            poster_candidates.append(Path(poster_name).name)

        # Backward-compatible guesses for legacy datasets.
        if movie_id:
            poster_candidates.extend([f"{movie_id}.jpg", f"{movie_id}.jpeg", f"{movie_id}.png"])

        # De-duplicate while keeping order.
        seen = set()
        ordered_candidates: list[str] = []
        for name in poster_candidates:
            if name and name not in seen:
                seen.add(name)
                ordered_candidates.append(name)

        # New Amazon_MM_2018 uses posters/, legacy Amazon used images/.
        for base_dir in (self.poster_dir, self.image_dir, self.dataset_dir):
            for name in ordered_candidates:
                path = base_dir / name
                if path.exists():
                    return path

        # Fallback: some posters are named like "{item_id}_hash.jpg".
        for base_dir in (self.poster_dir, self.image_dir):
            matches = sorted(base_dir.glob(f"{movie_id}_*.jpg"))
            if matches:
                return matches[0]

        return None


class _BaseEngine:
    def __init__(self) -> None:
        self.item2idx: dict[Any, int] = {}
        self.idx2item: list[Any] = []

    def _to_item_index(self, item_id: str) -> int | None:
        for candidate in candidate_item_keys(item_id):
            if candidate in self.item2idx:
                return int(self.item2idx[candidate])
        return None

    def _to_item_id(self, raw_item: Any) -> str:
        return normalize_item_id(raw_item)


class SASRecEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        config = artifact["config"]
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]
        self.max_len = int(config["max_len"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SASRec(
            num_items=int(config["num_items"]),
            max_len=int(config["max_len"]),
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        mapped_history = []
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                mapped_history.append(idx)
        if not mapped_history:
            return []

        seq = np.zeros(self.max_len, dtype=np.int64)
        recent = mapped_history[-self.max_len :]
        seq[-len(recent) :] = recent

        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scores = self.model.predict_scores(seq_tensor)[0]
        for item_idx in set(mapped_history):
            scores[item_idx] = -1e9

        k = min(topk, len(self.idx2item) - 1)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        rec_movie_ids = []
        for item_idx in top_indices:
            if item_idx <= 0:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
        return rec_movie_ids


class LightGCNEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        config = artifact["config"]
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]
        self.max_history = int(config.get("max_history", 50))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_embeddings = artifact["final_item_embeddings"].to(self.device)

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        mapped_history = []
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                mapped_history.append(idx)
        if not mapped_history:
            return []
        mapped_history = mapped_history[-self.max_history :]

        hist_tensor = torch.LongTensor(mapped_history).to(self.device)
        with torch.no_grad():
            user_emb = self.item_embeddings[hist_tensor].mean(dim=0, keepdim=True)
            scores = torch.matmul(user_emb, self.item_embeddings.transpose(0, 1))[0]
            scores[0] = -1e9
            for item_idx in set(mapped_history):
                scores[item_idx] = -1e9

        k = min(topk, len(self.idx2item) - 1)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        rec_movie_ids = []
        for item_idx in top_indices:
            if item_idx <= 0:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
        return rec_movie_ids


class MultVAEEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        config = artifact["config"]
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]
        self.max_history = int(config.get("max_history", 50))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiVAE(
            input_dim=int(config["input_dim"]),
            p_dims=[int(config["hidden_dim"]), int(config["latent_dim"])],
            dropout=float(config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        mapped_history = []
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                mapped_history.append(idx)
        if not mapped_history:
            return []
        mapped_history = mapped_history[-self.max_history :]

        input_dim = self.model.input_dim
        x = torch.zeros((1, input_dim), dtype=torch.float32, device=self.device)
        x[0, torch.LongTensor(mapped_history).to(self.device)] = 1.0

        with torch.no_grad():
            scores = self.model.recommend_logits(x)[0]
        scores[0] = -1e9
        for item_idx in set(mapped_history):
            scores[item_idx] = -1e9

        k = min(topk, len(self.idx2item) - 1)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        rec_movie_ids = []
        for item_idx in top_indices:
            if item_idx <= 0:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
        return rec_movie_ids


class PopRecEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]

        ranked_indices = artifact.get("ranked_item_indices", [])
        if ranked_indices:
            self.ranked_indices = [int(x) for x in ranked_indices if int(x) > 0]
            return

        counts = artifact.get("item_popularity", None)
        if counts is None:
            self.ranked_indices = []
            return

        if isinstance(counts, torch.Tensor):
            counts_tensor = counts
        else:
            counts_tensor = torch.tensor(counts, dtype=torch.float32)
        ranking = torch.argsort(counts_tensor, descending=True).tolist()
        self.ranked_indices = [int(x) for x in ranking if int(x) > 0]

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        seen_indices = set()
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                seen_indices.add(idx)

        rec_movie_ids = []
        for item_idx in self.ranked_indices:
            if item_idx <= 0 or item_idx in seen_indices:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
            if len(rec_movie_ids) >= topk:
                break
        return rec_movie_ids


class BPRMFEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        config = artifact["config"]
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]
        self.max_history = int(config.get("max_history", 50))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_embeddings = artifact["final_item_embeddings"].to(self.device)

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        mapped_history = []
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                mapped_history.append(idx)
        if not mapped_history:
            return []
        mapped_history = mapped_history[-self.max_history :]

        hist_tensor = torch.LongTensor(mapped_history).to(self.device)
        with torch.no_grad():
            user_emb = self.item_embeddings[hist_tensor].mean(dim=0, keepdim=True)
            scores = torch.matmul(user_emb, self.item_embeddings.transpose(0, 1))[0]
            scores[0] = -1e9
            for item_idx in set(mapped_history):
                scores[item_idx] = -1e9

        k = min(topk, len(self.idx2item) - 1)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        rec_movie_ids = []
        for item_idx in top_indices:
            if item_idx <= 0:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
        return rec_movie_ids


class GRU4RecEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        config = artifact["config"]
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]
        self.max_len = int(config["max_len"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GRU4Rec(
            num_items=int(config["num_items"]),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        mapped_history = []
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                mapped_history.append(idx)
        if not mapped_history:
            return []

        seq = np.zeros(self.max_len, dtype=np.int64)
        recent = mapped_history[-self.max_len :]
        seq[-len(recent) :] = recent
        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.model.predict_scores(seq_tensor)[0]
        for item_idx in set(mapped_history):
            scores[item_idx] = -1e9

        k = min(topk, len(self.idx2item) - 1)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        rec_movie_ids = []
        for item_idx in top_indices:
            if item_idx <= 0:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
        return rec_movie_ids


class BERT4RecEngine(_BaseEngine):
    def __init__(self, artifact_path: str):
        super().__init__()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        config = artifact["config"]
        self.item2idx = artifact["item2idx"]
        self.idx2item = artifact["idx2item"]
        self.max_len = int(config["max_len"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BERT4Rec(
            num_items=int(config["num_items"]),
            max_len=int(config["max_len"]),
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def recommend(self, history_movie_ids: list[str], topk: int) -> list[str]:
        mapped_history = []
        for mid in history_movie_ids:
            idx = self._to_item_index(mid)
            if idx is not None:
                mapped_history.append(idx)
        if not mapped_history:
            return []

        # Reserve one slot for [MASK] at the end.
        recent = mapped_history[-(self.max_len - 1) :]
        input_ids = np.zeros(self.max_len, dtype=np.int64)
        input_ids[-(len(recent) + 1) : -1] = recent
        input_ids[-1] = self.model.mask_token
        seq_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores = self.model.predict_scores(seq_tensor)[0]
        for item_idx in set(mapped_history):
            scores[item_idx] = -1e9

        k = min(topk, len(self.idx2item) - 1)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        rec_movie_ids = []
        for item_idx in top_indices:
            if item_idx <= 0:
                continue
            rec_movie_ids.append(self._to_item_id(self.idx2item[item_idx]))
        return rec_movie_ids


class MovieRecommender:
    def __init__(
        self,
        dataset_key: str,
        dataset_dir: str,
        random_seed: int = 42,
        sasrec_artifact_path: str | None = None,
        lightgcn_artifact_path: str | None = None,
        multvae_artifact_path: str | None = None,
        poprec_artifact_path: str | None = None,
        bprmf_artifact_path: str | None = None,
        gru4rec_artifact_path: str | None = None,
        bert4rec_artifact_path: str | None = None,
    ):
        self.dataset_key = dataset_key
        self.dataset_dir = str(dataset_dir)
        self.engines: dict[str, object] = {}

        if sasrec_artifact_path and Path(sasrec_artifact_path).exists():
            self.engines["sasrec"] = SASRecEngine(sasrec_artifact_path)
        if lightgcn_artifact_path and Path(lightgcn_artifact_path).exists():
            self.engines["lightgcn"] = LightGCNEngine(lightgcn_artifact_path)
        if multvae_artifact_path and Path(multvae_artifact_path).exists():
            self.engines["multvae"] = MultVAEEngine(multvae_artifact_path)
        if poprec_artifact_path and Path(poprec_artifact_path).exists():
            self.engines["poprec"] = PopRecEngine(poprec_artifact_path)
        if bprmf_artifact_path and Path(bprmf_artifact_path).exists():
            self.engines["bprmf"] = BPRMFEngine(bprmf_artifact_path)
        if gru4rec_artifact_path and Path(gru4rec_artifact_path).exists():
            self.engines["gru4rec"] = GRU4RecEngine(gru4rec_artifact_path)
        if bert4rec_artifact_path and Path(bert4rec_artifact_path).exists():
            self.engines["bert4rec"] = BERT4RecEngine(bert4rec_artifact_path)

        item_whitelist: set[str] | None = None
        if self.engines:
            item_whitelist = set()
            for engine in self.engines.values():
                for raw_item_id in engine.item2idx.keys():
                    item_id = normalize_item_id(raw_item_id)
                    if not item_id or item_id == "0":
                        continue
                    item_whitelist.add(item_id)

        self.catalog = MovieCatalog(
            dataset_key=dataset_key,
            dataset_dir=dataset_dir,
            item_whitelist=item_whitelist,
            random_seed=random_seed,
        )

    def available_models(self) -> list[str]:
        return sorted(self.engines.keys())

    def default_model(self) -> str:
        if "sasrec" in self.engines:
            return "sasrec"
        models = self.available_models()
        return models[0] if models else ""

    def normalize_model_name(self, model_name: str | None) -> str:
        if not model_name:
            return self.default_model()
        name = str(model_name).strip().lower()
        if name in self.engines:
            return name
        return self.default_model()

    def random_movie_ids(self, n: int, avoid_last_same: bool = False) -> list[str]:
        return self.catalog.random_movie_ids(n, avoid_last_same=avoid_last_same)

    @staticmethod
    def _unique_keep_order(movie_ids: list[str]) -> list[str]:
        seen = set()
        unique_ids = []
        for mid in movie_ids:
            if mid in seen:
                continue
            seen.add(mid)
            unique_ids.append(mid)
        return unique_ids

    def recommend_ids(self, history_movie_ids: list[str], model_name: str, topk: int = 200) -> list[str]:
        model_name = self.normalize_model_name(model_name)
        engine = self.engines.get(model_name)
        if engine is None:
            return self.random_movie_ids(min(topk, len(self.catalog.movie_ids)))

        rec_ids = engine.recommend(history_movie_ids, topk=topk)
        rec_ids = [mid for mid in rec_ids if self.catalog.has_movie(mid)]
        rec_ids = self._unique_keep_order(rec_ids)
        if not rec_ids:
            return self.random_movie_ids(min(topk, len(self.catalog.movie_ids)))
        return rec_ids

    def movie_card(self, movie_id: str) -> dict[str, str]:
        return self.catalog.movie_card(movie_id)

    def poster_path(self, movie_id: str) -> Path | None:
        return self.catalog.poster_path(movie_id)
