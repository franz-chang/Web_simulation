from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset_utils import load_user_sequences
from poprec import build_popularity_counts


def remap_items(user_sequences: dict[int | str, list[int | str]]) -> tuple[dict[int | str, list[int]], dict[int | str, int], list[int | str]]:
    unique_items = sorted({item for seq in user_sequences.values() for item in seq}, key=lambda x: str(x))
    item2idx = {item: idx for idx, item in enumerate(unique_items, start=1)}
    idx2item = [0] + unique_items
    remapped = {
        uid: [item2idx[item] for item in seq if item in item2idx]
        for uid, seq in user_sequences.items()
    }
    return remapped, item2idx, idx2item


def split_sequences(
    user_sequences: dict[int | str, list[int]],
) -> tuple[dict[int | str, list[int]], dict[int | str, int], dict[int | str, int]]:
    train, valid, test = {}, {}, {}
    for uid, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        train[uid] = seq[:-2]
        valid[uid] = seq[-2]
        test[uid] = seq[-1]
    return train, valid, test


def parse_ks(raw: str) -> list[int]:
    ks = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not ks:
        raise ValueError("eval_ks cannot be empty")
    return ks


def evaluate(
    ranked_items: list[int],
    user_train: dict[int | str, list[int]],
    user_target: dict[int | str, int],
    ks: list[int],
) -> dict[str, float]:
    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    valid_users = 0

    for uid, target in user_target.items():
        history = set(user_train.get(uid, []))
        if not history:
            continue

        rank = None
        pos = 0
        for item in ranked_items:
            if item in history and item != target:
                continue
            pos += 1
            if item == target:
                rank = pos
                break

        if rank is None:
            continue

        valid_users += 1
        for k in ks:
            if rank <= k:
                hit[k] += 1.0
                ndcg[k] += 1.0 / np.log2(rank + 1.0)

    if valid_users == 0:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}

    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"] = hit[k] / valid_users
        metrics[f"NDCG@{k}"] = ndcg[k] / valid_users
    return metrics


def train(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = parse_ks(args.eval_ks)

    print(f"Loading interactions from: {dataset_dir}")
    user_sequences = load_user_sequences(dataset_dir, min_seq_len=args.min_seq_len)
    remapped, item2idx, idx2item = remap_items(user_sequences)
    user_train, user_valid, user_test = split_sequences(remapped)
    user_train = {u: s for u, s in user_train.items() if len(s) >= 2}
    user_valid = {u: user_valid[u] for u in user_train if u in user_valid}
    user_test = {u: user_test[u] for u in user_train if u in user_test}

    counts, ranked_items = build_popularity_counts(user_train)
    print(f"Train users: {len(user_train)}, items: {len(idx2item) - 1}, eval_ks: {ks}")

    valid_metrics = evaluate(ranked_items, user_train, user_valid, ks)
    test_metrics = evaluate(ranked_items, user_train, user_test, ks)

    valid_text = " ".join([f"HR@{k}={valid_metrics[f'HR@{k}']:.4f} NDCG@{k}={valid_metrics[f'NDCG@{k}']:.4f}" for k in ks])
    test_text = " ".join([f"HR@{k}={test_metrics[f'HR@{k}']:.4f} NDCG@{k}={test_metrics[f'NDCG@{k}']:.4f}" for k in ks])
    print(f"Valid metrics: {valid_text}")
    print(f"Test metrics: {test_text}")

    artifact = {
        "config": {
            "num_items": len(idx2item) - 1,
        },
        "item2idx": item2idx,
        "idx2item": idx2item,
        "item_popularity": counts,
        "ranked_item_indices": ranked_items,
        "valid_metrics": {k: float(v) for k, v in valid_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    torch.save(artifact, output_path)
    print(f"Saved artifact to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PopRec baseline on MovieLens or Amazon review sequences")
    default_dataset_dir = Path(__file__).resolve().parent.parent / "WebSim_Dataset" / "MM-ML-1M-main"
    parser.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir))
    parser.add_argument("--output-model", type=str, default="artifacts/poprec_ml1m.pt")
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--eval-ks", type=str, default="10,20")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
