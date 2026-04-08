from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bert4rec import BERT4Rec, BERT4RecTrainDataset
from dataset_utils import load_user_sequences


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate(
    model: BERT4Rec,
    user_train: dict[int | str, list[int]],
    user_target: dict[int | str, int],
    max_len: int,
    device: torch.device,
    ks: list[int],
) -> dict[str, float]:
    model.eval()
    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    valid_users = 0

    eval_rows: list[tuple[int | str, list[int], int]] = []
    for uid, history in user_train.items():
        target = user_target.get(uid, 0)
        if target <= 0 or not history:
            continue
        eval_rows.append((uid, history, target))

    batch_size = 256
    for start in range(0, len(eval_rows), batch_size):
        chunk = eval_rows[start : start + batch_size]
        input_batch = np.zeros((len(chunk), max_len), dtype=np.int64)
        targets = []
        histories = []

        for row_idx, (_, history, target) in enumerate(chunk):
            recent = history[-(max_len - 1) :]
            if recent:
                input_batch[row_idx, -(len(recent) + 1) : -1] = np.array(recent, dtype=np.int64)
            input_batch[row_idx, -1] = model.mask_token
            targets.append(int(target))
            histories.append(history)

        seq_t = torch.LongTensor(input_batch).to(device)
        scores_batch = model.predict_scores(seq_t).detach().cpu()

        for row_idx, scores in enumerate(scores_batch):
            target = targets[row_idx]
            history = histories[row_idx]
            seen = set(history)
            if target in seen:
                seen.remove(target)
            if seen:
                scores[list(seen)] = -1e9

            target_score = float(scores[target].item())
            rank = int((scores > target_score).sum().item()) + 1
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
    set_seed(args.seed)
    ks = parse_ks(args.eval_ks)

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading interactions from: {dataset_dir}")
    user_sequences = load_user_sequences(dataset_dir, min_seq_len=args.min_seq_len)
    remapped, item2idx, idx2item = remap_items(user_sequences)
    user_train, user_valid, user_test = split_sequences(remapped)
    user_train = {u: s for u, s in user_train.items() if len(s) >= 2}
    user_valid = {u: user_valid[u] for u in user_train if u in user_valid}
    user_test = {u: user_test[u] for u in user_train if u in user_test}

    num_items = len(idx2item) - 1
    print(f"Train users: {len(user_train)}, items: {num_items}, eval_ks: {ks}")

    device = pick_device()
    print(f"Using device: {device}")
    model = BERT4Rec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    primary_k = ks[0]
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_dataset = BERT4RecTrainDataset(
            user_train=user_train,
            max_len=args.max_len,
            num_items=num_items,
            mask_prob=args.mask_prob,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        total_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = model.calculate_loss(input_ids, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, user_train, user_valid, args.max_len, device, ks)
        history.append({"epoch": epoch, "loss": avg_loss, **metrics})

        metric_text = " ".join(
            [f"HR@{k}={metrics[f'HR@{k}']:.4f} NDCG@{k}={metrics[f'NDCG@{k}']:.4f}" for k in ks]
        )
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f} - valid {metric_text}")

        score = metrics[f"HR@{primary_k}"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, user_train, user_test, args.max_len, device, ks)
    test_text = " ".join(
        [f"HR@{k}={test_metrics[f'HR@{k}']:.4f} NDCG@{k}={test_metrics[f'NDCG@{k}']:.4f}" for k in ks]
    )
    print(f"Best epoch by HR@{primary_k}: {best_epoch} ({best_metric:.4f})")
    print(f"Test metrics (best checkpoint): {test_text}")

    artifact = {
        "model_state_dict": best_state,
        "config": {
            "num_items": num_items,
            "max_len": args.max_len,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "mask_prob": args.mask_prob,
        },
        "best_epoch": best_epoch,
        "best_valid_hr": float(best_metric),
        "valid_history": history,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "item2idx": item2idx,
        "idx2item": idx2item,
    }
    torch.save(artifact, output_path)
    print(f"Saved artifact to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT4Rec on MovieLens or Amazon review sequences")
    default_dataset_dir = Path(__file__).resolve().parent.parent / "WebSim_Dataset" / "MM-ML-1M-main"
    parser.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir))
    parser.add_argument("--output-model", type=str, default="artifacts/bert4rec_ml1m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mask-prob", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--eval-ks", type=str, default="10,20")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
