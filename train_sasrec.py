import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_utils import load_user_sequences as load_user_sequences_generic
from sasrec import SASRec, SASRecTrainDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_user_sequences(dataset_source: str | Path, min_seq_len: int = 5) -> dict:
    return load_user_sequences_generic(dataset_source, min_seq_len=min_seq_len)


def remap_items(user_sequences: dict[int, list[int]]) -> tuple[dict[int, list[int]], dict[int, int], list[int]]:
    unique_items = sorted({item for seq in user_sequences.values() for item in seq})
    item2idx = {item: idx for idx, item in enumerate(unique_items, start=1)}
    idx2item = [0] + unique_items
    remapped = {
        uid: [item2idx[item] for item in seq]
        for uid, seq in user_sequences.items()
    }
    return remapped, item2idx, idx2item


def split_sequences(
    user_sequences: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    user_train: dict[int, list[int]] = {}
    user_valid: dict[int, int] = {}
    user_test: dict[int, int] = {}
    for uid, sequence in user_sequences.items():
        if len(sequence) < 3:
            continue
        user_train[uid] = sequence[:-2]
        user_valid[uid] = sequence[-2]
        user_test[uid] = sequence[-1]
    return user_train, user_valid, user_test


@torch.no_grad()
def evaluate(
    model: SASRec,
    user_train: dict[int, list[int]],
    user_target: dict[int, int],
    max_len: int,
    device: torch.device,
    ks: list[int],
) -> dict[str, float]:
    model.eval()
    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    valid_users = 0

    for uid, history in user_train.items():
        target = user_target.get(uid, 0)
        if target <= 0 or not history:
            continue

        seq = np.zeros(max_len, dtype=np.int64)
        recent = history[-max_len:]
        seq[-len(recent) :] = recent
        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)

        scores = model.predict_scores(seq_tensor)[0].detach().cpu()
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


def parse_ks(raw: str) -> list[int]:
    ks = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        ks.append(int(x))
    ks = sorted(set(ks))
    if not ks:
        raise ValueError("eval_ks cannot be empty")
    return ks


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_ks = parse_ks(args.eval_ks)

    print(f"Loading interactions from: {dataset_dir}")
    user_sequences = load_user_sequences(dataset_dir, min_seq_len=args.min_seq_len)
    remapped_sequences, item2idx, idx2item = remap_items(user_sequences)
    user_train, user_valid, user_test = split_sequences(remapped_sequences)
    user_train = {uid: seq for uid, seq in user_train.items() if len(seq) >= 2}
    user_valid = {uid: user_valid[uid] for uid in user_train if uid in user_valid}
    user_test = {uid: user_test[uid] for uid in user_train if uid in user_test}

    num_users = len(user_train)
    num_items = len(idx2item) - 1
    print(f"Train users: {num_users}, items: {num_items}, eval_ks: {eval_ks}")

    train_dataset = SASRecTrainDataset(user_train, num_items=num_items, max_len=args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    primary_k = eval_ks[0]
    best_metric = -1.0
    best_epoch = 0
    best_state_dict = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for seq, pos, neg in train_loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            loss = model.calculate_loss(seq, pos, neg)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, user_train, user_valid, args.max_len, device, eval_ks)
        history.append({"epoch": epoch, "loss": avg_loss, **metrics})

        metric_text = " ".join(
            [f"HR@{k}={metrics[f'HR@{k}']:.4f} NDCG@{k}={metrics[f'NDCG@{k}']:.4f}" for k in eval_ks]
        )
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f} - valid {metric_text}")

        score = metrics[f"HR@{primary_k}"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate(model, user_train, user_test, args.max_len, device, eval_ks)
    test_text = " ".join(
        [f"HR@{k}={test_metrics[f'HR@{k}']:.4f} NDCG@{k}={test_metrics[f'NDCG@{k}']:.4f}" for k in eval_ks]
    )
    print(f"Best epoch by HR@{primary_k}: {best_epoch} ({best_metric:.4f})")
    print(f"Test metrics (best checkpoint): {test_text}")

    artifact = {
        "model_state_dict": best_state_dict,
        "config": {
            "max_len": args.max_len,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "num_items": num_items,
        },
        "best_epoch": best_epoch,
        "best_valid_hr": best_metric,
        "valid_history": history,
        "test_metrics": test_metrics,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }
    torch.save(artifact, output_path)
    print(f"Saved artifact to: {output_path}")


def parse_args() -> argparse.Namespace:
    default_dataset_dir = Path(__file__).resolve().parent.parent / "WebSim_Dataset" / "MM-ML-1M-main"
    parser = argparse.ArgumentParser(description="Train SASRec on MovieLens or Amazon review sequences")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(default_dataset_dir),
        help="Dataset directory (MovieLens with ratings.dat, or Amazon with raw/*.json.gz)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="artifacts/sasrec_ml1m.pt",
        help="Output model artifact path",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--eval-ks", type=str, default="10,20")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
