import argparse
import random
from pathlib import Path

import numpy as np
import torch

from multvae import MultiVAE
from train_sasrec import load_user_sequences


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remap_items(user_sequences: dict[int, list[int]]) -> tuple[dict[int, list[int]], dict[int, int], list[int]]:
    unique_items = sorted({item for seq in user_sequences.values() for item in seq})
    item2idx = {item: i for i, item in enumerate(unique_items, start=1)}
    idx2item = [0] + unique_items
    remapped = {u: [item2idx[m] for m in seq if m in item2idx] for u, seq in user_sequences.items()}
    return remapped, item2idx, idx2item


def split_sequences(
    user_sequences: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    train = {}
    valid = {}
    test = {}
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


def build_batch_matrix(
    user_ids: list[int],
    user_hist: dict[int, list[int]],
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.zeros((len(user_ids), input_dim), dtype=torch.float32, device=device)
    for i, uid in enumerate(user_ids):
        items = user_hist.get(uid, [])
        if items:
            x[i, torch.LongTensor(items).to(device)] = 1.0
    return x


@torch.no_grad()
def evaluate(
    model: MultiVAE,
    user_train: dict[int, list[int]],
    user_target: dict[int, int],
    input_dim: int,
    device: torch.device,
    ks: list[int],
    max_history: int,
) -> dict[str, float]:
    model.eval()
    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    valid_users = 0

    for uid, target in user_target.items():
        history = user_train.get(uid, [])
        if not history:
            continue
        history = history[-max_history:]
        x = torch.zeros((1, input_dim), dtype=torch.float32, device=device)
        x[0, torch.LongTensor(history).to(device)] = 1.0
        logits = model.recommend_logits(x)[0].detach().cpu()
        logits[0] = -1e9
        for item in set(history):
            if item != target:
                logits[item] = -1e9
        target_score = float(logits[target].item())
        rank = int((logits > target_score).sum().item()) + 1
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

    users = sorted(user_train.keys())
    input_dim = len(idx2item)
    print(f"Train users: {len(users)}, items: {input_dim - 1}, eval_ks: {ks}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiVAE(
        input_dim=input_dim,
        p_dims=[args.hidden_dim, args.latent_dim],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_epoch = 0
    best_state = None
    history = []
    total_steps = 0
    primary_k = ks[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(users)
        total_loss = 0.0
        steps = 0
        for i in range(0, len(users), args.batch_size):
            batch_users = users[i : i + args.batch_size]
            x = build_batch_matrix(batch_users, user_train, input_dim, device)
            anneal = min(args.anneal_cap, total_steps / max(1.0, float(args.total_anneal_steps)))
            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            loss = model.loss_fn(logits, x, mu, logvar, anneal)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
            total_steps += 1

        avg_loss = total_loss / max(steps, 1)
        metrics = evaluate(
            model=model,
            user_train=user_train,
            user_target=user_valid,
            input_dim=input_dim,
            device=device,
            ks=ks,
            max_history=args.max_history,
        )
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

    test_metrics = evaluate(
        model=model,
        user_train=user_train,
        user_target=user_test,
        input_dim=input_dim,
        device=device,
        ks=ks,
        max_history=args.max_history,
    )
    test_text = " ".join(
        [f"HR@{k}={test_metrics[f'HR@{k}']:.4f} NDCG@{k}={test_metrics[f'NDCG@{k}']:.4f}" for k in ks]
    )
    print(f"Best epoch by HR@{primary_k}: {best_epoch} ({best_metric:.4f})")
    print(f"Test metrics (best checkpoint): {test_text}")

    artifact = {
        "model_state_dict": best_state,
        "config": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "dropout": args.dropout,
            "max_history": args.max_history,
        },
        "item2idx": item2idx,
        "idx2item": idx2item,
        "best_epoch": best_epoch,
        "best_valid_hr": float(best_metric),
        "valid_history": history,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    torch.save(artifact, output_path)
    print(f"Saved artifact to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mult-VAE on MovieLens or Amazon review sequences")
    default_dataset_dir = Path(__file__).resolve().parent.parent / "WebSim_Dataset" / "MM-ML-1M-main"
    parser.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir))
    parser.add_argument("--output-model", type=str, default="artifacts/multvae_ml1m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--anneal-cap", type=float, default=0.2)
    parser.add_argument("--total-anneal-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--eval-ks", type=str, default="10,20")
    parser.add_argument("--max-history", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
