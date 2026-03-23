import argparse
import random
from pathlib import Path

import numpy as np
import torch

from lightgcn import LightGCN, build_norm_adj
from train_sasrec import load_user_sequences


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remap_users_items(
    user_sequences: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, int], list[int], dict[int, int]]:
    user_ids = sorted(user_sequences.keys())
    user2idx = {u: i for i, u in enumerate(user_ids, start=1)}

    unique_items = sorted({item for seq in user_sequences.values() for item in seq})
    item2idx = {item: i for i, item in enumerate(unique_items, start=1)}
    idx2item = [0] + unique_items

    remapped = {}
    for raw_uid, sequence in user_sequences.items():
        uid = user2idx[raw_uid]
        remapped[uid] = [item2idx[m] for m in sequence if m in item2idx]
    return remapped, item2idx, idx2item, user2idx


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


def sample_triples(
    user_ids: np.ndarray,
    user_pos_list: list[list[int]],
    user_pos_set: list[set[int]],
    num_items: int,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_users = np.random.choice(user_ids, size=num_samples, replace=True)
    sampled_pos = np.zeros(num_samples, dtype=np.int64)
    sampled_neg = np.zeros(num_samples, dtype=np.int64)

    for idx, uid in enumerate(sampled_users):
        pos_item = random.choice(user_pos_list[uid])
        sampled_pos[idx] = pos_item
        neg_item = random.randint(1, num_items)
        while neg_item in user_pos_set[uid]:
            neg_item = random.randint(1, num_items)
        sampled_neg[idx] = neg_item
    return sampled_users, sampled_pos, sampled_neg


@torch.no_grad()
def evaluate(
    model: LightGCN,
    user_train: dict[int, list[int]],
    user_target: dict[int, int],
    ks: list[int],
) -> dict[str, float]:
    model.eval()
    user_emb, item_emb = model.computer()

    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    valid_users = 0

    for uid, target in user_target.items():
        history = user_train.get(uid, [])
        if not history or target <= 0:
            continue
        scores = torch.matmul(item_emb, user_emb[uid])
        scores = scores.clone()
        scores[0] = -1e9
        for item in set(history):
            if item != target:
                scores[item] = -1e9
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
    ks = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not ks:
        raise ValueError("eval_ks cannot be empty")
    return ks


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ks = parse_ks(args.eval_ks)

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading interactions from: {dataset_dir}")
    raw_sequences = load_user_sequences(dataset_dir, min_seq_len=args.min_seq_len)
    remapped, item2idx, idx2item, _ = remap_users_items(raw_sequences)
    user_train, user_valid, user_test = split_sequences(remapped)
    user_train = {u: s for u, s in user_train.items() if len(s) >= 2}
    user_valid = {u: user_valid[u] for u in user_train if u in user_valid}
    user_test = {u: user_test[u] for u in user_train if u in user_test}

    num_users = max(user_train.keys()) if user_train else 0
    num_items = len(idx2item) - 1
    train_sets = {u: set(seq) for u, seq in user_train.items()}
    print(f"Train users: {len(user_train)}, items: {num_items}, eval_ks: {ks}")

    norm_adj = build_norm_adj(num_users, num_items, train_sets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        norm_adj=norm_adj.to(device),
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reg_weight = args.reg_weight

    user_pos_list = [[] for _ in range(num_users + 1)]
    user_pos_set = [set() for _ in range(num_users + 1)]
    for uid, seq in user_train.items():
        dedup = sorted(set(seq))
        user_pos_list[uid] = dedup
        user_pos_set[uid] = set(dedup)
    user_ids = np.array(sorted(user_train.keys()), dtype=np.int64)

    best_metric = -1.0
    best_epoch = 0
    best_state_dict = None
    history = []
    primary_k = ks[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        users, pos_items, neg_items = sample_triples(
            user_ids=user_ids,
            user_pos_list=user_pos_list,
            user_pos_set=user_pos_set,
            num_items=num_items,
            num_samples=args.samples_per_epoch,
        )
        users_t = torch.LongTensor(users).to(device)
        pos_t = torch.LongTensor(pos_items).to(device)
        neg_t = torch.LongTensor(neg_items).to(device)

        optimizer.zero_grad()
        bpr_loss, reg_loss = model.bpr_loss(users_t, pos_t, neg_t)
        total_loss = bpr_loss + reg_weight * reg_loss
        total_loss.backward()
        optimizer.step()

        metrics = evaluate(model, user_train, user_valid, ks)
        row = {"epoch": epoch, "loss": float(total_loss.item()), **metrics}
        history.append(row)

        metric_text = " ".join(
            [f"HR@{k}={metrics[f'HR@{k}']:.4f} NDCG@{k}={metrics[f'NDCG@{k}']:.4f}" for k in ks]
        )
        print(f"Epoch {epoch}/{args.epochs} - loss: {float(total_loss.item()):.4f} - valid {metric_text}")

        score = metrics[f"HR@{primary_k}"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state_dict)
    model.eval()

    test_metrics = evaluate(model, user_train, user_test, ks)
    test_text = " ".join(
        [f"HR@{k}={test_metrics[f'HR@{k}']:.4f} NDCG@{k}={test_metrics[f'NDCG@{k}']:.4f}" for k in ks]
    )
    print(f"Best epoch by HR@{primary_k}: {best_epoch} ({best_metric:.4f})")
    print(f"Test metrics (best checkpoint): {test_text}")

    with torch.no_grad():
        _, item_emb = model.computer()
        final_item_embeddings = item_emb.detach().cpu()

    artifact = {
        "model_state_dict": best_state_dict,
        "config": {
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
            "n_layers": args.n_layers,
            "max_history": args.max_history,
        },
        "item2idx": item2idx,
        "idx2item": idx2item,
        "best_epoch": best_epoch,
        "best_valid_hr": float(best_metric),
        "valid_history": history,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "final_item_embeddings": final_item_embeddings,
    }
    torch.save(artifact, output_path)
    print(f"Saved artifact to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGCN on MovieLens or Amazon review sequences")
    default_dataset_dir = Path(__file__).resolve().parent.parent / "WebSim_Dataset" / "MM-ML-1M-main"
    parser.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir))
    parser.add_argument("--output-model", type=str, default="artifacts/lightgcn_ml1m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--eval-ks", type=str, default="10,20")
    parser.add_argument("--max-history", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
