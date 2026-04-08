from __future__ import annotations

import torch


def build_popularity_counts(user_train: dict[int | str, list[int]]) -> tuple[torch.Tensor, list[int]]:
    """Build item popularity counts and sorted ranking from train sequences.

    Item index 0 is reserved as padding and always kept at count 0.
    """
    max_item_idx = 0
    for seq in user_train.values():
        if seq:
            max_item_idx = max(max_item_idx, max(int(x) for x in seq))

    counts = torch.zeros(max_item_idx + 1, dtype=torch.float32)
    for seq in user_train.values():
        for item in seq:
            idx = int(item)
            if idx > 0:
                counts[idx] += 1.0

    ranked = torch.argsort(counts, descending=True).tolist()
    ranked = [idx for idx in ranked if idx > 0 and counts[idx] > 0]
    return counts, ranked
