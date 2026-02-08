"""Compatibility shim for rfdetr + transformers 5.x.

rfdetr imports `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`,
which was removed in transformers 5.0. This module patches it back in so rfdetr
can load without modification.
"""
from __future__ import annotations

_patched = False


def patch_transformers_for_rfdetr() -> None:
    global _patched
    if _patched:
        return

    import torch
    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(
            heads: list[int],
            n_heads: int,
            head_size: int,
            already_pruned_heads: set[int],
        ) -> tuple[set[int], torch.LongTensor]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index: torch.LongTensor = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    _patched = True
