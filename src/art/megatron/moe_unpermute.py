from typing import Any

import torch


class _TokenMajorUnpermuteFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        permuted_tokens: torch.Tensor,
        token_major_order: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> torch.Tensor:
        token_major = permuted_tokens.index_select(0, token_major_order)
        inverse_order = torch.empty_like(token_major_order)
        inverse_order[token_major_order] = torch.arange(
            token_major_order.numel(),
            device=token_major_order.device,
            dtype=token_major_order.dtype,
        )
        ctx.save_for_backward(inverse_order, token_counts)
        return torch.segment_reduce(token_major, reduce="sum", lengths=token_counts)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inverse_order, token_counts = ctx.saved_tensors
        grad_token_major = grad_output.repeat_interleave(token_counts, dim=0)
        grad_permuted = grad_token_major.index_select(0, inverse_order)
        return grad_permuted, None, None


def token_major_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    num_tokens = int(restore_shape[0])
    hidden_size = int(restore_shape[1])
    if num_tokens == 0 or sorted_indices.numel() == 0:
        return permuted_tokens.new_zeros((num_tokens, hidden_size))
    token_major_order = torch.argsort(sorted_indices, stable=True)
    token_counts = torch.bincount(sorted_indices, minlength=num_tokens)
    return _TokenMajorUnpermuteFn.apply(
        permuted_tokens,
        token_major_order,
        token_counts,
    )
