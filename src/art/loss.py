from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
import torch

from art.utils.group_aggregate import group_aggregate

from . import dev

if TYPE_CHECKING:
    from art.unsloth.service import TrainInputs


class Loss(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reduction: Literal["mean", "sum"]
    policy_loss: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor | None
    policy_loss_sum: torch.Tensor
    probs_corr: torch.Tensor
    kl_policy_ref: torch.Tensor | None = None


IMPORTANCE_SAMPLING_LEVEL_TOKEN = 0
IMPORTANCE_SAMPLING_LEVEL_SEQUENCE = 1
IMPORTANCE_SAMPLING_LEVEL_AVERAGE = 2
IMPORTANCE_SAMPLING_LEVEL_GEOMETRIC_AVERAGE = 3


def _importance_sampling_level_id(level: str) -> int:
    if level == "token":
        return IMPORTANCE_SAMPLING_LEVEL_TOKEN
    if level == "sequence":
        return IMPORTANCE_SAMPLING_LEVEL_SEQUENCE
    if level == "average":
        return IMPORTANCE_SAMPLING_LEVEL_AVERAGE
    if level == "geometric_average":
        return IMPORTANCE_SAMPLING_LEVEL_GEOMETRIC_AVERAGE
    raise ValueError(f"Unsupported importance_sampling_level: {level}")


@torch.compile
def _compiled_loss_core(
    *,
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    assistant_mask: torch.Tensor,
    weights: torch.Tensor,
    shifted_group_ids: torch.Tensor,
    ref_logprobs: torch.Tensor,
    has_ref_logprobs: bool,
    entropies: torch.Tensor,
    has_entropies: bool,
    original_logprobs: torch.Tensor,
    has_original_logprobs: bool,
    importance_sampling_level: int,
    ppo: bool,
    epsilon: float,
    epsilon_high: float,
    max_negative_advantage_importance_sampling_weight: float,
    has_max_negative_advantage_importance_sampling_weight: bool,
    mask_prob_ratio: bool,
    kimi_k2_tau: float,
    has_kimi_k2_tau: bool,
    kl_penalty_coef: float,
    has_kl_penalty: bool,
    truncated_importance_sampling: float,
    has_truncated_importance_sampling: bool,
    reduction_is_mean: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logprob_diff = new_logprobs - old_logprobs
    prob_ratio = torch.exp(logprob_diff)
    if importance_sampling_level != IMPORTANCE_SAMPLING_LEVEL_TOKEN:
        sequence_prob_ratio = torch.exp(
            group_aggregate(
                logprob_diff,
                by=shifted_group_ids * assistant_mask,
                reduce="mean",
            )
        )
        if importance_sampling_level == IMPORTANCE_SAMPLING_LEVEL_SEQUENCE:
            prob_ratio = sequence_prob_ratio
        elif importance_sampling_level == IMPORTANCE_SAMPLING_LEVEL_AVERAGE:
            prob_ratio = (prob_ratio + sequence_prob_ratio) / 2
        elif importance_sampling_level == IMPORTANCE_SAMPLING_LEVEL_GEOMETRIC_AVERAGE:
            prob_ratio = (prob_ratio**0.5) * (sequence_prob_ratio**0.5)
    if has_max_negative_advantage_importance_sampling_weight:
        prob_ratio = torch.clamp(
            prob_ratio, max=max_negative_advantage_importance_sampling_weight
        )
    if mask_prob_ratio:
        prob_ratio = torch.where(
            (prob_ratio > 1 - epsilon) & (prob_ratio < 1 + epsilon_high),
            prob_ratio,
            0.0,
        )
    if has_kimi_k2_tau:
        advantages = advantages - kimi_k2_tau * logprob_diff.detach()
    kl_policy_ref = new_logprobs.new_tensor(float("nan"))
    if has_kl_penalty:
        kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
        avg_kl = kl_per_token.sum() / (assistant_mask.sum() + 1e-6)
        kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
        advantages = advantages + kl_penalty
        kl_policy_ref = avg_kl
    if ppo:
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
    else:
        policy_loss = -(
            torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
            * advantages
            * new_logprobs
        )
    if has_truncated_importance_sampling:
        if has_original_logprobs:
            truncated_logprob_diff = old_logprobs - original_logprobs
            truncated_prob_ratio = torch.exp(truncated_logprob_diff)
        else:
            truncated_prob_ratio = prob_ratio
        policy_loss = (
            policy_loss
            * torch.clamp(
                truncated_prob_ratio, max=truncated_importance_sampling
            ).detach()
        )
    if has_ref_logprobs:
        kl_div = (
            torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1.0
        )
    else:
        kl_div = torch.zeros_like(policy_loss)
    policy_loss = policy_loss * weights * assistant_mask
    kl_div = kl_div * weights * assistant_mask
    denominator = assistant_mask.sum() + 1e-6 if reduction_is_mean else 1.0
    reduced_policy_loss = policy_loss.sum() / denominator
    kl = kl_div.sum() / denominator
    entropy = (
        shift_tensor(entropies, 0.0) * weights * assistant_mask
    ).sum() / denominator
    if not has_entropies:
        entropy = new_logprobs.new_tensor(float("nan"))
    return reduced_policy_loss, kl, entropy, policy_loss.sum(), kl_policy_ref


def loss_fn(
    inputs: "TrainInputs",
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropies: torch.Tensor | None,
    experimental_config: dev.TrainConfig,
    reduction: Literal["mean", "sum"] = "mean",
) -> Loss:
    old_logprobs = shift_tensor(inputs["logprobs"], float("nan"))
    advantages = shift_tensor(inputs["advantages"], 0.0)
    assistant_mask = shift_tensor(inputs["assistant_mask"], False).to(
        new_logprobs.dtype
    )
    weights = shift_tensor(inputs["weights"], 0.0)
    shifted_group_ids = shift_tensor(inputs["group_ids"], 0)
    old_logprobs_mask = ~torch.isnan(old_logprobs)
    probs_corr = torch.corrcoef(
        torch.stack(
            [
                torch.exp(old_logprobs[old_logprobs_mask]),
                torch.exp(new_logprobs[old_logprobs_mask]),
            ]
        )
    )[0, 1]
    # Assume missing old logprobs were sampled under the current policy
    old_logprobs = torch.where(
        torch.isnan(old_logprobs),
        new_logprobs.detach(),
        old_logprobs,
    )
    importance_sampling_level = _importance_sampling_level_id(
        experimental_config.get("importance_sampling_level", "token")
    )
    ppo = experimental_config.get("ppo", False)
    if ppo:
        epsilon_default = 0.2
        epsilon_high_default = None
    else:
        epsilon_default = 1.0
        epsilon_high_default = 4.0
    epsilon = experimental_config.get("epsilon", epsilon_default)
    epsilon_high = experimental_config.get("epsilon_high", epsilon_high_default)
    if epsilon_high is None:
        epsilon_high = epsilon
    max_negative_advantage_importance_sampling_weight = experimental_config.get(
        "max_negative_advantage_importance_sampling_weight", 0.0
    )
    has_max_negative_advantage_importance_sampling_weight = (
        "max_negative_advantage_importance_sampling_weight" in experimental_config
        and experimental_config["max_negative_advantage_importance_sampling_weight"]
        is not None
    )
    kimi_k2_tau = experimental_config.get("kimi_k2_tau", 0.0)
    has_kimi_k2_tau = (
        "kimi_k2_tau" in experimental_config
        and experimental_config["kimi_k2_tau"] is not None
    )
    normalized_kimi_k2_tau = 0.0 if kimi_k2_tau is None else float(kimi_k2_tau)
    kl_penalty_coef = experimental_config.get("kl_penalty_coef", 0.0)
    truncated_importance_sampling = experimental_config.get(
        "truncated_importance_sampling", 0.0
    )
    normalized_truncated_importance_sampling = (
        0.0
        if truncated_importance_sampling is None
        else float(truncated_importance_sampling)
    )
    original_logprobs = (
        shift_tensor(inputs["original_logprobs"], 0.0)  # ty:ignore[invalid-key]
        if "original_logprobs" in inputs
        else torch.zeros_like(new_logprobs)
    )
    if "original_logprobs" in inputs:
        original_logprobs = torch.where(
            torch.isnan(original_logprobs),
            new_logprobs.detach(),
            original_logprobs,
        )
    reduced_policy_loss, kl, entropy_tensor, policy_loss_sum, kl_policy_ref_tensor = (
        _compiled_loss_core(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            assistant_mask=assistant_mask,
            weights=weights,
            shifted_group_ids=shifted_group_ids,
            ref_logprobs=(
                ref_logprobs
                if ref_logprobs is not None
                else torch.zeros_like(new_logprobs)
            ),
            has_ref_logprobs=ref_logprobs is not None,
            entropies=entropies
            if entropies is not None
            else torch.zeros_like(new_logprobs),
            has_entropies=entropies is not None,
            original_logprobs=original_logprobs,
            has_original_logprobs="original_logprobs" in inputs,
            importance_sampling_level=importance_sampling_level,
            ppo=ppo,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            max_negative_advantage_importance_sampling_weight=max_negative_advantage_importance_sampling_weight,
            has_max_negative_advantage_importance_sampling_weight=has_max_negative_advantage_importance_sampling_weight,
            mask_prob_ratio=experimental_config.get("mask_prob_ratio", False),
            kimi_k2_tau=normalized_kimi_k2_tau,
            has_kimi_k2_tau=has_kimi_k2_tau,
            kl_penalty_coef=kl_penalty_coef,
            has_kl_penalty=kl_penalty_coef > 0 and ref_logprobs is not None,
            truncated_importance_sampling=normalized_truncated_importance_sampling,
            has_truncated_importance_sampling=(
                "truncated_importance_sampling" in experimental_config
                and experimental_config["truncated_importance_sampling"] is not None
            ),
            reduction_is_mean=reduction == "mean",
        )
    )
    entropy = entropy_tensor if entropies is not None else None
    kl_policy_ref = (
        kl_policy_ref_tensor
        if kl_penalty_coef > 0 and ref_logprobs is not None
        else None
    )
    return Loss(
        reduction=reduction,
        policy_loss=reduced_policy_loss,
        kl=kl,
        entropy=entropy,
        policy_loss_sum=policy_loss_sum,
        probs_corr=probs_corr,
        kl_policy_ref=kl_policy_ref,
    )


def shift_tensor(tensor: torch.Tensor, pad: int | float | bool) -> torch.Tensor:
    return torch.nn.functional.pad(tensor[:, 1:], (0, 1), value=pad)
