#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from megatron.core.transformer.moe import grouped_gemm_util
from pydantic import BaseModel, ConfigDict, Field
import torch


def _parse_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def _mean_ms(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_tokens: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    ffn_hidden_size: int = Field(gt=0)
    num_experts: int = Field(gt=0)
    top_k: int = Field(gt=0)
    dtype_name: str
    warmup: int = Field(ge=0)
    iters: int = Field(gt=0)
    seed: int
    hidden_scale: float = Field(gt=0.0)
    router_scale: float = Field(gt=0.0)
    weight_scale: float = Field(gt=0.0)
    correctness_tokens: int = Field(gt=0)
    train_expert_weights: bool = False

    @property
    def dtype(self) -> torch.dtype:
        return _parse_dtype(self.dtype_name)


class TimingPayload(BaseModel):
    router_ms: float
    dispatch_plan_ms: float
    dispatch_gather_ms: float
    moe_ms: float
    combine_ms: float
    forward_ms: float
    backward_ms: float
    step_ms: float


class CorrectnessPayload(BaseModel):
    output_max_abs: float
    output_mean_abs: float
    hidden_grad_max_abs: float
    hidden_grad_mean_abs: float
    gate_grad_max_abs: float
    gate_grad_mean_abs: float
    expert_fc1_grad_max_abs: float | None = None
    expert_fc2_grad_max_abs: float | None = None


class _TokenMajorCombineFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        down: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_order: torch.Tensor,
        routed_order: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        token_major = down.index_select(0, inverse_order).view(
            num_tokens,
            top_k,
            down.shape[1],
        )
        ctx.save_for_backward(token_major, topk_weights, routed_order)
        return (token_major * topk_weights.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        token_major, topk_weights, routed_order = ctx.saved_tensors
        grad_token_major = grad_output.unsqueeze(1) * topk_weights.unsqueeze(-1)
        grad_down = grad_token_major.reshape(-1, grad_output.shape[1]).index_select(
            0, routed_order
        )
        grad_topk_weights = torch.bmm(
            token_major,
            grad_output.unsqueeze(-1),
        ).squeeze(-1)
        return grad_down, grad_topk_weights, None, None, None, None


def _default_output_json_path(spec: BenchmarkSpec) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = (
        f"tokens_{spec.num_tokens}_hidden_{spec.hidden_size}_ffn_{spec.ffn_hidden_size}"
        f"_experts_{spec.num_experts}_topk_{spec.top_k}_{timestamp}.json"
    )
    return repo_root / ".local" / "bench_moe_router_dispatch_combine" / filename


def _build_problem(
    spec: BenchmarkSpec,
    *,
    device: torch.device,
    num_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    tokens = int(spec.num_tokens if num_tokens is None else num_tokens)
    dtype = spec.dtype
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    return {
        "flat_token_ids": (
            torch.arange(tokens, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(tokens, int(spec.top_k))
            .reshape(-1)
        ),
        "hidden_template": torch.randn(
            tokens,
            int(spec.hidden_size),
            device=device,
            dtype=dtype,
        )
        * float(spec.hidden_scale),
        "gate_template": torch.randn(
            tokens,
            int(spec.num_experts),
            device=device,
            dtype=torch.float32,
        )
        * float(spec.router_scale),
        "expert_fc1_weight": torch.randn(
            int(spec.num_experts),
            int(spec.hidden_size),
            2 * int(spec.ffn_hidden_size),
            device=device,
            dtype=dtype,
        )
        * float(spec.weight_scale),
        "expert_fc2_weight": torch.randn(
            int(spec.num_experts),
            int(spec.ffn_hidden_size),
            int(spec.hidden_size),
            device=device,
            dtype=dtype,
        )
        * float(spec.weight_scale),
        "loss_grad": torch.randn(
            tokens,
            int(spec.hidden_size),
            device=device,
            dtype=torch.float32,
        ),
    }


def _clone_iteration_state(
    problem: dict[str, torch.Tensor],
    *,
    train_expert_weights: bool,
) -> dict[str, torch.Tensor]:
    expert_fc1_weight = problem["expert_fc1_weight"].detach()
    expert_fc2_weight = problem["expert_fc2_weight"].detach()
    expert_fc1_weight.requires_grad_(train_expert_weights)
    expert_fc2_weight.requires_grad_(train_expert_weights)
    return {
        "flat_token_ids": problem["flat_token_ids"],
        "hidden_states": problem["hidden_template"].detach().requires_grad_(True),
        "gate_scores": problem["gate_template"].detach().requires_grad_(True),
        "expert_fc1_weight": expert_fc1_weight,
        "expert_fc2_weight": expert_fc2_weight,
        "loss_grad": problem["loss_grad"],
    }


def _combine_index_add(
    *,
    down: torch.Tensor,
    routed_token_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
) -> torch.Tensor:
    output = down.new_zeros((num_tokens, hidden_size))
    output.index_add_(0, routed_token_ids, down * routed_weights.unsqueeze(-1))
    return output


def _combine_token_major_sum(
    *,
    down: torch.Tensor,
    inverse_order: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    token_major = down.index_select(0, inverse_order).view(
        topk_weights.shape[0],
        topk_weights.shape[1],
        down.shape[1],
    )
    return (token_major * topk_weights.unsqueeze(-1)).sum(dim=1)


def _combine_token_major_manual(
    *,
    down: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_order: torch.Tensor,
    routed_order: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    return _TokenMajorCombineFn.apply(
        down,
        topk_weights,
        inverse_order,
        routed_order,
        num_tokens,
        top_k,
    )


def _run_iteration(
    *,
    spec: BenchmarkSpec,
    problem: dict[str, torch.Tensor],
    combine_backend: str,
    capture_payload: bool = False,
) -> dict[str, Any]:
    state = _clone_iteration_state(
        problem,
        train_expert_weights=bool(spec.train_expert_weights),
    )
    hidden_states = state["hidden_states"]
    gate_scores = state["gate_scores"]
    expert_fc1_weight = state["expert_fc1_weight"]
    expert_fc2_weight = state["expert_fc2_weight"]
    stream = torch.cuda.current_stream(hidden_states.device)
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in (
            "router",
            "dispatch_plan",
            "dispatch_gather",
            "moe",
            "combine",
            "forward",
            "backward",
            "step",
        )
    }

    events["step"][0].record(stream)
    events["forward"][0].record(stream)

    events["router"][0].record(stream)
    topk_values, topk_indices = torch.topk(gate_scores, k=int(spec.top_k), dim=-1)
    topk_weights = torch.softmax(topk_values, dim=-1).to(hidden_states.dtype)
    events["router"][1].record(stream)

    events["dispatch_plan"][0].record(stream)
    flat_expert_ids = topk_indices.reshape(-1).to(torch.long)
    flat_weights = topk_weights.reshape(-1)
    routed_order = torch.argsort(flat_expert_ids, stable=True)
    routed_token_ids = state["flat_token_ids"].index_select(0, routed_order)
    routed_weights = flat_weights.index_select(0, routed_order)
    tokens_per_expert = torch.bincount(
        flat_expert_ids, minlength=int(spec.num_experts)
    ).to(device="cpu", dtype=torch.int64)
    inverse_order = torch.empty_like(routed_order)
    inverse_order[routed_order] = torch.arange(
        routed_order.numel(),
        device=routed_order.device,
        dtype=routed_order.dtype,
    )
    events["dispatch_plan"][1].record(stream)

    events["dispatch_gather"][0].record(stream)
    dispatched = hidden_states.index_select(0, routed_token_ids)
    events["dispatch_gather"][1].record(stream)

    events["moe"][0].record(stream)
    fc1_out = grouped_gemm_util.ops.gmm(
        dispatched,
        expert_fc1_weight,
        tokens_per_expert,
        trans_b=False,
    )
    gate_out, up_out = torch.chunk(fc1_out, 2, dim=-1)
    activated = torch.nn.functional.silu(gate_out) * up_out
    down = grouped_gemm_util.ops.gmm(
        activated,
        expert_fc2_weight,
        tokens_per_expert,
        trans_b=False,
    )
    events["moe"][1].record(stream)

    events["combine"][0].record(stream)
    if combine_backend == "index_add":
        output = _combine_index_add(
            down=down,
            routed_token_ids=routed_token_ids,
            routed_weights=routed_weights,
            num_tokens=int(spec.num_tokens),
            hidden_size=int(spec.hidden_size),
        )
    elif combine_backend == "token_major_sum":
        output = _combine_token_major_sum(
            down=down,
            inverse_order=inverse_order,
            topk_weights=topk_weights,
        )
    elif combine_backend == "token_major_manual":
        output = _combine_token_major_manual(
            down=down,
            topk_weights=topk_weights,
            inverse_order=inverse_order,
            routed_order=routed_order,
            num_tokens=int(spec.num_tokens),
            top_k=int(spec.top_k),
        )
    else:
        raise ValueError(f"Unsupported combine backend {combine_backend!r}")
    events["combine"][1].record(stream)
    events["forward"][1].record(stream)

    loss = (output.float() * state["loss_grad"]).sum() / max(
        1, state["loss_grad"].numel()
    )
    events["backward"][0].record(stream)
    loss.backward()
    events["backward"][1].record(stream)
    events["step"][1].record(stream)
    torch.cuda.synchronize(hidden_states.device)

    payload = {
        "timing_ms": TimingPayload(
            router_ms=float(events["router"][0].elapsed_time(events["router"][1])),
            dispatch_plan_ms=float(
                events["dispatch_plan"][0].elapsed_time(events["dispatch_plan"][1])
            ),
            dispatch_gather_ms=float(
                events["dispatch_gather"][0].elapsed_time(events["dispatch_gather"][1])
            ),
            moe_ms=float(events["moe"][0].elapsed_time(events["moe"][1])),
            combine_ms=float(events["combine"][0].elapsed_time(events["combine"][1])),
            forward_ms=float(events["forward"][0].elapsed_time(events["forward"][1])),
            backward_ms=float(
                events["backward"][0].elapsed_time(events["backward"][1])
            ),
            step_ms=float(events["step"][0].elapsed_time(events["step"][1])),
        )
    }
    if capture_payload:
        payload.update(
            {
                "output": output.detach(),
                "hidden_grad": hidden_states.grad.detach(),
                "gate_grad": gate_scores.grad.detach(),
                "expert_fc1_grad": (
                    None
                    if expert_fc1_weight.grad is None
                    else expert_fc1_weight.grad.detach()
                ),
                "expert_fc2_grad": (
                    None
                    if expert_fc2_weight.grad is None
                    else expert_fc2_weight.grad.detach()
                ),
            }
        )
    return payload


def _warmup_all_backends(
    *,
    spec: BenchmarkSpec,
    problem: dict[str, torch.Tensor],
    variants: list[str],
    repeats: int,
) -> None:
    for _ in range(repeats):
        for variant in variants:
            _run_iteration(spec=spec, problem=problem, combine_backend=variant)
            torch.cuda.empty_cache()


def _aggregate_variant_result(
    *,
    spec: BenchmarkSpec,
    timings: list[TimingPayload],
) -> dict[str, Any]:
    mean_timing = TimingPayload(
        router_ms=_mean_ms([payload.router_ms for payload in timings]),
        dispatch_plan_ms=_mean_ms([payload.dispatch_plan_ms for payload in timings]),
        dispatch_gather_ms=_mean_ms(
            [payload.dispatch_gather_ms for payload in timings]
        ),
        moe_ms=_mean_ms([payload.moe_ms for payload in timings]),
        combine_ms=_mean_ms([payload.combine_ms for payload in timings]),
        forward_ms=_mean_ms([payload.forward_ms for payload in timings]),
        backward_ms=_mean_ms([payload.backward_ms for payload in timings]),
        step_ms=_mean_ms([payload.step_ms for payload in timings]),
    )
    step_s = mean_timing.step_ms / 1_000.0
    forward_s = mean_timing.forward_ms / 1_000.0
    routed_tokens = int(spec.num_tokens) * int(spec.top_k)
    return {
        "timing_ms": mean_timing.model_dump(),
        "throughput": {
            "tokens_per_s_step": float(spec.num_tokens) / step_s,
            "tokens_per_s_forward": float(spec.num_tokens) / forward_s,
            "routed_tokens_per_s_step": float(routed_tokens) / step_s,
            "routed_tokens_per_s_forward": float(routed_tokens) / forward_s,
        },
    }


def _correctness_delta(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    diff = (baseline - candidate).abs()
    return float(diff.max().item()), float(diff.float().mean().item())


def _measure_correctness(
    *,
    spec: BenchmarkSpec,
    device: torch.device,
    variants: list[str],
) -> dict[str, Any]:
    if "index_add" not in variants:
        return {}
    correctness_tokens = min(int(spec.correctness_tokens), int(spec.num_tokens))
    correctness_spec = spec.model_copy(update={"num_tokens": correctness_tokens})
    correctness_problem = _build_problem(
        correctness_spec,
        device=device,
    )
    baseline = _run_iteration(
        spec=correctness_spec,
        problem=correctness_problem,
        combine_backend="index_add",
        capture_payload=True,
    )
    results: dict[str, Any] = {}
    for variant in variants:
        if variant == "index_add":
            continue
        candidate = _run_iteration(
            spec=correctness_spec,
            problem=correctness_problem,
            combine_backend=variant,
            capture_payload=True,
        )
        output_max_abs, output_mean_abs = _correctness_delta(
            baseline["output"], candidate["output"]
        )
        hidden_grad_max_abs, hidden_grad_mean_abs = _correctness_delta(
            baseline["hidden_grad"], candidate["hidden_grad"]
        )
        gate_grad_max_abs, gate_grad_mean_abs = _correctness_delta(
            baseline["gate_grad"], candidate["gate_grad"]
        )
        expert_fc1_grad_max_abs = None
        expert_fc2_grad_max_abs = None
        if (
            baseline["expert_fc1_grad"] is not None
            and candidate["expert_fc1_grad"] is not None
        ):
            expert_fc1_grad_max_abs, _ = _correctness_delta(
                baseline["expert_fc1_grad"], candidate["expert_fc1_grad"]
            )
        if (
            baseline["expert_fc2_grad"] is not None
            and candidate["expert_fc2_grad"] is not None
        ):
            expert_fc2_grad_max_abs, _ = _correctness_delta(
                baseline["expert_fc2_grad"], candidate["expert_fc2_grad"]
            )
        results[variant] = CorrectnessPayload(
            output_max_abs=output_max_abs,
            output_mean_abs=output_mean_abs,
            hidden_grad_max_abs=hidden_grad_max_abs,
            hidden_grad_mean_abs=hidden_grad_mean_abs,
            gate_grad_max_abs=gate_grad_max_abs,
            gate_grad_mean_abs=gate_grad_mean_abs,
            expert_fc1_grad_max_abs=expert_fc1_grad_max_abs,
            expert_fc2_grad_max_abs=expert_fc2_grad_max_abs,
        ).model_dump()
    return results


def benchmark(spec: BenchmarkSpec, *, variants: list[str]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if spec.top_k > spec.num_experts:
        raise ValueError(
            f"top_k must be <= num_experts, got top_k={spec.top_k}, num_experts={spec.num_experts}"
        )
    device = torch.device("cuda")
    problem = _build_problem(spec, device=device)

    _warmup_all_backends(
        spec=spec,
        problem=problem,
        variants=variants,
        repeats=max(1, int(spec.warmup)),
    )

    results: dict[str, Any] = {}
    timings_by_variant = {variant: [] for variant in variants}
    for round_idx in range(int(spec.iters)):
        start_idx = round_idx % len(variants)
        ordered_variants = variants[start_idx:] + variants[:start_idx]
        for variant in ordered_variants:
            payload = _run_iteration(
                spec=spec,
                problem=problem,
                combine_backend=variant,
            )
            timings_by_variant[variant].append(payload["timing_ms"])
            torch.cuda.empty_cache()
    for variant, timings in timings_by_variant.items():
        results[variant] = _aggregate_variant_result(spec=spec, timings=timings)

    baseline_step_ms = (
        results["index_add"]["timing_ms"]["step_ms"] if "index_add" in results else None
    )
    if baseline_step_ms is not None:
        for variant, payload in results.items():
            step_ms = payload["timing_ms"]["step_ms"]
            payload["speedup_vs_index_add"] = (
                float(baseline_step_ms) / float(step_ms) if step_ms > 0.0 else None
            )

    return {
        "spec": spec.model_dump(),
        "derived": {
            "routed_tokens": int(spec.num_tokens) * int(spec.top_k),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "correctness": _measure_correctness(
            spec=spec, device=device, variants=variants
        ),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark isolated router -> dispatch -> grouped MoE -> combine on CUDA with "
            "the Qwen3-30B-A3B local MoE shape."
        )
    )
    parser.add_argument("--num-tokens", type=int, default=40960)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--router-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--correctness-tokens", type=int, default=1024)
    parser.add_argument(
        "--train-expert-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable expert weight gradients instead of the frozen-weight ART-like mode.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["index_add", "token_major_sum", "token_major_manual"],
        choices=["index_add", "token_major_sum", "token_major_manual"],
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BenchmarkSpec(
        num_tokens=int(args.num_tokens),
        hidden_size=int(args.hidden_size),
        ffn_hidden_size=int(args.ffn_hidden_size),
        num_experts=int(args.num_experts),
        top_k=int(args.top_k),
        dtype_name=str(args.dtype),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
        hidden_scale=float(args.hidden_scale),
        router_scale=float(args.router_scale),
        weight_scale=float(args.weight_scale),
        correctness_tokens=int(args.correctness_tokens),
        train_expert_weights=bool(args.train_expert_weights),
    )
    payload = benchmark(spec, variants=[str(variant) for variant in args.variants])
    output_path = args.json_out or _default_output_json_path(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
