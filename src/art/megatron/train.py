# isort: off
import os


def _set_cache_dir(env_var: str, default_path: str) -> None:
    if not os.environ.get(env_var):
        os.environ[env_var] = os.path.expanduser(default_path)
    os.makedirs(os.environ[env_var], exist_ok=True)


os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
_set_cache_dir("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torchinductor")
_set_cache_dir("TRITON_CACHE_DIR", "~/.triton/cache")
# isort: on

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import json
import math
import shutil
import time
from typing import Any, Callable, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file, save_file
import torch
from torch.distributed import all_reduce
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.megatron.bridge_adapter_compat import build_adapter_weights_by_base
from art.megatron.finalize_grads import finalize_model_grads_extended
from art.megatron.flex_attention import create_shared_prefix_attention_state
from art.megatron.job_protocol import (
    MegatronMergedTrainJob,
    MegatronSyncJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
    load_megatron_job,
)
from art.megatron.lora import (
    apply_lora_adapters,
)
from art.megatron.offload import (
    OffloadState,
    clear_optimizer_state,
    offload_to_cpu,
    reload_to_gpu,
)
from art.megatron.provider import get_provider_bundle
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
)
from art.preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)

DEFAULT_MODEL_IDENTIFIER = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    bridge: Any
    model: list[MegatronModule]
    optimizer: Any
    rank: int
    world_size: int
    moe_routing_replay_controller: MoeRoutingReplayController | None = None
    merged_weight_transfer_group: Any | None = None
    merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None = None


class TrainStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reduced_loss: torch.Tensor
    probs_corr: float
    new_logprobs: torch.Tensor
    update_successful: bool
    grad_norm: float
    num_zeros_in_grad: int | None


@dataclass
class MergedWeightExport:
    bridge: Any
    model: list[MegatronModule]
    model_config: Any
    conversion_tasks: list[Any]
    adapter_weights_by_base: dict[str, list[Any]]


def print0(rank: int, *values: Any) -> None:
    if rank == 0:
        print(*values)


def freeze_model(model_chunks: list[MegatronModule]) -> list[MegatronModule]:
    for module in model_chunks:
        for param in module.parameters():
            param.requires_grad = False
    return model_chunks


def _frozen_linear_grad_input(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if grad_output.dim() <= 2 or weight.dim() != 2:
        return grad_output.matmul(weight)
    try:
        grad_output_2d = grad_output.view(-1, int(grad_output.shape[-1]))
    except RuntimeError:
        grad_output_2d = grad_output.reshape(-1, int(grad_output.shape[-1]))
    grad_input_2d = grad_output_2d.matmul(weight)
    return grad_input_2d.reshape(*grad_output.shape[:-1], int(weight.shape[-1]))


def _install_fast_frozen_output_backward() -> None:
    from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight

    if getattr(LinearWithFrozenWeight.backward, "__art_fast_output_backward__", False):
        return

    def _fast_backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (weight,) = ctx.saved_tensors
        grad_input = _frozen_linear_grad_input(grad_output, weight)
        if ctx.allreduce_dgrad:
            all_reduce(grad_input, group=ctx.tp_group)
        return grad_input, None, None, None, None

    setattr(_fast_backward, "__art_fast_output_backward__", True)
    LinearWithFrozenWeight.backward = staticmethod(_fast_backward)


def _install_gpt_preprocess_hook(model_chunks: list[MegatronModule]) -> None:
    for chunk in model_chunks:
        module: Any = chunk
        while not isinstance(module, GPTModel) and hasattr(module, "module"):
            module = module.module
        if not isinstance(module, GPTModel):
            module = getattr(module, "language_model", None)
        if not isinstance(module, GPTModel):
            continue
        preprocess = module._preprocess

        def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
            preproc_output = list(_preprocess(*args, **kwargs))
            preproc_output[0].requires_grad = True  # type: ignore[index]
            table = preproc_output[1]  # [S, B, 1, D]  # type: ignore[index]
            embedding_dim = table.size(-1)
            table_flat = table.view(table.size(0), embedding_dim)
            position_ids = kwargs["position_ids"]  # [B, S]
            if position_ids.ndim != 2:
                return tuple(preproc_output)
            batch_size, sequence_length = position_ids.shape
            gathered = table_flat.index_select(0, position_ids.reshape(-1))
            gathered = (
                gathered.view(batch_size, sequence_length, embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            preproc_output[1] = gathered.unsqueeze(2)  # [S, B, 1, D]
            return tuple(preproc_output)

        module._preprocess = preprocess_hook  # type: ignore[attr-defined]


def _default_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True,
        lr=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
        adam_eps=1e-13,
    )


def configure_moe_routing_replay(
    runtime: TrainingRuntime,
    *,
    replay_bundle_path: str | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
    strict: bool = True,
) -> None:
    if runtime.moe_routing_replay_controller is not None:
        runtime.moe_routing_replay_controller.remove_router_patches()
        runtime.moe_routing_replay_controller = None

    if replay_bundle is not None and replay_bundle_path is not None:
        raise RuntimeError(
            "Provide either replay_bundle_path or replay_bundle, not both"
        )
    if replay_bundle is None and replay_bundle_path is None:
        return

    if replay_bundle is None:
        if replay_bundle_path is None:
            raise RuntimeError(
                "replay_bundle_path is required when replay_bundle is None"
            )
        replay_bundle = MoeRoutingReplayBundle.from_dir(replay_bundle_path)

    controller = MoeRoutingReplayController(
        bundle=replay_bundle,
        strict=strict,
    )
    controller.install_router_patches(runtime.model)
    runtime.moe_routing_replay_controller = controller


def build_training_runtime(
    *,
    model_identifier: str | None = None,
    provider_torch_dtype: torch.dtype = torch.bfloat16,
    provider_configure: Callable[[Any], None] | None = None,
    optimizer_config: OptimizerConfig | None = None,
    moe_routing_replay_path: str | None = None,
    moe_routing_replay_bundle: MoeRoutingReplayBundle | None = None,
    moe_routing_replay_strict: bool = True,
    print_env: bool = True,
    print_optimizer_stats: bool = True,
) -> TrainingRuntime:
    _install_fast_frozen_output_backward()
    provider_bundle = get_provider_bundle(
        model_identifier
        or os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER),
        torch_dtype=provider_torch_dtype,
    )
    provider = provider_bundle.provider
    if provider_configure is not None:
        provider_configure(provider)
    provider.register_pre_wrap_hook(freeze_model)
    provider.register_pre_wrap_hook(
        lambda chunks: apply_lora_adapters(chunks, provider)
    )

    model = cast(
        list[MegatronModule],
        provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(
                # memory and comm for this should be small anyways cause lora
                grad_reduce_in_fp32=True,
                average_in_collective=False,
            ),
            data_parallel_random_init=False,
        ),
    )

    if not torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError(
            "torch.distributed must be initialized before building runtime"
        )
    rank = torch.distributed.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]

    if rank == 0 and print_env:
        print("TORCHINDUCTOR_CACHE_DIR:", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        print("Resolved inductor cache_dir():", inductor_cache_dir())
        print("TRITON_CACHE_DIR:", os.environ["TRITON_CACHE_DIR"])

    _install_gpt_preprocess_hook(model)

    optimizer = get_megatron_optimizer(
        config=optimizer_config or _default_optimizer_config(),
        model_chunks=model,
    )

    if rank == 0 and print_optimizer_stats:
        num_params = sum(
            p.numel()
            for group in optimizer.param_groups
            if not group["is_decoupled_lr"]
            for p in group["params"]
        )
        print(f"Number of parameters in optimizer: {num_params:,}")
        total_params = sum(p.numel() for module in model for p in module.parameters())
        percent = (num_params / total_params) * 100 if total_params > 0 else 0
        print(f"Optimizer parameters as percent of total: {percent:0.2f}%")

    runtime = TrainingRuntime(
        provider=provider,
        bridge=provider_bundle.bridge,
        model=model,
        optimizer=optimizer,
        rank=rank,
        world_size=world_size,
    )
    configure_moe_routing_replay(
        runtime,
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
        strict=moe_routing_replay_strict,
    )
    return runtime


def iter_modules(model_chunks: list[MegatronModule]) -> Any:
    for chunk in model_chunks:
        for module in chunk.modules():
            yield module


def load_adapter_into_model(
    model_chunks: list[MegatronModule],
    adapter_model: dict[str, torch.Tensor],
    optimizer: Any | None = None,
) -> None:
    with torch.no_grad():
        for module in iter_modules(model_chunks):
            if hasattr(module, "load_lora"):
                module.load_lora(adapter_model)  # type: ignore[attr-defined]

    if optimizer is None:
        return
    optimizer.reload_model_params()


def maybe_load_adapter_into_model(
    model_chunks: list[MegatronModule],
    adapter_model_path: str,
    optimizer: Any | None = None,
    *,
    rank: int,
) -> dict[str, torch.Tensor]:
    if not os.path.exists(adapter_model_path):
        print0(rank, "No adapter model found at", adapter_model_path)
        return {}
    print0(rank, "Loading adapter model from", adapter_model_path)
    adapter_model = load_file(adapter_model_path)
    load_adapter_into_model(model_chunks, adapter_model, optimizer)
    return adapter_model


def collect_sharded_lora_state(
    model_chunks: list[MegatronModule],
    adapter_model: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    sharded_state_dict: dict[str, torch.Tensor] = {}
    sharded_state_manifest: dict[str, dict[str, Any]] = {}
    for module in iter_modules(model_chunks):
        if hasattr(module, "sharded_lora_state_dict"):
            module_sharded_lora_state_dict: dict[str, torch.Tensor] = (
                module.sharded_lora_state_dict()  # type: ignore[attr-defined]
            )
            for key, value in module_sharded_lora_state_dict.items():
                target_dtype = (
                    adapter_model[key].dtype if key in adapter_model else value.dtype
                )
                sharded_state_dict[key] = value.to(target_dtype)
        if hasattr(module, "sharded_lora_manifest"):
            module_sharded_lora_manifest: dict[str, dict[str, Any]] = (
                module.sharded_lora_manifest()  # type: ignore[attr-defined]
            )
            sharded_state_manifest.update(module_sharded_lora_manifest)
    return sharded_state_dict, sharded_state_manifest


@torch.no_grad()
def select_indexed_inputs(packed_tensors: PackedTensors, index: int) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value[index : index + 1]
            for key, value in packed_tensors.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _clone_packed_tensors(inputs: PackedTensors) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value.clone()
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _zero_contribution_inputs(template: PackedTensors) -> PackedTensors:
    dummy = _clone_packed_tensors(template)
    dummy["assistant_mask"].zero_()
    return dummy


def resolve_local_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int,
) -> int:
    dp_world_size = ps.get_data_parallel_world_size()
    if (
        global_grad_accumulation_sequences <= 0
        or global_grad_accumulation_sequences % dp_world_size != 0
    ):
        raise RuntimeError(
            "Invalid global grad accumulation / DP world size combination: "
            f"global_grad_accumulation_sequences={global_grad_accumulation_sequences}, "
            f"dp_world_size={dp_world_size}"
        )
    return global_grad_accumulation_sequences // dp_world_size


def build_micro_sample_indices(
    step_index: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int,
) -> list[int | None]:
    dp_rank = ps.get_data_parallel_rank()
    dp_world_size = ps.get_data_parallel_world_size()
    local_grad_accumulation_sequences = resolve_local_grad_accumulation_sequences(
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    base_global_sample_index = step_index * global_grad_accumulation_sequences
    global_step_indices: list[int | None] = []
    for offset in range(global_grad_accumulation_sequences):
        global_sample_index = base_global_sample_index + offset
        global_step_indices.append(
            global_sample_index if global_sample_index < num_sequences else None
        )
    return [
        global_step_indices[offset * dp_world_size + dp_rank]
        for offset in range(local_grad_accumulation_sequences)
    ]


def select_micro_inputs(
    packed_tensors: PackedTensors,
    sample_indices: list[int | None],
    zero_template: PackedTensors,
) -> list[PackedTensors]:
    return [
        _clone_packed_tensors(zero_template)
        if sample_index is None
        else select_indexed_inputs(packed_tensors, sample_index)
        for sample_index in sample_indices
    ]


def _move_inputs_to_device(inputs: PackedTensors, device: torch.device) -> None:
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)  # type: ignore[index]


def _attention_block_kwargs(
    model_chunk: MegatronModule,
    attention_state: Any,
) -> dict[str, Any]:
    model = model_chunk
    while hasattr(model, "module"):
        model = model.module  # type: ignore[assignment]
    if type(model).__name__ == "Qwen3VLModel":
        return {"extra_block_kwargs": {"attention_bias": attention_state}}
    return {"attention_bias": attention_state}


def _optimizer_step(
    optimizer: Any,
    learning_rate: float,
) -> tuple[bool, float, int | None]:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    update_successful, grad_norm, num_zeros_in_grad = cast(
        tuple[bool, float, int | None], optimizer.step()
    )
    optimizer.zero_grad()
    return update_successful, grad_norm, num_zeros_in_grad


def _reduce_loss(
    loss: torch.Tensor,
    op: Any = torch.distributed.ReduceOp.AVG,  # ty: ignore[possibly-missing-attribute]
    group: Any | None = None,
) -> torch.Tensor:
    reduced_loss = loss.detach().clone()
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        reduced_loss,
        op=op,
        group=group,
    )
    return reduced_loss


def _count_trainable_tokens(inputs: PackedTensors) -> float:
    assistant_mask = shift_tensor(inputs["assistant_mask"], False)
    return float(assistant_mask.sum().item())


def _local_trainable_token_count_tensor(
    micro_inputs: list[PackedTensors],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(_count_trainable_tokens(micro) for micro in micro_inputs)
    return torch.tensor([local_token_total], device=device, dtype=torch.float32)


def run_training_step(
    *,
    model_chunks: list[MegatronModule],
    optimizer: Any,
    learning_rate: float,
    inputs: PackedTensors | list[PackedTensors],
    config: types.TrainConfig,
    experimental_config: dev.TrainConfig,
    step_index: int,
    sample_index: int | list[int | None],
    ref_logprobs: torch.Tensor | None = None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
) -> TrainStepResult:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("run_training_step requires at least one packed sequence")

    if isinstance(sample_index, list):
        if len(sample_index) != len(micro_inputs):
            raise ValueError(
                "sample_index list length must match number of micro inputs: "
                f"{len(sample_index)} != {len(micro_inputs)}"
            )
        micro_sample_indices = sample_index
    else:
        assert len(micro_inputs) == 1
        micro_sample_indices = [sample_index]

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=micro_sample_indices,
            global_grad_accumulation_sequences=config.grad_accumulation_sequences,
        )

    device = next(model_chunks[0].parameters()).device

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]

    micro_count = len(micro_inputs)
    raw_loss_sum: torch.Tensor | None = None
    num_tokens = _local_trainable_token_count_tensor(micro_inputs, device=device)
    probs_corr_sum = 0.0
    new_logprobs: torch.Tensor | None = None

    for micro in micro_inputs:
        _move_inputs_to_device(micro, device)
        attention_state = create_shared_prefix_attention_state(
            group_ids=micro["group_ids"],
            parent_ids=micro["parent_ids"],
        )
        attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)

        model_output = model_chunks[0](
            input_ids=micro["tokens"],
            position_ids=micro["input_pos"],
            attention_mask=attention_mask,
            labels=shift_tensor(micro["tokens"], 0),
            extra_block_kwargs=_attention_block_kwargs(
                model_chunks[0],
                attention_state,
            ),
        )
        new_logprobs = -model_output

        loss_info = loss_fn(
            micro,  # ty: ignore[invalid-argument-type]
            new_logprobs,
            ref_logprobs,
            None,
            experimental_config,
            reduction="sum",
        )
        micro_loss = loss_info.policy_loss
        micro_loss.backward()
        probs_corr_sum += float(loss_info.probs_corr.item())
        detached_micro_loss = micro_loss.detach()
        if raw_loss_sum is None:
            raw_loss_sum = detached_micro_loss
        else:
            raw_loss_sum = raw_loss_sum + detached_micro_loss

    if new_logprobs is None or raw_loss_sum is None:
        raise RuntimeError("run_training_step did not produce outputs")

    finalize_model_grads_extended(model_chunks, num_tokens=num_tokens)
    update_successful, grad_norm, num_zeros_in_grad = _optimizer_step(
        optimizer,
        learning_rate,
    )
    global_num_tokens = max(num_tokens.item(), 1.0)
    reduced_loss = _reduce_loss(
        raw_loss_sum / global_num_tokens,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step()

    return TrainStepResult(
        reduced_loss=reduced_loss,
        probs_corr=probs_corr_sum / micro_count,
        new_logprobs=new_logprobs,
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def _is_art_adapter_param_name(name: str) -> bool:
    return any(
        segment in name
        for segment in (
            ".lora.",
            ".q_proj_lora.",
            ".k_proj_lora.",
            ".v_proj_lora.",
            ".qkv_lora.",
            ".z_lora.",
            ".gate_lora.",
            ".up_lora.",
        )
    )


def _canonical_art_param_name(name: str) -> str:
    segments = name.split(".")
    while segments and segments[0] == "module":
        segments = segments[1:]
    canonical: list[str] = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments):
            current = segments[i]
            nxt = segments[i + 1]
            if current in {
                "linear_proj",
                "linear_qkv",
                "in_proj",
                "linear_fc1",
                "linear_fc2",
            } and nxt == current:
                canonical.append(current)
                i += 2
                continue
            if current == "out_proj" and nxt == "linear_proj":
                canonical.append(current)
                i += 2
                continue
            if current == "row_parallel_lora" and nxt == "linear_proj":
                i += 2
                continue
        canonical.append(segments[i])
        i += 1
    return ".".join(canonical)


def _mapping_hf_weights_exist(mapping: Any, hf_keys: set[str]) -> bool:
    if getattr(mapping, "allow_hf_name_mismatch", False):
        return True
    hf_param = mapping.hf_param
    if isinstance(hf_param, str):
        return hf_param in hf_keys
    if isinstance(hf_param, dict):
        return all(param in hf_keys for param in hf_param.values())
    return False


def _build_art_conversion_tasks(runtime: TrainingRuntime) -> list[Any]:
    from itertools import chain

    from megatron.bridge.models.conversion.model_bridge import (
        WeightConversionTask,
        _megatron_local_name_to_global,
    )
    from megatron.bridge.models.conversion.utils import (
        get_module_and_param_from_name,
        persistent_buffers,
    )

    bridge = runtime.bridge
    mapping_registry = bridge._model_bridge.mapping_registry()
    hf_source = bridge.hf_pretrained.state.source
    hf_keys = set(hf_source.get_all_keys())
    model_config = runtime.model[0].config
    tasks: list[Any] = []
    for vp_stage, model in enumerate(runtime.model):
        for local_name, _ in chain(model.named_parameters(), persistent_buffers(model)):
            if "_extra_state" in local_name or _is_art_adapter_param_name(local_name):
                continue
            global_name = _megatron_local_name_to_global(
                runtime.model,
                model_config,
                _canonical_art_param_name(local_name),
                vp_stage,
            )
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if mapping is None or not _mapping_hf_weights_exist(mapping, hf_keys):
                continue
            local_module, local_weights = get_module_and_param_from_name(
                runtime.model,
                local_name,
                vp_stage,
            )
            if local_module is not None and not hasattr(local_module, "config"):
                setattr(local_module, "config", model_config)
            tasks.append(
                WeightConversionTask(
                    pp_rank=0,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )
            )
    return tasks


def _build_merged_weight_export(runtime: TrainingRuntime) -> MergedWeightExport:
    return MergedWeightExport(
        bridge=runtime.bridge,
        model=runtime.model,
        model_config=runtime.model[0].config,
        conversion_tasks=_build_art_conversion_tasks(runtime),
        adapter_weights_by_base=build_adapter_weights_by_base(runtime.model),
    )


def _iter_merged_vllm_weights(weight_export: MergedWeightExport) -> Any:
    # vLLM expects HF checkpoint names, but Megatron only has live trainer weights.
    # Convert through Bridge here, then merge ART's LoRA deltas into those tensors.
    bridge = weight_export.bridge
    model_bridge = bridge._model_bridge
    hf_state_dict = bridge.hf_pretrained.state
    grouped_buffers: dict[str, dict[int, torch.Tensor]] = {}
    for task in weight_export.conversion_tasks:
        converted_weights_dict = task.mapping.megatron_to_hf(
            task.param_weight,
            task.megatron_module,
        )
        adapter_weights = weight_export.adapter_weights_by_base.get(task.global_param_name)
        if adapter_weights is not None:
            converted_weights_dict = model_bridge._merge_lora_adapter_weights(
                weight_export.model,
                converted_weights_dict,
                adapter_weights,
            )
        if getattr(task.mapping, "is_grouped_export", False):
            merged_result = model_bridge._accumulate_grouped_export(
                task,
                converted_weights_dict,
                weight_export.model_config,
                grouped_buffers,
                hf_state_dict,
            )
            if merged_result is None:
                continue
            converted_weights_dict = merged_result
        else:
            converted_weights_dict = model_bridge.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )
        for hf_name, tensor in converted_weights_dict.items():
            yield hf_name, tensor


def _ensure_merged_weight_transfer_group(
    runtime: TrainingRuntime,
    spec: MergedWeightTransferSpec,
) -> None:
    assert runtime.rank == 0
    assert runtime.world_size == 1
    if runtime.merged_weight_transfer_init_info == spec.init_info:
        assert runtime.merged_weight_transfer_group is not None
        return
    import httpx
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    def _remote_init() -> None:
        response = httpx.post(
            f"{spec.vllm_base_url}/init_weight_transfer_engine",
            json={"init_info": spec.init_info.model_dump()},
            timeout=300.0,
        )
        response.raise_for_status()

    with ThreadPoolExecutor(max_workers=1) as executor:
        remote_future = executor.submit(_remote_init)
        time.sleep(1.0)
        runtime.merged_weight_transfer_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": spec.init_info.master_address,
                "master_port": spec.init_info.master_port,
                "world_size": spec.init_info.world_size,
            }
        )
        remote_future.result()
    runtime.merged_weight_transfer_init_info = spec.init_info


def _sync_merged_weights_to_vllm(
    runtime: TrainingRuntime,
    spec: MergedWeightTransferSpec,
    *,
    pause_generation: bool,
) -> None:
    assert runtime.rank == 0
    assert runtime.world_size == 1

    import httpx
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    _ensure_merged_weight_transfer_group(runtime, spec)
    weight_export = _build_merged_weight_export(runtime)

    def _send_weights() -> None:
        NCCLWeightTransferEngine.trainer_send_weights(
            _iter_merged_vllm_weights(weight_export),
            {"group": runtime.merged_weight_transfer_group},
        )

    with httpx.Client() as client:
        if pause_generation:
            response = client.post(
                f"{spec.vllm_base_url}/pause",
                params={"mode": "wait"},
                timeout=300.0,
            )
            response.raise_for_status()
        try:
            torch.cuda.synchronize()
            names: list[str] = []
            dtype_names: list[str] = []
            shapes: list[list[int]] = []
            for name, tensor in _iter_merged_vllm_weights(weight_export):
                names.append(name)
                dtype_names.append(str(tensor.dtype).removeprefix("torch."))
                shapes.append(list(tensor.shape))
            with ThreadPoolExecutor(max_workers=1) as executor:
                send_future = executor.submit(_send_weights)
                response = client.post(
                    f"{spec.vllm_base_url}/update_weights",
                    json={
                        "update_info": {
                            "names": names,
                            "dtype_names": dtype_names,
                            "shapes": shapes,
                            "is_checkpoint_format": True,
                        }
                    },
                    timeout=600.0,
                )
                response.raise_for_status()
                send_future.result()
            response = client.post(
                f"{spec.vllm_base_url}/art/set_served_model_name",
                json={"name": spec.served_model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            torch.cuda.synchronize()
        finally:
            if pause_generation:
                response = client.post(
                    f"{spec.vllm_base_url}/resume",
                    timeout=30.0,
                )
                response.raise_for_status()


def _run_service_loop(runtime: TrainingRuntime) -> None:
    offload_state = OffloadState()
    offload_to_cpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)

    while True:
        torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        job_names = sorted(
            job_name for job_name in os.listdir(jobs_dir) if job_name.endswith(".json")
        )
        if not job_names:
            time.sleep(1)
            continue

        wake_lock_path = "/tmp/megatron_vllm_waking"
        while os.path.exists(wake_lock_path):
            time.sleep(0.2)

        reload_to_gpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)

        job_name = job_names[0]
        job_path = os.path.join(jobs_dir, job_name)
        with open(job_path, "rb") as handle:
            job = load_megatron_job(handle.read())
        train_job = None if isinstance(job, MegatronSyncJob) else job
        if train_job is not None:
            config = train_job.config
            experimental_config = train_job.experimental_config
            configure_moe_routing_replay(
                runtime,
                replay_bundle_path=train_job.moe_routing_replay_path,
                strict=train_job.moe_routing_replay_strict,
            )

        print0(runtime.rank, "Loaded job from", job_path)
        print0(runtime.rank, "Job:", job)

        adapter_model_path = f"{job.lora_path}/adapter_model.safetensors"
        adapter_model = maybe_load_adapter_into_model(
            runtime.model,
            adapter_model_path,
            runtime.optimizer,
            rank=runtime.rank,
        )

        if isinstance(job, MegatronSyncJob):
            _sync_merged_weights_to_vllm(
                runtime,
                job.merged_weight_transfer,
                pause_generation=False,
            )
        else:
            assert train_job is not None
            optimizer_shard_path = os.path.join(
                train_job.optimizer_state_path,
                f"{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.pt",
            )
            if os.path.exists(optimizer_shard_path):
                print("Loading optimizer state from", optimizer_shard_path)
                runtime.optimizer.load_state_dict(torch.load(optimizer_shard_path))
            else:
                print(
                    "No optimizer state found at",
                    optimizer_shard_path,
                    "- resetting optimizer for new run",
                )
                clear_optimizer_state(runtime.optimizer)
                runtime.optimizer.reload_model_params()

            print0(
                runtime.rank,
                "Loading packed tensors from",
                train_job.disk_packed_tensors["dir"],
            )
            packed_tensors = packed_tensors_from_dir(**train_job.disk_packed_tensors)
            template = _clone_packed_tensors(select_indexed_inputs(packed_tensors, 0))
            zero_template = _zero_contribution_inputs(template)
            num_sequences = train_job.disk_packed_tensors["num_sequences"]
            global_grad_accumulation_sequences = config.grad_accumulation_sequences
            num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
            for step_index in range(num_steps):
                micro_indices = build_micro_sample_indices(
                    step_index=step_index,
                    num_sequences=num_sequences,
                    global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                )
                micro_inputs = select_micro_inputs(
                    packed_tensors, micro_indices, zero_template
                )
                step_result = run_training_step(
                    model_chunks=runtime.model,
                    optimizer=runtime.optimizer,
                    learning_rate=config.learning_rate,
                    inputs=micro_inputs,
                    config=config,
                    experimental_config=experimental_config,
                    ref_logprobs=None,
                    step_index=step_index,
                    sample_index=micro_indices,
                    moe_routing_replay_controller=runtime.moe_routing_replay_controller,
                )
                print0(
                    runtime.rank,
                    "Correlation between old and new probabilities:",
                    step_result.probs_corr,
                )

                if runtime.rank == 0:
                    with open(
                        "/tmp/megatron_training_log.jsonl", "a+", encoding="utf-8"
                    ) as log_file:
                        log_msg = json.dumps(
                            {
                                "loss": step_result.reduced_loss.item(),
                                "grad_norm": step_result.grad_norm,
                                "probs_corr": step_result.probs_corr,
                            }
                        )
                        print("Logging", log_msg)
                        log_file.write(log_msg + "\n")

            sharded_state_dict, sharded_state_manifest = collect_sharded_lora_state(
                runtime.model,
                adapter_model,
            )
            shard_path = os.path.join(
                job.lora_path,
                f"adapter_model-{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.safetensors",
            )
            manifest_path = os.path.join(
                job.lora_path,
                f"adapter_manifest-{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.json",
            )
            print("Saving adapter shard to", shard_path)
            save_file(sharded_state_dict, shard_path)
            print("Saving adapter shard manifest to", manifest_path)
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(sharded_state_manifest, manifest_file, sort_keys=True)

            print("Saving optimizer shard to", optimizer_shard_path)
            os.makedirs(train_job.optimizer_state_path, exist_ok=True)
            torch.save(runtime.optimizer.state_dict(), optimizer_shard_path)

            if isinstance(train_job, MegatronMergedTrainJob):
                _sync_merged_weights_to_vllm(
                    runtime,
                    train_job.merged_weight_transfer,
                    pause_generation=True,
                )

        offload_to_cpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)

        if train_job is not None:
            del packed_tensors
            del template
            del zero_template
            if "micro_inputs" in locals():
                del micro_inputs
        del adapter_model
        gc.collect()
        torch.cuda.empty_cache()

        torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        if runtime.rank == 0:
            os.remove(job_path)
            with open(
                "/tmp/megatron_training_log.jsonl", "a+", encoding="utf-8"
            ) as log_file:
                log_file.write("all done\n")
            if train_job is not None:
                shutil.rmtree(train_job.disk_packed_tensors["dir"])


def main() -> None:
    runtime = build_training_runtime(
        model_identifier=os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER)
    )
    _run_service_loop(runtime)


if __name__ == "__main__":
    main()
