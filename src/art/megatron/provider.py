import copy
from functools import partial
import inspect
from pathlib import Path
from typing import Callable, cast

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.state import (
    SafeTensorsStateSource,
    StateDict,
    StateSource,
)
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
import torch

from art.megatron.flex_attention import FlexDotProductAttention


def _resolve_layer_spec(
    base_layer_spec: ModuleSpec | Callable[[GPTModelProvider], ModuleSpec],
    config: GPTModelProvider,
    vp_stage: int | None = None,
) -> ModuleSpec:
    if isinstance(base_layer_spec, ModuleSpec):
        return copy.deepcopy(base_layer_spec)
    kwargs = (
        {"vp_stage": vp_stage}
        if vp_stage in inspect.signature(base_layer_spec).parameters
        else {}
    )
    return base_layer_spec(config, **kwargs)


class _CastingStateSource(StateSource):
    def __init__(self, source: StateSource, *, dtype: torch.dtype):
        self._source = source
        self._dtype = dtype

    def get_all_keys(self) -> list[str]:
        return self._source.get_all_keys()

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        loaded = self._source.load_tensors(keys)
        return {
            key: (
                value.to(dtype=self._dtype)
                if torch.is_floating_point(value) and value.dtype != self._dtype
                else value
            )
            for key, value in loaded.items()
        }

    def has_glob(self, pattern: str) -> bool:
        return self._source.has_glob(pattern)


def get_provider(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> GPTModelProvider:
    bridge = AutoBridge.from_hf_pretrained(
        model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    assert isinstance(bridge._model_bridge, Qwen3MoEBridge), (
        "Only Qwen3 MoE models are supported"
    )
    if torch_dtype != torch.bfloat16:
        model_name_or_path = bridge.hf_pretrained.model_name_or_path
        assert model_name_or_path is not None
        bridge.hf_pretrained._state_dict_accessor = StateDict(
            _CastingStateSource(
                SafeTensorsStateSource(cast(str | Path, model_name_or_path)),
                dtype=torch_dtype,
            )
        )
    provider = bridge.to_megatron_provider()
    setattr(provider, "art_bridge", bridge)
    base_layer_spec = provider.transformer_layer_spec

    def _flex_attention_layer_spec(
        config: GPTModelProvider, vp_stage: int | None = None
    ) -> ModuleSpec:
        layer_spec = _resolve_layer_spec(base_layer_spec, config, vp_stage)
        # Keep Megatron's standard layer stack and replace only core attention.
        layer_spec.submodules.self_attention.submodules.core_attention = (  # ty: ignore[unresolved-attribute]
            FlexDotProductAttention
        )
        return layer_spec

    provider.transformer_layer_spec = _flex_attention_layer_spec
    provider.attention_backend = AttnBackend.auto
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.tensor_model_parallel_size = min(2, torch.cuda.device_count())
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = torch.cuda.device_count()
    provider.expert_tensor_parallel_size = 1
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    # params are disabled anyways, but should know about this if we switch to full FT
    # because DP 'dummy' microbatches will unintentionally have loss for this
    provider.moe_aux_loss_coeff = 0.0
    # effectively just a flag modifying finalize_model_grads behavior for DPxCP
    provider.calculate_per_token_loss = True
    if provider.tensor_model_parallel_size > 1:
        provider.sequence_parallel = True
    provider.finalize()
    return provider
