import copy
from dataclasses import dataclass
import inspect
from pathlib import Path
from types import MethodType
from typing import Any, Callable, cast

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.state import (
    SafeTensorsStateSource,
    StateDict,
    StateSource,
)
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
    Qwen3VLSelfAttention,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen35VLMoEModelProvider,
    _patch_standard_attention_specs,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
import torch

from art.megatron.flex_attention import FlexDotProductAttention


@dataclass(frozen=True)
class ProviderBundle:
    provider: GPTModelProvider
    bridge: Any


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


def _patch_core_attention(layer_spec: object) -> None:
    submodules = getattr(layer_spec, "submodules", None)
    self_attention = getattr(submodules, "self_attention", None)
    attention_submodules = getattr(self_attention, "submodules", None)
    if attention_submodules is None or not hasattr(
        attention_submodules, "core_attention"
    ):
        return
    attention_submodules.core_attention = FlexDotProductAttention


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


def get_provider_bundle(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> ProviderBundle:
    bridge = AutoBridge.from_hf_pretrained(
        model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    assert isinstance(bridge._model_bridge, (Qwen3MoEBridge, Qwen35VLMoEBridge)), (
        "Only Qwen3 and Qwen3.5 MoE models are supported"
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
    if isinstance(provider, Qwen35VLMoEModelProvider):
        from megatron.bridge.models.gpt_provider import mtp_block_spec

        def _patch_qwen35_block_spec(block_spec: TransformerBlockSubmodules) -> None:
            _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)
            for layer_spec in block_spec.layer_specs:
                _patch_core_attention(layer_spec)

        def _qwen35_layer_spec(
            config: GPTModelProvider, vp_stage: int | None = None
        ) -> TransformerBlockSubmodules:
            block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config,
                vp_stage=vp_stage,
            )
            _patch_qwen35_block_spec(block_spec)
            return block_spec

        provider.transformer_layer_spec = _qwen35_layer_spec

        def _provide_qwen35_with_flex_attention(
            self: Qwen35VLMoEModelProvider,
            pre_process: bool | None = None,
            post_process: bool | None = None,
            vp_stage: int | None = None,
        ) -> Qwen3VLModel:
            language_transformer_config = self
            hf_vision_config = self.vision_config
            hf_vision_config.torch_dtype = self.params_dtype
            block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                language_transformer_config,
                vp_stage=vp_stage,
            )
            _patch_qwen35_block_spec(block_spec)
            model = Qwen3VLModel(
                language_transformer_config=language_transformer_config,
                language_transformer_layer_spec=block_spec,
                vision_transformer_config=hf_vision_config,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=self._pg_collection,
                mtp_block_spec=mtp_block_spec(self, vp_stage=vp_stage),
                vp_stage=vp_stage,
            )
            if (
                self.freeze_language_model
                or self.freeze_vision_model
                or self.freeze_vision_projection
            ):
                model.freeze(
                    freeze_language_model=self.freeze_language_model,
                    freeze_vision_model=self.freeze_vision_model,
                    freeze_vision_projection=self.freeze_vision_projection,
                )
            return model

        provider.provide = MethodType(_provide_qwen35_with_flex_attention, provider)
    base_layer_spec = provider.transformer_layer_spec

    def _flex_attention_layer_spec(
        config: GPTModelProvider, vp_stage: int | None = None
    ) -> ModuleSpec:
        layer_spec = _resolve_layer_spec(base_layer_spec, config, vp_stage)
        layer_specs = getattr(layer_spec, "layer_specs", None)
        if layer_specs is None:
            _patch_core_attention(layer_spec)
        else:
            for block_layer_spec in layer_specs:
                _patch_core_attention(block_layer_spec)
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
    # ART computes its own RL loss, so MTP only adds incompatible postprocess work.
    provider.mtp_enabled = False
    provider.mtp_num_layers = 0
    if provider.tensor_model_parallel_size > 1:
        provider.sequence_parallel = True
    provider.finalize()
    return ProviderBundle(provider=provider, bridge=bridge)


def get_provider(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> GPTModelProvider:
    return get_provider_bundle(model, torch_dtype=torch_dtype).provider
