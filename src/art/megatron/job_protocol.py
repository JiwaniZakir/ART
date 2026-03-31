from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter

from art import dev, types
from art.megatron.routing_replay import MoeRoutingReplayBundle
from art.preprocessing.pack import DiskPackedTensors


class MergedWeightTransferInitInfo(BaseModel):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


class MergedWeightTransferSpec(BaseModel):
    init_info: MergedWeightTransferInitInfo
    vllm_base_url: str
    served_model_name: str


class MegatronSyncJob(BaseModel):
    kind: Literal["sync"]
    lora_path: str
    merged_weight_transfer: MergedWeightTransferSpec


class _MegatronTrainJobBase(BaseModel):
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig
    moe_routing_replay_path: str | None = None
    moe_routing_replay_strict: bool = True


class MegatronLoraTrainJob(_MegatronTrainJobBase):
    kind: Literal["train_lora"]


class MegatronMergedTrainJob(_MegatronTrainJobBase):
    kind: Literal["train_merged"]
    merged_weight_transfer: MergedWeightTransferSpec


MegatronLoraTrainJob.model_rebuild(
    force=True,
    _types_namespace={"MoeRoutingReplayBundle": MoeRoutingReplayBundle},
)
MegatronMergedTrainJob.model_rebuild(
    force=True,
    _types_namespace={"MoeRoutingReplayBundle": MoeRoutingReplayBundle},
)

MegatronJob: TypeAlias = Annotated[
    MegatronSyncJob | MegatronLoraTrainJob | MegatronMergedTrainJob,
    Field(discriminator="kind"),
]


def dump_megatron_job(job: MegatronJob) -> str:
    return TypeAdapter(MegatronJob).dump_json(job).decode()


def load_megatron_job(raw: str | bytes) -> MegatronJob:
    return TypeAdapter(MegatronJob).validate_json(raw)
