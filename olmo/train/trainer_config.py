from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Union, cast,
)

import omegaconf
from omegaconf import OmegaConf as om
import torch
from omegaconf.errors import OmegaConfBaseException
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, MixedPrecisionPolicy

from olmo.config import BaseConfig, StrEnum
from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.inf_evaluator import InfDatasetEvaluatorConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.io import PathOrStr, read_file
from olmo.models.model_config import BaseModelConfig, get_model_types
from olmo.train.checkpointer import CheckpointerConfig
from olmo.models.sprvla.sprvla import SPRVLA
from olmo.models.model import FSDPWrapStrategy
from olmo.torch_util import get_local_world_size, get_world_size
from olmo.train.optim import OptimizerConfig, SchedulerConfig

__all__ = [
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "FSDPPrecision",
    "FSDPConfig",
]

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


log = logging.getLogger("trainer")


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = "ai2-llm"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1
    allow_resume: bool = False


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = "default"
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    dynamic: bool = False

    backend: str = "inductor"
    """
    The backend to use.
    """

    def compile_args(self):
        return self.asdict()


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """

    float = "float"


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


@dataclass
class FSDPConfig(BaseConfig):
    fsdp2: bool = True

    precision: FSDPPrecision = FSDPPrecision.pure

    # These other factors only affect FSDP1

    use_orig_params: bool = True

    wrapping_strategy: Optional[FSDPWrapStrategy] = None

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    hybrid_sharding_num_model_replicas: Optional[int] = None
    """
    The number of model instances, when using a hybrid sharding strategy.
    If not ``None``, this must divide the total number of nodes. If ``None``, the default,
    a model instance is used per node (as determined by ``get_world_size() // get_local_world_size()``).
    PyTorch's default HSDP behavior matches this default behavior.
    """

    def get_fsd_args(self, autocast_precision) -> Dict[str, Any]:
        if self.precision == FSDPPrecision.pure:
            mp = MixedPrecision(
                param_dtype=autocast_precision,
                reduce_dtype=autocast_precision,
                buffer_dtype=autocast_precision,
            )
        elif self.precision == FSDPPrecision.mixed:
            mp = MixedPrecision(
                param_dtype=autocast_precision,
                reduce_dtype=torch.float32,
                buffer_dtype=autocast_precision,
            )
        elif self.precision == FSDPPrecision.float:
            mp = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"{self.precision}")

        if self.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            num_model_replicas = self.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )

            if num_model_replicas <= 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")
            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
        else:
            # Given an explicit device mesh so FSDP uses DTensors, avoiding a checkpointing issue:
            # https://github.com/pytorch/pytorch/issues/132366#issuecomment-2264642034
            device_mesh = init_device_mesh("cuda", (get_world_size(),))
        return dict(
            device_mesh=device_mesh,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=mp,
            use_orig_params=self.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
            limit_all_gathers=True,
        )

    def get_fsd2_args(self, autocast_precision) -> Dict:
        if self.hybrid_sharding_num_model_replicas:
            raise NotImplementedError()

        if self.precision == FSDPPrecision.pure:
            mp = MixedPrecisionPolicy(
                param_dtype=autocast_precision,
                reduce_dtype=autocast_precision,
            )
        elif self.precision == FSDPPrecision.mixed:
            mp = MixedPrecisionPolicy(
                param_dtype=autocast_precision,
                reduce_dtype=torch.float32,
            )
        elif self.precision == FSDPPrecision.float:
            mp = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"{self.precision}")
        return dict(mp_policy=mp)


class BatchDivisor(StrEnum):
    global_batch = "global_batch"
    device_batch = "device_batch"
    instance = "instance"


@dataclass
class RuntimeData(BaseConfig):
    args: str
    hostname: str
    date: str
    world_size: int
    resuming_from: Optional[str]
    beaker_experiment_id: Optional[str]
    beaker_experiment_url: Optional[str]
    wandb_id: Optional[str]
    wandb_url: Optional[str]


@dataclass
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    Run name, used when logging 
    """

    model: BaseModelConfig = omegaconf.MISSING
    """
    Model to train
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    epoch: Optional[int] = None
    """
    Increment this when starting a new epoch.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    ft_llm: bool = True
    """
    Tune the LLM parameters
    """

    ft_vit: bool = True
    """
    Tune the image encoder parameters
    """

    ft_connector: bool = True
    """
    Tune the V/L connector parameters
    """

    # Do we fine-tune the input/output embeddings
    ft_embedding: str = "lm_head"
    """
    Tune the embedding layers
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When resuming, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When resuming, use this to fast-forward the dataloader beyond the last checkpoint.
    """

    evaluators: List[LossDatasetEvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 1000
    """
    How often (in terms of batches) to run evaluations.
    """

    inf_evaluators: List[InfDatasetEvaluatorConfig] = field(default_factory=list)
    """
    Inference Evaluation configurations.
    """

    inf_eval_interval: Optional[int] = -1
    """
    How often (in terms of batches) to run inference evaluations
    """

    eval_on_last_step: bool = True
    """Always run evaluations at the last step"""

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    checkpointer_config: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    """Checkpointing configuration."""

    canceled_check_interval: int = 50
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
    """

    save_interval: int = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_final_optim: bool = True
    """
    Save the final optimizer state
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    save_intermediate_unsharded_checkpoint: bool = False
    """Save an unsharded checkpoint at every intermediate saving step"""

    save_final_unsharded_checkpoint: bool = False
    """Save an unsharded checkpoint at the end of training"""

    save_every_n_epoch: Optional[float] = None
    """Save checkpoints at the end of every n training epoch"""

    save_interval_epoch: int = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite existing files
    """

    load_path: Optional[str] = None
    """
    The path to a sharded or unshared checkpoint to start from.
    """

    reset_optimizer_state: bool = False
    """
    Don't load try and load optimizer state from `load_path`
    """

    reset_trainer_state: bool = False
    """
    Don't load and load train state from `load_path`
    """

    initial_model_checkpoint: Optional[str] = None
    """
    Path to a checkpoint to use to initialize the model from, overriden by `load_path`
    """

    allow_resume: bool = False
    """
    Try to resume training if a checkpoint already exists in the checkpoint directory
    """

    max_duration: Union[int, str] = 10000
    """
    How long to train for.

    If specified without a unit (the default), the units are assumed to be steps.
    You can also specify this in terms of tokens, for example: `max_duration="2e12T"` means train until
    2 trillion tokens.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_microbatch_size: int = 16
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradient norms to this value if set.
    """

    multi_component_grad_norm: bool =True
    """
    Use separate grad norm for each component in multi-modal model
    """

    batch_divisor: Optional[BatchDivisor] = BatchDivisor.global_batch
    """
    How loss is normalized in distributed settings
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    beaker_log_interval: int = 50
    """
    How often to update beaker description with run progress 
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    gen1_gc_interval: Optional[int] = 1
    """
    How often (in steps) to run generation 1 garbage collection.
    Set to ``None`` to use automatic garbage collection (i.e. we don't mess with it).
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    activation_checkpointing: bool = True
    """
    Enable activation checkpointing
    """

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0 (z-loss).
    """

    softmax_auxiliary_loss_scale: float = 1e-4
    """
    The scale of the auxiliary loss function (z-loss).
    """

    time_limit: Optional[float] = None
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    """

    extra_steps_after_cancel: int = 10
    """
    Under certain conditions when a run is canceled we train for a few extra steps after saving
    the final checkpoint so that when the run is restarted from the latest checkpoint we have some
    overlap in metrics.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[Union[int, str]] = None
    """
    Stop at a specific step.
    """

    stop_after: Optional[int] = None
    """
    Stop after a specific number of steps.
    """

    fused_loss: Optional[bool] = None
    """
    Whether to use the fused CE loss function from `flash-attn`.
    """

    compile_loss: bool = False
    """
    Whether to compile the loss function
    """

    runtime_data: Optional[RuntimeData] = None
    """
    Data about the current run, filled in automatically 
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @classmethod
    def load(
        cls,
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        schema = om.structured(cls)
        try:
            raw = om.create(read_file(path))
            if key is not None:
                raw = raw[key]  # type: ignore

            # Make sure the schema has the correct model class
            model_name = raw.model.get("model_name", "sprvla")
            model_cls = get_model_types()[model_name]
            schema.model = om.structured(model_cls)

            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(TrainConfig, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(e)
