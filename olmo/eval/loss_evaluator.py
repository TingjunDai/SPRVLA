"""Class to build metrics for a model based on the loss"""
import dataclasses
import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import Dict, Optional, Union

import torch
import torchmetrics
import wandb
from torch.utils.data import DataLoader
from torchmetrics import Metric, MeanMetric
from tqdm import tqdm
from wandb.sdk.data_types.base_types.wb_value import WBValue

from olmo.config import BaseConfig, D
from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.save_eval_data_config import SaveEvalDataConfig
from olmo.models.sprvla.sprvla import SPRVLAConfig
from olmo.torch_util import move_to_device, get_world_size

__all__ = ["LossMetrics", "LossDatasetEvaluator", "LossDatasetEvaluatorConfig"]

log = logging.getLogger(__name__)


class LossMetrics:
    """Aggregates loss metrics from a forward pass"""

    def __init__(self, device, collect_outputs=False):
        self.eval_metrics: Dict[str, MeanMetric] = dict(
            CrossEntropyLoss=MeanMetric("error").to(device),
            ZLoss=MeanMetric("error").to(device),
            Accuracy=MeanMetric("error").to(device),
        )

    def reset(self) -> None:
        if isinstance(self.eval_metrics, Metric):
            self.eval_metrics.reset()
        else:
            for metric in self.eval_metrics.values():
                metric.reset()

    def compute(self) -> Dict[str, Union[float, WBValue]]:
        metrics = {}
        for k, v in self.eval_metrics.items():
            if k in ["HighResSelection", "HighResVals"]:
                metrics[k] = wandb.Histogram(v.compute().detach().cpu().numpy(), num_bins=100)
            elif v.weight > 0:
                metrics[k] = v.compute().item()
        return metrics

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        model_out,
        cross_entropy_loss: torch.Tensor,
        zloss: torch.Tensor
    ) -> None:
        loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
        total_weight = loss_masks.sum()
        labels = batch["labels"]
        pred = torch.argmax(model_out.logits, dim=-1)
        accuracy = ((pred.flatten() == labels.flatten()).float() * loss_masks.flatten()).sum().item()
        self.eval_metrics["CrossEntropyLoss"].update(cross_entropy_loss/total_weight, total_weight)
        if zloss is not None:
            self.eval_metrics["ZLoss"].update(zloss/total_weight, total_weight)
        self.eval_metrics["Accuracy"].update(accuracy/total_weight, total_weight)
        if model_out.metrics is not None:
            for name, val in model_out.metrics.items():
                if name in ["HighResSelection", "HighResVals"]:
                    if name not in self.eval_metrics:
                        self.eval_metrics[name] = torchmetrics.CatMetric("error")
                    self.eval_metrics[name].update(val)
                else:
                    if name not in self.eval_metrics:
                        self.eval_metrics[name] = MeanMetric("error").to(cross_entropy_loss.device)
                    try:
                        if isinstance(val, tuple):
                            self.eval_metrics[name].update(val[0]/val[1], val[1])
                        else:
                            self.eval_metrics[name].update(val, 1)
                    except Exception as e:
                        e.add_note(f"Error processing metric {name}")
                        raise e


@dataclass
class LossDatasetEvaluator:
    """Evaluates a model on a dataset based on its loss and other forward-pass metrics"""
    label: str
    eval_loader: DataLoader
    evaluator: LossMetrics
    num_batches: Optional[int] = None
    console_log_interval: Optional[int] = None
    z_loss: Optional[float] = None
    save_data: Optional[SaveEvalDataConfig] = None

    def run(self, model, device, autocast_precision, loss_fn=None, pbar=False):
        # Reset metrics.
        self.evaluator.reset()
        if loss_fn is None:
            from olmo.train.trainer import cross_entropy_loss as loss_fn

        # Initialize data loader iterator.
        eval_batches = iter(self.eval_loader)

        # Adjust how many batches to evaluate on.
        num_eval_batches = self.num_batches
        if num_eval_batches > 0:
            if isinstance(self.eval_loader, torch.utils.data.IterableDataset):
                num_eval_batches = None  # No defined length
            else:
                num_eval_batches = min(num_eval_batches, len(self.eval_loader))
            eval_batches = islice(eval_batches, num_eval_batches)

        # Run model over batches.
        viz_data = []
        with torch.inference_mode():
            for eval_step, batch in enumerate(tqdm(eval_batches, total=num_eval_batches, disable=not pbar)):
                batch = move_to_device(batch, device)
                response_mask = (batch["loss_masks"] > 0)
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    inputs = {k: v for k, v in batch.items() if k not in ["labels", "loss_masks", "metadata"]}
                    model_out = model(**inputs, response_mask=response_mask)
                logits = model_out.logits
                loss_masks = batch["loss_masks"]
                loss_masks = loss_masks * (loss_masks > 0)
                labels = batch["labels"].long()
                labels.masked_fill_(~(loss_masks > 0), -100)
                labels = labels.view(-1)
                logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
                ce_loss, z_loss = loss_fn(
                    logits_for_loss, labels, ignore_index=-100, reduction="none",
                    compute_z_loss=self.z_loss is not None, z_loss_scale=self.z_loss,
                )
                ce_loss = (ce_loss * loss_masks.view(-1)).sum()
                if z_loss is not None:
                    z_loss = (z_loss * loss_masks.view(-1)).sum()
                self.evaluator.update(batch, model_out, ce_loss, z_loss)

                # Maybe save internal data
                if self.save_data:
                    for i in range(len(response_mask)):
                        saved_data = {}
                        if self.save_data.example_metadata:
                            saved_data["example_metadata"] = batch["metadata"][i]
                        if self.save_data.post_processed_inputs:
                            saved_data["post_processed_inputs"] = {k: v[i].detach().cpu() for k, v in inputs.items()}
                        if self.save_data.model_internal_data:
                            saved_data["model_internal_data"] = {k: v[i].detach().cpu() for k, v in model_out.internal.items()}
                        viz_data.append(saved_data)

                if self.console_log_interval and not pbar:
                    if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.console_log_interval == 0:
                        log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")
        if self.save_data:
            return self.evaluator.compute(), viz_data
        else:
            return self.evaluator.compute()


@dataclass
class LossDatasetEvaluatorConfig(BaseConfig):
    """Configuration for a loss evaluation"""

    label: Optional[str] = None
    """Label to use when logging"""

    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    device_batch_size: int = 4
    """Batch size, can default to the eval batch set set in the global config"""

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    console_log_interval: Optional[int] = None
    """How often to log progress to console"""

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = config.copy()
        if getattr(config, "mm_evaluator", None):
            config.generative_evaluator = LossDatasetEvaluatorConfig.update_legacy_settings(config.generative_evaluator)
        if getattr(config, "data", None):
            config.data = DataLoaderConfig.update_legacy_settings(config.data)
        return config

    def build_dataset_evaluator(self, model_config: SPRVLAConfig, device, save_data: SaveEvalDataConfig=None) -> LossDatasetEvaluator:
        eval_loader = self.data.build_eval_dataloader(
            model_config, self.device_batch_size, for_inference=False,
            include_metadata=save_data and save_data.example_metadata
        )
        if self.max_examples is not None:
            num_batches = max(1, self.max_examples // (self.device_batch_size*get_world_size()))
        elif self.subset_num_batches is not None:
            num_batches = self.subset_num_batches
        else:
            num_batches = len(eval_loader)

        return LossDatasetEvaluator(
            label=self.label,
            eval_loader=eval_loader,
            evaluator=LossMetrics(device),
            num_batches=num_batches,
            console_log_interval=self.console_log_interval,
            save_data=save_data
        )
