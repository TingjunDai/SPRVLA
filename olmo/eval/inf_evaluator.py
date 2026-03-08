"""Class to evaluate models based on their generation outputs"""
import dataclasses
import itertools
import logging
from collections import defaultdict
from typing import List, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics
import wandb
from tqdm import tqdm

from .evaluators import (
    HtmlTable, CountEval, PointCountEval, ClockEval, VqaEval,
    SavePredictions, AndroidControlEval, MathVistaEval, PointingEval,
)
from ..config import BaseConfig
from ..data.data_loader import DataLoaderConfig
from ..torch_util import (
    get_global_rank,
    get_world_size,
    move_to_device,
)
from ..util import flatten_list

__all__ = ["InfEvaluator", "EvaluatorConfig", "InfDatasetEvaluator", "InfDatasetEvaluatorConfig"]

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InfEvaluator:
    """
    Evaluates the text outputs from a model on a task
    """
    metrics: List

    def __call__(self, predictions, example_metadata, tokenizer, device, step=None):
        inf_metrics = {}
        for metric in self.metrics:
            results = metric(example_metadata, predictions, step=step, tokenizer=tokenizer)
            assert all(k not in inf_metrics for k in results)
            inf_metrics.update(results)

        resolved_metrics = {}
        # sort so metrics are iterated on in the same order on all devices
        for k in sorted(inf_metrics):
            v = inf_metrics[k]
            if isinstance(v, torchmetrics.Metric):
                resolved_metrics[k] = v.to(device).compute().item()
            elif isinstance(v, HtmlTable):
                # Special case, we aggregate table rows from all devices to ensure we can always
                # have enough rows to show even if each device only eval-ed a few examples
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    all_rows = flatten_list([x.rows for x in all_predictions])
                    resolved_metrics[k] = wandb.Html(HtmlTable(all_rows).get_html())
                else:
                    dist.gather_object(v, None)
            elif isinstance(v, List):
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(v, all_predictions)
                    resolved_metrics[k] = []
                    for pred in all_predictions:
                        resolved_metrics[k] += pred
                else:
                    dist.gather_object(v, None)
            else:
                raise ValueError(f"Metric {v} not understood")

        for metric in self.metrics:
            if isinstance(metric, (CountEval, PointCountEval)):
                # Counting has a macro-score that should be computed once we have
                # scores from all devices
                counting_scores = {k: resolved_metrics[k] for
                                   k in list(resolved_metrics.keys()) if k.startswith("correct_")}
                resolved_metrics["per_category_average"] = np.mean(list(counting_scores.values()))
        return resolved_metrics


@dataclasses.dataclass
class EvaluatorConfig(BaseConfig):
    """Config for `Evaluator` objects that compute metrics"""

    n_to_log: int = 10
    """Num examples to log to console"""

    num_wandb_examples: int = 0
    """Num examples to log to Wandb as a HTML table"""

    save_predictions: Optional[str] = "_default"  # saves with default name to checkpoint dir
    """Where to save predictions files"""

    save_tokens: bool = False
    """If save predictions, should the tokens be saved"""

    vqa_eval: str = ''
    """name(s) of VQA-style eval to run, can be a comma seperated list"""

    # Other individual types of eval
    pointing_eval: bool = False
    count_eval: bool = False
    point_count_eval: bool = False
    android_eval: bool = False
    clock_eval: bool = False
    clock_bench_eval: bool = False # Clock reading benchmark, coco/openimg/movies
    math_vista_eval: bool = False
    temp_compass_eval: str = ''
    """TempCompass tasks to run evaluation on, either one of the tasks or 'all'"""
    temp_compass_disable_api: bool = False
    """Whether not to use ChatGPT evaluation for TempCompass"""

    def build(self, default_save_dir=None) -> InfEvaluator:
        evaluators = []
        save_predictions = self.save_predictions
        if save_predictions == "_default":
            if default_save_dir is None:
                logging.info(f"save_predictions is \"default\" but no default "
                             f"save dir set so predictions will not be saved")
            save_predictions = default_save_dir
        if save_predictions:
            evaluators.append(SavePredictions(
                save_predictions,
                log_examples=self.n_to_log,
                save_tokens=self.save_tokens
            ))

        if self.vqa_eval:
            evaluators.append(VqaEval(self.vqa_eval.split(","), self.num_wandb_examples))
        elif self.clock_eval:
            evaluators.append(ClockEval(self.num_wandb_examples))
        elif self.clock_bench_eval:
            evaluators.append(ClockEval(self.num_wandb_examples, is_test=True))
        elif self.math_vista_eval:
            evaluators.append(MathVistaEval(self.num_wandb_examples))
        elif self.point_count_eval:
            evaluators.append(PointCountEval(self.num_wandb_examples))
        elif self.count_eval:
            evaluators.append(CountEval(self.num_wandb_examples))
        elif self.android_eval:
            evaluators.append(AndroidControlEval(self.num_wandb_examples))
        else:
            pass
        return InfEvaluator(evaluators)


@dataclasses.dataclass
class InfDatasetEvaluator:
    """Evaluates a model on a dataset"""
    label: str
    dataloader: Any
    evaluator: InfEvaluator
    n_steps: int
    max_new_tokens: int = 448
    console_log_interval: Optional[int] = None

    def run(self, model, device, autocast_precision, is_distributed, pbar=False):
        eval_dataloader = self.dataloader
        eval_it = iter(eval_dataloader)
        n_steps = self.n_steps
        if n_steps is not None and 0 <= n_steps < len(self.dataloader):
            eval_it = itertools.islice(eval_it, 0, n_steps)
            total_steps = n_steps
        else:
            total_steps = len(eval_dataloader)

        all_metadata = []
        predictions = defaultdict(list)
        done_init = False
        pbar = pbar and get_global_rank() == 0
        for eval_step, batch in enumerate(tqdm(eval_it, total=total_steps, ncols=100, disable=not pbar)):
            if "metadata" in batch:
                batch_metadata = batch.pop("metadata")
            else:
                # Handle old-style data that used metadata/ prefix instead
                metadata = {k: batch.pop(k) for k in list(batch) if k.startswith("metadata/")}
                batch_metadata = []
                for i in range(len(batch["input_ids"])):
                    converted = {}
                    for k, v in metadata.items():
                        if isinstance(v[i], bytes):
                            converted[k] = v[i].decode("utf-8")
                        else:
                            converted[k] = v[i].tolist()
                    batch_metadata.append(converted)

            batch_inference = move_to_device(batch, device)
            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    olmo_gen_output = model.generate(
                        batch=batch_inference,
                        max_steps=self.max_new_tokens,
                        is_distributed=is_distributed
                    )

            pred = {
                "predictions": olmo_gen_output.token_ids[:, 0].detach().cpu().numpy(), # beam size of 1
                "prompts": batch_inference["input_ids"].detach().cpu().numpy(),
            }

            if hasattr(olmo_gen_output, 'internal') and olmo_gen_output.internal is not None and 'bmm' in olmo_gen_output.internal:
                pred['bmm'] = olmo_gen_output.internal['bmm'].to(torch.float32).detach().cpu().numpy()
                pred['high_res_indices'] = olmo_gen_output.internal['high_res_indices'].to(torch.float32).detach().cpu().numpy()

            valid_ixs = [i for i, md in enumerate(batch_metadata) if md.get("valid", True)]
            all_metadata += [batch_metadata[i] for i in valid_ixs]
            for k, v in pred.items():
                for ix in valid_ixs:
                    predictions[k].append(v[ix])

            # Log to console.
            if self.console_log_interval and not pbar:
                if eval_step + 1 == n_steps or (eval_step + 1) % self.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{total_steps}]")

        tokenizer = model.config.build_tokenizer()
        metrics = self.evaluator(predictions, all_metadata, tokenizer, device)
        return metrics


@dataclasses.dataclass
class InfDatasetEvaluatorConfig(BaseConfig):
    """Configuration for an inference evaluator"""

    label: Optional[str] = None
    """Label to use when logging"""

    data: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    evaluator: EvaluatorConfig = dataclasses.field(default_factory=EvaluatorConfig)
    """Evaluator to compute metrics from the generated outputs"""

    max_new_tokens: int = 448
    """Max number of tokens to generate"""

    device_batch_size: int = 4
    """Batch size"""

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    console_log_interval: Optional[int] = None
    """How often to log progress to console"""

    include_image: bool = False
    """Include image in the metadata"""

    def build_dataset_evaluator(
        self,
        model_config,
        default_save_dir,
        device,
    ) -> InfDatasetEvaluator:
        global_batch_size = self.device_batch_size * get_world_size()
        if self.max_examples and self.max_examples > 0:
            max_steps = max(self.max_examples // global_batch_size, 1)
        elif self.subset_num_batches:
            max_steps = self.subset_num_batches
        else:
            max_steps = None

        eval_loader = self.data.build_eval_dataloader(
            model_config,
            self.device_batch_size,
            for_inference=True,
            pad_batches=True,
            max_steps_for_padding=max_steps,
            include_image=self.include_image,
        )
        if self.max_examples is not None:
            num_batches = self.max_examples // self.device_batch_size*get_world_size()
        elif self.subset_num_batches is not None:
            num_batches = self.subset_num_batches
        else:
            num_batches = len(eval_loader)

        return InfDatasetEvaluator(
            label=self.label,
            dataloader=eval_loader,
            evaluator=self.evaluator.build(default_save_dir),
            n_steps=max_steps,
            max_new_tokens=self.max_new_tokens,
            console_log_interval=self.console_log_interval
        )
