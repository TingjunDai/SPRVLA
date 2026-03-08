import argparse
import os
import logging

import torch
import torch.nn as nn
from typing import Any, Dict, Union, Optional

from olmo.models.sprvla.sprvla import SPRVLA, SPRVLAConfig
from olmo.train.checkpointer import load_model_state, save_unsharded
from olmo.util import (
    prepare_cli_environment,
    resource_path
)

from olmo.train.trainer_config import TrainConfig


logger = logging.getLogger(__name__)


def count_parameters(
    module_or_cls: Union[nn.Module, type],
    *args: Any,
    trainable_only: bool = False,
    detailed: bool = False,
    **kwargs: Any,
) -> Union[int, Dict[str, Any]]:
    """
    Count parameters in a PyTorch nn.Module.

    Args:
        module_or_cls: An nn.Module instance OR an nn.Module subclass.
        *args/**kwargs: If a class is given, these are passed to its constructor.
        trainable_only: If True, count only parameters with requires_grad=True.
        detailed: If True, return a dict with totals and memory info.

    Returns:
        If detailed=False: an int (number of parameters).
        If detailed=True: a dict with keys:
            - total
            - trainable
            - frozen
            - param_bytes (sum of parameter storage in bytes)
            - by_dtype (mapping torch.dtype -> param count)
    """
    if isinstance(module_or_cls, nn.Module):
        model = module_or_cls
    elif isinstance(module_or_cls, type) and issubclass(module_or_cls, nn.Module):
        model = module_or_cls(*args, **kwargs)
    else:
        raise TypeError("module_or_cls must be an nn.Module instance or subclass.")

    params = list(model.parameters())

    if trainable_only:
        count = sum(p.numel() for p in params if p.requires_grad)
        if not detailed:
            return count
    else:
        count = sum(p.numel() for p in params)
        if not detailed:
            return count

    # Detailed breakdown
    trainable = sum(p.numel() for p in params if p.requires_grad)
    frozen = count - trainable
    param_bytes = sum(p.numel() * p.element_size() for p in params)

    by_dtype: Dict[torch.dtype, int] = {}
    for p in params:
        by_dtype[p.dtype] = by_dtype.get(p.dtype, 0) + p.numel()

    return {
        "total": count,
        "trainable": trainable,
        "frozen": frozen,
        "param_bytes": param_bytes,
        "by_dtype": by_dtype,
    }

def format_param_count(n_params: int, unit: str = "M", decimals: Optional[int] = None) -> str:
    """
    Format a raw parameter count as 'XM' or 'Y.BB'.
      - unit="M": millions, default decimals=0 (round to integer)
      - unit="B": billions, default decimals=1 (round to 1 decimal)
    """
    u = unit.upper()
    if u == "M":
        value = n_params / 1_000_000
        decimals = 0 if decimals is None else decimals
        return f"{value:.{decimals}f}M"
    elif u == "B":
        value = n_params / 1_000_000_000
        decimals = 1 if decimals is None else decimals
        return f"{value:.{decimals}f}B"
    else:
        raise ValueError("unit must be 'M' or 'B'")


def load_model_and_calculate(checkpoint_dir: str) -> None:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(checkpoint_dir, "config.yaml")
    config: TrainConfig = TrainConfig.load(config_path)
    # model_config: SPRVLAConfig = SPRVLAConfig.load(config_path, key="model", validate_paths=False)
    model_config = config.model


    logger.info(f"Loading model checkpoint from {checkpoint_dir}")
    with torch.device("meta"):
        model: SPRVLA = model_config.build_model()
    model.to_empty(device=torch.device("cpu"))
    # load_model_state(checkpoint_dir, model)

    logger.info(model)
    vit_num_params = count_parameters(model.vision_backbone.image_vit)
    connector_num_params = count_parameters(model.vision_backbone.image_projector) + count_parameters(model.vision_backbone.image_pooling_2d)
    llm_num_params = count_parameters(model.transformer)

    logger.info(f"vit num of params: {format_param_count(vit_num_params, 'M')}")
    logger.info(f"connector num of params: {format_param_count(connector_num_params, 'M')}")
    logger.info(f"llm num of params: {format_param_count(llm_num_params, 'B')}")
    


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, creates pytorch_model.bin, and save the toeknizer,"
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Location of SPRVLA checkpoint.",
    )
    args = parser.parse_args()
    prepare_cli_environment()
    load_model_and_calculate(args.checkpoint_dir)


if __name__ == "__main__":
    main()