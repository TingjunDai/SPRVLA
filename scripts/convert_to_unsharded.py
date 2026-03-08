import argparse
import os
import logging

import torch

from olmo.models.sprvla.sprvla import SPRVLA, SPRVLAConfig
from olmo.train.checkpointer import load_model_state, save_unsharded
from olmo.util import (
    prepare_cli_environment,
    resource_path
)

from olmo.train.trainer_config import TrainConfig


logger = logging.getLogger(__name__)


def convert_checkpoint(checkpoint_dir: str, output_dir: str) -> None:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(checkpoint_dir, "config.yaml")
    config: TrainConfig = TrainConfig.load(config_path)
    # model_config: SPRVLAConfig = SPRVLAConfig.load(config_path, key="model", validate_paths=False)
    model_config = config.model


    logger.info(f"Loading model checkpoint from {checkpoint_dir}")
    with torch.device("meta"):
        model: SPRVLA = model_config.build_model()
    model.to_empty(device=torch.device("cpu"))
    load_model_state(checkpoint_dir, model)

    logger.info(f"Save model config and checkpoint to {output_dir}")
    save_unsharded(output_dir, model, None, config, True)

    logger.info(f"Completed")


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, creates pytorch_model.bin, and save the toeknizer,"
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Location of SPRVLA checkpoint.",
    )
    parser.add_argument(
        "output_dir",
        help="Location to save the converted checkpoint.",
    )
    args = parser.parse_args()
    prepare_cli_environment()
    convert_checkpoint(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()