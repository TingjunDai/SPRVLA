"""
Convert 7B-D HF model to training checkpoint format (fixed version)
"""
import sys
from pathlib import Path

# Critical fix: Add HF model directory to Python path
# so that transformers can find configuration_sprvlaact.py and other files
def add_hf_model_to_path(hf_model_path: str):
    """Add the HF model directory to sys.path"""
    hf_dir = Path(hf_model_path).resolve()
    if str(hf_dir) not in sys.path:
        sys.path.insert(0, str(hf_dir))
        print(f"Added {hf_dir} to Python path")

import torch
from olmo.train.trainer_config import TrainConfig
from olmo.train.checkpointer import load_model_state_hf, save_unsharded
from olmo.train.distributed_checkpointing import save_model_and_optim_state
from omegaconf import OmegaConf


def convert_hf_to_checkpoint(
    hf_model_path: str,
    output_dir: str,
    config_path: str = None,
    format: str = "sharded",
):
    """
    Convert an HF-format 7B-D model to training checkpoint format.

    Args:
        hf_model_path: Path to the HF model
        output_dir: Output directory
        config_path: Path to TrainConfig
        format: "sharded" or "unsharded"
    """
    print("="*80)
    print("7B-D HF Model -> Training Checkpoint Format Conversion")
    print("="*80)
    print(f"Input: {hf_model_path}")
    print(f"Output: {output_dir}")
    print(f"Format: {format}")
    print("="*80)

    # Critical step: Add HF model directory to Python path
    print("\n[0/4] Preparing environment...")
    add_hf_model_to_path(hf_model_path)

    # Verify import works
    try:
        import configuration_sprvlaact
        print("  Successfully imported configuration_sprvlaact")
    except ImportError as e:
        print(f"  Warning: Could not import configuration_sprvlaact: {e}")
        print(f"  Will attempt to continue...")

    # 1. Load config
    if config_path:
        print(f"\n[1/4] Loading config: {config_path}")
        cfg = TrainConfig.load(config_path)
        model_cfg = cfg.model
    else:
        print(f"\n[1/4] Attempting to find config in HF directory")
        config_yaml = Path(hf_model_path) / "config.yaml"
        if config_yaml.exists():
            print(f"  Found config: {config_yaml}")
            cfg = TrainConfig.load(str(config_yaml))
            model_cfg = cfg.model
        else:
            raise ValueError(
                f"Must provide config_path or have config.yaml in {hf_model_path}\n"
                f"Hint: Use the config.yaml from your training checkpoint"
            )

    # 2. Create model
    print(f"\n[2/4] Creating model structure...")
    with torch.device("meta"):
        model = model_cfg.build_model()
    model.to_empty(device=torch.device("cpu"))
    print(f"  Model created: {model_cfg.model_name}")

    # 3. Load HF weights
    print(f"\n[3/4] Loading weights from HF format...")
    print(f"  Loading... (this may take a few minutes)")

    try:
        load_model_state_hf(hf_model_path, model)
        print(f"  HF weights loaded successfully")
    except ImportError as e:
        print(f"  Loading failed: {e}")
        print(f"\n  Trying fallback method (loading directly from safetensors)...")
        load_hf_from_safetensors(hf_model_path, model)
        print(f"  Loaded successfully using fallback method")

    # 4. Save as checkpoint format
    print(f"\n[4/4] Saving as {format} format...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if format == "sharded":
        model_and_optim_dir = output_path / "model_and_optim"
        print(f"  Saving to: {model_and_optim_dir}")
        print(f"  Saving... (this may take a few minutes)")

        save_model_and_optim_state(
            str(model_and_optim_dir),
            model,
            optim=None,
            save_overwrite=True,
        )

        if cfg:
            config_file = output_path / "config.yaml"
            with open(config_file, 'w') as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
            print(f"  Config saved")

        print(f"  Sharded checkpoint saved")

    elif format == "unsharded":
        print(f"  Saving... (this may take a few minutes)")
        save_unsharded(
            str(output_path),
            model,
            optim=None,
            config=cfg,
            overwrite=True
        )
        print(f"  Unsharded checkpoint saved")

    else:
        raise ValueError(f"Unsupported format: {format}")

    print("\n" + "="*80)
    print("Conversion complete!")
    print("="*80)

    if format == "sharded":
        print(f"\nOutput directory: {output_dir}/")
        print(f"  ├── model_and_optim/")
        print(f"  └── config.yaml")
    else:
        print(f"\nOutput directory: {output_dir}/")
        print(f"  ├── model.pt")
        print(f"  └── config.yaml")

    return output_path


def load_hf_from_safetensors(hf_model_path: str, model):
    """Fallback method: load directly from safetensors without using transformers"""
    from safetensors import safe_open
    import json

    hf_dir = Path(hf_model_path)
    index_file = hf_dir / "model.safetensors.index.json"

    print(f"  Loading from safetensors: {index_file}")

    with open(index_file) as f:
        index = json.load(f)

    state_dict = {}
    shard_files = set(index['weight_map'].values())

    print(f"  Found {len(shard_files)} shard files")
    for i, shard_file in enumerate(sorted(shard_files), 1):
        print(f"  [{i}/{len(shard_files)}] Loading {shard_file}...")
        with safe_open(hf_dir / shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # Convert HF format to olmo format
    print(f"  Converting format...")
    converted = {}
    for k, v in state_dict.items():
        if k == "lm_head.weight":
            converted["transformer.ff_out.weight"] = v
            continue

        # Remove model. prefix
        k = k[6:] if k.startswith("model.") else k

        # Convert structure
        k = k.replace(".self_attn.", ".").replace(".mlp.", ".")

        converted[k] = v

    print(f"  Loading into model... ({len(converted)} parameters)")
    model.load_state_dict(converted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HF model to training checkpoint format")
    parser.add_argument("--hf-model", default=None, required=True,
                       help="Path to the HF model")
    parser.add_argument("--output", default="./checkpoints/converted-sharded",
                       help="Output directory")
    parser.add_argument("--config", default=None,
                       help="Path to TrainConfig yaml")
    parser.add_argument("--format", default="sharded", choices=["sharded", "unsharded"],
                       help="Output format")

    args = parser.parse_args()

    # Convert
    output = convert_hf_to_checkpoint(
        hf_model_path=args.hf_model,
        output_dir=args.output,
        config_path=args.config,
        format=args.format
    )

    print("\n" + "="*80)
    print("Next step: Merge weights")
    print("="*80)
    print(f"\nConverted 7B-D checkpoint: {output}")
