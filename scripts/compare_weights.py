import argparse
import logging
import sys
from os.path import join

import torch

from olmo.models.sprvla.sprvla import SPRVLA
from olmo.train.checkpointer import load_model_state, load_model_state_hf
from olmo.util import prepare_cli_environment, resource_path
from olmo.train.trainer_config import TrainConfig

logger = logging.getLogger(__name__)

def _collect_named_tensors(model: torch.nn.Module):
    """Return a dict of name->tensor for both parameters and buffers."""
    out = {}
    for n, p in model.named_parameters():
        out[n] = p
    for n, b in model.named_buffers():
        # Avoid clobbering if a buffer and param share a name (rare).
        if n in out:
            out[f"{n}__BUFFER"] = b
        else:
            out[n] = b
    return out

def _compare_tensors(a: torch.Tensor, b: torch.Tensor, exact: bool, rtol: float, atol: float) -> tuple[bool, float]:
    """Return (equal, max_abs_diff). Casts to float32 when needed."""
    with torch.no_grad():
        a = a.detach().cpu()
        b = b.detach().cpu()
        if a.shape != b.shape:
            return False, float("inf")
        if exact and a.dtype == b.dtype:
            equal = torch.equal(a, b)
            mad = (a - b).abs().max().item() if not equal else 0.0
            return equal, mad
        # tolerant compare in common dtype
        a32 = a.float() if a.dtype != torch.float32 else a
        b32 = b.float() if b.dtype != torch.float32 else b
        equal = torch.allclose(a32, b32, rtol=rtol, atol=atol, equal_nan=True)
        mad = (a32 - b32).abs().max().item()
        return equal, mad

def convert_checkpoint(checkpoint_dir: str, checkpoint_dir_hf: str, exact: bool, rtol: float, atol: float) -> bool:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(checkpoint_dir, "config.yaml")
    config: TrainConfig = TrainConfig.load(config_path)
    model_config = config.model

    logger.info(f"Loading model checkpoint from {checkpoint_dir}")
    with torch.device("meta"):
        model: SPRVLA = model_config.build_model()
    model.to_empty(device=torch.device("cpu"))
    load_model_state(checkpoint_dir, model)

    logger.info(f"Loading model checkpoint from {checkpoint_dir_hf}")
    with torch.device("meta"):
        model_hf: SPRVLA = model_config.build_model()
    model_hf.to_empty(device=torch.device("cpu"))
    load_model_state_hf(checkpoint_dir_hf, model_hf)

    # Gather tensors
    t1 = _collect_named_tensors(model)
    t2 = _collect_named_tensors(model_hf)

    names1 = set(t1.keys())
    names2 = set(t2.keys())

    missing_in_hf = sorted(names1 - names2)
    extra_in_hf   = sorted(names2 - names1)

    if missing_in_hf:
        logger.warning("Missing in HF checkpoint: %s", ", ".join(missing_in_hf[:20]) + (" ..." if len(missing_in_hf) > 20 else ""))
    if extra_in_hf:
        logger.warning("Extra in HF checkpoint: %s", ", ".join(extra_in_hf[:20]) + (" ..." if len(extra_in_hf) > 20 else ""))

    # Compare common tensors
    diffs = []
    max_name = None
    max_mad = 0.0
    common = sorted(names1 & names2)
    for name in common:
        same, mad = _compare_tensors(t1[name], t2[name], exact=exact, rtol=rtol, atol=atol)
        if not same:
            diffs.append((name, t1[name].shape, t1[name].dtype, t2[name].dtype, mad))
        if mad > max_mad:
            max_mad, max_name = mad, name

    # Reporting
    logger.info("Compared %d tensors (params+buffers).", len(common))
    logger.info("Missing in HF: %d | Extra in HF: %d | Different: %d", len(missing_in_hf), len(extra_in_hf), len(diffs))
    if diffs:
        for name, shape, dt1, dt2, mad in diffs[:50]:
            logger.info(f"DIFF: {name} shape={tuple(shape)} dtype1={dt1} dtype2={dt2} max_abs_diff={mad:.6g}")
        if len(diffs) > 50:
            logger.info("... and %d more differing tensors", len(diffs) - 50)
    if max_name is not None:
        logger.info(f"Worst max_abs_diff: {max_mad:.6g} on {max_name}")

    success = not missing_in_hf and not extra_in_hf and not diffs
    if success:
        logger.info("All tensors match within the chosen criteria.")
    else:
        logger.warning("Models do NOT match.")
    return success

def main():
    parser = argparse.ArgumentParser(description="Compare weights between a native SPRVLA checkpoint and an HF checkpoint.")
    parser.add_argument("checkpoint_dir", help="Path to native SPRVLA checkpoint dir.")
    parser.add_argument("checkpoint_dir_hf", help="HF repo ID or local dir for HF weights.")
    parser.add_argument("--exact", action="store_true",
                        help="Require exact bitwise equality (only works if dtypes are identical).")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for approximate comparison.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for approximate comparison.")
    args = parser.parse_args()

    prepare_cli_environment()
    ok = convert_checkpoint(args.checkpoint_dir, args.checkpoint_dir_hf, args.exact, args.rtol, args.atol)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()