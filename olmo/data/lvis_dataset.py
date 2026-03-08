import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import datasets
from datasets import Image as HFImage
from PIL import Image

from olmo.data.dataset import Dataset, HfDataset


# --------------------------
# Helpers
# --------------------------

def _cast_if_present(ds: datasets.Dataset, cols: Tuple[str, ...]) -> datasets.Dataset:
    """Cast existing columns among `cols` to HF Image(decode=True)."""
    for c in cols:
        if c in ds.column_names:
            ds = ds.cast_column(c, HFImage(decode=True))
    return ds

def _to_pil(x: Any) -> Image.Image:
    """Convert common containers to PIL.Image (RGB)."""
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str) and os.path.exists(x):
        return Image.open(x).convert("RGB")
    if isinstance(x, dict):
        # HF Image feature may be dict-like {path:..., bytes:...}
        if x.get("path"):
            return Image.open(x["path"]).convert("RGB")
        if x.get("bytes") is not None:
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
    raise ValueError(f"Unsupported image container for PIL conversion: {type(x)}")

_IMG_TOKEN_PATTERNS = (
    "<image>/n",
    "/n<image>",
    "<image>",
)

def _clean_image_tokens(text: Optional[str]) -> Optional[str]:
    """Remove <image> token and common variants like '<image>/n', '/n<image>', etc."""
    if text is None:
        return None
    out = text
    for pat in _IMG_TOKEN_PATTERNS:
        out = out.replace(pat, "")
    # If any literal '/n' remain (unlikely), drop them too.
    # out = out.replace("/n", "")
    return out


# =============================================================================
# LVIS dataset (subset of the Pretraining Mixture)
# =============================================================================

class LVIS(HfDataset):
    """
    LVIS subset loader from `allenai/SPRVLA-Pretraining-Mixture`.

    - Columns: 'image', 'conversation' (singular)
      * 'image' is an image (HF Image feature or decodable to PIL)
      * 'conversation' is a dict with
            { "from": [...], "value": [...] }
        where 'from' is a sequence like ["human","gpt","human","gpt",...]
    - download(): prefetches via HuggingFace builder (into HF cache)
    - get(): returns:
        {
          "image": PIL.Image (optional if missing),
          "message_list": [
              {"question": <str>, "answer": <str or None>, "style": <style>},
              ...
          ],
          "metadata": {"id": <example id or None>}
        }
      * Ensures human/GPT pairing in sequence
      * Strips all '<image>' token variants from both questions and answers
    """

    PATH: str = "allenai/SPRVLA-Pretraining-Mixture"
    DEFAULT_NAME: str = "lvis"  # HF config name inside the mixture

    @classmethod
    def download(cls, n_procs: int = 1, dataset_name: Optional[str] = None):
        """
        Prefetch the HF dataset builder for this LVIS config into the local HF cache.
        Mirrors the pretraining-style download; `n_procs` is kept for API parity.
        """
        name = dataset_name or cls.DEFAULT_NAME
        builder = datasets.load_dataset_builder(cls.PATH, name=name)
        builder.download_and_prepare()

    def __init__(
        self,
        split: str = "train",
        style: str = "demo",
        keep_in_memory: bool = False,
        num_proc: int = 16,
        dataset_name: Optional[str] = None,
        **hf_kwargs,
    ):
        """
        Args:
            split: HF split (e.g., 'train', 'validation', 'test').
            style: Arbitrary string passed through in the get() output.
            keep_in_memory: Forwarded to base HfDataset.
            num_proc: Forwarded to base HfDataset (e.g., map parallelism in non-streaming mode).
            dataset_name: HF config name; defaults to 'lvis'.
            **hf_kwargs: Forwarded to datasets.load_dataset (e.g., streaming=True).
        """
        self.style = style
        self.dataset_name = dataset_name or self.DEFAULT_NAME

        # HfDataset base is expected to use self.PATH and the provided name
        super().__init__(
            split=split,
            keep_in_memory=keep_in_memory,
            name=self.dataset_name,
            num_proc=num_proc,
            **hf_kwargs,
        )

        # Ensure 'image' is decodable to a PIL.Image on access
        if isinstance(self.dataset, datasets.Dataset):
            self.dataset = _cast_if_present(self.dataset, ("image",))

    def __len__(self) -> int:
        return len(self.dataset)

    # ---- runtime helpers ----
    def _return_pil(self, img: Any) -> Image.Image:
        pil = img if isinstance(img, Image.Image) else _to_pil(img)
        return pil

    # ---- API ----
    def get(self, item, rng):
        ex: Dict[str, Any] = self.dataset[item]

        conv = ex.get("conversations", {})
        frm = conv.get("from", []) or []
        val = conv.get("value", []) or []
        n = min(len(frm), len(val))

        message_list: List[Dict[str, Optional[str]]] = []
        i = 0
        while i < n:
            if frm[i] == "human":
                question = _clean_image_tokens(val[i])
                answer = None
                if i + 1 < n and frm[i + 1] == "gpt":
                    answer = val[i + 1]
                    i += 1  # consume the paired GPT turn
                message_list.append({
                    "question": question,
                    "answer": answer,
                    "style": self.style,
                })
            i += 1

        img = ex.get("image", None)
        if img is not None:
            try:
                img = self._return_pil(img)
            except Exception:
                # If anything odd happens, leave the raw object; caller can decide.
                pass

        out = {
            "message_list": message_list,
        }
        if img is not None:
            out["image"] = img
        
        return out