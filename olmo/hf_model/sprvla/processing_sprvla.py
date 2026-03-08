"""
Processor class for SPRVLA.
"""
from typing import List, Optional, Union, Dict, Tuple

import PIL
from PIL import ImageFile, ImageOps

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import torch

from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

from transformers import AutoTokenizer
from image_processing_sprvla import SPRVLAImagesKwargs, SPRVLAImageProcessor


logger = logging.get_logger(__name__)


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = f"<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = f"<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = f"<im_start>"
IM_END_TOKEN = f"<im_end>"
IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"

EXTRA_TOKENS = (IM_START_TOKEN, IM_END_TOKEN, IMAGE_PATCH_TOKEN,
                IM_COL_TOKEN, IMAGE_PROMPT, IMAGE_LOW_RES_TOKEN)


DEMO_STYLES = [
    "point_count",
    "pointing",
    "cosyn_point",
    "user_qa",
    "long_caption",
    "short_caption",
    "correction_qa",
    "demo",
    "android_control",
]


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_special_token_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


def load_image(image: Union[PIL.Image.Image, np.ndarray]) -> np.ndarray:
    """Load image"""
    setup_pil()
    if isinstance(image, PIL.Image.Image):
        image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)
        return np.array(image)
    elif isinstance(image, np.ndarray):
        assert len(image.shape) == 3, "Image should have 3 dimensions"
        assert image.shape[2] == 3, "Image should have 3 channels"
        assert image.dtype == np.uint8, "Image should have uint8 type"
        return image
    else:
        raise ValueError("Image should be PIL.Image or np.ndarray")


class SPRVLAProcessorKwargs(ProcessingKwargs, total=False):
    """SPRVLA processor kwargs"""
    images_kwargs: SPRVLAImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class SPRVLAProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    optional_attributes = [
        "chat_template",
        "prompt_templates",
        "message_format",
        "system_prompt",
        "style",
        "always_start_with_space",
        "default_inference_len",
        "use_col_tokens",
        "image_padding_mask",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: SPRVLAImageProcessor = None,
        tokenizer: AutoTokenizer = None,
        chat_template: Optional[str] = None,
        prompt_templates: Optional[str] = "uber_model",
        message_format: Optional[str] = "role",
        system_prompt: Optional[str] = "demo_or_style",
        style: Optional[str] = "demo",
        always_start_with_space: Optional[bool] = False,
        default_inference_len: Optional[int] = 65,
        use_col_tokens: Optional[bool] = True,
        image_padding_mask: bool = False,
        **kwargs
    ) -> None:
        if tokenizer.padding_side != "left":
            logger.warning(f"Tokenizer {tokenizer.name_or_path} is not left-padded, padding side will be set to left")
            tokenizer.padding_side = "left"  # type: ignore
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            prompt_templates=prompt_templates,
            message_format=message_format,
            system_prompt=system_prompt,
            style=style,
            always_start_with_space=always_start_with_space,
            default_inference_len=default_inference_len,
            use_col_tokens=use_col_tokens,
            image_padding_mask=image_padding_mask,
        )
        self._special_tokens = None

    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens
    
    def get_user_prompt(self, text: TextInput) -> str:
        """Get user prompt"""
        if self.prompt_templates == "none":
            return ""
        elif self.prompt_templates == "uber_model":
            return text
        else:
            raise NotImplementedError(self.prompt_templates)
    
    def get_prefix(self) -> str:
        """Get prefix"""
        if self.system_prompt == "style_and_length":  # captioner
            assert self.style in ["long_caption"]
            style = self.style
            n = None if self.default_inference_len is None else str(self.default_inference_len)
            if n is not None and len(n) > 0:  # allow empty string to signal unconditioned
                prefix = style + " " + n + ":"
            else:
                prefix = style + " :"
        elif self.system_prompt == "demo_or_style":  # demo model
            if self.style in DEMO_STYLES:
                prefix = ""
            else:
                prefix = self.style + ":"
        else:
            raise NotImplementedError(self.system_prompt)
        return prefix
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt"""
        if self.message_format == "none":
            pass
        elif self.message_format == "role":
            prompt = "User: " + prompt + " Assistant:"
        else:
            raise NotImplementedError(self.message_format)
        
        if self.always_start_with_space:
            prompt = " " + prompt
        
        return prompt
    
    def get_prompt(self, text: TextInput) -> str:
        prompt = self.get_user_prompt(text)
        if self.system_prompt and self.system_prompt != "none":
            prefix = self.get_prefix()
            if len(prefix) > 0 and len(prompt) > 0:
                prompt = prefix + " " + prompt
            elif len(prefix) > 0:
                prompt = prefix
        prompt = self.format_prompt(prompt)
        return prompt

    def get_image_tokens(self, image_grid: np.ndarray):
        joint = []
        for h, w in image_grid:
            per_row = np.full(w, IMAGE_PATCH_TOKEN)
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint += [
                [IM_START_TOKEN],
                extra_tokens,
                [IM_END_TOKEN],
            ]
        return np.concatenate(joint)

    def insert_bos_numpy(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ):
        """
        Args:
            input_ids: [B, S] array with left padding
            attention_mask: [B, S] array (0 for pad, 1 for valid)
            bos_token_id: int
            pad_token_id: int
        Returns:
            input_ids_out: [B, S] or [B, S+1] array with bos inserted if needed
            attention_mask_out: same shape as input_ids_out
        """

        need_to_expand = len(input_ids.shape) == 1
        if need_to_expand:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]

        B, S = input_ids.shape

        # Handle zero-length sequence
        if S == 0:
            new_input_ids = np.full((B, 1), bos_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.ones((B, 1), dtype=attention_mask.dtype)
            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]
            return new_input_ids, new_attention_mask

        first_valid_index = (attention_mask == 1).argmax(axis=-1)  # [B]
        bos_already_present = np.all(input_ids[np.arange(B), first_valid_index] == bos_token_id)

        if bos_already_present:
            if need_to_expand:
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
            return input_ids, attention_mask
        else:
            new_input_ids = np.full((B, S+1), pad_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.zeros((B, S+1), dtype=attention_mask.dtype)

            src_idx = np.tile(np.arange(S), (B, 1))  # [B, S]
            valid_mask = src_idx >= first_valid_index[:, None]  # [B, S]
            tgt_idx = src_idx + 1  # shit right
            batch_idx = np.tile(np.arange(B)[:, None], (1, S))  # [B, S]

            # flatten valid_positions
            flat_vals = input_ids[valid_mask]
            flat_batch = batch_idx[valid_mask]
            flat_tgt = tgt_idx[valid_mask]

            new_input_ids[flat_batch, flat_tgt] = flat_vals
            new_attention_mask[flat_batch, flat_tgt] = 1
            
            insert_pos = first_valid_index
            new_input_ids[np.arange(B), insert_pos] = bos_token_id
            new_attention_mask[np.arange(B), insert_pos] = 1

            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]

            return new_input_ids, new_attention_mask

    def insert_bos_torch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_token_id: int,
        pad_token_id: int,
    ):
        """
        Args:
            input_ids: [B, S] tensor with left padding
            attention_mask: [B, S] tensor (0 for pad, 1 for valid)
            bos_token_id: int
            pad_token_id: int
        Returns:
            input_ids_out: [B, S] or [B, S+1] tensor with bos inserted if needed
            attention_mask_out: same shape as input_ids_out
        """
        
        B, S = input_ids.shape
        device = input_ids.device

        # Handle zero-length sequence
        if S == 0:
            new_input_ids = torch.full((B, 1), bos_token_id, dtype=input_ids.dtype, device=device)
            new_attention_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
            return new_input_ids, new_attention_mask

        first_valid_index = (attention_mask == 1).long().argmax(dim=-1)  # [B]
        bos_already_present = (input_ids[torch.arange(B), first_valid_index] == bos_token_id).all()

        if bos_already_present:
            return input_ids, attention_mask
        else:
            new_input_ids = torch.full((B, S+1), pad_token_id, dtype=input_ids.dtype, device=device)
            new_attention_mask = torch.zeros((B, S+1), dtype=attention_mask.dtype, device=device)

            src_idx = torch.arange(S, device=device).expand(B, S)  # [B, S]
            valid_mask = src_idx >= first_valid_index.unsqueeze(1)  # [B, S]
            tgt_idx = src_idx + 1  # shift right
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(src_idx)

            flat_vals = input_ids[valid_mask]
            flat_batch = batch_idx[valid_mask]
            flat_tgt = tgt_idx[valid_mask]

            new_input_ids[flat_batch, flat_tgt] = flat_vals
            new_attention_mask[flat_batch, flat_tgt] = 1

            insert_pos = first_valid_index
            batch_indices = torch.arange(B, device=device)
            new_input_ids[batch_indices, insert_pos] = bos_token_id
            new_attention_mask[batch_indices, insert_pos] = 1

            return new_input_ids, new_attention_mask

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: Union[ImageInput, List[ImageInput]] = None,
        apply_chat_template: bool = False,
        **kwargs: Unpack[SPRVLAProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            SPRVLAProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, (list, tuple)) and isinstance(images, (list, tuple)):
            if len(text) != len(images):
                raise ValueError("You have to provide the same number of text and images")
            if len(text) > 1 and not output_kwargs["text_kwargs"].get("padding", False):
                raise ValueError("You have to specify padding when you have multiple text inputs")

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}
        
        if apply_chat_template:
            text = [self.get_prompt(t) for t in text]
        
        prompt_strings = text
        if image_inputs.get("images", None) is not None:

            prompt_strings = []
            for idx, image_grids in enumerate(image_inputs.pop("image_grids")):
                if isinstance(image_grids, torch.Tensor):
                    image_grids = image_grids.cpu().numpy()
                if isinstance(images, (list, tuple)) and isinstance(images[idx], (list, tuple)):
                    image_grids = image_grids[~np.all(image_grids == -1, axis=-1)]
                    offset = 2 if len(images[idx]) < len(image_grids) else 1 # whether to use both low and high res images
                    all_image_strings = []
                    for i in range(0, len(image_grids), offset):
                        image_grids_i = image_grids[i:i+offset]
                        image_tokens = self.get_image_tokens(image_grids_i)
                        img_ix = i // offset
                        all_image_strings.append(f"Image {img_ix + 1}" + "".join(image_tokens))
                    image_string = "".join(all_image_strings)
                    prompt_strings.append(image_string + text[idx])
                else:
                    image_grids = image_grids[~np.all(image_grids == -1, axis=-1)]
                    assert len(image_grids) in [1, 2], "Only one or two crops are supported for single image inputs"
                    image_tokens = self.get_image_tokens(image_grids)
                    image_string = "".join(image_tokens)
                    prompt_strings.append(image_string + text[idx])
        
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        is_list = isinstance(input_ids, (list, tuple))
        if is_list:
            input_ids = np.array(input_ids)
            attention_mask = np.array(attention_mask)
        
        use_numpy = isinstance(attention_mask, np.ndarray)

        if use_numpy and np.issubdtype(input_ids.dtype, np.floating):
            input_ids = input_ids.astype(np.int64)
            attention_mask = attention_mask.astype(np.int64)
        elif not use_numpy and torch.is_floating_point(input_ids):
            input_ids = input_ids.to(torch.int64)
            attention_mask = attention_mask.to(torch.int64)
        
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if use_numpy:
            input_ids, attention_mask = self.insert_bos_numpy(
                input_ids, attention_mask, bos, self.tokenizer.pad_token_id
            )
        else:
            input_ids, attention_mask = self.insert_bos_torch(
                input_ids, attention_mask, bos, self.tokenizer.pad_token_id
            )
        if is_list:
            input_ids = input_ids.tolist()  # type: ignore
            attention_mask = attention_mask.tolist()  # type: ignore
        text_inputs["input_ids"] = input_ids
        text_inputs["attention_mask"] = attention_mask

        if kwargs.get("device", None) is not None:
            text_inputs = text_inputs.to(device=kwargs.get("device"), non_blocking=True)
        # there is no bos token in Qwen tokenizer
        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=output_kwargs["common_kwargs"]["return_tensors"]
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


SPRVLAProcessor.register_for_auto_class()