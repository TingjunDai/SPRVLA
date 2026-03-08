import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property, partial
from typing import List, Optional, Set, Tuple, TypedDict, Union, Dict, Any
from PIL.Image import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (BatchFeature, PretrainedConfig, ProcessorMixin,
                          TensorType, AutoTokenizer, AutoConfig)
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention import Attention
from vllm.attention.layer import MultiHeadAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import (MulAndSilu, QuickGELU,
                                                   SiluAndMul, get_act_fn)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptIndexTargets,
                                        PromptInsertion, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings, SupportsLoRA,
    SupportsMultiModal, SupportsPP, SupportsQuant
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, WeightsMapper, flatten_bn,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix, merge_multimodal_embeddings
)


import re
from transformers import Qwen2Tokenizer


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = f"<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = f"<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = f"<im_start>"
IM_END_TOKEN = f"<im_end>"
IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


NUM_RE = re.compile(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$')
DEPTH_RE = re.compile(r'<DEPTH_START>(.*?)<DEPTH_END>', re.DOTALL)
# One-level-nested [...] matcher: outer block that may contain inner [ ... ] lists
OUTER_BLOCK_RE = re.compile(r'\[(?:[^\[\]]|\[[^\[\]]*\])+\]')
# Regex patterns for extracting subtask-related information
SUBTASK_COUNT_RE = re.compile(r'The number of subtasks is (\d+)', re.IGNORECASE)
SUBTASK_GOALS_RE = re.compile(
    r'The positions of subtask goals are\s*(\[(?:[^\[\]]|\[[^\[\]]*\])+\])',
    re.IGNORECASE
)
TRAJECTORY_RE = re.compile(
    r'The trajectory of the end effector.*?is\s*(\[(?:[^\[\]]|\[[^\[\]]*\])+\])',
    re.IGNORECASE | re.DOTALL
)

def _is_number(s: str) -> bool:
    return bool(NUM_RE.match(s))

def _has_non_ascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s)

def _to_number(s: str):
    """Parse string number to int when possible, else float."""
    v = float(s)
    return int(v) if v.is_integer() else v

def extract_depth_string(text: str, include_tags: bool = False) -> list[str]:
    """
    Return all occurrences of depth strings.
    If include_tags=True, each item is '<DEPTH_START>...<DEPTH_END>';
    otherwise each item is just the inner '...'.
    """
    matches = list(DEPTH_RE.finditer(text))
    if include_tags:
        return [m.group(0) for m in matches]
    return [m.group(1) for m in matches]

def _parse_point_list(block: str, point_len: int | None = 2) -> list[list[float]] | None:
    """
    Helper function to parse a block like '[[x1,y1],[x2,y2],...]' into list of points.
    Returns None if parsing fails.
    """
    inner_strs = re.findall(r'\[([^\[\]]+)\]', block)
    if not inner_strs:
        return None
    
    rows: list[list[float]] = []
    for row in inner_strs:
        parts = [p.strip().strip('"').strip("'") for p in row.split(',')]
        if point_len is not None and len(parts) != point_len:
            return None
        if not all(_is_number(p) for p in parts):
            return None
        rows.append([_to_number(p) for p in parts])
    
    return rows if rows else None

def extract_trajectory(text: str, point_len: int = 2) -> list[list[float]] | None:
    """
    Extract end effector trajectory from text.
    
    Example:
        "The trajectory of the end effector in the first image is [[128, 68], [92, 109]]"
        -> [[128.0, 68.0], [92.0, 109.0]]
    
    Args:
        text: Input text
        point_len: Expected length of each point (default: 2 for [x, y])
    
    Returns:
        list[list[float]]: List of trajectory points, or None if not found
    """
    match = TRAJECTORY_RE.search(text)
    if not match:
        return None
    
    block = match.group(1)
    return _parse_point_list(block, point_len)

def extract_subtask_count(text: str) -> int | None:
    """
    Extract the number of subtasks from text.
    
    Example:
        "The number of subtasks is 2" -> 2
    
    Returns:
        int: Number of subtasks, or None if not found
    """
    match = SUBTASK_COUNT_RE.search(text)
    if match:
        return int(match.group(1))
    return None

def extract_subtask_goals(text: str, point_len: int = 2) -> list[list[float]] | None:
    """
    Extract subtask goal positions from text.
    
    Example:
        "The positions of subtask goals are [[82, 146], [178, 89]]"
        -> [[82.0, 146.0], [178.0, 89.0]]
    
    Args:
        text: Input text
        point_len: Expected length of each point (default: 2 for [x, y])
    
    Returns:
        list[list[float]]: List of goal positions, or None if not found
    """
    match = SUBTASK_GOALS_RE.search(text)
    if not match:
        return None
    
    block = match.group(1)
    return _parse_point_list(block, point_len)

def extract_trace_lists(
    text: str,
    point_len: int | None = 2,     # e.g., 2 for [x,y], 3 for [x,y,z]; None = any length ≥1
    min_points: int = 1
) -> list[list[list[float]]]:
    """
    Extract *numeric* lists-of-lists like [[140,225],[130,212],...].
    Returns a list of traces; each trace is a list of points (lists of numbers).

    Heuristic:
      - Find outer [ ... ] blocks that may contain inner lists
      - Keep blocks where every inner list is fully numeric
      - Enforce per-point length (point_len) and a minimum number of points (min_points)
    """
    traces: list[list[list[float]]] = []

    # Find outer blocks that can contain nested lists
    for block in OUTER_BLOCK_RE.findall(text):
        inner_strs = re.findall(r'\[([^\[\]]+)\]', block)  # contents of each inner [...]
        if len(inner_strs) < min_points:
            continue

        rows: list[list[float]] = []
        ok = True
        for row in inner_strs:
            parts = [p.strip().strip('"').strip("'") for p in row.split(',')]
            if point_len is not None and len(parts) != point_len:
                ok = False
                break
            if not all(_is_number(p) for p in parts):
                ok = False
                break
            rows.append([_to_number(p) for p in parts])

        if ok:
            traces.append(rows)

    return traces

def extract_action_token_lists(
    text: str,
    only_len: int | None = None,         # e.g., 7 if you expect 7-D actions
    require_non_ascii: bool = True       # set False if your tokens can be pure ASCII
) -> list[list[str]]:
    """
    Extract all [ ... ] groups split by commas, discard numeric lists,
    and return token lists (quotes stripped, whitespace trimmed).
    """
    lists = []
    # Match NON-nested bracketed groups: [ ... ] without inner [ or ]
    for inner in re.findall(r'\[([^\[\]]+)\]', text):
        parts = [p.strip().strip('"').strip("'") for p in inner.split(',')]

        if only_len is not None and len(parts) != only_len:
            continue

        # If *all* items are numeric -> not action tokens (like coordinates)
        if all(_is_number(p) for p in parts):
            continue

        # Optionally require at least one non-ASCII char across tokens (helps exclude plain words/numbers)
        if require_non_ascii and not any(_has_non_ascii(p) for p in parts):
            continue

        lists.append(parts)

    return lists


class SPRVLAImageInputs(TypedDict):
    images: Union[torch.Tensor, list[torch.Tensor]]
    """Shape: `(batch_size * num_images, num_crops, num_patch, patch_dim)`"""

    pooled_patches_idx: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_patch_tokens, num_pooled_patches)`
    """

    num_crops: torch.Tensor
    """Shape: `(batch_size * num_images)`"""

    num_pooled_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


@dataclass
class VitConfig:
    """Config for a vision transformer"""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    image_default_input_size: Tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 577
    use_cls_token: bool = False   # True for OpenCLIP
    patch_bias: bool = True       # False for OpenCLIP
    pre_layernorm: bool = False   # True for OpenCLIP

    def __post_init__(self):
        self.image_default_input_size = tuple(self.image_default_input_size)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class AdapterConfig:
    """Config for a vit-llm adapter"""

    vit_layers: Tuple[int, int] = (-3, -9)
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_act: str = "silu"
    intermediate_size: int = 18944
    text_hidden_size: int = 3584
    image_feature_dropout: float = 0.0,
    image_padding_embed: Optional[str] = None  # e.g., "pad_and_partial_pad"


@dataclass
class LlmConfig:
    """Configuration for a language model transformer"""

    hidden_size: int = 3584
    """
    The hidden size of the model.
    """

    num_attention_heads: int = 28
    """
    The number of self-attention heads.
    """

    num_key_value_heads: int = 4
    """
    The number of heads to use for keys and values.
    """

    head_dim: int = 128
    """
    The head dimensionality for the attention mechanism.
    """

    vocab_size: int = 152064
    """Vocabulary size of the model."""

    additional_vocab_size: int = 128
    """Number of additional tokens to have the input embeddings for"""

    qkv_bias: bool = True
    """
    Do QKV projection a bias
    """

    num_hidden_layers: int = 48
    """
    The number of layers/blocks.
    """

    intermediate_size: int = 18944
    """
    The hidden size for the MLP.
    """

    hidden_act: str = "silu"
    """
    The activation function to use within the MLP layers.
    """

    max_position_embeddings: int = 4096
    """
    Max positional embeddings to use in RoPE cache
    """

    rope_theta: float = 1000000.0
    """
    RoPE theta parameter.
    """

    use_qk_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    layer_norm_eps: float = 1e-6
    """
    epsilon for layer norms
    """

    norm_after: bool = False
    """Apply layer norm before and after the attention and MLP blocks."""


class ViTMLP(nn.Module):
    """MLP used in Vision Transformer."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=True, quant_config=quant_config)
        # Activation function.
        self.act = get_act_fn(hidden_act)
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=True, quant_config=quant_config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


class ViTMultiHeadDotProductAttention(nn.Module):
    """Multi-head attention used in Vision Transformer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        input_dim: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = head_dim

        assert self.head_dim == self.hidden_size // self.total_num_heads
        
        self.total_num_kv_heads = num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if input_dim is None:
            input_dim = self.hidden_size

        self.wq = ColumnParallelLinear(
            input_dim,
            self.total_num_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wk = ColumnParallelLinear(
            input_dim,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wv = ColumnParallelLinear(
            input_dim,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.scale = self.head_dim**-0.5
        self.attn = MultiHeadAttention(self.num_heads,
                                       self.head_dim,
                                       self.scale,
                                       num_kv_heads=self.num_kv_heads)

    def forward(self,
                inputs_q: torch.Tensor,
                inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, _ = self.wq(inputs_q)
        xk, _ = self.wk(inputs_k)
        xv, _ = self.wv(inputs_v)

        output = self.attn(xq, xk, xv)
        output, _ = self.wo(output)

        return output


class SPRVLAVisionBlock(nn.Module):
    """Residual attention block used in Vision Transformer."""

    def __init__(
        self,
        config: VitConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            quant_config=quant_config,
        )
        self.feed_forward = ViTMLP(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.attention_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SPRVLAVisionBlockCollection(nn.Module):
    """Collection of residual attention blocks used in Vision Transformer."""

    def __init__(
        self,
        config: VitConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([
            SPRVLAVisionBlock(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class SPRVLAVisionTransformer(nn.Module):
    """Vision Transformer used in Vision Backbone."""

    def __init__(
        self,
        config: VitConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        scale = config.hidden_size ** -0.5
        self.num_prefix_tokens: int = 1 if config.use_cls_token else 0
        if config.use_cls_token:
            self.class_embedding = nn.Parameter(
                torch.zeros(config.hidden_size)
            )
        self.patch_num = config.image_num_patch
        self.positional_embedding = nn.Parameter(
            torch.randn(config.image_num_pos, config.hidden_size) * scale)
        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=config.patch_bias,
        )
        # optional pre-LN
        self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) \
                      if config.pre_layernorm else None
        self.transformer = SPRVLAVisionBlockCollection(config, quant_config)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding
        if self.config.use_cls_token:
            cls_pos, pos_emb = pos_emb[:1], pos_emb[1:]   # split out CLS

        pos_emb = pos_emb.reshape(
            (
                int(math.sqrt(pos_emb.shape[0])),
                int(math.sqrt(pos_emb.shape[0])),
                pos_emb.shape[1]
            )
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)
        
        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        if self.config.use_cls_token:
            x = x + torch.cat([cls_pos[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        else:
            x = x + pos_emb[None, :, :].to(x.dtype)
        return x
    
    def forward(self,
                x: torch.Tensor,
                patch_num: Optional[int] = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.patch_num
        
        x = self.patch_embedding(x)

        if self.config.use_cls_token:
            x = torch.cat([_expand_token(self.class_embedding, x.size(0)).to(x.dtype), x], dim=1)

        x = self.add_pos_emb(x, patch_num)

        if self.pre_ln is not None:
            x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.merged_linear = MergedColumnParallelLinear(
            input_dim,
            [hidden_dim] * 2,
            bias=False,
            quant_config=quant_config,
        )
        # Activation function.
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            hidden_dim, output_dim, bias=False, quant_config=quant_config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.merged_linear(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class SPRVLAVisionBackbone(nn.Module, SupportsQuant):
    packed_modules_mapping = {"merged_linear": ["gate_proj", "up_proj"]}

    def __init__(
        self,
        vit_config: VitConfig,
        adapter_config: AdapterConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.vit_config = vit_config
        self.adapter_config = adapter_config

        self.vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(layer + vit_config.num_hidden_layers)
        
        last_layer_needed = max(self.vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            vit_config.num_hidden_layers = last_layer_needed
        self.image_vit = SPRVLAVisionTransformer(vit_config, quant_config)

        self.num_prefix_tokens: int = self.image_vit.num_prefix_tokens

        # optional pad_embed
        self.pad_embed = None
        if adapter_config.image_padding_embed == "pad_and_partial_pad":
            pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
            self.pad_embed = nn.Parameter(torch.zeros((2, pool_dim)))

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            quant_config=quant_config,
        )
        self.image_projector = ImageProjectorMLP(
            input_dim=adapter_config.hidden_size,
            hidden_dim=adapter_config.intermediate_size,
            output_dim=adapter_config.text_hidden_size,
            hidden_act=adapter_config.hidden_act,
            quant_config=quant_config,
        )
        self.image_feature_dropout = nn.Dropout(adapter_config.image_feature_dropout)

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
        image_features = image_features.view(B, T, N, -1)
        return image_features

    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        image_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.encode_image(images)

        # optional padding embeddings
        if self.pad_embed is not None and image_masks is not None:
            image_masks = image_masks.to(device=self.device)
            all_pad = (image_masks == 0).to(image_features.dtype)
            partial = torch.logical_and(image_masks < 1, ~ (image_masks == 0)).to(image_features.dtype)
            image_features = image_features + self.pad_embed[0][None,None,None,:] * all_pad[...,None] \
                            + self.pad_embed[1][None,None,None,:] * partial[...,None]
            
        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]
        
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
        batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]])

        # Now [batch, num_features, num_pooled_patches, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.to(self.dtype)[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])

        query = to_pool.mean(-2, keepdim=True)
        pooled_features = self.image_pooling_2d(query, to_pool)
        pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

        # MLP layer to map the feature.
        pooled_features = self.image_projector(pooled_features)
        return pooled_features.view(-1, pooled_features.shape[-1])[valid_token.flatten()]

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_linear", "gate_proj", 0),
            ("merged_linear", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class SPRVLAAttention(nn.Module):
    """SPRVLA's LLM Attention."""

    def __init__(
        self,
        config: LlmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        assert self.head_dim == self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
        )

        self.tp_rank: Optional[int] = None
        self.k_norm: Optional[nn.Module] = None
        self.q_norm: Optional[nn.Module] = None
        if config.use_qk_norm:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim,
                                  eps=config.layer_norm_eps)
            self.q_norm = RMSNorm(self.hidden_size,
                                  eps=config.layer_norm_eps)
        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def _apply_qk_norm(self, q: torch.Tensor,
                       k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm.forward_native(q)
        k = self.k_norm.forward_native(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LanguageModelMLP(nn.Module):
    """SPRVLA's LLM mlp."""

    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.up_gate_proj = MergedColumnParallelLinear(
            input_dim,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        # Activation function.
        assert hidden_act == "silu"
        self.act_fn = MulAndSilu()
        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            intermediate_size,
            input_dim,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        up_gate, _ = self.up_gate_proj(x)
        x = self.act_fn(up_gate)
        x, _ = self.down_proj(x)
        return x


class SPRVLADecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Attention block.
        self.self_attn = SPRVLAAttention(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # MLP block.
        self.mlp = LanguageModelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config,
        )

        # LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SPRVLADecoderNormAfterLayer(SPRVLADecoderLayer):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = None
        return hidden_states, residual


@support_torch_compile
class SPRVLALlm(nn.Module, SupportsQuant):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config

        kwargs = {}
        for field in fields(LlmConfig):
            kwargs[field.name] = getattr(config.llm_config, field.name)
        llm_config = LlmConfig(**kwargs)

        self.embedding_size = llm_config.vocab_size
        self.embedding_size += llm_config.additional_vocab_size or 0
        self.embed_tokens = VocabParallelEmbedding(
            self.embedding_size,
            llm_config.hidden_size,
            quant_config=quant_config,
        )

        decoder_layer = SPRVLADecoderNormAfterLayer if llm_config.norm_after \
            else SPRVLADecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            llm_config.num_hidden_layers,
            lambda prefix: decoder_layer(
                llm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(llm_config.hidden_size, eps=llm_config.layer_norm_eps)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], llm_config.hidden_size))

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Apply blocks one-by-one.
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _lowest_multiple(x: int, k: int) -> int:
    return (x // k) * k


def get_patches_grid_size(
    *,
    image_h: int,
    image_w: int,
    patch_size: int,
    pool_h: int,
    pool_w: int,
) -> tuple[int, int]:
    patch_h = image_h // patch_size
    patch_w = image_w // patch_size
    h_pad = _lowest_multiple(patch_h + pool_h - 1, pool_h) - patch_h
    w_pad = _lowest_multiple(patch_w + pool_w - 1, pool_w) - patch_w
    nrows = (patch_h + h_pad) // pool_h
    ncols = (patch_w + w_pad) // pool_w

    return nrows, ncols


def get_candidate_tilings(max_num: int) -> list[tuple[int, int]]:
    tilings = [(i, j) for i in range(1, max_num + 1)
               for j in range(1, max_num + 1) if i * j <= max_num]
    return sorted(tilings, key=lambda x: (x[0] * x[1], x[0]))


def select_tiling(
    *,
    height: int,
    width: int,
    patch_size: int,
    max_num_patches: int,
):
    tilings = get_candidate_tilings(max_num_patches)
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([height, width], dtype=np.float32)
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = required_scale_d.min(axis=-1, keepdims=True)

    if (required_scale < 1).all():
        ix = required_scale.argmax()
    else:
        ix = np.where(required_scale < 1.0, 10e9, required_scale).argmin()

    return candidate_tilings[ix]


def get_image_size(image: ImageInput) -> ImageSize:
    if isinstance(image, Image):
        return ImageSize(*image.size)
    elif isinstance(image, (np.ndarray, torch.Tensor)):
        assert image.ndim == 3
        h, w, c = image.shape
        assert c in [1, 3]
        return ImageSize(w, h)
    else:
        raise ValueError(f"Unknown image type: {type(image)}")


class SPRVLAProcessorWrapper:
    """
    Wraps :class:`SPRVLAProcessor` so that it can be called directly.
    """

    def __init__(self, processor: ProcessorMixin):
        super().__init__()

        self.processor = processor

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.processor.tokenizer.vocab  # type: ignore

    @cached_property
    def max_crops(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        max_crops = image_processor.max_crops
        assert isinstance(max_crops, int)

        return max_crops

    @cached_property
    def max_multi_image_crops(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        max_multi_image_crops = image_processor.max_multi_image_crops
        assert isinstance(max_multi_image_crops, int)
        return max_multi_image_crops

    @cached_property
    def image_pooling_h(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_pooling_h = image_processor.image_pooling_h
        assert isinstance(image_pooling_h, int)

        return image_pooling_h

    @cached_property
    def image_pooling_w(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_pooling_w = image_processor.image_pooling_w
        assert isinstance(image_pooling_w, int)

        return image_pooling_w

    @cached_property
    def base_image_input_size(self) -> tuple[int, int]:
        image_processor = self.processor.image_processor  # type: ignore

        base_image_input_size = image_processor.base_image_input_size
        if isinstance(base_image_input_size, int):
            return base_image_input_size, base_image_input_size

        return tuple(base_image_input_size)

    @cached_property
    def image_patch_size(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_patch_size = image_processor.image_patch_size
        assert isinstance(image_patch_size, int)

        return image_patch_size

    @cached_property
    def overlap_margins(self) -> tuple[int, int]:
        image_processor = self.processor.image_processor  # type: ignore

        left_margin, right_margin = image_processor.overlap_margins
        assert isinstance(left_margin, int)
        assert isinstance(right_margin, int)

        return left_margin, right_margin

    @cached_property
    def bos_token(self) -> str:
        return self.processor.tokenizer.bos_token or self.processor.tokenizer.eos_token

    @cached_property
    def image_patch_id(self) -> int:
        return self.vocab[IMAGE_PATCH_TOKEN]

    @cached_property
    def im_col_id(self) -> int:
        return self.vocab[IM_COL_TOKEN]

    @cached_property
    def im_start_id(self) -> int:
        return self.vocab[IM_START_TOKEN]

    @cached_property
    def im_end_id(self) -> int:
        return self.vocab[IM_END_TOKEN]

    def select_tiling(
        self,
        *,
        image_height: int,
        image_width: int,
        is_multi_image: bool,
    ) -> tuple[int, int]:
        max_crops = self.max_multi_image_crops if is_multi_image else self.max_crops
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d
        tiling_h, tiling_w = select_tiling(
            height=image_height - total_margin_pixels,
            width=image_width - total_margin_pixels,
            patch_size=crop_window_size,
            max_num_patches=max_crops,
        )

        return tiling_h, tiling_w
    
    def get_base_grid_size(self) -> tuple[int, int]:
        base_image_input_size = self.base_image_input_size

        return get_patches_grid_size(
            image_h=base_image_input_size[0],
            image_w=base_image_input_size[1],
            patch_size=self.image_patch_size,
            pool_h=self.image_pooling_h,
            pool_w=self.image_pooling_w,
        )

    def get_patches_grid_size(
        self,
        *,
        image_height: int,
        image_width: int,
        is_multi_image: bool,
    ) -> tuple[int, int]:
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d

        tiling_h, tiling_w = self.select_tiling(
            image_height=image_height,
            image_width=image_width,
            is_multi_image=is_multi_image,
        )

        h, w = [tiling_h * crop_window_size + total_margin_pixels, 
                tiling_w * crop_window_size + total_margin_pixels]
        nrows, ncols = get_patches_grid_size(
            image_h=h, image_w=w, patch_size=base_image_input_d,
            pool_h=self.image_pooling_h,
            pool_w=self.image_pooling_w,
        )

        return nrows, ncols

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        outputs = self.processor(  # type: ignore
            text, images, return_tensors=return_tensors, **kwargs)

        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]
        
        # outputs = {k: v.squeeze(0) if k not in ["input_ids", "attention_mask"] else v for k, v in outputs.items()}
        attention_mask: torch.Tensor = outputs.pop("attention_mask")

        if len(images) > 0:
            assert self.processor.image_processor.crop_mode == "overlap-and-resize-c2"
            num_crops = []
            for image in images:
                if isinstance(image, (list, tuple)):
                    tilings_per_ex = []
                    for img in image:
                        image_size = get_image_size(img)
                        tilings_per_ex.append(
                            self.select_tiling(
                                image_height=image_size.height,
                                image_width=image_size.width,
                                is_multi_image=True,
                            )
                        )
                    num_crops_per_ex = torch.tensor(tilings_per_ex).prod(-1) + 1
                    num_crops.append(num_crops_per_ex.sum())
                else:
                    image_size = get_image_size(image)
                    tiling = self.select_tiling(
                        image_height=image_size.height,
                        image_width=image_size.width,
                        is_multi_image=False,
                    )
                    num_crops.append(torch.tensor(tiling).prod() + 1)
            # For each image: tiling_h * tiling_w + extra
            num_crops = torch.tensor(num_crops)
            assert all([n == len(img) for n, img in zip(num_crops, outputs["images"])])
            outputs["num_crops"] = num_crops
            outputs["num_pooled_patches"] = torch.tensor([len(t) for t in outputs["pooled_patches_idx"]])
            outputs["num_patches"] = torch.tensor([np.prod(t.shape[:2]) for t in outputs["images"]])
            outputs["img_patch_id"] = self.image_patch_id
            outputs = {k: v.squeeze(0) if k in ["images", "pooled_patches_idx", "image_masks"] else v for k, v in outputs.items()}

        return BatchFeature(outputs)


class SPRVLAProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self, **kwargs: object) -> SPRVLAProcessorWrapper:
        processor = self.ctx.get_hf_processor(**kwargs)
        return SPRVLAProcessorWrapper(processor)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_height: int,
        image_width: int,
        is_multi_image: bool,
        processor: Optional[SPRVLAProcessorWrapper] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()
        
        hf_processor = processor.processor  # type: ignore
        image_processor = hf_processor.image_processor  # type: ignore
        assert image_processor.crop_mode == "overlap-and-resize-c2"

        resize_nrows, resize_cols = processor.get_base_grid_size()
        # start/end tokens + image patch token + col tokens
        extra = 2 + resize_nrows * (resize_cols + int(hf_processor.use_col_tokens))
        overlap_nrows, overlap_ncols = processor.get_patches_grid_size(
            image_height=image_height,
            image_width=image_width,
            is_multi_image=is_multi_image,
        )
        joint = 2 + overlap_nrows * (overlap_ncols + int(hf_processor.use_col_tokens))

        return extra + joint

    def get_image_size_with_most_features(self, num_images: int) -> ImageSize:
        processor = self.get_hf_processor()

        left_margin, right_margin = processor.overlap_margins
        base_image_input_size = processor.base_image_input_size
        base_image_input_d = processor.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d

        tilings = get_candidate_tilings(
            processor.max_multi_image_crops if num_images > 1 else processor.max_crops
        )
        largest_feature_size, largest_feature_pinpoint = 0, None

        for hr, wr in tilings:
            height = hr * crop_window_size + total_margin_pixels
            width = wr * crop_window_size + total_margin_pixels

            feat_size = self.get_num_image_tokens(
                image_height=height,
                image_width=width,
                is_multi_image=num_images > 1,
                processor=processor
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


class SPRVLADummyInputsBuilder(BaseDummyInputsBuilder[SPRVLAProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        processor = self.info.get_hf_processor()
        text = processor.processor.apply_chat_template(
            "", tokenize=False, add_generation_prompt=True,
        )

        return text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = \
            self.info.get_image_size_with_most_features(num_images)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class SPRVLAMultiModalProcessor(BaseMultiModalProcessor[SPRVLAProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_crops = hf_inputs.get("num_crops", torch.empty(0))
        num_images = len(num_crops)
        num_pooled_patches = hf_inputs.get("num_pooled_patches", torch.empty(0))

        return dict(
            images=MultiModalFieldConfig.flat_from_sizes("image", num_crops),
            pooled_patches_idx=MultiModalFieldConfig.flat_from_sizes(
                "image", num_pooled_patches),
            num_crops=MultiModalFieldConfig.batched("image"),
            num_pooled_patches=MultiModalFieldConfig.batched("image"),
            num_patches=MultiModalFieldConfig.batched("image"),
            img_patch_id=MultiModalFieldConfig.shared("image", num_images),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        img_patch_id = processor.image_patch_id
        img_col_id = processor.im_col_id
        img_start_id = processor.im_start_id
        img_end_id = processor.im_end_id
        use_col_tokens = processor.processor.use_col_tokens

        resize_nrows, resize_cols = processor.get_base_grid_size()
        extra_row = [img_patch_id] * resize_cols + [img_col_id] * int(use_col_tokens)
        extra_joint = (
            [img_start_id] + extra_row * resize_nrows + [img_end_id]
        )

        def get_insertion_sprvla(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            # TODO: I don't know why, but ImageProcessorItems assumes that the channel appears first for np.ndarray
            image = images.get(item_idx)
            is_multi_image = isinstance(image, (list, tuple))
            if is_multi_image:
                img_token_ids = []
                for i, img in enumerate(image):
                    image_size = get_image_size(img)
                    nrows, ncols = processor.get_patches_grid_size(
                        image_height=image_size.height,
                        image_width=image_size.width,
                        is_multi_image=True,
                    )
                    joint_row = [img_patch_id] * ncols + [img_col_id] * int(use_col_tokens)
                    joint = (
                        [img_start_id] + joint_row * nrows + [img_end_id]
                    )
                    token_ids = processor.processor.tokenizer.encode(
                        f"Image {i + 1}",
                        add_special_tokens=False,
                    )
                    img_token_ids += token_ids + extra_joint + joint
            else:
                image_size = get_image_size(image)

                nrows, ncols = processor.get_patches_grid_size(
                    image_height=image_size.height,
                    image_width=image_size.width,
                    is_multi_image=False,
                )

                joint_row = [img_patch_id] * ncols + [img_col_id] * int(use_col_tokens)
                joint = (
                    [img_start_id] + joint_row * nrows + [img_end_id]
                )
                img_token_ids = extra_joint + joint

            return PromptUpdateDetails.select_token_id(
                img_token_ids,
                embed_token_id=img_patch_id,
            )

        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.prefix(processor.bos_token),
                insertion=get_insertion_sprvla,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(SPRVLAMultiModalProcessor,
                                        info=SPRVLAProcessingInfo,
                                        dummy_inputs=SPRVLADummyInputsBuilder)
class SPRVLAForActionReasoning(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA,
                                     SupportsQuant):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # vision backbone mapping
            "image_projector.w1.": "image_projector.gate_proj.",
            "image_projector.w3.": "image_projector.up_proj.",
            "image_projector.w2.": "image_projector.down_proj.",
            # language backbone mapping
            "att_proj": "qkv_proj",
            "attn_out": "o_proj",
            "q_norm": "q_norm",
            "k_norm": "k_norm",
            "ff_proj": "up_gate_proj",
            "ff_out": "down_proj",
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        },
        orig_to_new_prefix={
            # vision backbone mapping
            "model.vision_backbone.": "vision_backbone.",
            # language backbone mapping
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "up_gate_proj": ["up_gate_proj"],  # language model
        "merged_linear": ["gate_proj", "up_proj"]  # image_projector
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        kwargs = {}
        for field in fields(VitConfig):
            kwargs[field.name] = getattr(config.vit_config, field.name)
        vit_config = VitConfig(**kwargs)

        kwargs = {}
        for field in fields(AdapterConfig):
            kwargs[field.name] = getattr(config.adapter_config, field.name)
        adapter_config = AdapterConfig(**kwargs)

        self.vision_backbone = SPRVLAVisionBackbone(vit_config, adapter_config, quant_config)
        self.model = SPRVLALlm(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))
        
        self.img_patch_id = None

        self.lm_head = ParallelLMHead(
            config.llm_config.vocab_size,
            config.llm_config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config.llm_config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Optional[SPRVLAImageInputs]:
        images = kwargs.pop("images", None)
        if images is None:
            return None

        if not isinstance(images, (torch.Tensor, list)):
            raise ValueError("Incorrect type of images. "
                             f"Got type: {type(images)}")
        
        pooled_patches_idx = kwargs.pop("pooled_patches_idx", None)
        if not isinstance(pooled_patches_idx, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pooled_patches_idx. "
                             f"Got type: {type(pooled_patches_idx)}")

        num_crops = kwargs.pop("num_crops", None)
        if not isinstance(num_crops, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_crops. "
                             f"Got type: {type(num_crops)}")
        
        num_pooled_patches = kwargs.pop("num_pooled_patches", None)
        if not isinstance(num_pooled_patches, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_pooled_patches. "
                             f"Got type: {type(num_pooled_patches)}")
        
        num_patches = kwargs.pop("num_patches", None)
        if not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_patches. "
                             f"Got type: {type(num_patches)}")

        img_patch_id = kwargs.pop("img_patch_id", None)
        if not isinstance(img_patch_id, torch.Tensor):
            raise ValueError("Incorrect type of img_patch_id. "
                             f"Got type: {type(img_patch_id)}")
        self.img_patch_id = img_patch_id.flatten().unique().item()

        num_crops = flatten_bn(num_crops, concat=True)
        num_pooled_patches = flatten_bn(num_pooled_patches, concat=True)
        num_patches = flatten_bn(num_patches, concat=True)

        return SPRVLAImageInputs(
            images=images,
            pooled_patches_idx=pooled_patches_idx,
            num_crops=num_crops,
            num_pooled_patches=num_pooled_patches,
            num_patches=num_patches,
        )

    def _process_image_input(
        self,
        image_input: SPRVLAImageInputs,
    ) -> list[torch.Tensor]:
        images = image_input["images"]
        pooled_patches_idx = image_input["pooled_patches_idx"]
        num_crops = image_input["num_crops"]
        num_pooled_patches = image_input["num_pooled_patches"]
        num_patches = image_input["num_patches"]

        accum_patches = num_patches.cumsum(dim=0)[:-1]
        for i in range(1, len(pooled_patches_idx)):
            pooled_patches_idx[i] += accum_patches[i - 1]
        
        # Call the vision backbone one the whole batch at one
        images_flat = flatten_bn(images, concat=True)
        pooled_patches_idx_flat = flatten_bn(pooled_patches_idx, concat=True)

        image_features_flat = self.vision_backbone(
            images=images_flat.unsqueeze(0),
            pooled_patches_idx=pooled_patches_idx_flat.unsqueeze(0),
        )

        assert len(image_features_flat) == num_pooled_patches.sum()
        return image_features_flat.split(num_pooled_patches.tolist(), dim=0)

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            assert self.img_patch_id is not None

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.img_patch_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> SamplerOutput:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.model(input_ids,
                                   positions,
                                   intermediate_tensors,
                                   inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        loader = AutoWeightsLoader(self)
        weights = _get_weights_with_merged_embedding(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector="vision_backbone.image_projector",
            tower_model="vision_backbone",
        )


# --- client-side parser ---
class SPRVLAParser:
    def __init__(self, tokenizer, norm_stats: Dict[str, Any], n_action_bins: int = 256):
        # --- Action parsing / de-tokenization setup ---
        # Stats dict expected under config.norm_stats (per-dataset key). If missing, default to empty.
        self.norm_stats = norm_stats or {}
        # Number of discretization bins used for action tokens, defaults to 256.
        self.n_action_bins = int(n_action_bins)
        # Precompute bin centers in [-1, 1] for inverse token to value mapping.
        self.bins = np.linspace(-1.0, 1.0, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        # Lazily constructed tokenizer for converting token strings to ids
        self._qwen_tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_dir: str):
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        tok = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")
        norm_stats = getattr(cfg, "norm_stats", {}) or {}
        n_action_bins = getattr(cfg, "n_action_bins", 256)
        return cls(tok, norm_stats, n_action_bins)

    # ===== Utilities for action parsing / un-normalization =====
    def _check_unnorm_key(self, unnorm_key: Optional[str]) -> str:
        """Validate and resolve which dataset key to use from self.norm_stats."""
        if not self.norm_stats:
            raise ValueError("No norm_stats found in config; cannot unnormalize actions.")
        if unnorm_key is None:
            if len(self.norm_stats) != 1:
                raise ValueError(
                    f"Model has multiple dataset stats; please pass `unnorm_key` from {list(self.norm_stats.keys())}"
                )
            return next(iter(self.norm_stats.keys()))
        if unnorm_key not in self.norm_stats:
            raise ValueError(f"`unnorm_key`={unnorm_key!r} not in {list(self.norm_stats.keys())}")
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Return action dimensionality from q01 stats length for the dataset key."""
        key = self._check_unnorm_key(unnorm_key)
        return len(self.norm_stats[key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Return the full action stats dict for a given dataset key."""
        key = self._check_unnorm_key(unnorm_key)
        return self.norm_stats[key]["action"]

    def parse_action(self, text: str, unnorm_key: Optional[str] = None) -> list:
        """
        Parse a generated text to extract one 1×D action token list, decode to continuous values,
        and unnormalize using dataset-specific stats from `config.norm_stats`.

        This follows the pipeline used in `experiments/robot/libero/main_libero_10_evaluation.py`:
        - Find bracketed token lists following the phrase "the action that the robot should take is" (case-insensitive),
          falling back to any bracketed list in the text.
        - Convert token strings → ids via Qwen2Tokenizer.
        - Map ids → discretized bin indices using: `discretized = vocab_size - token_id - 1` (clipped to bins)
        - Convert bins → normalized actions in [-1, 1] using precomputed `bin_centers`.
        - Unnormalize with q01/q99 and optional `mask` from norm_stats.

        Returns:
            List[float]: unnormalized action vector of length D.
        """
        # Resolve action dimension and stats
        action_dim = self.get_action_dim(unnorm_key)
        stats = self.get_action_stats(unnorm_key)
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
        # the gripper state should not be normalized
        mask[-1] = False

        # Lazily load the tokenizer (shared across calls)
        if self._qwen_tokenizer is None:
            self._qwen_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-7B")

        token_lists = extract_action_token_lists(text, only_len=action_dim)
        action_lists = []

        # Choose the first list (temporal aggregation, if any, should be done by the caller)
        for tokens in token_lists:

            # Convert tokens → ids (replace None with vocab_size to avoid negatives)
            ids = self._qwen_tokenizer.convert_tokens_to_ids(tokens)
            ids = [self._qwen_tokenizer.vocab_size if i is None else int(i) for i in ids]
            ids = np.asarray(ids, dtype=np.int64)

            # ids → discretized bin indices → normalized actions in [-1, 1]
            discretized = self._qwen_tokenizer.vocab_size - ids
            discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized = self.bin_centers[discretized]

            # Unnormalize using per-dimension statistics
            unnorm = 0.5 * (normalized + 1.0) * (q99 - q01) + q01
            actions = np.where(mask, unnorm, normalized)

            action_lists.append([float(x) for x in actions])

        # Return a Python list of float actions
        return action_lists

    def parse_trace(self, text: str) -> list:
        return extract_trace_lists(text, point_len=2, min_points=1)

    def parse_depth(self, text: str) -> list:
        return extract_depth_string(text, include_tags=True)

    def parse_subtask_count(self, text: str) -> int:
        return extract_subtask_count(text)

    def parse_subtask_point(self, text: str) -> list:
        return extract_subtask_goals(text)

    def parse_trace_new(self, text: str) -> list:
        return extract_trajectory(text)


def _get_weights_with_merged_embedding(
    weights: Iterable[Tuple[str, torch.Tensor]]
) -> Iterable[Tuple[str, torch.Tensor]]:
    embedding_weights = {}
    for name, weight in weights:
        if "wte.embedding" in name:
            embedding_weights["embedding"] = weight
        elif "wte.new_embedding" in name:
            embedding_weights["new_embedding"] = weight
        else:
            yield (name, weight)
    # this is compatible with most of quantization,
    # because they won't quantize embed_tokens
    embedding_weights = torch.cat(
        [embedding_weights["embedding"], embedding_weights["new_embedding"]],
        dim=0,
    )
    yield ("model.embed_tokens.weight", embedding_weights)