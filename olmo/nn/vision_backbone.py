import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional

import einops
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from olmo.config import BaseConfig, D, StrEnum
from olmo.nn.image_vit import VitConfig, VisionTransformer
from olmo.nn.llm import Activation
from olmo.torch_util import freeze_module
from olmo.util import resource_path
from torch.nn import functional as F
from torch.distributed.fsdp import fully_shard


class ImagePaddingEmbed(StrEnum):
    """How to embed image padding information"""
    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    """How to pool patch features"""
    attention = "attention"
    attention_meanq = "attention_meanq"
    attention_meanq_2x = "attention_meanq_2x"
    attention_meanq_4x = "attention_meanq_4x"
    attention_2wide = "attention_2wide"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    """How to project the pooled features into the LLM embedding space"""
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


@dataclass
class SPRVLAVisionBackboneConfig(BaseConfig):
    """Vision ViT and the Image/Language Connector"""

    vit: VitConfig = field(default_factory=VitConfig)
    """The vision ViT"""

    image_pooling_2d: ImagePooling2DType = ImagePooling2DType.attention_meanq
    """Layer to pool image features"""

    pooling_attention_mask: bool = False
    """Use an attention mask when pooling instead setting masked embeddings to 0"""

    image_projector: ImageProjectType = ImageProjectType.mlp
    """Layer to project pooled image features to the LLM embedding space"""

    image_padding_embed: Optional[ImagePaddingEmbed] = None
    """
    Image padding mode to use to tell the model what parts of the image are padding
    """

    vit_layers: Tuple = (-1,)
    """What layers to use from the VIT"""

    skip_unused_layers: bool = True
    """Don't load layers we don't need from the ViT"""

    image_feature_dropout: float = 0.0
    """Dropout for image patch features"""

    connector_activation_checkpointing: bool = True
    """Allow activation checkpoint on the connector components"""

    compile_vit: Optional[str] = "blocks"
    """How to compile the ViT"""

    def __post_init__(self):
        self.vit_layers = tuple(self.vit_layers)  # type: ignore[assignment]

    def build(self, llm_config, device):
        return SPRVLAVisionBackbone(self, llm_config, device)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "fix_image_padding" in config:
            assert config.image_padding_embed is None or config.fix_image_padding
            del config["fix_image_padding"]
        for k in ["image_pooling_h", "image_pooling_w"]:
            if k in config:
                assert config.pop(k) == 2
        config.vit = VitConfig.update_legacy_settings(config.vit)
        return config


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(self, config, input_dim: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.hidden_size = config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=device,
            )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        # Activation function.
        self.act = Activation.build(config.activation_type, split_inputs=True)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def reset_parameters(self):
        self.submodule.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class SPRVLAVisionBackbone(nn.Module):
    def __init__(self, config: SPRVLAVisionBackboneConfig, llm_config, device=None):
        super().__init__()
        self.config = config
        input_dim: int = None
        vit_cfg = config.vit
        pool_dim = vit_cfg.image_emb_dim * len(config.vit_layers)

        from olmo.nn.image_vit import ViTMultiHeadDotProductAttention

        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(config.vit, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.attention_2wide, ImagePooling2DType.attention_meanq_2x, ImagePooling2DType.attention_meanq_4x]:
            mha_cfg = deepcopy(config.vit)
            factor = 4 if config.image_pooling_2d ==ImagePooling2DType.attention_meanq_4x else 2
            mha_cfg.image_emb_dim *= factor
            mha_cfg.image_head_dim *= factor
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(mha_cfg, input_dim=pool_dim)
            input_dim = mha_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack]:
            self.image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * vit_cfg.image_emb_dim
            if config.image_pooling_2d == ImagePooling2DType.stack:
                input_dim *= 4
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")

        self.input_dim = input_dim

        if config.image_projector == ImageProjectType.mlp:
            self.image_projector = ImageProjectorMLP(llm_config, input_dim, device=device)
        elif config.image_projector == ImageProjectType.linear:
            self.image_projector = nn.Linear(input_dim, llm_config.d_model, bias=False, device=device)
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)

        self.vit_layers = []
        for layer in config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(config.vit.image_num_layers + layer)
        last_layer_needed = (max(self.vit_layers)+1)

        vit_cfg = self.config.vit
        if last_layer_needed < config.vit.image_num_layers:
            if self.config.skip_unused_layers:
                vit_cfg = replace(vit_cfg, image_num_layers=last_layer_needed)
                self.image_vit: VisionTransformer = vit_cfg.build(device)
            else:
                # We might need to keep the layers for checkpoint compatibility, but we
                # freeze them since unfrozen layers with no gradient confuses torch's distributed
                # optimizer checkpointer
                self.image_vit: VisionTransformer = vit_cfg.build(device)
                for block in self.image_vit.transformer.resblocks[last_layer_needed-1:]:
                    freeze_module(block)
        else:
            self.image_vit: VisionTransformer = vit_cfg.build(device)

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = vit_cfg.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=device))
            else:
                raise ValueError(config.image_padding_embed)

    @classmethod
    def build(cls, config: SPRVLAVisionBackboneConfig, outut_dim, device=None) -> 'SPRVLAVisionBackbone':
        return SPRVLAVisionBackbone(config, outut_dim, device)

    def reset_connector_parameters(self):
        if self.image_pooling_2d is not None:
            self.image_pooling_2d.reset_parameters()
        if self.image_projector == "2mlp":
            for module in self.image_projector:
                module.reset_parameters()
        elif self.image_projector == "linear":
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector.reset_parameters()
        if self.pad_embed is not None:
            nn.init.zeros_(self.pad_embed)

    def reset_parameters(self):
        self.reset_connector_parameters()
        self.image_vit.reset_parameters()

    def reset_with_pretrained_weights(self):
        self.reset_connector_parameters()  # resets the connector
        self.image_vit.reset_with_pretrained_weights()

    def apply_fsdp2(self, **kwargs):
        self.image_vit.apply_fsdp2(**kwargs)
        fully_shard(self.image_pooling_2d, **kwargs)
        fully_shard(self.image_projector, **kwargs)
        # For any remaining parameters in `self`, like the pad embed
        fully_shard(self, **kwargs)

    def apply_activation_checkpointing(self):
        self.image_vit.apply_activation_checkpointing()
        if self.config.connector_activation_checkpointing:
            self.image_projector = checkpoint_wrapper(self.image_projector)
            self.image_pooling_2d = checkpoint_wrapper(self.image_pooling_2d)

    def apply_compile(self, **kwargs):
        self.image_pooling_2d.compile(**kwargs)
        self.image_projector.compile(**kwargs)
        if self.config.compile_vit == "blocks":
            for block in self.image_vit.transformer.resblocks:
                block.compile(**kwargs)
        elif self.config.compile_vit is not None:
            raise NotImplementedError(self.config.compile_vit)

    def get_connector_parameters(self):
        vit_params = set(self.image_vit.parameters())
        return (p for p in self.parameters() if p not in vit_params)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
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

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor,
                pooled_patches_idx: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=torch.float32)
                all_pad = all_pad.to(dtype=torch.float32)
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]

        multiple_pooling = isinstance(pooled_patches_idx, (tuple, list))
        if not multiple_pooling:
            pooled_patches_idxs = [pooled_patches_idx]
        else:
            pooled_patches_idxs = pooled_patches_idx

        all_pooled_features = []
        for pooled_patches_idx in pooled_patches_idxs:
            valid = pooled_patches_idx >= 0
            valid_token = torch.any(valid, -1)

            # Use `pooled_patches_idx` to arange the features for image pooling
            batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
            batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]])

            # Now [batch, num_high_res_features, pool_dim, dim]
            to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
            to_pool = to_pool * valid.float()[:, :, :, None]
            to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
            if self.config.pooling_attention_mask:
                attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
            else:
                attn_mask = None

            if cfg.image_pooling_2d in [ImagePooling2DType.attention_meanq, ImagePooling2DType.attention_meanq_2x, ImagePooling2DType.attention_meanq_4x]:
                if self.config.pooling_attention_mask:
                    denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
                    denom = torch.where(denom == 0, 1, denom)
                    query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
                else:
                    query = to_pool.mean(-2, keepdim=True)
                pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
            elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
                pooled_features = self.image_pooling_2d(to_pool[:, :1, :], to_pool, attn_mask=attn_mask)
            else:
                pooled_features = to_pool

            pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

            # MLP layer to map the feature.
            if cfg.image_projector == ImageProjectType.mlpx2:
                for module in self.image_projector:
                    pooled_features = module(pooled_features)
            else:
                pooled_features = self.image_projector(pooled_features)
            all_pooled_features.append((pooled_features, valid_token))

        if multiple_pooling:
            return all_pooled_features
        else:
            image_features, valid_token = all_pooled_features[0]
            return image_features.view(-1, image_features.shape[-1])[valid_token.flatten()]
