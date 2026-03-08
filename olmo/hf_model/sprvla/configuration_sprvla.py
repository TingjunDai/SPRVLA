"""
SPRVLA configuration
"""

from typing import Tuple, Optional, Dict, Any

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SPRVLAVitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SPRVLAVisionTransformer`].
    It is used to instantiate a `SPRVLAVisionTransformer` according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import SPRVLAVitConfig, SPRVLAVisionTransformer

    >>> # Initializing a SPRVLAVitConfig
    >>> configuration = SPRVLAVitConfig()

    >>> # Initializing a SPRVLAVisionTransformer (with random weights)
    >>> model = SPRVLAVisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sprvla_vit"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        image_default_input_size: Tuple[int, int] = (378, 378),
        image_patch_size: int = 14,
        image_num_pos: int = 577,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        initializer_range: float = 0.02,
        float32_attention: bool = True,
        use_cls_token: bool = False,      # True for OpenCLIP
        patch_bias: bool = True,          # False for OpenCLIP
        pre_layernorm: bool = False,      # True for OpenCLIP
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.image_default_input_size = image_default_input_size
        self.image_patch_size = image_patch_size
        self.image_num_pos = image_num_pos
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.initializer_range = initializer_range
        self.float32_attention = float32_attention
        self.use_cls_token = use_cls_token
        self.patch_bias = patch_bias
        self.pre_layernorm = pre_layernorm

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class SPRVLAAdapterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of SPRVLAAdapter. With SPRVLAVitConfig,
    It is used to instantiate an SPRVLAVisionBackbone according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import SPRVLAVitConfig, SPRVLAAdapterConfig, SPRVLAVisionBackbone

    >>> # Initializing a SPRVLAVitConfig and a SPRVLAAdapterConfig
    >>> vit_config = SPRVLAVitConfig()
    >>> adapter_config = SPRVLAPoolingConfig()

    >>> # Initializing a SPRVLAVisionBackbone (with random weights)
    >>> model = SPRVLAVisionBackbone(vit_config, adapter_config)

    >>> # Accessing the model configuration
    >>> vit_configuration = model.vit_config
    >>> adapter_configuration = model.adapter_config
    ```"""

    def __init__(
        self,
        vit_layers: Tuple = (-3, -9),
        hidden_size: int = 1152,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = 18944,
        text_hidden_size: int = 3584,
        image_feature_dropout: float = 0.0,
        initializer_range: float = 0.02,
        # pooling_mode: str = "indices",            # "indices" (SigLIP) or "2x2_attention" (OpenCLIP)
        image_padding_embed: Optional[str] = None,  # e.g. "pad_and_partial_pad"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vit_layers = vit_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.text_hidden_size = text_hidden_size
        self.image_feature_dropout = image_feature_dropout
        self.initializer_range = initializer_range
        # self.pooling_mode = pooling_mode
        self.image_padding_embed = image_padding_embed


class SPRVLALlmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SPRVLALlm`]. It is used to instantiate a
    `SPRVLALlm` according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import SPRVLALlmConfig, SPRVLALlm

    >>> # Initializing a SPRVLALlmConfig
    >>> configuration = SPRVLALlmConfig()

    >>> # Initializing a SPRVLALlm (with random weights)
    >>> model = SPRVLALlm(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sprvla_llm"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "blocks.*.self_attn.att_proj": "colwise",
        "blocks.*.self_attn.attn_out": "rowwise",
        "blocks.*.mlp.ff_proj": "colwise",
        "blocks.*.mlp.ff_out": "rowwise",
    }
    base_model_pp_plan = {
        "wte": (["input_ids"], ["inputs_embeds"]),
        "blocks": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "ln_f": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: Optional[int] = 4,
        head_dim: int = 128,
        vocab_size: int = 152064,
        additional_vocab_size: int = 128,
        qkv_bias: bool = True,
        num_hidden_layers: int = 48,
        intermediate_size: int = 18944,
        hidden_act: str = "silu",
        embedding_dropout: float=0.0,
        attention_dropout: float=0.0,
        residual_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1000000.0,
        rope_scaling: Dict[str, Any] = None,
        use_qk_norm: bool = False,
        qk_norm_type: str = "olmo",
        layer_norm_eps: int = 1e-6,
        norm_after: bool = False,
        initializer_range: float = 0.02,
        use_cache=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.qkv_bias = qkv_bias
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.norm_after = norm_after
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)


class SPRVLAConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SPRVLAForActionReasoning`].
    It is used to instantiate an SPRVLA model according to the specified arguments, defining the model architecture.

    Example:

    ```python
    >>> from transformers import SPRVLAConfig, SPRVLAVitConfig, SPRVLAAdapterConfig, SPRVLALlmConfig

    >>> # Initializing a SPRVLAVitConfig
    >>> vit_config = SPRVLAVitConfig()

    >>> # Initializing a SPRVLAAdapterConfig
    >>> adapter_config = SPRVLAAdapterConfig()

    >>> # Initializing a SPRVLALlmConfig
    >>> llm_config = SPRVLALlmConfig()

    >>> # Initializing a SPRVLAConfig
    >>> configuration = SPRVLAConfig(vit_config, adapter_config, llm_config, image_patch_id=152069)

    >>> # Initializing a model
    >>> model = SPRVLAForActionReasoning(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sprvla"
    sub_configs = {
        "llm_config": SPRVLALlmConfig,
        "vit_config": SPRVLAVitConfig,
        "adapter_config": SPRVLAAdapterConfig,
    }

    def __init__(
        self,
        vit_config: SPRVLAVitConfig = None,
        adapter_config: SPRVLAAdapterConfig = None,
        llm_config: SPRVLALlmConfig = None,
        image_patch_id: int = None,
        initializer_range: float = 0.02,
        n_action_bins: int = 256,
        norm_stats: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vit_config is None:
            self.vit_config = SPRVLAVitConfig()
        elif isinstance(vit_config, dict):
            self.vit_config = SPRVLAVitConfig(**vit_config)
        else:
            self.vit_config = vit_config
        if adapter_config is None:
            self.adapter_config = SPRVLAAdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = SPRVLAAdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
        if llm_config is None:
            self.llm_config = SPRVLALlmConfig()
        elif isinstance(llm_config, dict):
            self.llm_config = SPRVLALlmConfig(**llm_config)
        else:
            self.llm_config = llm_config
        self.image_patch_id = image_patch_id
        self.initializer_range = initializer_range

        self.n_action_bins = n_action_bins
        self.norm_stats = norm_stats

    @property
    def image_num_patch(self):
        assert self.vit_config is not None
        return self.vit_config.image_num_patch
    
    @property
    def num_attention_heads(self):
        return self.llm_config.num_attention_heads
    
    @property
    def num_key_value_heads(self):
        return self.llm_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.llm_config.head_dim

    @property
    def num_hidden_layers(self):
        return self.llm_config.num_hidden_layers
    
    @property
    def hidden_size(self):
        return self.llm_config.hidden_size
    
    @property
    def vocab_size(self):
        return self.llm_config.vocab_size
    
    @property
    def max_position_embeddings(self):
        return self.llm_config.max_position_embeddings


SPRVLAVitConfig.register_for_auto_class()
SPRVLAAdapterConfig.register_for_auto_class()
SPRVLALlmConfig.register_for_auto_class()
SPRVLAConfig.register_for_auto_class()