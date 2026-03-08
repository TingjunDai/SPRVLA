import logging
from typing import Dict
from dataclasses import replace

from olmo.models.sprvla.data_formatter import DataFormatter
from olmo.models.sprvla.model_preprocessor import SPRVLAPreprocessorConfig
from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.inf_evaluator import InfDatasetEvaluatorConfig, EvaluatorConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig, AttentionType, LayerNormType, AttentionLayerNormType
from olmo.models.sprvla.sprvla import SPRVLAConfig
from olmo.tokenizer import TokenizerConfig
from olmo.nn.vision_backbone import SPRVLAVisionBackboneConfig

log = logging.getLogger(__name__)


DEBUG_MODEL = SPRVLAConfig(
    llm=LlmConfig(
        d_model=128,
        n_heads=2,
        n_layers=1,
        max_sequence_length=4096,
        additional_vocab_size=128,
        vocab_size=152064,
        rope=True,
        embedding_size=None,
        weight_tying=False,
        tokenizer=TokenizerConfig(
            identifier="Qwen/Qwen2-7B",
        )
    ),
    vision_backbone=SPRVLAVisionBackboneConfig(
        vit=VitConfig(image_num_layers=1)
    ),
    data_formatter=DataFormatter(),
    mm_preprocessor=SPRVLAPreprocessorConfig(crop_mode="resize", max_crops=1)
)


def get_evaluator(name) -> EvaluatorConfig:
    """Gets the default evaluator for task `name`"""

    if name in ["text_vqa", "okvqa", "coco_2014_vqa", "coco_2014_vqa_multi"]:
        return EvaluatorConfig(vqa_eval="vqa_score")
    elif name.startswith("math_vista"):
        return EvaluatorConfig(math_vista_eval=True)
    elif name == "a_okvqa_da":
        return EvaluatorConfig(vqa_eval="a_okvqa_score")
    elif name.startswith("android_control"):
        return EvaluatorConfig(android_eval=True)
    elif name == "vqa_v2_test":
        return EvaluatorConfig()
    elif name.startswith("chart_qa"):
        return EvaluatorConfig(vqa_eval="relaxed_correctness,scifi_relaxed_correctness,em")
    elif name in ["doc_qa", "info_qa", "st_qa"]:
        return EvaluatorConfig(vqa_eval="ansl,em")
    elif name in ["gqa", "tally_qa"]:
        return EvaluatorConfig(vqa_eval="em")
    elif name in ["science_qa", "a_okvqa_mc", "science_qa_img", "ai2_diagram", "ai2_diagram_v2", "ai2_diagram_v2_transparent", "muir_bench_mc"]:
        return EvaluatorConfig(vqa_eval="mc")
    elif name in ["ai2_diagram_v2_mix_transparent", "ai2_diagram_v2_mix_transparent_one_style"]:
        return EvaluatorConfig(vqa_eval="mc_ai2d_opaque,mc_ai2d_transparent")
    elif name.startswith("mmmu"):
        return EvaluatorConfig(vqa_eval="mmmu_score")
    elif name.startswith("countbench_qa") or name.startswith("pixmo_count"):
        return EvaluatorConfig(point_count_eval=True)
    elif name.startswith("real_world_qa"):
        return EvaluatorConfig(vqa_eval="real_world_qa_score")
    elif name == "pixmo_clocks":
        return EvaluatorConfig(clock_eval=True)
    elif name == "pointing_eval":
        return EvaluatorConfig(pointing_eval=True)
    elif name == "clock_bench":
        return EvaluatorConfig(clock_bench_eval=True)
    elif name in ["countbench_qa"]:
        return EvaluatorConfig(count_eval=True)
    elif name.startswith("temp_compass"):
        disable_api = "disable_api" in name
        name = name.replace("_disable_api", "")
        task = '_'.join(name.split("_")[2:]) if len(name.split("_")) > 2 else "all"
        return EvaluatorConfig(temp_compass_eval=task, temp_compass_disable_api=disable_api)
    elif name.startswith("plm_fgqa_eval"):
        return EvaluatorConfig(plm_fgqa_eval=True)
    elif name == "mlvu_gen":
        return EvaluatorConfig(mlvu_gen_eval=True)
    elif name == "ego_schema":
        return EvaluatorConfig(vqa_eval="ego_schema_mc")
    elif name == "perception_test":
        return EvaluatorConfig(vqa_eval="perception_test_mc")
    elif name == "nextqa_mc":
        return EvaluatorConfig(vqa_eval="nextqa_mc")
    elif name == "muir_bench":
        return EvaluatorConfig(vqa_eval="muir_bench_mc")
    elif name in ["dense_caption_eval", "user_qa", "vqa_v2_test", "intern_vid"]:
        # No metrics, but still save prediction file
        return EvaluatorConfig()
    else:
        raise NotImplementedError(name)


def get_default_max_tokens(name):
    if name == "dense_caption_eval":
        return 448
    elif name.startswith("named_entity"):
        max_new_tokens = 256
    elif name == "math_vista_demo":
        max_new_tokens = 384
    elif name in ["chart_qa_scifi", "chart_qa_ex", "chart_qa_exp", "chart_qa_prompting_explanation"] or name.endswith("_demo"):
        max_new_tokens = 256
    elif name.startswith("user_questions_for_elo"):
        max_new_tokens = 768  # Can have counts of 20+ so make sure there is room
    elif name in ["pointing_eval", "pointing"]:
        max_new_tokens = 192  # 192 is enought for counts <=10 in the point tag format
    elif "countbench_qa" in name or "pixmo_count" in name:
        max_new_tokens = 192
    elif name == "android_control_hl_cot":
        max_new_tokens = 64
    elif name.startswith("android_control"):
        max_new_tokens = 16
    else:
        max_new_tokens = 12
    return max_new_tokens


def get_evaluation(name, seq_len, max_examples, for_inference=True,
                   num_workers=2, device_batch_size=None,
                   persistent_workers=False, include_image=False) -> InfDatasetEvaluatorConfig:
    """Gets the default evaluation config for task (or task:split string) `name`"""
    if ":" in name:
        name, split = name.split(":")
    else:
        split = None

    if name == "chart_qa_weighted":
        name = "chart_qa"
    if name == "coco_2014_vqa_multi":
        name = "coco_2014_vqa"

    eval_only_tasks = ["mmmu", "mme", "math_vista", "real_world_qa", "seed_bench",
                       "mmbench", "sugar_crepe", "blink"]
    eval_only_tasks += [task_name + "_test" for task_name in eval_only_tasks]
    if name == "tall_qa_count":
        task_name = "tally_qa"
    elif name in eval_only_tasks:
        task_name = name + "_test" if not name.endswith("_test") else name
    else:
        task_name = name
    test_eval_tasks = ["mme_test", "real_world_qa_test", "real_world_qa_test", "count_bench",
                       "seed_bench_test", "sugar_crepe_test", "count_bench_from_caption", "pointing_test"]
    if split is None:
        split = "test" if task_name in test_eval_tasks else "validation"

    ds = DataLoaderConfig(
        dataset=task_name, sequence_length=seq_len,
        split=split, shuffle=True, 
        drop_last=max_examples is not None and max_examples >= 0,
        num_workers=num_workers, pad="to_max", pin_memory=True,
        seed=691203,
        persistent_workers=persistent_workers
    )

    if for_inference:
        evaluator = get_evaluator(name)
        evaluator.num_wandb_examples = 64
        evaluator.num_wandb_examples = 32
        evaluator.n_to_log = 0
        evaluator.save_predictions = None

        max_new_tokens = get_default_max_tokens(name)

        return InfDatasetEvaluatorConfig(
            max_examples=max_examples,
            device_batch_size=device_batch_size,
            max_new_tokens=max_new_tokens,
            evaluator=evaluator,
            label="ai2_diagram" if "ai2_diagram" in name else name,
            data=ds,
            console_log_interval="${console_log_interval}",  # Use log interval in top-level config
            include_image=include_image,
        )
            
    else:
        return LossDatasetEvaluatorConfig(
            max_examples=max_examples,
            device_batch_size=device_batch_size,
            label="ai2_diagram" if "ai2_diagram" in name else name,
            data=ds,
            console_log_interval="${console_log_interval}"  # Use log interval in top-level config
        )


DEBUG_VISION_BACKBONE = VitConfig(
    init_path=None,
    resize_mode="siglip",
    image_model_type="openai",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=128,
    image_num_heads=2,
    image_num_key_value_heads=2,
    image_num_layers=2,
    image_head_dim=64,
    image_mlp_dim=256,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


DEFAULT_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


SIGLIP_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=27,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=729, # no CLS token
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
    normalize="siglip"
)


SIGLIP2_VISION_BACKBONE = replace(
    SIGLIP_VISION_BACKBONE,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/siglip2-so400m-14-384.pt",
)


DINOV2_LARGE_336_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    image_model_type="dino",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
    normalize="dino",
)


METACLIP_L14_336_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


OLMOE = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmoe.pt",
    d_model=2048,
    n_heads=16,
    n_layers=16,
    mlp_ratio=1,
    activation_type='swiglu',
    block_type='moe',
    rope=True,
    rope_full_precision=True,
    rope_theta=10000.0,
    attention_type='sdpa',
    attention_layer_norm=True,
    residual_dropout=0.1,
    response_residual_dropout=0.0,
    embedding_dropout=0.0,
    layer_norm_type='rms',
    layer_norm_with_affine=True,
    layer_norm_eps=1e-05,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    max_position_embeddings=32768,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=50280,
    embedding_size=50304,
    additional_vocab_size=128,
    new_embedding_init_range=0.02,
    weight_tying=False,
    normalize_input_embeds=False,
    use_position_ids=True,

    # MOE parameters
    moe_num_experts=64,
    moe_top_k=8,
    moe_mlp_impl='sparse',
    moe_log_expert_assignment=False,
    moe_shared_expert=False,
    moe_lbl_in_fp32=False,
    moe_interleave=False,
    moe_loss_weight=0.0,
    moe_zloss_weight=0.0,
    moe_dropless=True,
    moe_capacity_factor=1.25,

    tokenizer=TokenizerConfig(
        identifier='allenai/OLMoE-1B-7B-0924',
    ),
    fix_pad_tokenizer=True,
)


OLMO_1024_PREVIEW = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo-1024-preview.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/dolma2-tokenizer",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


OLMO2_1124_7B = replace(
    OLMO_1024_PREVIEW,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo2-1124-7b.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-1124-7B",
    ),
)


OLMO2_1124_13B = replace(
    OLMO_1024_PREVIEW,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo2-1124-13b.pt",
    d_model=5120,
    n_heads=40,
    n_layers=40,
    mlp_hidden_size=27648,
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-1124-13B",
    ),
)


OLMO2_1124_13B_INSTRUCT = replace(
    OLMO2_1124_13B,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo2-1124-13b-instruct.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-1124-13B-Instruct",
    ),
)


OLMO2_0325_32B = replace(
    OLMO_1024_PREVIEW,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo2-0325-32b.pt",
    d_model=5120,
    n_heads=40,
    n_kv_heads=8,
    n_layers=64,
    mlp_hidden_size=55296,
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-0325-32B",
    ),
)


OLMO2_0325_32B_INSTRUCT = replace(
    OLMO2_0325_32B,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo2-0325-32b-instruct.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-0325-32B-Instruct",
    ),
)


QWEN2_7B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2-7b.pt",
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
    ),
)


QWEN25_15B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-1.5b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=1536,
    mlp_hidden_size=8960*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=12,
    n_kv_heads=2,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-1.5B",
    ),
)


QWEN25_3B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-3b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2048,
    mlp_hidden_size=11008*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=16,
    n_kv_heads=2,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-3B",
    ),
)


QWEN25_7B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-7b.pt",
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-7B",
    ),
)


QWEN25_14B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-14b.pt",
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=5120,
    mlp_hidden_size=13824*2,
    n_layers=48,
    additional_vocab_size=128,
    n_heads=40,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-14B",
    ),
)


QWEN25_14B_INSTRUCT = replace(
    QWEN25_14B,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-14b-instruct.pt",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-14B-Instruct",
    ),
    layer_norm_eps=1e-6,
    # The only difference is the layer norm eps
    # and the tokenizer identifier
)


QWEN2_72B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2-70b.pt",
    additional_vocab_size=128,
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=8192,
    mlp_hidden_size=29568*2,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-72B",
    ),
)


OMLO_19_13B = LlmConfig(
    d_model=5120,
    n_heads=40,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=40,
    mlp_ratio=4,
    mlp_hidden_size=27648,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    init_fn="normal",
    init_std=0.02,
    init_cutoff_factor=3.0,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/dolma2-tokenizer",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


LLAMA31_TULU31_8B = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/llama3.1-tulu3.1-8b.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=8,
    qkv_bias=False,
    n_layers=32,
    mlp_hidden_size=14336*2,
    block_type="llama",
    rope=True,
    rope_theta=500000.0,
    rope_type="llama3",
    rope_factor=8.0,
    rope_high_freq_factor=4.0,
    rope_low_freq_factor=1.0,
    rope_original_max_position_embeddings=8192,
    attention_dropout=0,
    residual_dropout=0,
    response_residual_dropout=0,
    layer_norm_type=LayerNormType.rms,
    layer_norm_eps=1e-5,
    max_sequence_length=4096,
    include_bias=False,
    embedding_dropout=0,
    vocab_size=128384, # multiple of 128
    additional_vocab_size=128,
    weight_tying=False,
    embedding_size=128384,
    tokenizer=TokenizerConfig(
        identifier="allenai/Llama-3.1-Tulu-3.1-8B",
    ),
)


QWEN3_8B_BASE = LlmConfig(
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen3-8b-base.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=False,
    include_bias=False,
    embedding_size=151936,
    d_model=4096,
    mlp_hidden_size=12288*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-8B-Base",
    ),
)


QWEN3_8B = replace(
    QWEN3_8B_BASE,
    init_path="${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen3-8b.pt",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-8B",
    ),
)


DEFAULT_LOAD_PATHS = {
    "openai": "${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    "siglip": "${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    "dinov2_large_336": "${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    "metaclip_l14_336": "${oc.env:SPRVLA_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    "olmoe": "${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmoe.pt",
    "olmo_1024_preview": "${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/olmo-1024-preview.pt",
    "qwen2.5_1.5b": "${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-1.5b.pt",
    "qwen2.5_3b": "${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2.5-3b.pt",
    "qwen2_7b": "${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2-7b.pt",
    "qwen2_72b": "${oc.env:SPRVLA_DATA_DIR}/pretrained_llms/qwen2-70b.pt",
}


VISION_BACKBONES: Dict[str, VitConfig] = {
    "debug": DEBUG_VISION_BACKBONE,
    "openai": DEFAULT_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "siglip2": SIGLIP2_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_LARGE_336_VISION_BACKBONE,
    "metaclip_l14_336": METACLIP_L14_336_VISION_BACKBONE,
}


LLMS: Dict[str, LlmConfig] = {
    "olmoe": OLMOE,
    "olmo_1024_preview": OLMO_1024_PREVIEW,
    "olmo2_1124_7b": OLMO2_1124_7B,
    "olmo2_1124_13b": OLMO2_1124_13B,
    "olmo2_1124_13b_instruct": OLMO2_1124_13B_INSTRUCT,
    "olmo2_0325_32b": OLMO2_0325_32B,
    "olmo2_0325_32b_instruct": OLMO2_0325_32B_INSTRUCT,
    "qwen2_7b": QWEN2_7B,
    "qwen2_72b": QWEN2_72B,
    "qwen2.5_14b_instruct": QWEN25_14B_INSTRUCT,
    "qwen2.5_14b": QWEN25_14B,
    "qwen2.5_7b": QWEN25_7B,
    "qwen2.5_3b": QWEN25_3B,
    "qwen2.5_1.5b": QWEN25_15B,
    "olmo1120_13b": OMLO_19_13B,
    "llama3.1_tulu3.1_8b": LLAMA31_TULU31_8B,
    "qwen3_8b_base": QWEN3_8B_BASE,
    "qwen3_8b": QWEN3_8B,
}

