import argparse
from typing import Optional
import numpy as np
import requests
import warnings
import logging
from io import BytesIO
import base64
import PIL
from PIL import Image, ImageFile, ImageOps
import os

import torch
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.sampling_params import SamplingParams
from olmo.vllm.sprvla.sprvla import SPRVLAForActionReasoning, SPRVLAParser
ModelRegistry.register_model("SPRVLAForActionReasoning", SPRVLAForActionReasoning)
_MULTIMODAL_MODELS["SPRVLAForActionReasoning"] = ("sprvla", "SPRVLAForActionReasoning")

from transformers import AutoProcessor


def apply_chat_template(processor: AutoProcessor, text: str):
    messages = [
        {
            "role": "user",
            "content": [dict(type="text", text=text)]
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt


def vllm_inference(model_dir: str):

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        # padding_side="left",
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0
    )

    parser = SPRVLAParser.from_pretrained(model_dir)

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
    )

    prompt = (
        "The task is pick up the black bowl on the stove and place it on the plate. "
        "What is the action that the robot should take. "
        "To figure out the action that the robot should take to pick up the black bowl on the stove and place it on the plate, let's think through it step by step. "
        "First, what is the depth map for this image? "
        "Second, what is the trajectory of the end effector? "
        "Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"
    )

    img1 = Image.open("/weka/oe-training-default/jiafeid/SPRVLA/data/libero_modified_proprio/libero_spatial_no_noops/primary/0000000/0000.png")
    img2 = Image.open("/weka/oe-training-default/jiafeid/SPRVLA/data/libero_modified_proprio/libero_spatial_no_noops/wrist/0000000/0000.png")
    img = [img1, img2]


    inputs = [
        {
            "prompt": apply_chat_template(processor, prompt),
            "multi_modal_data": {
                "image": [img]
            },
        },
    ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(generated_text)

    depth = parser.parse_depth(generated_text)
    print(depth)

    trace = parser.parse_trace(generated_text)
    print(trace)

    action = parser.parse_action(generated_text, unnorm_key="fractal20220817_data")
    print(action)


def main():

    model_dir = "/weka/oe-training-default/jiafeid/SPRVLA/checkpoints-sprvla/SPRVLA-7B-D-Pretrain-0812"

    vllm_inference(model_dir)


if __name__ == "__main__":
    main()