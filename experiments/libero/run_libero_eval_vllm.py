import os
import argparse
from collections import defaultdict, deque
import numpy as np
import cv2
from PIL import Image
import torch
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
)
from robot_utils import (
    DATE,
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)
import tqdm
import ast
from libero.libero import benchmark
from transformers import AutoProcessor
import math
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.sampling_params import SamplingParams
from sprvla import SPRVLAForActionReasoning, SPRVLAParser
import imageio
import json
import pickle

ModelRegistry.register_model("SPRVLAForActionReasoning", SPRVLAForActionReasoning)
_MULTIMODAL_MODELS["SPRVLAForActionReasoning"] = ("sprvla", "SPRVLAForActionReasoning")


class PromptManager:
    """State tracker for dynamic prompt switching (simplified: only detects subtask count changes)"""
    
    def __init__(self):
        # Subtask count tracking (last 4 entries)
        self.subtask_history = deque(maxlen=4)

        # Prompt state
        self.use_reset_prompt = False  # whether to use reset prompt
        self.reset_prompt_counter = 0  # number of times reset prompt has been used
        self.reset_prompt_duration = 3  # number of steps to keep using reset prompt

        # Original task description
        self.original_task_description = None
        
    def update_subtask_count(self, subtask_count):
        """Update subtask count history"""
        if subtask_count is not None:
            self.subtask_history.append(subtask_count)
            print(f"[PromptManager] Subtask history: {list(self.subtask_history)}")
    
    def check_subtask_trigger(self):
        """
        Check whether the subtask count change triggers a reset condition.

        Trigger conditions (based on n-2, n-1, n):
        1. n-2 is m, and both n-1 and n are m+1 or larger (increasing trend)
        2. n-2 is m, and both n-1 and n are m-2 or smaller (significant decreasing trend)

        Suppression condition (requires 4 history entries):
        - If n-3, n-1, n all have the same value and only n-2 is anomalous, do not trigger reset
        - Example: [2, 1, 2, 2] does not trigger (n-3=2, n-2=1, n-1=2, n=2)

        Requires at least 3 history entries to check basic trigger conditions.
        """
        if len(self.subtask_history) < 3:
            return False
        
        history = list(self.subtask_history)
        
        # Get recent values
        if len(history) == 3:
            # Only 3 values: [n-2, n-1, n]
            prev = history[0]      # n-2
            current = history[1]   # n-1
            next_val = history[2]  # n
            n_3 = None
        else:  # len(history) == 4
            # 4 values: [n-3, n-2, n-1, n]
            n_3 = history[0]       # n-3
            prev = history[1]      # n-2
            current = history[2]   # n-1
            next_val = history[3]  # n
        
        # Check basic trigger conditions (based on n-2, n-1, n)
        trigger_increase = current >= prev + 1 and next_val >= prev + 1
        trigger_decrease = current <= prev - 2 and next_val <= prev - 2
        
        if not (trigger_increase or trigger_decrease):
            return False
        
        # If 4 history values exist, check suppression condition
        if n_3 is not None:
            n_3_trigger_increase = current >= n_3 + 1 and next_val >= n_3 + 1
            n_3_trigger_decrease = current <= n_3 - 2 and next_val <= n_3 - 2
            if not (n_3_trigger_increase or n_3_trigger_decrease):
                print(f"[PromptManager] Detected isolated anomaly at frame n-2, not triggering reset: [{n_3}, {prev}, {current}, {next_val}]")
                return False
            # If n-3, n-1, n are all the same, n-2 is an isolated anomaly; do not trigger
            # if n_3 == current == next_val:
            #     print(f"[PromptManager] Detected isolated anomaly at frame n-2, not triggering reset: [{n_3}, {prev}, {current}, {next_val}]")
            #     return False
            # elif n_3 - 1 == current == next_val:
            #     print(f"[PromptManager] Detected isolated anomaly at frame n-2, not triggering reset: [{n_3}, {prev}, {current}, {next_val}]")
            #     return False
        
        # Trigger reset
        if trigger_increase:
            print(f"[PromptManager] Detected abnormal subtask count increase: history={history}")
        else:
            print(f"[PromptManager] Detected abnormal subtask count decrease: history={history}")
        
        return True
    
    def update_state(self, subtask_count):
        """
        Update state and determine whether to switch prompts.

        Args:
            subtask_count: current number of subtasks

        Returns:
            bool: whether the reset prompt should be used
        """
        # If currently using the reset prompt
        if self.use_reset_prompt:
            self.reset_prompt_counter += 1
            print(f"[PromptManager] Reset mode in progress: {self.reset_prompt_counter}/{self.reset_prompt_duration}")
            
            # Duration reached, revert to normal
            if self.reset_prompt_counter >= self.reset_prompt_duration:
                print(f"[PromptManager] Reset prompt duration complete, reverting to normal prompt")
                self.use_reset_prompt = False
                self.reset_prompt_counter = 0
                self.subtask_history.clear()
            
            return True
        
        # Update history
        self.update_subtask_count(subtask_count)
        
        # Check whether reset condition is triggered
        if self.check_subtask_trigger():
            print(f"[PromptManager] [WARNING] Reset condition triggered, switching to reset prompt")
            self.use_reset_prompt = True
            self.reset_prompt_counter = 0
            return True
        
        return False
    
    def get_prompt(self, original_task_description):
        """Get the prompt that should currently be used"""
        if self.original_task_description is None:
            self.original_task_description = original_task_description
        
        if self.use_reset_prompt:
            return "return to initial position"
        else:
            return self.original_task_description


def crop_and_resize_pil(img: Image.Image, crop_scale: float) -> Image.Image:
    """
    Center‐crop a PIL image to crop_scale of its area,
    then resize back to the ORIGINAL image size.
    """
    w, h = img.size
    # sqrt(crop_scale) to get relative side length
    rel = math.sqrt(crop_scale)
    cw, ch = int(w * rel), int(h * rel)
    left = (w - cw) // 2
    top  = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    # resize back to the original dimensions (w, h)
    return cropped.resize((w, h), Image.BILINEAR)


def center_crop_image(img: Image.Image) -> Image.Image:
    # fixed 0.9 area scale
    return crop_and_resize_pil(img, 0.9)


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    """
    if isinstance(action, list):
        action = np.array(action)
    
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    """
    if isinstance(action, list):
        action = np.array(action)
    
    action[..., -1] = action[..., -1] * -1.0
    return action


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


def annotate_frame(frame, traj, sub_count, sub_point, use_reset_prompt, reset_counter, reset_duration, max_subtasks=4):
    """
    Add annotations to the original frame and extend the canvas to accommodate subtask descriptions.

    Args:
        max_subtasks: maximum number of subtasks across the entire episode, used to ensure consistent frame sizes

    Returns:
        annotated_image: annotated extended image
        annotation_data: annotation data dictionary for the current frame
    """
    img_h, img_w = frame.shape[:2]
    
    # Calculate extra height needed based on max subtask count
    # Each subtask description takes ~25 pixels (including line spacing), plus 20 pixels for margins
    line_height = 25
    extra_height = max_subtasks * line_height + 20 if max_subtasks > 0 else 0
    
    # Create extended canvas
    extended_h = img_h + extra_height
    extended_img = np.zeros((extended_h, img_w, 3), dtype=np.uint8)
    extended_img[:img_h, :] = frame
    
    # Fill extra area with dark gray background if present
    if extra_height > 0:
        extended_img[img_h:, :] = (40, 40, 40)
    
    # Collect annotation data
    annotation_data = {
        'subtask_count': sub_count,
        'trajectory': traj.tolist() if isinstance(traj, np.ndarray) else traj if traj else None,
        'subtask_points': [],
        'reset_mode': use_reset_prompt,
        'reset_counter': reset_counter if use_reset_prompt else None,
        'reset_duration': reset_duration if use_reset_prompt else None
    }
    
    # 1. Draw trajectory line
    if traj is not None:
        for i in range(len(traj) - 1):
            p1 = tuple(map(int, traj[i]))
            p2 = tuple(map(int, traj[i + 1]))
            cv2.line(extended_img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
    
    # 2. Draw subtask points (circles and indices only in the original image area)
    if sub_point is not None and len(sub_point) > 0:
        for idx, subtask in enumerate(sub_point):
            # Extract coordinates and description
            if isinstance(subtask, dict) and 'position' in subtask:
                position = subtask['position']
                description = subtask.get('description', f'Subtask {idx+1}')
            else:
                position = subtask[:2] if len(subtask) >= 2 else subtask
                description = f'Subtask {idx+1}'
            
            center = tuple(map(int, position))
            
            # Save annotation data
            annotation_data['subtask_points'].append({
                'index': idx + 1,
                'position': position,
                'description': description
            })
            
            # Draw hollow circle and index number on the original image
            cv2.circle(extended_img, center, radius=8, 
                      color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            
            # Draw subtask index number
            text_number = str(idx + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(
                text_number, font, font_scale, thickness
            )
            text_pos = (center[0] - text_w // 2, center[1] + text_h // 2)
            
            cv2.putText(extended_img, text_number, text_pos,
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # 3. Draw subtask descriptions in the extended area
    if extra_height > 0 and sub_point:
        desc_y = img_h + 15  # Start 15 pixels below the original image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        for idx, subtask in enumerate(sub_point):
            if isinstance(subtask, dict) and 'description' in subtask:
                description = subtask['description']
            else:
                description = f'Subtask {idx+1}'
            
            # Draw index number and description
            text = f"{idx+1}. {description}"
            
            # Truncate overly long descriptions
            max_length = 80
            if len(text) > max_length:
                text = text[:max_length-3] + "..."
            
            cv2.putText(extended_img, text, (10, desc_y),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            desc_y += line_height
    
    # 4. Draw subtask count (top-left corner, matching reset indicator style)
    if sub_count is not None:
        text = f"Subtasks: {sub_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw semi-transparent background
        overlay = extended_img.copy()
        cv2.rectangle(overlay, (5, 5),
                    (10 + text_width, 10 + text_height + baseline),
                    (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, extended_img, 0.4, 0, extended_img)

        # Draw text
        cv2.putText(extended_img, text, (7, 7 + text_height),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # 5. Draw Reset mode status (top-right corner, matching subtask indicator style)
    if use_reset_prompt:
        reset_text = f"RESET ({reset_counter}/{reset_duration})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (tw, th), bl = cv2.getTextSize(reset_text, font, font_scale, thickness)
        
        # Draw semi-transparent background
        overlay = extended_img.copy()
        cv2.rectangle(overlay, (img_w - tw - 15, 5),
                    (img_w - 5, 10 + th + bl),
                    (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.6, extended_img, 0.4, 0, extended_img)
        
        cv2.putText(extended_img, reset_text, 
                   (img_w - tw - 13, 7 + th),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return extended_img, annotation_data


def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle numpy scalar types
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle strings (numpy and Python)
    if isinstance(obj, (np.str_, str)):
        return str(obj)
    
    # Recursively handle container types
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    
    # Return other types as-is (native Python types: int, float, bool, str, etc.)
    return obj


def save_rollout_video_dual(annotated_images, raw_images, annotation_data_list, 
                            episode_idx, success, task_description, checkpoint, task):
    """
    Save two versions of the video and annotation data, using the same path structure as save_rollout_video.

    Args:
        annotated_images: list of annotated frames
        raw_images: list of raw (unannotated) frames
        annotation_data_list: list of annotation data for each frame
        episode_idx: episode number
        success: whether the episode succeeded
        task_description: task description
        checkpoint: model checkpoint
        task: task name
    """
    # Use the same base path structure as save_rollout_video
    base_dir = f"./rollouts/{DATE}/{task}/{checkpoint}"
    
    # Create three subdirectories
    annotated_dir = os.path.join(base_dir, "annotated")
    raw_dir = os.path.join(base_dir, "raw")
    annotations_dir = os.path.join(base_dir, "annotations")
    
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Process task description, consistent with the original function
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    
    # Generate base filename
    base_filename = f"{DATE_TIME}--episode={episode_idx}--success={success}--task={processed_task_description}"
    
    # 1. Save annotated video
    if annotated_images:
        annotated_path = os.path.join(annotated_dir, f"{base_filename}.mp4")
        video_writer = imageio.get_writer(annotated_path, fps=30)
        for img in annotated_images:
            video_writer.append_data(img)
        video_writer.close()
        print(f"Saved annotated video: {annotated_path}")
    
    # 2. Save raw video
    if raw_images:
        raw_path = os.path.join(raw_dir, f"{base_filename}.mp4")
        video_writer = imageio.get_writer(raw_path, fps=30)
        for img in raw_images:
            video_writer.append_data(img)
        video_writer.close()
        print(f"Saved raw video: {raw_path}")
    
    # 3. Save annotation data (JSON format for easy viewing)
    if annotation_data_list:
        json_path = os.path.join(annotations_dir, f"{base_filename}.json")
        
        # Convert to serializable format
        serializable_data = {
            'episode': int(episode_idx),
            'success': bool(success),
            'task': task_description,
            'checkpoint': checkpoint,
            'total_frames': len(annotation_data_list),
            'frames': convert_to_serializable(annotation_data_list)
        }
        
        with open(json_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Saved annotation data: {json_path}")
        
        # 4. Also save in pickle format (preserves numpy arrays and other complex types without conversion)
        pickle_path = os.path.join(annotations_dir, f"{base_filename}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'episode': episode_idx,
                'success': success,
                'task': task_description,
                'checkpoint': checkpoint,
                'total_frames': len(annotation_data_list),
                'frames': annotation_data_list  # pickle can directly save numpy types
            }, f)
        print(f"Saved annotation pickle: {pickle_path}")


def step(img, wrist_img, language_instruction, model, processor, sampling_params, parser, unnorm_key):
    """
    Run the multimodal model to get a text, parse out the 8×7 action matrix,
    unnormalize, then temporally aggregate the first 6 DOFs (dims 0–5) while using
    the latest value for DOF 6. Return a single aggregated 7-D action vector and
    the annotated image.
    """
    image = Image.fromarray(img)
    wrist = Image.fromarray(wrist_img)
    image = center_crop_image(image)
    wrist = center_crop_image(wrist)
    imgs = [image, wrist]

    prompt = (
        f"The task is {language_instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {language_instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, how many subtasks are needed to complete this task, what is the semantic description of each subtask, and what are the goal positions for each subtask? "
        "Third, what is the trajectory of the end effector in the first image to reach the next subtask goal? "
        "Based on the depth map of the first image, the semantic description and goal position of each subtask, the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )

    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [dict(type="text", text=prompt)]
            }
        ], 
        tokenize=False, 
        add_generation_prompt=True,
    )

    inputs = [
        {
            "prompt": text,
            "multi_modal_data": {
                "image": [imgs]
            },
        },
    ]

    outputs = model.generate(inputs, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    depth = parser.parse_depth(generated_text)
    subtask_count = parser.parse_subtask_count(generated_text)
    # print(f"subtask_count: {subtask_count}")

    subtask_detail = parser.parse_subtask_ps(generated_text)
    # for subtask in subtask_detail:
        # print(f"subtask_description: {subtask['description']}")
        # print(f"subtask_position: {subtask['position']}")

    trace = parser.parse_trace_new(generated_text)
    # print(f"Trace: {trace}")

    action = parser.parse_action(generated_text, unnorm_key=unnorm_key)

    if (
        action is None
        or (isinstance(action, (list, tuple)) and len(action) == 0)
        or (isinstance(action, np.ndarray) and action.size == 0)
    ):
        raise ValueError("parse_action produced no action (None/empty).")
    
    annotated = np.array(img.copy())

    return action, annotated, trace, subtask_count, subtask_detail


def eval_libero(args, processor, model, sampling_params, parser, task_suite_name, checkpoint, seed, model_family, num_trials_per_task, num_steps_wait) -> None:

    set_seed_everywhere(seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Task suite: {task_suite_name}")

    # Get expected image dimensions
    resize_size = get_image_resize_size()

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for _ in tqdm.tqdm(range(1)):
        # Get task
        task_id = args.task_id
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
            last_gripper_state = -1
       
            # Create a new PromptManager for each episode
            prompt_manager = PromptManager()

            print(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            # Create three lists: annotated frames, raw frames, annotation data
            replay_images_annotated = []
            replay_images_raw = []
            annotation_data_list = []
            
            if task_suite_name == "libero_spatial":
                max_steps = 440
                unnorm_key = "libero_spatial_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_object":
                max_steps = 560
                unnorm_key = "libero_object_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_goal":
                max_steps = 600
                unnorm_key = "libero_goal_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_10":
                max_steps = 980
                unnorm_key = "libero_10_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_90":
                max_steps = 400
                print(f"Max steps: {max_steps}")

            print(f"Starting episode {task_episodes+1}...")
   
            timestep = 0
            outer_done = False
         
            while t < max_steps + num_steps_wait and not outer_done:
                # 1) Warm-up: ignore its 'done'
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_family))
                    t += 1
                    continue
                
                # Get the prompt that should currently be used
                current_task_description = prompt_manager.get_prompt(task_description)
                if current_task_description != task_description:
                    print(f"\n{'*'*60}")
                    print(f"[RESET] Using special prompt: '{current_task_description}'")
                    print(f"{'*'*60}\n")

                # 2) step action
                img = get_libero_image(obs, resize_size)
                wrist_img = get_libero_wrist_image(obs, resize_size)
                wait = False
                try:
                    action_matrix, annotated_image, traj, sub_count, sub_point = step(
                        img, wrist_img, current_task_description, model, processor, 
                        sampling_params, parser, unnorm_key
                    )
                    prompt_manager.update_state(sub_count)
                except Exception as e:
                    print(e)
                    action_matrix = np.zeros((1, 7), dtype=float)
                    action_matrix[:, -1] = last_gripper_state
                    annotated_image = img
                    traj = None
                    sub_count = None
                    sub_point = None
                    wait = True
                    print(f"error: {e}")

                action_num = 0
                # 3) Execute each of the N actions until done
                for single_action in action_matrix:
                    
                    if isinstance(single_action, str):
                        single_action = ast.literal_eval(single_action)
                    single_action = normalize_gripper_action(single_action, binarize=True)
                    single_action = invert_gripper_action(single_action)
                    obs, _, done, _ = env.step(single_action)
                    visualize = get_libero_image(obs, resize_size)

                    try:
                        # Get raw image (without annotations)
                        raw_frame = np.array(visualize.copy())
                        
                        # Create annotated image and annotation data
                        annotated_frame, frame_annotation = annotate_frame(
                            raw_frame,
                            traj,
                            sub_count,
                            sub_point,
                            prompt_manager.use_reset_prompt,
                            prompt_manager.reset_prompt_counter,
                            prompt_manager.reset_prompt_duration
                        )
                        
                        # Add timestamp and action info to annotation data
                        frame_annotation['timestep'] = t + action_num
                        frame_annotation['action'] = single_action.tolist() if isinstance(single_action, np.ndarray) else single_action
                        frame_annotation['done'] = done
                        
                        # Save to respective lists
                        replay_images_raw.append(raw_frame)
                        replay_images_annotated.append(annotated_frame)
                        annotation_data_list.append(frame_annotation)
                        
                    except Exception as e:
                        print(f"Frame annotation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Save raw frame even on failure
                        replay_images_raw.append(np.array(visualize))
                        replay_images_annotated.append(np.array(visualize))
                        annotation_data_list.append({
                            'timestep': t + action_num,
                            'error': str(e)
                        })

                    action_num += 1
   
                    if done:
                        outer_done = True
                        break
                
                # 4) Advance your loop counters
                timestep += 1
                if wait:
                    action_num = 1
                    
                t += action_num

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1

            # Save both versions of the video and annotation data
            save_rollout_video_dual(
                replay_images_annotated,
                replay_images_raw,
                annotation_data_list,
                total_episodes,
                success=done,
                task_description=task_description,
                checkpoint=checkpoint,
                task=task_suite_name
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     type=str, required=True)
    p.add_argument("--task_id",  type=int, required=False, default=None, 
                   help="Specific task ID (0-9). If not provided, will run all task IDs 0-9 for the specified task type.")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    task_suite_name = f"libero_{args.task}"
    ckpt = args.checkpoint
    seed = 7

    set_seed_everywhere(seed)

    processor = AutoProcessor.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        padding_side="left",
    )

    model = LLM(
        model=ckpt,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0
    )

    parser = SPRVLAParser.from_pretrained(ckpt)

    model_family = ckpt.replace("/", "-")
    num_trials_per_task = 50
    num_steps_wait = 10  
    
    if args.task_id is not None:
        print(f"Running single task ID: {args.task_id}")
        eval_libero(args, processor, model, sampling_params, parser, task_suite_name, ckpt, seed, model_family, num_trials_per_task, num_steps_wait)
    else:
        # Run all task IDs 0-9 for the specified task type
        print(f"Running all task IDs 0-9 for task type: {args.task}")
        for task_id in range(10):
            print(f"\n{'='*50}")
            print(f"Running task ID: {task_id}")
            print(f"{'='*50}")
            args.task_id = task_id
            eval_libero(args, processor, model, sampling_params, parser, task_suite_name, ckpt, seed, model_family, num_trials_per_task, num_steps_wait)


if __name__ == "__main__":
    main()
