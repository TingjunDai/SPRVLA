import dataclasses
import math
from typing import Optional, Any, List, Union, Dict

import numpy as np
from olmo import tokenizer


ATTEND_ALL_SUBSEGMENT_ID = 10000


def build_subsegment_pos_ids(subsegments):
    position_ids = np.zeros_like(subsegments, dtype=np.int32)
    for subsegment_id in np.unique(subsegments):
        mask = subsegments == subsegment_id
        position_ids[mask] = np.arange(mask.sum(), dtype=np.int32)
    return position_ids


@dataclasses.dataclass
class InterleavedTextPreprocessor:
    """
    Build batches from text that is interleaved with tokens from other modalities
    """
    tokenizer: Any = None
    loss_token_weighting: Optional[str] = None
    max_text_tokens: Optional[int] = None
    max_sequence_length: Optional[int] = None
    last_message_loss_only: bool = False
    max_answer_len: int = None

    def tokenize_message(self, message_list: List[str], bos=True, add_last_eos=True):
        if bos:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            text_token_ids = [bos]
            loss_mask = [0.0]
        else:
            text_token_ids = []
            loss_mask = []
        for msg_ix, message in enumerate(message_list):
            message_ids = self.tokenizer.encode(message)
            is_model = msg_ix % 2 == 1
            if is_model and (add_last_eos or msg_ix != len(message_list) - 1):
                message_ids.append(self.tokenizer.eos_token_id)

            if is_model and self.max_answer_len:
                message_ids = message_ids[:self.max_answer_len]

            has_loss = is_model and (
                not self.last_message_loss_only or (msg_ix == (len(message_list) - 1)))
            loss_mask += [has_loss] * len(message_ids)
            text_token_ids += message_ids
        text_token_ids = np.array(text_token_ids)
        is_prompt = text_token_ids == self.tokenizer.image_prompt_token_id
        return text_token_ids, np.array(loss_mask, dtype=np.float32)

    def tokenize_message_list(
        self,
        message_list: Union[List[str], List[List[str]]],
        n_mm_tokens: int,
        num_images: int = 1,
    ):
        """Handle multi-annotation data where we have many annotations for one multi-modal input"""
        assert len(message_list) > 0, "Given empty messages"
        # Multi-annotation data where we have many annotations for one multi-modal input
        before_ids = []
        after_ids = []
        before_losses = []
        after_losses = []
        before_subsegments = []
        after_subsegments = []
        n_tokens = 0
        for message_set_ix, message_tuple in enumerate(message_list):
            add_bos = message_set_ix == 0
            text_ids, text_loss = self.tokenize_message(
                message_tuple, bos=add_bos, add_last_eos=False)
            is_prompt = text_ids == self.tokenizer.image_prompt_token_id
            n_prompts = is_prompt.sum()
            if n_prompts == 1:
                image_idx = np.argmax(is_prompt)
                s, e = image_idx, image_idx+1
            elif n_prompts == 0 and add_bos:
                s, e = 1, 1
            elif n_prompts == 0:
                s, e = 0, 0
            else:
                raise NotImplementedError("Multi-message with multi images")

            if text_loss[e] != 0:
                raise ValueError("Must have a non-loss token after MM data")
            if self.max_sequence_length and message_set_ix != 0:
                if (n_mm_tokens + n_tokens + np.argmax(text_loss != 0)) >= self.max_sequence_length:
                    # This example would get no loss tokens anyway
                    break
            n_tokens += len(text_ids)
            before_ids.append(text_ids[:s])
            after_ids.append(text_ids[e:])
            before_losses.append(text_loss[:s])
            after_losses.append(text_loss[e:])
            before_subsegments.append(np.full(s, ATTEND_ALL_SUBSEGMENT_ID, dtype=np.int32))
            after_subsegments.append(np.full(len(text_ids[e:]), message_set_ix, dtype=np.int32))
            if self.max_text_tokens and (n_tokens >= self.max_text_tokens):
                break

        text_token_ids = np.concatenate([
            np.concatenate(before_ids),
            [self.tokenizer.image_prompt_token_id] * num_images,
            np.concatenate(after_ids),
            [self.tokenizer.eos_token_id],
        ])
        text_subsegments = np.concatenate([
            np.concatenate(before_subsegments),
            [ATTEND_ALL_SUBSEGMENT_ID] * num_images,
            np.concatenate(after_subsegments),
            after_subsegments[-1][-1:]  # for EOS
        ])
        text_loss_masks = np.concatenate([
            np.concatenate(before_losses),
            [0] * num_images,
            np.concatenate(after_losses),
            [0]  # for EOS
        ])
        if self.loss_token_weighting == "root_subsegments":
            text_loss_masks *= math.sqrt(1/len(before_ids))
        elif self.loss_token_weighting is not None:
            raise NotImplementedError(self.loss_token_weighting)
        return text_token_ids, text_loss_masks, text_subsegments

    def tokenize_and_interleave(
        self,
        message_list: Union[List[str], List[List[str]]],
        multi_model_tokens: List[np.ndarray],
        multi_model_pos_ids: Optional[List[np.ndarray]]=None,
        weight: Optional[float]=None
    ) -> Dict[str, np.ndarray]:
        """
        Build a batch by interleaving the text tokens from tokenizing `message_list` and the
        multi-modal tokens from `multi_model_tokens`

        `tokenizer.IMAGE_PROMPT` is used to show where the MM tokens should be inserted, if it is
        not present the MM tokens are inserted right after BOS

        If `message_list` is a list of lists, the batch is assumed to contain multiply-annotated
        MM data. The batch will include tokens from all messages but the MM tokens only once, and
        `subsegment_id` will indicate how to cross-attend between the tokens. Attending between
        tokens before the MM tokens will be allowed, but attending between tokens after the MM
        tokens will not.
        """
        if isinstance(message_list[0], list) and len(message_list) == 1:
            message_list = message_list[0]
        if isinstance(message_list[0], str):
            text_token_ids, text_loss_masks = self.tokenize_message(message_list)
            text_subsegments = None
            for_inference = len(message_list) % 2 == 1
        else:
            text_token_ids, text_loss_masks, text_subsegments = self.tokenize_message_list(
                message_list, sum(len(x) for x in multi_model_tokens), 0 if multi_model_tokens is None else len(multi_model_tokens))
            for_inference = False

        mm_idx = np.argwhere(text_token_ids == self.tokenizer.image_prompt_token_id)
        if len(mm_idx) == 0:
            if multi_model_tokens is not None:
                # Assume mm data should go right after BOS
                mm_idx = [1] * len(multi_model_tokens)
        else:
            mm_idx = mm_idx[:, 0]

        mm_tokens = []
        mm_loss_masks = []
        mm_subsegments = None if text_subsegments is None else []
        mm_position_ids = []
        on = 0
        on_pos = 0
        for i, token_ix in enumerate(mm_idx):
            mm_tokens.append(text_token_ids[on:token_ix])
            mm_loss_masks.append(text_loss_masks[on:token_ix])
            if text_subsegments is not None:
                mm_subsegments.append(text_subsegments[on:token_ix])
            if multi_model_pos_ids is not None:
                assert len(multi_model_tokens[-1]) == len(multi_model_pos_ids[-1])
                mm_position_ids.append(np.arange(on_pos, on_pos+len(mm_tokens[-1])))
                on_pos += len(mm_tokens[-1])

            vision_tokens = multi_model_tokens[i]
            mm_tokens.append(vision_tokens)
            mm_loss_masks.append(np.zeros_like(vision_tokens))
            if text_subsegments is not None:
                mm_subsegments.append(np.full([len(vision_tokens)], text_subsegments[token_ix]))
            if multi_model_pos_ids is not None:
                mm_position_ids.append(multi_model_pos_ids[i] + on_pos)
                on_pos += multi_model_pos_ids[i].max() + 1
            if text_token_ids[token_ix] == self.tokenizer.image_prompt_token_id:
                on = token_ix + 1  # Skip over the image prompt token
            else:
                on = token_ix

        mm_tokens.append(text_token_ids[on:])
        mm_loss_masks.append(text_loss_masks[on:])
        if text_subsegments is not None:
            mm_subsegments.append(text_subsegments[on:])
            n_pre_mm_tokens = sum(len(x) for x in mm_tokens[:-1])
            if not mm_position_ids:
                mm_position_ids = [np.arange(n_pre_mm_tokens)]
                on_pos = n_pre_mm_tokens
            mm_position_ids.append(on_pos + build_subsegment_pos_ids(text_subsegments[on:]))
        elif mm_position_ids:
            mm_position_ids.append(np.arange(on_pos, on_pos+len(mm_tokens[-1])))
        else:
            mm_position_ids = [np.arange(0, sum(len(x) for x in mm_tokens))]

        mm_tokens = np.concatenate(mm_tokens)
        mm_loss_masks = np.concatenate(mm_loss_masks)
        mm_position_ids = np.concatenate(mm_position_ids)
        if mm_subsegments is not None:
            mm_subsegments = np.concatenate(mm_subsegments)

        target_tokens = mm_tokens

        if (len(message_list) % 2 == 0) or mm_subsegments is not None:
            target_tokens = mm_tokens[1:]
            input_tokens = mm_tokens[:-1]
            mm_loss_masks = mm_loss_masks[1:]
            if mm_subsegments is not None:
                # The targets for subsegments in the middle need to end with EOS,
                # currently they end with whatever starts the next segment
                mm_subsegments = mm_subsegments[:-1]
                target_tokens = np.copy(target_tokens)
                for i in range(len(message_list)):
                    subsegment_mask = mm_subsegments == i
                    if not np.any(mm_subsegments == i):
                        assert (self.max_text_tokens or self.max_sequence_length) and i != 0
                        # Message skipped due hitting `self.max_text_tokens`
                        break
                    segment_end = np.argwhere(mm_subsegments == i)[-1, 0]
                    target_tokens[segment_end] = self.tokenizer.eos_token_id
                    assert mm_subsegments[segment_end-1] == i
                    assert mm_loss_masks[segment_end-1] != 0
                    mm_loss_masks[segment_end] = mm_loss_masks[segment_end-1]

            if mm_loss_masks[-1] == 0:
                raise RuntimeError("EOS should not be masked")
            mm_position_ids = mm_position_ids[:-1]
        else:
            # Presumably doing inference, but give a dummy target anyway for consistency
            assert mm_tokens[-1] != self.tokenizer.eos_token_id
            input_tokens = mm_tokens
            target_tokens = np.pad(mm_tokens[1:], [0, 1], constant_values=0)
        if weight is not None:
            mm_loss_masks *= weight

        out = {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "loss_masks": mm_loss_masks,
            "position_ids": mm_position_ids,
        }
        if mm_subsegments is not None:
            out["subsegment_ids"] = mm_subsegments

        # Some sanity checks
        if not all(len(v) == len(input_tokens) for v in out.values()):
            raise RuntimeError("Length mismatch")
        special_tokens = np.array([
            self.tokenizer.image_end_token_id,
            self.tokenizer.image_start_token_id,
            self.tokenizer.image_col_token_id,
            self.tokenizer.image_patch_token_id,
            self.tokenizer.image_low_res_token_id,
        ])[None, :]
        if np.any(target_tokens[mm_loss_masks != 0][:, None] == special_tokens):
            raise RuntimeError("A special token had a loss")
        return out
