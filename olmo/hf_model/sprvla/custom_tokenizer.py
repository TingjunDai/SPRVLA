# custom_tokenizer.py

from transformers import Qwen2Tokenizer, Qwen2TokenizerFast, GPT2TokenizerFast
from tokenizers.processors import Sequence, ByteLevel, TemplateProcessing

class Qwen2TokenizerWithBOS(Qwen2Tokenizer):
    def __init__(self, *args, add_bos_token=False, **kwargs):
        self.add_bos_token = add_bos_token
        super().__init__(*args, add_bos_token=add_bos_token, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id] if getattr(self, "add_bos_token", False) else []
        output = bos_token_ids + token_ids_0
        if token_ids_1 is None:
            return output
        return output + bos_token_ids + token_ids_1

class Qwen2TokenizerFastWithBOS(Qwen2TokenizerFast):
    def __init__(self, *args, add_bos_token=False, **kwargs):
        self.add_bos_token = add_bos_token
        super().__init__(*args, add_bos_token=add_bos_token, **kwargs)

        self._tokenizer.post_processor = Sequence(
            [
                ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
                TemplateProcessing(
                    single=f"{self.bos_token} $A:0",
                    pair=f"{self.bos_token} $A:0 {self.bos_token} $B:1",
                    special_tokens=[
                        (self.bos_token, self.bos_token_id),
                    ],
                ),
            ]
        )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id] if getattr(self, "add_bos_token", False) else []
        output = bos_token_ids + token_ids_0
        if token_ids_1 is None:
            return output
        return output + bos_token_ids + token_ids_1


class GPT2TokenizerFastWithBOS(GPT2TokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A:0",
            pair=f"{self.bos_token} $A:0 {self.bos_token} $B:1",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
            ],
        )