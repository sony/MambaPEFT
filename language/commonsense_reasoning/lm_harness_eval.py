import torch

import transformers
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "mamba_peft/src/"))
from mamba_peft.src.peft import PeftModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import MambaForCausalLM

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("MambaPEFT")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, 
                 pretrained="state-spaces/mamba-130m-hf",
                 peft_weights=None, 
                 max_length=2048, 
                 batch_size=None, 
                 device="cuda",
                 dtype=torch.float32,
                 trust_remote_code=False):
        self.peft_weights = peft_weights
        super().__init__(pretrained=pretrained,
                       tokenizer="EleutherAI/gpt-neox-20b",
                       max_length=max_length,
                       dtype=dtype,
                       trust_remote_code=trust_remote_code)
        
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size


    def _create_model(
        self,
        pretrained: str,
        dtype = "float32",
        # no `parallelize=True` options
        # no PEFT and quantization options
        # Mamba does not support arbitrary HF from_pretrained() args
        **kwargs,
    ) -> None:
        
        model = MambaForCausalLM.from_pretrained(
            pretrained,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self._model = PeftModel.from_pretrained(
            model,
            self.peft_weights,
            torch_dtype=torch.float32,
        )
        self._model.config.use_cache = False # Not fully implemented yet
        self._model.float()
        self._model.to(self._device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()