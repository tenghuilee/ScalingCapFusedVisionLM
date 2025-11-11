
import copy
import warnings
from typing import Optional

import torch
import transformers
import vlmeval
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import isimg, listinstr

from PIL import Image

class LLaVA_HF(vlmeval.BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = 'a272c74',
        device_map='cuda',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        max_new_tokens=512,
        **kwargs,
    ):
        """
        Wrapper model for vlmeval
        Args:
        - model_path (str): path to the model
        - revision (str): revision of the model;
            for more detail, see https://huggingface.co/llava-hf/llava-1.5-7b-hf/discussions/41
        """
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        self.max_new_tokens = max_new_tokens

        self.torch_dtype = torch_dtype

        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision,
        )

        self.kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
        )

    def adjust_kwargs(self, dataset):
        kwargs = copy.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 10
        elif listinstr(["seedbench"], dataset):
            kwargs['max_new_tokens'] = 3
        return kwargs

    def generate_inner(self, message: list[dict], dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs

        image_name = None
        question = None
        for line in message:
            _type = line.get("type")
            if _type == "image":
                image_name = line.get("value")
            elif _type == "text":
                question = line.get("value")

        assert image_name is not None and question is not None, "Image and question must be provided"

        text_prompt = self.processor.apply_chat_template([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }], add_generation_prompt=True)

        inputs = self.processor(
            images=Image.open(image_name),
            text=text_prompt,
            padding=True,
            return_tensors="pt",
        ).to(
            self.model.device,
            self.torch_dtype,
        )

        input_token_len = inputs.input_ids.shape[-1]

        # Generate
        generate_ids = self.model.generate(**inputs, **kwargs)

        answer = self.processor.decode(
            generate_ids[0][input_token_len:].cpu(),
            skip_special_tokens=True,
        ).strip()

        return answer
