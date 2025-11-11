import copy

import torch
import vlmeval
from PIL import Image
from transformers import (
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import isimg, listinstr


class InstructBlip_HF(vlmeval.BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = "Salesforce/instructblip-vicuna-7b",
        revision="52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92",
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        **kwargs,
    ):
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            # attn_implementation=attn_implementation,
        )
        self.processor = InstructBlipProcessor.from_pretrained(
            model_path, revision=revision
        )

        self.kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_length=256,
            min_length=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
        )

    def adjust_kwargs(self, dataset):
        kwargs = copy.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ["MCQ", "Y/N"]:
            kwargs["max_new_tokens"] = 32
        elif DATASET_TYPE(dataset) == "Caption" and "COCO" in dataset:
            kwargs["max_new_tokens"] = 32
        elif DATASET_TYPE(dataset) == "VQA":
            if listinstr(["OCRVQA", "ChartQA", "DocVQA"], dataset):
                kwargs["max_new_tokens"] = 100
            elif listinstr(["TextVQA"], dataset):
                kwargs["max_new_tokens"] = 10
        elif listinstr(["seedbench"], dataset):
            kwargs["max_new_tokens"] = 3
        return kwargs

    def generate_inner(self, message: list[dict], dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs

        image_name = None
        question = None

        image_name = None
        question = None
        for line in message:
            _type = line.get("type")
            if _type == "image":
                image_name = line.get("value")
            elif _type == "text":
                question = line.get("value")

        assert (
            image_name is not None and question is not None
        ), "Image and question must be provided"

        inputs = self.processor(
            images=Image.open(image_name).convert("RGB"),
            text=question,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            **kwargs,
        )
        generated_text = self.processor.decode(
            outputs[0].cpu(),
            skip_special_tokens=True,
        ).strip()

        return generated_text
