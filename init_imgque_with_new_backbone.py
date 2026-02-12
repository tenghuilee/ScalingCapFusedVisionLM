r"""

In this script, the weights of the LLM backbone will be replaced by new LLM backbone.

CAUTION: Please finetune the module with the new backbone before using it.
CAUTION: Please finetune the module with the new backbone before using it.
CAUTION: Please finetune the module with the new backbone before using it.

Don't forget to update chat_template.

The checkpoint should contain all avaliable modules, including,
1. The CLIP Vision Encoder
2. The IMGQUE Fuse module
3. The Projection Layer
4. The LLM backbone

The new checkpoint should contain all avaliable modules, including,
1. The CLIP Vision Encoder (DO NOT CHANGE)
2. The IMGQUE Fuse module (DO NOT CHANGE)
3. The Projection Layer (DO NOT CHANGE)
4. The LLM backbone (CHANGE)
"""

import argparse
import os

import torch
import transformers

import tensorfusionvlm.model.modeling_imghd_multi_que_llm as modeling_base
from tensorfusionvlm.auto_models import (AutoModelForImageMultiQuery,
                                         load_tokenizers_processor)

# disable gradient globally
torch.set_grad_enabled(False)

_args = argparse.ArgumentParser()
_args.add_argument("--ckpt_path", type=str, required=True)
_args.add_argument("--new_llm_backbone", type=str, required=True)
_args.add_argument("--save_path", type=str, required=True)
_args.add_argument("--further_check", action="store_true", help="If set, we will load the checkpoint from currently saved path and check if the new LLM backbone is loaded correctly.")
args = _args.parse_args()

# 1. Load the checkpoint
print("Loading the tokenizer, processors from:", args.ckpt_path)
llm_tokenizer, clip_tokenizer, image_processor = load_tokenizers_processor(
    args.ckpt_path,
)


print("Loading the module from:", args.ckpt_path)
model = AutoModelForImageMultiQuery.from_pretrained(
    args.ckpt_path,
)  # type: modeling_base.ImageMultiQueryCLIPLLMModel

old_llm_backbone = model.llm_backbone

print("Updating the LLM backbone to:", args.new_llm_backbone)
new_llm_backbone = transformers.AutoModelForCausalLM.from_pretrained(
    args.new_llm_backbone,
    torch_dtype=model.llm_backbone.dtype,
    trust_remote_code=True,
)

assert old_llm_backbone.config.hidden_size == new_llm_backbone.config.hidden_size

del model.llm_backbone

model.llm_backbone = new_llm_backbone

model.config.llm_model_name_or_path = args.new_llm_backbone
model.config.llm_config = new_llm_backbone.config

# 2. Save the checkpoint
print("Saving the checkpoint to:", args.save_path)
model.save_pretrained(args.save_path)
llm_tokenizer.save_pretrained(args.save_path)
image_processor.save_pretrained(args.save_path)
clip_tokenizer.save_pretrained(os.path.join(args.save_path, "clip"))

print("Done!")

if args.further_check:
    new_model = AutoModelForImageMultiQuery.from_pretrained(args.save_path)

    old_state_dict = model.state_dict()
    new_state_dict = new_model.state_dict()

    for k in old_state_dict.keys():
        assert k in new_state_dict.keys(), f"{k} not in new state dict"
        assert torch.allclose(old_state_dict[k], new_state_dict[k]), f"{k} not close"
    
    print("All keys in old state dict are in new state dict and they are close!")
    print("Further check passed!")

