# %%
import os

import numpy as np
import torch
import torch.nn as nn
import transformers

import tensorfusionvlm.data_utils as data_utils
import tensorfusionvlm.model.modeling_imghd_multi_que_llm as modeling_base
from tensorfusionvlm.auto_models import (AutoModelForImageMultiQuery,
                                         load_tokenizers_processor)


# %%

checkpoint_path = "./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion_linear-"

checkpoint_path += "256_8_hd_llama-2_eye-ft-finetune"
# checkpoint_path += "16_8_hd_llama-2_eye-ft-finetune"

# checkpoint_path = "./checkpoints/debug_init_new_llm_backbone"

# conv_template = "vicuna_v1.1"
# conv_template = "qwen-7b-chat"
conv_template = "llama-2-vision"
# conv_template = "llama-3-vision"

ckpt_name = ""
# ckpt_name = None
# ckpt_name = "checkpoint-1660"

if ckpt_name is None:
    last_checkpoint_path = transformers.trainer_utils.get_last_checkpoint(checkpoint_path)
else:
    last_checkpoint_path = os.path.join(checkpoint_path, ckpt_name)
print(last_checkpoint_path)

main_config: modeling_base.ImageMultiQueryCLIPConfig = transformers.AutoConfig.from_pretrained(
    last_checkpoint_path)

llm_tokenizer, clip_tokenizer, image_processor = load_tokenizers_processor(
    checkpoint_path)


# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = AutoModelForImageMultiQuery.from_pretrained(
    last_checkpoint_path,
    torch_dtype=dtype, 
    attn_implementation="flash_attention_2",
)

#%%
print(model.imgque_backbone.clip_model.vision_model.token_LF)

#%%

model = model.eval().to(device) # type: modeling_base.ImageMultiQueryCLIPLLMModel

#%%

if conv_template == "llama-3-vision":
    eos_token = llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
else:
    eos_token = llm_tokenizer.eos_token_id

parser_util = data_utils.UtilChatAndQueryParser(
    tokenizer=llm_tokenizer,
    query_tokenizer=clip_tokenizer,
    image_processor=image_processor,
    length_que_end=main_config.length_que_end,
    length_que_end_short=main_config.length_que_end_short,
    append_special_padding=main_config.append_special_padding,
    _qed_replaced_query="",
)

generation_config = transformers.GenerationConfig(
    # temperature=0.99,
    # top_p=0.99,
    num_beams=1,
    # do_sample=True,
    do_sample=False,
    max_new_tokens=256,
    eos_token_id=eos_token,
    pad_token_id=llm_tokenizer.pad_token_id,
)

streamer = transformers.TextStreamer(
    llm_tokenizer,
    skip_prompt=True,
)

#%%

# set forward hook

# def debug_forward_hook(tag: str):
#     def __inner(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
#         print(f"{tag}, diff io {torch.mean(torch.square(input[0] - output[0]))}")
#     return __inner

# for i in range(0, 20):
#     sub_module = model.imgque_backbone.imgque_model.layers[i] # type: nn.Module
#     # remove forward hook if exists
#     if hasattr(sub_module, "_forward_hooks"):
#         sub_module._forward_hooks.clear()
#     sub_module.register_forward_hook(debug_forward_hook(f"layer_{i}"))

#%%

record = data_utils.ChatRecordSimple.from_dict([
    # {"role": "template", "content": conv_template},
    # {"role": "system", "content": "You are a helpful assistant. Please answer the question based on the image."},
    # {"role": "system", "content": "You can read all text from the image."},
    # {"role": "system", "content": """You can read all text from the image. 
    #     If the image is about table, please extract all text from the table.
    #     If the text is about math, writen in Latex Format, start with '\(' and end with '\)'.
    #     If the text is about programming, writen the code with complete syntax.
    #     If the text is unclear to be determined, please write <unknwon>.
    #     Remove unnecessary empty white spaces."""},
    # {"role": "image", "content": "./__hidden/x07.png"},
    {"role": "image", "content": "./__hidden/04.jpg"},
    {"role": "user", "content": "What time is this?"},
    {"role": "assistant", "content": "This is a photo of a cat."},
    {"role": "user", "content": "Describe the given image with detail."},
    # {"role": "user", "content": "What is this and which the city this object possibly located? Answer with the reason why you give such answer."},
    # {"role": "user", "content": "Please write a python program Using OpenCV to read the image."},
    # {"role": "user", "content": "Tell me what is this and where this place located and how can we go to this place from guangzhou? Step-by-step guide with points."},
    # {"role": "user", "content": "Teach me how to cook this. step-by-step guide with points."},
    # {"role": "user", "content": "Think deep and more. What is the meaning of this image?"},
    # {"role": "user", "content": "Tell me the long history behind this object."},
    # {"role": "user", "content": "What is it? Answer the reason why you give such answer."},
    # {"role": "user", "content": "Write an interesting story based on the given image. Please write it down."},
    # {"role": "user", "content": "Read all text from the given image. Do not miss any readable characters."},
    # {"role": "user", "content": "Convert the given document into better format."},
    # {"role": "user", "content": "Convert the given table into latex format."},
    # {"role": "user", "content": "Create a html web page based on the given image."},
    {"role": "assistant", "content": ""},
])

predict = model.generate(
    generation_config=generation_config,
    streamer=streamer,
    **parser_util.parse_record(
        record, 
        conv_template=conv_template,
        enable_que_end=True,
        dtype=torch.bfloat16,
        device=device,
    )
)

#%%