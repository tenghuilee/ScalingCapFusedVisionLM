# %%
import os
from collections import OrderedDict
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from PIL import Image
from safetensors import safe_open
# Eval on coco test dataset
import tensorfusionvlm.data_utils as data_utils
from tensorfusionvlm.constants import EnumTokenType, DEFAULT_ANSWER_PLACEHOLDER

import tensorfusionvlm.model.modeling_query_adapt_clip as query_adapt_clip

#%%

ckpt_path = "./checkpoints/modeling_img_multi_que_cl_64_8_layer24"

tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_path)
image_processor = transformers.CLIPImageProcessor.from_pretrained(ckpt_path)

model = query_adapt_clip.AutoModelForVLM.from_pretrained(ckpt_path)

#%%
parser = data_utils.ChatQEDQueryParser(
    tokenizer=tokenizer,
    length_que_end=model.config.length_que_end,
    length_que_end_short=model.config.length_que_end_short,
    append_special_padding=True,
)

parser.append(data_utils.EnumTokenType.QUE, "Please describe this image.")
parser.append(data_utils.EnumTokenType.QUE, "What is in the left?")


outputs = parser.parse(True)

# %%

image_helper = data_utils.ImageHelperWithProcessor(
    processor=image_processor,
)

#%%

image_path = "./__hidden/01.jpg"

pixel_values = image_helper.load_image(image_path)

#%%

model_outputs = model.forward(
    pixel_values=pixel_values.unsqueeze(0),
    input_ids=outputs.input_ids.unsqueeze(0),
    input_tps=outputs.input_tps.unsqueeze(0),
    attention_mask=outputs.attention_mask.unsqueeze(0),
)

# %%

mask = data_utils.EnumTokenType.is_the_type(model_outputs.input_tps, EnumTokenType.QED_EOS)

# %%
picked_last_hidden_state = model_outputs.last_hidden_state[mask]
# %%

cctcc = torch.matmul(picked_last_hidden_state, picked_last_hidden_state.transpose(0, 1))

# cctcc = torch.softmax(cctcc, dim=-1)

cc = cctcc.data.cpu().float().numpy()

plt.imshow(cc)
plt.axis('off')
plt.colorbar()
plt.show()
# %%
