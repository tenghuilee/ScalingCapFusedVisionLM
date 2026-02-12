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
from tensorfusionvlm.data_utils import CocoCaptionMixed, CocoCaptionMixedDataset

import tensorfusionvlm.model.modeling_query_adapt_clip as query_adapt_clip

ROOT_PATH = "./checkpoints/modeling_query_adapt_cl_baai_svit_llava_warmup_ft_clip_text_adamw_hf"

# %%
base_model_name = "openai/clip-vit-large-patch14-336"

tokenizer = transformers.CLIPTokenizer.from_pretrained(base_model_name)
processor = transformers.CLIPProcessor.from_pretrained(base_model_name)

# %%
# I don't know why the `.from_pretrained()` method always give us the wrong checkpoint.
# model = query_adapt_clip.ModelingQueryAdaptCLIPModel.from_pretrained(ROOT_PATH, use_safetensors=True)

# # use this below instead
config = query_adapt_clip.ModelingQueryAdaptCLIPModelConfig.from_pretrained(ROOT_PATH)
config.base_name_or_path = None
model = query_adapt_clip.QueryAdaptCLIPModel(config)

state_dict = {}
with safe_open(os.path.join(ROOT_PATH, "model.safetensors"), framework="pt", device=0) as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        state_dict[key] = tensor
model.load_state_dict(state_dict)

model = model.eval()

# %%

image = [
    Image.open(f"./__hidden/{d:02d}.jpg")
    for d in range(5)
]

# Two different querys on the same image

queries = [
    "What's the color of the car?",
    "How many people in the image?",
    "What's the color of the tree?",
    "How many dogs in the image?",
    "List all objects.",
]

answers = [
    "red", "green", "black", "white",
    "one", "two", "three", "zero", "car", "human", "cat", "dog",
]

# %%
# text to token_index
que_text_input = processor(text=queries, padding=True, truncation=True, return_tensors="pt")
que_input_ids, que_attention_mask = que_text_input.input_ids, que_text_input.attention_mask

ans_text_input = processor(text=answers, padding=True, truncation=True, return_tensors="pt")
ans_input_ids, ans_attention_mask = ans_text_input.input_ids, ans_text_input.attention_mask

visual_input = processor(images=image, padding=True, truncation=True, return_tensors="pt")
pixel_values = visual_input.pixel_values

# %%

outputs = model.forward(
    # pixel_values=torch.cat([pixel_values for _ in queries], dim=0),
    pixel_values=pixel_values,
    que_ids=torch.stack([que_input_ids[4] for _ in queries], dim=0),
    # que_attention_mask=que_attention_mask,
    que_attention_mask=torch.stack([que_attention_mask[4] for _ in queries], dim=0),
    ans_ids=ans_input_ids,
    ans_attention_mask=ans_attention_mask,
    return_loss=False,
)

#%%
with np.printoptions(precision=3, suppress=True):
    print(torch.softmax(outputs.logits_per_image, dim=-1).data.cpu().numpy())

# %%

# test low rankness

# %%
imgque_outputs = model.get_imgque(
    input_ids=que_input_ids[1:2],
    attention_mask=que_attention_mask[1:2],
    pixel_values=pixel_values,
)

# %%

# (1, 586, 1024)
last_hidden_state = imgque_outputs.last_hidden_state # type: torch.Tensor
# %%

# %%
U, S, V = torch.svd(last_hidden_state[0])

# %%
plt.plot(S.data.cpu().numpy())
plt.show()
# %%

SS = S.data.cpu().numpy()

# accumulate
SS_accum = np.cumsum(SS) / np.sum(SS)
plt.plot(SS_accum)
plt.show()
# %%

# compute reconstruction error

reconstruction_error = []

norm_base = torch.norm(last_hidden_state[0]).data.cpu().numpy()

for rank in range(10, S.shape[0], 10):
    U_ = U[:, :rank]
    S_ = S[:rank]
    V_ = V[:, :rank]

    last_hidden_state_ = torch.matmul(U_, torch.diag(S_))
    last_hidden_state_ = torch.matmul(last_hidden_state_, V_.t())

    reconstruction_error.append(
        torch.norm(last_hidden_state_ - last_hidden_state[0]).data.cpu().numpy() / norm_base
    )

plt.plot(range(10, S.shape[0], 10), reconstruction_error)
plt.show()

# %%

plt.hist(last_hidden_state.data.cpu().numpy().flatten(), bins=500)
plt.show()
# %%
