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

import tensorfusionvlm.model.modeling_query_adapt_clip_ae_exam as query_adapt_clip

ROOT_PATH = "./checkpoints/exam_modeling_query_adapt_cl_baai_svit_llava_warmup_ft_clip_text/checkpoint-700"

# %%
base_model_name = "openai/clip-vit-large-patch14-336"

tokenizer = transformers.CLIPTokenizer.from_pretrained(base_model_name)
processor = transformers.CLIPProcessor.from_pretrained(base_model_name)

# %%
# I don't know why the `.from_pretrained()` method always give us the wrong checkpoint.
# model = query_adapt_clip.ModelingQueryAdaptCLIPModel.from_pretrained(ROOT_PATH, use_safetensors=True)

# # use this below instead
config = query_adapt_clip.ModelingQueryAdaptCLIPModelConfig.from_pretrained(
    ROOT_PATH)
config.base_name_or_path = None
model = query_adapt_clip.ModelingQueryAdaptCLIPExam(config)

state_dict = {}
with safe_open(os.path.join(ROOT_PATH, "model.safetensors"), framework="pt", device=0) as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        state_dict[key] = tensor
model.load_state_dict(state_dict)

model = model.eval()

#%%
image = Image.open("./__hidden/08.jpg")
query = "Describe the image in detail"

prepare_inputs = processor(text=query, images=image, padding=True, return_tensors="pt")
que_input_ids = prepare_inputs.input_ids
que_attention_mask = prepare_inputs.attention_mask
pixel_values = prepare_inputs.pixel_values
# %%

with torch.no_grad():
    outputs = model.forward(
        pixel_values=pixel_values,
        que_ids=que_input_ids,
        que_attention_mask=que_attention_mask,
        return_loss=True,
    )

# %% 
print(f"loss: {outputs.loss}, loss_recover: {outputs.loss_recover}, loss_lowrank: {outputs.loss_lowrank}")

#%%
with np.printoptions(precision=3, suppress=True):
    print(outputs.imgque_input.data.cpu().numpy())
    print(outputs.imgque_output.data.cpu().numpy())

    print(outputs.imgque_middle.data.cpu().numpy())
    print(outputs.imgque_lowrank.data.cpu().numpy())

# %%
U, S, V = torch.svd(outputs.imgque_middle[0])


# %%
SS = S.data.cpu().numpy()

plt.plot(SS)
plt.show()

# accumulation
SS_acc = np.cumsum(SS) / np.sum(SS)
plt.plot(SS_acc)
plt.show()

# %%

reconstruction_error = []

norm_base = torch.norm(outputs.imgque_middle[0]).data.cpu().numpy()

for rank in range(10, S.shape[0], 10):
    U_ = U[:, :rank]
    S_ = S[:rank]
    V_ = V[:, :rank]

    last_hidden_state_ = torch.matmul(U_, torch.diag(S_))
    last_hidden_state_ = torch.matmul(last_hidden_state_, V_.t())

    reconstruction_error.append(
        torch.norm(last_hidden_state_ -
                   outputs.imgque_middle[0]).data.cpu().numpy() / norm_base
    )

plt.plot(range(10, S.shape[0], 10), reconstruction_error)
plt.show()

# %%
