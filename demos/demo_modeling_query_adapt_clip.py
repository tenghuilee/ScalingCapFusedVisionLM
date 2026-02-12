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

print(model.logit_scale.exp())
# %%
# check clip model is the same as original model
clip_model = transformers.CLIPModel.from_pretrained(base_model_name)
clip_model = clip_model.eval()

# %%

# Initialize the name_param_diff dictionary with OrderedDict for each encoder layer
name_param_diff = {
    "encoder.layers": [OrderedDict() for _ in range(len(clip_model.text_model.encoder.layers))],
}

# Iterate over the named parameters of the new model
for name, param_new in model.clip_model.text_model.named_parameters():
    param_old = clip_model.text_model.get_parameter(name)
    diff = torch.norm(param_new.data - param_old.data).item()

    # Check if the parameter name starts with "encoder.layers."
    if name.startswith("encoder.layers."):
        # Extract the layer index and sub-name
        _, _, layer_idx, *sub_name_parts = name.split(".")
        layer_idx = int(layer_idx)
        sub_name = ".".join(sub_name_parts)
        # Store the difference in the appropriate OrderedDict
        name_param_diff["encoder.layers"][layer_idx][sub_name] = diff
    else:
        name_param_diff[name] = diff

name_param_diff["text_projection"] = torch.norm(
    model.clip_model.text_projection.weight - clip_model.text_projection.weight
)

# Remove the encoder layers from the name_param_diff dictionary
encoder_layers = name_param_diff.pop("encoder.layers")

# Print the differences for non-encoder layer parameters
for key, value in name_param_diff.items():
    print(f"{key}, {value:.5f}")

print(" ,", ",".join(encoder_layers[0].keys()))
# Print the differences for each encoder layer, sorted by sub-name (OrderedDict maintains the order)
for layer_idx, layer_dict in enumerate(encoder_layers):
    sorted_vals = [f"{v:.5f}" for v in layer_dict.values()]
    print(f"encoder.layers.{layer_idx}, {','.join(sorted_vals)}")


#%%

coco = CocoCaptionMixed(
    root="./datasets/coco/val2017/",
    annFile="./datasets/coco/annotations_trainval2017/captions_val2017.json",
    mix_items=10,
)


#%%
image, captions, caption_masks = coco[6]

# plt.imshow(image)
# plt.axis("off")
# plt.show()

# for tgx, msk in zip(captions, caption_masks):
#     print(msk, tgx)

visual_inputs = processor(images=image, return_tensors="pt", padding=True)

query_inputs = processor(text="Describe the image in detail.", return_tensors="pt")
que_ids, que_attention_mask = query_inputs["input_ids"], query_inputs["attention_mask"]

ans_inputs = processor(text=captions, padding=True, return_tensors="pt")
ans_ids, ans_attention_mask = ans_inputs["input_ids"], ans_inputs["attention_mask"]

# %%

output = model.forward(
    pixel_values=visual_inputs["pixel_values"],
    que_ids=que_ids,
    que_attention_mask=que_attention_mask,
    ans_ids=ans_ids,
    ans_attention_mask=ans_attention_mask,
    return_loss=False,
)

# %%
text_embeds = output.text_embeds
image_embeds = output.image_embeds
imgque_embeds = output.imgque_embeds

# print(text_embeds.shape, image_embeds.shape, query_align_embeds.shape)

similarity = torch.matmul(image_embeds, text_embeds.t()) * 100
similarity = torch.softmax(similarity, dim=1)
with np.printoptions(precision=4, suppress=True):
    print(similarity.data.cpu().numpy().reshape(-1, 5))


similarity = torch.matmul(imgque_embeds, text_embeds.t()) * 100
similarity = torch.softmax(similarity, dim=1)
# numpy round to 3 decimals
with np.printoptions(precision=4, suppress=True):
    print(similarity.data.cpu().numpy().reshape(-1, 5))

similarity = torch.matmul(image_embeds, imgque_embeds.t())
print(similarity)
# %%
