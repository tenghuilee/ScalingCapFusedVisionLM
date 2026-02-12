#%%
import random
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import transformers
from PIL import Image
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPModel,
                          CLIPProcessor, CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModel)

from tensorfusionvlm.utils import ImageHelperWithProcessor

VISION_BACKBONE="openai/clip-vit-large-patch14-336"

#%%

model = CLIPModel.from_pretrained(VISION_BACKBONE)
processor = CLIPProcessor.from_pretrained(VISION_BACKBONE)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("./__hidden/01.jpg")

#%%
# inputs = processor(text=["one cat", "two cats", "dog"],
#                    images=image, return_tensors="pt", padding=True)

visual_inputs = processor(images=image, return_tensors="pt", padding=True)
image_features = model.vision_model(**visual_inputs)[0]
print("shape of image_features", image_features.shape)
visual_projed = model.visual_projection(image_features)
# visual_projed = model.get_image_features(**inputs)
visual_projed = visual_projed / torch.norm(visual_projed, dim=-1, keepdim=True)
print(visual_projed.shape)
# image_features = model.get_image_features(**inputs)
# print(image_features.shape)

#%%
txt_inputs = processor(text=["How many animals in the image? The image contains a dog. The image contains two cats. This image is about a sofa. Zebra, rocks"], return_tensors="pt", padding=True)
input_ids = txt_inputs.input_ids[0]
txt_cells = []
for _j in range(input_ids.size(0)):
    txt_cells.append(processor.tokenizer.decode(input_ids[_j]))

txt_features = model.text_model(**txt_inputs)[0]
print("shape of text features", txt_features.shape)
# model.get_text_features(**txt_inputs)
txt_projed = model.text_projection(txt_features) # type: torch.Tensor
txt_projed = txt_projed / torch.norm(txt_projed, dim=-1, keepdim=True)
# print(txt_projed)

#%%
similarity = torch.matmul(
    txt_projed.view(-1, txt_projed.size(-1)),
    visual_projed[0].T,
)
similarity = similarity.reshape(txt_projed.size(0), txt_projed.size(1), -1)
print(similarity.shape)

# similarity = similarity.softmax(dim=1)

U, S, V = torch.svd(similarity[0])
plt.plot(S.data.cpu().numpy())
plt.show()

#%%
plt.figure(figsize=(8,4))
plt.plot(similarity[:,:,0].squeeze().data.cpu().numpy())
plt.xticks(np.arange(len(txt_cells))+1, txt_cells, rotation=90)
plt.show()

# %%
plt.imshow(image)
plt.show()

plt.figure(figsize=(8,4))
plt.boxplot(similarity[0].data.cpu().numpy().T)
plt.xticks(np.arange(len(txt_cells))+1, txt_cells, rotation=90)
# plt.legend(["%d" % (_j) for _j in range(similarity.size(1))])
# plt.imshow(similarity[i].data.cpu().numpy(), cmap="hot", aspect="auto")
plt.show()

# #%%
# print(visual_inputs.pixel_values.shape)
# plt.imshow(visual_inputs.pixel_values[0].data.cpu().numpy().transpose(1,2,0))
# plt.show()

# %%

# %%
