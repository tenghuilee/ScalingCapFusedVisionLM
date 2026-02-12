#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
import transformers.models.phi3
import transformers.models.llama.tokenization_llama_fast
import tensorfusionvlm.data_utils as data_utils
from tensorfusionvlm.data_utils import EnumTokenType
from tokenizers import processors
import torchvision.transforms

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tensorfusionvlm.model.image_processing_phi3_v import Phi3VImageProcessor

from tensorfusionvlm.utils import ImageHelperWithPhi3Processor

torch.cuda.set_device("cuda:4")

#%%

# llm_name_or_path = "./llama_hf_mirror/llama_hf/llama-3-8b-instruct"
# llm_name_or_path = "microsoft/Phi-3-mini-4k-instruct"
# llm_name_or_path = "Qwen/Qwen2-7B-Instruct"
# llm_name_or_path = "microsoft/Phi-3-vision-128k-instruct"
llm_name_or_path = "./checkpoints/modeling_imghd_multi_que_cl_128_8_layer12"

# tokenizer = AutoTokenizer.from_pretrained()
# tokenizer = AutoTokenizer.from_pretrained()
# tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)

# main_config = AutoConfig.from_pretrained(llm_name_or_path)
# print(main_config)

# model = AutoModelForCausalLM.from_pretrained(llm_name_or_path, trust_remote_code=True)

img_helper = ImageHelperWithPhi3Processor(llm_name_or_path)

#%%

out, sinfo = img_helper.load_image(
    "./__hidden/Scarlet Lounge (Night).jpeg",
    return_shape_info=True,
    img_processor=torchvision.transforms.ToTensor(),
)
        

pixel_values = out["pixel_values"]

np_img = pixel_values[0].cpu().numpy().transpose(0, 2, 3, 1)


# put the first frame to the left
# put the rest 16 frames to 4x4 grid
fig = plt.figure(layout="constrained", figsize=(16, 8)) #, facecolor="black")

if np_img.shape[0] == 17:
    gs = GridSpec(4, 8, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    for i in range(4):
        for j in range(4,8):
            fig.add_subplot(gs[i, j])

    for i, ax in enumerate(fig.axes):
        ax.imshow(np_img[i])
        ax.axis("off")

    plt.show()
elif np_img.shape[0] == 10:
    gs = GridSpec(3, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    for i in range(3):
        for j in range(3,6):
            fig.add_subplot(gs[i, j])

    for i, ax in enumerate(fig.axes):
        ax.imshow(np_img[i])
        ax.axis("off")

    plt.show()
else:
    raise ValueError("Unsupported number of frames")

#%%


# messages = [
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
# ]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you solve the equation 2x + 3 = 7?"},
    {"role": "assistant", "content": "The solution to the equation 2x + 3 = 7 is x = 1."},
    {"role": "user", "content": "Please proof E=MC^2"},
]

# full_msg = tokenizer.apply_chat_template(
#     conversation=messages[1:],
#     # return_tensors="pt",
#     # return_dict=True,
#     tokenize=False,
#     tokenizer_kwargs={
#         "return_token_type_ids": True,
#     },
# )

# print(full_msg)

full_msg = tokenizer.apply_chat_template(
    conversation=messages,
    return_tensors="pt",
    return_dict=True,
    # tokenize=False,
    # tokenizer_kwargs={
    #     "return_token_type_ids": True,
    # },
)

print(full_msg)


#%%
# print(tokenizer.special_tokens_map)

# #%%

# print(tokenizer("<|assistant|>", add_special_tokens=False))

# assist_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

# print(full_msg)
# print(assist_id)

# pos = (full_msg == assist_id).int().argmax(-1)

# print(pos)

# full_msg[0, 0:pos+1] = -100

# print(full_msg)

model = AutoModelForCausalLM.from_pretrained(
    llm_name_or_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# messages = [
#     {"role": "system", "content": "Is the given sentence a query sentence? Answer with '<p>1</p>' if it is a query sentence, '<p>0</p>' if it is not."},
#     {"role": "user", "content": f"<p>The image is about</p>" },
# ]

#%%
# messages = [
#     # {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Can you solve the equation 2x + 3 = 7?"},
#     {"role": "assistant", "content": "The solution to the equation 2x + 3 = 7 is x = 1."},
# ]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

#%%