
#%%
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%

from tensorfusionvlm.vlmeval_extend_models import *

# %%

model = VisionZipLLaVA()

ret = model.generate([
    './__hidden/02.jpg',
    'What can you see in the image?',
])
print(ret)


# %%

del model
torch.cuda.empty_cache()

#%%

model = VisPrunerLLaVA(
    visual_token_num=64,
    important_ratio=0.5,
    max_new_tokens=128,
)


#%%
ret = model.generate([
    './__hidden/02.jpg',
    'What can you see in the image?',
])
print(ret)

#%%
