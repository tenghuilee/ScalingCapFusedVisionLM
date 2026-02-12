
#%%

from tensorfusionvlm.eval_models.imgque import ImageMultiQueryCLIPLLMModel
import transformers
from tensorfusionvlm.vlmeval_extend import supported_VLM


#%%
# for key in supported_VLM.keys():
#     print(key)

#%%


# model = supported_VLM['Phi-3-Vision']() # type: transformers.PreTrainedModel
# type: transformers.PreTrainedModel
# model = supported_VLM['deepseek_vl_1.3b']()
# model = supported_VLM['ImageMultiQueryCLIPLLMModel_vicuna7b']()
# model = supported_VLM['LLaVA-v1.5-7B-hf']()
model = supported_VLM['ImageMultiQueryCLIPLLMModel_llama-2_256']()

# model = ImageMultiQueryCLIPLLMModel(
#     model_path="./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion-64_8_qwen2-1.5B_orth-finetune",
#     conv_template="qwen-7b-chat",
# )

#%%
ret = model.generate([
    './__hidden/02.jpg',
    'What can you see in the image?',
])
print(ret)

#%%

