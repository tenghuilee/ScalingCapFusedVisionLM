from vlmeval.config import supported_VLM
from tensorfusionvlm.eval_models.imgque import ImageMultiQueryCLIPLLMModel
from tensorfusionvlm.eval_models.llavahf import LLaVA_HF
from tensorfusionvlm.eval_models.instructblip import InstructBlip_HF
import os

import itertools
from functools import partial

from tensorfusionvlm.vlmeval_extend_models import VisionZipLLaVA, VisPrunerLLaVA

# supported_VLM["ImageMultiQueryCLIPLLMModel_vicuna7b"] = partial(
#     ImageMultiQueryCLIPLLMModel,
#     model_path="./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion-64_8_vicuna_7b_orth-finetune",
#     conv_template="vicuna_v1.1",
# )

supported_VLM["LLaVA-v1.5-7B-hf"] = partial(
    LLaVA_HF,
    model_path = "llava-hf/llava-1.5-7b-hf",
    revision = 'a272c74',
)

supported_VLM["InstructBLIP-7B-hf"] = partial(
    InstructBlip_HF,
    model_path = "Salesforce/instructblip-vicuna-7b",
)


supported_VLM["ImageMultiQueryCLIPLLMModel_imgque_hd_256_v5_llama-2"] = partial(
    ImageMultiQueryCLIPLLMModel,
    model_path="./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion_linear-256_8_hd_llama-2_eye-v5-finetune",
    conv_template="llama-2-vision",
)

# llama-2
for flen in [1, 8, 16, 32, 64, 128, 256, 384, 512, 768]:
    supported_VLM[f"ImageMultiQueryCLIPLLMModel_llama-2_{flen}"] = partial(
        ImageMultiQueryCLIPLLMModel,
        model_path=f"./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion_linear-{flen}_8_hd_llama-2_eye-ft-finetune",
        conv_template="llama-2-vision",
    )

# llama-2 drop query without finetune
for flen in [1, 8, 16, 32, 64, 128, 256, 384, 512, 768]:
    supported_VLM[f"ImageMultiQueryCLIPLLMModel_DropQueryNoFT-llama-2_{flen}"] = partial(
        ImageMultiQueryCLIPLLMModel,
        model_path=f"./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion_linear-{flen}_8_hd_llama-2_eye-ft-finetune",
        conv_template="llama-2-vision",
        _qed_replaced_query="", # empty means drop the query
    )

# llama-2 drop query with finetune
for flen in [1, 8, 16, 32, 64, 128, 256, 384, 512, 768]:
    supported_VLM[f"ImageMultiQueryCLIPLLMModel_DropQueryFT-llama-2_{flen}"] = partial(
        ImageMultiQueryCLIPLLMModel,
        model_path=f"./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion_linear-{flen}_8_hd_llama-2_drop-query-ft-finetune",
        conv_template="llama-2-vision",
        _qed_replaced_query="", # empty means drop the query
    )

# vicuna 
for flen in [8, 16, 32, 64, 128, 256, 512]:
    supported_VLM[f"ImageMultiQueryCLIPLLMModel_vicuna_{flen}"] = partial(
        ImageMultiQueryCLIPLLMModel,
        model_path=f"./checkpoints/tensorfusionvlm-v_img_multi_que_clip_fusion_linear-{flen}_8_hd_vicuna-ft-finetune",
        conv_template="vicuna_v1.1",
    )


for flen in [16, 32, 64, 128, 256, 384, 512, 768]:
    supported_VLM[f"VisPrunerLLaVA_{flen}_0.5"] = partial(
        VisPrunerLLaVA,
        model_path='liuhaotian/llava-v1.5-7b',
        visual_token_num=flen,
        important_ratio=0.5,
        max_new_tokens=128,
    )

    supported_VLM[f"VisionZipLLaVA_{flen-10}_10"] = partial(
        VisionZipLLaVA,
        model_path='liuhaotian/llava-v1.5-7b',
        dominant=flen-10,
        contextual=10,
        max_new_tokens=128,
    )
