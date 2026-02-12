
#%%

import os
from collections import OrderedDict

import transformers
from safetensors import safe_open
import argparse

import tensorfusionvlm.model.modeling_query_adapt_clip as modeling_base

#%%
_arg = argparse.ArgumentParser()
_arg.add_argument("--root_path", type=str, default="./checkpoints/modeling_imghd_multi_que_cl_128_8_layer24/")
_arg.add_argument("--ckpt_name", type=str, default=None)
_arg.add_argument("--overwrite", action="store_true")
args = _arg.parse_args()

assert os.path.exists(args.root_path)

if args.ckpt_name is None:
    pending_files = []
    for f in os.listdir(args.root_path):
        if f.startswith("checkpoint-"):
            pending_files.append(f)
    if len(pending_files) == 0:
        print("no checkpoint found")
        exit(0)
    
    # find the latest checkpoint
    pending_files.sort(key=lambda x: int(x.removeprefix("checkpoint-")))
    args.ckpt_name = pending_files[-1]

print("using checkpoint", args.ckpt_name)

ckpt_path = os.path.join(args.root_path, args.ckpt_name)
assert os.path.exists(ckpt_path)

if os.path.exists(os.path.join(args.root_path, "config.json")):
    if not args.overwrite:
        print("config.json exists, exit")
        exit(0)

model_config = transformers.AutoConfig.from_pretrained(
    ckpt_path) # type: modeling_base.ImageHDMultiQueryCLIPConfig

model_config.text_config.max_position_embeddings = 1024

model = modeling_base.ImageHDMultiQueryCLIPModel(model_config)

state_dict = OrderedDict()
prefix = "clip_model.vision_model.vision_model."
prefix_new = "clip_model.vision_model."
with safe_open(os.path.join(ckpt_path, "model.safetensors"), "pt") as f:
    for k in f.keys():
        assert isinstance(k, str)
        # GlobalDebugUtil.print(f.get_tensor(k), name=k)
        if k.startswith("clip_model.vision_model.vision_model.post_layernorm"):
            # skip special layers
            print("skipping", k)
            continue

        params = f.get_tensor(k)
        if k.startswith(prefix):
            k = prefix_new + k.removeprefix(prefix)
        state_dict[k] = params

model.load_state_dict(state_dict)

model.save_pretrained(args.root_path)

model = modeling_base.AutoModelOnVLMForImageMultiQuery.from_pretrained(args.root_path)

print("done")
