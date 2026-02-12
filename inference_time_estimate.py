r"""
Compare the inference time of different models on a subset of MSCOCO images.



"""

import argparse
import json
import os
import random
import re
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import transformers
from tqdm.auto import tqdm

from tensorfusionvlm.imgque_model_clock import ImageMultiQueryModelClock


class ImagePicker:
    def __init__(self, image_path: str, num_images: int, image_rand_seed: int):
        self.image_path = image_path
        self.num_images = num_images
        self.image_rand_seed = image_rand_seed

        # list all avaliable images
        self.image_list = []
        for fname in os.listdir(self.image_path):
            if not fname.endswith('.jpg'):
                continue
            self.image_list.append(fname)
        # sort
        self.image_list.sort()
        # shuffle
        rpn = random.Random(self.image_rand_seed)
        rpn.shuffle(self.image_list)
        self.image_list = self.image_list[:self.num_images]
    
    def __len__(self):
        return len(self.image_list)

    def __iter__(self):
        for fname in self.image_list:
            image_path = os.path.join(self.image_path, fname)
            yield image_path

def generate_response_with_timing(
    model_name: str,
    image_picker: ImagePicker,
    output_file_jsonl: str,
    conv_template="llama-2-vision",
):

    model_clock = ImageMultiQueryModelClock(
        model_path=model_name,
        conv_template=conv_template,
    )

    # prepare one sample to warm up
    for path_to_img in image_picker:
        out = model_clock.generate_response_with_timing(
            messages=[
                { "role": "image", "content": path_to_img },
                { "role": "user", "content": "Describe the image in detail." },
            ],
            max_new_tokens=1,
        )
        break

    with open(output_file_jsonl, 'w') as fout:
        for path_to_img in tqdm(image_picker):
            out = model_clock.generate_response_with_timing(
                messages=[
                    { "role": "image", "content": path_to_img },
                    { "role": "user", "content": "Describe the image in detail." },
                ],
                max_new_tokens=256,
            )
            fout.write(json.dumps(out) + '\n')


def find_all_models(
    model_dir: str,
    model_name_re: str,
):
    ptn_model = re.compile(model_name_re)

    model_list = []
    for fname in os.listdir(model_dir):
        if ptn_model.match(fname):
            model_list.append(os.path.join(model_dir, fname))
    return model_list


def eval_inference_time(
    model_name: str,
    image_picker: ImagePicker,
    out_path_root: str,
    conv_template="llama-2-vision",
):
    print(f"Processing model: {model_name}")

    out_file_jsonl = os.path.join(
        out_path_root,
        f"inference_time_{os.path.basename(model_name)}.jsonl",
    )
    os.makedirs(out_path_root, exist_ok=True)

    generate_response_with_timing(
        model_name=model_name,
        image_picker=image_picker,
        output_file_jsonl=out_file_jsonl,
        conv_template=conv_template,
    )

    print(f"Saved results to: {out_file_jsonl}")

def eval_flops(
    model_name: str,
    image_picker: ImagePicker,
    out_path_root: str,
    conv_template="llama-2-vision",
):

    print(f"Processing model: {model_name}")

    out_file_jsonl = os.path.join(
        out_path_root,
        f"flops_{os.path.basename(model_name)}.jsonl",
    )
    os.makedirs(out_path_root, exist_ok=True)

    model_clock = ImageMultiQueryModelClock(
        model_path=model_name,
        conv_template=conv_template,
    )

    # prepare one sample to warm up
    for path_to_img in image_picker:
        out = model_clock.estimate_flop_for_prefill(
            messages=[
                { "role": "image", "content": path_to_img },
                { "role": "user", "content": "Describe the image in detail." },
            ],
        )
        print(out)
        break

    with open(out_file_jsonl, 'w') as fout:
        for path_to_img in tqdm(image_picker):
            out = model_clock.estimate_flop_for_prefill(
                messages=[
                    { "role": "image", "content": path_to_img },
                    { "role": "user", "content": "Describe the image in detail." },
                ],
            )
            fout.write(json.dumps(out) + '\n')

    print(f"Saved results to: {out_file_jsonl}")


if __name__ == "__main__":
    _args = argparse.ArgumentParser()
    _args.add_argument("--model_dir", type=str, default="./checkpoints")
    _args.add_argument("--model_name_re", type=str, default=r"tensorfusionvlm-v_img_multi_que_clip_fusion_linear-(\d+)_(\d+)_hd_llama-2_eye-ft-finetune")
    _args.add_argument("--image_path", type=str, default="./datasets/mscoco/test2017")
    _args.add_argument("--num_images", type=int, default=500)
    _args.add_argument("--image_rand_seed", type=int, default=122)
    _args.add_argument("--out_path_root", type=str, default="./eval_time_estimate")
    args = _args.parse_args()

    image_picker = ImagePicker(
        image_path=args.image_path,
        num_images=args.num_images,
        image_rand_seed=args.image_rand_seed,
    )

    for model_name in find_all_models(
        model_dir=args.model_dir,
        model_name_re=args.model_name_re,
    ):
        # eval_inference_time(model_name, image_picker, args.out_path_root)
        eval_flops(model_name, image_picker, args.out_path_root)
