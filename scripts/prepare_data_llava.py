import json
import os
from typing import Iterator
import re

# import PIL.Image
import imagesize

import data_basic
from tqdm import tqdm

class ProcessMain:
    SRC_JSON = "./datasets/llava_v1_5_mix665k.json"
    DST_JSONL = "./datasets/prepare_llava_v1_5_mix665k_image_only.jsonl"
    IMAGE_ROOT = "./datasets/soft_link_image_collection"

    MAX_ROUND = 48 # 24 * 2

    REMOVE_IMAGE_TOKEN = re.compile(r"\n*<image>\n*")

    image_size_cached = dict()

    def __init__(self):
        pass

    def get_image_size(self, image: str) -> tuple[int, int]:
        if image in self.image_size_cached:
            return self.image_size_cached[image]
        else:
            width, height = imagesize.get(os.path.join(self.IMAGE_ROOT, image))
            self.image_size_cached[image] = (width, height)
            return width, height
    
    def process_conv(self, image: str, conversations: dict, bbox: data_basic.BBoxFormaterWithSquarePadded) -> Iterator[data_basic.ChatRecord]:
        rec = data_basic.ChatRecord(image=image)

        if image is not None:
            image_size = self.get_image_size(image)
        else:
            image_size = (0, 0)

        conv_count = 0
        for i, conv in enumerate(conversations):
            _from = conv["from"]
            value = conv["value"]

            # do filter 
            if i == 0:
                if "Reference OCR token" in value:
                    return None
            # end filter

            value = re.sub(self.REMOVE_IMAGE_TOKEN, "", value).strip()
            if image is not None:
                value = bbox.apply(value, image_size)

            if _from == "human":
                assert i % 2 == 0
                rec.append_user(value)
            elif _from == "gpt":
                assert i % 2 == 1
                rec.append_assistant(value)
            else:
                raise ValueError(f"Unknown from: {_from}")
            conv_count += 1

            if conv_count >= self.MAX_ROUND:
                break
        
        return rec

    def process(self) -> None:
        with open(self.SRC_JSON, "r") as f:
            data = json.load(f)
        
        writer = data_basic.OutputWriter(self.DST_JSONL)
        bbox = data_basic.BBoxFormaterWithSquarePadded()

        for item in tqdm(data):

            # item = item # type: dict
            image = item.get("image", None)

            if image is None:
                # ignore no image instances
                continue

            try:
                rec = self.process_conv(image, item["conversations"], bbox)
                if rec is not None:
                    writer.write(rec)
                else:
                    print("Ignore", rec)
            except ValueError as e:
                print(e)
            
        writer.dump_summary()

if __name__ == "__main__":
    pmain = ProcessMain()
    pmain.process()

    # for k, v in pmain.cached_image_size.items():
    #     print(k, v)
