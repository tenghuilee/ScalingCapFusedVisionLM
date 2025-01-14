# %%
import argparse
import json
import logging
import multiprocessing 
import os

import torch
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel

# chart|diagram|figure|illustration|image|photo|picture|table

# python scripts/prepare_mPLUG_doc_nouga.py - -input_file "./datasets/prepare_mPLUG_Doc_struct_aware_parse_doc.jsonl" - -image_root "./datasets/soft_link_image_collection" - -rank - 1

_args = argparse.ArgumentParser()
_args.add_argument("--input_file", type=str, required=True)
_args.add_argument("--image_root", type=str, required=True)
_args.add_argument("--rank", type=int, default=0, help="rank of the process; -1 to use all available GPUs")
_args.add_argument("--world_size", type=int, default=None)
_args.add_argument("--batch_size", type=int, default=200)
_args.add_argument("--model_name", type=str, default="facebook/nougat-base")
args = _args.parse_args()

logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
)

class SimpleRender:
    def __init__(self, input_file: str, image_root: str, rank: int, world_size: int, batch_size: int, model_name: str):
        assert input_file.endswith(".jsonl")
        self.output_path = input_file.replace(
            ".jsonl",
            f"_{rank:02d}_of_{world_size:02d}_nougat.jsonl",
        ) # type: str
        self.input_file = input_file
        self.fout_writer = open(self.output_path, "a")
        self.image_root = image_root
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        self.processor = NougatProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.model.to(device=self.device, dtype=self.dtype)

        self.passed_images = set() # type: set[str]
        self._reset()
    
    def __del__(self):
        self.fout_writer.close()

    def _reset(self):
        with open(self.output_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.passed_images.add(item["image"])

    def process(self):
        image_names = []
        write_count = 0
        with open(self.input_file, "r") as f:
            for idx, line in enumerate(f):
                if idx % self.world_size != self.rank:
                    continue

                item = json.loads(line)
                image = item["image"]

                if image in self.passed_images:
                    continue

                image_names.append(image)

                if len(image_names) >= self.batch_size:
                    try:
                        write_count += self.process_batch(image_names)
                        logging.info("@[%02d/%02d] write %d images", self.rank, self.world_size, write_count)
                    except Exception as e:
                        logging.error("Error: %s", e)
                    
                    image_names = []
                self.passed_images.add(image)

        if len(image_names) > 0:
            write_count += self.process_batch(image_names)

        logging.info("@[%02d/%02d] write %d images", self.rank, self.world_size, write_count)
        return write_count

    @torch.no_grad()
    def process_batch(self, image_names: list[str]):
        pil_images = [
            Image.open(os.path.join(self.image_root, _img)).convert("RGB")
            for _img in image_names
        ]

        batch_image = self.processor(images=pil_images, return_tensors="pt", padding=True).pixel_values
        batch_image = batch_image.to(device=self.device, dtype=self.dtype)
        outputs = self.model.generate(
            batch_image,
            min_length=1,
            max_new_tokens=2048,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            do_sample=False,
        )
        sequences = self.processor.batch_decode(outputs, skip_special_tokens=True)
        sequences = self.processor.post_process_generation(sequences, fix_markdown=True)
        for img_name, sequence in zip(image_names, sequences):
            self.fout_writer.write(json.dumps({"image": img_name, "src": sequence}) + "\n")

        self.fout_writer.flush()

        return len(image_names)

def do_render(local_rank: int, world_size: int, args):
    render = SimpleRender(
        input_file=args.input_file,
        image_root=args.image_root,
        rank=local_rank,
        world_size=world_size,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    render.process()

if __name__ == "__main__":

    world_size = args.world_size if args.world_size is not None else torch.cuda.device_count()

    print("Runing")
    print("world_size:", world_size)
    print("rank:", args.rank)
    print("batch_size:", args.batch_size)
    print("model_name:", args.model_name)
    print("input_file:", args.input_file)
    print("image_root:", args.image_root)
    print("")

    if args.rank == -1:
        processes = [] # for i in range(5): # Create 5 processes p = multiprocessing.Process(target=worker, args=(i,)) processes.append(p) p.start() for p in processes: p.join()
        for local_rank in range(world_size):
            p = multiprocessing.Process(target=do_render, args=(local_rank, world_size, args))
            processes.append(p)
            p.start()

        # wait for all processes to finish
        for p in processes:
            p.join()
    else:
        if args.rank >= world_size:
            raise ValueError(f"Invalid rank {args.rank} for world size {world_size}")

        do_render(args.rank, world_size, args)
    
    print("Done")
