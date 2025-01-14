
# %%
from transformers import NougatProcessor, VisionEncoderDecoderModel
import json
import os
import torch

from PIL import Image
from pix2text import TextFormulaOCR
import progressbar
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/238
# export LD_LIBRARY_PATH=$HOME/Downloads/cudnn-linux-x86_64-9.0.0.312_cuda12-archive/lib
# export CPATH=$HOME/Downloads/cudnn-linux-x86_64-9.0.0.312_cuda12-archive/include

# %%


def is_valid_bbox(res_bbox: list[dict]) -> bool:
    for bbox in res_bbox:
        if bbox["type"] == "isolated":
            if bbox["score"] > 0.8:
                return True
    return False


class ImageDataset(Dataset):
    def __init__(self, image_root: str, src_path: str):
        super().__init__()
        self.image_root = image_root
        self.src_path = src_path
        self.image_paths = []
        with open(src_path, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                img = data['image']
                self.image_paths.append(img)

        self.transform = transforms.Compose([
            transforms.Resize(768),
            transforms.CenterCrop(768),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_src = self.image_paths[idx]
        img_path = os.path.join(self.image_root, img_src)
        img = Image.open(img_path).convert("RGB")
        # 768x768
        img = self.transform(img)
        return img, img_src


def collect_func(batch: list[tuple[Image.Image, str]]):
    img, batch_src = list(zip(*batch))
    return img, batch_src


# %%
p2t = TextFormulaOCR.from_config(device="cuda")

# %%

src_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc.jsonl"
image_root = "../datasets/soft_link_image_collection"
out_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math.txt"

# passed_lines = 0

# dataloader = DataLoader(ImageDataset(image_root, src_path), batch_size=32,
#                         shuffle=False, num_workers=32, collate_fn=collect_func)

# with open(out_path, 'w') as fout:
#     progress_bar = tqdm(dataloader, desc="Processing")
#     for img, img_src in progress_bar:
#         res = p2t.mfd(img, conf=0.5, stream=True, verbose=False)
#         for _img_src, res in zip(img_src, res):
#             if is_valid_bbox(res):
#                 fout.write(_img_src + "\n")
#                 passed_lines += 1
#                 progress_bar.set_postfix({"passed": passed_lines})

# %%


class ContextDataset(Dataset):
    def __init__(self, src_path: str, picked_path: str, image_root: str):
        super().__init__()
        self.src_path = src_path
        self.picked_path = picked_path
        self.image_root = image_root
        self.data_img = []
        self.data_ctx = []
        self.picked_img = set()
        with open(picked_path, 'r') as fin:
            for line in fin:
                self.picked_img.add(line.strip())
        with open(src_path, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                img = data['image']
                if img not in self.picked_img:
                    continue
                self.data_img.append(img)
                self.data_ctx.append(data['src'])

    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, idx):
        img, ctx = self.data_img[idx], self.data_ctx[idx]
        img_pil = Image.open(os.path.join(self.image_root, img)).convert("RGB")
        return img_pil, img, ctx

    @staticmethod
    def collate_fn(batch):
        pil, img, ctx = list(zip(*batch))
        return pil, img, ctx

# %%

model_name = "facebook/nougat-base"
device = "cuda"
dtype = torch.bfloat16
processor = NougatProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model = model.to(device=device, dtype=dtype)


# %%
dataloader = DataLoader(
    ContextDataset(src_path, out_path, image_root),
    batch_size=200,
    shuffle=False,
    num_workers=32,
    collate_fn=ContextDataset.collate_fn,
)

out_nouga_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math.jsonl"

with torch.no_grad(), open(out_nouga_path, 'w') as fout:
    for img, img_names, ctx in tqdm(dataloader, desc="Writing"):
        batch_image = processor(images=img, return_tensors="pt", padding=True).pixel_values
        batch_image = batch_image.to(device=device, dtype=dtype)
        
        outputs = model.generate(
            batch_image,
            min_length=1,
            max_new_tokens=2048,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            do_sample=False,
        )

        sequences = processor.batch_decode(outputs, skip_special_tokens=True)
        sequences = processor.post_process_generation(sequences, fix_markdown=False)
        for img_name, sequence in zip(img_names, sequences):
            fout.write(json.dumps({"image": img_name, "src": sequence}) + "\n")

print("Done")
