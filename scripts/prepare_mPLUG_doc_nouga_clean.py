#%%
import torch
import re
from sentence_transformers import SentenceTransformer
import json
from tqdm.auto import tqdm

import multiprocessing
import nltk

image_root = "../datasets/soft_link_image_collection"

# %%

class SimpleValidString:
    def __init__(
        self,
        device=None,
        token_len_greater_than=10,
        token_non_repeat_grater_than=0.12,
        token_non_repeat_window_size=256,
        seq_similirity_grater_than=0.5,
        len_diff_less_than=0.1,
    ):
        # Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        self.model = SentenceTransformer("all-mpnet-base-v2", device=device)

        self.token_len_greater_than=token_len_greater_than
        self.token_non_repeat_grater_than=token_non_repeat_grater_than
        self.token_non_repeat_window_size=token_non_repeat_window_size
        self.seq_similirity_grater_than=seq_similirity_grater_than
        self.len_diff_less_than=len_diff_less_than
    
    def token_length(self, text: list[str]) -> int:
        return len(text)

    def token_non_repeat_ratio(self, text: list[str], window_size=256) -> float:
        if len(text) > window_size:
            text = text[-window_size:]
        sub_vocalb = set(text)
        return len(sub_vocalb) / len(text)
    
    def token_length_diff_ratio(self, text1: list[str], text2: list[str]) -> float:
        """
        ratio: is the maximum allowed difference in length between two texts
        smaller is better
        """
        len1 = len(text1)
        len2 = len(text2)
        return abs(len1 - len2) / max(len1, len2) 

    @torch.no_grad()
    def similarity(self, dest: str, src: str) -> float:
        embedding = self.model.encode([
            dest,
            src,
        ])
        return self.model.similarity(embedding[0], embedding[1]).item()
    
    def is_valid(self, dest: str, src: str) -> bool:
        return self.is_valid_with_reason(dest, src)[0]
    
    def is_valid_with_reason(
        self,
        dest: str,
        src: str,
    ) -> bool:
        dst_token = nltk.word_tokenize(dest)
        src_token = nltk.word_tokenize(src)

        if len(dst_token) < self.token_len_greater_than:
            reason = f"dst_token too short: {len(dst_token)} < {self.token_len_greater_than}"
            return False, reason
        if len(src_token) < self.token_len_greater_than:
            reason = f"src_token too short: {len(src_token)} < {self.token_len_greater_than}"
            return False, reason
        ratio = self.token_non_repeat_ratio(
            src_token,
            window_size=self.token_non_repeat_window_size,
        )
        if ratio < self.token_non_repeat_grater_than:
            reason = f"src_token non repeat ratio too low: {ratio} < {self.token_non_repeat_grater_than}"
            return False, reason
        ratio = self.token_length_diff_ratio(
            dst_token, src_token)
        if ratio > self.len_diff_less_than:
            reason = f"dst_token and src_token length diff ratio too high: {ratio} > {self.len_diff_less_than}"
            return False, reason
        ratio = self.similarity(dest, src)

        if ratio < self.seq_similirity_grater_than:
            reason = f"dst_token and src_token similarity too low: {ratio} < {self.seq_similirity_grater_than}"
            return False, reason
        return True, "passed"

class FileMananger:
    def __init__(self, original_file: str, sub_file: str, output_file: str, droped_file: str):
        self.original_file = original_file
        self.output_file = output_file
        self.sub_file = sub_file
        self.droped_file = droped_file

        self.ostream = open(self.output_file, "a")
        self.droped_ostream = open(self.droped_file, "a")
        self.passed_cache = set() # type: set[str]

        self.passed_count = 0
        self.droped_count = 0

        self.original_cache = dict() # type: dict[str, str]
        self.cache_original_file()
        self._reset()

    
    def __del__(self):
        self.ostream.close()
        self.droped_ostream.close()
    
    def cache_original_file(self):
        print("cache original file")
        with open(self.original_file, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                image = data["image"]
                src = data["src"]
                self.original_cache[image] = src
    
    def _reset(self):
        self.passed_cache = set() # type: set[str]
        with open(self.output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                image = data["image"]
                self.passed_cache.add(image)
                self.passed_count += 1
        with open(self.droped_file, "r") as f:
            for line in f:
                data = json.loads(line)
                image = data["image"]
                self.droped_count += 1
                self.passed_cache.add(image)
    
    def write(self, image: str, src: str):
        self.passed_count += 1
        self.ostream.write(json.dumps({
            "image": image,
            "src": src,
        }) + "\n")
    
    def write_droped(self, image: str, src: str, reason: str):
        self.droped_count += 1
        self.droped_ostream.write(json.dumps({
            "image": image,
            "src": src,
            "reason": reason,
        }) + "\n")
    
    def flush(self):
        self.ostream.flush()
    
    def iter_sub_files(self):
        print("iter sub file", self.sub_file)
        with open(self.sub_file, "r") as f:
            tqdm_ui = tqdm(f)
            for line in tqdm_ui:
                data = json.loads(line)
                image = data["image"]
                if image in self.passed_cache:
                    continue
                
                src = data["src"] # type: str
                original_src = self.original_cache[image]
                original_src = original_src.strip()
                src = src.strip()
                yield image, original_src, src

                self.passed_cache.add(image)
                tqdm_ui.set_description(f"passed {self.passed_count}; droped {self.droped_count};")

# %%

def do_clean(
    checker: SimpleValidString,
    fm: FileMananger,
):

    for image, original_src, src in fm.iter_sub_files():
        ok, reason = checker.is_valid_with_reason(src, original_src)
        if ok:
            fm.write(image, src)
        else:
            fm.write_droped(image, src, reason)
        
def do_clean_sub(sub_idx, world_size, device):
    original_file = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc.jsonl"
    sub_file = f"../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_{sub_idx:02d}_of_{world_size:02d}_nougat.jsonl"
    output_file = f"../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_{sub_idx:02d}_of_{world_size:02d}_nougat_clean.jsonl"
    drop_file = f"../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_{sub_idx:02d}_of_{world_size:02d}_nougat_droped.jsonl"
    fm = FileMananger(
        original_file=original_file,
        sub_file=sub_file,
        output_file=output_file,
        droped_file=drop_file,
    )
    checker = SimpleValidString(
        device=device,
        seq_similirity_grater_than=0.7,
        len_diff_less_than=0.2,
    )
    do_clean(checker, fm)

def main_parallel(world_size = 9, device="cuda"):
    processor = [] # type: list[multiprocessing.Process]
    for sub_idx in range(world_size):
        p = multiprocessing.Process(
            target=do_clean_sub,
            args=(sub_idx, world_size, device),
        )
        p.start()
        processor.append(p)

    for proc in processor:
        proc.join()

def main_math_picked():
    original_file = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc.jsonl"
    sub_file = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math.jsonl"
    output_file = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math_clean.jsonl"
    drop_file = f"../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_math_droped.jsonl"
    fm = FileMananger(
        original_file=original_file,
        sub_file=sub_file,
        output_file=output_file,
        droped_file=drop_file,
    )
    checker = SimpleValidString(
        device="cuda",
        seq_similirity_grater_than=0.5,
        len_diff_less_than=0.7,
    )
    do_clean(checker, fm)

#%%
if __name__ == "__main__":
    main_math_picked()
    print("done")
    