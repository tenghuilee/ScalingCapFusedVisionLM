# %%
import json
from collections import Counter

import nltk
import numpy as np
from nltk.util import ngrams
from tqdm.auto import tqdm

# %% 

class FileOperator:
    def __init__(self):
        self.src_paths = [
            f"../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_{i:02d}_of_09_nougat_clean.jsonl"
            for i in range(9)
        ]

        self.accepted_file = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_nougat_clean_remove_dumpicated_pattens.jsonl"
        self.rejected_file = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc_nougat_clean_remove_dumpicated_pattens_rejected.jsonl"

        self.astream = open(self.accepted_file, "w")
        self.rstream = open(self.rejected_file, "w")
        self.rejected_count = 0
        self.accepted_count = 0
    
    def __del__(self):
        self.astream.close()
        self.rstream.close()
    
    def accept(self, item: dict):
        self.accepted_count += 1
        self.astream.write(json.dumps(item) + "\n")

    def reject(self, item: dict):
        self.rejected_count += 1
        self.rstream.write(json.dumps(item) + "\n")
    
    def iter(self):
        for r, src_path in enumerate(self.src_paths):
            with open(src_path, "r") as f:
                for line in f:
                    yield r, json.loads(line)

# %%

max_ratio = 0.12
window_size = 256

file_operator = FileOperator()

progress_bar = tqdm(file_operator.iter())

for idx, item in progress_bar:

    ctx = item["src"]

    tokens = nltk.word_tokenize(ctx)
    if len(tokens) < 10:
        file_operator.reject(item)
        continue

    is_accepted = False
    if len(tokens) > window_size:
        tokens = tokens[-window_size:]
    
    sub_vocab = set(tokens)
        
    if len(sub_vocab) / len(tokens) < max_ratio:
        file_operator.reject(item)
    else:
        file_operator.accept(item)
    
    progress_bar.set_description(f"Accepted: {file_operator.accepted_count}, Rejected: {file_operator.rejected_count}; @{idx} of 9")
        
# %%
