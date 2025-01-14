#%%
import json
import re
from tqdm.auto import tqdm
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance
import time

#%%

def dump_item(item: dict):
    for chat in item["messages"]:
        if chat["role"] == "user":
            print("\033[31m", chat["content"], "\033[0m")
        else:
            print("\033[32m", chat["content"], "\033[0m")

#%%

# data_src = "../datasets/prepare_llava_v1_5_mix665k.jsonl"
# data_src = "../datasets/prepare_baai_svit_complex_reasoning.jsonl"
data_src = "../datasets/prepare_merged_clean_v1.jsonl"
temp_src = "../datasets/temp_cache.jsonl"

_yn_patten = re.compile(r"Please answer yes or no\.", re.IGNORECASE)

all_queries = set()
with open(data_src, "r") as istream:
    flag = False
    progress = tqdm(istream)
    for line in progress:
        item = json.loads(line)
        if item["image"] is None:
            continue
        for message in item["messages"]:
            if message["role"] == "user":
                pass
                # all_queries.add(message["content"])
            elif message["role"] == "assistant":
                continue
            else:
                raise ValueError("unknown role")
            
            ctx = message["content"]
            if _yn_patten.search(ctx):
                flag = True
                break

            all_queries.add(ctx)
        if flag:
            break

#%%

for que in all_queries:
    if _yn_patten.search(que):
        print(que)
        break

#%%

class QuestionChecker:
    ocr_ref = "Reference OCR token"

    _ptn_question = re.compile(r"^(What|Who|When|Where|Why|How|Which|Whose|Is|Are|Can|Could|Do|Does|Did|What's|Who's|On which|On how|In which|In what) .+?\?$", re.IGNORECASE)

    _ptn_prov = re.compile(r"^Please provide (a|the)")

    _ptn_choice = re.compile(
        r"(Answer with the option's letter from the given choices directly)|(Answer the question using a single word or phrase).$")
    
    _pth_describe = re.compile(
        r"^Describe the"
    )

    def __init__(self):

        self.passed_count = 0

    def filter_out(self, que: str):
        if self.ocr_ref in que:
            return None
        
        if (
            (len(que) > 2 and que[-1] == "?")
            or self._ptn_choice.search(que)
            or self._pth_describe.search(que)
            or self._ptn_question.search(que)
            or self._ptn_prov.search(que)
        ):
            self.passed_count += 1
            return 0

        return 1

query_checker = QuestionChecker()

further_filted_set = set()
for que in all_queries:
    check_result = query_checker.filter_out(que)
    if check_result == 1:
        further_filted_set.add(que)
    # elif check_result == 0:
    #     print("\033[32m", que, "\033[0m")
    #     break

print(f"passed {query_checker.passed_count}")
print("filter out", len(further_filted_set))

with open("./temp_all_queries.txt", "w") as ostream:
    for que in further_filted_set:
        ostream.write(json.dumps({"que": que}) + "\n")

# %%
