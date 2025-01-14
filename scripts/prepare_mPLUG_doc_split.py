#%%
import re
import json
from tqdm.auto import tqdm

#%%
data_src = "../datasets/prepare_mPLUG_Doc_struct_aware_parse.jsonl"

out_doc = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_doc.jsonl"
out_md = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_md.jsonl"
out_other = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_other.jsonl"

fout_doc = open(out_doc, "w")
fout_md = open(out_md, "w")
fout_other = open(out_other, "w")

# statistic </(\w+)> in data_src
# There are two type of tags in the document
# 1. <doc> ... </doc>
# 2. <md> ... </md>
ptn_tag = re.compile(r".*?<(doc|md)>(.*?)</(doc|md)>", re.DOTALL)
tag_set = set() # type: set[str]
with open(data_src, "r") as f:
    for line in tqdm(f):
        line_data = json.loads(line)
        image = line_data["image"]
        que = line_data["messages"][0]["content"]
        ans = line_data["messages"][1]["content"]

        ctx = que + ans

        ctx_match = ptn_tag.match(ctx)
        if ctx_match is None:
            fout_other.write(line)
            continue

        tag = ctx_match.group(1)
        src = ctx_match.group(2)

        if tag == "doc":
            fout_doc.write(json.dumps({
                "image": image,
                "src": src,
            }) + "\n")
        elif tag == "md":
            fout_md.write(json.dumps({
                "image": image,
                "src": src,
            }) + "\n")
        else:
            print("\033[31mError: Unknown tag\033[0m")
            raise ValueError("Unknown tag")

fout_doc.close()
fout_other.close()
fout_md.close()

# %%
