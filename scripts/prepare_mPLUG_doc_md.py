#%%
import enum
import json
import os
import random
import re

import itertools
import mistletoe
import mistletoe.latex_renderer
import mistletoe.markdown_renderer
from mistletoe import block_token
from tqdm.auto import tqdm

#%%

class SimpleMarkdownRender(mistletoe.markdown_renderer.MarkdownRenderer):
    pass

class SimpleLatexRender(mistletoe.latex_renderer.LaTeXRenderer):

    def render_document(self, token):
        return self.render_inner(token)

class SimpleHttpRender(mistletoe.HtmlRenderer):
    def render_list(self, token: block_token.List) -> str:
        template = '<{tag}>\n{inner}\n</{tag}>'
        if token.start is not None:
            tag = 'ol'
        else:
            tag = 'ul'
        self._suppress_ptag_stack.append(not token.loose)
        inner = '\n'.join([self.render(child) for child in token.children])
        self._suppress_ptag_stack.pop()
        return template.format(tag=tag, inner=inner)

    def render_table_cell(self, token: block_token.TableCell, in_header=False) -> str:
        template = '<{tag}>{inner}</{tag}>\n'
        tag = 'th' if in_header else 'td'
        inner = self.render_inner(token)
        return template.format(tag=tag, inner=inner)

class RenderType(enum.Enum):
    HTML = 0
    MD = 1
    LaTeX = 2
    LaTeX_Complete = 3

class QueryPool:
    level_1 = ['Translate', 'Present', 'Reformat', 'Express', 'Format', 'Extract', 'Interpret',
                'Represent', 'Convert', 'Read', 'Transform', 'Turn', 'Output', 'Recreate', 'Render', 'Encode']
    level_2 = ["the", "the given", "the input", "the provided"]
    level_3 = ["image", "chart", "figure", "picture", "diagram", "illustration", "table"]
    level_4 = ["as", "into", "to", "in", "to fit"]
    _level_5 = {
        RenderType.MD: ["markdown", "Markdwon", "MARKDOWN", "md", "MD"],
        RenderType.HTML: ["html", "HTML", "XHTML"],
        RenderType.LaTeX: ["latex", "LaTeX", "Latex", "LATEX"],
        RenderType.LaTeX_Complete: ["complete latex", "Complete LaTeX", "COMPLETE LATEX", "Full Latex", "full latex", "FULL LATEX", "Full LaTeX"],
    }

    level_6 = ["", " format", " structure", " style", ".", " format.", " structure.", " style."]

    @property
    def level_5(self):
        ans = []
        for rt in RenderType:
            ans.extend(self._level_5[rt])
        return ans

    def __init__(self, seed=42):
        self.random = random.Random(seed)
    
    def iter_pool_list(self):
        for lx in itertools.product(
            self.level_1,
            self.level_2,
            self.level_3,
            self.level_4,
            self.level_5,
            self.level_6,
        ):
            yield ' '.join(lx[0:-1]) + lx[-1]
    
    def pool_list(self):
        return list(self.iter_pool_list())
    
    def convert_context_md(self, context: str, render_type: RenderType = RenderType.MD) -> str:
        # only support
        if render_type == RenderType.HTML:
            ans_txt = mistletoe.markdown(context, renderer=SimpleHttpRender)
        elif render_type == RenderType.MD:
            ans_txt = mistletoe.markdown(context, renderer=SimpleMarkdownRender)
        elif render_type == RenderType.LaTeX:
            ans_txt = mistletoe.markdown(context, renderer=SimpleLatexRender)
        elif render_type == RenderType.LaTeX_Complete:
            ans_txt = mistletoe.markdown(context, renderer=mistletoe.latex_renderer.LaTeXRenderer)
        else:
            raise ValueError(f"Only support {list(RenderType)}")

        return ans_txt
    
    def random_pool_list(self):
        l1 = self.random.choice(self.level_1)
        l2 = self.random.choice(self.level_2)
        l3 = self.random.choice(self.level_3)
        l4 = self.random.choice(self.level_4)
        l6 = self.random.choice(self.level_6)
        rt = self.random.choice(list(RenderType))
        l5 = self.random.choice(self._level_5[rt])
        return rt, f"{l1} {l2} {l3} {l4} {l5}{l6}"

#%%

data_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_md.jsonl"
output_path = "../datasets/prepare_mPLUG_Doc_struct_aware_parse_md_html_latex_mixed.jsonl"
ostream = open(output_path, "w")

que_pool = QueryPool()

#%%

with open(data_path, "r") as f:

    for line in tqdm(f):
        item = json.loads(line)
        image = item["image"]
        context = item["src"]

        rt, que = que_pool.random_pool_list()

        ans_txt = que_pool.convert_context_md(context, rt)
        
        ostream.write(json.dumps({
            "image": image,
            "messages": [
                {
                    "role": "user", 
                    "content": que,
                },
                {
                    "role": "assistant",
                    "content": ans_txt,
                }
            ]
        }) + "\n")

ostream.close()
print("Done!")

#%%
