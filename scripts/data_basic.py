"""
basic format of the dataset formats

Using JsonL format
{
    "image": "image_name",
    "messages": [
        {
            "role": "user",
            "content": "user's message"
        },
        {
            "role": "assistant",
            "content": "assistant's message",
        },
        ...
    ]
}
"""

import re
import io
import json
from typing import Tuple
from tqdm import tqdm
from dataclasses import dataclass, field

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

@dataclass
class ChatRecord:
    image: str = None
    messages: list[dict] = field(default_factory=list)

    def append(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        return self
    
    def append_user(self, content: str):
        return self.append(ROLE_USER, content)
    
    def append_assistant(self, content: str):
        return self.append(ROLE_ASSISTANT, content)
    
    def extend_with_list(self, messages: list[str]):
        for i, msg in enumerate(messages):
            if i % 2 == 0:
                self.append_user(msg)
            else:
                self.append_assistant(msg)
        return self

    def to_dict(self):
        return {
            "image": self.image,
            "messages": self.messages,
        }
    
    @property
    def num_messages(self):
        return len(self.messages)
    
    def to_json(self, out_file: io.TextIOBase = None):
        if out_file is None:
            return json.dumps(self.to_dict())
        else:
            if not out_file.writable():
                raise ValueError("The file is not writable.")
            
            json.dump(
                self.to_dict(),
                out_file,
            )

class OutputWriter:
    """Using Jsonl format"""
    def __init__(self, out_file: str):
        self.out_file = open(out_file, "w")
        self.summary_chat_rounds = dict()
        self.summary_num_records = 0
    
    def __del__(self):
        self.out_file.flush()
        self.out_file.close()
    
    def write(self, chat_record: ChatRecord):
        chat_record.to_json(self.out_file)
        self.out_file.write("\n")

        self.summary_num_records += 1

        assert chat_record.num_messages > 0, "The number of messages must be greater than 0."
        if chat_record.num_messages % 2 != 0:
            print(chat_record)
            raise ValueError("The number of messages must be even.")

        chat_rounds = chat_record.num_messages // 2
        
        if chat_rounds in self.summary_chat_rounds:
            self.summary_chat_rounds[chat_rounds] += 1
        else:
            self.summary_chat_rounds[chat_rounds] = 1
    
    def dump_summary(self):
        print("Summary:")
        for key in sorted(self.summary_chat_rounds.keys()):
            print(f"    {key}: {self.summary_chat_rounds[key]}")
        print(f"    Total: {self.summary_num_records}")

class BBoxFormater:
    """
    Converting bounding boxes with formats
     - [x0, y0, x1, y1] where x0, y0, x1, y1 are float numbers from [0, 1], 
    to the format of
    - <bbox>x0, y0, x1, y1</bbox> where x0, y0, x1, y1 are integers from [0, 999] 
    """

    def __init__(self):
        self.ptn_bracket_bbox_format = re.compile(r"\[([\d.]+), *([\d.]+), *([\d.]+), *([\d.]+)\]")
        self.ptn_bbox_format = re.compile(r"<bbox>([\d]+), *([\d]+), *([\d]+), *([\d]+)</bbox>")

    def _clip(self, x: str, min_value, max_value):
        return min(max(x, min_value), max_value)
    
    def re_match_to_float(self, re_match: re.Match, min_value=0.0, max_value=1.0):
        return [
            self._clip(float(re_match.group(i)), min_value, max_value)
            for i in range(1, 5)
        ]

    def replace_format(self, re_match: re.Match):
        ax = self.re_match_to_float(re_match)
        ax = [int(x * 999) for x in ax]
        return f"<bbox>{ax[0]:d},{ax[1]:d},{ax[2]:d},{ax[3]:d}</bbox>"

    def apply(self, src: str):
        return re.sub(self.ptn_bracket_bbox_format, self.replace_format, src)
    
    def has_bracket_bbox(self, src: str):
        return self.ptn_bracket_bbox_format.search(src) is not None

    def has_bbox(self, src: str):
        return self.ptn_bbox_format.search(src) is not None


class BBoxFormaterWithSquarePadded(BBoxFormater):

    def apply(self, src: str, image_size: Tuple[int, int]):
        w, h = image_size
        wh = max(w, h)

        if w > h:
            shift_w, shift_h = 0, (w - h) // 2
        else:
            shift_w, shift_h = (h - w) // 2, 0
        
        def __inner(re_match: re.Match):
            ax = self.re_match_to_float(re_match, min_value=0.0, max_value=1.0)

            ax = [
                (ax[0] * wh - shift_w) / w,
                (ax[1] * wh - shift_h) / h,
                (ax[2] * wh - shift_w) / w,
                (ax[3] * wh - shift_h) / h,
            ]

            ax = [int(self._clip(x, 0.0, 1.0) * 999) for x in ax]

            return f"<bbox>{ax[0]:d},{ax[1]:d},{ax[2]:d},{ax[3]:d}</bbox>"
        
        return re.sub(self.ptn_bracket_bbox_format, __inner, src)
