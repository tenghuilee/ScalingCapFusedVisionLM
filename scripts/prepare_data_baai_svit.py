import json
import os
from typing import Iterator

import data_basic
from tqdm import tqdm


class ProcessChatsBase:
    
    # implement for iterator
    def iter(self, conversations: list, bbox: data_basic.BBoxFormater) -> Iterator[data_basic.ChatRecord]:
        ...

class ProcessReferringQA(ProcessChatsBase):

    def __init__(self, max_round: int):
        self.max_round = max_round
    
    def iter(self, conversations: list, bbox: data_basic.BBoxFormater):
        max_round = self.max_round
        conv_counts = 0
        for conv in conversations:
            for _ in conv["content"]:
                conv_counts += 1
        
        count_split = conv_counts // 2
        while count_split > max_round:
            count_split //= 2
        max_round = count_split
        assert max_round > 0 
        
        rec = data_basic.ChatRecord()
        rec_count = 0

        for conv in conversations:
            try:
                for item in conv["content"]:
                    _from = item["from"]
                    value = item["value"]
                    if _from == "user":
                        rec.append_user(bbox.apply(value))
                    elif _from == "gpt":
                        rec.append_assistant(bbox.apply(value))
                    else:
                        raise ValueError(f"Unknown from: {_from}")
                    rec_count += 1
                    if rec_count >= 2*max_round:
                        yield rec
                        rec = data_basic.ChatRecord()
                        rec_count = 0

            except ValueError as e:
                print(f"Error: {e}. Ignore this conversation.")
                rec = data_basic.ChatRecord()
                rec_count = 0

        if rec_count > 0:
            yield rec


class ProcessConversation(ProcessChatsBase):

    """
    No any bounding boxes
    
    """

    # def iter(self, conversations: list, bbox: data_basic.BBoxFormater):
    #     rec = data_basic.ChatRecord()
    #     for conv in conversations:
    #         print(conv)
    #         print(len(conv["content"]) // 2)

    #         for item in conv["content"]:
    #             assert bbox.has_bbox(item["value"])
    #     yield rec

    def iter(self, conversations: list, bbox: data_basic.BBoxFormater):
        for conv in conversations:
            rec = data_basic.ChatRecord()
            for item in conv["content"]:
                _from = item["from"]
                value = item["value"]
                if _from == "user":
                    rec.append_user(bbox.apply(value))
                elif _from == "gpt":
                    rec.append_assistant(bbox.apply(value))
                else:
                    raise ValueError(f"Unknown from: {_from}")
            yield rec

class ProcessDetailDescription(ProcessChatsBase):
    """
    No any bounding boxes
    """

    def iter(self, conversations: list, bbox: data_basic.BBoxFormater):
        for conv in conversations:
            rec = data_basic.ChatRecord()
            for item in conv["content"]:
                _from = item["from"]
                value = item["value"]
                if _from == "user":
                    rec.append_user(bbox.apply(value))
                elif _from == "gpt":
                    rec.append_assistant(bbox.apply(value))
                else:
                    raise ValueError(f"Unknown from: {_from}")
            yield rec


class ProcessComplexReasoning(ProcessChatsBase):
    """
    No any bounding boxes
    """

    def __init__(self, max_round: int) -> None:
        super().__init__()
        self.max_round = max_round
    

    def iter(self, conversations: list, bbox: data_basic.BBoxFormater):
        max_round = self.max_round
        conv_counts = 0
        for conv in conversations:
            for _ in conv["content"]:
                conv_counts += 1
        
        count_split = conv_counts // 2
        while count_split > max_round:
            count_split //= 2
        max_round = count_split
        assert max_round > 0

        rec = data_basic.ChatRecord()
        rec_count = 0

        for conv in conversations:
            for item in conv["content"]:
                _from = item["from"]
                value = item["value"]
                if _from == "user":
                    rec.append_user(bbox.apply(value))
                elif _from == "gpt":
                    rec.append_assistant(bbox.apply(value))
                else:
                    raise ValueError(f"Unknown from: {_from}")
                
                rec_count += 1
                if rec_count >= 2 * max_round:
                    yield rec
                    rec = data_basic.ChatRecord()
                    rec_count = 0
        if rec_count > 0:
            yield rec


class ProcessMain:

    SRC_FOLDER = "./datasets/baai_svit"
    IMAGE_ROOT = "./datasets"

    JSON_PATH_DICT = {
        "complex_reasoning": ("complex_reasoning.json", ProcessComplexReasoning(max_round=10)),
        "conversation": ("conversation.json", ProcessConversation()),
        "detail_description": ("detail_description.json", ProcessDetailDescription()),
        "referring_qa": ("referring_qa.json", ProcessReferringQA(max_round=10)),
        # "svit": "svit.json",
    }

    VG_PATHS = [
        "vg/VG_100K",
        "vg/VG_100K_2",
    ]

    def __init__(self) -> None:

        self.image_files_cached = [
            set(os.listdir(os.path.join(self.IMAGE_ROOT, folder))) 
            for folder in self.VG_PATHS
        ]
    
    def get_image_path(self, image_id: str):
        image_name = f"{image_id}.jpg"
        for image_files, folder in zip(self.image_files_cached, self.VG_PATHS):
            if image_name in image_files:
                return os.path.join(folder, image_name)
        raise ValueError(f"Image {image_name} not found")

    def process(self, out_path: str, in_path: str, process: ProcessChatsBase):
        with open(in_path, "r") as f:
            data = json.load(f)
    
        writer = data_basic.OutputWriter(out_path)
        bbox = data_basic.BBoxFormater()

        for item in tqdm(data):
            image_id = item["image_id"]
            image_path = self.get_image_path(image_id)

            for rec in process.iter(item["conversations"], bbox):
                rec.image = image_path
                rec = rec # type: data_basic.ChatRecord

                # do filter
                qa0 = rec.messages[0]["content"]

                if (len(qa0) < 4):
                    print("Too short", rec)
                    continue

                writer.write(rec)
        
        writer.dump_summary()
    
if __name__ == "__main__":
    pmain = ProcessMain()
    for key, (src_json, func) in pmain.JSON_PATH_DICT.items():
        pmain.process(
            f"./datasets/prepare_baai_svit_{key}.jsonl",
            os.path.join(pmain.SRC_FOLDER, src_json),
            func,
        )

