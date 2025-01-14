# md5
import hashlib
import json
import os
import re
import sqlite3
import warnings
from typing import Iterator, Tuple

import data_basic
import PIL
import PIL.Image
from tqdm import tqdm


def iter_jsonl(jsonl_path: str) -> Iterator[Tuple[str, list]]:
    with open(jsonl_path, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)

            images = item["image"]
            if len(images) > 1:
                # do not support multi-image
                continue

            image = images[0]
            messages = item["messages"]

            yield image, messages


def quick_md5(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class ImageFileChecker:

    DB_FILE = "image_md5.db"

    def __init__(self, dst_root: str):
        self.dst_root = dst_root
        os.makedirs(os.path.dirname(self.dst_root), exist_ok=True)
        self.cursor = None

        self._init_table()

    def __del__(self):
        self.conn.commit()
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None

    def _init_table(self):
        self.conn = sqlite3.connect(os.path.join(self.dst_root, self.DB_FILE))
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS image_md5 (
            image_path TEXT PRIMARY KEY UNIQUE,
            md5 TEXT
        )""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS error_images (
            image_path TEXT PRIMARY KEY UNIQUE
        )""")
        self.conn.commit()

    def is_md5_cached(self, image: str) -> bool:
        self.cursor.execute(
            "SELECT md5 FROM image_md5 WHERE image_path=?", (image,))
        return self.cursor.fetchone() is not None

    def cache_md5(self, image: str, md5: str):
        self.cursor.execute(
            "INSERT INTO image_md5 (image_path, md5) VALUES (?, ?)", (image, md5))

    def get_md5(self, image: str) -> str:
        self.cursor.execute(
            "SELECT md5 FROM image_md5 WHERE image_path=?", (image,))
        ans = self.cursor.fetchone()
        if ans is None:
            return None
        else:
            return ans[0]
    
    def is_error_image(self, image: str) -> bool:
        self.cursor.execute(
            "SELECT image_path FROM error_images WHERE image_path=?", (image,))
        return self.cursor.fetchone() is not None

    def add_error_image(self, image: str):
        self.cursor.execute(
            "INSERT INTO error_images (image_path) VALUES (?)", (image,))
    
    def varify_and_move_files(self, jsonl_path: str, image_root: str):
        assert image_root != self.dst_root, "image_root should not be the same as dst_root"
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Image root {image_root} not found")

        passed_images = set()

        for image, _ in iter_jsonl(jsonl_path):
            if image in passed_images:
                continue

            if self.is_error_image(image):
                continue

            # 1. check image exists
            dst_path = os.path.join(self.dst_root, image)
            src_path = os.path.join(image_root, image)

            if not os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    self.add_error_image(image)
                    print(f"Image {src_path} not found")
                else:
                    if self.is_md5_cached(image):
                        continue
                    else:
                        try:
                            warnings.warn("Image not found, but dst exists")
                            _img = PIL.Image.open(dst_path)
                            _img.verify()
                            _img.close()

                            dst_md5 = quick_md5(dst_path)
                            self.cache_md5(image, dst_md5)
                        except Exception as e:
                            print(f"Error when computing md5 for {dst_path}: {e}")
                            self.add_error_image(image)
                continue

            try:
                src_md5 = quick_md5(src_path)
            except Exception as e:
                print(f"Error when computing md5 for {src_path}: {e}")
                self.add_error_image(image)
                continue

            dst_md5 = self.get_md5(image) 
            if dst_md5 is not None:
                assert src_md5 == dst_md5, f"md5 mismatch for {image}. {src_md5} != {dst_md5}"
                continue

            if os.path.exists(dst_path):
                # file collision
                # md5 should be located in self.md5_cache
                # this branch should not be reached
                # The cached configuration is wrong or something else
                warnings.warn(
                    f"md5 missing for {image} {dst_md5}. This should not happen.")
                
                dst_md5 = quick_md5(dst_path)
                assert src_md5 == dst_md5, f"md5 mismatch for {image}. {src_md5} != {dst_md5}"
                # re compute the md5
                self.cache_md5(image, dst_md5)
            else:
                # image is not in dst_root
                try:
                    # check image is valid
                    _img = PIL.Image.open(src_path)
                    _img.verify()
                    _img.close()
                
                    self.cache_md5(image, src_md5)

                    # move the image
                    image_sub_path = image.rsplit("/", maxsplit=1)[0]
                    os.makedirs(os.path.join(self.dst_root,
                                image_sub_path), exist_ok=True)
                    os.rename(src_path, dst_path)
                except Exception as e:
                    print(f"Error opening image {src_path}: {e}")
                    self.add_error_image(image)

            passed_images.add(image)
        
        self.conn.commit()
        return passed_images


class ProcessMain:

    REMOVE_IMAGE_TOKEN = re.compile(r"\n*<\|image\|>\n*")

    UPDATE_IMAGE_PTH = (
        re.compile(r"^\./imgs/"),
        "mPLUG_imgs/",
    )

    def __init__(self, checker: ImageFileChecker):
        self.checker = checker

    def process_conv(self, image: str, messages: dict) -> Iterator[data_basic.ChatRecord]:
        rec = data_basic.ChatRecord(image=image)

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            content = re.sub(self.REMOVE_IMAGE_TOKEN, "", content)
            if role == "user":
                rec.append_user(content)
            elif role == "assistant":
                rec.append_assistant(content)
            else:
                raise ValueError(f"Unknown role: {role}")

        yield rec

    def process(self, dst_path: str, src_path: str) -> None:

        print("Loading jsonl file...")
        print(src_path)
        print("->")
        print(dst_path)

        writer = data_basic.OutputWriter(dst_path)

        for image, messages in iter_jsonl(src_path):

            if not self.checker.is_md5_cached(image):
                # filter out images that are not in the dst_root
                continue

            new_img_name = re.sub(
                self.UPDATE_IMAGE_PTH[0], self.UPDATE_IMAGE_PTH[1], image
            )

            for rec in self.process_conv(new_img_name, messages):
                writer.write(rec)

        writer.dump_summary()

if __name__ == "__main__":
    checker = ImageFileChecker(
        "/llms/data/mPLUG_merged_imgs"
    )

    # checker.varify_and_move_files(
    #     "/llms/data/mPLUG_DocStruct4M/struct_aware_parse.jsonl",
    #     "/llms/data/mPLUG_DocStruct4M/",
    # )

    # checker.varify_and_move_files(
    #     "/llms/data/mPLUG_DocStruct4M/multi_grained_text_localization.jsonl",
    #     "/llms/data/mPLUG_DocStruct4M/",
    # )

    # checker.varify_and_move_files(
    #     "/llms/data/mPLUG_DocDownstream-1.0/train.jsonl",
    #     "/llms/data/mPLUG_DocDownstream-1.0/",
    # )

    # checker.varify_and_move_files(
    #     "/llms/data/mPLUG_DocReason25K/detailed_explanation.jsonl",
    #     "/llms/data/mPLUG_DocReason25K",
    # )

    # for task in ["text_grounding.jsonl", "text_recognition.jsonl"]:
    #     checker.varify_and_move_files(
    #         f"/llms/data/mPLUG_DocLocal4k/{task}",
    #         f"/llms/data/mPLUG_DocLocal4k",
    #     )
    # 

    mpross = ProcessMain(checker)

    task_list = [
        ("struct_aware_parse", "mPLUG_DocStruct4M/struct_aware_parse.jsonl"),
        ("multi_grained_text_localization", "mPLUG_DocStruct4M/multi_grained_text_localization.jsonl"),
        ("detailed_explanation", "mPLUG_DocReason25K/detailed_explanation.jsonl"),
        ("text_grounding", "mPLUG_DocLocal4k/text_grounding.jsonl"),
        ("text_recognition", "mPLUG_DocLocal4k/text_recognition.jsonl"),
        ("down_stream", "mPLUG_DocDownstream-1.0/train.jsonl"),
    ]

    for task, src_path in task_list:
        mpross.process(
            f"./datasets/prepare_mPLUG_Doc_{task}.jsonl",
            f"./datasets/{src_path}",
        )

