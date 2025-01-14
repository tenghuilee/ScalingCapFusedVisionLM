import abc
import functools
import json
import os
import re
from dataclasses import dataclass, field
from typing import (Callable, Dict, List, Optional, Sequence, SupportsIndex,
                    Tuple, Union)

import torch
import torch.utils
import transformers
from tokenizers import processors
from torch.utils.data import Dataset

from tensorfusionvlm.chat_parser import *
from tensorfusionvlm.constants import *
from tensorfusionvlm.constants import EnumTokenType
from tensorfusionvlm.utils import *


def _debug_func(func):
    def __inner(*args, **kwargs):
        print(f"Calling {func.__name__}")
        ans = func(*args, **kwargs)
        print(f"End call {func.__name__}")
        return ans
    return __inner

def append_qed_parser_tokens(
    tokenizer: transformers.PreTrainedTokenizer,
    length_que_end: int = 0,
    length_que_end_short: int = None,
    append_special_padding: bool = False,
):
    ChatQEDAnswerParser(tokenizer, length_que_end, length_que_end_short, append_special_padding)
    return len(tokenizer)

# try to decompose the dataset into smaller ones

class UnionChatDatasetPartition(abc.ABC):
    # Please implement this funcion in sub classes
    def get_item(self, chats: ChatRecord) -> dict: ...

    # Please implement this funcion in sub classes
    def collect_fn(self, instances: list[dict]) -> dict: ...

    def get_output_dict(self, **kwargs):
        return {k: v for k, v in kwargs.items()}


class UnionChatDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        partitions: list[UnionChatDatasetPartition],
        drop_no_image_instances: bool = False,
        remove_image_tag: str = DEFAULT_IMAGE_TOKEN, # deprecated: ignore image tag
        list_data_dict: list[dict] = None,
    ):
        """
        Required:
            - data_path: path to the data file
            - partitions: list of partitions
        Optional:
            - drop_no_image_instances: drop instances without image
            - remove_image_tag: remove image tag from text
               (deprecated: ignore this)

        """
        super().__init__()

        self.partitions = partitions
        assert len(self.partitions) > 0, "no partition is provided"
        self.drop_no_image_instances = drop_no_image_instances

        # filter list data dict
        def is_valid(sample: dict, drop: bool):
            item = ChatRecord.from_dict(sample)
            if drop and (not item.has_image):
                # no image
                return False
            len_convs = len(item.messages)
            if len_convs < 2:
                # has at least one question and one answer
                return False
            chat_char_count = 0
            for msg in item.messages:
                chat_char_count += len(msg)
            if chat_char_count < 25 or chat_char_count > 5000:
                return False
            return True

        if list_data_dict is None:
            list_data_dict = []
            with open(data_path, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    if is_valid(sample, drop_no_image_instances):
                        list_data_dict.append(sample)

        self.list_data_dict = list_data_dict

        self.enable_square_image_bbox_remap = False
        for i, par in enumerate(self.partitions):
            if not isinstance(par, (UCDP_Image_Base, )):
                continue
            if i == 0:
                continue
            raise ValueError("Please place image partition at the begining of the list.")
        

    def __len__(self):
        return len(self.list_data_dict)

    @torch.inference_mode()
    def __getitem__(self, index: int):
        sample = self.list_data_dict[index]
        chat = ChatRecord.from_dict(sample)

        outputs = dict()
        for partition in self.partitions:
            outputs.update(partition.get_item(chat))
        return outputs


class DataCollactorForUnionChatDataset:
    def __init__(self, partitions: list[UnionChatDatasetPartition]) -> None:
        self.partitions = partitions

    @torch.inference_mode()
    def __call__(self, instances: list[dict]) -> dict:
        outputs = dict()
        for partition in self.partitions:
            outputs.update(partition.collect_fn(instances))
        return outputs


class UCDPWraper_DropNoImageInstances(UnionChatDatasetPartition):

    def __init__(self, base_partition: UnionChatDatasetPartition, name_of_key: str):
        self.base_partition = base_partition
        self.name_of_key = name_of_key

    def get_item(self, chats: ChatRecord) -> Dict:
        if (chats.image is None) or (chats.image.lower() in ["none", "no", "unavailable", "undefined", ""]):
            outputs = self.base_partition.get_output_dict()
            outputs[self.name_of_key] = -1
        else:
            outputs = self.base_partition.get_item(chats)
            outputs[self.name_of_key] = 0
        return outputs

    def collect_fn(self, instances: List[Dict]) -> Dict:
        kept_instances = []
        # masks = []
        for i, inst in enumerate(instances):
            if inst[self.name_of_key] >= 0:
                # masks.append(i)
                kept_instances.append(inst)

        outputs = self.base_partition.collect_fn(kept_instances)
        # outputs[self.name_of_key] = torch.tensor(masks, dtype=torch.long)
        return outputs

    @staticmethod
    def wrap(name_of_key: str):
        def __inner(cls: UnionChatDatasetPartition):
            @functools.wraps(cls)
            def __inner_wraps(*args, **kwargs):
                return UCDPWraper_DropNoImageInstances(cls(*args, **kwargs), name_of_key)
            return __inner_wraps
        return __inner


class UCDP_Image_Base(UnionChatDatasetPartition):

    def __init__(
        self,
        image_folder: str,
        image_helper: ImageHelperBase,
        image_aspect_ratio: str = "pad",
    ):
        self.image_folder = image_folder
        self.image_aspect_ratio = image_aspect_ratio
        self.image_helper = image_helper
        if image_aspect_ratio == "pad":
            # self.ptn_bbox = re.compile(
            #     r"\[[0-9.]+, [0-9.]+, [0-9.]+, [0-9.]+\]")
            self.ptn_bbox = re.compile(
                r"<bbox>(\d+), *(\d+), *(\d+), *(\d+)</bbox>")
        else:
            raise NotImplementedError("image aspect ratio must be 'pad'")

    def map_bbox(self, src: str, shape_info: ImageHelperWithProcessor.ShapeInfo):
        def __inner(re_match: re.Match):
            ax = [
                min(max(float(re_match.group(i)) / 999, 0.0), 1.0)
                for i in range(1, 5)
            ]

            ax = self.image_helper.map_bbox_square(
                shape_info, ax)
            
            ax = [int(x * 999) for x in ax]

            return f"<bbox>{ax[0]:d},{ax[1]:d},{ax[2]:d},{ax[3]:d}</bbox>"

        return re.sub(self.ptn_bbox, __inner, src)

class UCDP_Image(UCDP_Image_Base):

    def __init__(
        self,
        image_folder: str,
        image_processor: transformers.CLIPImageProcessor,
        image_aspect_ratio: str = "pad",
    ):
        super().__init__(
            image_folder=image_folder,
            image_helper=ImageHelperWithProcessor(
                processor=image_processor,
                image_aspect_ratio=image_aspect_ratio,
            ),
            image_aspect_ratio=image_aspect_ratio,
        )

    def get_output_dict(self, pixel_values: torch.Tensor = None):
        return {
            "pixel_values": pixel_values,
        }

    def get_item(self, chats: ChatRecord) -> dict:

        pixel_values, shape_info = self.image_helper.load_image(
            self.image_folder, chats.image, return_shape_info=True)

        if self.ptn_bbox is not None:
            convs = [self.map_bbox(cc, shape_info) for cc in chats.messages]
            chats.messages = convs

        return self.get_output_dict(pixel_values)

    def collect_fn(self, instances: list[dict]) -> dict:
        return self.get_output_dict(torch.stack([
            inst["pixel_values"] for inst in instances
        ], dim=0))

class UCDP_HDImage(UCDP_Image_Base):

    def __init__(
        self,
        image_folder: str,
        image_processor: Union[str, Phi3VImageProcessor],
        drop_empty_crops: bool = False,
    ):
        super().__init__(
            image_folder=image_folder,
            image_helper=ImageHelperWithPhi3Processor(
                image_processor,
            ),
            image_aspect_ratio="pad",
        )
        self.drop_empty_crops = drop_empty_crops

        self.image_helper = self.image_helper # type: ImageHelperWithPhi3Processor
    
    def get_output_dict(
        self,
        pixel_values: torch.Tensor = None,
        image_sizes: torch.Tensor = None,
        num_img_tokens: torch.Tensor = None,
        crop_index: list[int] = None,
    ):
        return {
            'pixel_values': pixel_values,
            'image_sizes': image_sizes,
            'num_img_tokens': num_img_tokens,
            'crop_index': crop_index,
        }
    
    def get_item(self, chats: ChatRecord) -> dict:

        if chats.image is None:
            return self.get_output_dict()

        img = self.image_helper.load_PIL_image(
            self.image_folder, chats.image, do_convert_rgb=True)

        pad_img, shape_info = self.image_helper.image_expand2square(img)

        if self.ptn_bbox is not None:
            convs = [self.map_bbox(cc, shape_info) for cc in chats.messages]
            chats.messages = convs
        
        image_size = self.image_helper.calc_hd_image_size(img)
        num_img_tokens = self.image_helper.calc_hd_num_image_tokens_by_size(
            *image_size)
        
        chats.image_size = image_size
        chats.num_img_tokens = num_img_tokens
        
        return self.get_output_dict(
            pixel_values=(pad_img, img),
            image_sizes=image_size,
            num_img_tokens=num_img_tokens,
        )

    def collect_fn(self, instances: list[dict]) -> dict:
        pad_img, img = [], []
        crop_index_mapping = dict()
        _has_image_index = 0
        for i, inst in enumerate(instances):
            if inst["pixel_values"] is None:
                continue
            else:
                crop_index_mapping[_has_image_index] = i
                _has_image_index += 1
            _px = inst["pixel_values"]
            pad_img.append(_px[0])
            img.append(_px[1])

        if _has_image_index == 0:
            return self.get_output_dict()
        
        out = self.image_helper.preprocess(
            image=img,
            pad_image=pad_img,
            return_tensors='pt',
        )

        num_img_tokens = out["num_img_tokens"]
        image_sizes = out["image_sizes"]
        if _has_image_index != len(instances):
            re_mapped_image_size = torch.zeros(
                (len(instances), 2), dtype=torch.long, device=image_sizes.device
            )

            for _i, _j in crop_index_mapping.items():
                re_mapped_image_size[_j] = image_sizes[_i]
            image_sizes = re_mapped_image_size

        if self.drop_empty_crops:
            assert isinstance(self.image_helper, ImageHelperWithPhi3Processor)
            cout = self.image_helper.drop_empty_crops(out, return_torch_tensor=False)

            # remap index
            pixel_values = torch.cat(cout["pixel_values"], dim=0)
            crop_index = cout["crop_index"]

            assert pixel_values.shape[0] == len(crop_index), f"batch size of pixel_values and crop_index must be the same, but got {len(pixel_values)} and {len(crop_index)}"

            if _has_image_index != len(instances):
                # mapping the values of crop_index to the original index
                mapped_cidx = [crop_index_mapping[idx] for idx in crop_index]
                crop_index = mapped_cidx

            return self.get_output_dict(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                num_img_tokens=num_img_tokens,
                crop_index=torch.tensor(crop_index),
            )
        else:
            # TODO: not tested
            pixel_values = out["pixel_values"]
            if _has_image_index != len(instances):
                re_mapped_pixel_values = torch.zeros(
                    (len(instances), *pixel_values.shape[1:]),
                    dtype=pixel_values.dtype,
                    device=pixel_values.device
                )
                for _i, _j in crop_index_mapping.items():
                    re_mapped_pixel_values[_j] = pixel_values[_i]
                pixel_values = re_mapped_pixel_values
                
            return self.get_output_dict(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                num_img_tokens=num_img_tokens,
            )

class UCDP_CLIPQuery_WithPlaceholder(UnionChatDatasetPartition):

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        length_que_end: int = None,
        length_que_end_short: int = None,
        max_sequence_length: int = None,
        append_special_padding: bool = False,
        _drop_and_replaced_query: Optional[str] = None,
        _drop_and_replaced_query_fill_ends: Optional[int] = 0,
    ):
        """
        if _drop_and_replaced_query is None,
            query <qed> ....
        else if _drop_and_replaced_query is a string,
            _drop_and_replaced_query <qed> ....
        If you don't have any priority, you can use None.
        If you want to drop the query, you can use "".
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.append_special_padding = append_special_padding

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.length_que_end = length_que_end
        self.length_que_end_short = length_que_end_short

        if self.pad_token_id is None:
            raise ValueError("The tokenizer must have a pad_token_id.")
         
        if _drop_and_replaced_query is None:
            self.parser = ChatQEDQueryParser(
                tokenizer=self.tokenizer,
                length_que_end=self.length_que_end,
                length_que_end_short=self.length_que_end_short,
                append_special_padding=self.append_special_padding,
            )
        elif isinstance(_drop_and_replaced_query, str):
            self.parser = ChatQEDDropQueryParser(
                replaced_query=_drop_and_replaced_query,
                fill_ends=_drop_and_replaced_query_fill_ends,
                tokenizer=self.tokenizer,
                length_que_end=self.length_que_end,
                length_que_end_short=self.length_que_end_short,
                append_special_padding=self.append_special_padding,
            )
        else:
            raise ValueError(f"Invalid _drop_and_replaced_query: {_drop_and_replaced_query}.")

    def get_output_dict(
        self,
        que_input_ids: torch.Tensor = None,
        que_input_tps: torch.Tensor = None,
        que_attention_mask: torch.Tensor = None,
    ):
        return {
            "que_input_ids": que_input_ids,
            "que_input_tps": que_input_tps,
            "que_attention_mask": que_attention_mask,
        }

    def get_item(self, chats: ChatRecord) -> Dict:

        outputs = self.parser.parse(
            chats.apply(),
            enable_que_end=chats.has_image
        )

        return self.get_output_dict(
            que_input_ids=outputs.input_ids,
            que_input_tps=outputs.input_tps,
            que_attention_mask=outputs.attention_mask,
        )

    def collect_fn(self, instances: List[Dict]) -> Dict:
        que_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["que_input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        que_input_tps = torch.nn.utils.rnn.pad_sequence(
            [item["que_input_tps"] for item in instances],
            batch_first=True,
            padding_value=EnumTokenType.PAD.value,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["que_attention_mask"] for item in instances],
            batch_first=True,
            padding_value=0,
        )
        return self.get_output_dict(
            que_input_ids=que_input_ids,
            que_input_tps=que_input_tps,
            que_attention_mask=attention_mask,
        )


class UCDP_CLIPAnswer_WithPlaceholder(UnionChatDatasetPartition):

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        length_que_end: int = None,
        length_que_end_short: int = None,
        append_special_padding: bool = False,
        max_sequence_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        self.length_que_end = length_que_end
        self.length_que_end_short = length_que_end_short
        self.append_special_padding = append_special_padding

        self.parser = ChatQEDAnswerParser(
            tokenizer=self.tokenizer,
            length_que_end=self.length_que_end,
            length_que_end_short=self.length_que_end_short,
            append_special_padding=self.append_special_padding,
        )

    def get_output_dict(
        self,
        ans_input_ids: torch.Tensor = None,
        ans_input_tps: torch.Tensor = None,
        ans_attention_mask: torch.Tensor = None,
        ans_indexes: torch.Tensor = 0,
    ):
        return {
            "ans_input_ids": ans_input_ids,
            "ans_input_tps": ans_input_tps,
            "ans_attention_mask": ans_attention_mask,
            "ans_indexes": ans_indexes,
        }

    def get_item(self, chats: ChatRecord) -> Dict:
        # tokenize

        outputs = self.parser.parse(
            chats.apply(),
            enable_que_end=chats.has_image,
            max_seq_len=self.max_sequence_length,
        )

        return self.get_output_dict(
            ans_input_ids=outputs.input_ids,
            ans_input_tps=outputs.input_tps,
            ans_attention_mask=outputs.attention_mask,
        )

    def collect_fn(self, instances: List[Dict]) -> Dict:
        sum_c, max_l = 0, 0
        for item in instances:
            c, l = item["ans_input_ids"].shape
            max_l = max(max_l, l)
            sum_c += c

        out_shape = (sum_c, max_l)
        ans_input_ids = torch.full(
            out_shape, fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        ans_input_tps = torch.full(
            out_shape, fill_value=EnumTokenType.PAD.value, dtype=torch.long)
        ans_attention_mask = torch.full(
            out_shape, fill_value=0, dtype=torch.long)

        ans_indexes = []
        _j = 0
        for item in instances:
            ans_indexes.append(_j)
            c, l = item["ans_input_ids"].shape
            for _i in range(c):
                ans_input_ids[_j, 0:l] = item["ans_input_ids"][_i]
                ans_input_tps[_j, 0:l] = item["ans_input_tps"][_i]
                ans_attention_mask[_j, 0:l] = item["ans_attention_mask"][_i]
                _j += 1

        return self.get_output_dict(
            ans_input_ids=ans_input_ids,
            ans_input_tps=ans_input_tps,
            ans_attention_mask=ans_attention_mask,
            ans_indexes=torch.tensor(ans_indexes, dtype=torch.long),
        )

class UCDP_CLIPAnswer(UnionChatDatasetPartition):

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_sequence_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        self.parser = ChatAnswerParser(
            tokenizer=self.tokenizer,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id


    def get_output_dict(
        self,
        ans_input_ids: torch.Tensor = None,
        ans_input_tps: torch.Tensor = None,
        ans_attention_mask: torch.Tensor = None,
        ans_indexes: torch.Tensor = 0,
    ):
        return {
            "ans_input_ids": ans_input_ids,
            "ans_input_tps": ans_input_tps,
            "ans_attention_mask": ans_attention_mask,
            "ans_indexes": ans_indexes,
        }

    def get_item(self, chats: ChatRecord) -> dict:
        """Get the item for the LLM query and answer"""
        outputs = self.parser.parse(
            chats.apply(),
            max_seq_len=self.max_sequence_length,
        )

        return self.get_output_dict(
            ans_input_ids=outputs.input_ids,
            ans_input_tps=outputs.input_tps,
            ans_attention_mask=outputs.attention_mask,
        )

    def collect_fn(self, instances: List[Dict]) -> Dict:
        sum_c, max_l = 0, 0
        for item in instances:
            c, l = item["ans_input_ids"].shape
            max_l = max(max_l, l)
            sum_c += c

        out_shape = (sum_c, max_l)
        ans_input_ids = torch.full(
            out_shape, fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        ans_input_tps = torch.full(
            out_shape, fill_value=EnumTokenType.PAD.value, dtype=torch.long)
        ans_attention_mask = torch.full(
            out_shape, fill_value=0, dtype=torch.long)

        ans_indexes = []
        _j = 0
        for item in instances:
            ans_indexes.append(_j)
            c, l = item["ans_input_ids"].shape
            for _i in range(c):
                ans_input_ids[_j, 0:l] = item["ans_input_ids"][_i]
                ans_input_tps[_j, 0:l] = item["ans_input_tps"][_i]
                ans_attention_mask[_j, 0:l] = item["ans_attention_mask"][_i]
                _j += 1

        return self.get_output_dict(
            ans_input_ids=ans_input_ids,
            ans_input_tps=ans_input_tps,
            ans_attention_mask=ans_attention_mask,
            ans_indexes=torch.tensor(ans_indexes, dtype=torch.long),
        )

class UCDP_LLMQueryAnswer_Base(UnionChatDatasetPartition):
    """The base class for LLM query and answer

    This class will provide well formated LLM inputs by well organized queries, anwsers, system_prompt and other instruction characters.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        enable_position_ids: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.chat_template = chat_template

        self.enable_position_ids = enable_position_ids

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id

    def get_output_dict(
        self,
        input_ids: torch.Tensor = None,
        input_tps: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        output = {
            "input_ids": input_ids,
            "input_tps": input_tps,
            "attention_mask": attention_mask,
        }

        if self.enable_position_ids:
            output["position_ids"] = torch.cumsum(attention_mask, dim=-1)

        return output

    def collect_fn(self, instances: List[Dict]) -> Dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        input_tps = torch.nn.utils.rnn.pad_sequence(
            [item["input_tps"] for item in instances],
            batch_first=True,
            padding_value=EnumTokenType.PAD.value,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in instances],
            batch_first=True,
            padding_value=0,
        )

        return self.get_output_dict(
            input_ids=input_ids,
            input_tps=input_tps,
            attention_mask=attention_mask,
        )

class UCDP_LLMQueryAnswer_WithPlaceholder(UCDP_LLMQueryAnswer_Base):
    """The partition for LLM query and answer

    Add QED placeholder before each question.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        length_que_end: int = 0,
        length_que_end_short: int = None,
        enable_position_ids: bool = False
    ):
        super().__init__(
            tokenizer=tokenizer,
            chat_template=chat_template,
            enable_position_ids=enable_position_ids,
        )

        self.length_que_end = length_que_end
        self.length_que_end_short = length_que_end_short
        
        self.parser = ChatQEDParser(
            tokenizer=self.tokenizer,
            length_que_end=self.length_que_end,
            length_que_end_short=self.length_que_end_short,
        )

    @torch.inference_mode()
    def get_item(self, chats: ChatRecord) -> dict:
        """Get the item for the LLM query and answer"""

        outputs = self.parser.parse(
            chats.apply_template(self.chat_template),
            enable_que_end=chats.has_image,
        )

        return self.get_output_dict(
            input_ids=outputs.input_ids,
            input_tps=outputs.input_tps,
            attention_mask=outputs.attention_mask,
        )

class UCDP_LLMQueryAnswer_WithIMGPlaceholder(UCDP_LLMQueryAnswer_Base):
    """
    This class will provide well formatted LLM input tokens.

    The Image token will be instered before the first question.

    For example:
    system prompts, ... `<|image_token|> * n` question 1 ...

    The length of instered image tokens is `n', which will be given by the `num_image_tokens` argument provided by UCDP_Image_Base.

    If the number `n' is not provided, the image tokens will be ignored.
    """
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        enable_position_ids: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            chat_template=chat_template,
            enable_position_ids=enable_position_ids,
        )

        self.parser = ChatIMGParser(
            tokenizer=self.tokenizer,
        )
    
    @torch.inference_mode()
    def get_item(self, chats: ChatRecord) -> dict:
        """Get the item for the LLM query and answer"""

        pased_results = chats.apply_template(self.chat_template)

        assert chats.num_img_tokens is not None, "num_image_tokens is not provided"

        if chats.has_image:
            pased_results.num_image_tokens = chats.num_img_tokens 

        outputs = self.parser.parse(
            pased_results,
            enable_que_end=chats.has_image,
        )

        return self.get_output_dict(
            input_ids=outputs.input_ids,
            input_tps=outputs.input_tps,
            attention_mask=outputs.attention_mask,
        )

# simple implement of dropped no image instances

@UCDPWraper_DropNoImageInstances.wrap(name_of_key="image_instance_index")
class UCDPWraped_Image(UCDP_Image):
    pass

@UCDPWraper_DropNoImageInstances.wrap(name_of_key="imagehd_instance_index")
class UCDPWraped_HDImage(UCDP_HDImage): 
    pass

@UCDPWraper_DropNoImageInstances.wrap(name_of_key="instance_index")
class UCDPWraped_LLMQueryAnswer_WithPlaceholder(UCDP_LLMQueryAnswer_WithPlaceholder):
    pass


@UCDPWraper_DropNoImageInstances.wrap(name_of_key="que_instance_index")
class UCDPWraped_CLIPQuery_WithPlaceholder(UCDP_CLIPQuery_WithPlaceholder):
    pass

@UCDPWraper_DropNoImageInstances.wrap(name_of_key="ans_instance_index")
class UCDPWraped_CLIPAnswer_WithPlaceholder(UCDP_CLIPAnswer_WithPlaceholder):
    pass

