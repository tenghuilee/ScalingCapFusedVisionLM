import copy
import io
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import fastchat.conversation as fastchatConv
import torch
import transformers
from tokenizers import processors

import tensorfusionvlm.chat_template_extend as fastchatConvExtend
from tensorfusionvlm.constants import *
from tensorfusionvlm.constants import EnumTokenType
from tensorfusionvlm.model.image_processing_phi3_v import Phi3VImageProcessor
from tensorfusionvlm.utils import (ImageHelperWithPhi3Processor,
                                   ImageHelperWithProcessor)


class ChatRecordPasted:
    def __init__(self, num_image_tokens: int = 0):
        self.src_list = []  # type: List[str]
        self.tps_list = []  # type: List[EnumTokenType]
        self.num_image_tokens = num_image_tokens

    def append(self, src: str, token_type: EnumTokenType):
        self.src_list.append(src)
        self.tps_list.append(token_type)

    def get_src(self, token_type: EnumTokenType = None):
        if token_type is None:
            return [src for src in self.src_list]
        else:
            return [
                src for src, t in zip(self.src_list, self.tps_list) if (token_type.value & t.value) != 0
            ]

    def iter_src_tps(self, enum=False):
        index = 0
        for src, tp in zip(self.src_list, self.tps_list):
            if enum:
                yield index, src, tp
            else:
                yield src, tp
            index += 1

    def iter_src(self, enum=False, tps_filter: EnumTokenType = None):
        index = 0
        for src, tp in zip(self.src_list, self.tps_list):
            if tps_filter is not None and (tps_filter.value & tp.value) == 0:
                continue
            if enum:
                yield index, src
            else:
                yield src
            index += 1

    def __len__(self):
        return len(self.src_list)

    def __str__(self):
        return "".join(self.src_list)


@dataclass
class ChatConversation(fastchatConv.Conversation):

    def do_chat_parse(
        self,
        system_message: str = None,
        add_endoftext: bool = False,
    ) -> ChatRecordPasted:
        """
        get raw index_ids

        if image_tag is not none:
        detect image_tag in string and mark that place to EnumTokenType.IMG
        """

        pas = ChatRecordPasted()

        if system_message is None:
            system_message = self.system_message
        if system_message is None or system_message == "":
            system_prompt = None
        else:
            system_prompt = self.system_template.format(
                system_message=system_message)

        if self.sep_style == fastchatConv.SeparatorStyle.LLAMA2:
            # <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos><bos>[INST] Prompt [/INST] Answer <eos>
            # <bos>[INST] Prompt [/INST]
            pas.append("<s>", EnumTokenType.BOS)  # BOS; special token

            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        # system prompt + user query
                        if system_prompt is None:
                            pas.append(f"[INST] {message}", EnumTokenType.QUE)
                            pas.append(f"[/INST]", EnumTokenType.INST)
                        else:
                            pas.append(system_prompt, EnumTokenType.SYS)
                            pas.append(message, EnumTokenType.QUE)
                            pas.append(f"[/INST]", EnumTokenType.INST)
                    elif i % 2 == 1:
                        pas.append(f"{message} </s>", EnumTokenType.ANS)
                    else:
                        pas.append(f"<s>[INST]", EnumTokenType.INST)
                        pas.append(message, EnumTokenType.QUE)
                        pas.append(f"[/INST]", EnumTokenType.INST)
                else:
                    pass

        elif self.sep_style == fastchatConv.SeparatorStyle.ADD_COLON_TWO:
            # vicuna v1.1

            seps = [self.sep, self.sep2]

            pas.append("<s>", EnumTokenType.BOS)  # BOS; special token
            if system_prompt:
                pas.append(system_prompt + seps[0], EnumTokenType.SYS)
            else:
                pas.append(seps[0], EnumTokenType.SYS)

            for i, (role, message) in enumerate(self.messages):
                if message:
                    pas.append(role + ": ", EnumTokenType.INST)
                    if i % 2 == 0:
                        pas.append(message, EnumTokenType.QUE)
                        pas.append(seps[0], EnumTokenType.INST)
                        # NO BOS;
                        # _parser.append(EnumTokenType.BOS, seps[0])
                    else:
                        pas.append(message, EnumTokenType.ANS)
                        pas.append(seps[1], EnumTokenType.EOS)
                else:
                    pas.append(role + ":", EnumTokenType.INST)

        elif self.sep_style == fastchatConv.SeparatorStyle.CHATML:
            # qwen-7b-chat; Yi-34b-chat; ...
            eos = self.sep + "\n"
            if system_prompt:
                pas.append(system_prompt + eos, EnumTokenType.SYS)

            for i, (role, message) in enumerate(self.messages):
                if message:
                    pas.append(role + "\n", EnumTokenType.INST)
                    if i % 2 == 0:
                        pas.append(message, EnumTokenType.QUE)
                        pas.append(eos, EnumTokenType.INST)  # <|im_end|>
                    else:
                        pas.append(message, EnumTokenType.ANS)
                        pas.append(eos, EnumTokenType.EOS)  # <|im_end|>
                else:
                    pas.append(role + "\n", EnumTokenType.INST)

        elif self.sep_style == fastchatConvExtend.SeparatorStyleExtend.PHI_3:
            # phi-3
            # {% for message in messages %}
            #   {% if message['role'] == 'system' %}
            #       {{'<|system|>\n' + message['content'] + '<|end|>\n'}}
            #   {% elif message['role'] == 'user' %}
            #       {{'<|user|>\n' + message['content'] + '<|end|>\n'}}
            #   {% elif message['role'] == 'assistant' %}
            #       {{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}
            #   {% endif %}
            # {% endfor %}
            # {% if add_generation_prompt %}
            #   {{ '<|assistant|>\n' }}
            # {% else %}
            #   {{ eos_token }}
            # {% endif %}"

            # <|system|>
            # You are a helpful assistant.<|end|>
            # <|user|>
            # Can you solve the equation 2x + 3 = 7?<|end|>
            # <|assistant|>
            # The solution to the equation 2x + 3 = 7 is x = 1.<|end|>
            # <|endoftext|>

            if system_prompt:
                pas.append(system_prompt, EnumTokenType.SYS)
            for i, (role, message) in enumerate(self.messages):
                if message:
                    pas.append(role + "\n", EnumTokenType.INST)
                    if i % 2 == 0:
                        pas.append(message, EnumTokenType.QUE)
                    else:
                        pas.append(message, EnumTokenType.ANS)
                    # EOS
                    pas.append("<|end|>\n", EnumTokenType.EOS)
                else:
                    pas.append(role + "\n", EnumTokenType.INST)

            if add_endoftext:
                pas.append("<|endoftext|>", EnumTokenType.EOS)

        elif self.sep_style == fastchatConvExtend.SeparatorStyleExtend.LLAMA_3:
            # llama-3
            # {% set loop_messages = messages %}
            # {% for message in loop_messages %}
            #   {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}
            #   {% if loop.index0 == 0 %}
            #       {% set content = bos_token + content %}
            #   {% endif %}
            #   {{ content }}
            # {% endfor %}
            # {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
            pas.append("<|begin_of_text|>", EnumTokenType.BOS)
            if system_prompt:
                pas.append(system_prompt, EnumTokenType.SYS)

            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i % 2 == 0:
                        pas.append(role + "\n\n", EnumTokenType.INST)
                        pas.append(message, EnumTokenType.QUE)
                        pas.append("<|eot_id|>", EnumTokenType.INST)
                    else:
                        pas.append(role + "\n\n", EnumTokenType.INST)
                        pas.append(message, EnumTokenType.ANS)
                        pas.append("<|eot_id|>", EnumTokenType.EOS)
                else:
                    pas.append(role + "\n\n", EnumTokenType.INST)
            if add_endoftext:
                pas.append(self.roles[1] + "\n\n", EnumTokenType.INST)
        else:
            raise NotImplementedError()

        return pas


def get_conv_template(chat_template: Union[str | fastchatConv.Conversation | ChatConversation], **kwargs):
    if isinstance(chat_template, str):
        conversation = fastchatConv.get_conv_template(chat_template)
        if chat_template == "llama-2":
            # special fixed for llama-2
            conversation.sep = "<s>"
            conversation.sep2 = "</s>"
    elif isinstance(chat_template, fastchatConv.Conversation):
        conversation = chat_template
    elif isinstance(chat_template, ChatConversation):
        return chat_template
    else:
        raise ValueError(
            "chat_template must be a string or a Conversation object")
    data_dict = conversation.__dict__.copy()
    data_dict.update(kwargs)
    return ChatConversation(**data_dict)


@dataclass
class ChatRecordBase:
    """
    Requires:
    - image(str): the path of the image
    - messages(list): the list of conversations, where
        0, 2, 4, ...: even index is user
        1, 3, 5, ...: odd index is assistant
    - system_message(str): the system message

    Please make sure that these two variable is setted before 
    - image_size(tuple): the image size; writen elsewhere
    - num_img_tokens(int): the number of image tokens; writen elsewhere
    """

    image: str = field(default=None, metadata={"help": "Image path"})
    messages: List[str] = field(
        default_factory=list, metadata={"help": "messages"})
    system_message: str = None

    image_size: Optional[Tuple[int, int]] = field(default=None, metadata={"help": "Image size"})
    num_img_tokens: Optional[int] = field(default=None, metadata={"help": "Number of image tokens"})

    @classmethod
    def from_json(cls, src: str):
        return cls.from_dict(json.loads(src))

    @classmethod
    def from_dict(cls, src: dict):
        raise NotImplementedError("Please implement this method in subclass")

    @property
    def has_image(self):
        return self.image is not None

    def get_image_path(self, image_folder: str):
        if self.image is None:
            return None
        return os.path.join(image_folder, self.image)

    def to_dict(self):
        raise NotImplementedError("Please implement this method in subclass")

    def to_json(self, indent=4):
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self):
        return self.to_json()

    def load_image(
        self,
        image_aspect_ratio: str = "pad",
        processor: transformers.CLIPImageProcessor = None,
        image_folder: str = None,
    ):
        if self.image is None:
            return None
        return self.load_image_by_helper(
            ImageHelperWithProcessor(processor, image_aspect_ratio),
            image_folder=image_folder,
        )

    def load_image_by_helper(
        self,
        img_loader: ImageHelperWithProcessor,
        image_folder: str = None,
    ):
        if self.image is None:
            return None

        if image_folder is None:
            return img_loader.load_image(self.image)
        else:
            return img_loader.load_image(image_folder, self.image)

    def apply_template(
        self,
        chat_template: Union[str, ChatConversation],
        system_message: str = None,
        add_endoftext: bool = False,
    ) -> ChatRecordPasted:
        """
        - conversation_template: the conversation template form llava 
            if is NONE; use `self.default_chat_template` 
            Currently support:
            - llama-2
            - llama-3
            - vicuna_v1.1
            - microsoft-phi-3
            - qwen-7b-chat
        """

        chat_template = get_conv_template(chat_template)

        chat_template.messages = []
        for i, chats in enumerate(self.messages):
            chat_template.append_message(chat_template.roles[i % 2], chats)

        return chat_template.do_chat_parse(
            system_message=system_message or self.system_message,
            add_endoftext=add_endoftext,
        )

    def apply(self) -> ChatRecordPasted:
        out = ChatRecordPasted()
        roles = (EnumTokenType.QUE, EnumTokenType.ANS)
        for i, chats in enumerate(self.messages):
            out.append(chats, roles[i % 2])
        return out


@dataclass
class ChatRecord(ChatRecordBase):
    """
    Please use from_dict or from_json, and avoid to use __init__()
    ```
    {
        "image": "image_name",
        "messages": [
            {
                "role": "user",
                "content": "user's message",
            },
            {
                "role": "assistant",
                "content": "assistant's message",
            },
            ...
        ]
    }
    ```
    no image format
    ```
    {
        "image": None,
        "messages": [
            {
                "role": "user",
                "content": "user's message",
            },
            {
                "role": "assistant",
                "content": "assistant's message",
            },
            ...
        ]
    }
    ```
    """

    @classmethod
    def from_dict(cls, src: dict):
        image = src.get("image", None)  # type: Optional[str]
        if image is not None and image.lower() == "none":
            image = None
        messages = src.get("messages")  # type: list[dict[str, str]]

        out = ChatRecord(image=image)

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                assert i % 2 == 0
            elif role == "assistant":
                assert i % 2 == 1
            else:
                raise ValueError(f"Invalid role: {role}")

            out.messages.append(content)

        return out

    def to_dict(self):
        roles = ("user", "assistant")
        return {
            "image": self.image,
            "messages": [{
                "role": roles[i % 2],
                "content": conv,
            } for i, conv in enumerate(self.messages)]
        }


@dataclass
class ChatRecordSimple(ChatRecordBase):
    """
    [
        { "role": "image", "content": "path to image" },
        { "role": "system", "content": "You are a ..." },
        { "role": "user", "content": "What xxx?" },
        { "role": "assistant", "content": "I am a robot" },
        { "role": "user", "content": "What xxx?" },
        { "role": "assistant", "content": "I am a robot" },
    ]
    """

    @classmethod
    def from_dict(cls, src_list: list[dict[str, str]]):
        out = ChatRecordSimple()

        for line in src_list:
            role = line.get("role")
            content = line.get("content")
            _chat_round = len(out.messages) % 2
            if role == "image":
                out.image = content
            elif role == "system":
                out.system_message = content
            elif role == "user" and _chat_round == 0:
                out.messages.append(content)
            elif role == "assistant" and _chat_round == 1:
                out.messages.append(content)
            else:
                warnings.warn(
                    f"Conversation {line} is not valid, it will be ignored")
        return out

    def to_dict(self):
        out = []
        out.append({"role": "image", "content": self.image})
        out.append({"role": "system", "content": self.system_message})
        for i, conv in enumerate(self.messages):
            out.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": conv,
            })
        return out


@dataclass
class ChatRecordVLMEvla(ChatRecordBase):
    """
    Record for vlmeval VLMEvalKit
    [
        { "type": "image", "value": "path to image" },
        { "type": "text", "value": "You are a ..." },
        { "type": "text", "value": "What xxx?" },
        { "type": "text", "value": "I am a robot" },
        { "type": "text", "value": "What xxx?" },
    ]
    """

    @classmethod
    def from_dict(cls, src_list: list[dict[str, str]], add_endoftext=False):
        out = ChatRecordVLMEvla()

        for line in src_list:
            _type = line.get("type")
            value = line.get("value")
            if _type == "image":
                out.image = value
            elif _type == "text":
                out.messages.append(value)
            else:
                warnings.warn(
                    f"Conversation {line} is not valid, it will be ignored")

        if add_endoftext and len(out.messages) % 2 == 0:
            # 0, 2, 4, ...: query
            # 1, 3, 5, ...: response
            out.messages.append("")

        return out

    def to_dict(self):
        out = []
        out.append({"type": "image", "value": self.image})
        # out.append({"type": "text", "value": self.system_message})
        out.append({"type": "text", "value": conv} for conv in self.messages)
        return out


@dataclass
class ChatParserOutput:
    input_ids: torch.Tensor = None
    input_tps: torch.Tensor = None
    attention_mask: torch.Tensor = None


class ChatParserBase:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.pad_token_id = None
            self.eos_token_id = None
        else:
            self.pad_token_id = tokenizer.pad_token_id
            self.eos_token_id = tokenizer.eos_token_id
            if self.pad_token_id is None:
                self.pad_token_id = tokenizer.eos_token_id

    @torch.inference_mode()
    def parse_ids_tps(self, pasted_record: ChatRecordPasted, enable_que_end: bool = False):
        if self.tokenizer is None:
            return [], []
        raise NotImplementedError

    def to_string(self, pasted_record: ChatRecordPasted) -> str:
        return str(pasted_record)

    def parse(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
        max_seq_len: int = None,
    ) -> Union[ChatParserOutput, str]:
        if self.tokenizer is None:
            return str(self)

        ids, tps = self.parse_ids_tps(
            pasted_record=pasted_record,
            enable_que_end=enable_que_end,
        )

        if max_seq_len is not None:
            ids = ids[:max_seq_len]
            tps = tps[:max_seq_len]

        ids = torch.tensor(ids)
        tps = torch.tensor(tps)
        mask = tps.ne(EnumTokenType.PAD.value).int()
        return ChatParserOutput(input_ids=ids, input_tps=tps, attention_mask=mask)

class ChatQEDParserBase(ChatParserBase):
    """
    The base class for ChatQEDParser.

    Features:
    - init the `length_que_end' and `length_que_end_short'
    - set pad token
    - init placeholder tokens
    """
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        length_que_end: int = 0,
        length_que_end_short: int = None,
        append_special_padding: bool = False,
    ):
        super().__init__(tokenizer)

        self.append_special_padding = append_special_padding

        # -1 for eos
        self.length_que_end = max(length_que_end - 1, 0)
        if length_que_end_short is None:
            self.length_que_end_short = length_que_end
        else:
            self.length_que_end_short = max(length_que_end_short - 1, 0)

        assert self.length_que_end >= self.length_que_end_short, "length_que_end_short must be smaller than length_que_end"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if append_special_padding:
            self.placeholder_tokens = [
                f"<|placeholder_qed_{i}|>" for i in range(self.length_que_end)]

            self.tokenizer.add_tokens(
                self.placeholder_tokens, special_tokens=False)
            self.placeholder_tokens_ids = [self.tokenizer.convert_tokens_to_ids(
                toc) for toc in self.placeholder_tokens]
        else:
            self.placeholder_tokens = [
                self.tokenizer.pad_token for _ in range(self.length_que_end)]
            self.placeholder_tokens_ids = [
                self.tokenizer.pad_token_id for _ in range(self.length_que_end)]

        self.tps_qed_length_que_end = [
            EnumTokenType.QED.value] * self.length_que_end

class ChatQEDParser(ChatParserBase):
    """
    Parse chat records to well formatted sequence of tokens and token types.

    Parsed example:
    ```
    <<SYS>>You are a helpful assistant.<</SYS>> User: <|pad|>*n  ...
    ```
    The parse results are more suitable for the LLM.
    """
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        length_que_end: int = 0,
        length_que_end_short: int = None,
    ):
        super().__init__(tokenizer)

        self.length_que_end = length_que_end
        if length_que_end_short is None:
            self.length_que_end_short = length_que_end
        else:
            self.length_que_end_short = length_que_end_short

    def to_string(self, pasted_record: ChatRecordPasted) -> str:
        out_str = io.StringIO()
        index = 0
        for src, tps in pasted_record.iter_src_tps():
            if (self.length_que_end) > 0 and (tps == EnumTokenType.QUE):
                if index == 0:
                    out_str.write(f"<|qed len={self.length_que_end}|>")
                else:
                    out_str.write(f"<|qed len={self.length_que_end_short}|>")
                index += 1
            out_str.write(src)
        return out_str.getvalue()

    @torch.inference_mode()
    def parse_ids_tps(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
    ):
        """
        Return input_ids and input_tps
        """
        input_ids = self.tokenizer(
            pasted_record.get_src(),
            add_special_tokens=False,  # must be False
        ).input_ids

        out_ids = []
        out_tps = []

        _que_index = 0
        for ids, tps in zip(input_ids, pasted_record.tps_list):

            if enable_que_end and (tps == EnumTokenType.QUE):
                if _que_index == 0:
                    out_ids.extend([self.pad_token_id] * self.length_que_end)
                    out_tps.extend([EnumTokenType.QED.value]
                                   * self.length_que_end)
                else:
                    out_ids.extend([self.pad_token_id] *
                                   self.length_que_end_short)
                    out_tps.extend([EnumTokenType.QED.value]
                                   * self.length_que_end_short)
                _que_index += 1

            out_ids.extend(ids)
            out_tps.extend([tps.value] * len(ids))

        return out_ids, out_tps

class ChatIMGParser(ChatParserBase):
    """
    Parse chat records to well formatted sequence of tokens and token types.

    Parsed example:
    ```
    <<SYS>>You are a helpful assistant.<</SYS>> User: <|image|>*n  ...
    ```
    The parse results are more suitable for the LLM.
    """
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__(tokenizer)

        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        if not isinstance(image_token_id, int):
            # the token <|image|> is not in the tokenizer
            # use the <|eos|> token to replace
            image_token_id = self.tokenizer.eos_token_id

        self.image_token_id = image_token_id
    
    def to_string(self, pasted_record: ChatRecordPasted) -> str:
        out_str = io.StringIO()
        index = 0
        for src, tps in pasted_record.iter_src_tps():
            if (pasted_record.num_image_tokens) > 0 and (tps == EnumTokenType.QUE):
                if index == 0:
                    out_str.write(f"<|image|>*{pasted_record.num_image_tokens}")
                index += 1
            out_str.write(src)
        return out_str.getvalue()

    @torch.inference_mode()
    def parse_ids_tps(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False, 
    ):
        """
        Return input_ids and input_tps

        if enable_que_end:
            If pasted_record.num_image_tokens <= 0, 
            do not add the <|image|> tokens.
            Otherwise, add <|image|> * n tokens 
        """
        input_ids = self.tokenizer(
            pasted_record.get_src(),
            add_special_tokens=False,  # must be False
        ).input_ids

        out_ids = []
        out_tps = []

        num_image_tokens = pasted_record.num_image_tokens

        _que_index = 0
        for ids, tps in zip(input_ids, pasted_record.tps_list):

            if enable_que_end and (tps == EnumTokenType.QUE):
                if _que_index == 0:
                    out_ids.extend([self.image_token_id] * num_image_tokens)
                    out_tps.extend([EnumTokenType.IMG.value] * num_image_tokens)
                # else, do nothing
                _que_index += 1

            out_ids.extend(ids)
            out_tps.extend([tps.value] * len(ids))

        return out_ids, out_tps

class ChatQEDQueryParser(ChatQEDParserBase):

    def to_string(self, pasted_record: ChatRecordPasted):
        out_str = io.StringIO()
        for i, src in pasted_record.iter_src(enum=True, tps_filter=EnumTokenType.QUE):
            out_str.write(src)
            if i == 0:
                out_str.write(f"<|qed len={self.length_que_end}|>")
            else:
                out_str.write(f"<|qed len={self.length_que_end_short}|>")
            out_str.write("<|eos|>\n")
        return out_str.getvalue()

    @torch.inference_mode()
    def parse_ids_tps(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
    ):
        """
        if enable_que_end:
            que que que <|qed|>...<|qed|><|eos|>
        """

        input_ids = self.tokenizer(
            pasted_record.get_src(EnumTokenType.QUE),
            add_special_tokens=False,  # must be False
        ).input_ids

        out_ids = []
        out_tps = []

        _que_index = 0
        for ids in input_ids:
            out_ids.extend(ids)
            out_tps.extend([EnumTokenType.QUE.value] * len(ids))

            if enable_que_end:
                if _que_index == 0:
                    out_ids.extend(self.placeholder_tokens_ids)
                    out_tps.extend(self.tps_qed_length_que_end)
                    # out_ids.extend([self.pad_token_id] * self.length_que_end)
                    # out_tps.extend([EnumTokenType.QED.value] * self.length_que_end)
                else:
                    out_ids.extend(
                        self.placeholder_tokens_ids[0:self.length_que_end_short])
                    out_tps.extend(
                        self.tps_qed_length_que_end[0:self.length_que_end_short])
                    # out_ids.extend([self.pad_token_id] * self.length_que_end_short)
                    # out_tps.extend([EnumTokenType.QED.value] * self.length_que_end_short)

                _que_index += 1
                out_ids.append(self.eos_token_id)
                out_tps.append(EnumTokenType.EOS.value)

        return out_ids, out_tps

class ChatQEDDropQueryParser(ChatQEDQueryParser):
    """
    For Debug:
    Use with caution.

    All query will be replace by one debug placeholder.
    """

    def __init__(self, replaced_query="", fill_ends: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if replaced_query is None:
            replaced_query = ""
        if fill_ends is None:
            fill_ends = 0

        self.fill_ends = fill_ends
        
        self.replaced_query = replaced_query
        if replaced_query == "":
            self.replaced_query_ids = []
        else:
            self.replaced_query_ids = self.tokenizer(
                replaced_query, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def parse_ids_tps(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
    ):
        out_ids = []
        out_tps = []

        _que_index = 0
        for _ in pasted_record.iter_src(tps_filter=EnumTokenType.QUE):
            if len(self.replaced_query_ids) > 0:
                out_ids.extend(self.replaced_query_ids)
                out_tps.extend([EnumTokenType.QUE.value] * len(self.replaced_query_ids))

            if enable_que_end:
                if _que_index == 0:
                    out_ids.extend(self.placeholder_tokens_ids)
                    out_tps.extend(self.tps_qed_length_que_end)
                else:
                    out_ids.extend(
                        self.placeholder_tokens_ids[0:self.length_que_end_short])
                    out_tps.extend(
                        self.tps_qed_length_que_end[0:self.length_que_end_short])

                _que_index += 1
                out_ids.append(self.eos_token_id)
                out_tps.append(EnumTokenType.EOS.value)
        
        if self.fill_ends > 0:
            out_ids.extend([self.eos_token_id] * self.fill_ends)
            out_tps.extend([EnumTokenType.PAD.value] * self.fill_ends)

        return out_ids, out_tps

class ChatQEDAnswerParser(ChatQEDParserBase):

    def to_string(self, pasted_record: ChatRecordPasted) -> str:
        out = io.StringIO()
        for i, src in pasted_record.iter_src(enum=True, tps_filter=EnumTokenType.ANS):
            out.write(src)
            if i == 0:
                out.write(f"<|qed len={self.length_que_end}|>")
            else:
                out.write(f"<|qed len={self.length_que_end_short}|>")
            out.write("<|eos|>\n")
        return out.getvalue()

    @torch.inference_mode()
    def parse_ids_tps(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
    ):
        input_ids = self.tokenizer(
            pasted_record.get_src(EnumTokenType.ANS),
            add_special_tokens=False,  # must be False
        ).input_ids

        out_ids = []
        out_tps = []

        for i, ids in enumerate(input_ids):
            _ids = []
            _tps = []

            _ids.extend(ids)
            _tps.extend([EnumTokenType.ANS.value] * len(ids))
            if enable_que_end:
                if i == 0:
                    _ids.extend(self.placeholder_tokens_ids)
                    _tps.extend(self.tps_qed_length_que_end)
                else:
                    _ids.extend(
                        self.placeholder_tokens_ids[0:self.length_que_end_short])
                    _tps.extend(
                        self.tps_qed_length_que_end[0:self.length_que_end_short])

            _ids.append(self.eos_token_id)
            _tps.append(EnumTokenType.EOS.value)

            out_ids.append(_ids)
            out_tps.append(_tps)

        return out_ids, out_tps

    @torch.inference_mode()
    def parse(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
        max_seq_len: int = None,
    ) -> ChatParserOutput | str:
        ids, tps = self.parse_ids_tps(
            pasted_record=pasted_record, enable_que_end=enable_que_end)

        ids = [torch.tensor(_i) for _i in ids]
        tps = [torch.tensor(_i) for _i in tps]
        att_mask = [torch.ones_like(_i) for _i in ids]

        _ids = torch.nn.utils.rnn.pad_sequence(
            ids, batch_first=True, padding_value=self.pad_token_id)

        _tps = torch.nn.utils.rnn.pad_sequence(
            tps, batch_first=True, padding_value=EnumTokenType.PAD.value)

        att_mask = torch.nn.utils.rnn.pad_sequence(
            att_mask, batch_first=True, padding_value=0)

        if max_seq_len is not None:
            _ids = _ids[:, :max_seq_len]
            _tps = _tps[:, :max_seq_len]
            att_mask = att_mask[:, :max_seq_len]

        return ChatParserOutput(
            input_ids=_ids,
            input_tps=_tps,
            attention_mask=att_mask,
        )


class ChatAnswerParser(ChatParserBase):
    """
    Parse answers into ids and tps
    No que_end
    """
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
    ):
        super().__init__(tokenizer=tokenizer)

        if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            # buggy in fast model
            _tok = copy.deepcopy(tokenizer)
            _tok._tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
                pair=f"{tokenizer.bos_token} $A {tokenizer.eos_token} {tokenizer.bos_token} $B {tokenizer.eos_token}",
                special_tokens=[
                    (tokenizer.bos_token, tokenizer.bos_token_id),
                    (tokenizer.eos_token, tokenizer.eos_token_id),
                ],
            )
            self.tokenizer = _tok

    @torch.inference_mode()
    def parse_ids_tps(
        self,
        pasted_record: ChatRecordPasted,
        enable_que_end: bool = False,
        max_seq_len: int = None,
    ):
        if enable_que_end:
            warnings.warn("ChatAnswerParser does not support que_end")
            enable_que_end = False

        parse_outputs = self.tokenizer(
            pasted_record.get_src(EnumTokenType.ANS),
            add_special_tokens=True,
            max_length=max_seq_len,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        ids = parse_outputs.input_ids
        msk = parse_outputs.attention_mask

        tps = torch.full_like(ids, fill_value=EnumTokenType.ANS.value)
        tps[ids == self.tokenizer.eos_token_id] = EnumTokenType.EOS.value
        tps[msk == 0] = EnumTokenType.PAD.value

        # set eos token
        return ids, tps, parse_outputs.attention_mask

    def to_string(self, pasted_record: ChatRecordPasted) -> str:
        out = io.StringIO()
        for src in pasted_record.iter_src(tps_filter=EnumTokenType.ANS):
            out.write(src)
            out.write("\n")
        return out.getvalue()

    def parse(self, pasted_record: ChatRecordPasted, enable_que_end: bool = False, max_seq_len: int = None) -> ChatParserOutput | str:
        ids, tps, msk = self.parse_ids_tps(
            pasted_record=pasted_record, enable_que_end=enable_que_end, max_seq_len=max_seq_len)
        
        return ChatParserOutput(
            input_ids=ids,
            input_tps=tps,
            attention_mask=msk,
        )

class UtilChatAndQueryParser:
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        query_tokenizer: transformers.AutoTokenizer,
        image_processor: Union[transformers.CLIPImageProcessor,
                               Phi3VImageProcessor] = None,
        image_aspect_ratio: str = "pad",
        length_que_end: int = None,
        length_que_end_short: int = None,
        append_special_padding: bool = False,
        _qed_replaced_query: Optional[str] = None,
    ):
        """
        A helper class to parse chat and query.

        Required:
        - tokenizer: tokenizer for chat
        - query_tokenizer: tokenizer for query
        - image_processor: image processor for image

        Optional:
        - image_aspect_ratio: aspect ratio for image. Only support pad.
        - length_que_end: length of que_end. If None, disable que_end.
        - length_que_end_short: length of que_end_short. If None, disable que_end_short.
        - append_special_padding: append special padding to the end of the sequence.
        - _qed_query_parser_debug_placeholder: 
            use with caution. see `ChatQEDQueryParserDebug' for details.
            placeholder for debug. If None, disable debug.
        """

        if query_tokenizer is None:
            query_tokenizer = tokenizer

        self.chat_parser = ChatQEDParser(
            tokenizer=tokenizer,
            length_que_end=length_que_end,
            length_que_end_short=length_que_end_short,
        )

        self.is_hd_model = False

        if image_processor is not None:
            if isinstance(image_processor, transformers.CLIPImageProcessor):
                self.is_hd_model = False
                self.image_helper = ImageHelperWithProcessor(
                    processor=image_processor,
                    image_aspect_ratio=image_aspect_ratio,
                )
            elif isinstance(image_processor, Phi3VImageProcessor):
                self.is_hd_model = True
                self.image_helper = ImageHelperWithPhi3Processor(
                    processor=image_processor,
                )
                # aspect_ratio='pad'; always
        else:
            self.image_helper = None

        if _qed_replaced_query is None:
            self.query_parser = ChatQEDQueryParser(
                tokenizer=query_tokenizer,
                length_que_end=length_que_end,
                length_que_end_short=length_que_end_short,
                append_special_padding=append_special_padding,
            )
        else:
            self.query_parser = ChatQEDDropQueryParser(
                replaced_query=_qed_replaced_query,
                tokenizer=query_tokenizer,
                length_que_end=length_que_end,
                length_que_end_short=length_que_end_short,
                append_special_padding=append_special_padding,
            )

    def parse_record(
        self,
        record: ChatRecordBase,
        conv_template: str,
        system_message: str = None,
        add_endoftext: bool = False,
        enable_que_end: bool = False,
        image_folder: str = None,
        dtype=None,
        device=None,
    ):
        pasted_record = record.apply_template(
            conv_template,
            system_message=system_message,
            add_endoftext=add_endoftext,
        )

        chat_outputs = self.chat_parser.parse(
            pasted_record=pasted_record,
            enable_que_end=enable_que_end,
        )

        query_outputs = self.query_parser.parse(
            pasted_record=pasted_record,
            enable_que_end=enable_que_end,
        )

        out = {
            "input_ids": chat_outputs.input_ids.unsqueeze(0).to(device=device),
            "input_tps": chat_outputs.input_tps.unsqueeze(0).to(device=device),
            "attention_mask": chat_outputs.attention_mask.unsqueeze(0).to(device=device),
            "que_input_ids": query_outputs.input_ids.unsqueeze(0).to(device=device),
            "que_input_tps": query_outputs.input_tps.unsqueeze(0).to(device=device),
            "que_attention_mask": query_outputs.attention_mask.unsqueeze(0).to(device=device),
        }

        if self.image_helper is not None:
            if self.is_hd_model:
                img = self.image_helper.load_PIL_image(
                    image_root=image_folder,
                    image_name=record.image,
                )

                pad_img, shape_info = self.image_helper.image_expand2square(
                    img)

                pout = self.image_helper.preprocess(
                    image=img,
                    pad_image=pad_img,
                    return_tensors='pt',
                )

                cout = self.image_helper.drop_empty_crops(pout)

                out["pixel_values"] = cout["pixel_values"].to(
                    dtype=dtype, device=device)
                out["crop_index"] = cout["crop_index"].to(device=device)
            else:
                out["pixel_values"] = record.load_image_by_helper(
                    self.image_helper,
                    image_folder=image_folder,
                ).unsqueeze(0).to(dtype=dtype, device=device)

        return out

