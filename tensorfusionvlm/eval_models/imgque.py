import copy
from typing import Optional
import warnings

import torch
import transformers
import vlmeval
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import isimg, listinstr

from tensorfusionvlm import data_utils
from tensorfusionvlm.auto_models import AutoModelForImageMultiQuery, load_tokenizers_processor


class ImageMultiQueryCLIPLLMModel(vlmeval.BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        conv_template: str,
        _qed_replaced_query: Optional[str] = None,
        **kwargs,
    ):
        """
        Wrapper model for vlmeval 
        Args:
            model_path (str): path to the model
            conv_template (str): conversation template
            _qed_replaced_query (str):
                placeholder for query parser debug
                Please keep this parameter to None if you are not sure.
                When string is parsed, all que queries will be replaced with this string.
            **kwargs (dict): additional arguments
        """
        assert model_path is not None
        self.conv_template = conv_template
        self.system_prompt = kwargs.pop('system_prompt', None)

        llm_tokenizer, clip_tokenizer, image_processor = load_tokenizers_processor(model_path)
        if clip_tokenizer.pad_token is None:
            clip_tokenizer.pad_token = clip_tokenizer.eos_token
        self.llm_tokenizer = llm_tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.image_processor = image_processor
        self.model = AutoModelForImageMultiQuery.from_pretrained(
            model_path,
            device_map='cuda',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ) # type: transformers.PreTrainedModel

        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            num_return_sequences=1,
            use_cache=True,
            output_hidden_states=True,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )

        main_config = self.model.config

        self.parser_util = data_utils.UtilChatAndQueryParser(
            tokenizer=llm_tokenizer,
            query_tokenizer=clip_tokenizer,
            image_processor=image_processor,
            length_que_end=main_config.length_que_end,
            length_que_end_short=main_config.length_que_end_short,
            append_special_padding=main_config.append_special_padding,
            _qed_replaced_query=_qed_replaced_query,
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"Using kwargs: {self.kwargs}")
        torch.cuda.empty_cache()

    def adjust_kwargs(self, dataset):
        kwargs = copy.deepcopy(self.kwargs)
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 32
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 100
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 10
        elif listinstr(["seedbench"], dataset):
            kwargs['max_new_tokens'] = 3
        return kwargs
        
    def generate_inner(self, message: list[dict], dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs
        
        record = data_utils.ChatRecordVLMEvla.from_dict(
            message, add_endoftext=True)

        parser_output = self.parser_util.parse_record(
            record,
            conv_template=self.conv_template,
            system_message=self.system_prompt,
            enable_que_end=record.has_image,
            dtype=torch.bfloat16,
            device=self.model.device,
        )

        input_token_len = parser_output['input_ids'].shape[-1]

        pred = self.model.generate(
            **parser_output,
            **kwargs,
        )

        answer = self.llm_tokenizer.decode(
            pred[0][input_token_len:].cpu(),
            skip_special_tokens=True,
        ).strip()

        return answer

