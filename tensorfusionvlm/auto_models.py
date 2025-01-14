
from typing import Tuple, Union
import transformers.models.auto as auto_model
from collections import OrderedDict
import transformers
import os

from .model.modeling_query_adapt_clip import (
    ImageHDMultiQueryCLIPConfig,
    ImageHDMultiQueryCLIPModel,
    ImageMultiQueryCLIPConfig,
    ImageMultiQueryConfig,
    ImageMultiQueryCLIPModel,
    AutoModelOnVLMForImageMultiQuery,
)

from .model.image_processing_phi3_v import Phi3VImageProcessor

from .model.modeling_imghd_multi_que_llm import (
    ImageMultiQueryCLIPLLMConfig,
    ImageMultiQueryCLIPLLMModel,
)

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, CLIPImageProcessor, AutoTokenizer


class AutoModelForImageMultiQuery(auto_model.AutoModelForPreTraining):
    _model_mapping = OrderedDict({
        ImageMultiQueryCLIPConfig: ImageMultiQueryCLIPModel,
        ImageMultiQueryCLIPLLMConfig: ImageMultiQueryCLIPLLMModel,
        ImageHDMultiQueryCLIPConfig: ImageHDMultiQueryCLIPModel,
    })


# auto_model.AutoImageProcessor.register(
#     Im
# )
auto_model.AutoImageProcessor.register("Phi3VImageProcessor", Phi3VImageProcessor)


def load_tokenizers_processor(model_name_or_path: str) -> Tuple[
    Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None],
    CLIPImageProcessor
]:
    """
    Return:
    - llm_tokenizer
    - clip_tokenizer: if avaliable, else None
    - image_processor
    """

    llm_tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path)

    clip_tokenizer_path = os.path.join(model_name_or_path, "clip")

    if os.path.exists(clip_tokenizer_path):
        clip_tokenizer = AutoTokenizer.from_pretrained(
            clip_tokenizer_path)
        if clip_tokenizer.pad_token is None:
            clip_tokenizer.pad_token = clip_tokenizer.eos_token
    else:
        clip_tokenizer = None

    # image_processor = CLIPImageProcessor.from_pretrained(
    #     model_name_or_path)
    image_processor = auto_model.AutoImageProcessor.from_pretrained(
        model_name_or_path)

    assert isinstance(image_processor, CLIPImageProcessor) or isinstance(image_processor, Phi3VImageProcessor)

    return llm_tokenizer, clip_tokenizer, image_processor
