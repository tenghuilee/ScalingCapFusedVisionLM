import warnings
from abc import abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import transformers
import transformers.models.clip.modeling_clip as modeling_clip
import transformers.models.llama.modeling_llama as modeling_llama
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          GenerationConfig, LlamaForCausalLM,
                          LogitsProcessorList, PreTrainedModel,
                          StoppingCriteriaList)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

from tensorfusionvlm.model.fusions import (FusionConcatBaseReturn,
                                           load_fusion_model, FusionConcatBase)


from tensorfusionvlm.model.modeling_query_adapt_clip import (
    ImageHDMultiQueryCLIPConfig,
    ImageMultiQueryCLIPConfig, 
    ImageMultiQueryConfig, 
    ModelForVLM,
    AutoModelOnVLMForImageMultiQuery,
    ImageMultiQueryOutputWithPooling,
    clip_loss,
    out_of_distribution_mask,
)

from tensorfusionvlm.utils import GlobalDebugUtil
from tensorfusionvlm.constants import *


class ImgQueCLIPLLMOutput(CausalLMOutputWithPast):
    pass

class ImageMultiQueryCLIPLLMConfig(transformers.PretrainedConfig):
    model_type = "modeling_query_adapt_clip_llm"

    def __init__(
        self,
        # model config
        imgque_model_name_or_path: Optional[str] = None,
        llm_model_name_or_path: Optional[str] = None,
        imgque_config: Optional[Union[ImageMultiQueryCLIPConfig, ImageMultiQueryConfig, ImageHDMultiQueryCLIPConfig]] = None,
        llm_config: Optional[modeling_llama.LlamaConfig] = None,
        fusion_name: Optional[str] = None,
        hidden_size: Optional[int] = None,
        # enable_que_end: bool = True, # always true
        length_que_end: int = 8,
        length_que_end_short: Optional[int] = None,
        append_special_padding: bool = False,
        replace_imgque_encoder_by_encoder_with_layer: Optional[int] = None,
        imgque_vocab_size: Optional[int] = None,
        **kwargs,
    ):
        self.imgque_model_name_or_path = imgque_model_name_or_path
        self.llm_model_name_or_path = llm_model_name_or_path
        self.fusion_name = fusion_name
        
        self.enable_que_end = True
        self.length_que_end = length_que_end
        self.length_que_end_short = length_que_end_short
        self.append_special_padding = append_special_padding
        self.replace_imgque_encoder_by_encoder_with_layer = replace_imgque_encoder_by_encoder_with_layer

        super().__init__(**kwargs)

        if imgque_config is None:
            if imgque_model_name_or_path is None:
                imgque_config = None
            else:
                imgque_config = AutoConfig.from_pretrained(
                    imgque_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                )
        else:
            if isinstance(imgque_config, dict):
                imgque_config = AutoConfig.for_model(**imgque_config) # type: ignore

        self.imgque_config = imgque_config
        if (imgque_config is not None) and (imgque_vocab_size is not None):
            self.imgque_config.imgque_config.vocab_size = imgque_vocab_size

        if llm_config is None:
            if llm_model_name_or_path is None:
                llm_config = None
            else:
                llm_config = AutoConfig.from_pretrained(
                    llm_model_name_or_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                )
        else:
            if isinstance(llm_config, dict):
                llm_config = AutoConfig.for_model(**llm_config) # type: ignore

        self.llm_config = llm_config
        if hidden_size is None:
            if self.llm_config is None:
                hidden_size = 4096
            else:
                hidden_size = self.llm_config.hidden_size
        self.hidden_size = hidden_size
    
class ImageMultiQueryCLIPLLMModel(PreTrainedModel):
    config_class = ImageMultiQueryCLIPLLMConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: ImageMultiQueryCLIPLLMConfig,
        imgque_model_from_pretrained=False,
        llm_model_from_pretrained=False,
    ):
        super().__init__(config)
        self.config = config

        if imgque_model_from_pretrained:
            if config.imgque_model_name_or_path is not None:
                self.imgque_backbone = AutoModelOnVLMForImageMultiQuery.from_pretrained(
                    config.imgque_model_name_or_path,
                    attn_implementation=config._attn_implementation,
                    torch_dtype=config.torch_dtype,
                ) # type: ModelForVLM
            else:
                self.imgque_backbone = AutoModelOnVLMForImageMultiQuery.from_config(
                    config.imgque_config,
                    attn_implementation=config._attn_implementation,
                    torch_dtype=config.torch_dtype,
                    clip_from_pretrained=True,
                ) # type: ModelForVLM
        else:
            self.imgque_backbone = AutoModelOnVLMForImageMultiQuery.from_config(
                config.imgque_config,
                attn_implementation=config._attn_implementation,
                torch_dtype=config.torch_dtype,
            ) # type: ModelForVLM

        if llm_model_from_pretrained:
            assert config.llm_model_name_or_path is not None
            self.llm_backbone = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name_or_path,
                attn_implementation=config._attn_implementation,
                torch_dtype=config.torch_dtype,
            ) # type: transformers.ModelForCausalLM
            config.llm_config = self.llm_backbone.config
        else:
            self.llm_backbone = AutoModelForCausalLM.from_config(
                config.llm_config,
                attn_implementation=config._attn_implementation,
                torch_dtype=config.torch_dtype,
            ) # type: transformers.ModelForCausalLM

        self.fusion_backbone = load_fusion_model(
            config.fusion_name,
            img_embed_size=config.imgque_config.hidden_size,
            img_embed_len=self.imgque_backbone.num_patches,
            txt_embed_size=config.llm_config.hidden_size,
            config=config
        ).to(config.torch_dtype) # type: FusionConcatBase

    @property
    def llm_input_embeds(self) -> nn.Embedding:
        return self.llm_backbone.get_input_embeddings()

    @property
    def llm_output_embeds(self):
        return self.llm_backbone.get_output_embeddings()

    # def load_state_dict(self, state_dict, strict = True, assign = False):
    #     print("\033[32mLoading state dict\033[0m")
    #     ans = super().load_state_dict(state_dict, strict, assign)
    #     generator = np.random.default_rng(1)
    #     imgque_model = self.imgque_backbone.imgque_model # 
    #     with torch.no_grad():
    #         # special init 
    #         for layer_idx in [16, 17, 18, 19]:
    #             _sub_model = imgque_model.layers[layer_idx] # type: nn.Module
    #             for _name, _weight in _sub_model.named_parameters():
    #                 if "proj" in _name:
    #                     _weight_np = generator.normal(0, scale=0.02, size=_weight.shape)
    #                     _weight.copy_(torch.from_numpy(_weight_np).to(dtype=_weight.dtype, device=_weight.device))
    #     return ans

    def forward(
        self,
        # text
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_tps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # image
        pixel_values: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        crop_index: Optional[torch.LongTensor] = None,
        # query
        que_input_ids: Optional[torch.LongTensor] = None,
        que_input_tps: Optional[torch.IntTensor] = None,
        que_attention_mask: Optional[torch.LongTensor] = None,
        # others
        return_loss: Optional[bool] = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # index
        instance_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, ImgQueCLIPLLMOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # prepare image embeds
        if vision_embeds is None:
            if pixel_values is None or pixel_values.size(0) == 0:
                warnings.warn(
                    "Both `pixel_values` and `vision_embeds` are None. Fallback to text mode."
                )
                out = self.llm_backbone.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    labels=input_ids if return_loss else None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
                return out
            else:
                # ! the batch size of vision_embeds could be less than that of input_ids
                # vision_embeds = self.imgque_backbone.get_image_embeds(pixel_values, keep_cls_patch=True)
                pass

        # prepare text embeds
        if inputs_embeds is None:
            inputs_embeds = self.llm_input_embeds(input_ids) # type: torch.Tensor

        if inputs_embeds.size(1) == 1:
            # inference mode 
            fusion_embeds = inputs_embeds
            # imgque_embeds = None # type: torch.Tensor
        else:
            # imgque
            imgque_outputs = self.imgque_backbone.forward(
                input_ids=que_input_ids,
                input_tps=que_input_tps,
                attention_mask=que_attention_mask,
                pixel_values=pixel_values,
                vision_embeds=None,
                crop_index=crop_index,
                return_dict=True,
            ) # type: ImageMultiQueryOutputWithPooling

            imgque_last_hidden_state = imgque_outputs.last_hidden_state # type: torch.Tensor

            # fusion
            fusion_embeds = self.fusion_backbone(
                input_embeds = inputs_embeds,
                input_tps = input_tps,
                imgque_embeds = imgque_last_hidden_state,
                que_input_tps = imgque_outputs.input_tps,
                instance_index=instance_index,
            )

        # llm
        llm_outputs = self.llm_backbone.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=fusion_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        loss = None
        if return_loss and (input_ids is not None) and (input_tps is not None):
            loss = self.fusion_backbone.compute_loss(llm_outputs.logits, input_ids, input_tps)        

        if not return_dict:
            outputs = llm_outputs[1:]
            return (loss,) + outputs if loss is not None else outputs
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=llm_outputs.logits,
            past_key_values=llm_outputs.past_key_values,
            hidden_states=llm_outputs.hidden_states,
            attentions=llm_outputs.attentions,
        )
        
    @torch.no_grad()
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        inputs, model_input_name, model_kwargs = super(
        )._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

        if inputs is not None:
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)

        input_tps = model_kwargs.get("input_tps", None)
        assert input_tps is not None, "input_tps must be provided"
        if input_tps.ndim == 1:
            input_tps = input_tps.unsqueeze(0)
            model_kwargs["input_tps"] = input_tps

        for key in [
            "input_ids",
            "input_tps",
            "que_input_ids",
            "que_input_tps",
            "attention_mask",
            "que_attention_mask",
        ]:
            if key not in model_kwargs:
                continue
            _var = model_kwargs[key]
            if _var is None:
                continue
            if _var.ndim == 1:
                _var = _var.unsqueeze(0)
                model_kwargs[key] = _var

        if model_kwargs.get("attention_mask", None) is None:
            model_kwargs["attention_mask"] = input_tps.ne(
                EnumTokenType.PAD.value).long()

        if model_kwargs.get("que_attention_mask", None) is None:
            que_input_tps = model_kwargs.get("que_input_tps", None)
            if que_input_tps is not None:
                model_kwargs["que_attention_mask"] = que_input_tps.ne(
                    EnumTokenType.PAD.value).long()

        pixel_values = model_kwargs.pop("pixel_values", None)
        if pixel_values is not None:
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            # TODO: this sould be carefully checked for HD Model
            if self.imgque_backbone.is_hd_model:
                model_kwargs["pixel_values"] = pixel_values
                assert "crop_index" in model_kwargs, "crop_index must be provided"
                # crop_index = model_kwargs["crop_index"] # type: torch.Tensor
                # if crop_index.ndim == 1:
                #     crop_index = crop_index.unsqueeze(0)
                # model_kwargs["crop_index"] = crop_index
            else:
                model_kwargs["vision_embeds"] = self.imgque_backbone.get_image_embeds(
                    pixel_values, keep_cls_patch=True)
        
        if "return_loss" in model_kwargs:
            # Do not need to compute loss
            model_kwargs["return_loss"] = False # type: ignore

        return inputs, model_input_name, model_kwargs 

    @torch.no_grad()
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_tps: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        # query
        que_input_ids: Optional[torch.LongTensor] = None,
        que_input_tps: Optional[torch.IntTensor] = None,
        que_attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # copy from huggingface transformers
        # rewrite this for `.generate` method
        # this will call many times
        past_length = 0

        # TODO:
        if past_key_values is not None:
            if not isinstance(past_key_values, transformers.cache_utils.Cache):
                past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

            cache_length = past_key_values.get_seq_length()
            assert past_key_values.seen_tokens is not None
            past_length = past_key_values.seen_tokens # type: int
            max_cache_length = past_key_values.get_max_length()
            # else:
            #     cache_length = past_length = past_key_values[0][0].shape[2]
            #     max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            assert input_ids is not None
            assert input_tps is not None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -
                                      (attention_mask.size(1) - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif EnumTokenType.IMG.value in input_tps:
                # TODO: fixed this later BUG
                input_ids = input_ids[:, input_ids.size(1)-1:input_ids.size(1)]
                input_tps = torch.zeros_like(
                    input_ids).fill_(EnumTokenType.ANS.value)
            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            assert input_ids is not None
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if past_key_value := getattr(self.llm_backbone.model.layers[0].self_attn, "past_key_value", None):
            # generation with static cache
            past_length = past_key_value.get_seq_length()
            assert input_ids is not None and position_ids is not None
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = kwargs.get("cache_position", None)
        if cache_position is None:
            assert position_ids is not None
            cache_position = torch.arange(
                past_length, past_length + position_ids.shape[-1], device=position_ids.device
            )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "input_tps": input_tps,
            "pixel_values": pixel_values,
            "crop_index": kwargs.get("crop_index", None),
            "vision_embeds": vision_embeds,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            "que_input_ids": que_input_ids,
            "que_input_tps": que_input_tps,
            "que_attention_mask": que_attention_mask,
        }) # type: ignore

        return model_inputs

AutoConfig.register("modeling_query_adapt_clip_llm",
                    ImageMultiQueryCLIPLLMConfig)
AutoModel.register(ImageMultiQueryCLIPLLMConfig, ImageMultiQueryCLIPLLMModel)

