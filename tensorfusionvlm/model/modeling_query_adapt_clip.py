from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import transformers
import transformers.modeling_outputs
import transformers.models.auto as auto_model
import transformers.models.clip.modeling_clip as modeling_clip
import transformers.models.llama.modeling_llama as modeling_llama
from deprecated import deprecated
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          PreTrainedModel)
from transformers.cache_utils import Cache

from tensorfusionvlm.constants import EnumTokenType

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html


def contrastive_loss(logits: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return torchF.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device), label_smoothing=label_smoothing)


def clip_loss(similarity: torch.Tensor, label_smoothing: float = 0.0, **kwargs) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, label_smoothing)
    image_loss = contrastive_loss(similarity.t(), label_smoothing)
    return (caption_loss + image_loss) / 2.0


def clip_loss_with_pooler_output(
    similarity: torch.Tensor, label_smoothing: float = 0.0,
    vision_pooler_output: torch.Tensor = None,
    text_pooler_output: torch.Tensor = None,
    pooler_output_weight: float = 1.0,
):
    return clip_loss(similarity, label_smoothing) + \
        pooler_output_weight * \
        torchF.mse_loss(vision_pooler_output, text_pooler_output)


@torch.no_grad()
def out_of_distribution_mask(input: torch.Tensor, sigma: float = 1.0):
    """
    1 std: 68.27%
    2 std: 95.45%
    3 std: 99.73%
    """
    x_std, x_mean = torch.std_mean(input)
    x_min = x_mean - sigma * x_std
    x_max = x_mean + sigma * x_std
    return torch.logical_or(input < x_min, input > x_max)


@torch.no_grad()
def in_the_distribution_mask(input: torch.Tensor, sigma: float = 1.0):
    """
    1 std: 68.27%
    2 std: 95.45%
    3 std: 99.73%
    """
    x_std, x_mean = torch.std_mean(input)
    x_min = x_mean - sigma * x_std
    x_max = x_mean + sigma * x_std
    return torch.logical_and(input > x_min, input < x_max)

###############################################################
# configures and outputs


class ModelingQueryAdaptCLIPModelConfig(modeling_clip.CLIPConfig):
    model_type = "modeling_query_align_clip"
    keys_to_ignore_at_inference = ["base_model_name_or_path"]

    def __init__(
        self,
        adapt_max_position_embeddings: int = 2048,
        hidden_size: int = 1024,
        base_name_or_path: str = None,
        align_num_hidden_layers: int = 32,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.base_name_or_path = base_name_or_path

        super().__init__(
            **kwargs,
        )

        if align_num_hidden_layers is None:
            self.align_num_hidden_layers = self.text_config.num_hidden_layers
        else:
            self.align_num_hidden_layers = align_num_hidden_layers

        self.query_align_config = modeling_llama.LlamaConfig(
            vocab_size=self.text_config.vocab_size,
            hidden_size=self.vision_config.hidden_size,
            intermediate_size=self.vision_config.intermediate_size,
            num_hidden_layers=self.align_num_hidden_layers,
            # num_attention_heads=self.vision_config.hidden_size//128,
            num_attention_heads=self.vision_config.hidden_size//64,
            # num_attention_heads=self.vision_config.hidden_size//32,
            max_position_embeddings=adapt_max_position_embeddings,
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            attn_implementation=self._attn_implementation,
            torch_dtype=self.torch_dtype,
            initializer_range=1e-3,
        )


class ImageMultiQueryCLIPConfig(modeling_clip.CLIPConfig):
    model_type = "modeling_image_multi_query_clip"
    keys_to_ignore_at_inference = ["base_model_name_or_path"]

    def __init__(
        self,
        max_position_embeddings: int = 2048,
        hidden_size: int = 1024,
        base_name_or_path: str = None,
        num_hidden_layers: int = None,
        vocab_size: int = None,
        length_que_end: int = None,
        length_que_end_short: int = None,
        append_special_padding: bool = False,
        **kwargs,
    ):
        self.base_name_or_path = base_name_or_path
        if base_name_or_path is not None:
            basic_config = transformers.AutoConfig.from_pretrained(
                base_name_or_path)
            basic_config_dict = basic_config.to_dict()
            basic_config_dict.update(kwargs)
        else:
            basic_config_dict = kwargs

        super().__init__(
            **basic_config_dict,
        )

        if vocab_size is None:
            self.vocab_size = self.text_config.vocab_size
        else:
            self.vocab_size = vocab_size

        if num_hidden_layers is None:
            self.num_hidden_layers = self.text_config.num_hidden_layers
        else:
            self.num_hidden_layers = num_hidden_layers

        self.length_que_end = length_que_end
        self.length_que_end_short = length_que_end_short
        self.append_special_padding = append_special_padding

        if self.pad_token_id is None:
            self.pad_token_id = self.text_config.pad_token_id
        if self.bos_token_id is None:
            self.bos_token_id = self.text_config.bos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id

        self.hidden_size = self.vision_config.hidden_size

        self.imgque_config = modeling_llama.LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.vision_config.hidden_size,
            intermediate_size=self.vision_config.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            # num_attention_heads=self.vision_config.hidden_size//128,
            num_attention_heads=self.vision_config.hidden_size//64,
            # num_attention_heads=self.vision_config.hidden_size//32,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            attn_implementation=self._attn_implementation,
            torch_dtype=self.torch_dtype,
        )


class CLIPVisionHDConfig(modeling_clip.CLIPVisionConfig):
    model_type = "clip_vision_hd"

    def __init__(
        self,
        enable_local_merge: bool = True,
        enable_token_crlf: bool = True,
        vembed_merge_kernel_size: int = 3,
        vembed_merge_stride: int = 2,
        vembed_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_token_crlf = enable_token_crlf
        self.enable_local_merge = enable_local_merge
        self.vembed_merge_kernel_size = vembed_merge_kernel_size
        self.vembed_merge_stride = vembed_merge_stride

        if vembed_hidden_size is None:
            vembed_hidden_size = self.hidden_size
        self.vembed_hidden_size = vembed_hidden_size


class ImageHDMultiQueryCLIPConfig(ImageMultiQueryCLIPConfig):
    model_type = "modeling_image_hd_multi_query_clip"
    keys_to_ignore_at_inference = ["base_model_name_or_path"]

    # the rest is the same as ImageMultiQueryCLIPConfig
    def __init__(
        self,
        enable_local_merge: bool = True,
        enable_token_crlf: bool = True,
        vembed_merge_kernel_size: int = 3,
        vembed_merge_stride: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_config = CLIPVisionHDConfig.from_dict(
            self.vision_config.to_dict())  # type: CLIPVisionHDConfig
        self.vision_config.enable_local_merge = enable_local_merge
        self.vision_config.enable_token_crlf = enable_token_crlf
        self.vision_config.vembed_merge_kernel_size = vembed_merge_kernel_size
        self.vision_config.vembed_merge_stride = vembed_merge_stride

        if self.vision_config.pad_token_id is None:
            self.vision_config.pad_token_id = self.pad_token_id
        if self.vision_config.bos_token_id is None:
            self.vision_config.bos_token_id = self.bos_token_id
        if self.vision_config.eos_token_id is None:
            self.vision_config.eos_token_id = self.eos_token_id


class ImageMultiQueryConfig(modeling_llama.LlamaConfig):
    model_type = "modeling_image_multi_query"

    def __init__(
        self,
        image_size: int = 14 * 8 * 4,  # the max image size
        patch_size: int = 14,  # the patch size
        num_channels: int = 3,
        text_num_hidden_layers: int = 8,
        num_hidden_layers: int = 16,
        hidden_size: int = 2048,  # 4096//2
        intermediate_size: int = 5504,  # 11008 // 2
        qed_repeated_count: int = 15,
        logit_scale_init_value: float = 4.605170,
        vocab_size: int = 32000,
        vision_embedding_scale: int = 1,
        # only avaliable is vision_embedding_scale > 1
        vision_embedding_mid_channels: int = 32,
        length_que_end: int = None,
        length_que_end_short: int = None,
        append_special_padding: bool = False,
        **kwargs,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_channels = num_channels

        self.text_num_hidden_layers = text_num_hidden_layers
        self.qed_repeated_count = qed_repeated_count
        self.logit_scale_init_value = logit_scale_init_value
        self.vision_embedding_scale = vision_embedding_scale
        self.vision_embedding_mid_channels = vision_embedding_mid_channels
        self.length_que_end = length_que_end
        self.length_que_end_short = length_que_end_short
        self.append_special_padding = append_special_padding

        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            **kwargs,
        )

    def get_text_config(self):
        out = self.from_dict(self.to_diff_dict())
        out.num_hidden_layers = self.text_num_hidden_layers
        return out


class ImageHDCLIPConfig(modeling_clip.CLIPConfig):
    model_type = "modeling_image_hd_clip"

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        hidden_size: int = 1024,
        vembed_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
        if base_model_name_or_path is not None:
            _conf = modeling_clip.CLIPVisionConfig.from_pretrained(
                base_model_name_or_path)
        else:
            _conf = self.vision_config

        self.vision_config = CLIPVisionHDConfig(
            vembed_hidden_size=vembed_hidden_size,
            **_conf.to_dict(),
        )  # type: CLIPVisionHDConfig


@dataclass
class QueryAdaptCLIPOutput(modeling_clip.CLIPOutput):
    imgque_embeds: Optional[torch.Tensor] = None


@dataclass
class ImageMultiQueryOutputWithPooling(transformers.modeling_outputs.ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    input_tps: torch.LongTensor = None
    attention_mask: torch.IntTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ImageHDMultiQueryOutput(transformers.modeling_outputs.ModelOutput):
    embeds: torch.FloatTensor = None
    input_tps: torch.LongTensor = None
    attention_mask: torch.LongTensor = None


@dataclass
class ImageMultiQueryCLIPOutput(modeling_clip.CLIPOutput):
    imgque_embeds: torch.FloatTensor = None
    imgque_model_output: ImageMultiQueryOutputWithPooling = None

##################################################################


class ImageQueryTransformer(modeling_llama.LlamaModel):
    supports_gradient_checkpointing = True

    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config)

        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

        if self.eos_token_id == 2:
            """
            Sorry, we do not support the eos_token_id=2 for now.
            Please refer the code of CLIP for more details.
            The original commend is:

            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            """
            raise ValueError("We do not support the eos_token_id=2 for now.")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, modeling_clip.BaseModelOutputWithPooling]:
        """

        Args:
        - input_ids: the ids of 'query'
        - vision_embeds: the vision embeds from vision encoder
        - attention_mask: the attention_mask of 'input_ids' !!!
        - output_attentions:
        - output_hidden_states:
        - return_dict:

        Returns:
        - the output of the CLIPTextTransformer
        """
        # update the input_embeds
        input_embeds = self.embed_tokens(input_ids)  # type: torch.Tensor
        input_embeds = torch.cat([
            vision_embeds,
            input_embeds,
        ], dim=1)

        bsz, patch_count = vision_embeds.shape[:2]

        # update attention mask
        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones(
                    (bsz, patch_count), device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask,
            ], dim=1)

        super_outputs = super().forward(
            input_ids=None,  # must set this to none
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = super_outputs[0]  # type: torch.Tensor

        with torch.no_grad():
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            first_position_of_eos_token = (input_ids.to(
                dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1)
            # !!!important `+ patch_count` in pooled_output; since the input_ids got shifted
            first_position_of_eos_token += patch_count

        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(bsz, device=last_hidden_state.device),
            first_position_of_eos_token,
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + super_outputs[1:]

        return modeling_clip.BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=super_outputs.hidden_states,
            attentions=super_outputs.attentions,
        )


class QueryAdaptCLIPModel(modeling_clip.CLIPPreTrainedModel):
    config_class = ModelingQueryAdaptCLIPModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ModelingQueryAdaptCLIPModelConfig):
        super().__init__(config)

        self.query_align_config = config.query_align_config
        self.projection_dim = config.projection_dim

        if config.base_name_or_path is not None:
            self.clip_model = modeling_clip.CLIPModel.from_pretrained(
                config.base_name_or_path, use_safetensors=True)
        else:
            self.clip_model = modeling_clip.CLIPModel(config)

        self.query_align_model = ImageQueryTransformer(self.query_align_config)
        self.query_align_projection = nn.Linear(
            self.query_align_config.hidden_size, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value))

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        que_ids: Optional[torch.LongTensor] = None,
        que_attention_mask: Optional[torch.Tensor] = None,
        ans_ids: Optional[torch.LongTensor] = None,
        ans_attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QueryAdaptCLIPOutput]:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # type: modeling_clip.BaseModelOutputWithPooling

        vision_last_hidden_state = vision_outputs[0]  # type: torch.Tensor

        # [1] is the pooled output
        if self.training:
            image_embeds = None
        else:
            image_embeds = vision_outputs[1]  # type: torch.Tensor
            image_embeds = self.clip_model.visual_projection(image_embeds)
            image_embeds = image_embeds / \
                image_embeds.norm(p=2, dim=-1, keepdim=True)

        if self.training and self.is_gradient_checkpointing:
            vision_last_hidden_state.requires_grad_(True)

        text_outputs = self.clip_model.text_model(
            input_ids=ans_ids,
            attention_mask=ans_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # type: modeling_clip.BaseModelOutputWithPooling
        text_embeds = text_outputs[1]  # type: torch.Tensor
        text_embeds = self.clip_model.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # compute query_align
        imgque_outputs = self.query_align_model(
            input_ids=que_ids,
            # last_hidden_state, remove the cls_patch
            vision_embeds=vision_last_hidden_state[:, 1:],
            attention_mask=que_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output
        imgque_embeds = imgque_outputs[1]  # type: torch.Tensor
        imgque_embeds = self.query_align_projection(imgque_embeds)
        imgque_embeds = imgque_embeds / \
            imgque_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(
            imgque_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            loss = modeling_clip.clip_loss(logits_per_image)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds,
                      image_embeds, text_outputs, vision_outputs, imgque_embeds)
            return ((loss,) + output) if loss is not None else output

        return QueryAdaptCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
            imgque_embeds=imgque_embeds,
        )

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Require:
        - input_ids: the input ids of the answer 
        - attention_mask: the attention mask of the answer

        Return:
        - text_features: the pooled features of the answer 
        """
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = text_outputs[1]  # type: torch.Tensor
        text_features = self.clip_model.text_projection(pooled_output)

        return text_features

    def get_ans_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """An alias of get_text_features()"""
        return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
        )

        pooled_output = vision_outputs[1]  # type: torch.Tensor
        image_features = self.clip_model.visual_projection(pooled_output)

        return image_features

    def get_imgque_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        # the last hidden state of the vision encoder, cls_patch included
        vision_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Require:
        - input_ids: the input ids of the query encoder
        - attention_mask: the attention mask of the query encoder
        - pixel_values: the pixel values of the image encoder
        - vision_embeds: the last hidden state of the vision encoder, cls_patch included
            if vision_embeds is None, pixel_values is required

        Return:
        - imgque_features: the pooled hidden state of the query encoder
        """

        imgque_outputs = self.get_imgque(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            vision_embeds=vision_embeds,
        )
        pooled_output = imgque_outputs[1]  # type: torch.Tensor
        imgque_features = self.query_align_projection(pooled_output)

        return imgque_features

    def get_imgque(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        # the last hidden state of the vision encoder, cls_patch included
        vision_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if vision_embeds is None:
            if pixel_values is None:
                raise ValueError(
                    "Either pixel_values or last_hidden_state has to be provided.")

            vision_outputs = self.clip_model.vision_model(
                pixel_values=pixel_values,
            )
            vision_embeds = vision_outputs[0]  # type: torch.Tensor

        return self.query_align_model(
            input_ids=input_ids,
            # the cls_patch is not included
            vision_embeds=vision_embeds[:, 1:],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionTransformerNoPool(nn.Module):
    """
    copy from transformers.models.clip.modeling_clip.CLIPVisionTransformer
    with post_layernorm removed
    """

    def __init__(self, config: modeling_clip.CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = modeling_clip.CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = modeling_clip.CLIPEncoder(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[Tuple | modeling_clip.BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        return self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionModelNoPool(modeling_clip.CLIPPreTrainedModel):
    """copy from transformers.models.clip.modeling_clip.CLIPVisionModel
    with CLIPVisionTransformer.post_layernorm removed
    """
    config_class = modeling_clip.CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: modeling_clip.CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformerNoPool(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, modeling_clip.BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionHDTransformer(nn.Module):
    config_class = CLIPVisionHDConfig

    """
    self.post_layernorm is removed
    """

    def __init__(self, config: CLIPVisionHDConfig):
        super().__init__()

        self.config = config

        embed_dim = config.hidden_size

        self.embeddings = modeling_clip.CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = modeling_clip.CLIPEncoder(config)
        self.pos_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # two special tokens for the \n token of gloabl image and sub images
        if config.enable_token_crlf:
            self.token_LF = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size), requires_grad=True)

        if config.enable_local_merge:
            self.num_sub_path = config.image_size // config.patch_size  # 24
            assert self.num_sub_path % 2 == 0, f"num_sub_path {self.num_sub_path} should be even"
            self.num_merged_patch = self.num_sub_path // 2

            self.vision_embed_merge = nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.vembed_hidden_size,
                kernel_size=config.vembed_merge_kernel_size,
                stride=config.vembed_merge_stride,
                padding=(config.vembed_merge_kernel_size -
                         1) // 2,  # same padding
                bias=False,
            )

        self.init_additional_weights()

    @torch.no_grad()
    def init_additional_weights(self):
        if self.config.enable_token_crlf:
            self.token_LF.normal_(std=self.config.initializer_range)
        if self.config.enable_local_merge:
            self.vision_embed_merge.weight.normal_(
                std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    def local_merge_tokens(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        """
        vision_embeds: (B, num_patches, hidden_size)
        """
        if not self.config.enable_local_merge:
            return vision_embeds

        # DO not add any normalization on vision_embeds, at here.
        # Since the output of ViT already has a layernorm

        B, _, hidden_size = vision_embeds.shape

        vision_embeds = (
            vision_embeds.reshape(
                B, self.num_sub_path, self.num_sub_path, hidden_size)
            .permute(0, 3, 1, 2)  # N, hidden_size, num_patches, num_patches
        )

        vision_embed_merge = self.vision_embed_merge(vision_embeds)

        vision_embed_down = torchF.interpolate(
            vision_embeds,
            size=(vision_embed_merge.shape[2], vision_embed_merge.shape[3]),
            mode="nearest",
        )

        vision_embeds = vision_embed_merge + vision_embed_down

        vision_embeds = (
            vision_embeds.reshape(B, hidden_size, -1)
            .permute(0, 2, 1)  # N, num_patches, hidden_size
        )
        return vision_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        drop_cls_patch: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> modeling_clip.BaseModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. image embedding
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # type: modeling_clip.BaseModelOutput

        # last hidden state
        last_hidden_state = self.pos_layrnorm(
            encoder_outputs[0])  # type: torch.Tensor
        cls_patch = last_hidden_state[:, 0:1]  # shape (B, hidden size)
        last_hidden_state = last_hidden_state[:, 1:]  # remove the cls_patch

        # shape (B, num_tokens, hidden size)
        if self.config.enable_local_merge:
            last_hidden_state = self.local_merge_tokens(last_hidden_state)

        # 2. add special tokens
        if self.config.enable_token_crlf:
            token_LF = self.token_LF.expand(
                last_hidden_state.shape[0], 1, -1).to(last_hidden_state.device)
            # shape (B, num_tokens + 1, hidden size)
            last_hidden_state = torch.cat([last_hidden_state, token_LF], dim=1)

        if not drop_cls_patch:
            last_hidden_state = torch.cat(
                [cls_patch, last_hidden_state], dim=1)

        if not return_dict:
            return last_hidden_state, encoder_outputs[1], encoder_outputs[2]

        return modeling_clip.BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionHDTransformerWithInputTPS(CLIPVisionHDTransformer):
    """This code insperied by Phi-3-Vision"""

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        crop_index: torch.LongTensor,
        inputs_embeds: torch.LongTensor,
        input_tps: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageHDMultiQueryOutput]:
        """
        Use crop_index to indicate the crop and its batch index.

        Please refer [utils.py](../utils.py@ImageHelperWithPhi3Processor.drop_empty_crops) for the details of the crop_index.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len, hidden_size = inputs_embeds.shape

        vision_outputs = super().forward(
            pixel_values=pixel_values,
            drop_cls_patch=True,
            return_dict=True,
        )  # type: modeling_clip.BaseModelOutput

        vision_embeds = vision_outputs.last_hidden_state
        num_tokens = vision_embeds.shape[1]

        if isinstance(crop_index, torch.Tensor):
            crop_index = crop_index.data.cpu().tolist()

        count_prefix = Counter(crop_index)
        max_count_prefix = max(count_prefix.values())
        # prefix + original size
        max_tps_len = max_count_prefix * num_tokens + seq_len

        out_embeds = torch.full(
            (batch_size, max_tps_len, hidden_size),
            fill_value=0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        _begin, _end = 0, 0
        for _i in range(batch_size):
            _count = count_prefix[_i]
            _prefix_len = _count * num_tokens
            _tail = _prefix_len + inputs_embeds.shape[1]

            _end += _count
            # (_prefix_len, hidden size) <=> (_prefix_len, hidden size)
            # copy vision_embeds to out_embeds
            out_embeds[_i,
                       0:_prefix_len] = vision_embeds[_begin:_end].reshape(-1, hidden_size)
            # copy inputs_embeds to out_embeds
            out_embeds[_i, _prefix_len:_tail] = inputs_embeds[_i]

            # update begin
            _begin = _end

        # concat tps
        with torch.no_grad():
            out_tps = torch.full(
                (batch_size, max_tps_len),
                fill_value=EnumTokenType.PAD.value,
                dtype=input_tps.dtype,
                device=input_tps.device,
            )
            if attention_mask is not None:
                out_attn_mask = torch.full_like(
                    out_tps, fill_value=0)
            else:
                out_attn_mask = None

            for _idx, _count in count_prefix.items():
                _prefix_len = _count * num_tokens
                _tail = _prefix_len + input_tps.shape[1]
                out_tps[_idx, 0:_prefix_len] = EnumTokenType.IMG.value
                out_tps[_idx, _prefix_len:_tail] = input_tps[_idx]

                if out_attn_mask is not None:
                    out_attn_mask[_idx, 0:_prefix_len] = 1
                    out_attn_mask[_idx,
                                  _prefix_len:_tail] = attention_mask[_idx]

            if out_attn_mask is None:
                # not pad is 1
                out_attn_mask = EnumTokenType.not_the_type(
                    out_tps, EnumTokenType.PAD)

        if not return_dict:
            return (out_embeds, out_tps, out_attn_mask)

        # no pooling is outputed
        return ImageHDMultiQueryOutput(
            embeds=out_embeds,
            input_tps=out_tps,
            attention_mask=out_attn_mask,
        )


class CLIPVisionHDModelNoPool(modeling_clip.CLIPPreTrainedModel):
    config_class = CLIPVisionHDConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionHDConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionHDTransformerWithInputTPS(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        crop_index: torch.LongTensor,
        inputs_embeds: torch.LongTensor,
        input_tps: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageHDMultiQueryOutput]:
        return self.vision_model(
            pixel_values=pixel_values,
            crop_index=crop_index,
            inputs_embeds=inputs_embeds,
            input_tps=input_tps,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )


class ImageQueryModelForVLM(transformers.PreTrainedModel):
    config_class = ModelingQueryAdaptCLIPModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ModelingQueryAdaptCLIPModelConfig):
        super().__init__(config)

        self.query_align_model = ImageQueryTransformer(
            config.query_align_config)
        self.clip_model = CLIPVisionModelNoPool(config.vision_config)

    @property
    def num_patches(self):
        return self.clip_model.vision_model.embeddings.num_patches

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        # the last hidden state of the vision encoder, cls_patch included
        vision_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, modeling_clip.BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if vision_embeds is None:
            if pixel_values is None:
                raise ValueError(
                    "Either pixel_values or last_hidden_state has to be provided.")

            vision_outputs = self.clip_model.vision_model(
                pixel_values=pixel_values,
            )
            vision_embeds = vision_outputs[0]  # type: torch.Tensor

        return self.query_align_model(
            input_ids=input_ids,
            # the cls_patch is not included
            vision_embeds=vision_embeds[:, 1:],
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_image_embeds(self, pixel_values: torch.FloatTensor, keep_cls_patch: bool = True) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
        )
        if not keep_cls_patch:
            vision_outputs = vision_outputs[0][:, 1:]
        else:
            return vision_outputs[0]  # type: torch.Tensor


###################################
##### Image Multi Query Model #####
###################################

class ImageMultiQueryTransformer(modeling_llama.LlamaModel):
    supports_gradient_checkpointing = True

    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config)

        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_tps: Optional[torch.LongTensor] = None,
        vision_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.IntTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageMultiQueryOutputWithPooling]:
        """
        Args:
        - input_ids: the ids of 'query'
        - vision_embeds: the vision embeds from vision encoder
        - attention_mask: the attention_mask of 'input_ids' !!!
        - output_attentions:
        - output_hidden_states:
        - return_dict:

        Returns:
        - the output of the CLIPTextTransformer
        """

        inputs_embeds = self.embed_tokens(input_ids)  # type: torch.Tensor
        inputs_embeds = torch.cat([
            vision_embeds,
            inputs_embeds,
        ], dim=1)

        bsz, patch_count = vision_embeds.shape[:2]

        # update attention mask
        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones(
                    (bsz, patch_count),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ], dim=1)

        if input_tps is not None:
            input_tps = torch.cat([torch.full(
                (bsz, patch_count),
                fill_value=EnumTokenType.IMG.value,
                dtype=input_tps.dtype,
                device=input_tps.device,
            ), input_tps], dim=1)

        super_outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = super_outputs[0]  # type: torch.Tensor

        with torch.no_grad():
            # pick the eos position
            eos_mask = EnumTokenType.is_the_type(input_tps, EnumTokenType.EOS)

        pooled_output = last_hidden_state[eos_mask]

        if not return_dict:
            return (
                last_hidden_state,
                pooled_output,
                input_tps,
                attention_mask,
            ) + super_outputs[1:]

        return ImageMultiQueryOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            input_tps=input_tps,
            attention_mask=attention_mask,
            hidden_states=super_outputs.hidden_states,
            attentions=super_outputs.attentions,
        )


class ImageHDMultiQueryTransformer(modeling_llama.LlamaModel):
    supports_gradient_checkpointing = True

    def __init__(self, config: modeling_llama.LlamaConfig):
        super().__init__(config)

        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id


class CLIPTextTransformerWithInputTPS(modeling_clip.CLIPTextTransformer):
    """Copy from transformers.models.clip.CLIPTextTransformer"""
    config_class = modeling_clip.CLIPTextConfig

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_tps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageMultiQueryOutputWithPooling]:
        r"""
        mainly copy from `CLIPTextTransformer`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        if input_tps is None:
            raise ValueError("You have to specify input_tps")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = modeling_clip._create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = modeling_clip._prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        with torch.no_grad():
            eos_mask = EnumTokenType.is_the_type(input_tps, EnumTokenType.EOS)

        pooled_output = last_hidden_state[eos_mask]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return ImageMultiQueryOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            input_tps=input_tps,
            attention_mask=attention_mask,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


class TextTransformerWithInputTPS(modeling_llama.LlamaModel):

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        input_tps: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache,
                                        List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, ImageMultiQueryOutputWithPooling]:

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        last_hidden_state = output[0]  # type: torch.Tensor

        with torch.no_grad():
            eos_mask = EnumTokenType.is_the_type(input_tps, EnumTokenType.EOS)

        pooler_output = last_hidden_state[eos_mask]

        if not return_dict:
            return (last_hidden_state, pooler_output) + output[1:]

        return ImageMultiQueryOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            input_tps=input_tps,
            attention_mask=attention_mask,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class CLIPModelWithInputTPSBase(modeling_clip.CLIPPreTrainedModel):
    config_class = modeling_clip.CLIPConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: modeling_clip.CLIPConfig,
        text_class: Type[transformers.PreTrainedModel],
        vision_class: Type[transformers.PreTrainedModel],
        with_text_model=True,
        with_vision_model=True,
        with_text_projection=True,
        with_vision_projection=True,
        with_logit_scale=True,
    ):
        """Copy from transformers.models.clip.CLIPModel"""
        super().__init__(config)

        if with_text_model and (not isinstance(config.text_config, text_class.config_class)):
            raise ValueError(
                f"config.text_config is expected to be of type {text_class.config_class} but is of type {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, vision_class.config_class):
            raise ValueError(
                f"config.vision_config is expected to be of type {vision_class.config_class} but is of type {type(config.vision_config)}."
            )

        self.with_text_model = with_text_model
        self.with_vision_model = with_vision_model
        self.with_text_projection = with_text_projection
        self.with_vision_projection = with_vision_projection
        self.with_logit_scale = with_logit_scale
        self.projection_dim = config.projection_dim

        if with_text_model:
            text_config = config.text_config
            self.text_embed_dim = text_config.hidden_size
            self.text_model = text_class(text_config)

            if with_text_projection:
                self.text_projection = nn.Linear(
                    self.text_embed_dim, self.projection_dim, bias=False)

        if with_vision_model:
            vision_config = config.vision_config
            self.vision_embed_dim = vision_config.hidden_size
            self.vision_model = vision_class(vision_config)
            if with_vision_projection:
                self.visual_projection = nn.Linear(
                    self.vision_embed_dim, self.projection_dim, bias=False)

        if with_logit_scale:
            self.logit_scale = nn.Parameter(
                torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, modeling_clip.CLIPModel):
            if self.with_text_projection:
                nn.init.normal_(
                    module.text_projection.weight,
                    std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
                )
            if self.with_vision_projection:
                nn.init.normal_(
                    module.visual_projection.weight,
                    std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
                )
        elif isinstance(module, modeling_clip.CLIPVisionModelWithProjection):
            if self.with_vision_projection:
                nn.init.normal_(
                    module.visual_projection.weight,
                    std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
                )
        elif isinstance(module, modeling_clip.CLIPTextModelWithProjection):
            if self.with_text_projection:
                nn.init.normal_(
                    module.text_projection.weight,
                    std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
                )
        else:
            super()._init_weights(module)

    def update_text_model_position_embedding(self, new_size: int):
        """
        we try to extend the position embedding of text model
        Call this function after all checkpoints are loaded.
        """
        new_position_embedding = self._update_embeding_layer(
            new_size, self.text_model.embeddings.position_embedding
        )
        if new_position_embedding is not self.text_model.embeddings.position_embedding:
            self.text_model.embeddings.position_embedding = new_position_embedding

            # update config
            self.config.text_config.max_position_embeddings = new_size
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.text_model.embeddings.register_buffer(
                "position_ids", torch.arange(new_size).expand((1, -1)), persistent=False
            )

        return self

    def update_text_model_vocab_size(self, new_size: int):
        """
        try to update the vocab size of text transformer
        Call this function after all checkpoints are loaded.
        """
        new_embeding_layer = self._update_embeding_layer(
            new_size, self.text_model.embeddings.token_embedding)
        if new_embeding_layer is not self.text_model.embeddings.token_embedding:
            self.text_model.embeddings.token_embedding = new_embeding_layer
            # update config
            self.config.text_config.vocab_size = new_size

        return self

    def _update_embeding_layer(self, new_size: int, old_embeding_layer: nn.Embedding):
        """
        update the embeding layer from text transformer
        """
        if new_size <= old_embeding_layer.num_embeddings:
            # do nothing
            return old_embeding_layer

        # extend the token embedding
        old_size = old_embeding_layer.num_embeddings
        new_embeding_layer = nn.Embedding(new_size, old_embeding_layer.embedding_dim).to(
            device=old_embeding_layer.weight.device,
            dtype=old_embeding_layer.weight.dtype,
        )

        with torch.no_grad():
            # initialize the new position embedding
            new_embeding_layer.weight[0:old_size].copy_(
                old_embeding_layer.weight)

            # init the new parts
            _std, _mean = torch.std_mean(old_embeding_layer.weight)
            new_embeding_layer.weight[old_size:new_size].normal_(
                mean=_mean.item(), std=_std.item()
            )

        # setup gradient
        new_embeding_layer.weight.requires_grad_(
            old_embeding_layer.weight.requires_grad)

        return new_embeding_layer

    def forward(self, *args, **kwargs):
        # TODO: the forward method is copied from transformers.models.clip.CLIPModel
        # and it is not implement with input_tps
        raise NotImplementedError


class CLIPModelWithInputTPS(CLIPModelWithInputTPSBase):
    config_class = modeling_clip.CLIPConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: modeling_clip.CLIPConfig,
        with_text_model=True,
        with_vision_model=True,
        with_text_projection=True,
        with_vision_projection=True,
        with_logit_scale=True,
    ):
        """Copy from transformers.models.clip.CLIPModel"""
        super().__init__(
            config,
            text_class=CLIPTextTransformerWithInputTPS,
            vision_class=modeling_clip.CLIPVisionTransformer,
            with_text_model=with_text_model,
            with_vision_model=with_vision_model,
            with_text_projection=with_text_projection,
            with_vision_projection=with_vision_projection,
            with_logit_scale=with_logit_scale,
        )


class CLIPHDModelWithInputTPS(CLIPModelWithInputTPSBase):
    """
    text_class = CLIPTextTransformerWithInputTPS
    vision_class = CLIPVisionHDTransformerWithInputTPS
    """
    config_class = ImageHDMultiQueryCLIPConfig

    def __init__(
        self,
        config: ImageHDMultiQueryCLIPConfig,
        with_text_model=True,
        with_vision_model=True,
        with_text_projection=True,
        with_vision_projection=True,
        with_logit_scale=True,
    ):
        super().__init__(
            config,
            text_class=CLIPTextTransformerWithInputTPS,
            vision_class=CLIPVisionHDTransformerWithInputTPS,
            with_text_model=with_text_model,
            with_vision_model=with_vision_model,
            with_text_projection=with_text_projection,
            with_vision_projection=with_vision_projection,
            with_logit_scale=with_logit_scale,
        )


class ModelForVLM(PreTrainedModel):
    is_hd_model = False

    def get_image_embeds(self, pixel_values: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_patches(self):
        raise NotImplementedError


@deprecated("Use ImageHDMultiQueryCLIPModel instead")
class ImageMultiQueryCLIPModel(modeling_clip.CLIPPreTrainedModel):
    config_class = ImageMultiQueryCLIPConfig
    supports_gradient_checkpointing = True
    is_hd_model = False

    def __init__(
        self,
        config: ImageMultiQueryCLIPConfig,
        clip_model_from_pretrained: bool = False,
    ):
        super().__init__(config)

        self.imgque_config = config.imgque_config
        self.projection_dim = config.projection_dim

        if clip_model_from_pretrained:
            assert config.base_name_or_path is not None, "Please provide the base_name_or_path"
            self.clip_model = CLIPModelWithInputTPS.from_pretrained(
                config.base_name_or_path,
                with_text_model=True,
                with_vision_model=True,
                with_text_projection=True,
                with_vision_projection=False,
                with_logit_scale=False,
            )  # type: CLIPModelWithInputTPS
        else:
            self.clip_model = CLIPModelWithInputTPS(
                config,
                with_text_model=True,
                with_vision_model=True,
                with_text_projection=True,
                with_vision_projection=False,
                with_logit_scale=False,
            )

        assert config.text_config.eos_token_id != 2, "We do not support the eos_token_id=2; Please read the funciton forward of CLIPTextTransformer"

        self.imgque_model = ImageMultiQueryTransformer(self.imgque_config)
        self.imgque_projection = nn.Linear(
            self.imgque_config.hidden_size, self.projection_dim, bias=False)

        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value))

    def update_text_model_position_embedding(self, new_size: int):
        """
        Extend the position embedding of the text model.
        """
        return self.clip_model.update_text_model_position_embedding(new_size)

    def update_text_model_vocab_size(self, new_size: int):
        """
        Update the vocab size of the text model.
        """
        return self.clip_model.update_text_model_vocab_size(new_size)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        que_input_ids: Optional[torch.LongTensor] = None,
        que_input_tps: Optional[torch.LongTensor] = None,
        que_attention_mask: Optional[torch.Tensor] = None,
        ans_input_ids: Optional[torch.LongTensor] = None,
        ans_input_tps: Optional[torch.LongTensor] = None,
        ans_attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ImageMultiQueryCLIPOutput]:
        r"""
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # type: modeling_clip.BaseModelOutputWithPooling

        # last hidden state
        vision_last_hidden_state = vision_outputs[0]  # type: torch.Tensor

        if self.training and self.is_gradient_checkpointing:
            vision_last_hidden_state.requires_grad_(True)

        text_outputs = self.clip_model.text_model.forward(
            input_ids=ans_input_ids,
            input_tps=ans_input_tps,
            attention_mask=ans_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # type: ImageMultiQueryOutputWithPooling

        # pooled_output
        text_embeds = text_outputs[1]  # type: torch.Tensor
        text_embeds = self.clip_model.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # compute query_align
        imgque_outputs = self.imgque_model.forward(
            input_ids=que_input_ids,
            input_tps=que_input_tps,
            attention_mask=que_attention_mask,
            # drop cls_patch
            vision_embeds=vision_last_hidden_state[:, 1:],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # que_mask = EnumTokenType.is_the_type(que_input_tps, EnumTokenType.EOS)
        # ans_mask = EnumTokenType.is_the_type(ans_input_tps, EnumTokenType.EOS)

        # assert torch.sum(que_mask) == torch.sum(ans_mask)

        # pooled_output
        imgque_embeds = imgque_outputs[1]  # type: torch.Tensor
        imgque_embeds = self.imgque_projection(imgque_embeds)
        imgque_embeds = imgque_embeds / \
            imgque_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(imgque_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_image * logit_scale) + \
                0.01 * torch.norm(logits_per_image)

            # add contrastive loss on imgque_outputs
            imgque_last_hidden_state = imgque_outputs[0]
            with torch.no_grad():
                qed_eos_mask = EnumTokenType.is_the_type(
                    imgque_outputs.input_tps, EnumTokenType.QED_EOS)

            imgque_qed_eos = imgque_last_hidden_state[qed_eos_mask]

            _iqe_iqeT = torch.matmul(imgque_qed_eos, imgque_qed_eos.t())
            loss_iqe = clip_loss(_iqe_iqeT * logit_scale) + 0.01 * \
                torch.norm(imgque_last_hidden_state, dim=-1).mean()
            loss = loss + loss_iqe

        if not return_dict:
            outputs = (logits_per_image, logits_per_text, text_embeds,
                       text_outputs, vision_outputs, imgque_embeds)
            return (loss, ) + outputs if loss is not None else outputs

        return ImageMultiQueryCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
            imgque_embeds=imgque_embeds,
        )


class ImageHDMultiQueryCLIPModel(modeling_clip.CLIPPreTrainedModel):
    base_model_prefix = "clip_hd"
    config_class = ImageHDMultiQueryCLIPConfig
    supports_gradient_checkpointing = True
    is_hd_model = True

    def __init__(
        self,
        config: ImageHDMultiQueryCLIPConfig,
        clip_model_from_pretrained: bool = False,
    ):
        super().__init__(config)

        self.imgque_config = config.imgque_config
        self.projection_dim = config.projection_dim

        self.clip_model = CLIPHDModelWithInputTPS(
            config,
            with_text_model=True,
            with_vision_model=True,
            with_text_projection=True,
            with_vision_projection=False,
            with_logit_scale=False,
        )

        if clip_model_from_pretrained:
            assert config.base_name_or_path is not None, "Please provide the base_name_or_path"

            _clip_model = transformers.AutoModel.from_pretrained(
                config.base_name_or_path,
            )
            missing_keys, unexpected_keys = self.clip_model.load_state_dict(
                _clip_model.state_dict(), strict=False)
            if len(unexpected_keys) > 0:
                print("Unexpected keys:", unexpected_keys)
            if len(missing_keys) > 0:
                print("Missing keys:", missing_keys)

        # assert config.text_config.eos_token_id != 2, "We do not support the eos_token_id=2; Please read the funciton forward of CLIPTextTransformer"

        self.imgque_model = ImageHDMultiQueryTransformer(self.imgque_config)
        self.imgque_projection = nn.Linear(
            self.imgque_config.hidden_size, self.projection_dim, bias=False)

        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value))

    def update_text_model_position_embedding(self, new_size: int):
        """
        Extend the position embedding of the text model.
        """
        return self.clip_model.update_text_model_position_embedding(new_size)

    def update_text_model_vocab_size(self, new_size: int):
        """
        Update the vocab size of the text model.
        """
        return self.clip_model.update_text_model_vocab_size(new_size)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        crop_index: Optional[torch.LongTensor] = None,
        que_input_ids: Optional[torch.LongTensor] = None,
        que_input_tps: Optional[torch.LongTensor] = None,
        que_attention_mask: Optional[torch.Tensor] = None,
        ans_input_ids: Optional[torch.LongTensor] = None,
        ans_input_tps: Optional[torch.LongTensor] = None,
        ans_attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ImageMultiQueryCLIPOutput]:
        r"""
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        que_inputs_embeds = self.imgque_model.embed_tokens(que_input_ids)

        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            crop_index=crop_index,
            inputs_embeds=que_inputs_embeds,
            input_tps=que_input_tps,
            attention_mask=que_attention_mask,
            return_dict=True,
        )  # type: ImageHDMultiQueryOutput

        # GlobalDebugUtil.print(vision_outputs[0], "vision outputs last hidden state")

        text_outputs = self.clip_model.text_model.forward(
            input_ids=ans_input_ids,
            input_tps=ans_input_tps,
            attention_mask=ans_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # type: ImageMultiQueryOutputWithPooling

        # pooled_output
        text_embeds = text_outputs[1]  # type: torch.Tensor

        text_embeds = self.clip_model.text_projection(text_embeds)

        # compute mean, std
        text_embeds_clamp_mask = out_of_distribution_mask(text_embeds)

        # remove outlier
        # GlobalDebugUtil.print(text_embeds, "text embeds before clamp")
        # text_embeds = torch.clamp(text_embeds, min=text_embeds_clamp_min, max=text_embeds_clamp_max)
        text_embeds = text_embeds.masked_fill(text_embeds_clamp_mask, 0)
        # GlobalDebugUtil.print(text_embeds, "text embeds after clamp")
        # print(torch.norm(text_embeds, dim=-1))
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # compute query_align
        imgque_outputs = self.imgque_model.forward(
            input_ids=None,
            attention_mask=vision_outputs.attention_mask,
            inputs_embeds=vision_outputs.embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )  # type: modeling_llama.BaseModelOutputWithPast

        with torch.no_grad():
            imgque_eos_mask = EnumTokenType.is_the_type(
                vision_outputs.input_tps, EnumTokenType.EOS)

        #
        imgque_outputs_last_hidden_state = imgque_outputs.last_hidden_state  # type: torch.Tensor

        # type: torch.Tensor
        imgque_embeds = imgque_outputs_last_hidden_state[imgque_eos_mask]
        imgque_embeds = self.imgque_projection(imgque_embeds)
        imgque_embeds = imgque_embeds / \
            imgque_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(imgque_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            # _eye_cl = torch.eye(logits_per_image.shape[0], dtype=logits_per_image.dtype, device=logits_per_image.device)
            # loss = torchF.mse_loss(logits_per_image, _eye_cl)
            loss = clip_loss(logits_per_image * logit_scale)
            # with np.printoptions(precision=2, suppress=True):
            #     print(torch.relu(logits_per_image * logit_scale).data.cpu().float().numpy())

            # add contrastive loss on imgque_outputs
            with torch.no_grad():
                qed_eos_mask = EnumTokenType.is_the_type(
                    vision_outputs.input_tps, EnumTokenType.QED_EOS)

            imgque_qed_eos = imgque_outputs_last_hidden_state[qed_eos_mask]
            imgque_qed_eos = imgque_qed_eos / \
                imgque_qed_eos.norm(dim=-1, keepdim=True)
            # imgque_mask = out_of_distribution_mask(imgque_outputs_last_hidden_state, 3)
            # GlobalDebugUtil.print(imgque_outputs_last_hidden_state, "imgque last hidden state before mask")
            # imgque_outputs_last_hidden_state = imgque_outputs_last_hidden_state.masked_fill(imgque_mask, 0)
            # GlobalDebugUtil.print(imgque_outputs_last_hidden_state, "imgque last hidden state after mask")

            _iqe_iqeT = torch.matmul(imgque_qed_eos, imgque_qed_eos.t())

            with torch.no_grad():
                _eye = torch.eye(
                    _iqe_iqeT.shape[0], dtype=_iqe_iqeT.dtype, device=_iqe_iqeT.device)

            loss = loss + torchF.mse_loss(_iqe_iqeT, _eye)

        if not return_dict:
            outputs = (logits_per_image, logits_per_text, text_embeds,
                       text_outputs, vision_outputs, imgque_embeds)
            return (loss, ) + outputs if loss is not None else outputs

        return ImageMultiQueryCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
            imgque_embeds=imgque_embeds,
        )


class ImageMultiQueryCLIPModelForVLM(ModelForVLM):
    config_class = ImageMultiQueryCLIPConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    is_hd_model = False

    def __init__(self, config: ImageMultiQueryCLIPConfig):
        """
        No Projection; No CLIP Text Encoder;

        Only contains
        - CLIPVisionModel
        - ImageMultiQueryTransformer
        """
        super().__init__(config)
        self.config = config
        assert config.text_config.eos_token_id != 2, "eos_token_id should not be 2"

        self.clip_model = CLIPVisionModelNoPool(config.vision_config)
        self.imgque_model = ImageMultiQueryTransformer(config.imgque_config)

    def get_input_embeddings(self):
        return self.imgque_model.embed_tokens

    def set_input_embeddings(self, value):
        self.imgque_model.embed_tokens = value

    @property
    def num_patches(self):
        return self.clip_model.vision_model.embeddings.num_patches

    def get_image_embeds(self, pixel_values: torch.FloatTensor, keep_cls_patch: bool = True) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values)
        vision_embeds = vision_outputs[0]  # type: torch.Tensor
        if not keep_cls_patch:
            vision_embeds = vision_embeds[:, 1:]
        return vision_embeds

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_embeds: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_tps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ImageMultiQueryOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if vision_embeds is None:
            if pixel_values is None:
                raise ValueError(
                    "Both `pixel_values` and `vision_embeds` cannot be `None`")

            vision_outputs = self.clip_model.vision_model(
                pixel_values=pixel_values)
            vision_embeds = vision_outputs[0]  # type: torch.Tensor

        return self.imgque_model.forward(
            input_ids=input_ids,
            input_tps=input_tps,
            attention_mask=attention_mask,
            # drop the cls_patch
            vision_embeds=vision_embeds[:, 1:],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ImageHDMultiQueryCLIPModelForVLM(ModelForVLM):
    config_class = ImageHDMultiQueryCLIPConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    is_hd_model = True

    def __init__(
        self,
        config: ImageHDMultiQueryCLIPConfig,
    ):
        super().__init__(config)
        self.config = config

        self.clip_model = CLIPVisionHDModelNoPool(config.vision_config)

        self.imgque_model = ImageHDMultiQueryTransformer(config.imgque_config)

    def num_patches(self):
        return None  # Not a valid value

    def get_input_embeddings(self):
        return self.imgque_model.embed_tokens

    def set_input_embeddings(self, value):
        self.imgque_model.embed_tokens = value

    def get_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_vision_embeddings(pixel_values=pixel_values)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        crop_index: Optional[torch.LongTensor] = None,
        # vision_embeds: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_tps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        assert pixel_values.shape[0] == crop_index.shape[0], "batch size of pixel_values and crop_index should be the same"

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # currently, do not support input vision_embeds directly
        # todo: support input vision_embeds later
        # if vision_embeds is None:
        #     if pixel_values is None:
        #         raise ValueError("You have to specify either pixel_values or vision_embeds")

        inputs_embeds = self.imgque_model.embed_tokens(input_ids)

        clip_outputs = self.clip_model.forward(
            pixel_values=pixel_values,
            crop_index=crop_index,
            inputs_embeds=inputs_embeds,
            input_tps=input_tps,
            attention_mask=attention_mask,
            return_dict=True,
        )

        imgque_outputs = self.imgque_model.forward(
            input_ids=None,
            attention_mask=clip_outputs.attention_mask,
            position_ids=None,
            inputs_embeds=clip_outputs.embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = imgque_outputs.last_hidden_state

        if not return_dict:
            return (last_hidden_state, clip_outputs.input_tps, clip_outputs.attention_mask) + imgque_outputs[2:]

        return ImageMultiQueryOutputWithPooling(
            last_hidden_state=last_hidden_state,
            input_tps=clip_outputs.input_tps,
            attention_mask=clip_outputs.attention_mask,
            hidden_states=imgque_outputs.hidden_states,
            attentions=imgque_outputs.attentions,
        )


class ImageHDCLIPVisionModelForVLM(ModelForVLM):
    base_model_prefix = "clip_hd"
    config_class = ImageHDCLIPConfig
    supports_gradient_checkpointing = True
    is_hd_model = True

    def __init__(
        self,
        config: ImageHDCLIPConfig,
        vision_from_pretrained: bool = False,
    ):
        super().__init__(config)

        if vision_from_pretrained:
            assert config.base_model_name_or_path is not None, "base_model_name_or_path should be specified when clip_model_from_pretrained is True"
            self.clip_model = CLIPModelWithInputTPSBase.from_pretrained(
                config.base_model_name_or_path,
                config=config,
                text_class=None,
                vision_class=CLIPVisionHDTransformer,
                with_text_model=False,
                with_text_projection=False,
                with_vision_model=True,
                with_vision_projection=False,
                with_logit_scale=False,
            )
        else:
            self.clip_model = CLIPModelWithInputTPSBase(
                config,
                text_class=None,
                vision_class=CLIPVisionHDTransformer,
                with_text_model=False,
                with_text_projection=False,
                with_vision_model=True,
                with_vision_projection=False,
                with_logit_scale=False,
            )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        drop_cls_patch: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return self.clip_model.vision_model(
            pixel_values=pixel_values,
            drop_cls_patch=drop_cls_patch,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


AutoConfig.register("modeling_query_align_clip",
                    ModelingQueryAdaptCLIPModelConfig)
AutoConfig.register("modeling_image_multi_query_clip",
                    ImageMultiQueryCLIPConfig)
AutoConfig.register("modeling_image_multi_query", ImageMultiQueryConfig)
AutoConfig.register("modeling_image_hd_multi_query_clip",
                    ImageHDMultiQueryCLIPConfig)
AutoConfig.register("clip_vision_hd", CLIPVisionHDConfig)
AutoConfig.register("modeling_image_hd_clip", ImageHDCLIPConfig)

AutoModel.register(ModelingQueryAdaptCLIPModelConfig, QueryAdaptCLIPModel)
AutoModel.register(ImageMultiQueryCLIPConfig, ImageMultiQueryCLIPModel)
AutoModel.register(ImageMultiQueryCLIPConfig, ImageMultiQueryCLIPModelForVLM)
AutoModel.register(ImageHDMultiQueryCLIPConfig,
                   ImageHDMultiQueryCLIPModelForVLM)
AutoModel.register(CLIPVisionHDConfig, CLIPVisionHDModelNoPool)


class AutoModelOnVLMForImageMultiQuery(auto_model.AutoModelForPreTraining):
    _model_mapping = OrderedDict({
        ImageMultiQueryCLIPConfig: ImageMultiQueryCLIPModelForVLM,
        ImageHDMultiQueryCLIPConfig: ImageHDMultiQueryCLIPModelForVLM,
        ImageHDCLIPConfig: ImageHDCLIPVisionModelForVLM,
    })
