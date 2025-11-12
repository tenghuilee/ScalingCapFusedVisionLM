from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as torchF
from transformers.modeling_outputs import ModelOutput

from tensorfusionvlm.constants import EnumTokenType

from .fusion_base import *
from .. import configuration_base 

# for all sub module;
# 1. extend FusionBase
# 2. add to __init__.py


class FusionBase(nn.Module):
    model_name = "base"
    enable_vq = False
    enable_clip_text_encoder = False
    _subclass_cls = []

    def __init__(
        self,
        img_embed_size: int,
        img_embed_len: int,
        txt_embed_size: int,
        config : configuration_base.TensorFusionVLMConfig = None,
    ):
        super().__init__()
        self.img_embed_size = img_embed_size
        self.img_embed_len = img_embed_len
        self.txt_embed_size = txt_embed_size
        self.config = config

        self.loss_focus_type = EnumTokenType.ANS_EOS
        self.enable_fusion = True
        self.label_smoothing = 0.0

        self._build_model()

    @property
    def length_que_end(self) -> int:
        return self.config.length_que_end
    
    def _build_model(self):
        # please implememt the create model stuff here
        pass

    @abstractmethod
    def _fusion(
        self,
        images_embeds: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_tps: torch.LongTensor,
        **kwargs,
    ):
        """
        Fusion method

        Require:
        - images_embeds: the embed images from vision_backbone (batch size, image token length, image embed dimmension)
        - inputs_embeds: the embed input_ids from llm embed layer (batch size, text token length, text embed dimmension)
        - input_tps: (batch size, text token length)
            the values stand for 
                PAD = -100 # the pad sequence
                INST = 1 # [INST] [/INST] bos<s>, eos </s>
                SYS = 2 # <<SYS>>\nxxxx\n<<SYS>>\n\n
                IMG = 3 # image token
                QUE = 4 # query
                ANS = 5 # answer from gpt
                BOS = 6 # begin of string
                EOS = 7 # end of string
            Please use EnumTOkenType.xxx.value

        Return:
        - text tokens
            - this will be feeded to LLM to get the output
        """
        raise NotImplementedError()

    @abstractmethod
    def _img_projection(self, images_embeds: torch.Tensor):
        """
        Map image tokens to the txt tokens.

        Image Pixel Values 
        => Image tokens ==============> Projected Image Tokens 
                        (this function)
        => Concat or whatever

        If you want to enable the support by
        - TensorFusionVLMPatch
        Please implement this function.
        """
        raise NotImplementedError("The funcion of ")
    
    @abstractmethod
    def enable_img_projection_patch(self, patch_module: nn.Module):
        raise NotImplementedError("This module do not support patch for img_projection")
    
    def _compute_loss(
        self,
        logist: torch.Tensor,
        input_ids: torch.LongTensor,
        input_tps: torch.LongTensor,
        model_output = None, # do not remove this; it is used in other place
    ):
        """
        The logist is not shifted

        For causal LLM

        input:  A B C D
        predic: B C D E
        the label is right shift by 1
        """
        # the basic corss entropy loss
        with torch.no_grad():
            input_tps = input_tps[..., 1:]
            # mask = torch.logical_not(torch.logical_or(
            #     input_tps == EnumTokenType.ANS.value,
            #     input_tps == EnumTokenType.EOS.value,
            # ))
            mask = EnumTokenType.not_the_type(input_tps, self.loss_focus_type)

            # random pick labels
            # mask = torch.logical_or(
            #     mask,
            #     torch.rand(mask.shape, device=input_ids.device) < 0.1,
            # )
            
            labels = input_ids[..., 1:].detach().clone()
            labels[mask] = -100 # ignore index
            labels = labels.contiguous()

        logist_shift = logist[..., :-1, :].contiguous()

        # print(predict.shape, labels.shape, labels)
        return torchF.cross_entropy(
            logist_shift.view(-1, logist_shift.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, *args, **kwargs):
        return self._fusion(*args, **kwargs)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._subclass_cls.append(cls)

    def compute_loss(
        self,
        logist: torch.Tensor,
        input_ids: torch.LongTensor,
        input_tps: torch.LongTensor,
        network_output = None,
    ):
        return self._compute_loss(logist, input_ids, input_tps, network_output)
    
    def _prepare_model_inputs(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs: Dict[str, torch.Tensor] | None = None,
    ):
        return inputs_tensor, model_kwargs

@dataclass
class FusionConcatBaseReturn(ModelOutput):
    """
    Return type of the concat_base model.
    """
    inputs_embeds: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    input_tps: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    image_instance_mask: Optional[torch.LongTensor] = None
    cache_position: Optional[torch.LongTensor] = None
    cached_image_cat: Optional[torch.Tensor] = None
    cached_image_cat_vq: Optional[torch.Tensor] = None
    cached_image_embeds: Optional[torch.Tensor] = None
    cached_fusion_value: Optional[Dict[str, torch.Tensor]] = None
    cached_input_embeds: Optional[torch.Tensor] = None

class FusionConcatBase(FusionBase):
    model_name = "concat_base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config is None:
            self.pad_token_id = 0
        else:
            self.pad_token_id = getattr(self.config, "pad_token_id", 0)
            if self.pad_token_id is None:
                self.pad_token_id = 0
    
    def check_image_instance_mask(self, image_instance_mask: torch.LongTensor, inputs_embeds: torch.Tensor, image_embeds: torch.Tensor):
        if image_instance_mask is None:
            assert inputs_embeds.shape[0] == image_embeds.shape[0], "the batch size of inputs_embeds and image_embeds must be the same if image_instance_mask is not provided"
            image_instance_mask = torch.ones((inputs_embeds.shape[0],), dtype=torch.long, device=inputs_embeds.device)
        return image_instance_mask

    def concat_inputs_for_multimodal(
        self,
        *,
        images_embeds: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_tps: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        max_sequence_len: Optional[int] = None, # if = None, or <= 0; not use this
        image_instance_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare inputs for multimodal LLM

        Warning: 
        Only the first `image tag` is took into consideration.

        Please preprocess the images_embeds or inputs_embeds to make sure
        ```
        images_embeds.shape[2] == inputs_embeds.shape[2]
        ```
        This function do not have any projections or alignments.


        This function will search the position of `image tag` and insert the `images embeds` to that position.

        1. search the input_tps: find where `image tag` located

        Requires:
        - images_embeds: (batch size, image embed length, dim)
        - inputs_embeds: (batch size, text token length, dim)
        - input_tps: (batch size, text token length)
        - attention_mask: 
        - position_ids:
        - input_ids:
        - past_key_values:
        - cache_position:
        """

        image_instance_mask = self.check_image_instance_mask(image_instance_mask, inputs_embeds, images_embeds)

        # check if there is a IMG tag in the input_tps
        if (images_embeds is None) or (past_key_values is not None) or (input_tps.size(1) == 1) or (not torch.any(input_tps == EnumTokenType.IMG.value)):
            if (past_key_values is not None) and (images_embeds is not None) and (input_tps.size(1) == 1):
                # inference mode
                # update attention_mask and position_ids
                # copy from llava
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

            return FusionConcatBaseReturn(
                inputs_embeds=inputs_embeds,
                input_tps=input_tps,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_ids=input_ids,
                cache_position=cache_position,
            )

        bsz, txt_seq_len = input_tps.shape
        _, img_seq_len, hidden_size = images_embeds.shape
        out_seq_len = txt_seq_len + img_seq_len

        if input_ids is None:
            output_ids = None
        else:
            output_ids = torch.full(
                (bsz, out_seq_len),
                fill_value=self.config.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        outputs_embeds = inputs_embeds.new_zeros(bsz, out_seq_len, hidden_size)

        outputs_tps = torch.full(
            (bsz, out_seq_len),
            fill_value=EnumTokenType.PAD.value, # set all as pad
            dtype=input_tps.dtype,
            device=input_tps.device,
        )

        outputs_attention_mask = torch.zeros(
            (bsz, out_seq_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        with torch.no_grad():
            if position_ids.size(0) == 1:
                # expand dim 0
                position_ids = position_ids.expand(bsz, -1)

        outputs_position_ids = torch.zeros(
            (bsz, out_seq_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        # copy input to outputs
        outputs_embeds[:, :txt_seq_len] = inputs_embeds
        outputs_tps[:, :txt_seq_len] = input_tps
        outputs_attention_mask[:, :txt_seq_len] = attention_mask
        outputs_position_ids[:, :txt_seq_len] = position_ids
        if output_ids is not None:
            output_ids[:, :txt_seq_len] = input_ids
        # end of copy

        if cache_position is not None:
            assert bsz == 1, "Param cache_position works in generation mode. Currently, only support batch size == 1. We are truing to support batch size > 1 later."
            outputs_cache_position = True # Just make it not None, 
        else:
            outputs_cache_position = None

        max_output_embed_len = 0

        def batch_indexes():
            _i = 0
            _j = 0
            while _i < bsz:
                _no = (image_instance_mask[_i] == 0)

                yield _no, _i, _j
                _i += 1
                if not _no:
                    _j += 1

        for no_concat, batch_idx, img_idx in batch_indexes():
            if no_concat:
                continue
            # TODO: this only works for single image, Error for multiple image
            with torch.no_grad():
                img_indices = torch.nonzero(input_tps[batch_idx] == EnumTokenType.IMG.value)
            # print(img_indices)
            if img_indices.numel() > 0:
                ll = img_indices[0].item()
                rr = img_indices[-1].item() + 1
                # print(ll, rr)
                # copy to outputs_embeds
                outputs_embeds[batch_idx, 0:ll] = inputs_embeds[batch_idx, 0:ll]
                outputs_embeds[batch_idx, ll:ll+img_seq_len] = images_embeds[img_idx, 0:img_seq_len]
                _outputs_embeds_len = ll - rr + img_seq_len+ txt_seq_len
                outputs_embeds[batch_idx, ll+img_seq_len:_outputs_embeds_len] = inputs_embeds[batch_idx, rr:txt_seq_len]

                if output_ids is not None:
                    output_ids[batch_idx, 0:ll] = input_ids[batch_idx, 0:ll]
                    output_ids[batch_idx, ll:ll+img_seq_len] = self.pad_token_id 
                    output_ids[batch_idx, ll+img_seq_len:_outputs_embeds_len] = input_ids[batch_idx, rr:txt_seq_len]

                # update the attention_mask
                outputs_attention_mask[batch_idx, 0:_outputs_embeds_len] = 1

                # update the output_tps
                outputs_tps[batch_idx, 0:ll] = input_tps[batch_idx, 0:ll]
                outputs_tps[batch_idx, ll:ll+img_seq_len] = EnumTokenType.IMG.value
                outputs_tps[batch_idx, ll+img_seq_len:_outputs_embeds_len] = input_tps[batch_idx, rr:txt_seq_len]


                position_ids_left = position_ids[batch_idx, 0].item()
                outputs_position_ids[batch_idx, 0:_outputs_embeds_len] = torch.arange(
                    position_ids_left, position_ids_left + _outputs_embeds_len, dtype=position_ids.dtype, device=position_ids.device,
                )

                max_output_embed_len = max(_outputs_embeds_len, max_output_embed_len)
        
        if max_sequence_len is not None and max_sequence_len > 0:
            max_output_embed_len = min(max_output_embed_len, max_sequence_len)

        return FusionConcatBaseReturn(
            input_ids=output_ids[:, 0:max_output_embed_len],
            inputs_embeds=outputs_embeds[:, 0:max_output_embed_len],
            input_tps=outputs_tps[:, 0:max_output_embed_len],
            attention_mask=outputs_attention_mask[:, 0:max_output_embed_len],
            position_ids=outputs_position_ids[:, 0:max_output_embed_len],
            cache_position=outputs_position_ids[0, 0:max_output_embed_len] if outputs_cache_position is not None else None, # TODO: fix this later
            image_instance_mask=image_instance_mask,
        )

    def fill_que_end_for_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_tps: torch.LongTensor,
        fusion_clip_text_img: torch.Tensor,
        length_que_end: int,
        image_instance_mask: torch.Tensor = None,
        inplace: bool = False,
    ):
        image_instance_mask = self.check_image_instance_mask(image_instance_mask, inputs_embeds, fusion_clip_text_img)

        if inplace:
            output = inputs_embeds
        else:
            output = inputs_embeds.clone()


        # find the qed blocks
        with torch.no_grad():
            image_instance_mask_bool = (image_instance_mask == 1)
            # shape (bsz, sequence length)
            qed_mask = EnumTokenType.is_the_type(
                input_tps[image_instance_mask_bool], EnumTokenType.QED).int()
            # Get the indices where qed_mask is 1
            qed_indices = (qed_mask == 1).nonzero(as_tuple=False)

        # for _bsz_idx in range(qed_mask.shape[0]):
        #     _que_idx = 0
        #     _i = 0
        #     while _i < qed_mask.shape[1]:
        #         if qed_mask[_bsz_idx, _i] == 1:
        #             print("copied to output", _bsz_idx, _i, _que_idx)
        #             # the sequence of qed_mask is 0 0 0 1 1 1 1 ....
        #             # There are `length_que_end` 1
        #             # since they are all the same, just skip
        #             output[_bsz_idx, _i:_i+length_que_end] = fusion_clip_text_img[_bsz_idx, _que_idx, :, :]
        #             _i += length_que_end
        #             _que_idx += 1
        #         else:
        #             _i += 1

        # length_que_end = max(length_que_end, fusion_clip_text_img.shape[2])

        end_idx_shift = min(length_que_end, fusion_clip_text_img.shape[2])

        inputs_embeds_len = inputs_embeds.shape[1]

        # Process each batch
        _j = 0
        for _bsz_idx in range(input_tps.shape[0]):
            if image_instance_mask[_bsz_idx] == 0:
                continue

            # Filter indices for the current batch
            batch_indices = qed_indices[qed_indices[:, 0] == _j][:, 1]

            if len(batch_indices) > 0:
                # Split indices into blocks of length_que_end
                for block_start in range(0, len(batch_indices), length_que_end):
                    start_idx = batch_indices[block_start]
                    end_idx = start_idx + end_idx_shift  
                    if end_idx >= inputs_embeds_len:
                        # index out of range
                        break

                    # Calculate the index for fusion_clip_text_img
                    _que_idx = block_start // length_que_end
                    # print(f"copied to output {_que_idx}, {start_idx} -> {end_idx} (+{end_idx_shift})")
                    target = fusion_clip_text_img[_j, _que_idx, 0:end_idx_shift, :]
                    output[_bsz_idx, start_idx:end_idx] = target
            
            _j += 1

        return output
