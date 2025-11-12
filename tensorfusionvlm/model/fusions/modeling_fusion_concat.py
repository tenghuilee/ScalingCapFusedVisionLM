import torch
import torch.nn as nn

from tensorfusionvlm.constants import EnumTokenType

from .fusion_base import FusionConcatBase

class ImgQueCLIPFusion(FusionConcatBase):
    model_name = "img_multi_que_clip_fusion_linear"
    enable_clip_text_encoder = True

    def _build_model(self):
        import tensorfusionvlm.model.modeling_imghd_multi_que_llm as clip_llm

        # don't need project
        assert isinstance(self.config, clip_llm.ImageMultiQueryCLIPLLMConfig)
        config = self.config  # type: clip_llm.ImageMultiQueryCLIPLLMConfig

        assert config.imgque_config is not None
        assert config.llm_config is not None
        assert config.imgque_config.hidden_size is not None
        assert config.llm_config.hidden_size is not None

        self.imgque_proj = nn.Linear(config.imgque_config.hidden_size, config.llm_config.hidden_size, bias=False)

        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        """
        new add after 
        "./checkpoints/modeling_img_multi_que_zero_2x_16_cl_qed15_eos1_layer16"
        """
        self.imgque_proj.weight.normal_(mean=0.0, std=0.001)
        if self.imgque_proj.bias is not None:
            self.imgque_proj.bias.zero_()

    def _fusion( # type: ignore
        self,
        input_embeds: torch.Tensor,
        input_tps: torch.Tensor,
        imgque_embeds: torch.Tensor,
        que_input_tps: torch.Tensor,
        **kwargs,
    ):
        if input_embeds.size(1) == 1:
            return input_embeds

        with torch.no_grad():
            # Find the place holder and insert the imgque to the input_embeds
            input_mask = EnumTokenType.is_the_type(input_tps, EnumTokenType.QED)
            imgque_mask = EnumTokenType.is_the_type(que_input_tps, EnumTokenType.QED_EOS)
        
        if imgque_mask.sum() == 0:
            return input_embeds, None

        # projection
        proj_imgque_embeds = self.imgque_proj(imgque_embeds[imgque_mask])

        # GlobalDebugUtil.print(imgque_embeds, "imgque_embeds")
        # GlobalDebugUtil.pause()

        # Clone input_embeds to avoid in-place modifications
        input_embeds = input_embeds.clone()
        # Assign the projected image embeddings to the corresponding positions in input_embeds
        input_embeds[input_mask] = proj_imgque_embeds        
        
        return input_embeds 

class SimpleLinearFusion(FusionConcatBase):
    model_name = "imghd_simple_linear"
    enable_clip_text_encoder = True

    def _build_model(self):
        self.m_proj = nn.Linear(
            self.img_embed_size,
            self.txt_embed_size,
            bias=False,
        )
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        """
        new add after 
        "./checkpoints/modeling_img_multi_que_zero_2x_16_cl_qed15_eos1_layer16"
        """
        self.m_proj.weight.normal_(mean=0.0, std=0.001)
        if self.m_proj.bias is not None:
            self.m_proj.bias.zero_()

    def _fusion( # type: ignore
        self,
        vision_embeds: torch.Tensor,
        **kwargs,
    ):
        return self.m_proj(vision_embeds)
