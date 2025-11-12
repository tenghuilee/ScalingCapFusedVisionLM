import unittest
import torch
from tensorfusionvlm.model.modeling_query_adapt_clip import *
from torch.utils.data import DataLoader

import tensorfusionvlm.data_utils as data_utils
import tensorfusionvlm.model.modeling_imghd_multi_que_llm as modeling_base
from tensorfusionvlm.auto_models import AutoModelForImageMultiQuery, AutoModelOnVLMForImageMultiQuery
import tensorfusionvlm.model.modeling_query_adapt_clip as modeling_vision

from tensorfusionvlm.utils import GlobalDebugUtil
GlobalDebugUtil.enable_print_histogram(True).enable_print_l2(False).enable_print_values(False)

class TestModelingQueryAlignClip(unittest.TestCase):
    def test_forward(self):
        config = ModelingQueryAdaptCLIPModelConfig()
        model = QueryAdaptCLIPModel(config)

        input_ids = torch.randint(0, 1000, (16, 32))  # Batch size 1, Sequence length 32
        pixel_values = torch.randn(16, 3, 224, 224)  # Batch size 1, Channels 3, Height 224, Width 224
        attention_mask = torch.ones(16, 32)  # Batch size 1, Sequence length 32

        output = model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_loss=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        print(output.loss.item())
    
    def test_imagehd_multi_query_clip_model_for_vlm(self):

        config = ImageHDMultiQueryCLIPConfig(
            max_position_embeddings=4096,
            hidden_size=1024,
            base_name_or_path="openai/clip-vit-large-patch14-336",
            num_hidden_layers=2,
            vocab_size=32000,
        )

        config.pad_token_id = 0

        model = ImageHDMultiQueryCLIPModelForVLM(
            config=config,
            clip_from_pretrained=True,
        )

        print(model)

        # create inputs
        pixel_values = torch.randn(17*2, 3, 336, 336)  # Batch size 1, Channels 3, Height 224, Width 224
        crop_index = torch.tensor([0]*17 + [1]*17)

        input_ids = torch.zeros((2, 32), dtype=torch.long)
        input_tps = torch.ones((2, 32), dtype=torch.long)
        attention_mask = torch.ones((2, 32), dtype=torch.long)

        output = model(
            pixel_values=pixel_values,
            crop_index=crop_index,
            input_ids=input_ids,
            input_tps=input_tps,
            attention_mask=attention_mask,
        )
        print(output)
        print(output.input_tps.shape)

    def test_model_llm_hd(self):

        # train from scratch
        config = modeling_base.ImageMultiQueryCLIPLLMConfig(
            imgque_model_name_or_path="./checkpoints/modeling_imghd_multi_que_cl_128_8_layer12",
            llm_model_name_or_path="./llama_hf_mirror/llama_hf/llama-2-7b-chat",
            fusion_name="img_multi_que_clip_fusion",
            length_que_end=128,
            length_que_end_short=8,
            append_special_padding=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        model = modeling_base.ImageMultiQueryCLIPLLMModel(
            config,
            imgque_model_from_pretrained=True,
            llm_model_from_pretrained=True,
            apply_cl_loss_on_image_encoder=True,
        ).to(torch.bfloat16).cuda()

        # create inputs

        clip_tokenizer = transformers.AutoTokenizer.from_pretrained(config.imgque_model_name_or_path)
        llm_tokenizer = transformers.AutoTokenizer.from_pretrained(config.llm_model_name_or_path)

        train_dataset = data_utils.UnionChatDataset(
            data_path="./datasets/tiny_dataset_for_debug.json",
            drop_no_image_instances=True,
            partitions=[
                data_utils.UCDP_HDImage(
                    image_folder="./datasets/soft_link_image_collection",
                    image_processor=data_utils.Phi3VImageProcessor.from_pretrained(
                        config.imgque_model_name_or_path,
                    ),
                    drop_empty_crops=True,
                ),
                data_utils.UCDP_LLMQueryAnswer_WithPlaceholder(
                    tokenizer=llm_tokenizer,
                    chat_template="llama-2-vision",
                    length_que_end=config.length_que_end,
                    length_que_end_short=config.length_que_end_short,
                ),
                data_utils.UCDP_CLIPQuery_WithPlaceholder(
                    tokenizer=clip_tokenizer,
                    length_que_end=config.length_que_end,
                    length_que_end_short=config.length_que_end_short,
                    max_sequence_length=40960,
                    append_special_padding=True,
                )
            ]
        )

        data_collator = data_utils.DataCollactorForUnionChatDataset(
            train_dataset.partitions)
        
        for batch in DataLoader(
            train_dataset,
            batch_size=2,
            collate_fn=data_collator,
        ):
            batch["return_loss"] = True
            batch["return_dict"] = True
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            with torch.no_grad():
                out = model(**batch)
                print(out.loss)

    def test_modeling_imghd_multi_que_on_vlm(self):
        ckpt_path = "./checkpoints/modeling_imghd_multi_que_cl_128_8_layer12"


        model = AutoModelForImageMultiQuery.from_pretrained(
            ckpt_path)

        for name, param in model.named_parameters():
            GlobalDebugUtil.print(param, name=name)

    def test_modeling_imghd_multi_que_cl(self):

        ckpt_path = "./checkpoints/modeling_imghd_multi_que_cl_128_8_layer12"

        model = AutoModelForImageMultiQuery.from_pretrained(
            ckpt_path,
            use_safetensors=True,
            # torch_dtype=torch.float16,
        ) # type: modeling_base.ImageHDMultiQueryCLIPModel

        model = model.eval().cuda()

        config = model.config # type: modeling_base.ImageHDMultiQueryCLIPConfig

        clip_tokenizer = transformers.AutoTokenizer.from_pretrained(config._name_or_path)

        train_dataset = data_utils.UnionChatDataset(
            data_path="./datasets/tiny_dataset_for_debug.json",
            drop_no_image_instances=True,
            partitions=[
                data_utils.UCDP_HDImage(
                    image_folder="./datasets/soft_link_image_collection",
                    image_processor=data_utils.Phi3VImageProcessor.from_pretrained(
                        config._name_or_path,
                    ),
                    drop_empty_crops=True,
                ),
                data_utils.UCDP_CLIPQuery_WithPlaceholder(
                    tokenizer=clip_tokenizer,
                    length_que_end=config.length_que_end,
                    length_que_end_short=config.length_que_end_short,
                    max_sequence_length=40960,
                    append_special_padding=True,
                ),
                data_utils.UCDP_CLIPAnswer_WithPlaceholder(
                    tokenizer=clip_tokenizer,
                    length_que_end=1,
                    length_que_end_short=1,
                    max_sequence_length=40960,
                    append_special_padding=True,
                )
            ]
        )

        data_collator = data_utils.DataCollactorForUnionChatDataset(
            train_dataset.partitions)
        
        for batch in DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=data_collator,
        ):
            batch["return_loss"] = True
            batch["return_dict"] = True
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            with torch.no_grad():
                out = model(**batch)
                print(f"loss: {out.loss}")
    
    @torch.no_grad()
    def test_modeling_imghd_clip(self):
        config = modeling_vision.ImageHDCLIPConfig(
            base_model_name_or_path="openai/clip-vit-large-patch14-336",
            hidden_size=1024,
        )

        self.assertEqual(config.vision_config.image_size, 336)
        self.assertEqual(config.vision_config.patch_size, 14)

        model = modeling_vision.ImageHDCLIPVisionModelForVLM(config, vision_from_pretrained=True)
        clip_model = transformers.AutoModel.from_pretrained("openai/clip-vit-large-patch14-336")

        for name, param in clip_model.named_parameters():
            if not name.startswith("vision_model"):
                continue
            if name.startswith("vision_model.post_layernorm"):
                continue
            try:
                vlm_param = model.clip_model.get_parameter(name)
            except AttributeError as e:
                print(name)
                print(e)
                break
            diff = torchF.mse_loss(vlm_param, param).item()
            self.assertLessEqual(diff, 1e-6)



