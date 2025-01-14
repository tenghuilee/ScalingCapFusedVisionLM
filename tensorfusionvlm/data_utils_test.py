import unittest
from tensorfusionvlm.data_utils import *
from tensorfusionvlm.auto_models import *
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class Args:
    data_path = "./datasets/adj_maxed_2048_qed16_finetune.json"
    image_folder = "./datasets/soft_link_image_collection"
    ckpt_path = ""
    chat_template = ""
    image_aspect_ratio = "pad"
    length_que_end = 64
    length_que_end_short = 8


class DataUtilsTest(unittest.TestCase):
    def test_load_data(self):
        args = Args()

        llm_tokenizer, clip_tokenizer, image_processor = load_tokenizers_processor(
            args.ckpt_path
        )

        train_dataset = UnionChatDataset(
            data_path=args.data_path,
            drop_no_image_instances=True,
            partitions=[
                UCDP_Image(
                    image_folder=args.image_folder,
                    image_processor=image_processor,
                    image_aspect_ratio=args.image_aspect_ratio,
                ),
                UCDP_CLIPQuery_WithPlaceholder(
                    tokenizer=llm_tokenizer,
                    length_que_end=args.length_que_end,
                    length_que_end_short=args.length_que_end_short,
                    append_special_padding=True,
                ),
                UCDP_CLIPAnswer_WithPlaceholder(
                    tokenizer=clip_tokenizer,
                    length_que_end=1,
                    length_que_end_short=1,
                ),
                UCDP_LLMQueryAnswer_WithPlaceholder(
                    tokenizer=llm_tokenizer,
                    chat_template=args.chat_template,
                    length_que_end=args.length_que_end,
                    length_que_end_short=args.length_que_end_short,
                ),
            ],
        )

        data_collector = DataCollactorForUnionChatDataset(train_dataset.partitions)

        for i, batch in enumerate(
            DataLoader(
                dataset=train_dataset,
                batch_size=2,
                collate_fn=data_collector,
            )
        ):
            print(batch)
            if i > 10:
                break
