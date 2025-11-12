import os.path
import string

import pandas as pd
import torch
import vlmeval
import vlmeval.dataset
from PIL import Image

import llava
import visionzip
import vispruner


class VisionZipLLaVA(vlmeval.LLaVA):

    def __init__(
        self,
        model_path='liuhaotian/llava-v1.5-7b',
        dominant=54,
        contextual=10,
        **kwargs,
    ):
        # warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert os.path.exists(model_path) or vlmeval.splitlen(model_path) == 2
        self.system_prompt = (
            'A chat between a curious human and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = '</s>'

        model_name = llava.get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            llava.load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map="cpu",
            )
        )

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True)  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        self.dominant = dominant
        self.contextual = contextual

        self.model = visionzip.visionzip(
            self.model,
            dominant=dominant,
            contextual=contextual,
        )


class VisPrunerLLaVA(vlmeval.BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path='liuhaotian/llava-v1.5-7b',
        visual_token_num=576,
        important_ratio=0.5,
        **kwargs,
    ):
        # warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert os.path.exists(model_path) or vlmeval.splitlen(model_path) == 2
        self.system_prompt = (
            'A chat between a curious human and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = '</s>'

        model_name = vispruner.get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            vispruner.load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map="cpu",
                # kwargs for vispruner
                visual_token_num=visual_token_num,
                important_ratio=important_ratio,
            )
        )
        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True)  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if vlmeval.dataset.DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if (
            'hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if vlmeval.cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if vlmeval.cn_string(
                prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])
        return text, images

    def chat_inner(self, message, dataset=None):

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += 'USER: ' if utter['role'] == 'user' else 'ASSISTANT: '
            content, images_sub = self.concat_tilist(utter['content'])
            prompt += content
            images.extend(images_sub)
            prompt += ' ' if utter['role'] == 'user' else self.stop_str
        assert message[-1]['role'] == 'user', message
        prompt += 'ASSISTANT: '

        images = [Image.open(s).convert('RGB') for s in images]

        class __args:
            image_aspect_ratio = 'pad'
        args = __args()
        image_tensor = vispruner.process_images(
            images, self.image_processor, args).to('cuda', dtype=torch.float16)

        input_ids = vispruner.tokenizer_image_token(
            prompt, self.tokenizer, vispruner.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [self.stop_str]
        stopping_criteria = vispruner.KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        return output

    def generate_inner(self, message, dataset=None):

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert('RGB') for s in images]

        class __args:
            image_aspect_ratio = 'pad'
        args = __args()
        if images:
            image_tensor = vispruner.process_images(
                images, self.image_processor, args).to('cuda', dtype=torch.float16)
        else:
            image_tensor = None

        prompt = self.system_prompt + 'USER: ' + content + ' ASSISTANT: '

        input_ids = vispruner.tokenizer_image_token(
            prompt, self.tokenizer, vispruner.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [self.stop_str]
        stopping_criteria = vispruner.KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            try:
                output_ids, visual_token_num = self.model.generate(
                    input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
            except Exception as e:
                print(e)
                return ""

            # print(visual_token_num)
        output = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        return output
