import sys
import unittest

import copy
from tensorfusionvlm.chat_parser import *
import difflib
import transformers


class TestPromptOutput(unittest.TestCase):

    def _chat_parser_general(
        self,
        tokenizer_name: str,
        conversation_template: str,
        add_endoftext: bool = True,
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

        if conversation_template in ["llama-2", "llama-3"]:
            tokenizer.pad_token = tokenizer.eos_token
        
        messages = [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant",
                "content": "The weather is currently sunny with a temperature of 25 degrees Celsius."},
            {"role": "user", "content": "Can you proof the Pythagorean theorem?"},
            {"role": "assistant", "content": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides."},
        ]

        target = tokenizer.apply_chat_template(messages) # type: torch.Tensor

        chat_record = ChatRecordSimple.from_dict(messages)
        chat_parser = ChatQEDParser(tokenizer=tokenizer)

        predict = chat_parser.parse_ids_tps(chat_record.apply_template(
            conversation_template,
            add_endoftext=add_endoftext))[0]

        # self.assertEqual(str(parser), tokenizer.apply_chat_template(messages, tokenize=False))
        
        self.assertEqual(target, predict)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you solve the equation 2x + 3 = 7?"},
            {"role": "assistant", "content": "The solution to the equation 2x + 3 = 7 is x = 1."},
            {"role": "user", "content": "Can you proof the Pythagorean theorem?"},
            {"role": "assistant", "content": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides."},
        ]

        target = tokenizer.apply_chat_template(messages) # type: torch.Tensor

        chat_record = ChatRecordSimple.from_dict(messages)
        predict = chat_parser.parse_ids_tps(chat_record.apply_template(
            conversation_template, add_endoftext=add_endoftext))[0]
        
        # self.assertEqual(str(parser), tokenizer.apply_chat_template(messages, tokenize=False))

        self.assertEqual(target, predict)

    
    def test_llama_2(self):
        self._chat_parser_general(
            tokenizer_name="./llama_hf_mirror/llama_hf/llama-2-7b-chat",
            conversation_template="llama-2",
            add_endoftext=True,
        )
    
    def test_phi_3(self):
        self._chat_parser_general(
            tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
            conversation_template="microsoft-phi-3",
            add_endoftext=True,
        )

    def test_qwen_7b(self):
        self._chat_parser_general(
            tokenizer_name="Qwen/Qwen2-7B-Instruct",
            conversation_template="qwen-7b-chat",
            add_endoftext=True,
        )
    
    def test_llama_3(self):
        self._chat_parser_general(
            tokenizer_name="./llama_hf_mirror/llama_hf/llama-3-8b-instruct",
            conversation_template="llama-3",
            add_endoftext=True,
        )
