# The extension of chat templates

from fastchat.conversation import *

_max_super_class = max([x.value for x in list(SeparatorStyle)])


class SeparatorStyleExtend(IntEnum):
    PHI_3 = _max_super_class + 1
    LLAMA_3 = auto()


#   "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",

# <|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# Can you solve the equation 2x + 3 = 7?<|end|>
# <|assistant|>
# The solution to the equation 2x + 3 = 7 is x = 1.<|end|>
# <|endoftext|>

register_conv_template(
    Conversation(
        name="microsoft-phi-3",
        system_template="<|system|>\n{system_message}<|end|>\n",
        system_message=None,
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyleExtend.PHI_3,
        sep="",
        sep2="<|end|>",
    )
)

register_conv_template(
    Conversation(
        name="phi-3",
        system_template="{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyleExtend.PHI_3,
        sep="",
        sep2="<|end|>",
    )
)

register_conv_template(
    Conversation(
        name="llama-2-vision",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep="<s>",
        sep2=" </s>",
    )
)

register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        system_message=None,
        roles=(
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
        ),
        sep_style=SeparatorStyleExtend.LLAMA_3,
        sep="<|begin_of_text|>",
        sep2="<|eot_id|>",
    )
)


register_conv_template(
    Conversation(
        name="llama-3-vision",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        system_message="You are a helpful assistant.",
        roles=(
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
        ),
        sep_style=SeparatorStyleExtend.LLAMA_3,
        sep="<|begin_of_text|>",
        sep2="<|eot_id|>",
    )
)
