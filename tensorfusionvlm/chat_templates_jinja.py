
# Chat templates,
# reference fastchat/conversation.py

vicuna = """
{%- if messages[0]['role'] == 'system' -%}
    {%- set loop_messages = messages[1:] -%}
    {%- set system_message = messages[0]['content'].strip() + ' '  -%}
{%- else -%}
    {%- set loop_messages = messages -%}
    {%- set system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n" -%}
{%- endif -%}
{{ bos_token + system_message }}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif -%}
    {%- set content = message['content'] -%}
    {%- if message['role'] == 'user' -%}
        {{ 'USER: ' + content.strip() + ' ' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ 'ASSISTANT: ' + content.strip() + eos_token }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ 'ASSISTANT:' }}
{%- endif -%}
"""

llama2 = """{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if loop_messages|length == 0 and system_message %}{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""

def get_chat_template(llm_name: str):
    llm_name = llm_name.lower()
    if 'vicuna' in llm_name:
        return vicuna
    else:
        return None
