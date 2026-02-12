"""
In this script, we try to load the pretrained LLaVA model proveded by huggingface.
"""

from PIL import Image
# import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model: LlavaForConditionalGeneration = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf").cuda()
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "<image>\nUSER: Please describe this image.\nASSISTANT:"
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open("./__hidden/000000003188.jpg")

inputs = processor(text=prompt, images=image, return_tensors="pt")

inputs_to_cuda = {k: v.cuda() for k, v in inputs.items()}

# Generate
generate_ids = model.generate(**inputs_to_cuda, max_length=512)
output = processor.batch_decode(generate_ids, skip_special_tokens=True,
                       clean_up_tokenization_spaces=False)[0]

print(output)

