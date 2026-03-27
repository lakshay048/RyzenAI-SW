# generate_inference_model.py
# This script generates the inference_model.json file for the Llama-3.2 model.
import json
import os
from transformers import AutoTokenizer

model_path = "phi-4-mini-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix"

print('Loading the tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path)
chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "{Content}"},
]


template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

json_template = {
  "Name": "phi-4-mini-instruct-awq-g128-int4-asym-bf16-onnx-ryzen-strix",
  "PromptTemplate": {
    "assistant": "{Content}",
    "prompt": template
  }
}

print('Generating inference_model.json with the following template:')
print(json.dumps(json_template, indent=2))
json_file = os.path.join(model_path, "inference_model.json")

with open(json_file, "w") as f:
    json.dump(json_template, f, indent=2)