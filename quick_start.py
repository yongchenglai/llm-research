# quick_start.py
# python quick_start.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained(
    'FlagAlpha/Atom-7B-Chat',
    device_map=device_map,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    trust_remote_code=True,
    attn_implementation="flash_attention_2")

model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    'FlagAlpha/Atom-7B-Chat',
    use_fast=False)

tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(
    ['<s>Human: 详细介绍一下中国\n</s><s>Assistant: '],
    return_tensors="pt",
    add_special_tokens=False).input_ids

if torch.cuda.is_available():
    input_ids = input_ids.to('cuda')

generate_input = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    "pad_token_id": tokenizer.pad_token_id
}

generate_ids = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)


