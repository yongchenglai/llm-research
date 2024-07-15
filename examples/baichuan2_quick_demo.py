# baichuan2_quick_demo.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "baichuan-inc/Baichuan2-7B-Chat"
    parser.add_argument("--model_name_or_path", type=str, help='mode name or path')
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
       args.model_name_or_path,
       use_fast=False,
       trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
       args.model_name_or_path,
       device_map="auto",
       torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
       trust_remote_code=True)

    model.generation_config = GenerationConfig.from_pretrained(
       args.model_name_or_path)

    print(model)

    messages = []
    messages.append({"role": "user", "content": "解释一下“温故而知新”"})
    response = model.chat(tokenizer, messages)
    print(response)





