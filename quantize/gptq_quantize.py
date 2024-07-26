# gptq_quantize.py
# https://github.com/AutoGPTQ/AutoGPTQ
# pip install auto-gptq optimum

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import argparse
import time
import torch
import json
import random
from datasets import Dataset
import logging

def load_data(data_path, tokenizer, n_samples):

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description='autoawq')
    parser.add_argument("--pretrained_model_dir", type=str, help='model path')
    parser.add_argument("--quantized_model_dir", type=str, help='quant path')
    parser.add_argument("--num_samples", type=int, default=128,
                        help="how many samples will be used to quantize model")
    args = parser.parse_args()

    model_path = args.pretrained_model_dir
    quant_path = args.quantized_model_dir

    quantize_config = BaseQuantizeConfig(
        bits=8,  # 4 or 8
        group_size=128,
        damp_percent=0.01,
        # set to False can significantly speed up inference
        # but the perplexity may slightly bad
        desc_act=False,
        static_groups=False,
        sym=True,
        true_sequential=True,
        model_name_or_path=None,
        model_file_base_name="model"
    )
    max_len = 8192

    # Load your tokenizer and model with AutoGPTQ
    # To learn about loading model to multiple GPUs,
    # visit https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/tutorial/
    # 02-Advanced-Model-Loading-and-Best-Practice.md
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load un-quantized model, by default,
    # the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
    print(model)
    # quantize model, the examples should be list of dict
    # whose keys can only be "input_ids" and "attention_mask"
    # model.quantize(examples)
    examples = load_data("dataset/alpaca_data_cleaned.json",
                         tokenizer, args.num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"],
         "attention_mask": example["attention_mask"]} for example in examples
    ]

    start = time.time()
    model.quantize(examples_for_quant, batch_size=1)
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    # save quantized model
    model.save_quantized(save_dir=quant_path)

    # save quantized model using safetensors
    # model.save_quantized(save_dir=quant_path, use_safetensors=True)

