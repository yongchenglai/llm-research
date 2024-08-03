# gptqmodel_quantize_autoround.py
import torch
from gptqmodel import GPTQModel
from gptqmodel.quantization.config import AutoRoundQuantizeConfig  # noqa: E402
from transformers import AutoTokenizer
import argparse


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description='autoawq')
    parser.add_argument("--pretrained_model_dir", type=str, help='model path')
    parser.add_argument("--quantized_model_dir", type=str, help='quant path')
    args = parser.parse_args()

    pretrained_model_id = args.pretrained_model_dir
    quantized_model_id = args.quantized_model_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
    calibration_dataset = [
        tokenizer(
            "gptqmodel is an easy-to-use model quantization library "
            "with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = AutoRoundQuantizeConfig(
        bits=4,  # 4-bit
        group_size=128  # 128 is good balance between quality and performance
    )

    model = GPTQModel.from_pretrained(
        pretrained_model_id,
        quantize_config=quantize_config,
    )

    model.quantize(calibration_dataset)

    model.save_quantized(quantized_model_id)

    tokenizer.save_pretrained(quantized_model_id)

    del model

    model = GPTQModel.from_quantized(
        quantized_model_id,
        device="cuda:0",
    )

    input_ids = torch.ones((1, 1), dtype=torch.long, device="cuda:0")
    outputs = model(input_ids=input_ids)
    print(f"output logits {outputs.logits.shape}: \n", outputs.logits)
    # inference with model.generate
    print(
        tokenizer.decode(
            model.generate(
                **tokenizer("gptqmodel is", return_tensors="pt").to(model.device)
            )[0]
        )
    )
