# awq_quantize.py
# Quantize Your Own Model with AutoAWQ
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from transformers import AwqConfig, AutoConfig
import argparse
from huggingface_hub import HfApi

'''
python awq_quantize.py \
--pretrained_model_dir='./Qwen/Qwen2-7B-Instruct/' \
--quantized_model_dir='./Qwen/Qwen2-7B-Instruct-AWQ/'
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='autoawq')
    parser.add_argument("--pretrained_model_dir", type=str, help='model path')
    parser.add_argument("--quantized_model_dir", type=str, help='quant path')
    args = parser.parse_args()

    model_path = args.pretrained_model_dir
    quant_path = args.quantized_model_dir

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        # device_map="auto",
        safetensors=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)

    print(model)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # modify the config file so that it is compatible with transformers integration
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    # the pretrained transformers model is stored
    # in the model attribute + we need to pass a dict
    model.model.config.quantization_config = quantization_config

    # save model weights
    model.save_quantized(quant_path, safetensors=True, shard_size="5GB")
    tokenizer.save_pretrained(quant_path)

