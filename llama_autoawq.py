# llama_autoawq.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from transformers import AwqConfig, AutoConfig
from huggingface_hub import HfApi
import argparse


if __name__ == "__main__":
    '''
    python llama_autoawq.py \
    --model_path='./FlagAlpha/Atom-7B-Chat/' \
    --quant_path='./FlagAlpha/Atom-7B-Chat-awq/'
    '''
    parser = argparse.ArgumentParser(description='autoawq')
    parser.add_argument("--model_path", type=str, help='model path')
    parser.add_argument("--quant_path", type=str, help='quant path')
    args = parser.parse_args()

    model_path = args.model_path
    quant_path = args.quant_path

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)

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
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


