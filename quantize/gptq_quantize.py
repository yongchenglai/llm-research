# gptq_quantize.py
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='autoawq')
    parser.add_argument("--model_path", type=str, help='model path')
    parser.add_argument("--quant_path", type=str, help='quant path')
    args = parser.parse_args()

    model_path = args.model_path
    quant_path = args.quant_path

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
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config)



