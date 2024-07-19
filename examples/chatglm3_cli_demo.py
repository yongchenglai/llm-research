# chatglm3_cli_demo.py
import os
import torch
import platform
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import argparse

# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# add .quantize(bits=4, device="cuda").cuda() before .eval() to use int4 model
# must use cuda to load int4 model

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用ChatGLM3-6B模型,输入内容即可进行对话,clear清空对话历史,stop终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str,
                            help='mode name or path')
        args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        # torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True)

    model.quantize(bits=4, device="cuda").cuda()
    model.eval()
    print(model)

    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values \
                in model.stream_chat(
                    tokenizer,
                    query,
                    history=history,
                    top_p=1,
                    temperature=0.01,
                    past_key_values=past_key_values,
                    return_past_key_values=True):

            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()
