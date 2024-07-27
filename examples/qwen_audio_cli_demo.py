# qwen_audio_cli_demo.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
import torch
import argparse

torch.manual_seed(1234)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Qwen-Audio-Chat Demo")
    parser.add_argument("--model_name_or_path", type=str,
                        default='Qwen/Qwen-Audio-Chat',
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--audio", type=str,
                        default='/models/audio/1272-128104-0000.flac')
    parser.add_argument("--text", type=str,
                        default='what is that sound?',)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # attn_implementation="flash_attention",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
        ),
        low_cpu_mem_usage=True,
    )

    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    model.eval()
    # print(model)

    query = tokenizer.from_list_format([
        {'audio': args.audio},
        {'text': args.text},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)



