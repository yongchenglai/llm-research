# llama_chat_gradio.py
import gradio as gr
import time
from transformers import AutoTokenizer, \
    AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
import torch, sys, os
import json
import pandas
import argparse


with gr.Blocks() as demo:
    # gr.Markdown("""<h1><center>AI智能助手</center></h1>""")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    state = gr.State()

    with gr.Row():
        sent_bt = gr.Button("发送")
        re_generate = gr.Button("重新回答")
        clear = gr.Button("新话题")

    with gr.Accordion("生成参数", open=False):
        slider_temp = gr.Slider(
            minimum=0,
            maximum=1,
            label="temperature",
            value=0.3)

        slider_top_p = gr.Slider(
            minimum=0.5,
            maximum=1,
            label="top_p",
            value=0.95)

        slider_context_times = gr.Slider(
            minimum=0,
            maximum=5,
            label="上文轮次",
            value=0,
            step=2.0)

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, temperature, top_p, slider_context_times):
        if pandas.isnull(history[-1][1])==False:
            history[-1][1] = None
            yield history

        slider_context_times = int(slider_context_times)
        history_true = history[1:-1]
        prompt = ''
        if slider_context_times > 0:
            prompt += '\n'.join([("<s>Human: "+one_chat[0].replace('<br>', '\n')+'\n</s>'
                if one_chat[0] else '')
                    + "<s>Assistant: "+one_chat[1].replace('<br>', '\n')+'\n</s>'
                for one_chat in history_true[-slider_context_times:]])

        prompt += "<s>Human: "+history[-1][0].replace('<br>', '\n')+"\n</s><s>Assistant:"

        inputs = tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=False)

        input_ids = tokenizer(
            inputs,  # [prompt],
            return_tensors="pt",
            add_special_tokens=False).input_ids[:, -512:].to('cuda')

        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": 1.3,
            "streamer": streamer,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

        # Use Thread to run generation in background
        # Otherwise, the process is blocked until generation is complete
        # and no streaming effect can be observed.
        thread = Thread(target=model.generate, kwargs=generate_input)
        thread.start()

        start_time = time.time()
        bot_message = ""
        print('Human:', history[-1][0])
        print('Assistant: ', end='', flush=True)
        for new_text in streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue
            if new_text != '</s>':
                bot_message += new_text
            if 'Human:' in bot_message:
                bot_message = bot_message.split('Human:')[0]
            history[-1][1] = bot_message
            yield history
        end_time = time.time()

        print()
        print('生成耗时：', end_time-start_time,
              '文字长度：', len(bot_message),
              '字耗时：', (end_time-start_time)/len(bot_message))

    msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        fn=bot,
        inputs=[chatbot, slider_temp, slider_top_p, slider_context_times],
        outputs=chatbot
    )

    sent_bt.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[chatbot, slider_temp, slider_top_p, slider_context_times],
        outputs=chatbot
    )

    re_generate.click(
        fn=bot,
        inputs=[chatbot, slider_temp, slider_top_p, slider_context_times],
        outputs=chatbot)

    clear.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help='mode name or path')
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument('--quant', type=int, choices=[4, 8], default=0,
                        help='Enable 4-bit or 8-bit precision loading')
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token


    # Load the model
    if args.quant == 4:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map="auto",
            torch_dtype=args.torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=args.torch_dtype,
                llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
            ),
            low_cpu_mem_usage=True,
        )
    elif args.quant == 8:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map="auto",
            torch_dtype=args.torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=args.torch_dtype,
            ),
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map="auto",
            torch_dtype=args.torch_dtype,
            trust_remote_code=True
        )

    model.eval()
    print(model)

    # Streaming Mode
    # Besides using TextStreamer, we can also use TextIteratorStreamer
    # which stores print-ready text in a queue,
    # to be used by a downstream application as an iterator:
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    demo.queue().launch(
        share=args.share,
        debug=True,
        server_name=args.server_name,
        server_port=args.server_port,
    )


