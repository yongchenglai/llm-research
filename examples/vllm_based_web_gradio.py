# vllm_based_web_gradio.py
from typing import List
import argparse
import gradio as gr
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="openbmb/MiniCPM-1B-sft-bf16")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    # for MiniCPM-1B and MiniCPM-2B  model, max_tokens should be set to 2048
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--quantization", type=str, default=None)

    # 生成参数
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--repetition_penalty", type=float, default=1.02)

    args = parser.parse_args()

    return args


def vllm_gen(dialog: List, top_p: float, temperature: float, max_dec_len: int):
    """generate model output with huggingface api

    Args:
        query (str): actual model input.
        top_p (float): only the smallest set of most probable tokens with probabilities
                       that add up to top_p or higher are kept for generation.
        temperature (float): Strictly positive float value used to
                             modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        str: real-time generation results of hf model
    """
    assert len(dialog) % 2 == 1
    prompt = tokenizer.apply_chat_template(
        dialog,
        tokenize=False,
        add_generation_prompt=False)
    token_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop": "<|im_end|>",
        "stop_token_ids": token_ids,
        "ignore_eos": False,
        "max_tokens": max_dec_len,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)[0]
    generated_text = outputs.outputs[0].text
    return generated_text


def generate(
    chat_history: List,
    query: str,
    top_p: float,
    temperature: float,
    max_dec_len: int,
):
    """generate after hitting "submit" button

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]].
                             list that stores all QA records
        query (str): query of current round
        top_p (float): only the smallest set of most probable tokens with probabilities
                        that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to
                             modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n], [q_n+1, a_n+1]].
              chat_history + QA of current round.
    """
    assert query != "", "Input must not be empty!!!"
    # apply chat template
    model_input = []
    for q, a in chat_history:
        model_input.append({"role": "user", "content": q})
        model_input.append({"role": "assistant", "content": a})
    model_input.append({"role": "user", "content": query})
    # yield model generation
    model_output = vllm_gen(model_input, top_p, temperature, max_dec_len)
    chat_history.append([query, model_output])
    return gr.update(value=""), chat_history


def regenerate(chat_history: List, top_p: float, temperature: float, max_dec_len: int):
    """re-generate the answer of last round's query

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]].
                             list that stores all QA records
        top_p (float): only the smallest set of most probable tokens with
                       probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. chat_history
    """
    assert len(chat_history) >= 1, "History is empty. Nothing to regenerate!!"
    # apply chat template
    model_input = []
    for q, a in chat_history[:-1]:
        model_input.append({"role": "user", "content": q})
        model_input.append({"role": "assistant", "content": a})
    model_input.append({"role": "user", "content": chat_history[-1][0]})
    # yield model generation
    model_output = vllm_gen(model_input, top_p, temperature, max_dec_len)
    chat_history[-1][1] = model_output
    return gr.update(value=""), chat_history


def clear_history():
    """clear all chat history

    Returns:
        List: empty chat history
    """
    return []


def reverse_last_round(chat_history):
    """reverse last round QA and keep the chat history before
    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]].
        list that stores all QA records
    Returns:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n-1, a_n-1]].
        chat_history without last round.
    """
    assert len(chat_history) >= 1, "History is empty. Nothing to reverse!!"
    return chat_history[:-1]


if __name__ == "__main__":
    args = get_args()

    # init model torch dtype
    torch_dtype = args.torch_dtype
    if torch_dtype == "" or torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype == "float32":
        torch_dtype = torch.float32
    elif torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Invalid torch dtype: {torch_dtype}")

    # init model and tokenizer
    path = args.model_path
    llm = LLM(
        model=path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch_dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        quantization=args.quantization
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True)

    print(llm)

    # launch gradio demo
    # with gr.Blocks(theme="soft") as demo:
    with gr.Blocks() as demo:
        # gr.Markdown("""# MiniCPM Gradio Demo""")

        with gr.Row():
            """"""
            with gr.Column(scale=1):
                top_p = gr.Slider(0, 1, value=0.8, step=0.1, label="top_p")
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="temperature")
                max_dec_len = gr.Slider(
                    minimum=1, maximum=args.max_tokens,
                    value=args.max_tokens, step=1, label="max_tokens")

            with gr.Column(scale=5):
                # chatbot = gr.Chatbot(bubble_full_width=False, height=400)
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(
                    label="User", placeholder="Input your query here!", lines=4)
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")
                    regen = gr.Button("Regenerate")
                    reverse = gr.Button("Reverse")

        submit.click(
            fn=generate,
            inputs=[chatbot, user_input, top_p, temperature, max_dec_len],
            # inputs=[chatbot, user_input, args.top_p, args.temperature, args.max_tokens],
            outputs=[user_input, chatbot])
        regen.click(
            fn=regenerate,
            inputs=[chatbot, top_p, temperature, max_dec_len],
            # inputs=[chatbot, args.top_p, args.temperature, args.max_tokens],
            outputs=[user_input, chatbot])
        clear.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot])
        reverse.click(
            fn=reverse_last_round,
            inputs=[chatbot],
            outputs=[chatbot])

    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True)




