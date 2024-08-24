# qwen2_audio_web_gradio.py
import gradio as gr
import modelscope_studio as mgr
import librosa
from transformers import Qwen2AudioForConditionalGeneration
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import argparse


def _get_args():

    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="'Qwen/Qwen2-Audio-7B-Instruct'")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument('--quant', type=int, choices=[4, 8], default=0,
                        help='Enable 4-bit or 8-bit precision loading')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
    parser.add_argument('--multi-gpus', action='store_true', default=False,
                        help='use multi-gpus')
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--inbrowser", action="store_true", default=True,
                        help="Automatically launch the interface "
                             "in a new tab on the default browser.")

    args = parser.parse_args()

    return args


def add_text(chatbot, task_history, input):
    text_content = input.text
    content = []
    if len(input.files) > 0:
        for i in input.files:
            content.append({'type': 'audio', 'audio_url': i.path})
    if text_content:
        content.append({'type': 'text', 'text': text_content})
    task_history.append({"role": "user", "content": content})

    chatbot.append([{
        "text": input.text,
        "files": input.files,
    }, None])
    return chatbot, task_history, None


def add_file(chatbot, task_history, audio_file):
    """Add audio file to the chat history."""
    task_history.append({"role": "user", "content": [{"audio": audio_file.name}]})
    chatbot.append((f"[Audio file: {audio_file.name}]", None))
    return chatbot, task_history


def reset_user_input():
    """Reset the user input field."""
    return gr.Textbox.update(value='')


def reset_state(task_history):
    """Reset the chat history."""
    return [], []


def regenerate(chatbot, task_history):
    """Regenerate the last bot response."""
    if task_history and task_history[-1]['role'] == 'assistant':
        task_history.pop()
        chatbot.pop()

    if task_history:
        chatbot, task_history = predict(chatbot, task_history)
        
    return chatbot, task_history


def predict(chatbot, task_history):
    """Generate a response from the model."""
    print(f"{task_history=}")
    print(f"{chatbot=}")

    text = processor.apply_chat_template(
        conversation=task_history,
        add_generation_prompt=True,
        tokenize=False)

    audios = []
    for message in task_history:
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "audio":
                    audios.append(librosa.load(
                        element['audio_url'],
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    print(f"{text=}")
    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True)

    if not _get_args().cpu_only:
        inputs["input_ids"] = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    # Convert a list of lists of token ids into a list of strings by calling decode.
    response = processor.batch_decode(
        sequences=generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    print(f"{response=}")

    task_history.append({'role': 'assistant', 'content': response})
    chatbot.append((None, response))  # Add the response to chatbot

    return chatbot, task_history


def _launch_demo(args):
    with gr.Blocks() as demo:
        # gr.Markdown("""<center><font size=8>Qwen2-Audio-Instruct Bot</center>""")
        chatbot = mgr.Chatbot(label='Qwen2-Audio-7B-Instruct',
                              elem_classes="control-height",
                              height=400)

        user_input = mgr.MultimodalInput(
            interactive=True,
            sources=['microphone', 'upload'],
            submit_button_props=dict(value="Submit"),
            upload_button_props=dict(value="Upload", show_progress=True),
        )
        task_history = gr.State([])

        with gr.Row():
            empty_bin = gr.Button("Clear History")
            regen_btn = gr.Button("Regenerate")

        user_input.submit(
            fn=add_text,
            inputs=[chatbot, task_history, user_input],
            outputs=[chatbot, task_history, user_input]
        ).then(
            predict,
            [chatbot, task_history],
            [chatbot, task_history],
            show_progress=True
        )

        empty_bin.click(fn=reset_state,
                        outputs=[chatbot, task_history],
                        show_progress=True)

        regen_btn.click(fn=regenerate,
                        inputs=[chatbot, task_history],
                        outputs=[chatbot, task_history],
                        show_progress=True)

    demo.queue().launch(
        share=args.share,
        debug=True,
        show_api=False,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
        ssl_verify=False,
    )


if __name__ == "__main__":

    args = _get_args()

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"
    """
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    )
    """

    if args.quant == 4:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=args.torch_dtype,
            # attn_implementation="flash_attention_2",
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
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map=device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=args.torch_dtype,
            ),
            low_cpu_mem_usage=True
        )
    else:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map=device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=True
        )

    model.eval()
    print(model)
    model.generation_config.max_new_tokens = 2048  # For chat.

    print("generation_config", model.generation_config)
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # resume_download=True,
    )

    _launch_demo(args)

