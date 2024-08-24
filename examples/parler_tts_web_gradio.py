# parler_tts_web_gradio.py
import gradio as gr
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoFeatureExtractor, AutoTokenizer, set_seed
from transformers import BitsAndBytesConfig
import argparse


default_text = "Please surprise me and speak in whatever voice you enjoy."
default_description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

examples = [
    [
        "Hey, how are you doing today?",
        "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    ],
    [
        "'This is the best time of my life, Bartley,' she said happily.",
        "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio.",
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.	",
        "A male speaker with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
    ],
    [
        "montrose also after having experienced still more variety of good and bad fortune threw down his arms and retired out of the kingdom",
        "A male speaker with a low-pitched voice delivering his words at a fast pace in a small, confined space with a lot of background noise and an animated tone.",
    ],
]


css = """
        #share-btn-container {
            display: flex;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            background-color: #000000;
            justify-content: center;
            align-items: center;
            border-radius: 9999px !important;
            width: 13rem;
            margin-top: 10px;
            margin-left: auto;
            flex: unset !important;
        }
        #share-btn {
            all: initial;
            color: #ffffff;
            font-weight: 600;
            cursor: pointer;
            font-family: 'IBM Plex Sans', sans-serif;
            margin-left: 0.5rem !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            right:0;
        }
        #share-btn * {
            all: unset !important;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
"""


if __name__ == "__main__":

    # Argparser
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="parler-tts/parler-tts-large-v1")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument('--quant', type=int, choices=[4, 8], default=0,
                        help='Enable 4-bit or 8-bit precision loading')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface "
                             "in a new tab on the default browser.")
    args = parser.parse_args()

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path)

    SAMPLE_RATE = feature_extractor.sampling_rate
    SEED = 41

    model = ParlerTTSForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path)
    model.to(device)
    print(model)

    def gen_tts(text, description):

        inputs = tokenizer(description, return_tensors="pt").to(device)
        prompt = tokenizer(text, return_tensors="pt").to(device)

        set_seed(SEED)
        generation = model.generate(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            do_sample=True,
            temperature=1.0
        )
        audio_arr = generation.cpu().numpy().squeeze()

        return (SAMPLE_RATE, audio_arr)


    with gr.Blocks(css=css) as demo:
        # gr.Markdown("# Parler-TTS </div>")
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    lines=2,
                    value=default_text,
                    elem_id="input_text")
                description = gr.Textbox(
                    label="Description",
                    lines=2,
                    value=default_description,
                    elem_id="input_description")
                run_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column():
                audio_out = gr.Audio(
                    label="Parler-TTS generation",
                    type="numpy",
                    elem_id="audio_out")

        gr.Examples(
            examples=examples,
            fn=gen_tts,
            inputs=[input_text, description],
            outputs=[audio_out],
            cache_examples=True)

        run_button.click(
            fn=gen_tts,
            inputs=[input_text, description],
            outputs=[audio_out],
            queue=True)

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
