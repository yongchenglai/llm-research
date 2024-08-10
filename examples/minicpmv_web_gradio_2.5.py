#!/usr/bin/env python
# encoding: utf-8
# minicpmv_web_gradio_2.5.py
import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import GPUtil


# README, How to run demo on different devices

# For Nvidia GPUs.
# python web_demo_2.5.py --device cuda

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo_2.5.py --device mps


# Load model
# model_path = 'openbmb/MiniCPM-Llama3-V-2_5'
# if 'int4' in model_path:

# if not args.is_4bit:
#    if device == 'mps':
#        print('Error: running int4 model with bitsandbytes on Mac is not supported right now.')
#        exit()
#    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
# else:


ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 2.5'

form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    #'value': 'Beam Search',
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}
# Beam Form
num_beams_slider = {
    'minimum': 0,
    'maximum': 5,
    'value': 3,
    'step': 1,
    'interactive': True,
    'label': 'Num Beams'
}

repetition_penalty_slider = {
    'minimum': 0,
    'maximum': 3,
    'value': 1.2,
    'step': 0.01,
    'interactive': True,
    'label': 'Repetition Penalty'
}

repetition_penalty_slider2 = {
    'minimum': 0,
    'maximum': 3,
    'value': 1.05,
    'step': 0.01,
    'interactive': True,
    'label': 'Repetition Penalty'
}

max_new_tokens_slider = {
    'minimum': 1,
    'maximum': 4096,
    'value': 1024,
    'step': 1,
    'interactive': True,
    'label': 'Max New Tokens'    
}


top_p_slider = {
    'minimum': 0,
    'maximum': 1,
    'value': 0.8,
    'step': 0.05,
    'interactive': True,
    'label': 'Top P'    
}

top_k_slider = {
    'minimum': 0,
    'maximum': 200,
    'value': 100,
    'step': 1,
    'interactive': True,
    'label': 'Top K'    
}

temperature_slider = {
    'minimum': 0,
    'maximum': 2,
    'value': 0.7,
    'step': 0.05,
    'interactive': True,
    'label': 'Temperature'    
}


def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )


def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    default_params = {"num_beams": 3,
                      "repetition_penalty": 1.2,
                      "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None, None
    try:
        image = img.convert('RGB')
        answer = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        answer = res.replace('</box>', '')
        return 0, answer, None, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None


def upload_img(image, _chatbot, _app_session):
    image = Image.fromarray(image)

    _app_session['sts'] = None
    _app_session['ctx'] = []
    _app_session['img'] = image
    _chatbot.append(('', 'Image uploaded successfully, you can talk to me now'))
    return _chatbot, _app_session


def respond(
    _question,
    _chat_bot,
    _app_cfg,
    params_form,
    num_beams,
    repetition_penalty,
    repetition_penalty_2,
    top_p,
    top_k,
    temperature,
):
    if _app_cfg.get('ctx', None) is None:
        _chat_bot.append((_question, 'Please upload an image to start'))
        return '', _chat_bot, _app_cfg

    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({"role": "user", "content": _question})
    else:
        _context = [{"role": "user", "content": _question}] 
    print('<User>:', _question)

    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': num_beams,
            'repetition_penalty': repetition_penalty,
            "max_new_tokens": 896 
        }
    else:
        params = {
            'sampling': True,
            'top_p': top_p,
            'top_k': top_k,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty_2,
            "max_new_tokens": 896 
        }
    code, _answer, _, sts = chat(_app_cfg['img'], _context, None, params)
    print('<Assistant>:', _answer)

    _context.append({"role": "assistant", "content": _answer}) 
    _chat_bot.append((_question, _answer))
    if code == 0:
        _app_cfg['ctx']=_context
        _app_cfg['sts']=sts
    return '', _chat_bot, _app_cfg


def regenerate_button_clicked(
    _question,
    _chat_bot,
    _app_cfg,
    params_form,
    num_beams,
    repetition_penalty,
    repetition_penalty_2,
    top_p,
    top_k,
    temperature,
):
    if len(_chat_bot) <= 1:
        _chat_bot.append(('Regenerate', 'No question for regeneration.'))
        return '', _chat_bot, _app_cfg
    elif _chat_bot[-1][0] == 'Regenerate':
        return '', _chat_bot, _app_cfg
    else:
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
    return respond(_question, _chat_bot, _app_cfg, params_form, num_beams, 
                   repetition_penalty, repetition_penalty_2,
                   top_p, top_k, temperature)



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            params_form = create_component(form_radio, comp='Radio')
            with gr.Accordion("Beam Search") as beams_according:
                num_beams = create_component(num_beams_slider)
                repetition_penalty = create_component(repetition_penalty_slider)

            with gr.Accordion("Sampling") as sampling_according:
                top_p = create_component(top_p_slider)
                top_k = create_component(top_k_slider)
                temperature = create_component(temperature_slider)
                repetition_penalty_2 = create_component(repetition_penalty_slider2)
            regenerate = create_component({'value': 'Regenerate'}, comp='Button')

        with gr.Column(scale=3, min_width=500):
            app_session = gr.State({'sts': None, 'ctx': None, 'img': None})
            bt_pic = gr.Image(label="Upload an image to start")
            chat_bot = gr.Chatbot(label=f"Chat with {model_name}")
            txt_message = gr.Textbox(label="Input text")
            
            regenerate.click(
                regenerate_button_clicked,
                [txt_message, chat_bot, app_session, params_form,
                 num_beams, repetition_penalty,
                 repetition_penalty_2, top_p, top_k, temperature],
                [txt_message, chat_bot, app_session]
            )
            txt_message.submit(
                respond, 
                [txt_message, chat_bot, app_session, params_form,
                 num_beams, repetition_penalty,
                 repetition_penalty_2, top_p, top_k, temperature],
                [txt_message, chat_bot, app_session]
            )
            bt_pic.upload(lambda: None, None, chat_bot, queue=False).\
                then(upload_img, inputs=[bt_pic, chat_bot, app_session],
                     outputs=[chat_bot, app_session])


if __name__ == "__main__":

    # Argparser
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="openbmb/MiniCPM-Llama3-V-2_5")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument('--quant', type=int, choices=[4, 8], default=0,
                        help='Enable 4-bit or 8-bit precision loading')
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    # parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
    args = parser.parse_args()

    # device = args.device
    # assert device in ['cuda', 'mps']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.bfloat16 \
        if torch.cuda.is_available() and \
        torch.cuda.get_device_capability()[0] >= 8 \
        else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True)

    model_path = args.model_name_or_path

    # Load the model
    if args.quant == 4:
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=TORCH_TYPE,
                llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
            ),
            low_cpu_mem_usage=True,
        )
    elif args.quant == 8:
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=TORCH_TYPE,
            ),
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True
        )

    model.eval()
    print(model)

    gpu_usage = GPUtil.getGPUs()[0].memoryUsed
    print(f"量化后显存占用: {round(gpu_usage / 1024, 2)}GB")

    # launch
    demo.launch(
        share=args.share,
        debug=True,
        show_api=False,
        server_port=args.server_port,
        server_name=args.server_name,
    )




