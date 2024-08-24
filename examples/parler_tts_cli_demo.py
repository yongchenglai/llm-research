# parler_tts_cli_demo.py
"""
docker exec -it parler-tts bash
python3 parler_tts_cli_demo.py
"""
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "/models/parler-tts/parler-tts-large-v1"

    tts_model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tts_model)

    prompt = "Hey, how are you doing today?"
    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

    input_ids = tts_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write("/models/parler_tts_out.wav", audio_arr, model.config.sampling_rate)


