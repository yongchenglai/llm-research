# whisper_cli_demo.py
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import BitsAndBytesConfig
from datasets import load_dataset


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_path = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
            llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
        ),
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(model)

    processor = AutoProcessor.from_pretrained(model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample = dataset[0]["audio"]

    # result = pipe(sample)
    result = pipe("./audio/es.mp3")
    print(result["text"])




