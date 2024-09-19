# minicpm3_rag_lora_cli_demo.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if __name__ == "__main__":

    base_model_dir = "/models/openbmb/MiniCPM3-4B"
    lora_model_dir = "/models/openbmb/MiniCPM3-RAG-LoRA"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(lora_model_dir)
    model = PeftModel.from_pretrained(model, lora_model_dir)

    passages_list = [
        "In the novel 'The Silent Watcher,' the lead character is named Alex Carter. Alex is a private detective who uncovers a series of mysterious events in a small town.",
        "Set in a quiet town, 'The Silent Watcher' follows Alex Carter, a former police officer turned private investigator, as he unravels the town's dark secrets.",
        "'The Silent Watcher' revolves around Alex Carter's journey as he confronts his past while solving complex cases in his hometown."]
    instruction = "Q: What is the name of the lead character in the novel 'The Silent Watcher'?\nA:"

    passages = '\n'.join(passages_list)
    input_text = 'Background:\n' + passages + '\n\n' + instruction

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False)

    outputs = model.chat(tokenizer, prompt, temperature=0.8, top_p=0.8)
    # The lead character in the novel 'The Silent Watcher' is named Alex Carter.
    print(outputs[0])

