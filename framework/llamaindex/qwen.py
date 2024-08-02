# qwen.py
"""
Langchain
This guide helps you build a question-answering application based on a
local knowledge base using Qwen2-7B-Instruct with langchain.
The goal is to establish a knowledge base Q&A solution.

Basic Usage:
The implementation process of this project includes
loading files -> reading text -> segmenting text -> vectorizing text
-> vectorizing questions -> matching the top k most similar text vectors
with the question vectors -> incorporating the matched text as context
along with the question into the prompt -> submitting to the
Qwen2-7B-Instruct to generate an answer. Below is an example:

pip install langchain==0.0.174
pip install faiss-gpu
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

class Qwen(LLM, ABC):
     max_token: int = 10000
     temperature: float = 0.01
     top_p = 0.9
     history_len: int = 3

     def __init__(self):
         super().__init__()

     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
         messages = [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt}
         ]
         text = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         model_inputs = tokenizer([text], return_tensors="pt").to(device)
         generated_ids = model.generate(
             model_inputs.input_ids,
             max_new_tokens=512
         )
         generated_ids = [
             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
         ]

         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         return response

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}


