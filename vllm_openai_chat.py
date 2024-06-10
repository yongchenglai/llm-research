# vllm_openai_chat.py
# Set OpenAI's API key and API base to use vLLM's API server.
from openai import OpenAI

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8090/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "说一下中国"},
    ]
)

print("Chat response:", chat_response)

