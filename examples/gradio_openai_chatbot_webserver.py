# gradio_openai_chatbot_webserver.py
"""
chmod 755 gradio_openai_chatbot_webserver.py
python gradio_openai_chatbot_webserver.py \
--model-url="http://192.168.31.160:8090/v1" \
--model="qwen2-7b" \
--api-key="token-abc123" \
--server_host="0.0.0.0" \
--server_port=7860 \
--share
"""
import gradio as gr
import argparse
from openai import OpenAI


def predict(message, history):
    # Convert chat history to OpenAI format
    history_openai_format = [{
        "role": "system",
        "content": "You are a great ai assistant."
    }]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})

    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            'repetition_penalty':
            1,
            'stop_token_ids': [
                int(id.strip()) for id in args.stop_token_ids.split(',')
                if id.strip()
            ] if args.stop_token_ids else []
        })

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description='Chatbot Interface with Customizable Parameters')
    parser.add_argument('--model-url', type=str, default='http://localhost:8000/v1',
                        help='Model URL')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name for the chatbot')
    parser.add_argument('--api-key', type=str, default="EMPTY", help='openai_api_key')
    parser.add_argument('--temp', type=float, default=0.8,
                        help='Temperature for text generation')
    parser.add_argument('--stop-token-ids', type=str, default='',
                        help='Comma-separated stop token IDs')
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--server_host", type=str, default='0.0.0.0')
    parser.add_argument("--server_port", type=int, default=7860)

    # Parse the arguments
    args = parser.parse_args()

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = args.api_key
    openai_api_base = args.model_url

    # Create an OpenAI client to interact with the API server
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Create and launch a chat interface with Gradio
    gr.ChatInterface(predict).queue().launch(
        server_name=args.server_host,
        server_port=args.server_port,
        debug=True,
        share=args.share)


