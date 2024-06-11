# gguf_api_test.py
# 基于requests进行服务测试

import requests

url = "http://localhost:8090/completion"
headers = {"Content-Type": "application/json"}
data = {
    "prompt": "system\n你是一位智能助手!\nuser\n你好,请介绍一下埃隆马斯克。请用中文回答。\nassistant\n",
    "n_predict": 128
}
response = requests.post(url, headers=headers, json=data)
result = response.json()["content"]

for line in result.split("\n"):
    print(line)


