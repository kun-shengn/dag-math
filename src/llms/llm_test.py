import logging

from openai import OpenAI
from config.log_config import setup_logger

# 配置 logger
logger = setup_logger(__name__)

# 1. 设置客户端指向本地 vLLM
client = OpenAI(
    api_key="EMPTY",  # vLLM 默认不需要 Key，但客户端库必须填一个
    base_url="http://127.0.0.1:5566/v1",  # 注意端口是 8080，且后面要加 /v1
)

# 2. 获取模型名称 (或者你直接写路径)
models = client.models.list()
model_name = models.data[0].id
print(models)
print(f"正在使用的模型: {model_name}")

# 3. 发送请求 (完全兼容 GPT-4/3.5 的写法)
chat_response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ],
    temperature=0.7,
    max_tokens=100,
)

# logger.info("回答:", chat_response.choices[0].message.content)
logger.info("回答: %s", chat_response)