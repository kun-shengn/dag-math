import logging

from openai import OpenAI
from src.utils.utils import extract_boxed_content
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from config.log_config import setup_logger
from config.llm_config import LLMConfig_Qwen3_8B
from prompts.qwen3_8B_prompt import qwen3_8B_prompt_system

# 配置 logger
logger = setup_logger(__name__)

# ---------------------------------------------------------
# 定义大模型服务的抽象接口
# ---------------------------------------------------------

class LLM_response(ABC):
    """
    抽象的 LLM 回复接口。
    """
    @abstractmethod
    def generate_response(self, content: str, temperature: float = 0) -> str:
        #生成带思考的回复
        pass

    @abstractmethod
    def generate_response_nothinking(self, content: str, temperature: float = 0) -> str:
        #生成无思考的回复（思考过程被过滤掉了）
        pass

# ---------------------------------------------------------
# Qwen3-8B的大模型服务实现
# ---------------------------------------------------------

class LLM_response_Qwen3_8B(LLM_response):
    """
    基于 OpenAI SDK 的 Qwen3-8B 回复实现。
    """
    def __init__(self):
        self.llmconfig = LLMConfig_Qwen3_8B()
        self.client = OpenAI(base_url = self.llmconfig.base_url, api_key = self.llmconfig.api_key)
        self.model_name = self.llmconfig.model_name

    def generate_response(self, content: str = 'Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.', temperature: float = 0) -> str:
        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": qwen3_8B_prompt_system},
                    {"role": "user", "content": content}
                ],
                temperature = temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

    def generate_response_nothinking(self, content: str = 'Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.', temperature: float = 0) -> str:
        try:
            response = self.generate_response(content, temperature)
            extracted = extract_boxed_content(response)
            if extracted is not None:
                return response, extracted
            else:
                logger.warning("No boxed content found in the response.")
                return response, extracted  # 如果没有找到，返回原始响应以供调试

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

# ---------------------------------------------------------
if __name__ == "__main__":
    # QWEN3-8B 模型测试
    Qwen3_8b_response = LLM_response_Qwen3_8B()
    logger.info(f"Qwen3-8B 模型: {Qwen3_8b_response.generate_response_nothinking()}")