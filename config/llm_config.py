from abc import ABC
from dataclasses import dataclass

#用于存放open格式大模型相关参数
@dataclass(kw_only=True)
class AbstractLLMConfig(ABC):
    base_url: str  
    api_key: str
    model_name: str

#qwen3-8b的配置类
@dataclass(kw_only=True)
class LLMConfig_Qwen3_8B(AbstractLLMConfig):
    base_url: str = "http://127.0.0.1:5566/v1"
    api_key: str = "EMPTY"  
    model_name: str = "/hdd0/zhongqishu/model/Qwen3-8B/" 

#qwen2.5-math-7B的配置类
@dataclass(kw_only=True)
class LLMConfig_Qwen2_5_math_7B(AbstractLLMConfig):
    base_url: str = "http://127.0.0.1:5566/v1"
    api_key: str = "EMPTY"  
    model_name: str = "/hdd0/zhongqishu/model/Qwen2.5-math-7B/" 

#qwen3.5-plus的配置类
@dataclass(kw_only=True)
class LLMConfig_Qwen3_5_Plus(AbstractLLMConfig):
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-a212b33c87f74711a8a978305a56ebe8 "
    model_name: str = "qwen-plus" 

