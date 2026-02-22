from dataclasses import dataclass
from abc import ABC

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

