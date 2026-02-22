import os
import json
import time
import logging
import numpy as np

from datetime import datetime
from abc import ABC, abstractmethod
from config.log_config import setup_logger
from config.llm_config import LLMConfig_Qwen3_8B
from src.utils.utils import load_jsonl
from src.eval.schemas import llmEvalLog
from src.llms.llm import LLM_response_Qwen3_8B

# 配置 logger
logger = setup_logger(__name__)

# ---------------------------------------------------------
# 定义AIME2025数据集的抽象接口
# ---------------------------------------------------------

class TestAIME2025(ABC):
    """
    对于AIME2025数据集的测试类，主要验证数据加载和模型推理的正确性。
    """
    @abstractmethod
    def load_data(self,file: str)-> list:
        """
        加载AIME2025数据集。
        """
        pass

    @abstractmethod
    def save_results(self, results: dict, file: str):
        """
        将测试结果保存到文件中。
        """
        pass

    @abstractmethod
    def test_model_responses_pass1(self, dataset: list)-> dict:
        """
        将模型的回答加入字典中
        """
        pass

    @abstractmethod
    def test_model_ACC(self):
        """
        测试模型准确率。
        """
        pass

class Qwen_8B_TestAIME2025(TestAIME2025):
    """
    Qwen3-8B模型在AIME2025数据集上的测试实现。
    """
    def __init__(self):
        super().__init__()
        logger.info("Initialized Qwen_8B_TestAIME2025")

    def load_data(self,file: str):
        test_set = load_jsonl(file)
        return test_set
    
    def save_results(self, results: dict):
        # 获取当前日期，格式为 "YYYYMMDD"
        current_date = datetime.now().strftime("%Y%m%d")
        
        # 动态生成保存路径
        dir_path = f"tests/outcome/{current_date}"
        file_path = os.path.join(dir_path, "Qwen_8B_TestAIME2025.jsonl")
        
        # 确保路径存在
        if not os.path.exists(dir_path):  # 如果目录不存在
            os.makedirs(dir_path)        # 创建目录

        # 保存结果到文件
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False) 
            f.write('\n')

        # 打印保存路径，便于调试
        logger.info(f"Results saved to: {file_path}")    

    def test_model_responses_pass1(self, dataset: list):
        llm_response = LLM_response_Qwen3_8B()
        test_result = []
        idx = 0
        for data in dataset:
            pred = []
            question = data['question']
            response,extracted = llm_response.generate_response_nothinking(question)
            pred.append(extracted)
            test_result_dict: llmEvalLog = {
                "idx": idx,
                "question": question,
                "answer": data['answer'],
                "response": response,
                "pred": pred
            }
            logger.info(f"Generated result for idx {idx}: {test_result_dict}")
            self.save_results(test_result_dict)  # 每生成一个结果就保存一次，避免数据丢失
            idx += 1
            # test_result.append(test_result_dict)
        return test_result

    def test_model_ACC(self):
        pass


if __name__ == "__main__":
    test = Qwen_8B_TestAIME2025()
    dataset = test.load_data("data/aime25/test.jsonl")
    results = test.test_model_responses_pass1(dataset)