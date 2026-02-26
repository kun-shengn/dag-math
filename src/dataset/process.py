import logging

from abc import ABC, abstractmethod
from config.log_config import setup_logger
from datasets import load_dataset, Dataset # 导入 datasets 库，用于加载和处理数据集
from prompts.qwen3_4B_prompt import qwen3_4B_prompt_system_RL
from src.dataset.cleansing import DataFilter_Deepmath_103k

# 配置 logger
logger = setup_logger(__name__)


class dataset_processor(ABC):
    """数据处理器，处理成数据集的形式"""
    @abstractmethod
    def process_dataset(self, dataset_dir) -> Dataset:
        pass

class DeepMath_processer(dataset_processor):
    def process_dataset(self, dataset_dir = r"D:\code\mess\datasets\deepmath-103k", prompt_system = qwen3_4B_prompt_system_RL, prompt_instructions = "", chunk_size=1000) -> Dataset:
        dataset = load_dataset(
            "parquet", # 或者 "csv", "parquet"
            data_dir=dataset_dir
        )

        # 使用 DataFilter_Deepmath_103k 进行筛选
        data_filter = DataFilter_Deepmath_103k()
        filtered_dataset = dataset.filter(
            lambda example: data_filter.filter_data_difficulty(example, min_score=5.5, max_score=8.0) is not None
        )
        logger.info(f"原始数据集大小: {len(dataset['train'])}. 筛选后数据集大小: {len(filtered_dataset['train'])}")

        # 从文本中提取 "####" 后面的答案
        def extract_hash_answer(text: str) -> str | None:
            try:
                return text.split("####")[1].strip() # 使用 "####" 分割文本，取第二部分并去除首尾空格
            except IndexError:
                return None # 如果分割失败，返回 None
            
        # 处理批次数据
        def process_batch(batch):
            # 构建 prompt 列表，每个 prompt 包含 system prompt, example conversation, 和用户问题
            prompts = [[
                {'role': 'system', 'content': prompt_system + "\n" + prompt_instructions}, # 系统提示词
                {'role': 'user', 'content': "What is 2+2?"}, # 示例用户问题
                {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"}, # 示例助手回答，包含 reasoning 和 answer 标签
                {'role': 'user', 'content': q.strip()} # 当前批次的用户问题
            ] for q in batch['question']] # 遍历批次中的问题

            return {
                'prompt': prompts, # 返回构建好的 prompt 列表
                'answer': [a for a in batch['final_answer']] # 返回提取出的答案列表
            }   

        return filtered_dataset.map(process_batch, batched=True, batch_size=chunk_size) # 使用 map 函数对数据集进行批处理    

if __name__ == "__main__":
    processor = DeepMath_processer()
    dataset = processor.process_dataset(dataset_dir = r"D:\code\mess\datasets\deepmath-103k")
    for i in range(5):
        logger.info(dataset['train'][i])
    logger.info(dataset)