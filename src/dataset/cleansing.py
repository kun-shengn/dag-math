import json
import random
import logging

from datasets import load_dataset
from abc import ABC, abstractmethod
from config.log_config import setup_logger

# 配置 logger
logger = setup_logger(__name__)

class DataLoader(ABC):
    """数据加载器：负责读取原始数据集"""
    @abstractmethod
    def load_data(self, file_path: str):
        """加载数据集"""
        pass

class DataLoader_DeepMath_103k(DataLoader):
    """DeepMath-103k 数据加载器"""
    def load_data(self, file_path: str):
        """加载 DeepMath-103k 数据集"""
        # 指定一个空间充足的缓存目录
        cache_dir = "/mnt/iso/zqs_workspace/cache/huggingface" 
        
        dataset = load_dataset(
            "parquet", # 或者 "csv", "parquet"
            data_dir=file_path,
            cache_dir=cache_dir  # 添加这一行
        )
        return dataset['train']  # 返回训练集
    
# ---------------------------------------------------------

class DataFilter(ABC):
    """数据筛选器：负责执行筛选逻辑"""
    @abstractmethod
    def filter_data(self, data: list, min_score: float, max_score: float) -> list:
        """筛选符合条件的数据"""
        pass

class DataFilter_Deepmath_103k(DataFilter):
    """DeepMath-103k 数据筛选器"""
    def filter_data(self, data: list, min_score: float, max_score: float) -> list:
        """筛选符合条件的数据"""
        pass    

    def filter_data_difficulty(self, data: dict, min_score: float, max_score: float) -> list:
        """根据分数范围筛选数据"""
        if min_score <= float(data['difficulty']) <= max_score:
            return data
        return None
    
# ---------------------------------------------------------

class DataSaver(ABC):
    """数据保存器：负责将处理后的数据写入磁盘"""
    @abstractmethod
    def save_data_list(self, data: list, output_path: str):
        """保存列表数据到指定路径"""
        pass

class JSONLDataSaver(DataSaver):
    """JSONL 数据保存器"""
    def save_data(self, data: dict, output_path: str):
        """将单个字典数据保存为 JSONL 格式"""
        with open(output_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def save_data_list(self, data: list, output_path: str):
        """将列表数据保存为 JSONL 格式"""
        with open(output_path, 'a+', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')    

# ---------------------------------------------------------

if __name__ == "__main__":
    # 加载数据
    data_loader = DataLoader_DeepMath_103k()
    dataset = data_loader.load_data("/mnt/iso/zqs_workspace/dataset/deepmath-103k")
    data_filter = DataFilter_Deepmath_103k()   
    data_saver = JSONLDataSaver()
    output_path = "/mnt/iso/zqs_workspace/code/dag-math/dataset/deepmath.jsonl"
    count_filter = 0 # 计数器，记录符合条件的数据数量
    count_sum = 0 # 计数器，记录总数据数量
    for item in dataset:
        count_sum += 1
        if data_filter.filter_data_difficulty(item, min_score=6.0, max_score=8.0):
            # logger.info(f"Saved {item} to {output_path}")
            data_saver.save_data(item, output_path)
            count_filter += 1
        logger.info(f"Total items: {count_sum}, Filtered items: {count_filter}")
    # # 输出 dataset 的列名和特征信息
    # logger.info(f"列名: {dataset.column_names}")
    # logger.info(f"特征详情: {dataset.features}")
     # 保存到 JSONL 文件
