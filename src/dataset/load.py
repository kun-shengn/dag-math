import logging

from datasets import load_dataset
from config.log_config import setup_logger

# 配置 logger
logger = setup_logger(__name__)

# 加载数据集
dataset = load_dataset(
    "parquet", # 或者 "csv", "parquet"
    data_dir=r"D:\code\mess\datasets\deepmath-103k"
)
# 查看数据集的结构
logger.info(dataset)
# 查看数据集的结构
logger.info(type(dataset))

# 访问训练集
train_dataset = dataset['train']

# 访问测试集
# test_dataset = dataset['test']

# 访问第一个样本
first_sample = train_dataset[0]
logger.info(first_sample)
logger.info(f"{first_sample['question']}\n{first_sample['final_answer']}")
