import logging
import os
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """配置并返回一个 logger"""
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # 定义日志文件路径
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    info_log_file = os.path.join(log_dir, f"info-{current_date}.log")
    error_log_file = os.path.join(log_dir, f"error-{current_date}.log")
    
    # 创建 logger
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # 防止重复添加 handler
        logger.setLevel(logging.DEBUG)  # 设置最低日志级别
        
        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # INFO 级别日志处理器
        info_handler = logging.FileHandler(info_log_file, mode='a', encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)
        
        # ERROR 级别日志处理器
        error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # 控制台日志处理器（可选）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger