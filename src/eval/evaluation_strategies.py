from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Dict
from grader import math_equal_process
from src.eval.parser import parse_ground_truth

class EvaluationStrategy(ABC):
    """
    定义评估策略的抽象基类 (接口)。
    """
    @abstractmethod
    def parse_ground_truth(self, sample: Dict) -> Tuple[Any, Any]:
        """解析出标准答案。"""
        pass

    @abstractmethod
    def get_comparison_function(self):
        """返回用于比较预测和真值的函数。"""
        pass

class MathEvaluationStrategy(EvaluationStrategy):
    """
    针对数学问题的具体评估策略。
    """
    def __init__(self, data_name: str):
        self.data_name = data_name

    def parse_ground_truth(self, sample: Dict) -> Tuple[str, Any]:
        """使用现有的解析器为数学问题解析答案。"""
        return parse_ground_truth(sample, self.data_name)

    def get_comparison_function(self):
        """返回数学问题的评分函数。"""
        return math_equal_process