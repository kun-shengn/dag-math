from typing import TypedDict, List, Optional, Any
# 如果是 Python 3.11+，直接用: from typing import NotRequired
from typing_extensions import NotRequired 

class llmEvalLog(TypedDict):
    """
    LLM 评测日志结构定义
    """
    
    # ==========================
    # ✅ 1. 必须填写的字段 (Required)
    # ==========================
    idx: int
    question: str
    answer: str
    pred: List[str]  # 根据你的JSON，pred是一个列表

    # ==========================
    # ❓ 2. 可选字段 (Not Required)
    # 字典里可以完全没有这些 Key，也不会报错
    # ==========================
    
    # 基础信息
    gt: NotRequired[str]
    gt_cot: NotRequired[str]
    level: NotRequired[str]
    solution: NotRequired[str]
    
    # LLM 交互详情
    system: NotRequired[str]
    query: NotRequired[str]
    
    # 复杂的列表结构
    response: NotRequired[List[str]]
    code: NotRequired[List[str]]
    
    # 评分结果 (允许由空值组成的列表)
    # List[Optional[str]] 意味着列表里可以是字符串，也可以是 None
    report: NotRequired[List[Optional[str]]] 
    
    score: NotRequired[List[bool]]
    
    # 嵌套列表 (二维数组)
    pred_score: NotRequired[List[List[float]]]
    
    # 历史记录 (可能为空)
    history: NotRequired[List[Any]]