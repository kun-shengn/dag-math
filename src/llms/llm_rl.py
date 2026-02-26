import os # 导入 os 库，用于与操作系统交互，例如设置环境变量
import re # 导入 re 库，用于正则表达式操作，例如提取答案
import torch # 导入 PyTorch 库
import logging

from abc import ABC, abstractmethod
from config.log_config import setup_logger
from src.dataset.process import DeepMath_processer
from config.rl_llm_config import Qwen3_4B_GRPO_Config_DeepMath, grpo_config
from datasets import load_dataset, Dataset # 导入 datasets 库，用于加载和处理数据集
from transformers import AutoTokenizer, AutoModelForCausalLM # 导入 transformers 库，用于加载预训练模型和 tokenizer
from trl.trainer import GRPOConfig, GRPOTrainer # 导入 trl 库中的 GRPOConfig 和 GRPOTrainer，用于 GRPO 训练

# 配置 logger
logger = setup_logger(__name__)

# ---------------------------------------------------------
# 定义大模型GRPO强化学习训练接口
# ---------------------------------------------------------
class LLM_GRPO(ABC):
    """大模型强化学习训练接口"""
    @abstractmethod
    def train(self, model_dir: str):
        pass

class Qwen3_4B_LLM_GRPO(LLM_GRPO):
    """基于 Qwen3-4B 模型的 GRPO 强化学习训练实现"""
    def __init__(self):
        pass

        # 从 XML 格式文本中提取答案，例如从 <answer>42</answer> 中提取 "42"
    def extract_xml_answer(self, text: str) -> str:
        try:
            answer = text.split("<answer>")[-1].split("</answer>")[0].strip() # 使用 <answer> 和 </answer> 分割文本，提取答案并去除首尾空格
            return answer
        except IndexError:
            return "" # 如果提取失败，返回空字符串
        
        # 定义奖励函数 (reward functions)

    # 格式奖励函数：检查模型输出是否符合 XML 格式要求 (包含 <reasoning> 和 <answer> 标签)
    def format_reward_func(self, completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has the correct format."""
        pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$" # 定义正则表达式，匹配以 <reasoning> 开头，</reasoning> 结尾，中间任意字符，然后是 <answer> 开头，</answer> 结尾，中间任意字符的格式
        responses = [completion[0]["content"] for completion in completions] # 从 completions 中提取模型生成的文本内容
        matches = [bool(re.match(pattern, r)) for r in responses] # 使用正则表达式匹配生成的文本是否符合格式
        return [1.0 if match else 0.0 for match in matches] # 如果匹配，奖励 1.0，否则奖励 0.0

    # 正确性奖励函数：检查模型生成的答案是否与标准答案一致
    def correctness_reward_func(self, prompts, completions, answer, **kwargs) -> list[float]:
        """Reward function that checks if the answer is correct."""
        responses = [completion[0]['content'] for completion in completions] # 提取模型生成的文本内容
        extracted_responses = [self.extract_xml_answer(r) for r in responses] # 从生成的文本中提取 XML 格式的答案
        print(f"Question: {prompts[0][-1]['content']}\nAnswer: {answer[0]}\nResponse: {responses[0]}\nExtracted: {extracted_responses[0]}") # 打印问题、标准答案、模型完整输出和提取出的答案，用于调试
        print(''.join('✅' if r == a else '❌' for r, a in zip(extracted_responses, answer))) # 打印 ✅ 或 ❌，表示答案是否正确
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)] # 如果答案正确，奖励 2.0，否则奖励 0.0

    def train(self, model_dir: str, dataset_dir: str = r"D:\code\mess\datasets\deepmath-103k"):
        # 训练代码示例
        logger.info("Starting GRPO training for Qwen3-4B model...")
        processer = DeepMath_processer()
        dataset = processer.process_dataset(dataset_dir = dataset_dir)
        # 设置显存相关的环境变量，限制 PyTorch 显存分配策略，防止 OOM
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        training_args = grpo_config(Qwen3_4B_GRPO_Config_DeepMath()) # 初始化 GRPO 训练参数配置

        # 加载预训练模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, # 模型名称
            torch_dtype=torch.bfloat16, # 指定数据类型为 bfloat16
            # attn_implementation="flash_attention_2", # T4 不支持 flash_attention_2，如果使用 A100 等 Ampere 架构 GPU 可以启用
            device_map="auto", # 自动选择设备 (GPU 或 CPU)
        )

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, # tokenizer 名称
            model_max_length=training_args.max_completion_length, # 设置 tokenizer 最大长度
        )
        tokenizer.pad_token = tokenizer.eos_token # 将 pad token 设置为 eos token

        # 初始化 GRPO Trainer
        trainer = GRPOTrainer(
            model=model, # 传入模型
            processing_class=tokenizer, # 传入 tokenizer
            reward_funcs=[ # 传入奖励函数列表
                self.format_reward_func, # 格式奖励函数
                self.correctness_reward_func # 正确性奖励函数
            ],
            args=training_args, # 传入训练参数
            train_dataset=dataset, # 传入训练数据集
        )

        # 开始训练
        trainer.train()

# ---------------------------------------------------------
if __name__ == "__main__":
    grpo_trainer = Qwen3_4B_LLM_GRPO()
    grpo_trainer.train(model_dir = "/mnt/iso/zqs_workspace/model/Qwen3-4B", dataset_dir = "/mnt/iso/zqs_workspace/dataset/deepmath-103k")