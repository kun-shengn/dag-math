import os # 导入 os 库，用于与操作系统交互，例如设置环境变量
import re # 导入 re 库，用于正则表达式操作，例如提取答案
import torch # 导入 PyTorch 库
from datasets import load_dataset, Dataset # 导入 datasets 库，用于加载和处理数据集
from transformers import AutoTokenizer, AutoModelForCausalLM # 导入 transformers 库，用于加载预训练模型和 tokenizer
from trl.trainer import GRPOConfig, GRPOTrainer # 导入 trl 库中的 GRPOConfig 和 GRPOTrainer，用于 GRPO 训练

# 定义 R1 风格的系统提示词，用于引导模型进行 reasoning 和 answer 的格式输出
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

# 定义任务特定的指令，这里要求答案必须是单个整数
TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."

# 数据预处理函数
def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset(dataset_name, 'main')[split] # 加载指定数据集和 split (例如 'train')

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
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS}, # 系统提示词
            {'role': 'user', 'content': "What is 2+2?"}, # 示例用户问题
            {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"}, # 示例助手回答，包含 reasoning 和 answer 标签
            {'role': 'user', 'content': q.strip()} # 当前批次的用户问题
        ] for q in batch['question']] # 遍历批次中的问题

        return {
            'prompt': prompts, # 返回构建好的 prompt 列表
            'answer': [extract_hash_answer(a) for a in batch['answer']] # 返回提取出的答案列表
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size) # 使用 map 函数对数据集进行批处理

dataset_name = 'openai/gsm8k' # 设置数据集名称为 gsm8k (数学应用题数据集)
dataset = preprocess_dataset(dataset_name, chunk_size=500) # 预处理数据集，批次大小为 500

# 从 XML 格式文本中提取答案，例如从 <answer>42</answer> 中提取 "42"
def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip() # 使用 <answer> 和 </answer> 分割文本，提取答案并去除首尾空格
        return answer
    except IndexError:
        return "" # 如果提取失败，返回空字符串

# 定义奖励函数 (reward functions)

# 格式奖励函数：检查模型输出是否符合 XML 格式要求 (包含 <reasoning> 和 <answer> 标签)
def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$" # 定义正则表达式，匹配以 <reasoning> 开头，</reasoning> 结尾，中间任意字符，然后是 <answer> 开头，</answer> 结尾，中间任意字符的格式
    responses = [completion[0]["content"] for completion in completions] # 从 completions 中提取模型生成的文本内容
    matches = [bool(re.match(pattern, r)) for r in responses] # 使用正则表达式匹配生成的文本是否符合格式
    return [1.0 if match else 0.0 for match in matches] # 如果匹配，奖励 1.0，否则奖励 0.0

# 正确性奖励函数：检查模型生成的答案是否与标准答案一致
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the answer is correct."""
    responses = [completion[0]['content'] for completion in completions] # 提取模型生成的文本内容
    extracted_responses = [extract_xml_answer(r) for r in responses] # 从生成的文本中提取 XML 格式的答案
    print(f"Question: {prompts[0][-1]['content']}\nAnswer: {answer[0]}\nResponse: {responses[0]}\nExtracted: {extracted_responses[0]}") # 打印问题、标准答案、模型完整输出和提取出的答案，用于调试
    print(''.join('✅' if r == a else '❌' for r, a in zip(extracted_responses, answer))) # 打印 ✅ 或 ❌，表示答案是否正确
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)] # 如果答案正确，奖励 2.0，否则奖励 0.0

# model_name = "Qwen/Qwen2.5-0.5B" # 可以选择不带 Instruct 的版本，这里使用 Instruct 版本
model_name = "Qwen/Qwen2.5-0.5B-Instruct" # 设置模型名称为 Qwen2.5-0.5B-Instruct

output_dir = f"outputs/{model_name.split('/')[-1]}-GRPO" # 设置输出目录，例如 outputs/Qwen2.5-0.5B-Instruct-GRPO
run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}" # 设置 run 名称，例如 Qwen2.5-0.5B-Instruct-gsm8k

# 设置显存相关的环境变量，限制 PyTorch 显存分配策略，防止 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

max_prompt_length=256 # 设置最大 prompt 长度
max_completion_length=512 # 设置最大 completion 长度

# 配置 GRPO 训练参数
training_args = GRPOConfig(
    output_dir=output_dir, # 输出目录
    run_name=run_name, # run 名称
    learning_rate=1e-5, # 学习率
    beta=0.005, # divergence coefficient，控制策略偏离参考模型的程度，值越大更新越保守，默认 0.04
    optim="adamw_8bit", # 使用 8-bit AdamW 优化器
    adam_beta1=0.9, # AdamW beta1 参数
    adam_beta2=0.99, # AdamW beta2 参数
    weight_decay=0.1, # 权重衰减
    warmup_ratio=0.1, # 学习率 warmup 比例
    lr_scheduler_type='cosine', # 学习率调度器类型为 cosine
    logging_steps=1, # 每隔多少步记录日志
    bf16=True, # 使用 bfloat16 精度训练
    per_device_train_batch_size=4, # 每个设备上的训练 batch size
    num_generations=4,  # group size，GRPO 算法中的 group size
    gradient_accumulation_steps=4, # 梯度累积步数，用于增大有效 batch size
    max_prompt_length=max_prompt_length, # 最大 prompt 长度
    max_completion_length=max_completion_length, # 最大 completion 长度
    num_train_epochs=1, # 训练 epoch 数
    save_steps=100, # 每隔多少步保存模型 checkpoint
    max_grad_norm=0.1, # 最大梯度裁剪范数
    report_to="wandb", # 使用 wandb 记录训练日志 (需要安装 wandb 并登录)
    log_on_each_node=False, # 只在主节点记录日志
    use_vllm=True, # 使用 vllm 进行推理加速
    vllm_init_kwargs={ # vllm 初始化参数
        "device": "cuda:0", # 指定设备为 cuda:0
        "gpu_memory_utilization": 0.3, # 设置 vllm 显存利用率为 30%
        "max_model_len": max_prompt_length + max_completion_length, # 设置 vllm 最大模型长度
        "dtype": "half", # 设置 vllm 数据类型为 half (float16/bfloat16，根据 GPU 支持自动选择)
        # "enable_chunked_prefill": True, # 启用 chunked prefill (vllm 的优化技术)
        # "max_num_batched_tokens": 2048, # 设置 vllm 最大 batch tokens 数
    },
    gradient_checkpointing=True, # 启用梯度检查点，减少显存占用
    gradient_checkpointing_kwargs={"use_reentrant": False}, # 梯度检查点参数，use_reentrant=False 可以节省显存
    logit_computation_mini_batch_size=1, # logit 计算的 mini batch size，用于进一步减少显存占用
    enable_profiling=False # 是否启用 profiling
)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_name, # 模型名称
    torch_dtype=torch.bfloat16, # 指定数据类型为 bfloat16
    # attn_implementation="flash_attention_2", # T4 不支持 flash_attention_2，如果使用 A100 等 Ampere 架构 GPU 可以启用
    device_map="auto", # 自动选择设备 (GPU 或 CPU)
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, # tokenizer 名称
    model_max_length=training_args.max_completion_length, # 设置 tokenizer 最大长度
)
tokenizer.pad_token = tokenizer.eos_token # 将 pad token 设置为 eos token

# 初始化 GRPO Trainer
trainer = GRPOTrainer(
    model=model, # 传入模型
    processing_class=tokenizer, # 传入 tokenizer
    reward_funcs=[ # 传入奖励函数列表
        format_reward_func, # 格式奖励函数
        correctness_reward_func # 正确性奖励函数
    ],
    args=training_args, # 传入训练参数
    train_dataset=dataset, # 传入训练数据集
)

# 开始训练
trainer.train()