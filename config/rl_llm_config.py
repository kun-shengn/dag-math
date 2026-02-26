from trl import GRPOConfig
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Qwen3_0_6B_GRPO_Config_DeepMath():
    #Qwen3-0.6B 的 GRPO 配置，包含训练参数和 vllm 参数
    output_dir=f"model/Qwen3_0.6B_GRPO_{datetime.now().strftime('%Y_%m_%d')}", # 输出目录
    run_name="Qwen3-0.6B-DeepMath103K", # 设置 run 名称，例如 Qwen2.5-0.5B-Instruct-gsm8k, # run 名称
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
    max_prompt_length=256, # 最大 prompt 长度
    max_completion_length=512, # 最大 completion 长度
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

class Qwen3_4B_GRPO_Config_DeepMath():
    #Qwen3-4B 的 GRPO 配置，包含训练参数和 vllm 参数
    output_dir=f"model/Qwen3_4B_GRPO_{datetime.now().strftime('%Y_%m_%d')}", # 输出目录
    run_name="Qwen3-4B-DeepMath103K", # 设置 run 名称，例如 Qwen2.5-0.5B-Instruct-gsm8k, # run 名称
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
    max_prompt_length=256, # 最大 prompt 长度
    max_completion_length=512, # 最大 completion 长度
    num_train_epochs=1, # 训练 epoch 数
    save_steps=100, # 每隔多少步保存模型 checkpoint
    max_grad_norm=0.1, # 最大梯度裁剪范数
    report_to="wandb", # 使用 wandb 记录训练日志 (需要安装 wandb 并登录)
    log_on_each_node=False, # 只在主节点记录日志
    use_vllm=True, # 使用 vllm 进行推理加速
    vllm_init_kwargs={ # vllm 初始化参数
        "tensor_parallel_size": 4, # 使用 3 张 GPU 进行张量并行
        "gpu_memory_utilization": 0.5, # 为每张 GPU 分配 80% 的显存
        "max_model_len": 2048, # 适当增加最大模型长度以处理更长的序列
        "dtype": "bfloat16", # L40 支持 bfloat16，通常比 float16 更稳定
    },
    enable_profiling=False # 是否启用 profiling

# ---------------------------------------------------------

def grpo_config(grpo_config) -> GRPOConfig:
    return GRPOConfig(
        output_dir=grpo_config.output_dir,  # 输出目录
        run_name=grpo_config.run_name,  # 设置 run 名称
        learning_rate=grpo_config.learning_rate,  # 学习率
        beta=grpo_config.beta,  # divergence coefficient
        optim=grpo_config.optim,  # 优化器
        adam_beta1=grpo_config.adam_beta1,  # AdamW beta1 参数
        adam_beta2=grpo_config.adam_beta2,  # AdamW beta2 参数
        weight_decay=grpo_config.weight_decay,  # 权重衰减
        warmup_ratio=grpo_config.warmup_ratio,  # 学习率 warmup 比例
        lr_scheduler_type=grpo_config.lr_scheduler_type,  # 学习率调度器类型
        logging_steps=grpo_config.logging_steps,  # 日志记录步数
        bf16=grpo_config.bf16,  # 是否使用 bfloat16 精度
        per_device_train_batch_size=grpo_config.per_device_train_batch_size,  # 每设备 batch size
        num_generations=grpo_config.num_generations,  # group size
        gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,  # 梯度累积步数
        max_prompt_length=grpo_config.max_prompt_length,  # 最大 prompt 长度
        max_completion_length=grpo_config.max_completion_length,  # 最大 completion 长度
        num_train_epochs=grpo_config.num_train_epochs,  # 训练 epoch 数
        save_steps=grpo_config.save_steps,  # 保存 checkpoint 的步数
        max_grad_norm=grpo_config.max_grad_norm,  # 最大梯度裁剪范数
        report_to=grpo_config.report_to,  # 日志记录工具
        log_on_each_node=grpo_config.log_on_each_node,  # 是否在每个节点记录日志
        use_vllm=grpo_config.use_vllm,  # 是否使用 vllm
        vllm_init_kwargs=grpo_config.vllm_init_kwargs,  # vllm 初始化参数
        gradient_checkpointing=grpo_config.gradient_checkpointing,  # 是否启用梯度检查点
        gradient_checkpointing_kwargs=grpo_config.gradient_checkpointing_kwargs,  # 梯度检查点参数
        logit_computation_mini_batch_size=grpo_config.logit_computation_mini_batch_size,  # logit 计算 mini batch size
        enable_profiling=grpo_config.enable_profiling  # 是否启用 profiling
    )