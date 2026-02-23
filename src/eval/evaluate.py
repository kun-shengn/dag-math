import logging
import argparse
import numpy as np
from typing import List
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from src.utils.utils import load_jsonl
from config.log_config import setup_logger
from metrics_calculator import calculate_metrics
from evaluation_strategies import EvaluationStrategy, MathEvaluationStrategy

# 配置 logger
logger = setup_logger(__name__)

def evaluate(samples: list, strategy: EvaluationStrategy, max_num_samples=None):
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx'])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        logger.info(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    # 1. 使用策略来准备数据
    for sample in samples:
        sample['gt_cot'], sample['gt'] = strategy.parse_ground_truth(sample)
    
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample.get('pred', [])]
    comparison_func = strategy.get_comparison_function()

    scores = []
    timeout_cnt = 0

    # 2. 执行评估
    if params: # 只有在有预测需要评估时才启动进程池
        with ProcessPool(max_workers=1) as pool:
            future = pool.map(comparison_func, params, timeout=3)
            iterator = future.result()
            with tqdm(total=len(params), desc="Evaluate") as progress_bar:
                while True:
                    try:
                        result = next(iterator)
                        scores.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logger.warning(f"TimeoutError: {error}")
                        scores.append(False)
                        timeout_cnt += 1
                    except Exception as error:
                        logger.error(f"An unexpected error occurred: {error}")
                        # 根据需要决定是否退出或记录为失败
                        scores.append(False)
                    progress_bar.update(1)
    
    # 3. 使用独立的计算器来计算指标
    result_json = calculate_metrics(samples, scores, timeout_cnt)

    logger.info(result_json)
    return samples, result_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="aime25", help="Name of the dataset, e.g., 'math'.")
    # parser.add_argument("--file_path", type=str, default="data/eval_rm_maj_example/math_cot_100.jsonl", help="Path to the JSONL file with predictions.")
    parser.add_argument("--file_path", type=str, default="tests/outcome/2026-02-22/Qwen2-5_math_7B_TestAIME2025.jsonl", help="Path to the JSONL file with predictions.")
    parser.add_argument("--max_num_samples", type=int, default=None, help="Maximum number of samples to evaluate.")
    args = parser.parse_args()

    # 依赖注入：创建并传入具体的策略
    # 如果未来有其他评估类型，只需在这里实例化不同的策略类
    strategy = MathEvaluationStrategy(data_name=args.data_name)
    
    samples = list(load_jsonl(args.file_path))
    logger.info(f"Loaded {samples} samples from {args.file_path}")
    if not samples:
        logger.warning("No samples found in the file.")
        return

    evaluate(samples=samples, strategy=strategy, max_num_samples=args.max_num_samples)


if __name__ == "__main__":
    main()