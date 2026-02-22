import numpy as np
import itertools
from typing import Union, List, Dict

def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def calculate_metrics(samples: List[Dict], scores: List[bool], timeout_cnt: int) -> Dict:
    """
    根据评分结果计算所有指标。
    """
    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max((len(s) for s in score_mat), default=0)
    if max_len == 0: # Handle case with no predictions
        return {
            "num_samples": len(samples), "num_scores": len(scores), "timeout_samples": timeout_cnt,
            "empty_samples": len([s for s in samples if not s.get('pred') or not s['pred'][-1]]),
            "acc": 0, "pass_acc": 0, "pass@k": {},
        }


    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + ([s[-1]] if s else [False]) * (max_len - len(s))

    score_mat_np = np.array(score_mat)
    num_correct = np.sum(score_mat_np, axis=1)

    k_values = [1]
    power = 1
    while 2**power <= max_len:
        k_values.append(2**power)
        power += 1

    pass_at_k = {
        k: float(np.round(np.mean(estimate_pass_at_k(max_len, num_correct, k)) * 100, decimals=1))
        for k in k_values
    }

    row_eval = [any(row) for row in score_mat]
    pass_acc = np.mean(row_eval) if row_eval else 0
    mean_score = float(np.round(score_mat_np.mean() * 100, decimals=1)) if score_mat_np.size > 0 else 0

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s.get('pred') or not s['pred'][-1]]),
        "acc": mean_score,
        "pass_acc": np.round(pass_acc * 100, decimals=1),
        "pass@k": pass_at_k,
    }

    if samples and "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            type_key = sample.get('type')
            if type_key not in type_scores:
                type_scores[type_key] = []
            # Use the last score for type accuracy as in the original script
            if sample['score']:
                type_scores[type_key].append(sample['score'][-1])
        
        type_acc = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items() if v}
        result_json['type_acc'] = {k: v for k, v in sorted(type_acc.items())}

    return result_json