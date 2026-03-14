"""
eval_utils.py - Shared utilities for quality evaluation.
"""

import json
import os
import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class EvalResult:
    backend: str
    model_name: str
    quantization: str
    task: str           # "mmlu" or "truthfulqa"
    accuracy: float     # 0.0 - 1.0
    num_samples: int
    hardware: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat()


def save_eval_result(result: EvalResult, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"  Eval result saved: {output_path}")


def load_all_results(results_dir: str) -> List[dict]:
    """Load all JSON results (benchmark + eval) from results directory."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and not fname.endswith("_latencies.json"):
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                    data["_source_file"] = fname
                    results.append(data)
            except Exception as e:
                print(f"  Warning: could not load {fname}: {e}")
    return results


def extract_lm_eval_accuracy(lm_eval_output_path: str, task: str) -> Optional[float]:
    """
    Parse accuracy from lm-evaluation-harness output JSON.
    lm-eval stores results under results[task_name][metric].
    """
    try:
        with open(lm_eval_output_path) as f:
            data = json.load(f)
        results = data.get("results", {})
        # mmlu is an aggregate; look for 'mmlu' key or average over subtasks
        if task in results:
            entry = results[task]
            # Try common metric keys
            for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if key in entry:
                    return float(entry[key])
        # Fallback: average over all subtasks containing the task name
        subtask_accs = []
        for k, v in results.items():
            if task in k:
                for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                    if key in v:
                        subtask_accs.append(float(v[key]))
                        break
        if subtask_accs:
            return sum(subtask_accs) / len(subtask_accs)
    except Exception as e:
        print(f"  Could not parse lm-eval output: {e}")
    return None
