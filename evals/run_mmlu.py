"""
run_mmlu.py - MMLU evaluation via lm-evaluation-harness.

MMLU (Massive Multitask Language Understanding) measures factual knowledge
across 57 subjects. We use it as the primary quality signal for quantization
degradation.

Usage:
    # HF backend (supports all quantizations via bitsandbytes)
    python evals/run_mmlu.py --config configs/benchmark_config.yaml \\
        --backend hf --quantization fp16

    python evals/run_mmlu.py --config configs/benchmark_config.yaml \\
        --backend hf --quantization int8

    # vLLM backend
    python evals/run_mmlu.py --config configs/benchmark_config.yaml \\
        --backend vllm --quantization fp16

Prerequisites:
    pip install lm-eval
"""

import argparse
import subprocess
import sys
import os
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from evals.eval_utils import EvalResult, save_eval_result, extract_lm_eval_accuracy


def build_lm_eval_command(
    backend: str,
    model_name: str,
    quantization: str,
    num_samples: int,
    output_path: str,
) -> list:
    """Build the lm_eval CLI command."""

    if backend == "hf":
        model_args = f"pretrained={model_name}"
        if quantization == "int8":
            model_args += ",load_in_8bit=True"
        elif quantization == "int4":
            model_args += ",load_in_4bit=True,bnb_4bit_quant_type=nf4"
        return [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", "mmlu",
            "--num_fewshot", "5",
            "--limit", str(num_samples),
            "--output_path", output_path,
            "--log_samples",
        ]

    elif backend == "vllm":
        return [
            sys.executable, "-m", "lm_eval",
            "--model", "vllm",
            "--model_args", f"pretrained={model_name},dtype=float16,gpu_memory_utilization=0.85",
            "--tasks", "mmlu",
            "--num_fewshot", "5",
            "--limit", str(num_samples),
            "--output_path", output_path,
            "--log_samples",
        ]

    else:
        raise ValueError(f"Backend '{backend}' not supported for MMLU eval via lm-eval. Use 'hf' or 'vllm'.")


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", required=True, choices=["hf", "vllm"])
    parser.add_argument("--quantization", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    num_samples = config["eval"]["mmlu"]["num_samples"]
    hardware = config.get("hardware", {}).get("gpu", "unknown")

    lm_eval_output_dir = f"results/lm_eval/mmlu_{args.backend}_{args.quantization}"
    os.makedirs(lm_eval_output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  MMLU Eval | backend={args.backend} | quant={args.quantization}")
    print(f"  Model: {model_name}")
    print(f"  Samples: {num_samples} (5-shot)")
    print(f"{'#'*60}")

    cmd = build_lm_eval_command(
        backend=args.backend,
        model_name=model_name,
        quantization=args.quantization,
        num_samples=num_samples,
        output_path=lm_eval_output_dir,
    )

    print(f"  Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, capture_output=False)  # stream output live

    if proc.returncode != 0:
        print(f"\n  lm_eval failed with exit code {proc.returncode}")
        sys.exit(1)

    # Parse accuracy from output
    # lm-eval writes a JSON file named after the model/task
    # Find it in the output directory
    lm_eval_json = None
    for root, dirs, files in os.walk(lm_eval_output_dir):
        for fname in files:
            if fname.endswith(".json") and "results" in fname.lower():
                lm_eval_json = os.path.join(root, fname)
                break
        if not lm_eval_json:
            # Try any json file
            for fname in files:
                if fname.endswith(".json"):
                    lm_eval_json = os.path.join(root, fname)
                    break

    accuracy = None
    if lm_eval_json:
        accuracy = extract_lm_eval_accuracy(lm_eval_json, "mmlu")

    if accuracy is None:
        print("  Could not auto-parse accuracy. Check lm_eval output manually.")
        print(f"  lm-eval results directory: {lm_eval_output_dir}")
    else:
        print(f"\n  MMLU Accuracy ({args.backend} / {args.quantization}): {accuracy:.4f} ({accuracy*100:.1f}%)")

        result = EvalResult(
            backend=args.backend,
            model_name=model_name,
            quantization=args.quantization,
            task="mmlu",
            accuracy=accuracy,
            num_samples=num_samples,
            hardware=hardware,
        )
        save_eval_result(result, f"results/eval_mmlu_{args.backend}_{args.quantization}.json")

        # Log to tracker
        tracking = config.get("tracking", {})
        if tracking.get("backend") == "mlflow":
            try:
                import mlflow
                mlflow.set_experiment(tracking["experiment_name"])
                with mlflow.start_run(run_name=f"mmlu_{args.backend}_{args.quantization}"):
                    mlflow.log_params({
                        "backend": args.backend,
                        "model": model_name,
                        "quantization": args.quantization,
                        "task": "mmlu",
                        "num_samples": num_samples,
                    })
                    mlflow.log_metric("mmlu_accuracy", accuracy)
                print("  Logged to MLflow.")
            except Exception as e:
                print(f"  MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
