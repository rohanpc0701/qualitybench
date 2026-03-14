"""
run_truthfulqa.py - TruthfulQA evaluation via lm-evaluation-harness.

TruthfulQA measures whether a model generates truthful answers by testing
on questions where humans often give false answers due to misconceptions.
We use the MC2 (multiple-choice, multi-true) variant.

Usage:
    python evals/run_truthfulqa.py --config configs/benchmark_config.yaml \\
        --backend hf --quantization fp16
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
            "--tasks", "truthfulqa_mc2",
            "--num_fewshot", "0",       # TruthfulQA is 0-shot
            "--limit", str(num_samples),
            "--output_path", output_path,
        ]

    elif backend == "vllm":
        return [
            sys.executable, "-m", "lm_eval",
            "--model", "vllm",
            "--model_args", f"pretrained={model_name},dtype=float16,gpu_memory_utilization=0.85",
            "--tasks", "truthfulqa_mc2",
            "--num_fewshot", "0",
            "--limit", str(num_samples),
            "--output_path", output_path,
        ]

    else:
        raise ValueError(f"Backend '{backend}' not supported. Use 'hf' or 'vllm'.")


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA evaluation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", required=True, choices=["hf", "vllm"])
    parser.add_argument("--quantization", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    num_samples = config["eval"]["truthfulqa"]["num_samples"]
    hardware = config.get("hardware", {}).get("gpu", "unknown")

    lm_eval_output_dir = f"results/lm_eval/truthfulqa_{args.backend}_{args.quantization}"
    os.makedirs(lm_eval_output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  TruthfulQA Eval | backend={args.backend} | quant={args.quantization}")
    print(f"  Model: {model_name}")
    print(f"  Samples: {num_samples} (0-shot, MC2)")
    print(f"{'#'*60}")

    cmd = build_lm_eval_command(
        backend=args.backend,
        model_name=model_name,
        quantization=args.quantization,
        num_samples=num_samples,
        output_path=lm_eval_output_dir,
    )

    print(f"  Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, capture_output=False)

    if proc.returncode != 0:
        print(f"\n  lm_eval failed with exit code {proc.returncode}")
        sys.exit(1)

    # Parse accuracy
    lm_eval_json = None
    for root, dirs, files in os.walk(lm_eval_output_dir):
        for fname in files:
            if fname.endswith(".json"):
                lm_eval_json = os.path.join(root, fname)
                break

    accuracy = None
    if lm_eval_json:
        accuracy = extract_lm_eval_accuracy(lm_eval_json, "truthfulqa_mc2")

    if accuracy is None:
        print("  Could not auto-parse accuracy. Check lm_eval output manually.")
        print(f"  lm-eval results directory: {lm_eval_output_dir}")
    else:
        print(f"\n  TruthfulQA MC2 ({args.backend} / {args.quantization}): {accuracy:.4f} ({accuracy*100:.1f}%)")

        result = EvalResult(
            backend=args.backend,
            model_name=model_name,
            quantization=args.quantization,
            task="truthfulqa_mc2",
            accuracy=accuracy,
            num_samples=num_samples,
            hardware=hardware,
        )
        save_eval_result(result, f"results/eval_truthfulqa_{args.backend}_{args.quantization}.json")

        tracking = config.get("tracking", {})
        if tracking.get("backend") == "mlflow":
            try:
                import mlflow
                mlflow.set_experiment(tracking["experiment_name"])
                with mlflow.start_run(run_name=f"truthfulqa_{args.backend}_{args.quantization}"):
                    mlflow.log_params({
                        "backend": args.backend,
                        "model": model_name,
                        "quantization": args.quantization,
                        "task": "truthfulqa_mc2",
                        "num_samples": num_samples,
                    })
                    mlflow.log_metric("truthfulqa_mc2_accuracy", accuracy)
                print("  Logged to MLflow.")
            except Exception as e:
                print(f"  MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
