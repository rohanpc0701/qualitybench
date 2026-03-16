"""
run_vllm.py - vLLM benchmark runner.

vLLM's key advantages to profile here:
  - PagedAttention: eliminates KV-cache memory fragmentation
  - Continuous batching: saturates GPU even with variable-length sequences
  - This should show clearly higher throughput vs HF Transformers FP16 baseline.

Usage:
    python benchmarks/run_vllm.py --config configs/benchmark_config.yaml
"""

import argparse
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from benchmarks.profiler import Profiler, log_to_tracker


def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark runner")
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests to send (tests batching behavior)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    prompt = config["profiling"]["prompt"]
    max_new_tokens = config["profiling"]["max_new_tokens"]
    warmup_runs = config["profiling"]["num_warmup_runs"]
    benchmark_runs = config["profiling"]["num_benchmark_runs"]
    hardware = config.get("hardware", {}).get("gpu", "unknown")

    # vLLM natively supports fp16; INT8/INT4 available via --quantization awq/gptq flags
    # For this run we benchmark fp16 (apples-to-apples with HF baseline)
    quantization = "fp16"

    print(f"\n{'#'*60}")
    print(f"  vLLM | {model_name} | {quantization} | batch={args.batch_size}")
    print(f"{'#'*60}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
        sys.exit(1)

    print("  Loading vLLM engine (this downloads model on first run)...")
    llm = LLM(
        model=model_name,
        dtype="float16",
        # gpu_memory_utilization=0.90,  # leave 10% headroom; tune if OOM
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
    )

    # Build prompt batch
    prompts = [prompt] * args.batch_size

    def generate_fn():
        outputs = llm.generate(prompts, sampling_params)
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return {"num_tokens": total_tokens // args.batch_size}  # avg per request

    profiler = Profiler(
        backend="vllm",
        model_name=model_name,
        quantization=quantization,
        hardware=hardware,
    )

    result = profiler.profile_generation(
        generate_fn,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        batch_size=args.batch_size,
    )

    profiler.print_summary(result)

    tag = f"bs{args.batch_size}" if args.batch_size > 1 else ""
    output_path = f"results/vllm_{quantization}{('_' + tag) if tag else ''}.json"
    profiler.save_result(result, output_path)

    log_to_tracker(result, config, run_name=f"vllm_{quantization}")

    print("\n  NOTE: Compare vLLM throughput vs HF Transformers fp16 to see")
    print("  PagedAttention + continuous batching advantage.\n")


if __name__ == "__main__":
    main()
