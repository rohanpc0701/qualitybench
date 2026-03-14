"""
run_llamacpp.py - llama.cpp benchmark runner via llama-cpp-python.

llama.cpp shines in:
  - Single-batch / low-concurrency latency
  - GGUF quantized models (Q4_K_M, Q8_0)
  - Mixed CPU+GPU execution

Key hypothesis to validate:
  llama.cpp Q8_0 can beat vLLM FP16 on p50 single-request latency due to different
  memory layout (row-major GGUF vs HBM-heavy CUDA tensors).

Usage:
    # Download GGUF first (e.g. from HuggingFace TheBloke):
    python benchmarks/run_llamacpp.py \\
        --config configs/benchmark_config.yaml \\
        --model-path /path/to/mistral-7b-v0.1.Q4_K_M.gguf \\
        --quantization q4_k_m

    python benchmarks/run_llamacpp.py \\
        --config configs/benchmark_config.yaml \\
        --model-path /path/to/mistral-7b-v0.1.Q8_0.gguf \\
        --quantization q8_0
"""

import argparse
import sys
import os
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from benchmarks.profiler import Profiler, log_to_tracker


QUANT_CHOICES = ["q4_k_m", "q8_0"]

# Approximate GGUF sizes for documentation purposes
GGUF_SIZES_GB = {
    "q4_k_m": 4.1,
    "q8_0": 7.7,
}


def main():
    parser = argparse.ArgumentParser(description="llama.cpp benchmark runner")
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Absolute path to the GGUF model file",
    )
    parser.add_argument(
        "--quantization",
        required=True,
        choices=QUANT_CHOICES,
        help="GGUF quantization level",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all). Use 0 for pure CPU.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: GGUF model not found at {args.model_path}")
        print("Download GGUF models from: https://huggingface.co/TheBloke")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    prompt = config["profiling"]["prompt"]
    max_new_tokens = config["profiling"]["max_new_tokens"]
    warmup_runs = config["profiling"]["num_warmup_runs"]
    benchmark_runs = config["profiling"]["num_benchmark_runs"]
    hardware = config.get("hardware", {}).get("gpu", "unknown")

    print(f"\n{'#'*60}")
    print(f"  llama.cpp | {args.quantization} | n_gpu_layers={args.n_gpu_layers}")
    print(f"  File: {os.path.basename(args.model_path)}")
    size = GGUF_SIZES_GB.get(args.quantization, "?")
    print(f"  Approx size: {size} GB")
    print(f"{'#'*60}")

    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed.")
        print("Install with: pip install llama-cpp-python")
        print("For GPU support: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
        sys.exit(1)

    print("  Loading model...", flush=True)
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=2048,
        verbose=False,
    )

    def generate_fn():
        output = llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0,
            echo=False,
            stop=["</s>", "\n\n"],
        )
        num_tokens = output["usage"]["completion_tokens"]
        return {"num_tokens": num_tokens}

    profiler = Profiler(
        backend="llamacpp",
        model_name=model_name,
        quantization=args.quantization,
        hardware=hardware,
    )

    result = profiler.profile_generation(
        generate_fn,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )

    profiler.print_summary(result)

    output_path = f"results/llamacpp_{args.quantization}.json"
    profiler.save_result(result, output_path)

    log_to_tracker(result, config, run_name=f"llamacpp_{args.quantization}")

    print("\n  COMPARE: Check if p50 latency beats vLLM fp16 -- llama.cpp Q8_0 often wins")
    print("  on single-request latency due to tighter GGUF memory layout.\n")


if __name__ == "__main__":
    main()
