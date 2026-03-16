"""
run_hf_transformers.py - HuggingFace Transformers benchmark runner.

Supports: FP16 baseline, INT8 (bitsandbytes), INT4-NF4 (bitsandbytes QLoRA-style).

Usage:
    python benchmarks/run_hf_transformers.py --config configs/benchmark_config.yaml
    python benchmarks/run_hf_transformers.py --config configs/benchmark_config.yaml --quantization int8
"""

import argparse
import sys
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from benchmarks.profiler import Profiler, log_to_tracker


QUANT_CHOICES = ["fp16", "int8", "int4"]


def load_model_and_tokenizer(model_name: str, quantization: str):
    print(f"  Loading {model_name} [{quantization}] ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bits/param
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

    else:
        raise ValueError(f"Unknown quantization: {quantization}. Choose from {QUANT_CHOICES}")

    model.eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description="HF Transformers benchmark runner")
    parser.add_argument("--config", required=True, help="Path to benchmark_config.yaml")
    parser.add_argument(
        "--quantization",
        choices=QUANT_CHOICES,
        default=None,
        help="Override quantization (default: run all from config)",
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

    # Resolve which quantizations to run
    if args.quantization:
        quantizations = [args.quantization]
    else:
        # Find hf_transformers backend in config
        hf_backend = next(
            (b for b in config["backends"] if b["name"] == "hf_transformers"), None
        )
        quantizations = hf_backend["quantizations"] if hf_backend else ["fp16"]

    for quant in quantizations:
        print(f"\n{'#'*60}")
        print(f"  HF Transformers | {model_name} | {quant}")
        print(f"{'#'*60}")

        tokenizer, model = load_model_and_tokenizer(model_name, quant)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def generate_fn():
            with torch.inference_mode():
                output = model.generate(  # noqa: F821
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            num_generated = output.shape[1] - inputs["input_ids"].shape[1]
            return {"num_tokens": num_generated}

        profiler = Profiler(
            backend="hf_transformers",
            model_name=model_name,
            quantization=quant,
            hardware=hardware,
        )

        result = profiler.profile_generation(
            generate_fn,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )

        profiler.print_summary(result)

        output_path = f"results/hf_transformers_{quant}.json"
        profiler.save_result(result, output_path)

        run_name = f"hf_{quant}"
        log_to_tracker(result, config, run_name)

        # Free VRAM before next quantization variant
        del model
        torch.cuda.empty_cache()
        print("  GPU cache cleared.")


if __name__ == "__main__":
    main()
