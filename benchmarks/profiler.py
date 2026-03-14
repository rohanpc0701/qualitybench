"""
profiler.py - Shared profiling utilities for QualityBench.

Captures:
  - p50 and p99 latency (not just mean)
  - Throughput in tokens/sec
  - Peak GPU memory via torch.cuda AND nvidia-smi subprocess (ground truth)
"""

import time
import subprocess
import json
import os
import datetime
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional


@dataclass
class ProfileResult:
    backend: str
    model_name: str
    quantization: str
    batch_size: int
    num_tokens_generated: int   # average per run
    latencies_ms: List[float]   # all raw timings
    p50_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    throughput_tokens_per_sec: float
    peak_gpu_memory_mb: float   # torch peak (most reliable for tracking across runs)
    nvidia_smi_used_mb: float   # snapshot from nvidia-smi at peak
    hardware: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class Profiler:
    def __init__(
        self,
        backend: str,
        model_name: str,
        quantization: str,
        hardware: str = "T4",
    ):
        self.backend = backend
        self.model_name = model_name
        self.quantization = quantization
        self.hardware = hardware

    # ------------------------------------------------------------------
    # GPU memory helpers
    # ------------------------------------------------------------------

    def _torch_memory_mb(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    def _nvidia_smi_memory_mb(self) -> Optional[float]:
        """Return current used GPU memory from nvidia-smi (in MB)."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                timeout=5,
            ).decode().strip()
            return float(out.split("\n")[0])
        except Exception:
            return None

    def _reset_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Core profiling loop
    # ------------------------------------------------------------------

    def profile_generation(
        self,
        generate_fn: Callable[[], Dict],
        warmup_runs: int = 3,
        benchmark_runs: int = 50,
        batch_size: int = 1,
    ) -> ProfileResult:
        """
        Profile a generation callable.

        generate_fn must return a dict with at least {'num_tokens': int}.
        """
        self._reset_peak_memory()

        # --- Warmup (not timed) ---
        print(f"  Warmup ({warmup_runs} runs)...", flush=True)
        for _ in range(warmup_runs):
            generate_fn()

        self._reset_peak_memory()  # reset after warmup so peak reflects real runs

        # --- Benchmark ---
        print(f"  Benchmarking ({benchmark_runs} runs)...", flush=True)
        latencies_ms: List[float] = []
        total_tokens = 0
        peak_nvidia_smi = 0.0

        for i in range(benchmark_runs):
            self._sync_cuda()
            t0 = time.perf_counter()

            result = generate_fn()

            self._sync_cuda()
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000
            latencies_ms.append(latency_ms)
            total_tokens += result.get("num_tokens", 0)

            # Sample nvidia-smi periodically (every 10 runs) to avoid subprocess overhead
            if i % 10 == 0:
                smi = self._nvidia_smi_memory_mb()
                if smi and smi > peak_nvidia_smi:
                    peak_nvidia_smi = smi

        arr = np.array(latencies_ms)
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        mean = float(np.mean(arr))

        total_time_s = arr.sum() / 1000.0
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

        torch_mem = self._torch_memory_mb()
        peak_gpu_mb = torch_mem.get("peak_mb", peak_nvidia_smi)

        return ProfileResult(
            backend=self.backend,
            model_name=self.model_name,
            quantization=self.quantization,
            batch_size=batch_size,
            num_tokens_generated=total_tokens // benchmark_runs if benchmark_runs else 0,
            latencies_ms=latencies_ms,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            mean_latency_ms=mean,
            throughput_tokens_per_sec=throughput,
            peak_gpu_memory_mb=peak_gpu_mb,
            nvidia_smi_used_mb=peak_nvidia_smi,
            hardware=self.hardware,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_result(self, result: ProfileResult, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = asdict(result)
        # Drop raw latency list from summary file to keep it readable;
        # save full latencies in a separate file for distribution plots.
        raw_path = output_path.replace(".json", "_latencies.json")
        with open(raw_path, "w") as f:
            json.dump({"latencies_ms": data.pop("latencies_ms")}, f)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Saved: {output_path}")
        print(f"  Saved: {raw_path}")

    def print_summary(self, result: ProfileResult):
        print(f"\n{'='*55}")
        print(f"  Backend     : {result.backend}")
        print(f"  Quantization: {result.quantization}")
        print(f"  p50 latency : {result.p50_latency_ms:.1f} ms")
        print(f"  p99 latency : {result.p99_latency_ms:.1f} ms")
        print(f"  Mean latency: {result.mean_latency_ms:.1f} ms")
        print(f"  Throughput  : {result.throughput_tokens_per_sec:.1f} tok/s")
        print(f"  Peak GPU mem: {result.peak_gpu_memory_mb:.0f} MB")
        print(f"{'='*55}\n")


# ------------------------------------------------------------------
# MLflow / W&B logging helper
# ------------------------------------------------------------------

def log_to_tracker(result: ProfileResult, config: dict, run_name: str):
    tracking = config.get("tracking", {})
    backend = tracking.get("backend", "none")
    experiment = tracking.get("experiment_name", "qualitybench")

    if backend == "mlflow":
        try:
            import mlflow
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "backend": result.backend,
                    "model": result.model_name,
                    "quantization": result.quantization,
                    "hardware": result.hardware,
                    "batch_size": result.batch_size,
                })
                mlflow.log_metrics({
                    "p50_latency_ms": result.p50_latency_ms,
                    "p99_latency_ms": result.p99_latency_ms,
                    "mean_latency_ms": result.mean_latency_ms,
                    "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                    "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
                })
            print(f"  Logged to MLflow run: {run_name}")
        except Exception as e:
            print(f"  MLflow logging failed: {e}")

    elif backend == "wandb":
        try:
            import wandb
            wandb.init(project=experiment, name=run_name, reinit=True)
            wandb.config.update({
                "backend": result.backend,
                "model": result.model_name,
                "quantization": result.quantization,
                "hardware": result.hardware,
            })
            wandb.log({
                "p50_latency_ms": result.p50_latency_ms,
                "p99_latency_ms": result.p99_latency_ms,
                "mean_latency_ms": result.mean_latency_ms,
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
            })
            wandb.finish()
            print(f"  Logged to W&B run: {run_name}")
        except Exception as e:
            print(f"  W&B logging failed: {e}")
