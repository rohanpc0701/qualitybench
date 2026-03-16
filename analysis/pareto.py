"""
pareto.py - Pareto frontier computation and visualization.

Dimensions:
  - Quality    : MMLU accuracy (higher is better)
  - Throughput : tokens/sec (higher is better)
  - Efficiency : tokens/sec per GB of GPU memory (higher is better)
  - Latency    : p50 latency ms (lower is better)

A config is Pareto-optimal if no other config is strictly better on ALL dimensions.
We support 2D and 3D Pareto plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from evals.eval_utils import load_all_results


# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------

def build_combined_df(results_dir: str) -> pd.DataFrame:
    """
    Merge benchmark results (throughput/latency/memory) with eval results
    (accuracy) into a single DataFrame keyed on (backend, quantization).
    """
    records = load_all_results(results_dir)

    bench_rows = []
    eval_rows = []

    for r in records:
        # Distinguish benchmark results from eval results
        if "throughput_tokens_per_sec" in r:
            bench_rows.append(r)
        elif "accuracy" in r and "task" in r:
            eval_rows.append(r)

    df_bench = pd.DataFrame(bench_rows) if bench_rows else pd.DataFrame()
    df_eval = pd.DataFrame(eval_rows) if eval_rows else pd.DataFrame()

    if df_bench.empty:
        print("Warning: No benchmark results found in", results_dir)
        return pd.DataFrame()

    # Compute efficiency metric
    if "peak_gpu_memory_mb" in df_bench.columns:
        df_bench["efficiency_tps_per_gb"] = (
            df_bench["throughput_tokens_per_sec"]
            / (df_bench["peak_gpu_memory_mb"] / 1024).clip(lower=0.1)
        )
    else:
        df_bench["efficiency_tps_per_gb"] = df_bench.get("throughput_tokens_per_sec", 0)

    df_bench["label"] = df_bench["backend"] + " / " + df_bench["quantization"]

    if df_eval.empty:
        df_bench["mmlu_accuracy"] = None
        df_bench["truthfulqa_accuracy"] = None
        return df_bench

    # Pivot eval results: one row per (backend, quantization), columns per task
    mmlu = df_eval[df_eval["task"] == "mmlu"][["backend", "quantization", "accuracy"]].rename(
        columns={"accuracy": "mmlu_accuracy"}
    )
    tqa = df_eval[df_eval["task"].str.contains("truthfulqa", na=False)][
        ["backend", "quantization", "accuracy"]
    ].rename(columns={"accuracy": "truthfulqa_accuracy"})

    df = df_bench.copy()
    if not mmlu.empty:
        df = df.merge(mmlu, on=["backend", "quantization"], how="left")
    else:
        df["mmlu_accuracy"] = None

    if not tqa.empty:
        df = df.merge(tqa, on=["backend", "quantization"], how="left")
    else:
        df["truthfulqa_accuracy"] = None

    return df


# ---------------------------------------------------------------------------
# Pareto computation
# ---------------------------------------------------------------------------

def is_pareto_optimal(costs: np.ndarray) -> np.ndarray:
    """
    Find Pareto-optimal points where all objectives are to be MAXIMIZED.

    Args:
        costs: shape (n, d), each row is a point, each column an objective.
               Pass negated values for objectives you want to minimize.
    Returns:
        Boolean mask, True = Pareto optimal.
    """
    n = costs.shape[0]
    is_opt = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_opt[i]:
            continue
        # A point j dominates i if j >= i on all objectives AND j > i on at least one
        dominates = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
        dominates[i] = False
        if dominates.any():
            is_opt[i] = False
    return is_opt


def compute_pareto(df: pd.DataFrame, objectives: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Add a 'pareto_optimal' column to df.

    objectives: list of (column_name, 'max'|'min') tuples
    """
    valid = df.dropna(subset=[col for col, _ in objectives]).copy()
    if valid.empty:
        df["pareto_optimal"] = False
        return df

    cost_matrix = np.column_stack([
        valid[col].values if direction == "max" else -valid[col].values
        for col, direction in objectives
    ])

    mask = is_pareto_optimal(cost_matrix)
    valid["pareto_optimal"] = mask

    df = df.copy()
    df["pareto_optimal"] = False
    df.loc[valid.index, "pareto_optimal"] = valid["pareto_optimal"]
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

BACKEND_COLORS = {
    "hf_transformers": "#636EFA",
    "vllm": "#EF553B",
    "llamacpp": "#00CC96",
}

QUANT_SYMBOLS = {
    "fp16": "circle",
    "int8": "square",
    "int4": "diamond",
    "q4_k_m": "cross",
    "q8_0": "x",
}


def plot_throughput_vs_memory(df: pd.DataFrame, output_path: str = "results/pareto_throughput_memory.html"):
    """
    2D Pareto plot: Throughput vs GPU Memory.
    Pareto-optimal = high throughput AND low memory.
    """
    df = compute_pareto(
        df,
        objectives=[("throughput_tokens_per_sec", "max"), ("peak_gpu_memory_mb", "min")],
    )

    fig = go.Figure()

    for is_pareto, group_label, color, size, symbol in [
        (False, "Sub-optimal", "lightgray", 10, "circle"),
        (True, "Pareto Optimal ★", "crimson", 16, "star"),
    ]:
        sub = df[df["pareto_optimal"] == is_pareto]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["throughput_tokens_per_sec"],
            y=sub["peak_gpu_memory_mb"],
            mode="markers+text" if is_pareto else "markers",
            text=sub["label"] if is_pareto else None,
            textposition="top center",
            marker=dict(color=color, size=size, symbol=symbol,
                        line=dict(width=1, color="black") if is_pareto else {}),
            customdata=sub[["backend", "quantization", "p50_latency_ms", "p99_latency_ms"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Throughput: %{x:.1f} tok/s<br>"
                "GPU Memory: %{y:.0f} MB<br>"
                "p50: %{customdata[2]:.1f}ms | p99: %{customdata[3]:.1f}ms"
                "<extra></extra>"
            ) if is_pareto else (
                "<b>%{marker.symbol}</b><br>"
                "Throughput: %{x:.1f} tok/s<br>"
                "GPU Memory: %{y:.0f} MB<extra></extra>"
            ),
            name=group_label,
        ))

    fig.update_layout(
        title="<b>QualityBench: Throughput vs GPU Memory (Pareto Frontier)</b>",
        xaxis_title="Throughput (tokens/sec)  →  higher is better",
        yaxis_title="Peak GPU Memory (MB)  ↑  lower is better",
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        annotations=[dict(
            text="★ = Pareto-optimal (no other config is faster AND cheaper in memory)",
            xref="paper", yref="paper", x=0, y=-0.12,
            showarrow=False, font=dict(size=11, color="gray"),
        )],
    )

    _save_fig(fig, output_path)
    return fig


def plot_quality_vs_efficiency(df: pd.DataFrame, output_path: str = "results/pareto_quality_efficiency.html"):
    """
    2D Pareto plot: MMLU accuracy vs tokens/sec-per-GB.
    The core quality-efficiency tradeoff plot.
    """
    if "mmlu_accuracy" not in df.columns or df["mmlu_accuracy"].isna().all():
        print("  No MMLU results available yet for quality-efficiency plot.")
        return None

    df = compute_pareto(
        df,
        objectives=[("mmlu_accuracy", "max"), ("efficiency_tps_per_gb", "max")],
    )
    sub = df.dropna(subset=["mmlu_accuracy"])

    fig = go.Figure()

    non_p = sub[~sub["pareto_optimal"]]
    par = sub[sub["pareto_optimal"]]

    if not non_p.empty:
        fig.add_trace(go.Scatter(
            x=non_p["efficiency_tps_per_gb"],
            y=non_p["mmlu_accuracy"] * 100,
            mode="markers",
            marker=dict(color="lightgray", size=10),
            text=non_p["label"],
            hovertemplate="<b>%{text}</b><br>Efficiency: %{x:.1f} tok/s/GB<br>MMLU: %{y:.1f}%<extra></extra>",
            name="Sub-optimal",
        ))

    if not par.empty:
        fig.add_trace(go.Scatter(
            x=par["efficiency_tps_per_gb"],
            y=par["mmlu_accuracy"] * 100,
            mode="markers+text",
            text=par["label"],
            textposition="top center",
            marker=dict(color="crimson", size=16, symbol="star", line=dict(width=1, color="black")),
            hovertemplate="<b>%{text}</b><br>Efficiency: %{x:.1f} tok/s/GB<br>MMLU: %{y:.1f}%<extra></extra>",
            name="Pareto Optimal ★",
        ))

    fig.update_layout(
        title="<b>QualityBench: Quality vs Efficiency (Pareto Frontier)</b>",
        xaxis_title="Efficiency (tok/s per GB GPU memory)  →  higher is better",
        yaxis_title="MMLU Accuracy (%)  →  higher is better",
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    _save_fig(fig, output_path)
    return fig


def plot_latency_comparison(df: pd.DataFrame, output_path: str = "results/latency_comparison.html"):
    """Bar chart comparing p50 and p99 latency across all configs."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="p50 Latency",
        x=df["label"],
        y=df["p50_latency_ms"],
        marker_color="steelblue",
        error_y=None,
    ))

    fig.add_trace(go.Bar(
        name="p99 Latency",
        x=df["label"],
        y=df["p99_latency_ms"],
        marker_color="tomato",
    ))

    fig.update_layout(
        title="<b>QualityBench: p50 vs p99 Latency by Configuration</b>",
        xaxis_title="Backend / Quantization",
        yaxis_title="Latency (ms)  ↓  lower is better",
        barmode="group",
        height=500,
        template="plotly_white",
        xaxis_tickangle=-30,
        annotations=[dict(
            text="p99 > p50 gap indicates tail-latency risk — critical for production SLOs",
            xref="paper", yref="paper", x=0, y=-0.18,
            showarrow=False, font=dict(size=11, color="gray"),
        )],
    )

    _save_fig(fig, output_path)
    return fig


def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(path)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Pareto frontier and generate plots")
    parser.add_argument("--results-dir", default="results", help="Directory with JSON result files")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir} ...")
    df = build_combined_df(args.results_dir)

    if df.empty:
        print("No results found. Run benchmarks first.")
        sys.exit(1)

    print(f"Found {len(df)} configurations:\n")
    cols = ["label", "throughput_tokens_per_sec", "p50_latency_ms", "p99_latency_ms",
            "peak_gpu_memory_mb", "efficiency_tps_per_gb", "mmlu_accuracy"]
    show_cols = [c for c in cols if c in df.columns]
    print(df[show_cols].to_string(index=False))
    print()

    plot_throughput_vs_memory(df)
    plot_quality_vs_efficiency(df)
    plot_latency_comparison(df)

    print("\nAll plots generated.")
