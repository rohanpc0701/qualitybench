"""
QualityBench Dashboard - Streamlit app.

Shows:
  1. Latency distribution (p50/p99 bar chart)
  2. Throughput comparison
  3. Quality vs Efficiency Pareto plot
  4. Recommended config card by use-case
  5. Raw results table

Deploy: streamlit run dashboard/app.py
"""

import sys
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysis.pareto import (
    build_combined_df,
    compute_pareto,
    plot_throughput_vs_memory,
    plot_quality_vs_efficiency,
    plot_latency_comparison,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QualityBench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚡ QualityBench")
st.markdown(
    "**LLM Inference Quality-Efficiency Tradeoff Benchmarking**  \n"
    "Profiling vLLM · llama.cpp · HuggingFace Transformers across FP16 / INT8 / INT4-GPTQ"
)
st.divider()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(RESULTS_DIR):
        return pd.DataFrame()
    return build_combined_df(RESULTS_DIR)

df = load_data()

if df.empty:
    st.warning(
        "No benchmark results found yet. Run benchmarks first:\n\n"
        "```bash\n"
        "python benchmarks/run_hf_transformers.py --config configs/benchmark_config.yaml\n"
        "python benchmarks/run_vllm.py --config configs/benchmark_config.yaml\n"
        "```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    all_backends = sorted(df["backend"].unique().tolist())
    selected_backends = st.multiselect("Backends", all_backends, default=all_backends)

    all_quants = sorted(df["quantization"].unique().tolist())
    selected_quants = st.multiselect("Quantizations", all_quants, default=all_quants)

    st.divider()
    st.caption("Hardware: " + df["hardware"].iloc[0] if "hardware" in df.columns else "")
    st.caption("Model: " + df["model_name"].iloc[0] if "model_name" in df.columns else "")
    st.caption("Reload to pick up new results (cached 60s).")

filtered = df[
    df["backend"].isin(selected_backends) & df["quantization"].isin(selected_quants)
]

if filtered.empty:
    st.info("No data matches selected filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Key metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Configs profiled", len(filtered))
col2.metric(
    "Best throughput",
    f"{filtered['throughput_tokens_per_sec'].max():.0f} tok/s",
    help="Max throughput across all configs",
)
col3.metric(
    "Best p50 latency",
    f"{filtered['p50_latency_ms'].min():.0f} ms",
    help="Lowest p50 latency (single request)",
)
if "mmlu_accuracy" in filtered.columns and not filtered["mmlu_accuracy"].isna().all():
    col4.metric(
        "Best MMLU",
        f"{filtered['mmlu_accuracy'].max()*100:.1f}%",
    )
else:
    col4.metric("MMLU evals", "Pending")

st.divider()

# ---------------------------------------------------------------------------
# Recommended config card
# ---------------------------------------------------------------------------
st.subheader("Recommended Configuration")
tab_low_lat, tab_high_tput, tab_cost_eff = st.tabs(
    ["🏎️ Low Latency", "🚀 High Throughput", "💰 Cost Efficient"]
)

with tab_low_lat:
    best = filtered.loc[filtered["p50_latency_ms"].idxmin()]
    st.success(f"**{best['label']}**")
    cols = st.columns(3)
    cols[0].metric("p50 latency", f"{best['p50_latency_ms']:.1f} ms")
    cols[1].metric("p99 latency", f"{best['p99_latency_ms']:.1f} ms")
    cols[2].metric("Throughput", f"{best['throughput_tokens_per_sec']:.0f} tok/s")
    st.caption("Best for: chatbots, interactive assistants, low-concurrency APIs")

with tab_high_tput:
    best = filtered.loc[filtered["throughput_tokens_per_sec"].idxmax()]
    st.success(f"**{best['label']}**")
    cols = st.columns(3)
    cols[0].metric("Throughput", f"{best['throughput_tokens_per_sec']:.0f} tok/s")
    cols[1].metric("p50 latency", f"{best['p50_latency_ms']:.1f} ms")
    cols[2].metric("GPU Memory", f"{best['peak_gpu_memory_mb']:.0f} MB")
    st.caption("Best for: batch inference, offline processing, high-concurrency serving")

with tab_cost_eff:
    if "efficiency_tps_per_gb" in filtered.columns:
        best = filtered.loc[filtered["efficiency_tps_per_gb"].idxmax()]
        st.success(f"**{best['label']}**")
        cols = st.columns(3)
        cols[0].metric("Efficiency", f"{best['efficiency_tps_per_gb']:.1f} tok/s/GB")
        cols[1].metric("Throughput", f"{best['throughput_tokens_per_sec']:.0f} tok/s")
        cols[2].metric("GPU Memory", f"{best['peak_gpu_memory_mb']:.0f} MB")
        st.caption("Best for: cost-constrained deployments, smaller GPU instances")
    else:
        st.info("Efficiency metric unavailable (missing GPU memory data).")

st.divider()

# ---------------------------------------------------------------------------
# Latency comparison
# ---------------------------------------------------------------------------
st.subheader("p50 vs p99 Latency")
st.caption("p99 tail latency is critical for production SLOs — mean latency hides outliers.")

fig_lat = go.Figure()
fig_lat.add_trace(go.Bar(
    name="p50",
    x=filtered["label"],
    y=filtered["p50_latency_ms"],
    marker_color="steelblue",
))
fig_lat.add_trace(go.Bar(
    name="p99",
    x=filtered["label"],
    y=filtered["p99_latency_ms"],
    marker_color="tomato",
))
fig_lat.update_layout(
    barmode="group",
    xaxis_tickangle=-30,
    yaxis_title="Latency (ms)",
    height=420,
    template="plotly_white",
    margin=dict(b=100),
)
st.plotly_chart(fig_lat, use_container_width=True)

# ---------------------------------------------------------------------------
# Throughput comparison
# ---------------------------------------------------------------------------
st.subheader("Throughput Comparison")

fig_tput = go.Figure(go.Bar(
    x=filtered["label"],
    y=filtered["throughput_tokens_per_sec"],
    marker=dict(
        color=filtered["throughput_tokens_per_sec"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="tok/s"),
    ),
    hovertemplate="<b>%{x}</b><br>%{y:.1f} tok/s<extra></extra>",
))
fig_tput.update_layout(
    xaxis_tickangle=-30,
    yaxis_title="Throughput (tokens/sec)",
    height=380,
    template="plotly_white",
    margin=dict(b=100),
)
st.plotly_chart(fig_tput, use_container_width=True)

# ---------------------------------------------------------------------------
# Pareto plot: Throughput vs Memory
# ---------------------------------------------------------------------------
st.subheader("Pareto Frontier: Throughput vs GPU Memory")
st.caption("Red stars = Pareto-optimal configurations (no other config is simultaneously faster and more memory-efficient).")

pareto_df = compute_pareto(
    filtered,
    objectives=[("throughput_tokens_per_sec", "max"), ("peak_gpu_memory_mb", "min")],
)

fig_pareto = go.Figure()
non_p = pareto_df[~pareto_df["pareto_optimal"]]
par = pareto_df[pareto_df["pareto_optimal"]]

if not non_p.empty:
    fig_pareto.add_trace(go.Scatter(
        x=non_p["throughput_tokens_per_sec"],
        y=non_p["peak_gpu_memory_mb"],
        mode="markers",
        text=non_p["label"],
        marker=dict(color="lightgray", size=10),
        hovertemplate="<b>%{text}</b><br>%{x:.1f} tok/s | %{y:.0f} MB<extra></extra>",
        name="Sub-optimal",
    ))

if not par.empty:
    fig_pareto.add_trace(go.Scatter(
        x=par["throughput_tokens_per_sec"],
        y=par["peak_gpu_memory_mb"],
        mode="markers+text",
        text=par["label"],
        textposition="top center",
        marker=dict(color="crimson", size=15, symbol="star", line=dict(width=1, color="black")),
        hovertemplate="<b>%{text}</b><br>%{x:.1f} tok/s | %{y:.0f} MB<extra></extra>",
        name="Pareto Optimal ★",
    ))

fig_pareto.update_layout(
    xaxis_title="Throughput (tok/s) →",
    yaxis_title="Peak GPU Memory (MB) ↑ lower better",
    height=500,
    template="plotly_white",
)
st.plotly_chart(fig_pareto, use_container_width=True)

# ---------------------------------------------------------------------------
# Quality vs Efficiency (only if eval data available)
# ---------------------------------------------------------------------------
if "mmlu_accuracy" in filtered.columns and not filtered["mmlu_accuracy"].isna().all():
    st.subheader("Quality vs Efficiency (Pareto)")
    st.caption("How much quality do you give up for each unit of efficiency gain?")

    qe_df = filtered.dropna(subset=["mmlu_accuracy", "efficiency_tps_per_gb"])
    qe_df = compute_pareto(qe_df, objectives=[("mmlu_accuracy", "max"), ("efficiency_tps_per_gb", "max")])

    fig_qe = go.Figure()
    non_p_qe = qe_df[~qe_df["pareto_optimal"]]
    par_qe = qe_df[qe_df["pareto_optimal"]]

    if not non_p_qe.empty:
        fig_qe.add_trace(go.Scatter(
            x=non_p_qe["efficiency_tps_per_gb"],
            y=non_p_qe["mmlu_accuracy"] * 100,
            mode="markers",
            text=non_p_qe["label"],
            marker=dict(color="lightgray", size=10),
            hovertemplate="<b>%{text}</b><br>Efficiency: %{x:.1f}<br>MMLU: %{y:.1f}%<extra></extra>",
            name="Sub-optimal",
        ))
    if not par_qe.empty:
        fig_qe.add_trace(go.Scatter(
            x=par_qe["efficiency_tps_per_gb"],
            y=par_qe["mmlu_accuracy"] * 100,
            mode="markers+text",
            text=par_qe["label"],
            textposition="top center",
            marker=dict(color="crimson", size=15, symbol="star", line=dict(width=1, color="black")),
            hovertemplate="<b>%{text}</b><br>Efficiency: %{x:.1f}<br>MMLU: %{y:.1f}%<extra></extra>",
            name="Pareto Optimal ★",
        ))

    fig_qe.update_layout(
        xaxis_title="Efficiency (tok/s per GB) →",
        yaxis_title="MMLU Accuracy (%) →",
        height=500,
        template="plotly_white",
    )
    st.plotly_chart(fig_qe, use_container_width=True)

# ---------------------------------------------------------------------------
# Raw results table
# ---------------------------------------------------------------------------
with st.expander("Raw Results Table"):
    display_cols = [
        c for c in [
            "label", "backend", "quantization",
            "throughput_tokens_per_sec", "p50_latency_ms", "p99_latency_ms",
            "peak_gpu_memory_mb", "efficiency_tps_per_gb",
            "mmlu_accuracy", "truthfulqa_accuracy",
        ] if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_cols].sort_values("throughput_tokens_per_sec", ascending=False),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "QualityBench · Built with Streamlit · "
    "[GitHub](https://github.com/yourusername/qualitybench)  "
    "Hardware: T4 GPU (Google Colab)"
)
